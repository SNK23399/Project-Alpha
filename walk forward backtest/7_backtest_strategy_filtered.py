"""
Step 7: Walk-Forward Backtest with Monte Carlo Feature Filtering
================================================================

Same as Step 6 but with an additional Monte Carlo filtering step.

At each rebalancing point:
1. Get candidate features (pass MIN_ALPHA and MIN_HIT_RATE filters)
2. Load PRECOMPUTED MC statistics (from 5b_precompute_mc_hitrates.py)
3. HYBRID MC filter using CI lower bounds (same approach for both metrics):
   a) Hit rate filter: Wilson CI lower bound > baseline (e.g., 50%)
   b) Alpha filter: t-CI lower bound > baseline (e.g., 0%)
   c) Selectivity: keep top X% by ranking metric (alpha, hitrate, or combined)
4. Run greedy forward selection on the filtered candidates

The combined ranking option weights alpha and hit rate like greedy does.

IMPORTANT: Run 5b_precompute_mc_hitrates.py first to generate MC statistics!

Usage:
    python 7_backtest_strategy_filtered.py

Output:
    data/backtest_results/filtered_backtest_summary.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import time
from numba import njit, prange

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================
# CONFIGURATION
# ============================================================

# Decay weighting
DECAY_HALF_LIFE_MONTHS = 54

# Single horizon
HOLDING_MONTHS = 1

# N values to test
N_SATELLITES_TO_TEST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Training parameters
MIN_TRAINING_MONTHS = 36
REOPTIMIZATION_FREQUENCY = 1

# Feature selection criteria
MIN_ALPHA = 0.001        # Minimum 0.1% alpha
MIN_HIT_RATE = 0.55      # Minimum 55% hit rate
MAX_ENSEMBLE_SIZE = 20   # Maximum features in ensemble
MIN_IMPROVEMENT = 0.0001 # Minimum improvement to add feature

# Monte Carlo filtering settings
MC_TOP_PERCENTILE = 1.0     # Keep all features that pass significance filters (no additional cutoff)
MC_CONFIDENCE_LEVEL = 0.95  # Confidence level for both Wilson CI and Alpha CI

# Dynamic baseline: use per-date mean of random MC performance as threshold
# Feature CI lower bound must exceed that date's random baseline
MC_USE_DYNAMIC_BASELINE = True  # If False, uses fixed thresholds below
MC_BASELINE_RATE = 0.50     # Fixed hit rate threshold (used if dynamic=False)
MC_BASELINE_ALPHA = 0.0     # Fixed alpha threshold (used if dynamic=False)

# Filter logic: 'AND' = must pass both HR and Alpha, 'OR' = pass either one
MC_FILTER_LOGIC = 'OR'      # 'AND' or 'OR'

# Ranking method for selectivity filter
MC_RANK_BY = 'combined'     # 'alpha', 'hitrate', or 'combined'
MC_COMBINED_ALPHA_WEIGHT = 0.7  # Weight for alpha in combined score (hitrate gets 1-this)

# Data and output directories
DATA_DIR = Path(__file__).parent / 'data'
OUTPUT_DIR = DATA_DIR / 'backtest_results'


# ============================================================
# NUMBA CPU FUNCTIONS (for greedy selection)
# ============================================================

@njit(cache=True)
def evaluate_feature_weighted(feature_alpha, feature_hit, feat_idx, n_satellites, date_mask, weights):
    """Numba-optimized feature evaluation with decay weights."""
    n_dates = len(date_mask)
    sum_alpha = 0.0
    sum_hit = 0.0
    sum_weight = 0.0

    for i in range(n_dates):
        if date_mask[i] and weights[i] > 0:
            alpha = feature_alpha[i, feat_idx, n_satellites - 1]
            hit = feature_hit[i, feat_idx, n_satellites - 1]
            if not np.isnan(alpha):
                sum_alpha += alpha * weights[i]
                sum_hit += hit * weights[i]
                sum_weight += weights[i]

    if sum_weight == 0:
        return -999.0, -999.0, 0

    return sum_alpha / sum_weight, sum_hit / sum_weight, int(sum_weight)


@njit(cache=True, parallel=True)
def evaluate_all_features_weighted(feature_alpha, feature_hit, n_satellites, date_mask, weights):
    """Evaluate ALL features with decay weights in parallel."""
    n_features = feature_alpha.shape[1]
    avg_alphas = np.empty(n_features, dtype=np.float64)
    hit_rates = np.empty(n_features, dtype=np.float64)
    n_valids = np.empty(n_features, dtype=np.int64)

    for feat_idx in prange(n_features):
        avg_alpha, hit_rate, n_valid = evaluate_feature_weighted(
            feature_alpha, feature_hit, feat_idx, n_satellites, date_mask, weights
        )
        avg_alphas[feat_idx] = avg_alpha
        hit_rates[feat_idx] = hit_rate
        n_valids[feat_idx] = n_valid

    return avg_alphas, hit_rates, n_valids


@njit(cache=True)
def compute_ensemble_alpha_single_date(rankings_slice, feature_indices, alpha_row, alpha_valid_row, n_satellites):
    """Compute ensemble alpha for a single date (for greedy selection)."""
    n_isins = rankings_slice.shape[0]
    n_features = len(feature_indices)

    scores = np.empty(n_isins, dtype=np.float64)
    for i in range(n_isins):
        score_sum = 0.0
        count = 0
        for j in range(n_features):
            feat_idx = feature_indices[j]
            val = rankings_slice[i, feat_idx]
            if not np.isnan(val):
                score_sum += val
                count += 1
        if count > 0:
            scores[i] = score_sum / count
        else:
            scores[i] = np.nan

    n_valid = 0
    for i in range(n_isins):
        if not np.isnan(scores[i]):
            n_valid += 1

    if n_valid < n_satellites:
        return np.nan

    selected_indices = np.empty(n_satellites, dtype=np.int64)
    used = np.zeros(n_isins, dtype=np.bool_)

    for k in range(n_satellites):
        best_idx = -1
        best_score = -np.inf
        for i in range(n_isins):
            if not used[i] and not np.isnan(scores[i]) and scores[i] > best_score:
                best_score = scores[i]
                best_idx = i
        if best_idx >= 0:
            selected_indices[k] = best_idx
            used[best_idx] = True
        else:
            return np.nan

    alpha_sum = 0.0
    alpha_count = 0
    for k in range(n_satellites):
        idx = selected_indices[k]
        if alpha_valid_row[idx]:
            alpha_sum += alpha_row[idx]
            alpha_count += 1

    if alpha_count == 0:
        return np.nan

    return alpha_sum / alpha_count


@njit(cache=True, parallel=True)
def compute_ensemble_alpha_all_dates(rankings, feature_indices, alpha_matrix, alpha_valid, date_mask, n_satellites):
    """Compute ensemble alpha for all dates in parallel."""
    n_dates = rankings.shape[0]
    alphas = np.empty(n_dates, dtype=np.float64)

    for date_idx in prange(n_dates):
        if not date_mask[date_idx]:
            alphas[date_idx] = np.nan
            continue
        alphas[date_idx] = compute_ensemble_alpha_single_date(
            rankings[date_idx], feature_indices, alpha_matrix[date_idx], alpha_valid[date_idx], n_satellites
        )

    return alphas


@njit(cache=True)
def select_top_n_isins(rankings_slice, feature_indices, n_satellites, alpha_valid_mask):
    """Select top-N ISINs based on ensemble of features."""
    n_isins = rankings_slice.shape[0]
    n_features = len(feature_indices)

    scores = np.empty(n_isins, dtype=np.float64)
    for i in range(n_isins):
        if alpha_valid_mask is not None and not alpha_valid_mask[i]:
            scores[i] = np.nan
            continue

        score_sum = 0.0
        count = 0
        for j in range(n_features):
            feat_idx = feature_indices[j]
            val = rankings_slice[i, feat_idx]
            if not np.isnan(val):
                score_sum += val
                count += 1
        if count > 0:
            scores[i] = score_sum / count
        else:
            scores[i] = np.nan

    n_valid = 0
    for i in range(n_isins):
        if not np.isnan(scores[i]):
            n_valid += 1

    if n_valid < n_satellites:
        return np.array([-1], dtype=np.int64)

    selected = np.empty(n_satellites, dtype=np.int64)
    used = np.zeros(n_isins, dtype=np.bool_)

    for k in range(n_satellites):
        best_idx = -1
        best_score = -np.inf
        for i in range(n_isins):
            if not used[i] and not np.isnan(scores[i]) and scores[i] > best_score:
                best_score = scores[i]
                best_idx = i
        selected[k] = best_idx
        if best_idx >= 0:
            used[best_idx] = True

    return selected


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load precomputed data for the single 1-month horizon."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    horizon_label = f"{HOLDING_MONTHS}month"

    alpha_file = DATA_DIR / f'forward_alpha_{horizon_label}.parquet'
    if not alpha_file.exists():
        raise FileNotFoundError(f"Forward alpha file not found: {alpha_file}")

    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])

    rankings_file = DATA_DIR / f'rankings_matrix_{horizon_label}.npz'
    if not rankings_file.exists():
        raise FileNotFoundError(f"Rankings matrix not found: {rankings_file}")

    npz_data = np.load(rankings_file, allow_pickle=True)
    rankings = npz_data['rankings'].astype(np.float64)
    dates = pd.to_datetime(npz_data['dates'])
    isins = npz_data['isins']
    feature_names = list(npz_data['features'])

    feature_alpha_file = DATA_DIR / f'feature_alpha_{horizon_label}.npz'
    if not feature_alpha_file.exists():
        raise FileNotFoundError(f"Feature alpha file not found: {feature_alpha_file}")

    fa_data = np.load(feature_alpha_file, allow_pickle=True)
    feature_alpha = fa_data['feature_alpha'].astype(np.float64)
    feature_hit = fa_data['feature_hit'].astype(np.float64)

    # Load precomputed MC hit rates
    mc_file = DATA_DIR / f'mc_hitrates_{horizon_label}.npz'
    if not mc_file.exists():
        raise FileNotFoundError(
            f"MC hit rates file not found: {mc_file}\n"
            "Run 5b_precompute_mc_hitrates.py first!"
        )

    mc_data = np.load(mc_file, allow_pickle=True)
    mc_hitrates = mc_data['mc_hitrates']           # (n_n_values, n_dates, n_features)
    mc_samples = mc_data['mc_samples']             # (n_n_values, n_dates, n_features) - sample counts
    mc_alpha_mean = mc_data['mc_alpha_mean']       # (n_n_values, n_dates, n_features) - mean alpha
    mc_alpha_std = mc_data['mc_alpha_std']         # (n_n_values, n_dates, n_features) - std alpha
    mc_candidate_masks = mc_data['candidate_masks'] # (n_n_values, n_dates, n_features)
    mc_inverted_masks = mc_data['inverted_masks']   # (n_n_values, n_dates, n_features)
    mc_n_satellites = mc_data['n_satellites']       # (n_n_values,)
    mc_test_start_idx = int(mc_data['test_start_idx'])

    # Build N to index mapping
    n_to_mc_idx = {int(n): i for i, n in enumerate(mc_n_satellites)}

    n_dates = len(dates)
    n_isins = len(isins)

    isin_to_idx = {isin: idx for idx, isin in enumerate(isins)}
    date_to_idx = {date: idx for idx, date in enumerate(dates)}

    alpha_matrix = np.full((n_dates, n_isins), np.nan, dtype=np.float64)

    for date, group in alpha_df.groupby('date'):
        if date not in date_to_idx:
            continue
        date_idx = date_to_idx[date]
        for isin, alpha in zip(group['isin'].values, group['forward_alpha'].values):
            if isin in isin_to_idx:
                isin_idx = isin_to_idx[isin]
                alpha_matrix[date_idx, isin_idx] = alpha

    alpha_valid = ~np.isnan(alpha_matrix)

    print(f"  [OK] {horizon_label}: {rankings.shape}")
    print(f"  Dates: {dates[0].date()} to {dates[-1].date()} ({len(dates)} months)")
    print(f"  ISINs: {len(isins)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  [OK] MC data loaded: hitrates, samples, alpha_mean, alpha_std {mc_hitrates.shape}")

    return {
        'alpha_matrix': alpha_matrix,
        'alpha_valid': alpha_valid,
        'rankings': rankings,
        'dates': dates,
        'isins': isins,
        'isin_to_idx': isin_to_idx,
        'date_to_idx': date_to_idx,
        'feature_names': feature_names,
        'feature_alpha': feature_alpha,
        'feature_hit': feature_hit,
        # MC data
        'mc_hitrates': mc_hitrates,
        'mc_samples': mc_samples,
        'mc_alpha_mean': mc_alpha_mean,
        'mc_alpha_std': mc_alpha_std,
        'mc_candidate_masks': mc_candidate_masks,
        'mc_inverted_masks': mc_inverted_masks,
        'mc_n_satellites': mc_n_satellites,
        'mc_test_start_idx': mc_test_start_idx,
        'n_to_mc_idx': n_to_mc_idx
    }


# ============================================================
# MC FILTERING USING PRECOMPUTED RESULTS
# ============================================================

def wilson_ci_lower_bound(hit_rate, n_samples, confidence=0.95):
    """
    Calculate the lower bound of Wilson confidence interval.

    This is more accurate than normal approximation for binomial proportions,
    especially for small samples or extreme proportions.

    Args:
        hit_rate: Observed hit rate (0-1)
        n_samples: Number of samples
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Lower bound of the confidence interval
    """
    from scipy import stats

    if n_samples == 0:
        return 0.0

    z = stats.norm.ppf(1 - (1 - confidence) / 2)  # ~1.96 for 95%
    z2 = z * z

    # Wilson score interval formula
    denominator = 1 + z2 / n_samples
    center = (hit_rate + z2 / (2 * n_samples)) / denominator
    margin = (z / denominator) * np.sqrt(
        hit_rate * (1 - hit_rate) / n_samples + z2 / (4 * n_samples * n_samples)
    )

    return center - margin


def alpha_ci_lower_bound(mean_alpha, std_alpha, n_samples, confidence=0.95):
    """
    Calculate the lower bound of confidence interval for mean alpha.

    Uses t-distribution since we're estimating population mean from sample.
    This is analogous to Wilson CI for hit rate - we get a lower bound
    that we're X% confident the true mean exceeds.

    Args:
        mean_alpha: Sample mean of alpha values
        std_alpha: Sample standard deviation of alpha values
        n_samples: Number of samples
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Lower bound of the confidence interval for mean alpha
    """
    from scipy import stats

    if n_samples < 2 or std_alpha <= 0 or np.isnan(mean_alpha) or np.isnan(std_alpha):
        return -np.inf  # Cannot compute, return very low value

    # Standard error of the mean
    se = std_alpha / np.sqrt(n_samples)

    # t critical value for one-sided CI
    # For lower bound only, we use (1 - confidence) quantile
    t_crit = stats.t.ppf(1 - confidence, df=n_samples - 1)

    # Lower bound: mean - t_crit * SE
    # Note: t_crit is negative for confidence > 0.5, so this adds the margin
    lower_bound = mean_alpha + t_crit * se

    return lower_bound


def mc_filter_candidates_precomputed(data, candidates, test_idx, n_satellites):
    """
    Filter candidates using PRECOMPUTED MC statistics with HYBRID approach:

    Step 1: Hit rate significance filter (Wilson CI lower bound > baseline)
    Step 2: Alpha significance filter (Alpha CI lower bound > baseline)
    Step 3: Selectivity filter (keep top X% by ranking metric)

    Both filters use confidence interval lower bounds for consistency.
    Baseline can be dynamic (per-date mean) or fixed.

    Returns: filtered list of candidates
    """
    if len(candidates) < 2:
        return candidates  # Not enough to filter

    n_to_mc_idx = data['n_to_mc_idx']

    # Check if we have precomputed data for this N
    if n_satellites not in n_to_mc_idx:
        print(f"  Warning: No precomputed MC data for N={n_satellites}, skipping MC filter")
        return candidates

    mc_idx = n_to_mc_idx[n_satellites]
    mc_hitrates = data['mc_hitrates'][mc_idx]        # (n_dates, n_features)
    mc_samples = data['mc_samples'][mc_idx]          # (n_dates, n_features) - actual sample counts
    mc_alpha_mean = data['mc_alpha_mean'][mc_idx]    # (n_dates, n_features) - mean alpha
    mc_alpha_std = data['mc_alpha_std'][mc_idx]      # (n_dates, n_features) - std alpha
    mc_candidate_mask = data['mc_candidate_masks'][mc_idx]  # (n_dates, n_features)

    # Compute baseline thresholds
    if MC_USE_DYNAMIC_BASELINE:
        # Dynamic: use this date's mean across all candidate features
        date_hr_values = mc_hitrates[test_idx, mc_candidate_mask[test_idx]]
        date_alpha_values = mc_alpha_mean[test_idx, mc_candidate_mask[test_idx]]

        # Filter out NaN values
        valid_hr = date_hr_values[~np.isnan(date_hr_values)]
        valid_alpha = date_alpha_values[~np.isnan(date_alpha_values)]

        baseline_hr = np.mean(valid_hr) if len(valid_hr) > 0 else MC_BASELINE_RATE
        baseline_alpha = np.mean(valid_alpha) if len(valid_alpha) > 0 else MC_BASELINE_ALPHA
    else:
        # Fixed thresholds
        baseline_hr = MC_BASELINE_RATE
        baseline_alpha = MC_BASELINE_ALPHA

    # Step 1 & 2: Filter by hit rate AND alpha CI lower bounds
    significant_candidates = []
    for cand in candidates:
        feat_idx = cand['idx']

        # Check if this feature was a candidate in the precomputed data
        if mc_candidate_mask[test_idx, feat_idx]:
            mc_hr = mc_hitrates[test_idx, feat_idx]
            n_samples = mc_samples[test_idx, feat_idx]
            mean_alpha = mc_alpha_mean[test_idx, feat_idx]
            std_alpha = mc_alpha_std[test_idx, feat_idx]

            if not np.isnan(mc_hr) and n_samples > 0:
                # Hit rate significance: Wilson CI lower bound > baseline
                hr_ci_lower = wilson_ci_lower_bound(
                    mc_hr, n_samples, MC_CONFIDENCE_LEVEL
                )
                hr_significant = hr_ci_lower > baseline_hr

                # Alpha significance: Alpha CI lower bound > baseline
                alpha_ci_lower = alpha_ci_lower_bound(
                    mean_alpha, std_alpha, n_samples, MC_CONFIDENCE_LEVEL
                )
                alpha_significant = alpha_ci_lower > baseline_alpha

                # Check significance based on filter logic
                if MC_FILTER_LOGIC == 'OR':
                    passes_filter = hr_significant or alpha_significant
                else:  # 'AND'
                    passes_filter = hr_significant and alpha_significant

                if passes_filter:
                    significant_candidates.append({
                        'cand': cand,
                        'mc_hr': mc_hr,
                        'mc_alpha': mean_alpha,
                        'hr_ci_lower': hr_ci_lower,
                        'alpha_ci_lower': alpha_ci_lower
                    })

    # Step 3: Apply selectivity filter - keep top X% by ranking metric
    if len(significant_candidates) > 0:
        # Calculate ranking score based on MC_RANK_BY
        if MC_RANK_BY == 'alpha':
            # Rank by alpha CI lower bound
            significant_candidates.sort(key=lambda x: x['alpha_ci_lower'], reverse=True)
        elif MC_RANK_BY == 'hitrate':
            # Rank by hit rate CI lower bound
            significant_candidates.sort(key=lambda x: x['hr_ci_lower'], reverse=True)
        else:  # 'combined'
            # Normalize and combine (like greedy does)
            # Get min/max for normalization
            alphas = [c['alpha_ci_lower'] for c in significant_candidates]
            hrs = [c['hr_ci_lower'] for c in significant_candidates]

            alpha_min, alpha_max = min(alphas), max(alphas)
            hr_min, hr_max = min(hrs), max(hrs)

            alpha_range = alpha_max - alpha_min if alpha_max > alpha_min else 1.0
            hr_range = hr_max - hr_min if hr_max > hr_min else 1.0

            for c in significant_candidates:
                alpha_norm = (c['alpha_ci_lower'] - alpha_min) / alpha_range
                hr_norm = (c['hr_ci_lower'] - hr_min) / hr_range
                c['combined_score'] = MC_COMBINED_ALPHA_WEIGHT * alpha_norm + (1 - MC_COMBINED_ALPHA_WEIGHT) * hr_norm

            significant_candidates.sort(key=lambda x: x['combined_score'], reverse=True)

        # Keep top MC_TOP_PERCENTILE
        n_keep = max(1, int(len(significant_candidates) * MC_TOP_PERCENTILE))
        filtered = [c['cand'] for c in significant_candidates[:n_keep]]
    else:
        filtered = []

    # Fallback: if no candidates pass, return top candidate by combined metric
    if len(filtered) == 0 and len(candidates) > 0:
        candidate_scores = []
        for cand in candidates:
            feat_idx = cand['idx']
            if mc_candidate_mask[test_idx, feat_idx]:
                mc_hr = mc_hitrates[test_idx, feat_idx]
                mean_alpha = mc_alpha_mean[test_idx, feat_idx]
                if not np.isnan(mc_hr) and not np.isnan(mean_alpha):
                    # Use raw values for fallback ranking
                    score = MC_COMBINED_ALPHA_WEIGHT * mean_alpha + (1 - MC_COMBINED_ALPHA_WEIGHT) * mc_hr
                    candidate_scores.append((cand, score))

        if candidate_scores:
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            # Return top candidate only as fallback
            filtered = [candidate_scores[0][0]]

    return filtered


# ============================================================
# FEATURE SELECTION WITH MC FILTERING + GREEDY
# ============================================================

def fast_greedy_search_with_mc_filter(data, n_satellites, train_mask, test_idx):
    """
    Feature selection with:
    1. Initial filtering (MIN_ALPHA, MIN_HIT_RATE)
    2. Monte Carlo filtering (keep top 33% by MC hit rate) - uses precomputed results
    3. Greedy forward selection on filtered candidates
    """
    rankings = data['rankings'].copy()
    feature_alpha = data['feature_alpha']
    feature_hit = data['feature_hit']
    feature_names = data['feature_names']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    # Get precomputed inversion mask
    n_to_mc_idx = data['n_to_mc_idx']
    mc_idx = n_to_mc_idx.get(n_satellites, 0)
    mc_inverted_mask = data['mc_inverted_masks'][mc_idx]  # (n_dates, n_features)

    n_features = len(feature_names)
    train_mask_np = np.array(train_mask, dtype=np.bool_)

    # Create exponential decay weights
    n_dates = len(train_mask)
    weights = np.zeros(n_dates, dtype=np.float64)
    decay_rate = np.log(2) / DECAY_HALF_LIFE_MONTHS

    for i in range(n_dates):
        if train_mask[i]:
            months_ago = test_idx - i
            weights[i] = np.exp(-decay_rate * months_ago)

    # Step 1: Evaluate ALL features with decay weights
    avg_alphas, hit_rates, n_valids = evaluate_all_features_weighted(
        feature_alpha, feature_hit, n_satellites, train_mask_np, weights
    )

    # Step 2: Filter initial candidates
    positive_mask = (avg_alphas >= MIN_ALPHA) & (hit_rates >= MIN_HIT_RATE) & (n_valids > 0)
    negative_mask = (avg_alphas <= -MIN_ALPHA) & ((1 - hit_rates) >= MIN_HIT_RATE) & (n_valids > 0)

    candidates = []

    # Positive features
    pos_indices = np.where(positive_mask)[0]
    for idx in pos_indices:
        candidates.append({
            'idx': idx,
            'type': 'positive',
            'alpha': avg_alphas[idx],
            'name': feature_names[idx]
        })

    # Negative features (invert rankings)
    neg_indices = np.where(negative_mask)[0]
    for idx in neg_indices:
        rankings[:, :, idx] = 1.0 - rankings[:, :, idx]
        candidates.append({
            'idx': idx,
            'type': 'inverted',
            'alpha': -avg_alphas[idx],
            'name': feature_names[idx]
        })

    n_initial_candidates = len(candidates)

    if len(candidates) == 0:
        return [], None, rankings, {'n_initial': 0, 'n_after_mc': 0}

    # Step 3: Monte Carlo filtering - keep top 33% using precomputed results
    if len(candidates) >= 3:
        candidates = mc_filter_candidates_precomputed(
            data, candidates, test_idx, n_satellites
        )

    n_after_mc = len(candidates)

    if len(candidates) == 0:
        return [], None, rankings, {'n_initial': n_initial_candidates, 'n_after_mc': 0}

    # Sort by alpha (descending)
    candidates.sort(key=lambda x: x['alpha'], reverse=True)

    # Step 4: Greedy forward selection
    selected = []
    best_alpha = -np.inf
    current_indices = np.array([], dtype=np.int64)

    for iteration in range(min(MAX_ENSEMBLE_SIZE, len(candidates))):
        best_add = None
        best_add_alpha = -np.inf

        remaining = [c for c in candidates if c['idx'] not in current_indices]

        if len(remaining) == 0:
            break

        for cand in remaining:
            feat_idx = cand['idx']
            test_indices = np.append(current_indices, feat_idx).astype(np.int64)

            alphas = compute_ensemble_alpha_all_dates(
                rankings, test_indices, alpha_matrix, alpha_valid, train_mask_np, n_satellites
            )

            valid_alphas = alphas[~np.isnan(alphas)]
            if len(valid_alphas) == 0:
                continue

            avg_alpha = np.mean(valid_alphas)

            if avg_alpha > best_add_alpha:
                best_add_alpha = avg_alpha
                best_add = cand

        improvement = best_add_alpha - best_alpha if best_alpha > -np.inf else best_add_alpha

        if best_add is None or improvement < MIN_IMPROVEMENT:
            break

        selected.append({
            'idx': best_add['idx'],
            'name': best_add['name'],
            'type': best_add['type'],
            'improvement': improvement
        })
        current_indices = np.append(current_indices, best_add['idx']).astype(np.int64)
        best_alpha = best_add_alpha

    # Compute final performance
    best_perf = None
    if len(selected) > 0:
        final_indices = np.array([f['idx'] for f in selected], dtype=np.int64)
        alphas = compute_ensemble_alpha_all_dates(
            rankings, final_indices, alpha_matrix, alpha_valid, train_mask_np, n_satellites
        )
        valid_alphas = alphas[~np.isnan(alphas)]
        if len(valid_alphas) > 0:
            best_perf = {
                'avg_alpha': np.mean(valid_alphas),
                'std_alpha': np.std(valid_alphas),
                'hit_rate': np.mean(valid_alphas > 0),
                'n_periods': len(valid_alphas)
            }

    extra_info = {
        'n_initial': n_initial_candidates,
        'n_after_mc': n_after_mc
    }

    return selected, best_perf, rankings, extra_info


# ============================================================
# SELECT TOP-N ETFs
# ============================================================

def select_etfs(data, test_idx, n_satellites, feature_indices, rankings):
    """Select top-N ETFs based on feature ensemble scores."""
    if feature_indices is None or len(feature_indices) == 0:
        return None

    alpha_valid_mask = data['alpha_valid'][test_idx]
    feature_arr = np.array(feature_indices, dtype=np.int64)

    selected = select_top_n_isins(rankings[test_idx], feature_arr, n_satellites, alpha_valid_mask)
    if selected[0] == -1:
        return None
    return selected.tolist()


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================

def walk_forward_backtest(data, n_satellites, show_progress=True):
    """Run walk-forward backtest with MC-filtered feature selection."""
    dates = data['dates']
    isins = data['isins']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0]

    if len(test_start_idx) == 0:
        return pd.DataFrame()

    test_start_idx = test_start_idx[0]

    if show_progress:
        print(f"\n  Testing N={n_satellites}")
        print(f"  Test period: {dates[test_start_idx].date()} to {dates[-1].date()}")
        if MC_USE_DYNAMIC_BASELINE:
            print(f"  MC filtering: CI lower bounds > per-date random baseline (dynamic)")
        else:
            print(f"  MC filtering: Hit rate CI > {MC_BASELINE_RATE:.0%}, Alpha CI > {MC_BASELINE_ALPHA:.2%}")

    results = []
    feature_indices = None
    rankings = None
    months_since_reopt = REOPTIMIZATION_FREQUENCY

    iterator = range(test_start_idx, len(dates))
    if show_progress:
        iterator = tqdm(iterator, desc=f"N={n_satellites}")

    for test_idx in iterator:
        test_date = dates[test_idx]

        # Re-optimize feature selection periodically
        if months_since_reopt >= REOPTIMIZATION_FREQUENCY:
            train_mask = dates < test_date

            # Use MC-filtered feature selection with precomputed results
            selected, train_perf, modified_rankings, extra_info = fast_greedy_search_with_mc_filter(
                data, n_satellites, train_mask, test_idx
            )

            if len(selected) > 0:
                feature_indices = [f['idx'] for f in selected]
                rankings = modified_rankings
            else:
                feature_indices = None
                rankings = None

            months_since_reopt = 0

        months_since_reopt += 1

        # Skip if no valid features
        if feature_indices is None:
            continue

        # Select top-N ETFs
        selected_isin_indices = select_etfs(
            data, test_idx, n_satellites, feature_indices, rankings
        )

        if selected_isin_indices is None or len(selected_isin_indices) < n_satellites:
            continue

        # Compute average alpha for selected ETFs
        alphas = []
        selected_isins = []

        for isin_idx in selected_isin_indices:
            if alpha_valid[test_idx, isin_idx]:
                alphas.append(alpha_matrix[test_idx, isin_idx])
                selected_isins.append(isins[isin_idx])

        if len(alphas) == 0:
            continue

        avg_alpha = np.mean(alphas)
        results.append({
            'date': test_date,
            'n_satellites': n_satellites,
            'avg_alpha': avg_alpha,
            'n_selected': len(selected_isins),
            'selected_isins': ','.join(selected_isins)
        })

    return pd.DataFrame(results)


# ============================================================
# ANALYZE RESULTS
# ============================================================

def analyze_results(results_df, n_satellites):
    """Analyze backtest results."""
    if len(results_df) == 0:
        return None

    avg_alpha = results_df['avg_alpha'].mean()
    std_alpha = results_df['avg_alpha'].std()
    hit_rate = (results_df['avg_alpha'] > 0).mean()
    sharpe = avg_alpha / std_alpha if std_alpha > 0 else 0

    results_df = results_df.copy()
    results_df['cumulative'] = (1 + results_df['avg_alpha']).cumprod() - 1
    total_return = results_df['cumulative'].iloc[-1]

    return {
        'n_satellites': n_satellites,
        'decay_half_life': DECAY_HALF_LIFE_MONTHS,
        'mc_confidence': MC_CONFIDENCE_LEVEL,
        'mc_dynamic_baseline': MC_USE_DYNAMIC_BASELINE,
        'n_periods': len(results_df),
        'avg_alpha': avg_alpha,
        'std_alpha': std_alpha,
        'annual_alpha': avg_alpha * 12,
        'hit_rate': hit_rate,
        'sharpe': sharpe,
        'total_return': total_return
    }


# ============================================================
# MAIN
# ============================================================

def main():
    """Run walk-forward backtest with MC feature filtering."""
    print("=" * 60)
    print("WALK-FORWARD BACKTEST WITH MC FILTERING (PRECOMPUTED)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Decay half-life: {DECAY_HALF_LIFE_MONTHS} months")
    print(f"  Holding period: {HOLDING_MONTHS} month")
    print(f"  N values to test: {N_SATELLITES_TO_TEST}")
    print(f"\nMonte Carlo Filtering (CI Lower Bounds):")
    print(f"  Using PRECOMPUTED MC hit rates and alpha statistics")
    print(f"  Confidence level: {MC_CONFIDENCE_LEVEL:.0%}")
    if MC_USE_DYNAMIC_BASELINE:
        print(f"  Baseline: DYNAMIC (per-date mean of random MC performance)")
    else:
        print(f"  Hit rate filter: Wilson CI lower bound > {MC_BASELINE_RATE:.0%}")
        print(f"  Alpha filter: t-CI lower bound > {MC_BASELINE_ALPHA:.2%}")
    print(f"  Filter logic: {MC_FILTER_LOGIC} (pass {'either' if MC_FILTER_LOGIC == 'OR' else 'both'} HR or Alpha test)")
    print(f"  Ranking method: {MC_RANK_BY}" + (f" (alpha weight: {MC_COMBINED_ALPHA_WEIGHT:.0%})" if MC_RANK_BY == 'combined' else ""))
    print(f"  Selectivity: top {MC_TOP_PERCENTILE:.0%} of significant features")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data (including precomputed MC results)
    data = load_data()

    # Warm up Numba
    print("\nWarming up Numba JIT compilation...")
    warmup_start = time.time()
    warmup_mask = np.ones(10, dtype=np.bool_)
    warmup_weights = np.ones(10, dtype=np.float64)
    _ = evaluate_all_features_weighted(
        data['feature_alpha'][:10],
        data['feature_hit'][:10],
        1,
        warmup_mask,
        warmup_weights
    )
    print(f"  JIT warmup complete in {time.time() - warmup_start:.1f}s")

    all_stats = []
    all_results = {}

    start_time = time.time()

    # Test each N value
    for test_num, n in enumerate(N_SATELLITES_TO_TEST, 1):
        print(f"\n{'='*60}")
        print(f"TEST {test_num}/{len(N_SATELLITES_TO_TEST)}: N={n}")
        print(f"{'='*60}")

        results_df = walk_forward_backtest(data, n, show_progress=True)

        if len(results_df) > 0:
            all_results[n] = results_df

            stats = analyze_results(results_df, n)
            if stats:
                all_stats.append(stats)

                print(f"\n  Results:")
                print(f"    Periods: {stats['n_periods']}")
                print(f"    Monthly Alpha: {stats['avg_alpha']*100:.2f}%")
                print(f"    Annual Alpha: {stats['annual_alpha']*100:.1f}%")
                print(f"    Hit Rate: {stats['hit_rate']:.2%}")
                print(f"    Sharpe: {stats['sharpe']:.3f}")

                # Save individual results
                output_file = OUTPUT_DIR / f'filtered_backtest_N{n}.csv'
                results_df.to_csv(output_file, index=False)
        else:
            print(f"  No results")

    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - MC FILTERED STRATEGY (PRECOMPUTED)")
    print("=" * 60)
    print(f"\nMC Filtering ({MC_CONFIDENCE_LEVEL:.0%} CI):")
    if MC_USE_DYNAMIC_BASELINE:
        print(f"  Baseline: dynamic (per-date mean)")
    else:
        print(f"  Hit rate: CI lower > {MC_BASELINE_RATE:.0%}")
        print(f"  Alpha: CI lower > {MC_BASELINE_ALPHA:.2%}")
    print(f"  Logic: {MC_FILTER_LOGIC}")
    print(f"  Ranking: {MC_RANK_BY}")
    print(f"  Selectivity: top {MC_TOP_PERCENTILE:.0%}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if len(all_stats) > 0:
        summary_df = pd.DataFrame(all_stats)

        print("\n" + "-" * 80)
        print(f"{'N':>3} {'Periods':>8} {'Monthly':>10} {'Annual':>10} {'Hit Rate':>10} {'Sharpe':>8}")
        print("-" * 80)

        for _, row in summary_df.sort_values('n_satellites').iterrows():
            print(f"{int(row['n_satellites']):>3} {int(row['n_periods']):>8} "
                  f"{row['avg_alpha']*100:>9.2f}% {row['annual_alpha']*100:>9.1f}% "
                  f"{row['hit_rate']:>9.1%} {row['sharpe']:>8.3f}")

        print("-" * 80)

        # Find best by different metrics
        best_sharpe = summary_df.loc[summary_df['sharpe'].idxmax()]
        best_alpha = summary_df.loc[summary_df['annual_alpha'].idxmax()]
        best_hit = summary_df.loc[summary_df['hit_rate'].idxmax()]

        print(f"\nBest by Sharpe:     N={int(best_sharpe['n_satellites'])}, "
              f"Alpha={best_sharpe['annual_alpha']*100:.1f}%/yr, "
              f"Hit={best_sharpe['hit_rate']:.1%}, "
              f"Sharpe={best_sharpe['sharpe']:.3f}")

        print(f"Best by Alpha:      N={int(best_alpha['n_satellites'])}, "
              f"Alpha={best_alpha['annual_alpha']*100:.1f}%/yr, "
              f"Hit={best_alpha['hit_rate']:.1%}, "
              f"Sharpe={best_alpha['sharpe']:.3f}")

        print(f"Best by Hit Rate:   N={int(best_hit['n_satellites'])}, "
              f"Alpha={best_hit['annual_alpha']*100:.1f}%/yr, "
              f"Hit={best_hit['hit_rate']:.1%}, "
              f"Sharpe={best_hit['sharpe']:.3f}")

        # Save summary
        summary_file = OUTPUT_DIR / 'filtered_backtest_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\n[SAVED] {summary_file}")


if __name__ == '__main__':
    main()
