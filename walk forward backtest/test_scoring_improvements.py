"""
Experimental: Feature-Level Filtering and Scoring Improvements
==============================================================

This script tests different approaches to improve feature selection
by penalizing features that frequently lead to bad picks.

Approaches tested:
1. Feature hit-rate decay: Weight features by their recent hit rate
2. Feature penalty system: Penalize features that cause losses
3. Adaptive thresholds: Dynamic MIN_ALPHA and MIN_HIT_RATE based on recent performance

Based on 6_backtest_strategy.py but with experimental modifications.
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

# Base decay weighting
DECAY_HALF_LIFE_MONTHS = 54  # Center of 48-60 significant range

# Single horizon
HOLDING_MONTHS = 1

# N values to test
N_SATELLITES_TO_TEST = [5]  # Focus on N=5 for faster experimentation

# Training parameters
MIN_TRAINING_MONTHS = 36  # Minimum training history
REOPTIMIZATION_FREQUENCY = 1  # Re-optimize every N months

# BASE Feature selection criteria (will be modified by experiments)
BASE_MIN_ALPHA = 0.001        # Minimum 0.1% alpha
BASE_MIN_HIT_RATE = 0.55      # Minimum 55% hit rate
MAX_ENSEMBLE_SIZE = 20        # Maximum features in ensemble
MIN_IMPROVEMENT = 0.0001      # Minimum improvement to add feature

# EXPERIMENTAL: Feature scoring adjustments
# Approach 1: Recent hit rate weighting
FEATURE_HIT_RATE_LOOKBACK = 12  # Months to look back for recent performance
FEATURE_HIT_RATE_WEIGHT = 0.3   # Weight for recent hit rate in scoring

# Approach 2: Feature penalty system
ENABLE_FEATURE_PENALTY = False
FEATURE_PENALTY_THRESHOLD = 0.4  # Features with hit rate < this get penalized
FEATURE_PENALTY_FACTOR = 0.5     # Multiply alpha by this for bad features

# Approach 3: Adaptive thresholds
ENABLE_ADAPTIVE_THRESHOLDS = False
ADAPTIVE_LOOKBACK = 24  # Months to compute adaptive thresholds

# Data and output directories
DATA_DIR = Path(__file__).parent / 'data'
OUTPUT_DIR = DATA_DIR / 'backtest_results'


# ============================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# ============================================================

@njit(cache=True)
def evaluate_feature_weighted(feature_alpha, feature_hit, feat_idx, n_satellites, date_mask, weights):
    """
    Numba-optimized feature evaluation with decay weights.
    Returns: (avg_alpha, hit_rate, n_valid) or (-999, -999, 0) if invalid
    """
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
    """
    Evaluate ALL features with decay weights in parallel using Numba prange.
    Returns arrays of (avg_alpha, hit_rate, n_valid) for each feature.
    """
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
def evaluate_feature_recent(feature_alpha, feature_hit, feat_idx, n_satellites, date_mask, lookback_start):
    """
    Evaluate feature performance over RECENT period only (last N months).
    Returns: (avg_alpha, hit_rate, n_valid)
    """
    n_dates = len(date_mask)
    sum_alpha = 0.0
    sum_hit = 0.0
    n_valid = 0

    for i in range(lookback_start, n_dates):
        if date_mask[i]:
            alpha = feature_alpha[i, feat_idx, n_satellites - 1]
            hit = feature_hit[i, feat_idx, n_satellites - 1]
            if not np.isnan(alpha):
                sum_alpha += alpha
                sum_hit += hit
                n_valid += 1

    if n_valid == 0:
        return -999.0, -999.0, 0

    return sum_alpha / n_valid, sum_hit / n_valid, n_valid


@njit(cache=True, parallel=True)
def evaluate_all_features_recent(feature_alpha, feature_hit, n_satellites, date_mask, lookback_start):
    """
    Evaluate ALL features over RECENT period in parallel.
    """
    n_features = feature_alpha.shape[1]
    avg_alphas = np.empty(n_features, dtype=np.float64)
    hit_rates = np.empty(n_features, dtype=np.float64)
    n_valids = np.empty(n_features, dtype=np.int64)

    for feat_idx in prange(n_features):
        avg_alpha, hit_rate, n_valid = evaluate_feature_recent(
            feature_alpha, feature_hit, feat_idx, n_satellites, date_mask, lookback_start
        )
        avg_alphas[feat_idx] = avg_alpha
        hit_rates[feat_idx] = hit_rate
        n_valids[feat_idx] = n_valid

    return avg_alphas, hit_rates, n_valids


@njit(cache=True)
def compute_ensemble_alpha_single_date(rankings_slice, feature_indices, alpha_row, alpha_valid_row, n_satellites):
    """
    Compute ensemble alpha for a single date.
    rankings_slice: (n_isins, n_features)
    feature_indices: array of feature indices to use
    Returns: average alpha or NaN if not enough valid
    """
    n_isins = rankings_slice.shape[0]
    n_features = len(feature_indices)

    # Compute scores as mean of selected feature rankings
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

    # Count valid scores
    n_valid = 0
    for i in range(n_isins):
        if not np.isnan(scores[i]):
            n_valid += 1

    if n_valid < n_satellites:
        return np.nan

    # Find top-N by score (higher is better)
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

    # Compute average alpha of selected
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
    """
    Compute ensemble alpha for all dates in parallel.
    Returns array of alphas for each date (NaN for invalid/skipped dates).
    """
    n_dates = rankings.shape[0]
    alphas = np.empty(n_dates, dtype=np.float64)

    for date_idx in prange(n_dates):
        if not date_mask[date_idx]:
            alphas[date_idx] = np.nan
            continue

        alphas[date_idx] = compute_ensemble_alpha_single_date(
            rankings[date_idx],
            feature_indices,
            alpha_matrix[date_idx],
            alpha_valid[date_idx],
            n_satellites
        )

    return alphas


@njit(cache=True)
def select_top_n_isins(rankings_slice, feature_indices, n_satellites, alpha_valid_mask=None):
    """
    Select top-N ISINs based on ensemble of features.
    Returns array of selected ISIN indices, or [-1] if not enough valid.
    """
    n_isins = rankings_slice.shape[0]
    n_features = len(feature_indices)

    # Compute scores (only for ISINs with valid alpha if mask provided)
    scores = np.empty(n_isins, dtype=np.float64)
    for i in range(n_isins):
        # Skip ISINs without valid alpha
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

    # Count valid
    n_valid = 0
    for i in range(n_isins):
        if not np.isnan(scores[i]):
            n_valid += 1

    if n_valid < n_satellites:
        return np.array([-1], dtype=np.int64)  # Signal failure

    # Select top-N
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
# LOAD DATA
# ============================================================

def load_data():
    """Load precomputed data for the single 1-month horizon."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    data_dir = DATA_DIR
    horizon_label = f"{HOLDING_MONTHS}month"

    # Load forward alpha
    alpha_file = data_dir / f'forward_alpha_{horizon_label}.parquet'
    if not alpha_file.exists():
        raise FileNotFoundError(f"Forward alpha file not found: {alpha_file}")

    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])

    # Load rankings matrix
    rankings_file = data_dir / f'rankings_matrix_{horizon_label}.npz'
    if not rankings_file.exists():
        raise FileNotFoundError(f"Rankings matrix not found: {rankings_file}")

    npz_data = np.load(rankings_file, allow_pickle=True)
    rankings = npz_data['rankings'].astype(np.float64)
    dates = pd.to_datetime(npz_data['dates'])
    isins = npz_data['isins']
    feature_names = list(npz_data['features'])

    # Load feature-alpha matrix
    feature_alpha_file = data_dir / f'feature_alpha_{horizon_label}.npz'
    if not feature_alpha_file.exists():
        raise FileNotFoundError(f"Feature alpha file not found: {feature_alpha_file}")

    fa_data = np.load(feature_alpha_file, allow_pickle=True)
    feature_alpha = fa_data['feature_alpha'].astype(np.float64)
    feature_hit = fa_data['feature_hit'].astype(np.float64)

    # Convert alpha_df to 2D numpy array
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
        'feature_hit': feature_hit
    }


# ============================================================
# EXPERIMENTAL FEATURE SELECTION APPROACHES
# ============================================================

def greedy_search_baseline(data, n_satellites, train_mask, test_idx, min_alpha=BASE_MIN_ALPHA, min_hit_rate=BASE_MIN_HIT_RATE):
    """
    BASELINE: Original greedy search with decay weighting.
    This is identical to 6_backtest_strategy.py
    """
    rankings = data['rankings'].copy()
    feature_alpha = data['feature_alpha']
    feature_hit = data['feature_hit']
    feature_names = data['feature_names']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

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

    # Evaluate ALL features with decay weights
    avg_alphas, hit_rates, n_valids = evaluate_all_features_weighted(
        feature_alpha, feature_hit, n_satellites, train_mask_np, weights
    )

    # Filter candidates
    positive_mask = (avg_alphas >= min_alpha) & (hit_rates >= min_hit_rate) & (n_valids > 0)
    negative_mask = (avg_alphas <= -min_alpha) & ((1 - hit_rates) >= min_hit_rate) & (n_valids > 0)

    # Prepare candidates
    candidates = []

    pos_indices = np.where(positive_mask)[0]
    for idx in pos_indices:
        candidates.append((idx, 'positive', avg_alphas[idx]))

    neg_indices = np.where(negative_mask)[0]
    for idx in neg_indices:
        rankings[:, :, idx] = 1.0 - rankings[:, :, idx]
        candidates.append((idx, 'inverted', -avg_alphas[idx]))

    if len(candidates) == 0:
        return [], None, rankings

    candidates.sort(key=lambda x: x[2], reverse=True)

    # Greedy forward selection
    selected = []
    best_alpha = -np.inf
    current_indices = np.array([], dtype=np.int64)

    for iteration in range(min(MAX_ENSEMBLE_SIZE, len(candidates))):
        best_add = None
        best_add_alpha = -np.inf

        remaining = [(idx, ptype, alpha) for idx, ptype, alpha in candidates
                     if idx not in current_indices]

        if len(remaining) == 0:
            break

        for feat_idx, pred_type, _ in remaining:
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
                best_add = feat_idx

        improvement = best_add_alpha - best_alpha if best_alpha > -np.inf else best_add_alpha

        if best_add is None or improvement < MIN_IMPROVEMENT:
            break

        selected.append({
            'idx': best_add,
            'name': feature_names[best_add],
            'improvement': improvement
        })
        current_indices = np.append(current_indices, best_add).astype(np.int64)
        best_alpha = best_add_alpha

    return selected, best_alpha if len(selected) > 0 else None, rankings


def greedy_search_with_recent_hit_penalty(data, n_satellites, train_mask, test_idx):
    """
    APPROACH 1: Weight feature scores by recent hit rate.

    Features with poor RECENT hit rate (last 12 months) get penalized
    in the scoring, even if their overall decay-weighted stats are good.
    """
    rankings = data['rankings'].copy()
    feature_alpha = data['feature_alpha']
    feature_hit = data['feature_hit']
    feature_names = data['feature_names']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    n_features = len(feature_names)
    train_mask_np = np.array(train_mask, dtype=np.bool_)

    # Create exponential decay weights (same as baseline)
    n_dates = len(train_mask)
    weights = np.zeros(n_dates, dtype=np.float64)
    decay_rate = np.log(2) / DECAY_HALF_LIFE_MONTHS

    for i in range(n_dates):
        if train_mask[i]:
            months_ago = test_idx - i
            weights[i] = np.exp(-decay_rate * months_ago)

    # Evaluate ALL features with decay weights (full history)
    avg_alphas, hit_rates, n_valids = evaluate_all_features_weighted(
        feature_alpha, feature_hit, n_satellites, train_mask_np, weights
    )

    # Also evaluate RECENT performance (last 12 months)
    lookback_start = max(0, test_idx - FEATURE_HIT_RATE_LOOKBACK)
    recent_alphas, recent_hit_rates, recent_n_valids = evaluate_all_features_recent(
        feature_alpha, feature_hit, n_satellites, train_mask_np, lookback_start
    )

    # Adjust alpha scores based on recent hit rate
    # If recent hit rate is low, penalize the alpha score
    adjusted_alphas = np.copy(avg_alphas)
    for i in range(n_features):
        if recent_n_valids[i] >= 6:  # Need at least 6 months of recent data
            recent_hr = recent_hit_rates[i]
            if recent_hr < 0.5:  # Feature has been performing poorly recently
                # Penalize by reducing effective alpha
                penalty = 1.0 - FEATURE_HIT_RATE_WEIGHT * (0.5 - recent_hr)
                adjusted_alphas[i] = avg_alphas[i] * penalty

    # Filter candidates using adjusted alphas but original hit rates
    positive_mask = (adjusted_alphas >= BASE_MIN_ALPHA) & (hit_rates >= BASE_MIN_HIT_RATE) & (n_valids > 0)
    negative_mask = (adjusted_alphas <= -BASE_MIN_ALPHA) & ((1 - hit_rates) >= BASE_MIN_HIT_RATE) & (n_valids > 0)

    candidates = []

    pos_indices = np.where(positive_mask)[0]
    for idx in pos_indices:
        candidates.append((idx, 'positive', adjusted_alphas[idx]))

    neg_indices = np.where(negative_mask)[0]
    for idx in neg_indices:
        rankings[:, :, idx] = 1.0 - rankings[:, :, idx]
        candidates.append((idx, 'inverted', -adjusted_alphas[idx]))

    if len(candidates) == 0:
        return [], None, rankings

    candidates.sort(key=lambda x: x[2], reverse=True)

    # Greedy forward selection (same as baseline)
    selected = []
    best_alpha = -np.inf
    current_indices = np.array([], dtype=np.int64)

    for iteration in range(min(MAX_ENSEMBLE_SIZE, len(candidates))):
        best_add = None
        best_add_alpha = -np.inf

        remaining = [(idx, ptype, alpha) for idx, ptype, alpha in candidates
                     if idx not in current_indices]

        if len(remaining) == 0:
            break

        for feat_idx, pred_type, _ in remaining:
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
                best_add = feat_idx

        improvement = best_add_alpha - best_alpha if best_alpha > -np.inf else best_add_alpha

        if best_add is None or improvement < MIN_IMPROVEMENT:
            break

        selected.append({
            'idx': best_add,
            'name': feature_names[best_add],
            'improvement': improvement
        })
        current_indices = np.append(current_indices, best_add).astype(np.int64)
        best_alpha = best_add_alpha

    return selected, best_alpha if len(selected) > 0 else None, rankings


def greedy_search_with_stricter_thresholds(data, n_satellites, train_mask, test_idx):
    """
    APPROACH 2: Use stricter minimum thresholds.

    Increase MIN_HIT_RATE from 0.55 to 0.60 to be more selective.
    """
    return greedy_search_baseline(
        data, n_satellites, train_mask, test_idx,
        min_alpha=BASE_MIN_ALPHA,
        min_hit_rate=0.60  # Stricter threshold
    )


def greedy_search_with_recent_only(data, n_satellites, train_mask, test_idx):
    """
    APPROACH 3: Only consider recent feature performance.

    Instead of using full decay-weighted history, only look at
    the last 24 months to determine feature quality.
    """
    rankings = data['rankings'].copy()
    feature_alpha = data['feature_alpha']
    feature_hit = data['feature_hit']
    feature_names = data['feature_names']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    n_features = len(feature_names)
    train_mask_np = np.array(train_mask, dtype=np.bool_)

    # Only look at recent data (24 months)
    lookback_start = max(0, test_idx - 24)

    # Create mask that only includes recent dates
    recent_mask = np.zeros(len(train_mask), dtype=np.bool_)
    for i in range(lookback_start, len(train_mask)):
        recent_mask[i] = train_mask[i]

    # Create flat weights for recent period (no decay within recent window)
    weights = np.zeros(len(train_mask), dtype=np.float64)
    for i in range(len(train_mask)):
        if recent_mask[i]:
            weights[i] = 1.0

    # Evaluate features on recent data only
    avg_alphas, hit_rates, n_valids = evaluate_all_features_weighted(
        feature_alpha, feature_hit, n_satellites, recent_mask, weights
    )

    # Filter candidates
    positive_mask = (avg_alphas >= BASE_MIN_ALPHA) & (hit_rates >= BASE_MIN_HIT_RATE) & (n_valids >= 12)
    negative_mask = (avg_alphas <= -BASE_MIN_ALPHA) & ((1 - hit_rates) >= BASE_MIN_HIT_RATE) & (n_valids >= 12)

    candidates = []

    pos_indices = np.where(positive_mask)[0]
    for idx in pos_indices:
        candidates.append((idx, 'positive', avg_alphas[idx]))

    neg_indices = np.where(negative_mask)[0]
    for idx in neg_indices:
        rankings[:, :, idx] = 1.0 - rankings[:, :, idx]
        candidates.append((idx, 'inverted', -avg_alphas[idx]))

    if len(candidates) == 0:
        return [], None, rankings

    candidates.sort(key=lambda x: x[2], reverse=True)

    # Greedy forward selection using FULL train mask for evaluation
    # (we filter candidates on recent, but evaluate ensemble on all history)
    selected = []
    best_alpha = -np.inf
    current_indices = np.array([], dtype=np.int64)

    for iteration in range(min(MAX_ENSEMBLE_SIZE, len(candidates))):
        best_add = None
        best_add_alpha = -np.inf

        remaining = [(idx, ptype, alpha) for idx, ptype, alpha in candidates
                     if idx not in current_indices]

        if len(remaining) == 0:
            break

        for feat_idx, pred_type, _ in remaining:
            test_indices = np.append(current_indices, feat_idx).astype(np.int64)

            # Evaluate on full training set
            alphas = compute_ensemble_alpha_all_dates(
                rankings, test_indices, alpha_matrix, alpha_valid, train_mask_np, n_satellites
            )

            valid_alphas = alphas[~np.isnan(alphas)]
            if len(valid_alphas) == 0:
                continue

            avg_alpha = np.mean(valid_alphas)

            if avg_alpha > best_add_alpha:
                best_add_alpha = avg_alpha
                best_add = feat_idx

        improvement = best_add_alpha - best_alpha if best_alpha > -np.inf else best_add_alpha

        if best_add is None or improvement < MIN_IMPROVEMENT:
            break

        selected.append({
            'idx': best_add,
            'name': feature_names[best_add],
            'improvement': improvement
        })
        current_indices = np.append(current_indices, best_add).astype(np.int64)
        best_alpha = best_add_alpha

    return selected, best_alpha if len(selected) > 0 else None, rankings


def greedy_search_with_feature_decay(data, n_satellites, train_mask, test_idx):
    """
    APPROACH 4: Apply additional decay to poorly performing features.

    Features that have been performing worse recently get additional
    time-decay on top of the standard decay.
    """
    rankings = data['rankings'].copy()
    feature_alpha = data['feature_alpha']
    feature_hit = data['feature_hit']
    feature_names = data['feature_names']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    n_features = len(feature_names)
    train_mask_np = np.array(train_mask, dtype=np.bool_)

    # Create exponential decay weights
    n_dates = len(train_mask)
    base_weights = np.zeros(n_dates, dtype=np.float64)
    decay_rate = np.log(2) / DECAY_HALF_LIFE_MONTHS

    for i in range(n_dates):
        if train_mask[i]:
            months_ago = test_idx - i
            base_weights[i] = np.exp(-decay_rate * months_ago)

    # First pass: evaluate features to find poorly performing ones
    avg_alphas, hit_rates, n_valids = evaluate_all_features_weighted(
        feature_alpha, feature_hit, n_satellites, train_mask_np, base_weights
    )

    # Also get recent performance
    lookback_start = max(0, test_idx - 12)
    recent_alphas, recent_hit_rates, recent_n_valids = evaluate_all_features_recent(
        feature_alpha, feature_hit, n_satellites, train_mask_np, lookback_start
    )

    # Create per-feature adjusted weights
    # For features with declining performance, apply steeper decay
    feature_weights = []
    for feat_idx in range(n_features):
        if recent_n_valids[feat_idx] >= 6:
            # Compare recent vs overall hit rate
            overall_hr = hit_rates[feat_idx]
            recent_hr = recent_hit_rates[feat_idx]

            if recent_hr < overall_hr - 0.1:  # Feature is declining
                # Apply steeper decay (shorter half-life)
                steep_decay_rate = np.log(2) / (DECAY_HALF_LIFE_MONTHS / 2)
                weights = np.zeros(n_dates, dtype=np.float64)
                for i in range(n_dates):
                    if train_mask[i]:
                        months_ago = test_idx - i
                        weights[i] = np.exp(-steep_decay_rate * months_ago)
                feature_weights.append(weights)
            else:
                feature_weights.append(base_weights)
        else:
            feature_weights.append(base_weights)

    # Re-evaluate features with adjusted weights
    adjusted_alphas = np.empty(n_features, dtype=np.float64)
    adjusted_hit_rates = np.empty(n_features, dtype=np.float64)

    for feat_idx in range(n_features):
        weights = feature_weights[feat_idx]
        avg_alpha, hit_rate, n_valid = evaluate_feature_weighted(
            feature_alpha, feature_hit, feat_idx, n_satellites, train_mask_np, weights
        )
        adjusted_alphas[feat_idx] = avg_alpha
        adjusted_hit_rates[feat_idx] = hit_rate

    # Filter candidates
    positive_mask = (adjusted_alphas >= BASE_MIN_ALPHA) & (adjusted_hit_rates >= BASE_MIN_HIT_RATE) & (n_valids > 0)
    negative_mask = (adjusted_alphas <= -BASE_MIN_ALPHA) & ((1 - adjusted_hit_rates) >= BASE_MIN_HIT_RATE) & (n_valids > 0)

    candidates = []

    pos_indices = np.where(positive_mask)[0]
    for idx in pos_indices:
        candidates.append((idx, 'positive', adjusted_alphas[idx]))

    neg_indices = np.where(negative_mask)[0]
    for idx in neg_indices:
        rankings[:, :, idx] = 1.0 - rankings[:, :, idx]
        candidates.append((idx, 'inverted', -adjusted_alphas[idx]))

    if len(candidates) == 0:
        return [], None, rankings

    candidates.sort(key=lambda x: x[2], reverse=True)

    # Greedy forward selection
    selected = []
    best_alpha = -np.inf
    current_indices = np.array([], dtype=np.int64)

    for iteration in range(min(MAX_ENSEMBLE_SIZE, len(candidates))):
        best_add = None
        best_add_alpha = -np.inf

        remaining = [(idx, ptype, alpha) for idx, ptype, alpha in candidates
                     if idx not in current_indices]

        if len(remaining) == 0:
            break

        for feat_idx, pred_type, _ in remaining:
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
                best_add = feat_idx

        improvement = best_add_alpha - best_alpha if best_alpha > -np.inf else best_add_alpha

        if best_add is None or improvement < MIN_IMPROVEMENT:
            break

        selected.append({
            'idx': best_add,
            'name': feature_names[best_add],
            'improvement': improvement
        })
        current_indices = np.append(current_indices, best_add).astype(np.int64)
        best_alpha = best_add_alpha

    return selected, best_alpha if len(selected) > 0 else None, rankings


# ============================================================
# SELECT ETFs
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

def walk_forward_backtest(data, n_satellites, approach='baseline', show_progress=True):
    """
    Run walk-forward backtest with specified feature selection approach.

    Args:
        approach: 'baseline', 'recent_hit_penalty', 'stricter', 'recent_only', 'feature_decay'
    """
    dates = data['dates']
    isins = data['isins']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0]

    if len(test_start_idx) == 0:
        return pd.DataFrame()

    test_start_idx = test_start_idx[0]

    # Select approach function
    approach_funcs = {
        'baseline': greedy_search_baseline,
        'recent_hit_penalty': greedy_search_with_recent_hit_penalty,
        'stricter': greedy_search_with_stricter_thresholds,
        'recent_only': greedy_search_with_recent_only,
        'feature_decay': greedy_search_with_feature_decay,
    }

    if approach not in approach_funcs:
        raise ValueError(f"Unknown approach: {approach}")

    search_func = approach_funcs[approach]

    if show_progress:
        print(f"\n  Testing approach: {approach}")
        print(f"  Test period: {dates[test_start_idx].date()} to {dates[-1].date()}")

    results = []
    feature_indices = None
    rankings = None
    months_since_reopt = REOPTIMIZATION_FREQUENCY

    iterator = range(test_start_idx, len(dates))
    if show_progress:
        iterator = tqdm(iterator, desc=f"{approach[:15]:15s}")

    for test_idx in iterator:
        test_date = dates[test_idx]

        # Re-optimize feature selection periodically
        if months_since_reopt >= REOPTIMIZATION_FREQUENCY:
            train_mask = dates < test_date

            # Use specified approach
            selected, train_perf, modified_rankings = search_func(
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

def analyze_results(results_df, approach_name):
    """Analyze backtest results."""
    if len(results_df) == 0:
        return None

    n_months = len(results_df)
    avg_alpha = results_df['avg_alpha'].mean()
    hit_rate = (results_df['avg_alpha'] > 0).mean()
    total_alpha = results_df['avg_alpha'].sum()
    ann_alpha = avg_alpha * 12

    return {
        'approach': approach_name,
        'n_months': n_months,
        'avg_alpha': avg_alpha,
        'ann_alpha_pct': ann_alpha * 100,
        'hit_rate': hit_rate,
        'total_alpha': total_alpha
    }


# ============================================================
# MAIN
# ============================================================

def main():
    """Run experimental backtests comparing different approaches."""
    print("=" * 70)
    print("EXPERIMENTAL: FEATURE-LEVEL FILTERING APPROACHES")
    print("=" * 70)

    # Load data
    data = load_data()

    # Define approaches to test
    approaches = [
        'baseline',           # Original approach
        'recent_hit_penalty', # Penalize features with poor recent hit rate
        'stricter',           # Stricter min hit rate (0.60 vs 0.55)
        'recent_only',        # Only consider last 24 months for feature quality
        'feature_decay',      # Apply steeper decay to declining features
    ]

    n_satellites = 5  # Focus on N=5

    all_results = []

    for approach in approaches:
        print(f"\n{'='*60}")
        print(f"Testing: {approach}")
        print('='*60)

        start_time = time.time()
        results_df = walk_forward_backtest(data, n_satellites, approach=approach, show_progress=True)
        elapsed = time.time() - start_time

        metrics = analyze_results(results_df, approach)
        if metrics:
            metrics['time_sec'] = elapsed
            all_results.append(metrics)

            print(f"\n  Results for {approach}:")
            print(f"    Months: {metrics['n_months']}")
            print(f"    Avg Alpha: {metrics['avg_alpha']*100:.3f}%/month")
            print(f"    Ann Alpha: {metrics['ann_alpha_pct']:.1f}%/year")
            print(f"    Hit Rate: {metrics['hit_rate']*100:.1f}%")
            print(f"    Time: {elapsed:.1f}s")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values('ann_alpha_pct', ascending=False)

    print("\nRanked by Annual Alpha:")
    print("-" * 70)
    for _, row in summary_df.iterrows():
        print(f"  {row['approach']:25s}  Ann: {row['ann_alpha_pct']:6.1f}%  Hit: {row['hit_rate']*100:5.1f}%")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / 'feature_filtering_comparison.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return summary_df


if __name__ == "__main__":
    main()
