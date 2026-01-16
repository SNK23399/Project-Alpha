"""
Step 7: Multi-Horizon Consensus Backtest (Walk-Forward Pipeline)
================================================================

ULTRA-OPTIMIZED VERSION with:
- Numba JIT compilation for all hot loops (~100x speedup)
- Vectorized feature evaluation using precomputed matrices
- Parallel processing with ThreadPoolExecutor
- Monthly reoptimization with cached computations

This script implements multi-horizon consensus voting for ETF selection:
- Loads precomputed data for multiple holding horizons (1-12 months)
- At each test date, selects ETFs using consensus across horizons
- Measures performance using 1-month forward alpha (primary horizon)

Consensus Methods:
- primary_only: Just use 1-month rankings (baseline)
- unanimous: ETF must be top-N in ALL horizons
- majority: ETF must be top-N in >50% of horizons
- weighted_avg: Weighted average of rankings across horizons
- primary_veto: Use 1-month but veto if 2+ longer horizons disagree

Usage:
    python 7_multi_horizon_backtest.py

Output:
    data/backtest_results/multi_horizon_summary.csv
    data/backtest_results/multi_horizon_N{n}_{method}.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
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

# Horizons to use for consensus
PRIMARY_HORIZON = 1  # Primary signal (1 month) - used for alpha measurement

# Different confirmation horizon configurations to test
CONFIRMATION_CONFIGS = {
    'none': [],                           # Baseline: primary only
    'short': [2, 3],                      # Short-term confirmation (2-3 months)
    'medium': [2, 3, 4, 5, 6],            # Medium-term confirmation (2-6 months)
    'long': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # All horizons (2-12 months)
}

# Default configuration (can be overridden)
CONFIRMATION_HORIZONS = CONFIRMATION_CONFIGS['long']
ALL_HORIZONS = [PRIMARY_HORIZON] + list(set(h for horizons in CONFIRMATION_CONFIGS.values() for h in horizons))

# N values to test
N_SATELLITES_TO_TEST = [1, 3, 5, 10]

# Consensus methods to test
CONSENSUS_METHODS = [
    'primary_only',      # Baseline: just use 1-month
    'unanimous',         # All horizons must agree (with fallback)
    'majority',          # At least 50% of horizons must agree
    'weighted_avg',      # Weighted average of rankings
    'primary_veto',      # Use 1-month but veto if 2+ longer horizons disagree
]

# Test all combinations of methods and confirmation configs
TEST_CONFIRMATION_CONFIGS = True  # Set to False to only test 'long' config

# Weights for weighted average (shorter horizons get more weight)
HORIZON_WEIGHTS = {
    1: 0.30,   # 30% weight on 1-month (strongest signal)
    2: 0.20,   # 20% weight on 2-month
    3: 0.15,   # 15% weight on 3-month
    4: 0.10,   # 10% weight on 4-month
    5: 0.07,   # 7% weight on 5-month
    6: 0.05,   # 5% weight on 6-month
    7: 0.04,   # 4% weight on 7-month
    8: 0.03,   # 3% weight on 8-month
    9: 0.02,   # 2% weight on 9-month
    10: 0.015, # 1.5% weight on 10-month
    11: 0.015, # 1.5% weight on 11-month
    12: 0.02,  # 2% weight on 12-month
}  # Total = 100%

# Training parameters
MIN_TRAINING_MONTHS = 36  # Minimum training history
REOPTIMIZATION_FREQUENCY = 1  # Re-optimize every N months

# Feature selection criteria
MIN_ALPHA = 0.001        # Minimum 0.1% alpha
MIN_HIT_RATE = 0.55      # Minimum 55% hit rate
MAX_ENSEMBLE_SIZE = 20   # Maximum features in ensemble
MIN_IMPROVEMENT = 0.0001 # Minimum improvement to add feature

# Parallel processing
N_CORES = max(1, cpu_count() - 1)

# Data and output directories
DATA_DIR = Path(__file__).parent / 'data'
OUTPUT_DIR = DATA_DIR / 'backtest_results'


# ============================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# ============================================================

@njit(cache=True)
def evaluate_feature_numba(feature_alpha, feature_hit, feat_idx, n_satellites, date_mask):
    """
    Numba-optimized feature evaluation.
    Returns: (avg_alpha, hit_rate, n_valid) or (-999, -999, 0) if invalid
    """
    n_dates = len(date_mask)
    sum_alpha = 0.0
    sum_hit = 0.0
    n_valid = 0

    for i in range(n_dates):
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
def evaluate_all_features_parallel(feature_alpha, feature_hit, n_satellites, date_mask):
    """
    Evaluate ALL features in parallel using Numba prange.
    Returns arrays of (avg_alpha, hit_rate, n_valid) for each feature.
    """
    n_features = feature_alpha.shape[1]
    avg_alphas = np.empty(n_features, dtype=np.float64)
    hit_rates = np.empty(n_features, dtype=np.float64)
    n_valids = np.empty(n_features, dtype=np.int64)

    for feat_idx in prange(n_features):
        avg_alpha, hit_rate, n_valid = evaluate_feature_numba(
            feature_alpha, feature_hit, feat_idx, n_satellites, date_mask
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
    valid_count = np.zeros(n_isins, dtype=np.int64)

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
            valid_count[i] = count
        else:
            scores[i] = np.nan
            valid_count[i] = 0

    # Count valid scores
    n_valid = 0
    for i in range(n_isins):
        if not np.isnan(scores[i]):
            n_valid += 1

    if n_valid < n_satellites:
        return np.nan

    # Find top-N by score (higher is better)
    # Simple selection: find indices with highest scores
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
    Returns array of selected ISIN indices, or empty array if not enough valid.

    Args:
        rankings_slice: 2D array (n_isins, n_features) of rankings for this date
        feature_indices: array of feature indices to use
        n_satellites: number of ISINs to select
        alpha_valid_mask: optional boolean array indicating which ISINs have valid alpha
                         (only these ISINs will be considered for selection)
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
# LOAD DATA FOR ALL HORIZONS
# ============================================================

def load_horizon_data(holding_months):
    """Load precomputed data for a single horizon."""
    data_dir = DATA_DIR

    # Load forward alpha
    alpha_file = data_dir / f'forward_alpha_{holding_months}month.parquet'
    if not alpha_file.exists():
        return None

    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])

    # Load rankings matrix
    rankings_file = data_dir / f'rankings_matrix_{holding_months}month.npz'
    if not rankings_file.exists():
        return None

    npz_data = np.load(rankings_file, allow_pickle=True)
    rankings = npz_data['rankings'].astype(np.float64)  # Ensure float64 for numba
    dates = pd.to_datetime(npz_data['dates'])
    isins = npz_data['isins']
    feature_names = list(npz_data['features'])

    # Load feature-alpha matrix
    feature_alpha_file = data_dir / f'feature_alpha_{holding_months}month.npz'
    feature_alpha = None
    feature_hit = None

    if feature_alpha_file.exists():
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


def load_all_horizons():
    """Load data for all horizons."""
    print("=" * 60)
    print("LOADING DATA FOR ALL HORIZONS")
    print("=" * 60)

    horizon_data = {}

    for h in ALL_HORIZONS:
        data = load_horizon_data(h)
        if data is not None:
            horizon_data[h] = data
            print(f"  [OK] {h}-month: {data['rankings'].shape}")
        else:
            print(f"  [--] {h}-month: NOT FOUND")

    if PRIMARY_HORIZON not in horizon_data:
        raise FileNotFoundError(f"Primary horizon ({PRIMARY_HORIZON}-month) data not found!")

    print(f"\nLoaded {len(horizon_data)} horizons: {list(horizon_data.keys())}")

    return horizon_data


# ============================================================
# FAST FEATURE SELECTION (using precomputed feature-alpha)
# ============================================================

def fast_greedy_search_optimized(data, n_satellites, train_mask):
    """
    Ultra-fast greedy ensemble search using Numba-parallelized functions.
    """
    rankings = data['rankings'].copy()
    feature_alpha = data['feature_alpha']
    feature_hit = data['feature_hit']
    feature_names = data['feature_names']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    n_features = len(feature_names)
    train_mask_np = np.array(train_mask, dtype=np.bool_)

    # Step 1: Evaluate ALL features in parallel (Numba)
    avg_alphas, hit_rates, n_valids = evaluate_all_features_parallel(
        feature_alpha, feature_hit, n_satellites, train_mask_np
    )

    # Step 2: Filter candidates (vectorized)
    positive_mask = (avg_alphas >= MIN_ALPHA) & (hit_rates >= MIN_HIT_RATE) & (n_valids > 0)
    negative_mask = (avg_alphas <= -MIN_ALPHA) & ((1 - hit_rates) >= MIN_HIT_RATE) & (n_valids > 0)

    # Prepare candidates list
    candidates = []

    # Positive features
    pos_indices = np.where(positive_mask)[0]
    for idx in pos_indices:
        candidates.append((idx, 'positive', avg_alphas[idx]))

    # Negative features (invert rankings)
    neg_indices = np.where(negative_mask)[0]
    for idx in neg_indices:
        rankings[:, :, idx] = 1.0 - rankings[:, :, idx]
        candidates.append((idx, 'inverted', -avg_alphas[idx]))

    if len(candidates) == 0:
        return [], None, rankings

    # Sort by alpha (descending)
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Step 3: Greedy forward selection with Numba-accelerated ensemble evaluation
    selected = []
    best_alpha = -np.inf
    current_indices = np.array([], dtype=np.int64)

    for iteration in range(min(MAX_ENSEMBLE_SIZE, len(candidates))):
        best_add = None
        best_add_alpha = -np.inf

        # Try adding each remaining candidate
        remaining = [(idx, ptype, alpha) for idx, ptype, alpha in candidates
                     if idx not in current_indices]

        if len(remaining) == 0:
            break

        for feat_idx, pred_type, _ in remaining:
            test_indices = np.append(current_indices, feat_idx).astype(np.int64)

            # Compute ensemble alpha using Numba
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

        # Check improvement
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

    return selected, best_perf, rankings


# ============================================================
# CONSENSUS METHODS
# ============================================================

def apply_consensus(horizon_data, test_idx, n_satellites, method, feature_indices_per_horizon, rankings_per_horizon, confirmation_horizons=None, alpha_valid_mask=None):
    """
    Apply consensus method to select ETFs across multiple horizons.

    Args:
        confirmation_horizons: List of confirmation horizons to use (default: CONFIRMATION_HORIZONS)
        alpha_valid_mask: Boolean array indicating which ISINs have valid alpha for this date
                         (only these ISINs will be considered for selection)
    """
    if confirmation_horizons is None:
        confirmation_horizons = CONFIRMATION_HORIZONS

    primary_data = horizon_data[PRIMARY_HORIZON]
    isins = primary_data['isins']
    n_isins = len(isins)

    if method == 'primary_only':
        # Just use 1-month rankings
        if PRIMARY_HORIZON not in feature_indices_per_horizon:
            return None

        features = feature_indices_per_horizon[PRIMARY_HORIZON]
        if features is None or len(features) == 0:
            return None

        rankings = rankings_per_horizon[PRIMARY_HORIZON]
        feature_arr = np.array(features, dtype=np.int64)

        selected = select_top_n_isins(rankings[test_idx], feature_arr, n_satellites, alpha_valid_mask)
        if selected[0] == -1:
            return None
        return selected.tolist()

    elif method == 'unanimous':
        # ETF must be in top-N for ALL available horizons
        # Use ISIN names to ensure proper mapping between horizons
        top_sets = []  # Sets of ISIN names (not indices)
        primary_isin_to_idx = primary_data['isin_to_idx']

        # Only check primary + confirmation horizons
        horizons_to_check = [PRIMARY_HORIZON] + list(confirmation_horizons)
        for h in horizons_to_check:
            features = feature_indices_per_horizon.get(h)
            if h not in horizon_data or features is None or len(features) == 0:
                continue

            rankings = rankings_per_horizon[h]
            h_data = horizon_data[h]

            test_date = primary_data['dates'][test_idx]
            if test_date not in h_data['date_to_idx']:
                continue
            h_test_idx = h_data['date_to_idx'][test_date]

            feature_arr = np.array(features, dtype=np.int64)
            h_isins = h_data['isins']
            h_n_isins = len(h_isins)
            # Take top 2*N to allow flexibility
            selected = select_top_n_isins(rankings[h_test_idx], feature_arr, min(n_satellites * 2, h_n_isins))
            if selected[0] != -1:
                # Convert to ISIN names
                selected_isins = set(h_isins[idx] for idx in selected)
                top_sets.append(selected_isins)

        if len(top_sets) == 0:
            return None

        # Intersection of all sets (by ISIN name)
        unanimous_isins = set.intersection(*top_sets)

        if len(unanimous_isins) >= n_satellites:
            # Convert back to primary indices and use primary scores to pick best N
            features = feature_indices_per_horizon.get(PRIMARY_HORIZON, [])
            if len(features) == 0:
                # Just return first N by primary index
                return [primary_isin_to_idx[isin] for isin in list(unanimous_isins)[:n_satellites]
                        if isin in primary_isin_to_idx]

            rankings = rankings_per_horizon[PRIMARY_HORIZON]
            feature_arr = np.array(features, dtype=np.int64)

            # Compute scores for unanimous candidates - only include ISINs with valid alpha
            scores = []
            for isin in unanimous_isins:
                if isin not in primary_isin_to_idx:
                    continue
                idx = primary_isin_to_idx[isin]
                # Skip if this ISIN doesn't have valid alpha
                if alpha_valid_mask is not None and not alpha_valid_mask[idx]:
                    continue
                score_sum = 0.0
                count = 0
                for feat_idx in feature_arr:
                    val = rankings[test_idx, idx, feat_idx]
                    if not np.isnan(val):
                        score_sum += val
                        count += 1
                if count > 0:
                    scores.append((idx, score_sum / count))
                else:
                    scores.append((idx, -999))

            scores.sort(key=lambda x: x[1], reverse=True)
            return [idx for idx, _ in scores[:n_satellites]]
        else:
            # Fall back to primary
            return apply_consensus(horizon_data, test_idx, n_satellites, 'primary_only',
                                   feature_indices_per_horizon, rankings_per_horizon,
                                   alpha_valid_mask=alpha_valid_mask)

    elif method == 'majority':
        # Count votes across horizons
        # Use ISIN names to map between horizons (they may have different ISIN counts)
        vote_counts = np.zeros(n_isins, dtype=np.int32)
        n_horizons_checked = 0
        primary_isin_to_idx = primary_data['isin_to_idx']

        # Only check primary + confirmation horizons
        horizons_to_check = [PRIMARY_HORIZON] + list(confirmation_horizons)
        for h in horizons_to_check:
            features = feature_indices_per_horizon.get(h)
            if h not in horizon_data or features is None or len(features) == 0:
                continue

            rankings = rankings_per_horizon[h]
            h_data = horizon_data[h]

            test_date = primary_data['dates'][test_idx]
            if test_date not in h_data['date_to_idx']:
                continue
            h_test_idx = h_data['date_to_idx'][test_date]

            feature_arr = np.array(features, dtype=np.int64)
            h_isins = h_data['isins']
            h_n_isins = len(h_isins)
            n_top = min(n_satellites * 2, h_n_isins)
            selected = select_top_n_isins(rankings[h_test_idx], feature_arr, n_top)

            if selected[0] != -1:
                for h_idx in selected:
                    # Map horizon ISIN index to primary ISIN index
                    h_isin = h_isins[h_idx]
                    if h_isin in primary_isin_to_idx:
                        primary_idx = primary_isin_to_idx[h_isin]
                        vote_counts[primary_idx] += 1
                n_horizons_checked += 1

        if n_horizons_checked == 0:
            return None

        # Select ETFs with most votes, breaking ties with primary score
        features = feature_indices_per_horizon.get(PRIMARY_HORIZON, [])
        if len(features) > 0:
            rankings = rankings_per_horizon[PRIMARY_HORIZON]
            feature_arr = np.array(features, dtype=np.int64)
            primary_scores = np.zeros(n_isins)
            for i in range(n_isins):
                score_sum = 0.0
                count = 0
                for feat_idx in feature_arr:
                    val = rankings[test_idx, i, feat_idx]
                    if not np.isnan(val):
                        score_sum += val
                        count += 1
                primary_scores[i] = score_sum / count if count > 0 else -999
        else:
            primary_scores = np.zeros(n_isins)

        # Create (votes, primary_score, idx) tuples - only for ISINs with valid alpha
        candidates = [(vote_counts[i], primary_scores[i], i)
                      for i in range(n_isins)
                      if vote_counts[i] > 0 and (alpha_valid_mask is None or alpha_valid_mask[i])]
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

        return [idx for _, _, idx in candidates[:n_satellites]]

    elif method == 'weighted_avg':
        # Weighted average of scores across horizons
        # Use ISIN names to map between horizons (they may have different ISIN counts)
        weighted_scores = np.zeros(n_isins, dtype=np.float64)
        weight_counts = np.zeros(n_isins, dtype=np.float64)
        primary_isin_to_idx = primary_data['isin_to_idx']

        # Only check primary + confirmation horizons
        horizons_to_check = [PRIMARY_HORIZON] + list(confirmation_horizons)
        for h in horizons_to_check:
            features = feature_indices_per_horizon.get(h)
            if h not in horizon_data or features is None or len(features) == 0:
                continue

            weight = HORIZON_WEIGHTS.get(h, 0.01)
            rankings = rankings_per_horizon[h]
            h_data = horizon_data[h]

            test_date = primary_data['dates'][test_idx]
            if test_date not in h_data['date_to_idx']:
                continue
            h_test_idx = h_data['date_to_idx'][test_date]

            feature_arr = np.array(features, dtype=np.int64)
            h_isins = h_data['isins']
            h_n_isins = len(h_isins)

            for h_i in range(h_n_isins):
                h_isin = h_isins[h_i]
                # Map to primary ISIN index
                if h_isin not in primary_isin_to_idx:
                    continue
                primary_i = primary_isin_to_idx[h_isin]

                score_sum = 0.0
                count = 0
                for feat_idx in feature_arr:
                    val = rankings[h_test_idx, h_i, feat_idx]
                    if not np.isnan(val):
                        score_sum += val
                        count += 1
                if count > 0:
                    weighted_scores[primary_i] += weight * (score_sum / count)
                    weight_counts[primary_i] += weight

        # Normalize by total weight per ISIN - only consider ISINs with valid alpha
        valid_mask = weight_counts > 0
        if alpha_valid_mask is not None:
            valid_mask = valid_mask & alpha_valid_mask

        if valid_mask.sum() < n_satellites:
            return None

        weighted_scores[valid_mask] /= weight_counts[valid_mask]

        valid_indices = np.where(valid_mask)[0]
        valid_scores = weighted_scores[valid_mask]

        top_k = np.argpartition(valid_scores, -n_satellites)[-n_satellites:]
        return valid_indices[top_k].tolist()

    elif method == 'primary_veto':
        # Use primary, but veto if 2+ longer horizons disagree
        features = feature_indices_per_horizon.get(PRIMARY_HORIZON, [])
        if len(features) == 0:
            return None

        rankings = rankings_per_horizon[PRIMARY_HORIZON]
        feature_arr = np.array(features, dtype=np.int64)
        primary_isin_to_idx = primary_data['isin_to_idx']

        primary_scores = np.zeros(n_isins)
        for i in range(n_isins):
            score_sum = 0.0
            count = 0
            for feat_idx in feature_arr:
                val = rankings[test_idx, i, feat_idx]
                if not np.isnan(val):
                    score_sum += val
                    count += 1
            primary_scores[i] = score_sum / count if count > 0 else np.nan

        # Count disagreements for each ISIN
        disagreements = np.zeros(n_isins, dtype=np.int32)

        for h in confirmation_horizons:
            h_features = feature_indices_per_horizon.get(h)
            if h not in horizon_data or h_features is None or len(h_features) == 0:
                continue

            h_rankings = rankings_per_horizon[h]
            h_data = horizon_data[h]

            test_date = primary_data['dates'][test_idx]
            if test_date not in h_data['date_to_idx']:
                continue
            h_test_idx = h_data['date_to_idx'][test_date]

            h_feature_arr = np.array(h_features, dtype=np.int64)
            h_isins = h_data['isins']
            h_n_isins = len(h_isins)

            # Compute scores for this horizon's ISINs
            h_scores_local = np.zeros(h_n_isins)
            for h_i in range(h_n_isins):
                score_sum = 0.0
                count = 0
                for feat_idx in h_feature_arr:
                    val = h_rankings[h_test_idx, h_i, feat_idx]
                    if not np.isnan(val):
                        score_sum += val
                        count += 1
                h_scores_local[h_i] = score_sum / count if count > 0 else np.nan

            # Mark ISINs in bottom half as disagreeing
            valid_mask_local = ~np.isnan(h_scores_local)
            if valid_mask_local.sum() > 0:
                median_score = np.nanmedian(h_scores_local[valid_mask_local])
                for h_i in range(h_n_isins):
                    if valid_mask_local[h_i] and h_scores_local[h_i] < median_score:
                        h_isin = h_isins[h_i]
                        if h_isin in primary_isin_to_idx:
                            primary_i = primary_isin_to_idx[h_isin]
                            disagreements[primary_i] += 1

        # Select from primary, but skip if 2+ disagreements
        # Only consider ISINs with valid alpha
        valid_mask = ~np.isnan(primary_scores)
        if alpha_valid_mask is not None:
            valid_mask = valid_mask & alpha_valid_mask

        if valid_mask.sum() < n_satellites:
            return None

        # Sort by primary score, then filter by disagreements
        candidates = [(primary_scores[i], disagreements[i], i)
                      for i in range(n_isins) if valid_mask[i]]
        candidates.sort(key=lambda x: x[0], reverse=True)

        selected = []
        for score, n_disagree, idx in candidates:
            if n_disagree < 2:  # Accept if fewer than 2 disagreements
                selected.append(idx)
                if len(selected) >= n_satellites:
                    break

        # Fall back if not enough
        if len(selected) < n_satellites:
            for score, n_disagree, idx in candidates:
                if idx not in selected:
                    selected.append(idx)
                    if len(selected) >= n_satellites:
                        break

        return selected

    else:
        raise ValueError(f"Unknown consensus method: {method}")


# ============================================================
# WALK-FORWARD BACKTEST WITH CONSENSUS
# ============================================================

def walk_forward_backtest_consensus(horizon_data, n_satellites, method, confirmation_horizons=None, show_progress=True):
    """
    Run walk-forward backtest with multi-horizon consensus.

    OPTIMIZED: Uses Numba-JIT compiled functions for feature selection.

    Args:
        horizon_data: Dict of horizon -> data
        n_satellites: Number of satellites to select
        method: Consensus method name
        confirmation_horizons: List of confirmation horizons to use (default: CONFIRMATION_HORIZONS)
        show_progress: Show progress bar
    """
    if confirmation_horizons is None:
        confirmation_horizons = CONFIRMATION_HORIZONS

    primary_data = horizon_data[PRIMARY_HORIZON]
    dates = primary_data['dates']
    isins = primary_data['isins']
    alpha_matrix = primary_data['alpha_matrix']
    alpha_valid = primary_data['alpha_valid']

    # Find start date
    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0]

    if len(test_start_idx) == 0:
        return pd.DataFrame()

    test_start_idx = test_start_idx[0]

    if show_progress:
        print(f"\n  Testing N={n_satellites}, Method={method}, Confirm={len(confirmation_horizons)} horizons")
        print(f"  Test period: {dates[test_start_idx].date()} to {dates[-1].date()}")

    results = []
    feature_indices_per_horizon = {}
    rankings_per_horizon = {}
    months_since_reopt = REOPTIMIZATION_FREQUENCY  # Force first optimization

    # Walk forward through time
    iterator = range(test_start_idx, len(dates))
    if show_progress:
        iterator = tqdm(iterator, desc=f"N={n_satellites} {method[:8]:8s}")

    for test_idx in iterator:
        test_date = dates[test_idx]

        # Re-optimize if needed
        if months_since_reopt >= REOPTIMIZATION_FREQUENCY:
            # Run feature selection on PRIMARY horizon
            train_mask = dates < test_date

            selected, train_perf, modified_rankings = fast_greedy_search_optimized(
                primary_data, n_satellites, train_mask
            )

            if len(selected) > 0:
                feature_indices_per_horizon[PRIMARY_HORIZON] = [f['idx'] for f in selected]
                rankings_per_horizon[PRIMARY_HORIZON] = modified_rankings
            else:
                feature_indices_per_horizon[PRIMARY_HORIZON] = None
                rankings_per_horizon[PRIMARY_HORIZON] = None

            # For confirmation horizons, use the SAME feature indices but their own rankings
            for h in confirmation_horizons:
                if h not in horizon_data:
                    feature_indices_per_horizon[h] = None
                    rankings_per_horizon[h] = None
                    continue

                h_data = horizon_data[h]
                if test_date not in h_data['date_to_idx']:
                    feature_indices_per_horizon[h] = None
                    rankings_per_horizon[h] = None
                    continue

                if feature_indices_per_horizon[PRIMARY_HORIZON] is not None:
                    feature_indices_per_horizon[h] = feature_indices_per_horizon[PRIMARY_HORIZON]
                    rankings_per_horizon[h] = h_data['rankings'].copy()
                else:
                    feature_indices_per_horizon[h] = None
                    rankings_per_horizon[h] = None

            months_since_reopt = 0

        months_since_reopt += 1

        # Check if primary horizon has features
        if feature_indices_per_horizon.get(PRIMARY_HORIZON) is None:
            continue

        # Apply consensus method to select ETFs
        # Pass confirmation_horizons and alpha_valid_mask to apply_consensus
        # The alpha_valid_mask ensures we only select ISINs that have forward alpha data
        selected_isin_indices = apply_consensus(
            horizon_data, test_idx, n_satellites, method,
            feature_indices_per_horizon, rankings_per_horizon,
            confirmation_horizons=confirmation_horizons,
            alpha_valid_mask=alpha_valid[test_idx]
        )

        if selected_isin_indices is None or len(selected_isin_indices) < n_satellites:
            continue

        # Measure alpha using PRIMARY horizon (1-month forward)
        # All selected ISINs should have valid alpha now (due to alpha_valid_mask)
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
            'method': method,
            'avg_alpha': avg_alpha,
            'n_selected': len(selected_isins),
            'selected_isins': ','.join(selected_isins)
        })

    return pd.DataFrame(results)


# ============================================================
# ANALYZE RESULTS
# ============================================================

def analyze_results(results_df, n_satellites, method):
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
        'method': method,
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
    """Run multi-horizon consensus backtest with different confirmation configurations."""
    print("=" * 60)
    print("MULTI-HORIZON CONSENSUS BACKTEST")
    print("=" * 60)
    print(f"\nPrimary horizon: {PRIMARY_HORIZON} month")
    print(f"N values to test: {N_SATELLITES_TO_TEST}")
    print(f"Consensus methods: {CONSENSUS_METHODS}")
    print(f"Confirmation configs: {list(CONFIRMATION_CONFIGS.keys())}")
    print(f"Test all configs: {TEST_CONFIRMATION_CONFIGS}")
    print(f"CPU cores available: {N_CORES}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data for all horizons
    horizon_data = load_all_horizons()

    # Warm up Numba JIT (first call compiles)
    print("\nWarming up Numba JIT compilation...")
    warmup_start = time.time()
    primary_data = horizon_data[PRIMARY_HORIZON]
    _ = evaluate_all_features_parallel(
        primary_data['feature_alpha'][:10],
        primary_data['feature_hit'][:10],
        1,
        np.ones(10, dtype=np.bool_)
    )
    print(f"  JIT warmup complete in {time.time() - warmup_start:.1f}s")

    # Determine which configs to test
    if TEST_CONFIRMATION_CONFIGS:
        configs_to_test = CONFIRMATION_CONFIGS
    else:
        configs_to_test = {'long': CONFIRMATION_CONFIGS['long']}

    # Run backtests
    all_stats = []
    all_results = {}

    # Calculate total tests (accounting for skipped tests when no confirmation horizons)
    total_tests = 0
    for config_name, confirm_horizons in configs_to_test.items():
        if len(confirm_horizons) == 0:
            # Only 'primary_only' method runs when no confirmation horizons
            total_tests += len(N_SATELLITES_TO_TEST) * 1
        else:
            # All methods run
            total_tests += len(N_SATELLITES_TO_TEST) * len(CONSENSUS_METHODS)
    test_num = 0

    start_time = time.time()

    for config_name, confirm_horizons in configs_to_test.items():
        print(f"\n{'#'*60}")
        print(f"# CONFIRMATION CONFIG: {config_name.upper()}")
        print(f"# Horizons: {confirm_horizons if confirm_horizons else 'None (primary only)'}")
        print(f"{'#'*60}")

        for n in N_SATELLITES_TO_TEST:
            for method in CONSENSUS_METHODS:
                # Skip non-primary methods if no confirmation horizons
                if len(confirm_horizons) == 0 and method != 'primary_only':
                    continue

                test_num += 1
                print(f"\n{'='*60}")
                print(f"TEST {test_num}/{total_tests}: Config={config_name}, N={n}, Method={method}")
                print(f"{'='*60}")

                results_df = walk_forward_backtest_consensus(
                    horizon_data, n, method,
                    confirmation_horizons=confirm_horizons,
                    show_progress=True
                )

                if len(results_df) > 0:
                    all_results[(config_name, n, method)] = results_df

                    stats = analyze_results(results_df, n, method)
                    if stats:
                        stats['config'] = config_name
                        stats['n_confirm'] = len(confirm_horizons)
                        all_stats.append(stats)

                        print(f"\n  Results:")
                        print(f"    Periods: {stats['n_periods']}")
                        print(f"    Monthly Alpha: {stats['avg_alpha']*100:.2f}%")
                        print(f"    Annual Alpha: {stats['annual_alpha']*100:.1f}%")
                        print(f"    Hit Rate: {stats['hit_rate']:.2%}")
                        print(f"    Sharpe: {stats['sharpe']:.3f}")

                    # Save individual results
                    output_file = OUTPUT_DIR / f'multi_horizon_{config_name}_N{n}_{method}.csv'
                    results_df.to_csv(output_file, index=False)
                else:
                    print(f"  No results")

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if len(all_stats) > 0:
        summary_df = pd.DataFrame(all_stats)

        # Print comparison table
        print("\n" + "-" * 100)
        print(f"{'Config':<8} {'N':>3} {'Method':<15} {'Periods':>8} {'Monthly':>10} {'Annual':>10} {'Hit Rate':>10} {'Sharpe':>8}")
        print("-" * 100)

        for _, row in summary_df.sort_values(['config', 'n_satellites', 'sharpe'], ascending=[True, True, False]).iterrows():
            print(f"{row['config']:<8} {int(row['n_satellites']):>3} {row['method']:<15} {int(row['n_periods']):>8} "
                  f"{row['avg_alpha']*100:>9.2f}% {row['annual_alpha']*100:>9.1f}% "
                  f"{row['hit_rate']:>9.1%} {row['sharpe']:>8.3f}")

        print("-" * 100)

        # Best by config
        print("\nBest configuration per confirmation horizon config (by Sharpe):")
        for config_name in configs_to_test.keys():
            config_df = summary_df[summary_df['config'] == config_name]
            if len(config_df) > 0:
                best = config_df.loc[config_df['sharpe'].idxmax()]
                print(f"  {config_name}: N={int(best['n_satellites'])}, Method={best['method']}, "
                      f"Alpha={best['annual_alpha']*100:.1f}%, "
                      f"Hit={best['hit_rate']:.1%}, "
                      f"Sharpe={best['sharpe']:.3f}")

        # Overall best
        best_overall = summary_df.loc[summary_df['sharpe'].idxmax()]
        print(f"\nOverall best (by Sharpe):")
        print(f"  Config={best_overall['config']}, N={int(best_overall['n_satellites'])}, Method={best_overall['method']}")
        print(f"  Annual Alpha: {best_overall['annual_alpha']*100:.1f}%")
        print(f"  Hit Rate: {best_overall['hit_rate']:.2%}")
        print(f"  Sharpe: {best_overall['sharpe']:.3f}")

        # Save summary
        summary_file = OUTPUT_DIR / 'multi_horizon_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\n[SAVED] {summary_file}")


if __name__ == '__main__':
    main()
