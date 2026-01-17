"""
Test Script: Strategy Improvement Ideas
=======================================

This script tests several potential improvements to the walk-forward backtest:

1. IC-Weighted Ensemble: Weight features by their Information Coefficient
3. Regime-Adaptive Strategy: Use different features in different market regimes
4. Time-Series Signal Features: Add signal trend/acceleration features
6. Dynamic N Selection: Adapt N based on signal confidence (range 1-5)
7. Multi-Factor Interaction Features: Create new signals by combining factors
   (e.g., momentum/volatility = "momentum per unit of risk")
8. Feature Stability Scoring: Prefer features that are stable over time

Run this script to compare improvements against the baseline strategy.

Usage:
    python test_strategy_improvements.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================
# CONFIGURATION
# ============================================================

HOLDING_MONTHS = 1
N_SATELLITES = 5  # Our chosen N from analysis
N_RANGE = [1, 2, 3, 4, 5]  # Statistically equivalent range for dynamic N

MIN_TRAINING_MONTHS = 36
MIN_ALPHA = 0.001
MIN_HIT_RATE = 0.55
MAX_ENSEMBLE_SIZE = 20
MIN_IMPROVEMENT = 0.0001

DATA_DIR = Path(__file__).parent / 'data'
OUTPUT_DIR = DATA_DIR / 'improvement_tests'

# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load all precomputed data."""
    print(f"\n{'='*60}")
    print(f"LOADING DATA")
    print(f"{'='*60}")

    # Load forward alpha
    alpha_file = DATA_DIR / f'forward_alpha_{HOLDING_MONTHS}month.parquet'
    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])
    print(f"[OK] Forward alpha: {len(alpha_df):,} observations")

    # Load rankings matrix
    rankings_file = DATA_DIR / f'rankings_matrix_{HOLDING_MONTHS}month.npz'
    npz_data = np.load(rankings_file, allow_pickle=True)
    rankings = npz_data['rankings'].copy()
    dates = pd.to_datetime(npz_data['dates'])
    isins = npz_data['isins']
    feature_names = list(npz_data['features'])
    print(f"[OK] Rankings matrix: {rankings.shape}")

    # Load feature-alpha matrix
    feature_alpha_file = DATA_DIR / f'feature_alpha_{HOLDING_MONTHS}month.npz'
    fa_data = np.load(feature_alpha_file, allow_pickle=True)
    feature_alpha = fa_data['feature_alpha']
    feature_hit = fa_data['feature_hit']
    print(f"[OK] Feature-alpha matrix: {feature_alpha.shape}")

    # Create alpha matrix for fast lookups
    n_dates = len(dates)
    n_isins = len(isins)
    isin_to_idx = {isin: idx for idx, isin in enumerate(isins)}
    date_to_idx = {date: idx for idx, date in enumerate(dates)}

    alpha_matrix = np.full((n_dates, n_isins), np.nan, dtype=np.float32)
    for date, group in alpha_df.groupby('date'):
        if date not in date_to_idx:
            continue
        date_idx = date_to_idx[date]
        for isin, alpha in zip(group['isin'].values, group['forward_alpha'].values):
            if isin in isin_to_idx:
                alpha_matrix[date_idx, isin_to_idx[isin]] = alpha

    alpha_valid = ~np.isnan(alpha_matrix)
    print(f"[OK] Alpha matrix: {alpha_matrix.shape}")

    return {
        'alpha_matrix': alpha_matrix,
        'alpha_valid': alpha_valid,
        'rankings': rankings,
        'dates': dates,
        'isins': isins,
        'feature_names': feature_names,
        'feature_alpha': feature_alpha,
        'feature_hit': feature_hit,
        'isin_to_idx': isin_to_idx,
        'date_to_idx': date_to_idx,
    }


# ============================================================
# BASELINE STRATEGY (from 6_fast_backtest.py)
# ============================================================

def evaluate_ensemble(data, feature_indices, n_satellites, date_mask):
    """Evaluate an ensemble of features."""
    rankings = data['rankings']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    train_indices = np.where(date_mask)[0]
    if len(train_indices) == 0:
        return None

    alphas_list = []

    for date_idx in train_indices:
        feature_rankings = rankings[date_idx, :, :][:, feature_indices]
        scores = np.nanmean(feature_rankings, axis=1)
        valid_mask = ~np.isnan(scores) & alpha_valid[date_idx]

        if valid_mask.sum() < n_satellites:
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]

        top_k = np.argpartition(valid_scores, -n_satellites)[-n_satellites:]
        top_isin_indices = valid_indices[top_k]

        selected_alphas = [alpha_matrix[date_idx, idx] for idx in top_isin_indices
                          if alpha_valid[date_idx, idx]]

        if len(selected_alphas) > 0:
            alphas_list.append(np.mean(selected_alphas))

    if len(alphas_list) == 0:
        return None

    alphas_arr = np.array(alphas_list)
    return {
        'avg_alpha': np.mean(alphas_arr),
        'std_alpha': np.std(alphas_arr),
        'hit_rate': np.mean(alphas_arr > 0),
        'n_periods': len(alphas_arr),
        'alphas': alphas_arr
    }


def greedy_feature_selection(data, n_satellites, train_mask, return_ics=False):
    """Baseline greedy feature selection."""
    rankings = data['rankings'].copy()
    feature_alpha = data['feature_alpha']
    feature_hit = data['feature_hit']
    feature_names = data['feature_names']
    n_features = len(feature_names)

    # Pre-filter candidates
    candidates = []
    candidate_ics = {}  # Store IC for each candidate

    for feat_idx in range(n_features):
        alphas = feature_alpha[train_mask, feat_idx, n_satellites - 1]
        hits = feature_hit[train_mask, feat_idx, n_satellites - 1]
        valid = ~np.isnan(alphas)

        if valid.sum() == 0:
            continue

        avg_alpha = np.mean(alphas[valid])
        hit_rate = np.mean(hits[valid])

        # Calculate IC (correlation between feature rank and forward alpha)
        # Using avg_alpha as proxy for IC
        ic = avg_alpha

        if avg_alpha >= MIN_ALPHA and hit_rate >= MIN_HIT_RATE:
            candidates.append((feat_idx, 'positive', avg_alpha))
            candidate_ics[feat_idx] = ic
        elif avg_alpha <= -MIN_ALPHA and (1 - hit_rate) >= MIN_HIT_RATE:
            rankings[:, :, feat_idx] = 1.0 - rankings[:, :, feat_idx]
            candidates.append((feat_idx, 'inverted', -avg_alpha))
            candidate_ics[feat_idx] = -ic

    if len(candidates) == 0:
        return [], None, rankings, {}

    candidates.sort(key=lambda x: x[2], reverse=True)

    # Greedy selection
    selected = []
    best_perf = None
    data_copy = data.copy()
    data_copy['rankings'] = rankings

    for iteration in range(min(MAX_ENSEMBLE_SIZE, len(candidates))):
        current_indices = [f['idx'] for f in selected]
        remaining = [(idx, ptype, alpha) for idx, ptype, alpha in candidates
                    if idx not in current_indices]

        if len(remaining) == 0:
            break

        best_add = None
        best_add_perf = None
        best_improvement = 0

        for feat_idx, pred_type, _ in remaining:
            test_indices = current_indices + [feat_idx]
            perf = evaluate_ensemble(data_copy, test_indices, n_satellites, train_mask)

            if perf is None:
                continue

            improvement = perf['avg_alpha'] if best_perf is None else perf['avg_alpha'] - best_perf['avg_alpha']

            if improvement > best_improvement:
                best_improvement = improvement
                best_add = feat_idx
                best_add_perf = perf

        if best_add is None or best_improvement < MIN_IMPROVEMENT:
            break

        selected.append({
            'idx': best_add,
            'name': feature_names[best_add],
            'improvement': best_improvement,
            'ic': candidate_ics.get(best_add, 0)
        })
        best_perf = best_add_perf

    return selected, best_perf, rankings, candidate_ics


def run_baseline_backtest(data, n_satellites=N_SATELLITES):
    """Run baseline walk-forward backtest."""
    dates = data['dates']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0][0]

    results = []

    for test_idx in tqdm(range(test_start_idx, len(dates)), desc="Baseline"):
        test_date = dates[test_idx]
        train_mask = dates < test_date

        selected, train_perf, modified_rankings, _ = greedy_feature_selection(
            data, n_satellites, train_mask
        )

        if len(selected) == 0:
            continue

        feature_indices = [f['idx'] for f in selected]

        # Select ETFs
        feature_rankings = modified_rankings[test_idx][:, feature_indices]
        scores = np.nanmean(feature_rankings, axis=1)
        valid_mask = ~np.isnan(scores) & alpha_valid[test_idx]

        if valid_mask.sum() < n_satellites:
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]
        top_k = np.argpartition(valid_scores, -n_satellites)[-n_satellites:]
        selected_indices = valid_indices[top_k]

        alphas = [alpha_matrix[test_idx, idx] for idx in selected_indices]

        results.append({
            'date': test_date,
            'avg_alpha': np.mean(alphas),
            'n_features': len(selected),
            'n_satellites': n_satellites
        })

    return pd.DataFrame(results)


# ============================================================
# IMPROVEMENT 1: IC-WEIGHTED ENSEMBLE
# ============================================================

def run_ic_weighted_backtest(data, n_satellites=N_SATELLITES):
    """
    Improvement 1: Weight features by their Information Coefficient (IC)
    instead of equal weighting.

    score = sum(IC_i * rank_i) / sum(IC_i)
    """
    dates = data['dates']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0][0]

    results = []

    for test_idx in tqdm(range(test_start_idx, len(dates)), desc="IC-Weighted"):
        test_date = dates[test_idx]
        train_mask = dates < test_date

        selected, train_perf, modified_rankings, candidate_ics = greedy_feature_selection(
            data, n_satellites, train_mask, return_ics=True
        )

        if len(selected) == 0:
            continue

        feature_indices = [f['idx'] for f in selected]
        feature_ics = np.array([f['ic'] for f in selected])

        # Normalize ICs to weights (ensure positive)
        feature_ics = np.maximum(feature_ics, 0.0001)  # Floor at small positive
        ic_weights = feature_ics / feature_ics.sum()

        # IC-weighted scoring
        feature_rankings = modified_rankings[test_idx][:, feature_indices]

        # Weighted average instead of simple mean
        scores = np.zeros(feature_rankings.shape[0])
        for i, (feat_idx, weight) in enumerate(zip(feature_indices, ic_weights)):
            valid = ~np.isnan(feature_rankings[:, i])
            scores[valid] += weight * feature_rankings[valid, i]

        valid_mask = ~np.isnan(np.nanmean(feature_rankings, axis=1)) & alpha_valid[test_idx]

        if valid_mask.sum() < n_satellites:
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]
        top_k = np.argpartition(valid_scores, -n_satellites)[-n_satellites:]
        selected_indices = valid_indices[top_k]

        alphas = [alpha_matrix[test_idx, idx] for idx in selected_indices]

        results.append({
            'date': test_date,
            'avg_alpha': np.mean(alphas),
            'n_features': len(selected),
            'n_satellites': n_satellites
        })

    return pd.DataFrame(results)


# ============================================================
# IMPROVEMENT 3: REGIME-ADAPTIVE STRATEGY
# ============================================================

def compute_market_regime(data, date_idx, lookback=63):
    """
    Compute market regime based on:
    - Volatility regime (high/low)
    - Trend regime (bull/bear)

    Returns: 'high_vol_bull', 'high_vol_bear', 'low_vol_bull', 'low_vol_bear'
    """
    alpha_matrix = data['alpha_matrix']

    # Get lookback window
    start_idx = max(0, date_idx - lookback)

    # Use cross-sectional mean alpha as market proxy
    market_alphas = np.nanmean(alpha_matrix[start_idx:date_idx], axis=1)
    market_alphas = market_alphas[~np.isnan(market_alphas)]

    if len(market_alphas) < 20:
        return 'unknown'

    # Volatility regime
    vol = np.std(market_alphas)
    vol_threshold = 0.03  # 3% monthly volatility threshold
    high_vol = vol > vol_threshold

    # Trend regime (cumulative return over period)
    cum_return = np.sum(market_alphas)
    bull = cum_return > 0

    if high_vol and bull:
        return 'high_vol_bull'
    elif high_vol and not bull:
        return 'high_vol_bear'
    elif not high_vol and bull:
        return 'low_vol_bull'
    else:
        return 'low_vol_bear'


def get_regime_feature_preferences(regime):
    """
    Return feature name patterns to prefer/avoid based on regime.

    In high volatility: prefer low-vol, quality signals
    In bear markets: prefer defensive signals
    In bull markets: prefer momentum signals
    """
    preferences = {
        'high_vol_bull': {
            'prefer': ['sharpe', 'sortino', 'calmar', 'vol_'],
            'avoid': ['momentum', 'cum_return']
        },
        'high_vol_bear': {
            'prefer': ['sharpe', 'sortino', 'drawdown', 'vol_'],
            'avoid': ['momentum', 'cum_return', 'rs_']
        },
        'low_vol_bull': {
            'prefer': ['momentum', 'rs_', 'cum_return', 'cum_alpha'],
            'avoid': []
        },
        'low_vol_bear': {
            'prefer': ['sharpe', 'info_ratio', 'alpha'],
            'avoid': ['momentum']
        },
        'unknown': {
            'prefer': [],
            'avoid': []
        }
    }
    return preferences.get(regime, preferences['unknown'])


def run_regime_adaptive_backtest(data, n_satellites=N_SATELLITES):
    """
    Improvement 3: Adapt feature selection based on market regime.
    """
    dates = data['dates']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']
    feature_names = data['feature_names']

    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0][0]

    results = []
    regime_counts = {'high_vol_bull': 0, 'high_vol_bear': 0,
                     'low_vol_bull': 0, 'low_vol_bear': 0, 'unknown': 0}

    for test_idx in tqdm(range(test_start_idx, len(dates)), desc="Regime-Adaptive"):
        test_date = dates[test_idx]
        train_mask = dates < test_date

        # Detect regime
        regime = compute_market_regime(data, test_idx)
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        prefs = get_regime_feature_preferences(regime)

        # Get base selection
        selected, train_perf, modified_rankings, _ = greedy_feature_selection(
            data, n_satellites, train_mask
        )

        if len(selected) == 0:
            continue

        # Re-rank features based on regime preferences
        for feat in selected:
            feat_name = feat['name'].lower()
            bonus = 0

            for pattern in prefs['prefer']:
                if pattern in feat_name:
                    bonus += 0.001  # Small bonus
                    break

            for pattern in prefs['avoid']:
                if pattern in feat_name:
                    bonus -= 0.001  # Small penalty
                    break

            feat['regime_adjusted_ic'] = feat['ic'] + bonus

        # Sort by regime-adjusted IC and take top features
        selected.sort(key=lambda x: x.get('regime_adjusted_ic', x['ic']), reverse=True)
        selected = selected[:min(len(selected), MAX_ENSEMBLE_SIZE)]

        feature_indices = [f['idx'] for f in selected]

        # Select ETFs
        feature_rankings = modified_rankings[test_idx][:, feature_indices]
        scores = np.nanmean(feature_rankings, axis=1)
        valid_mask = ~np.isnan(scores) & alpha_valid[test_idx]

        if valid_mask.sum() < n_satellites:
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]
        top_k = np.argpartition(valid_scores, -n_satellites)[-n_satellites:]
        selected_indices = valid_indices[top_k]

        alphas = [alpha_matrix[test_idx, idx] for idx in selected_indices]

        results.append({
            'date': test_date,
            'avg_alpha': np.mean(alphas),
            'n_features': len(selected),
            'n_satellites': n_satellites,
            'regime': regime
        })

    print(f"\nRegime distribution: {regime_counts}")
    return pd.DataFrame(results)


# ============================================================
# IMPROVEMENT 4: TIME-SERIES SIGNAL FEATURES
# ============================================================

def compute_signal_momentum(rankings, feature_idx, date_idx, lookback=5):
    """
    Compute momentum/trend of the signal itself.

    Returns:
    - signal_trend: Average change in rank over lookback period
    - signal_acceleration: Change in trend
    """
    if date_idx < lookback + 1:
        return 0, 0

    # Get historical ranks for this feature
    hist_ranks = rankings[date_idx-lookback:date_idx, :, feature_idx]

    # Mean rank per date
    mean_ranks = np.nanmean(hist_ranks, axis=1)

    if len(mean_ranks) < 2:
        return 0, 0

    # Trend: average change
    changes = np.diff(mean_ranks)
    trend = np.mean(changes) if len(changes) > 0 else 0

    # Acceleration: change in trend
    if len(changes) >= 2:
        accel = changes[-1] - changes[0]
    else:
        accel = 0

    return trend, accel


def run_timeseries_features_backtest(data, n_satellites=N_SATELLITES):
    """
    Improvement 4: Add time-series features about signal behavior.

    Boost features that are trending up (momentum in the signal itself).
    """
    dates = data['dates']
    rankings = data['rankings']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0][0]

    results = []

    for test_idx in tqdm(range(test_start_idx, len(dates)), desc="TimeSeries Features"):
        test_date = dates[test_idx]
        train_mask = dates < test_date

        selected, train_perf, modified_rankings, _ = greedy_feature_selection(
            data, n_satellites, train_mask
        )

        if len(selected) == 0:
            continue

        # Compute signal momentum for each selected feature
        for feat in selected:
            trend, accel = compute_signal_momentum(
                modified_rankings, feat['idx'], test_idx, lookback=5
            )
            feat['signal_trend'] = trend
            feat['signal_accel'] = accel

            # Boost IC for features with positive trend (signal getting stronger)
            trend_bonus = 0.0005 * trend if trend > 0 else 0
            feat['trend_adjusted_ic'] = feat['ic'] + trend_bonus

        feature_indices = [f['idx'] for f in selected]

        # Weight by trend-adjusted IC
        ics = np.array([f.get('trend_adjusted_ic', f['ic']) for f in selected])
        ics = np.maximum(ics, 0.0001)
        weights = ics / ics.sum()

        # Weighted scoring
        feature_rankings = modified_rankings[test_idx][:, feature_indices]
        scores = np.zeros(feature_rankings.shape[0])

        for i, weight in enumerate(weights):
            valid = ~np.isnan(feature_rankings[:, i])
            scores[valid] += weight * feature_rankings[valid, i]

        valid_mask = ~np.isnan(np.nanmean(feature_rankings, axis=1)) & alpha_valid[test_idx]

        if valid_mask.sum() < n_satellites:
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]
        top_k = np.argpartition(valid_scores, -n_satellites)[-n_satellites:]
        selected_indices = valid_indices[top_k]

        alphas = [alpha_matrix[test_idx, idx] for idx in selected_indices]

        results.append({
            'date': test_date,
            'avg_alpha': np.mean(alphas),
            'n_features': len(selected),
            'n_satellites': n_satellites
        })

    return pd.DataFrame(results)


# ============================================================
# IMPROVEMENT 6: DYNAMIC N SELECTION
# ============================================================

def compute_signal_confidence(data, selected_features, modified_rankings, test_idx, alpha_valid):
    """
    Compute confidence in the signal based on:
    - Dispersion of scores (high dispersion = more confident)
    - Agreement between features
    """
    feature_indices = [f['idx'] for f in selected_features]
    feature_rankings = modified_rankings[test_idx][:, feature_indices]

    # Score dispersion
    scores = np.nanmean(feature_rankings, axis=1)
    valid_mask = ~np.isnan(scores) & alpha_valid[test_idx]
    valid_scores = scores[valid_mask]

    if len(valid_scores) < 5:
        return 0.5  # Neutral confidence

    # Dispersion: std of top scores vs overall std
    top_scores = np.sort(valid_scores)[-10:]
    score_std = np.std(valid_scores)
    top_std = np.std(top_scores)

    # High dispersion in top = high confidence
    dispersion_ratio = score_std / (top_std + 0.001)

    # Feature agreement: correlation between feature rankings
    if len(feature_indices) > 1:
        # Check if features agree on top picks
        top_5_per_feature = []
        for i in range(len(feature_indices)):
            feat_ranks = feature_rankings[:, i]
            valid = ~np.isnan(feat_ranks) & alpha_valid[test_idx]
            if valid.sum() >= 5:
                top_5 = set(np.argsort(feat_ranks[valid])[-5:])
                top_5_per_feature.append(top_5)

        if len(top_5_per_feature) >= 2:
            # Jaccard similarity between feature top picks
            intersection = top_5_per_feature[0]
            for s in top_5_per_feature[1:]:
                intersection = intersection.intersection(s)
            agreement = len(intersection) / 5
        else:
            agreement = 0.5
    else:
        agreement = 0.5

    # Combine into confidence score (0-1)
    confidence = 0.5 * min(dispersion_ratio, 2) / 2 + 0.5 * agreement
    return np.clip(confidence, 0, 1)


def run_dynamic_n_backtest(data):
    """
    Improvement 6: Dynamically select N based on signal confidence.

    N_RANGE = [1, 2, 3, 4, 5] (statistically equivalent range)

    High confidence → lower N (concentrate on best picks)
    Low confidence → higher N (diversify)
    """
    dates = data['dates']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0][0]

    results = []
    n_distribution = {n: 0 for n in N_RANGE}

    for test_idx in tqdm(range(test_start_idx, len(dates)), desc="Dynamic N"):
        test_date = dates[test_idx]
        train_mask = dates < test_date

        # Use middle N for feature selection
        base_n = 3
        selected, train_perf, modified_rankings, _ = greedy_feature_selection(
            data, base_n, train_mask
        )

        if len(selected) == 0:
            continue

        # Compute confidence
        confidence = compute_signal_confidence(
            data, selected, modified_rankings, test_idx, alpha_valid
        )

        # Map confidence to N
        # High confidence (>0.7) → N=1 or 2 (concentrate)
        # Medium confidence (0.4-0.7) → N=3
        # Low confidence (<0.4) → N=4 or 5 (diversify)
        if confidence > 0.7:
            dynamic_n = 1
        elif confidence > 0.55:
            dynamic_n = 2
        elif confidence > 0.4:
            dynamic_n = 3
        elif confidence > 0.25:
            dynamic_n = 4
        else:
            dynamic_n = 5

        n_distribution[dynamic_n] += 1

        feature_indices = [f['idx'] for f in selected]

        # Select ETFs with dynamic N
        feature_rankings = modified_rankings[test_idx][:, feature_indices]
        scores = np.nanmean(feature_rankings, axis=1)
        valid_mask = ~np.isnan(scores) & alpha_valid[test_idx]

        if valid_mask.sum() < dynamic_n:
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]
        top_k = np.argpartition(valid_scores, -dynamic_n)[-dynamic_n:]
        selected_indices = valid_indices[top_k]

        alphas = [alpha_matrix[test_idx, idx] for idx in selected_indices]

        results.append({
            'date': test_date,
            'avg_alpha': np.mean(alphas),
            'n_features': len(selected),
            'n_satellites': dynamic_n,
            'confidence': confidence
        })

    print(f"\nN distribution: {n_distribution}")
    return pd.DataFrame(results)


# ============================================================
# IMPROVEMENT 8: FEATURE STABILITY SCORING
# ============================================================

def compute_feature_stability(data, feature_idx, n_satellites, date_idx, lookback_months=12):
    """
    Compute how stable a feature's performance has been over recent months.

    Stable features have consistent positive alpha across time.
    """
    feature_alpha = data['feature_alpha']
    dates = data['dates']

    # Get lookback window
    start_idx = max(0, date_idx - lookback_months)

    # Get alphas for this feature over lookback period
    alphas = feature_alpha[start_idx:date_idx, feature_idx, n_satellites - 1]
    valid = ~np.isnan(alphas)

    if valid.sum() < 6:
        return 0.5  # Neutral stability

    alphas = alphas[valid]

    # Stability metrics:
    # 1. Hit rate over period
    hit_rate = np.mean(alphas > 0)

    # 2. Consistency: 1 - (std / mean) capped
    if np.mean(alphas) > 0:
        cv = np.std(alphas) / np.mean(alphas)
        consistency = max(0, 1 - cv)
    else:
        consistency = 0

    # 3. No recent failures: check last 3 months
    recent_hit = np.mean(alphas[-3:] > 0) if len(alphas) >= 3 else 0.5

    # Combine
    stability = 0.4 * hit_rate + 0.3 * consistency + 0.3 * recent_hit
    return np.clip(stability, 0, 1)


def run_stability_weighted_backtest(data, n_satellites=N_SATELLITES):
    """
    Improvement 8: Weight features by their stability over time.

    Stable features get higher weight in the ensemble.
    """
    dates = data['dates']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0][0]

    results = []

    for test_idx in tqdm(range(test_start_idx, len(dates)), desc="Stability-Weighted"):
        test_date = dates[test_idx]
        train_mask = dates < test_date

        selected, train_perf, modified_rankings, _ = greedy_feature_selection(
            data, n_satellites, train_mask
        )

        if len(selected) == 0:
            continue

        # Compute stability for each selected feature
        for feat in selected:
            stability = compute_feature_stability(
                data, feat['idx'], n_satellites, test_idx
            )
            feat['stability'] = stability

            # Combined weight: IC * stability
            feat['stability_adjusted_ic'] = feat['ic'] * (0.5 + 0.5 * stability)

        feature_indices = [f['idx'] for f in selected]

        # Weight by stability-adjusted IC
        ics = np.array([f.get('stability_adjusted_ic', f['ic']) for f in selected])
        ics = np.maximum(ics, 0.0001)
        weights = ics / ics.sum()

        # Weighted scoring
        feature_rankings = modified_rankings[test_idx][:, feature_indices]
        scores = np.zeros(feature_rankings.shape[0])

        for i, weight in enumerate(weights):
            valid = ~np.isnan(feature_rankings[:, i])
            scores[valid] += weight * feature_rankings[valid, i]

        valid_mask = ~np.isnan(np.nanmean(feature_rankings, axis=1)) & alpha_valid[test_idx]

        if valid_mask.sum() < n_satellites:
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]
        top_k = np.argpartition(valid_scores, -n_satellites)[-n_satellites:]
        selected_indices = valid_indices[top_k]

        alphas = [alpha_matrix[test_idx, idx] for idx in selected_indices]

        results.append({
            'date': test_date,
            'avg_alpha': np.mean(alphas),
            'n_features': len(selected),
            'n_satellites': n_satellites
        })

    return pd.DataFrame(results)


# ============================================================
# IMPROVEMENT 7: MULTI-FACTOR INTERACTION FEATURES
# ============================================================

# Define interaction pairs: (factor1_pattern, factor2_pattern, operation, name)
# These create NEW signals by combining two factors at the signal level
INTERACTION_PAIRS = [
    # Momentum per unit of risk (Sharpe-like at signal level)
    ('momentum_63d', 'vol_21d', 'divide', 'mom_per_vol'),
    ('momentum_126d', 'vol_63d', 'divide', 'mom126_per_vol'),

    # Momentum with quality confirmation
    ('momentum_63d', 'sharpe_63d', 'multiply', 'mom_x_sharpe'),
    ('cum_return_63d', 'sortino_63d', 'multiply', 'return_x_sortino'),

    # Alpha with low drawdown (quality alpha)
    ('cum_alpha_core_63d', 'drawdown', 'multiply_inv', 'alpha_low_dd'),

    # Relative strength with trend confirmation
    ('rs_63d', 'high_52w_proximity', 'multiply', 'rs_near_high'),

    # Mean reversion candidates (low momentum but high quality)
    ('sharpe_126d', 'momentum_63d', 'divide_inv', 'quality_over_mom'),
]


def find_feature_by_pattern(feature_names, pattern):
    """Find feature indices matching a pattern."""
    matches = []
    for idx, name in enumerate(feature_names):
        # Match base signal name (before filter suffix)
        base_name = name.split('__')[0] if '__' in name else name
        if pattern in base_name:
            matches.append(idx)
    return matches


def create_interaction_signal(rankings, feat1_idx, feat2_idx, operation, date_idx):
    """
    Create interaction signal by combining two features.

    Operations:
    - 'multiply': f1 * f2 (both high = good)
    - 'divide': f1 / f2 (high f1, low f2 = good)
    - 'multiply_inv': f1 * (1 - f2) (high f1, low f2 = good)
    - 'divide_inv': f1 / (1 - f2 + 0.01) (high f1, low f2 = good)
    """
    r1 = rankings[date_idx, :, feat1_idx]
    r2 = rankings[date_idx, :, feat2_idx]

    # Handle NaN
    valid = ~np.isnan(r1) & ~np.isnan(r2)

    result = np.full(len(r1), np.nan)

    if operation == 'multiply':
        result[valid] = r1[valid] * r2[valid]
    elif operation == 'divide':
        # Avoid division by zero, invert r2 so low vol = high score
        r2_safe = np.maximum(r2[valid], 0.01)
        result[valid] = r1[valid] / r2_safe
    elif operation == 'multiply_inv':
        # High f1 AND low f2 = good
        result[valid] = r1[valid] * (1 - r2[valid])
    elif operation == 'divide_inv':
        # High f1 relative to low f2
        r2_inv = 1 - r2[valid] + 0.01
        result[valid] = r1[valid] / r2_inv

    # Re-rank to 0-1 scale
    valid_result = ~np.isnan(result)
    if valid_result.sum() > 1:
        ranks = np.argsort(np.argsort(result[valid_result]))
        result[valid_result] = ranks / (len(ranks) - 1)

    return result


def run_interaction_features_backtest(data, n_satellites=N_SATELLITES):
    """
    Improvement 7: Create new signals by combining factors at signal level.

    Example: momentum_63d / vol_21d = "momentum per unit of risk"
    This is different from averaging ranks - it creates fundamentally new signals.
    """
    dates = data['dates']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']
    feature_names = data['feature_names']

    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0][0]

    # Pre-compute feature indices for interaction pairs
    interaction_configs = []
    for pattern1, pattern2, operation, name in INTERACTION_PAIRS:
        feat1_matches = find_feature_by_pattern(feature_names, pattern1)
        feat2_matches = find_feature_by_pattern(feature_names, pattern2)

        if feat1_matches and feat2_matches:
            # Use first match with raw filter for simplicity
            # Prefer 'raw' or 'ema_21d' filtered versions
            feat1_idx = feat1_matches[0]
            feat2_idx = feat2_matches[0]

            for f1 in feat1_matches:
                if '__raw' in feature_names[f1] or '__ema_21d' in feature_names[f1]:
                    feat1_idx = f1
                    break
            for f2 in feat2_matches:
                if '__raw' in feature_names[f2] or '__ema_21d' in feature_names[f2]:
                    feat2_idx = f2
                    break

            interaction_configs.append({
                'feat1_idx': feat1_idx,
                'feat2_idx': feat2_idx,
                'operation': operation,
                'name': name,
                'feat1_name': feature_names[feat1_idx],
                'feat2_name': feature_names[feat2_idx]
            })

    print(f"\nCreated {len(interaction_configs)} interaction features:")
    for cfg in interaction_configs:
        print(f"  - {cfg['name']}: {cfg['feat1_name']} {cfg['operation']} {cfg['feat2_name']}")

    results = []

    for test_idx in tqdm(range(test_start_idx, len(dates)), desc="Interaction Features"):
        test_date = dates[test_idx]
        train_mask = dates < test_date

        # Get baseline selected features
        selected, train_perf, modified_rankings, _ = greedy_feature_selection(
            data, n_satellites, train_mask
        )

        if len(selected) == 0:
            continue

        # Create interaction signals for this date
        interaction_signals = []
        for cfg in interaction_configs:
            signal = create_interaction_signal(
                modified_rankings,
                cfg['feat1_idx'],
                cfg['feat2_idx'],
                cfg['operation'],
                test_idx
            )
            interaction_signals.append(signal)

        # Combine: average of base features + interaction features
        feature_indices = [f['idx'] for f in selected]
        base_rankings = modified_rankings[test_idx][:, feature_indices]
        base_scores = np.nanmean(base_rankings, axis=1)

        # Average interaction signals
        if interaction_signals:
            interaction_matrix = np.column_stack(interaction_signals)
            interaction_scores = np.nanmean(interaction_matrix, axis=1)

            # Combine: 60% base, 40% interaction
            combined_scores = np.where(
                ~np.isnan(base_scores) & ~np.isnan(interaction_scores),
                0.6 * base_scores + 0.4 * interaction_scores,
                base_scores  # Fallback to base if interaction not available
            )
        else:
            combined_scores = base_scores

        valid_mask = ~np.isnan(combined_scores) & alpha_valid[test_idx]

        if valid_mask.sum() < n_satellites:
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_scores = combined_scores[valid_mask]
        top_k = np.argpartition(valid_scores, -n_satellites)[-n_satellites:]
        selected_indices = valid_indices[top_k]

        alphas = [alpha_matrix[test_idx, idx] for idx in selected_indices]

        results.append({
            'date': test_date,
            'avg_alpha': np.mean(alphas),
            'n_features': len(selected),
            'n_interactions': len(interaction_configs),
            'n_satellites': n_satellites
        })

    return pd.DataFrame(results)


# ============================================================
# ANALYSIS & COMPARISON
# ============================================================

def analyze_results(results_df, name):
    """Compute performance metrics."""
    if len(results_df) == 0:
        return None

    avg_alpha = results_df['avg_alpha'].mean()
    std_alpha = results_df['avg_alpha'].std()
    hit_rate = (results_df['avg_alpha'] > 0).mean()
    sharpe = avg_alpha / std_alpha if std_alpha > 0 else 0

    # Drawdown
    cumulative = (1 + results_df['avg_alpha']).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    total_return = cumulative.iloc[-1] - 1

    return {
        'name': name,
        'n_periods': len(results_df),
        'avg_alpha': avg_alpha,
        'std_alpha': std_alpha,
        'annual_alpha': avg_alpha * 12,
        'hit_rate': hit_rate,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'total_return': total_return
    }


def print_comparison(all_results):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)

    print(f"\n{'Strategy':<25} {'Ann.Alpha':>10} {'Hit Rate':>10} {'Sharpe':>10} {'MaxDD':>10} {'Total':>10}")
    print("-" * 80)

    baseline = None
    for r in all_results:
        if r['name'] == 'Baseline':
            baseline = r

        print(f"{r['name']:<25} {r['annual_alpha']*100:>9.2f}% {r['hit_rate']:>9.1%} "
              f"{r['sharpe']:>10.3f} {r['max_drawdown']*100:>9.1f}% {r['total_return']*100:>9.1f}%")

    if baseline:
        print("\n" + "-" * 80)
        print("IMPROVEMENT VS BASELINE:")
        print("-" * 80)

        for r in all_results:
            if r['name'] == 'Baseline':
                continue

            alpha_diff = (r['annual_alpha'] - baseline['annual_alpha']) * 100
            hit_diff = (r['hit_rate'] - baseline['hit_rate']) * 100
            sharpe_diff = r['sharpe'] - baseline['sharpe']
            dd_diff = (r['max_drawdown'] - baseline['max_drawdown']) * 100

            print(f"{r['name']:<25} {alpha_diff:>+9.2f}% {hit_diff:>+9.1f}pp "
                  f"{sharpe_diff:>+10.3f} {dd_diff:>+9.1f}% ")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("STRATEGY IMPROVEMENT TESTS")
    print("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_data()

    all_results = []
    all_dfs = {}

    # 1. Baseline
    print("\n" + "=" * 60)
    print("Running BASELINE strategy...")
    print("=" * 60)
    df_baseline = run_baseline_backtest(data)
    all_dfs['Baseline'] = df_baseline
    all_results.append(analyze_results(df_baseline, 'Baseline'))

    # 2. IC-Weighted
    print("\n" + "=" * 60)
    print("Running IC-WEIGHTED strategy...")
    print("=" * 60)
    df_ic = run_ic_weighted_backtest(data)
    all_dfs['IC-Weighted'] = df_ic
    all_results.append(analyze_results(df_ic, 'IC-Weighted'))

    # 3. Regime-Adaptive
    print("\n" + "=" * 60)
    print("Running REGIME-ADAPTIVE strategy...")
    print("=" * 60)
    df_regime = run_regime_adaptive_backtest(data)
    all_dfs['Regime-Adaptive'] = df_regime
    all_results.append(analyze_results(df_regime, 'Regime-Adaptive'))

    # 4. Time-Series Features
    print("\n" + "=" * 60)
    print("Running TIME-SERIES FEATURES strategy...")
    print("=" * 60)
    df_ts = run_timeseries_features_backtest(data)
    all_dfs['TimeSeries-Features'] = df_ts
    all_results.append(analyze_results(df_ts, 'TimeSeries-Features'))

    # 5. Dynamic N
    print("\n" + "=" * 60)
    print("Running DYNAMIC N strategy...")
    print("=" * 60)
    df_dynamic = run_dynamic_n_backtest(data)
    all_dfs['Dynamic-N'] = df_dynamic
    all_results.append(analyze_results(df_dynamic, 'Dynamic-N'))

    # 6. Stability-Weighted
    print("\n" + "=" * 60)
    print("Running STABILITY-WEIGHTED strategy...")
    print("=" * 60)
    df_stability = run_stability_weighted_backtest(data)
    all_dfs['Stability-Weighted'] = df_stability
    all_results.append(analyze_results(df_stability, 'Stability-Weighted'))

    # 7. Interaction Features
    print("\n" + "=" * 60)
    print("Running INTERACTION FEATURES strategy...")
    print("=" * 60)
    df_interaction = run_interaction_features_backtest(data)
    all_dfs['Interaction-Features'] = df_interaction
    all_results.append(analyze_results(df_interaction, 'Interaction-Features'))

    # Print comparison
    print_comparison(all_results)

    # Save results
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(OUTPUT_DIR / 'improvement_comparison.csv', index=False)

    for name, df in all_dfs.items():
        df.to_csv(OUTPUT_DIR / f'{name.lower().replace("-", "_")}_results.csv', index=False)

    print(f"\n[SAVED] Results to {OUTPUT_DIR}")

    # Return for notebook use
    return all_results, all_dfs


if __name__ == '__main__':
    main()
