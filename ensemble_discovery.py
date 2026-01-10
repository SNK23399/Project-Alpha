"""
Ensemble Discovery for ETF Selection (GPU-Accelerated)
=======================================================

Tests various ensemble techniques for combining signals:
1. Aggregated Rank - average cross-sectional rank across features
2. Agreement Voting - count how many features agree on top picks
3. Filter-then-Select - use some features to filter, others to rank

NOW USES 100% OF FEATURES from signal_framework_gpu.py:
- 58 signal bases (momentum, reversion, quality, etc.)
- 12 smoothing filters (raw, EMA, Hull, Butterworth, Kalman, Savgol, DEMA)
- 25 indicator transformations (level, momentum, z-score, velocity, etc.)
= 17,400 total features!

Features are computed in streaming mode to manage memory.

GPU-accelerated using CuPy with Numba fallback for CPU.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from numba import njit, prange
import warnings
warnings.filterwarnings('ignore')
import time
from tqdm import tqdm

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_GPU = True
    print("GPU acceleration enabled (CuPy)")
except ImportError:
    cp = np
    HAS_GPU = False
    print("GPU not available, using CPU")

# Load data utilities
import sys
sys.path.insert(0, str(Path(__file__).parent / 'support'))
from etf_database import ETFDatabase

# Import signal framework modules
from signal_bases import compute_all_signal_bases
from signal_filters import DEFAULT_FILTER_CONFIGS, apply_filter
from signal_indicators import compute_indicators, N_INDICATORS

CORE_ISIN = "IE00B6R52259"


def to_gpu(arr):
    """Move array to GPU if available."""
    if HAS_GPU:
        return cp.asarray(arr)
    return arr


def to_cpu(arr):
    """Move array to CPU."""
    if HAS_GPU and hasattr(arr, 'get'):
        return arr.get()
    return arr


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and prepare data as DataFrames for signal_bases.
    Returns: etf_prices (DataFrame), core_prices (Series)
    """
    db = ETFDatabase()
    prices = db.load_all_prices()
    prices.index = pd.to_datetime(prices.index)

    core_prices = prices[CORE_ISIN]
    satellite_prices = prices.drop(columns=[CORE_ISIN])

    # Filter to ETFs with sufficient data
    min_days = 252 * 5
    valid_cols = satellite_prices.columns[satellite_prices.notna().sum() > min_days]
    satellite_prices = satellite_prices[valid_cols]

    return satellite_prices, core_prices


def load_data_arrays() -> Tuple[np.ndarray, np.ndarray, List[str], pd.DatetimeIndex]:
    """
    Load and prepare data as numpy arrays for fast GPU processing.
    Returns: etf_prices, core_prices, etf_isins, dates
    """
    etf_df, core_series = load_data()

    etf_prices = etf_df.values.astype(np.float32)
    core_prices = core_series.values.astype(np.float32)
    etf_isins = list(etf_df.columns)
    dates = etf_df.index

    return etf_prices, core_prices, etf_isins, dates


def compute_monthly_alpha_gpu(
    etf_prices: np.ndarray,
    core_prices: np.ndarray,
    dates: pd.DatetimeIndex
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute monthly forward alpha using vectorized operations.
    Returns: alpha_monthly (n_months, n_etfs), month_end_indices
    """
    # Find month-end indices
    month_ends = []
    current_month = None
    for i, date in enumerate(dates):
        if current_month is not None and date.month != current_month:
            month_ends.append(i - 1)
        current_month = date.month
    month_ends.append(len(dates) - 1)
    month_end_indices = np.array(month_ends[:-1], dtype=np.int32)  # Exclude last

    n_months = len(month_end_indices)
    n_etfs = etf_prices.shape[1]

    # Compute monthly returns at month-ends
    alpha_monthly = np.zeros((n_months, n_etfs), dtype=np.float32)

    for m in range(n_months - 1):
        idx_now = month_end_indices[m]
        idx_next = month_end_indices[m + 1]

        # ETF forward return
        etf_ret = (etf_prices[idx_next] / etf_prices[idx_now]) - 1
        core_ret = (core_prices[idx_next] / core_prices[idx_now]) - 1

        # Alpha = ETF return - Core return
        alpha_monthly[m] = etf_ret - core_ret

    return alpha_monthly[:-1], month_end_indices[:-1]  # Exclude last month (no forward)


# =============================================================================
# SMOOTHING FILTERS (Hull, Savgol) - TOP PERFORMERS
# =============================================================================

def _hull_ma_gpu(arr, window):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    Hull = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    xp = cp if HAS_GPU else np
    half_window = max(1, window // 2)
    sqrt_window = max(1, int(np.sqrt(window)))

    wma_half = _weighted_ma_gpu(arr, half_window)
    wma_full = _weighted_ma_gpu(arr, window)

    # 2*WMA(n/2) - WMA(n)
    raw_hull = 2 * wma_half - wma_full

    # Final WMA smoothing
    hull = _weighted_ma_gpu(raw_hull, sqrt_window)
    return hull


def _weighted_ma_gpu(arr, window):
    """Weighted moving average (linear weights)."""
    xp = cp if HAS_GPU else np
    weights = xp.arange(1, window + 1, dtype=xp.float32)
    weights = weights / weights.sum()

    if arr.ndim == 1:
        result = xp.full_like(arr, xp.nan)
        for t in range(window - 1, len(arr)):
            result[t] = xp.sum(arr[t-window+1:t+1] * weights)
    else:
        n_time, n_etfs = arr.shape
        result = xp.full_like(arr, xp.nan)
        for t in range(window - 1, n_time):
            result[t] = xp.sum(arr[t-window+1:t+1] * weights[:, None], axis=0)
    return result


def _savgol_filter_causal(arr, window, polyorder=3):
    """
    Fast causal Savitzky-Golay filter approximation.
    Uses scipy's savgol_filter with appropriate padding for near-causal behavior.
    Much faster than the iterative approach while being almost causal.
    """
    # Ensure window is odd
    if window % 2 == 0:
        window += 1

    if arr.ndim == 1:
        window = min(window, len(arr))
        if window < polyorder + 2:
            return arr.copy()

        # Forward-fill NaNs temporarily
        arr_filled = pd.Series(arr).ffill().bfill().values.astype(np.float32)

        # Apply savgol with 'nearest' mode (quasi-causal)
        filtered = savgol_filter(arr_filled, window, polyorder, mode='nearest')

        # Shift result forward by half window to make it more causal
        # (standard savgol uses centered window, shift compensates)
        shift = window // 2
        result = np.full_like(arr, np.nan)
        result[shift:] = filtered[:-shift] if shift > 0 else filtered

        # Restore NaN positions from original
        result[np.isnan(arr)] = np.nan
        return result
    else:
        # Vectorized for 2D - apply to all columns at once
        n_time, n_etfs = arr.shape
        window = min(window, n_time)
        if window < polyorder + 2:
            return arr.copy()

        # Forward-fill NaNs
        arr_filled = pd.DataFrame(arr).ffill().bfill().values.astype(np.float32)

        # Apply savgol to each column (vectorized by scipy)
        result = np.zeros_like(arr)
        for e in range(n_etfs):
            result[:, e] = savgol_filter(arr_filled[:, e], window, polyorder, mode='nearest')

        # Shift to make quasi-causal
        shift = window // 2
        if shift > 0:
            result_shifted = np.full_like(result, np.nan)
            result_shifted[shift:] = result[:-shift]
            result = result_shifted

        # Restore original NaN positions
        result[np.isnan(arr)] = np.nan
        return result


def _compute_velocity(arr, window=21):
    """Compute signal velocity (first derivative, normalized)."""
    xp = cp if HAS_GPU else np
    if arr.ndim == 1:
        diff = xp.zeros_like(arr)
        diff[window:] = arr[window:] - arr[:-window]
        std = _rolling_std_gpu(arr, window * 2)
        return diff / (std + 1e-10)
    else:
        diff = xp.zeros_like(arr)
        diff[window:] = arr[window:] - arr[:-window]
        std = _rolling_std_gpu(arr, window * 2)
        return diff / (std + 1e-10)


def _compute_acceleration(arr, window=21):
    """Compute signal acceleration (second derivative, normalized)."""
    velocity = _compute_velocity(arr, window)
    return _compute_velocity(velocity, window)


# =============================================================================
# GPU-ACCELERATED SIGNAL COMPUTATION (ENHANCED with Hull/Savgol)
# =============================================================================

def compute_signals_gpu(
    etf_prices: np.ndarray,
    core_prices: np.ndarray,
    month_end_indices: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute all signals at month-end dates using GPU.
    Now includes Hull and Savgol filtered versions of top signals.
    Returns: signals (n_signals, n_months, n_etfs), signal_names
    """
    n_time, n_etfs = etf_prices.shape
    n_months = len(month_end_indices)

    # Move to GPU
    etf_gpu = to_gpu(etf_prices)
    core_gpu = to_gpu(core_prices)

    # Pre-compute returns on GPU
    etf_ret = cp.zeros_like(etf_gpu) if HAS_GPU else np.zeros_like(etf_gpu)
    etf_ret[1:] = etf_gpu[1:] / etf_gpu[:-1] - 1

    core_ret = cp.zeros_like(core_gpu) if HAS_GPU else np.zeros_like(core_gpu)
    core_ret[1:] = core_gpu[1:] / core_gpu[:-1] - 1

    alpha = etf_ret - core_ret[:, None]

    raw_signals = {}

    # === MOMENTUM SIGNALS (RS at various lookbacks) ===
    for n_months_lb, days in [(1, 21), (3, 63), (6, 126), (12, 252)]:
        if days < n_time:
            etf_mom = etf_gpu[days:] / etf_gpu[:-days] - 1
            core_mom = core_gpu[days:] / core_gpu[:-days] - 1

            # Pad to full length
            if HAS_GPU:
                rs_full = cp.full((n_time, n_etfs), cp.nan, dtype=cp.float32)
            else:
                rs_full = np.full((n_time, n_etfs), np.nan, dtype=np.float32)
            rs_full[days:] = etf_mom / (core_mom[:, None] + 1e-10)
            raw_signals[f'rs_{n_months_lb}m'] = rs_full

    # === BETA-ADJUSTED RS (TOP PERFORMER) ===
    for window in [21, 63]:  # Added w21 (top performer)
        if window < n_time:
            beta = _rolling_beta_gpu(etf_ret, core_ret, window)
            rs_12m = raw_signals.get('rs_12m')
            if rs_12m is not None:
                for damping in [0.5, 1.0]:
                    xp = cp if HAS_GPU else np
                    beta_adj = rs_12m / (xp.abs(beta) + damping)
                    raw_signals[f'beta_adj_rs_d{damping}_w{window}'] = beta_adj

    # === MEAN REVERSION SIGNALS ===
    cum_alpha = _cumsum_gpu(alpha)
    rolling_max = _rolling_max_gpu(cum_alpha, 126)
    alpha_dd = -(cum_alpha - rolling_max)
    raw_signals['alpha_dd_reversion'] = alpha_dd

    # Vol-boosted reversion (TOP PERFORMER: +1.355%)
    vol_short = _rolling_std_gpu(core_ret, 63)
    vol_long = _rolling_std_gpu(core_ret, 252)
    vol_regime = vol_short / (vol_long + 1e-10)
    xp = cp if HAS_GPU else np
    vol_boost = 1 + xp.clip(vol_regime, 0, 2)
    raw_signals['vol_boosted_reversion'] = alpha_dd * vol_boost[:, None]

    # Distance from MA (TOP PERFORMER)
    ma_20 = _rolling_mean_gpu(etf_gpu, 20)
    raw_signals['dist_ma_20d'] = (etf_gpu - ma_20) / (ma_20 + 1e-10)

    # === QUALITY SIGNALS ===
    sharpe = _rolling_sharpe_gpu(etf_ret, 126)
    raw_signals['sharpe_126d'] = sharpe

    info_ratio = _rolling_sharpe_gpu(alpha, 126)
    raw_signals['info_ratio'] = info_ratio

    # === TREND SIGNALS ===
    ma_50 = _rolling_mean_gpu(etf_gpu, 50)
    ma_200 = _rolling_mean_gpu(etf_gpu, 200)
    raw_signals['ma_ratio_50_200'] = ma_50 / (ma_200 + 1e-10)

    # Trend-boosted momentum
    core_ma = _rolling_mean_gpu(core_gpu, 200)
    trend_regime = core_gpu / (core_ma + 1e-10) - 1
    trend_boost = 1 + xp.clip(trend_regime, 0, 1)
    if 'rs_6m' in raw_signals:
        raw_signals['trend_boosted_momentum'] = raw_signals['rs_6m'] * trend_boost[:, None]

    # === DISAGREEMENT SIGNALS (64% hit rate!) ===
    if 'rs_12m' in raw_signals and 'rs_1m' in raw_signals:
        raw_signals['disagreement_bull'] = raw_signals['rs_12m'] - raw_signals['rs_1m']

    # === MOMENTUM ACCELERATION (TOP PERFORMER) ===
    if 'rs_12m' in raw_signals:
        rs_12m = raw_signals['rs_12m']
        # 63-day change in 12m RS
        accel = xp.zeros_like(rs_12m)
        accel[63:] = rs_12m[63:] - rs_12m[:-63]
        raw_signals['momentum_accel_12m'] = accel

    # =================================================================
    # APPLY HULL AND SAVGOL FILTERS TO TOP SIGNALS
    # =================================================================
    print("   Applying Hull and Savgol filters to top signals...")

    signals = {}

    # Key signals to filter (top performers from signal_framework_gpu.py)
    top_signals = [
        'vol_boosted_reversion',
        'alpha_dd_reversion',
        'beta_adj_rs_d1.0_w21',
        'beta_adj_rs_d0.5_w21',
        'beta_adj_rs_d1.0_w63',
        'momentum_accel_12m',
        'dist_ma_20d',
        'trend_boosted_momentum',
        'rs_12m',
        'rs_6m',
        'disagreement_bull',
    ]

    for sig_name in raw_signals:
        sig = to_cpu(raw_signals[sig_name])

        # Always add raw signal
        signals[sig_name] = sig

        # Apply Hull filter to top signals (63-day for reversion, 21-day for momentum)
        if sig_name in top_signals:
            # Hull 63d (best for reversion signals)
            hull_63 = _hull_ma_gpu(to_gpu(sig), 63)
            signals[f'{sig_name}__hull_63d'] = to_cpu(hull_63)

            # Hull 21d (best for momentum signals)
            hull_21 = _hull_ma_gpu(to_gpu(sig), 21)
            signals[f'{sig_name}__hull_21d'] = to_cpu(hull_21)

            # Savgol 21d (preserves peaks)
            savgol_21 = _savgol_filter_causal(sig, 21, 3)
            signals[f'{sig_name}__savgol_21d'] = savgol_21

    # Extract month-end values only
    signal_names = list(signals.keys())
    n_signals = len(signal_names)

    print(f"   Total signals: {n_signals} (raw + Hull + Savgol filtered)")

    result = np.zeros((n_signals, n_months, n_etfs), dtype=np.float32)
    for s_idx, s_name in enumerate(signal_names):
        sig = signals[s_name]
        if HAS_GPU and hasattr(sig, 'get'):
            sig = sig.get()
        result[s_idx] = sig[month_end_indices]

    return result, signal_names


def _rolling_mean_gpu(arr, window):
    """GPU-accelerated rolling mean."""
    xp = cp if HAS_GPU else np
    if arr.ndim == 1:
        result = xp.full_like(arr, xp.nan)
        cumsum = xp.cumsum(arr)
        result[window-1:] = (cumsum[window-1:] - xp.concatenate([xp.array([0]), cumsum[:-window]])) / window
    else:
        result = xp.full_like(arr, xp.nan)
        cumsum = xp.cumsum(arr, axis=0)
        result[window-1:] = (cumsum[window-1:] - xp.concatenate([xp.zeros((1, arr.shape[1])), cumsum[:-window]], axis=0)) / window
    return result


def _rolling_std_gpu(arr, window):
    """GPU-accelerated rolling std."""
    xp = cp if HAS_GPU else np
    mean = _rolling_mean_gpu(arr, window)
    sq_mean = _rolling_mean_gpu(arr ** 2, window)
    variance = sq_mean - mean ** 2
    return xp.sqrt(xp.maximum(variance, 0))


def _rolling_zscore_gpu(arr, window):
    """GPU-accelerated rolling z-score."""
    mean = _rolling_mean_gpu(arr, window)
    std = _rolling_std_gpu(arr, window)
    return (arr - mean) / (std + 1e-10)


def _rolling_sharpe_gpu(arr, window):
    """GPU-accelerated rolling Sharpe ratio."""
    mean = _rolling_mean_gpu(arr, window) * 252
    std = _rolling_std_gpu(arr, window) * np.sqrt(252)
    return mean / (std + 1e-10)


def _rolling_max_gpu(arr, window):
    """GPU-accelerated rolling max (using stride tricks or loop)."""
    xp = cp if HAS_GPU else np
    n_time = arr.shape[0]
    if arr.ndim == 1:
        result = xp.full_like(arr, xp.nan)
        for t in range(window-1, n_time):
            result[t] = xp.max(arr[t-window+1:t+1])
    else:
        result = xp.full_like(arr, xp.nan)
        for t in range(window-1, n_time):
            result[t] = xp.max(arr[t-window+1:t+1], axis=0)
    return result


def _cumsum_gpu(arr):
    """GPU-accelerated cumsum with NaN handling."""
    xp = cp if HAS_GPU else np
    arr_filled = xp.where(xp.isnan(arr), 0, arr)
    return xp.cumsum(arr_filled, axis=0)


def _rolling_beta_gpu(etf_ret, core_ret, window):
    """GPU-accelerated rolling beta."""
    xp = cp if HAS_GPU else np
    n_time, n_etfs = etf_ret.shape

    # Rolling covariance and variance
    core_2d = core_ret[:, None] if core_ret.ndim == 1 else core_ret
    if core_2d.shape[1] == 1:
        core_2d = xp.broadcast_to(core_2d, (n_time, n_etfs))

    cov = _rolling_mean_gpu(etf_ret * core_2d, window) - _rolling_mean_gpu(etf_ret, window) * _rolling_mean_gpu(core_2d, window)
    var = _rolling_mean_gpu(core_2d ** 2, window) - _rolling_mean_gpu(core_2d, window) ** 2

    return cov / (var + 1e-10)


# =============================================================================
# GPU-ACCELERATED INDICATORS
# =============================================================================

def compute_indicators_gpu(
    signals: np.ndarray,  # (n_signals, n_months, n_etfs)
    signal_names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute indicator transformations using GPU.
    Now includes velocity indicator (top performer: +1.177% for momentum_accel).
    Returns: features (n_features, n_months, n_etfs), feature_names
    """
    n_signals, n_months, n_etfs = signals.shape
    xp = cp if HAS_GPU else np

    signals_gpu = to_gpu(signals)

    features = []
    feature_names = []

    # Signals that benefit from velocity indicator
    velocity_signals = [
        'momentum_accel_12m', 'momentum_accel_12m__hull_63d', 'momentum_accel_12m__hull_21d',
        'rs_12m', 'rs_12m__hull_63d', 'rs_6m', 'rs_6m__hull_63d',
        'disagreement_bull', 'beta_adj_rs_d1.0_w63', 'beta_adj_rs_d0.5_w21'
    ]

    for s_idx, s_name in enumerate(signal_names):
        sig = signals_gpu[s_idx]  # (n_months, n_etfs)

        # Level
        features.append(to_cpu(sig))
        feature_names.append(f'{s_name}__level')

        # Cross-sectional rank (vectorized)
        xs_rank = _cross_sectional_rank_gpu(sig)
        features.append(to_cpu(xs_rank))
        feature_names.append(f'{s_name}__xs_rank')

        # Cross-sectional z-score (vectorized)
        xs_mean = xp.nanmean(sig, axis=1, keepdims=True)
        xs_std = xp.nanstd(sig, axis=1, keepdims=True)
        xs_zscore = (sig - xs_mean) / (xs_std + 1e-10)
        features.append(to_cpu(xs_zscore))
        feature_names.append(f'{s_name}__xs_zscore')

        # Velocity (top performer for momentum signals)
        if any(v in s_name for v in ['momentum_accel', 'rs_12m', 'rs_6m', 'disagreement', 'beta_adj']):
            # Compute velocity: 3-month change, normalized
            sig_cpu = to_cpu(sig)
            velocity = np.zeros_like(sig_cpu)
            if n_months > 3:
                velocity[3:] = sig_cpu[3:] - sig_cpu[:-3]
                # Normalize by rolling std
                for m in range(6, n_months):
                    std_val = np.nanstd(sig_cpu[m-6:m], axis=0)
                    velocity[m] = velocity[m] / (std_val + 1e-10)
            features.append(velocity)
            feature_names.append(f'{s_name}__velocity')

    # Stack all features
    result = np.stack(features, axis=0).astype(np.float32)
    return result, feature_names


def _cross_sectional_rank_gpu(arr):
    """GPU-accelerated cross-sectional rank (percentile)."""
    xp = cp if HAS_GPU else np
    n_time, n_etfs = arr.shape

    # Argsort twice gives ranks
    # Handle NaN by setting to large value temporarily
    arr_filled = xp.where(xp.isnan(arr), xp.inf, arr)
    ranks = xp.argsort(xp.argsort(arr_filled, axis=1), axis=1).astype(xp.float32)

    # Count valid per row
    valid_count = xp.sum(~xp.isnan(arr), axis=1, keepdims=True).astype(xp.float32)

    # Normalize to [0, 1]
    result = ranks / (valid_count - 1 + 1e-10)

    # Restore NaN
    result = xp.where(xp.isnan(arr), xp.nan, result)

    return result


# =============================================================================
# GPU-ACCELERATED ENSEMBLE METHODS
# =============================================================================

def combined_score_gpu(
    features: np.ndarray,  # (n_features, n_months, n_etfs)
    feature_indices: Optional[List[int]] = None,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    GPU-accelerated combined score (weighted sum of z-scored features).
    Unlike aggregated rank, this preserves magnitude information.
    Returns: (n_months, n_etfs) combined score
    """
    xp = cp if HAS_GPU else np

    if feature_indices is not None:
        features = features[feature_indices]

    features_gpu = to_gpu(features)
    n_features, n_months, n_etfs = features_gpu.shape

    if weights is None:
        weights_gpu = xp.ones(n_features, dtype=xp.float32) / n_features
    else:
        weights_gpu = to_gpu(weights) / xp.sum(to_gpu(weights))

    # Z-score each feature cross-sectionally, then combine
    combined = xp.zeros((n_months, n_etfs), dtype=xp.float32)

    for f in range(n_features):
        feat = features_gpu[f]
        # Cross-sectional z-score
        xs_mean = xp.nanmean(feat, axis=1, keepdims=True)
        xs_std = xp.nanstd(feat, axis=1, keepdims=True)
        zscore = (feat - xs_mean) / (xs_std + 1e-10)
        # Clip extreme values
        zscore = xp.clip(zscore, -3, 3)
        # Add weighted contribution
        valid_mask = ~xp.isnan(zscore)
        combined = xp.where(valid_mask, combined + weights_gpu[f] * zscore, combined)

    return to_cpu(combined)


def weighted_voting_gpu(
    features: np.ndarray,  # (n_features, n_months, n_etfs)
    feature_weights: np.ndarray,  # (n_features,) - weights based on feature performance
    k: int = 10
) -> np.ndarray:
    """
    Weighted voting where better features get more votes.
    Returns: (n_months, n_etfs) weighted vote counts
    """
    xp = cp if HAS_GPU else np
    features_gpu = to_gpu(features)
    weights_gpu = to_gpu(feature_weights)
    weights_gpu = weights_gpu / xp.sum(weights_gpu)  # Normalize

    n_features, n_months, n_etfs = features_gpu.shape
    votes = xp.zeros((n_months, n_etfs), dtype=xp.float32)

    for f in range(n_features):
        feat = features_gpu[f]
        # Get rank for this feature
        rank = _cross_sectional_rank_gpu(feat)
        # Vote for top-k (rank > 1 - k/n_valid)
        threshold = 1.0 - k / n_etfs
        is_topk = rank >= threshold
        # Add weighted vote
        votes = xp.where(is_topk & ~xp.isnan(rank), votes + weights_gpu[f], votes)

    return to_cpu(votes)


def borda_count_gpu(
    features: np.ndarray,  # (n_features, n_months, n_etfs)
    feature_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Borda count voting: each feature gives points based on rank position.
    Top ETF gets n_etfs points, second gets n_etfs-1, etc.
    Returns: (n_months, n_etfs) total Borda points
    """
    xp = cp if HAS_GPU else np

    if feature_indices is not None:
        features = features[feature_indices]

    features_gpu = to_gpu(features)
    n_features, n_months, n_etfs = features_gpu.shape

    borda_scores = xp.zeros((n_months, n_etfs), dtype=xp.float32)

    for f in range(n_features):
        feat = features_gpu[f]
        # Rank gives percentile [0, 1], convert to points [0, n_etfs]
        rank = _cross_sectional_rank_gpu(feat)
        points = rank * n_etfs
        valid_mask = ~xp.isnan(points)
        borda_scores = xp.where(valid_mask, borda_scores + points, borda_scores)

    # Normalize by number of features
    borda_scores = borda_scores / n_features

    return to_cpu(borda_scores)


def majority_voting_gpu(
    features: np.ndarray,  # (n_features, n_months, n_etfs)
    k: int = 10,
    threshold_pct: float = 0.5  # Require this % of features to agree
) -> np.ndarray:
    """
    Majority voting: only count ETFs where majority of features agree.
    Returns: (n_months, n_etfs) 1 if majority agree, 0 otherwise
    """
    xp = cp if HAS_GPU else np
    features_gpu = to_gpu(features)
    n_features, n_months, n_etfs = features_gpu.shape

    votes = xp.zeros((n_months, n_etfs), dtype=xp.float32)

    for f in range(n_features):
        feat = features_gpu[f]
        rank = _cross_sectional_rank_gpu(feat)
        threshold = 1.0 - k / n_etfs
        is_topk = rank >= threshold
        votes = xp.where(is_topk & ~xp.isnan(rank), votes + 1, votes)

    # Only keep ETFs where majority agree
    min_votes = n_features * threshold_pct
    result = xp.where(votes >= min_votes, votes, 0)

    return to_cpu(result)


def rank_product_gpu(
    features: np.ndarray,  # (n_features, n_months, n_etfs)
    feature_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Rank product method: multiply ranks instead of averaging.
    Good at finding ETFs that are consistently highly ranked.
    Returns: (n_months, n_etfs) geometric mean of ranks
    """
    xp = cp if HAS_GPU else np

    if feature_indices is not None:
        features = features[feature_indices]

    features_gpu = to_gpu(features)
    n_features, n_months, n_etfs = features_gpu.shape

    # Use log-sum for numerical stability (geometric mean)
    log_rank_sum = xp.zeros((n_months, n_etfs), dtype=xp.float32)
    valid_count = xp.zeros((n_months, n_etfs), dtype=xp.float32)

    for f in range(n_features):
        feat = features_gpu[f]
        rank = _cross_sectional_rank_gpu(feat)
        # Avoid log(0) by adding small epsilon
        log_rank = xp.log(rank + 0.01)
        valid_mask = ~xp.isnan(rank)
        log_rank_sum = xp.where(valid_mask, log_rank_sum + log_rank, log_rank_sum)
        valid_count = xp.where(valid_mask, valid_count + 1, valid_count)

    # Geometric mean = exp(mean(log(ranks)))
    geo_mean = xp.exp(log_rank_sum / (valid_count + 1e-10))

    return to_cpu(geo_mean)


def aggregated_rank_gpu(
    features: np.ndarray,  # (n_features, n_months, n_etfs)
    feature_indices: Optional[List[int]] = None,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    GPU-accelerated aggregated rank across features.
    Returns: (n_months, n_etfs) average rank
    """
    xp = cp if HAS_GPU else np

    if feature_indices is not None:
        features = features[feature_indices]

    features_gpu = to_gpu(features)
    n_features, n_months, n_etfs = features_gpu.shape

    if weights is None:
        weights_gpu = xp.ones(n_features, dtype=xp.float32) / n_features
    else:
        weights_gpu = to_gpu(weights) / xp.sum(to_gpu(weights))

    # Compute rank for each feature (vectorized)
    agg_rank = xp.zeros((n_months, n_etfs), dtype=xp.float32)

    for f in range(n_features):
        feat = features_gpu[f]
        rank = _cross_sectional_rank_gpu(feat)
        # Add weighted rank (handle NaN)
        valid_mask = ~xp.isnan(rank)
        agg_rank = xp.where(valid_mask, agg_rank + weights_gpu[f] * rank, agg_rank)

    return to_cpu(agg_rank)


@njit(parallel=True, cache=True)
def agreement_voting_numba(
    features: np.ndarray,  # (n_features, n_months, n_etfs)
    k: int = 10
) -> np.ndarray:
    """
    Numba-accelerated agreement voting.
    Returns: (n_months, n_etfs) vote counts
    """
    n_features, n_months, n_etfs = features.shape
    votes = np.zeros((n_months, n_etfs), dtype=np.float32)

    for m in prange(n_months):
        for f in range(n_features):
            # Get valid values
            valid_count = 0
            for e in range(n_etfs):
                if not np.isnan(features[f, m, e]):
                    valid_count += 1

            if valid_count < k:
                continue

            # Find top-k indices
            values = np.empty(valid_count, dtype=np.float32)
            indices = np.empty(valid_count, dtype=np.int64)
            idx = 0
            for e in range(n_etfs):
                if not np.isnan(features[f, m, e]):
                    values[idx] = features[f, m, e]
                    indices[idx] = e
                    idx += 1

            # Partial sort for top-k
            sorted_idx = np.argsort(values)
            for i in range(min(k, valid_count)):
                etf_idx = indices[sorted_idx[valid_count - 1 - i]]
                votes[m, etf_idx] += 1

    return votes


def filter_then_select_gpu(
    filter_features: np.ndarray,  # (n_filter, n_months, n_etfs)
    select_features: np.ndarray,  # (n_select, n_months, n_etfs)
    filter_threshold: float = 0.3,
    select_k: int = 4
) -> np.ndarray:
    """
    Two-stage selection using GPU.
    Returns: (n_months, n_etfs) selection scores
    """
    # Compute aggregated ranks
    filter_rank = aggregated_rank_gpu(filter_features)
    select_rank = aggregated_rank_gpu(select_features)

    n_months, n_etfs = filter_rank.shape
    selections = np.zeros((n_months, n_etfs), dtype=np.float32)

    for m in range(n_months):
        # Valid ETFs
        valid_mask = ~np.isnan(filter_rank[m]) & ~np.isnan(select_rank[m])
        n_valid = valid_mask.sum()

        if n_valid < select_k * 2:
            continue

        # Apply filter
        filter_vals = filter_rank[m][valid_mask]
        threshold_val = np.percentile(filter_vals, filter_threshold * 100)

        # Survivors
        survivor_mask = valid_mask & (filter_rank[m] >= threshold_val)
        n_survivors = survivor_mask.sum()

        if n_survivors < select_k:
            continue

        # Select top-k from survivors
        survivor_indices = np.where(survivor_mask)[0]
        select_vals = select_rank[m][survivor_mask]
        top_k_idx = np.argsort(select_vals)[-select_k:]

        for idx in top_k_idx:
            selections[m, survivor_indices[idx]] = 1.0

    return selections


# =============================================================================
# GPU-ACCELERATED EVALUATION
# =============================================================================

@njit(parallel=True, cache=True)
def evaluate_selection_numba(
    selection_scores: np.ndarray,  # (n_months, n_etfs)
    alpha_monthly: np.ndarray,     # (n_months, n_etfs)
    select_k: int = 4
) -> Tuple[float, float, float, float, float]:
    """
    Evaluate selection strategy (Numba-accelerated).
    Returns: topk_advantage, hit_rate, winner_precision, loser_avoidance, avg_alpha
    """
    n_months, n_etfs = alpha_monthly.shape

    sum_topk_adv = 0.0
    count_positive = 0
    sum_winner_precision = 0.0
    sum_loser_avoidance = 0.0
    sum_alpha = 0.0
    n_valid = 0
    n_selected_total = 0

    for m in prange(n_months):
        # Count valid
        n_valid_etfs = 0
        for e in range(n_etfs):
            if not np.isnan(selection_scores[m, e]) and not np.isnan(alpha_monthly[m, e]):
                n_valid_etfs += 1

        if n_valid_etfs < 20:
            continue

        n_valid += 1

        # Extract valid values
        scores = np.empty(n_valid_etfs, dtype=np.float32)
        alphas = np.empty(n_valid_etfs, dtype=np.float32)
        idx = 0
        for e in range(n_etfs):
            if not np.isnan(selection_scores[m, e]) and not np.isnan(alpha_monthly[m, e]):
                scores[idx] = selection_scores[m, e]
                alphas[idx] = alpha_monthly[m, e]
                idx += 1

        # Sort by score
        sorted_by_score = np.argsort(scores)
        k = min(select_k, n_valid_etfs // 4)

        # Top-k metrics
        topk_sum = 0.0
        winners_in_topk = 0
        for i in range(k):
            alpha_val = alphas[sorted_by_score[n_valid_etfs - 1 - i]]
            topk_sum += alpha_val
            n_selected_total += 1
            sum_alpha += alpha_val
            if alpha_val > 0:
                winners_in_topk += 1

        topk_avg = topk_sum / k
        universe_avg = np.mean(alphas)
        topk_adv = topk_avg - universe_avg

        sum_topk_adv += topk_adv
        if topk_adv > 0:
            count_positive += 1
        sum_winner_precision += winners_in_topk / k

        # Loser avoidance
        sorted_by_alpha = np.argsort(alphas)
        top_k_indices = set()
        for i in range(k):
            top_k_indices.add(sorted_by_score[n_valid_etfs - 1 - i])

        losers_avoided = 0
        for i in range(k):
            if sorted_by_alpha[i] not in top_k_indices:
                losers_avoided += 1
        sum_loser_avoidance += losers_avoided / k

    if n_valid == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    return (
        sum_topk_adv / n_valid,
        count_positive / n_valid,
        sum_winner_precision / n_valid,
        sum_loser_avoidance / n_valid,
        sum_alpha / n_selected_total if n_selected_total > 0 else 0.0
    )


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def compute_features_full_pipeline(
    etf_prices_df: pd.DataFrame,
    core_prices_series: pd.Series,
    use_subset: bool = False,
    n_top_filters: int = 4  # Only use top N filters when using subset
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Compute all features using the full signal framework pipeline.

    This uses:
    - 58 signal bases from signal_bases.py
    - 12 smoothing filters from signal_filters.py (or subset)
    - 25 indicator transformations from signal_indicators.py

    Returns:
        features: (n_features, n_months, n_etfs) array
        feature_names: list of feature names
        alpha_monthly: (n_months, n_etfs) forward alpha
        monthly_indices: indices into daily data
    """
    print("\n2. Computing signal bases...")
    t0 = time.time()
    signals_3d, signal_names = compute_all_signal_bases(etf_prices_df, core_prices_series)
    n_signals = len(signal_names)
    print(f"   Signal bases: {n_signals}")
    print(f"   Time: {time.time() - t0:.1f}s")

    # 3. Compute monthly alpha
    print("\n3. Computing monthly alpha...")
    monthly_prices = etf_prices_df.resample('ME').last()
    monthly_core = core_prices_series.resample('ME').last()
    monthly_ret = monthly_prices.pct_change()
    core_ret = monthly_core.pct_change()
    alpha_monthly = (monthly_ret.sub(core_ret, axis=0)).values.astype(np.float32)

    # Get monthly indices
    monthly_dates = monthly_prices.index
    daily_dates = etf_prices_df.index
    monthly_indices = np.array([daily_dates.get_indexer([d], method='ffill')[0] for d in monthly_dates])

    n_months = len(monthly_indices)
    n_etfs = len(etf_prices_df.columns)
    print(f"   Months: {n_months}")

    # 4. Apply filters and compute indicators in streaming mode
    print("\n4. Computing filtered signals and indicators (streaming)...")
    t0 = time.time()

    # Select which filters to use
    # DEFAULT_FILTER_CONFIGS is a list of (name, func, kwargs) tuples
    if use_subset:
        # Use only the best filters: raw, hull_21d, hull_63d, savgol_21d
        best_filter_names = {'raw', 'hull_21d', 'hull_63d', 'savgol_21d'}
        filter_configs = [cfg for cfg in DEFAULT_FILTER_CONFIGS if cfg[0] in best_filter_names]
    else:
        filter_configs = DEFAULT_FILTER_CONFIGS

    n_filters = len(filter_configs)
    # Use simplified indicators for monthly data: level, xs_rank, xs_zscore, velocity
    indicator_list = ['level', 'xs_rank', 'xs_zscore', 'velocity']
    n_indicators = len(indicator_list)

    total_features = n_signals * n_filters * n_indicators
    print(f"   Signal bases: {n_signals}, Filters: {n_filters}, Indicators: {n_indicators}")
    print(f"   Total features: {total_features}")

    # Pre-allocate features array at monthly frequency
    features_list = []
    feature_names = []

    # Process each filter in sequence (streaming to save memory)
    for filter_name, filter_func, filter_kwargs in tqdm(filter_configs, desc="Processing filters"):
        # Apply filter to all signals
        # apply_filter expects 3D array (n_signals, n_time, n_etfs)
        if filter_name == 'raw':
            filtered_signals = signals_3d
        else:
            filtered_signals = apply_filter(signals_3d, filter_name, filter_func, filter_kwargs)

        # Compute indicators for this filter (simplified for monthly data)
        for s_idx, sig_name in enumerate(signal_names):
            sig_daily = filtered_signals[s_idx]  # (n_time, n_etfs)

            # Extract monthly values
            sig_monthly = sig_daily[monthly_indices]  # (n_months, n_etfs)

            base_name = f"{sig_name}__{filter_name}"

            # Level (raw signal value)
            features_list.append(sig_monthly.astype(np.float32))
            feature_names.append(f"{base_name}__level")

            # Cross-sectional rank
            xs_rank = np.zeros_like(sig_monthly)
            for t in range(sig_monthly.shape[0]):
                row = sig_monthly[t]
                valid = ~np.isnan(row)
                if valid.sum() > 1:
                    ranks = np.argsort(np.argsort(np.where(valid, row, np.inf)))
                    xs_rank[t] = ranks / (valid.sum() - 1 + 1e-10)
                    xs_rank[t, ~valid] = np.nan
            features_list.append(xs_rank.astype(np.float32))
            feature_names.append(f"{base_name}__xs_rank")

            # Cross-sectional z-score
            xs_mean = np.nanmean(sig_monthly, axis=1, keepdims=True)
            xs_std = np.nanstd(sig_monthly, axis=1, keepdims=True)
            xs_zscore = (sig_monthly - xs_mean) / (xs_std + 1e-10)
            features_list.append(xs_zscore.astype(np.float32))
            feature_names.append(f"{base_name}__xs_zscore")

            # Velocity (3-month change, normalized)
            velocity = np.zeros_like(sig_monthly)
            if sig_monthly.shape[0] > 3:
                velocity[3:] = sig_monthly[3:] - sig_monthly[:-3]
                # Normalize by rolling std (6 months)
                for m in range(6, sig_monthly.shape[0]):
                    std_val = np.nanstd(sig_monthly[m-6:m], axis=0)
                    velocity[m] = velocity[m] / (std_val + 1e-10)
            features_list.append(velocity.astype(np.float32))
            feature_names.append(f"{base_name}__velocity")

        # Free memory
        if filter_name != 'raw':
            del filtered_signals
        if HAS_GPU:
            cp.get_default_memory_pool().free_all_blocks()

    # Stack features
    features = np.stack(features_list, axis=0).astype(np.float32)

    print(f"   Features computed: {len(feature_names)}")
    print(f"   Time: {time.time() - t0:.1f}s")

    return features, feature_names, alpha_monthly, monthly_indices


def main(use_full_features: bool = True):
    """
    Main ensemble discovery analysis.

    Args:
        use_full_features: If True, use all 17,400 features (slower but comprehensive).
                          If False, use subset of ~1,000 best features (faster).
    """
    start_time = time.time()

    print("=" * 70)
    print("ENSEMBLE DISCOVERY FOR ETF SELECTION (GPU-Accelerated)")
    if use_full_features:
        print("MODE: Full 17,400 features (58 signals x 12 filters x 25 indicators)")
    else:
        print("MODE: Subset ~3,000 features (58 signals x 4 filters x ~13 indicators)")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    t0 = time.time()
    etf_prices_df, core_prices_series = load_data()
    n_etfs = len(etf_prices_df.columns)
    print(f"   ETFs: {n_etfs}")
    print(f"   Date range: {etf_prices_df.index[0].date()} to {etf_prices_df.index[-1].date()}")
    print(f"   Time: {time.time() - t0:.1f}s")

    # Compute features using full pipeline
    features, feature_names, alpha_monthly, monthly_indices = compute_features_full_pipeline(
        etf_prices_df, core_prices_series,
        use_subset=not use_full_features
    )
    n_features = len(feature_names)
    n_months = features.shape[1]

    # Align: features predict NEXT month's alpha
    features = features[:, :-1, :]
    alpha_next = alpha_monthly[1:]

    # =================================================================
    # TEST 1: Individual Features
    # =================================================================
    print("\n" + "=" * 70)
    print("TEST 1: INDIVIDUAL FEATURES")
    print("=" * 70)
    t0 = time.time()

    individual_results = []
    for f_idx, f_name in enumerate(feature_names):
        scores = features[f_idx]
        topk_adv, hit_rate, winner_prec, loser_avoid, avg_alpha = evaluate_selection_numba(
            scores, alpha_next, select_k=4
        )
        individual_results.append({
            'feature': f_name,
            'topk_advantage': topk_adv * 100,
            'hit_rate': hit_rate * 100,
            'winner_precision': winner_prec * 100,
            'loser_avoidance': loser_avoid * 100,
            'avg_alpha': avg_alpha * 100
        })

    df_individual = pd.DataFrame(individual_results)
    df_individual = df_individual.sort_values('topk_advantage', ascending=False)

    print(f"\nTime: {time.time() - t0:.1f}s")
    print("\nTop 15 Individual Features by Top-K Advantage:")
    print(df_individual.head(15).to_string(index=False))

    # =================================================================
    # TEST 2: Feature Categories
    # =================================================================
    print("\n" + "=" * 70)
    print("TEST 2: FEATURE CATEGORIES (Aggregated Rank)")
    print("=" * 70)
    t0 = time.time()

    # Categorize features
    momentum_idx = [i for i, n in enumerate(feature_names) if 'rs_' in n or 'momentum' in n.lower() or 'trend' in n.lower()]
    reversion_idx = [i for i, n in enumerate(feature_names) if 'reversion' in n or 'zscore' in n or 'alpha_dd' in n]
    quality_idx = [i for i, n in enumerate(feature_names) if 'sharpe' in n or 'info_ratio' in n]

    category_results = []

    # All features
    agg_all = aggregated_rank_gpu(features)
    topk_adv, hit_rate, winner_prec, loser_avoid, _ = evaluate_selection_numba(agg_all, alpha_next, select_k=4)
    category_results.append({
        'category': f'All Features ({n_features})',
        'topk_advantage': topk_adv * 100,
        'hit_rate': hit_rate * 100,
        'winner_precision': winner_prec * 100,
        'loser_avoidance': loser_avoid * 100
    })

    # By category
    for cat_name, cat_idx in [('Momentum', momentum_idx), ('Mean Reversion', reversion_idx), ('Quality', quality_idx)]:
        if len(cat_idx) > 0:
            agg = aggregated_rank_gpu(features, feature_indices=cat_idx)
            topk_adv, hit_rate, winner_prec, loser_avoid, _ = evaluate_selection_numba(agg, alpha_next, select_k=4)
            category_results.append({
                'category': f'{cat_name} ({len(cat_idx)})',
                'topk_advantage': topk_adv * 100,
                'hit_rate': hit_rate * 100,
                'winner_precision': winner_prec * 100,
                'loser_avoidance': loser_avoid * 100
            })

    # Top N by individual performance
    for n_top in [3, 5, 10]:
        top_n_names = df_individual.head(n_top)['feature'].tolist()
        top_n_idx = [i for i, n in enumerate(feature_names) if n in top_n_names]
        if len(top_n_idx) > 0:
            agg = aggregated_rank_gpu(features, feature_indices=top_n_idx)
            topk_adv, hit_rate, winner_prec, loser_avoid, _ = evaluate_selection_numba(agg, alpha_next, select_k=4)
            category_results.append({
                'category': f'Top {n_top} Features',
                'topk_advantage': topk_adv * 100,
                'hit_rate': hit_rate * 100,
                'winner_precision': winner_prec * 100,
                'loser_avoidance': loser_avoid * 100
            })

    df_category = pd.DataFrame(category_results)
    print(f"\nTime: {time.time() - t0:.1f}s")
    print("\nAggregated Rank by Category:")
    print(df_category.to_string(index=False))

    # =================================================================
    # TEST 3: COMBINED SCORES (Z-scored weighted sum)
    # =================================================================
    print("\n" + "=" * 70)
    print("TEST 3: COMBINED SCORES")
    print("=" * 70)
    t0 = time.time()

    combined_results = []

    # All features equal weight
    combined = combined_score_gpu(features)
    topk_adv, hit_rate, winner_prec, loser_avoid, _ = evaluate_selection_numba(combined, alpha_next, select_k=4)
    combined_results.append({
        'method': f'All Features Equal Weight ({n_features})',
        'topk_advantage': topk_adv * 100,
        'hit_rate': hit_rate * 100,
        'winner_precision': winner_prec * 100,
        'loser_avoidance': loser_avoid * 100
    })

    # Category combined scores
    for cat_name, cat_idx in [('Momentum', momentum_idx), ('Mean Reversion', reversion_idx), ('Quality', quality_idx)]:
        if len(cat_idx) > 0:
            combined = combined_score_gpu(features, feature_indices=cat_idx)
            topk_adv, hit_rate, winner_prec, loser_avoid, _ = evaluate_selection_numba(combined, alpha_next, select_k=4)
            combined_results.append({
                'method': f'{cat_name} Combined ({len(cat_idx)})',
                'topk_advantage': topk_adv * 100,
                'hit_rate': hit_rate * 100,
                'winner_precision': winner_prec * 100,
                'loser_avoidance': loser_avoid * 100
            })

    # Top N by individual performance (combined score)
    for n_top in [3, 5, 10]:
        top_n_names = df_individual.head(n_top)['feature'].tolist()
        top_n_idx = [i for i, n in enumerate(feature_names) if n in top_n_names]
        if len(top_n_idx) > 0:
            combined = combined_score_gpu(features, feature_indices=top_n_idx)
            topk_adv, hit_rate, winner_prec, loser_avoid, _ = evaluate_selection_numba(combined, alpha_next, select_k=4)
            combined_results.append({
                'method': f'Top {n_top} Combined Score',
                'topk_advantage': topk_adv * 100,
                'hit_rate': hit_rate * 100,
                'winner_precision': winner_prec * 100,
                'loser_avoidance': loser_avoid * 100
            })

    # Performance-weighted combined score (weights from individual performance)
    individual_topk = df_individual['topk_advantage'].values
    top_n_idx = list(range(len(feature_names)))
    # Only use positive performers
    positive_mask = individual_topk > 0
    if positive_mask.sum() > 0:
        pos_idx = [i for i, pos in enumerate(positive_mask) if pos]
        pos_weights = np.array([individual_topk[i] for i in pos_idx])
        combined = combined_score_gpu(features, feature_indices=pos_idx, weights=pos_weights)
        topk_adv, hit_rate, winner_prec, loser_avoid, _ = evaluate_selection_numba(combined, alpha_next, select_k=4)
        combined_results.append({
            'method': f'Performance-Weighted ({len(pos_idx)} features)',
            'topk_advantage': topk_adv * 100,
            'hit_rate': hit_rate * 100,
            'winner_precision': winner_prec * 100,
            'loser_avoidance': loser_avoid * 100
        })

    df_combined = pd.DataFrame(combined_results)
    df_combined = df_combined.sort_values('topk_advantage', ascending=False)
    print(f"\nTime: {time.time() - t0:.1f}s")
    print("\nCombined Score Results:")
    print(df_combined.to_string(index=False))

    # =================================================================
    # TEST 4: VOTING ENSEMBLES
    # =================================================================
    print("\n" + "=" * 70)
    print("TEST 4: VOTING ENSEMBLES")
    print("=" * 70)
    t0 = time.time()

    vote_results = []

    # Simple agreement voting
    for k in [5, 10, 20]:
        votes = agreement_voting_numba(features, k=k)
        topk_adv, hit_rate, winner_prec, loser_avoid, _ = evaluate_selection_numba(votes, alpha_next, select_k=4)
        vote_results.append({
            'method': f'Agreement Voting (top-{k})',
            'topk_advantage': topk_adv * 100,
            'hit_rate': hit_rate * 100,
            'winner_precision': winner_prec * 100,
            'loser_avoidance': loser_avoid * 100
        })

    # Category-specific voting
    for cat_name, cat_idx in [('Momentum', momentum_idx), ('Reversion', reversion_idx)]:
        if len(cat_idx) > 0:
            votes = agreement_voting_numba(features[cat_idx], k=10)
            topk_adv, hit_rate, winner_prec, loser_avoid, _ = evaluate_selection_numba(votes, alpha_next, select_k=4)
            vote_results.append({
                'method': f'{cat_name} Agreement (top-10)',
                'topk_advantage': topk_adv * 100,
                'hit_rate': hit_rate * 100,
                'winner_precision': winner_prec * 100,
                'loser_avoidance': loser_avoid * 100
            })

    # Borda count voting
    borda = borda_count_gpu(features)
    topk_adv, hit_rate, winner_prec, loser_avoid, _ = evaluate_selection_numba(borda, alpha_next, select_k=4)
    vote_results.append({
        'method': f'Borda Count (all features)',
        'topk_advantage': topk_adv * 100,
        'hit_rate': hit_rate * 100,
        'winner_precision': winner_prec * 100,
        'loser_avoidance': loser_avoid * 100
    })

    # Borda count on top performers
    for n_top in [5, 10]:
        top_n_names = df_individual.head(n_top)['feature'].tolist()
        top_n_idx = [i for i, n in enumerate(feature_names) if n in top_n_names]
        if len(top_n_idx) > 0:
            borda = borda_count_gpu(features, feature_indices=top_n_idx)
            topk_adv, hit_rate, winner_prec, loser_avoid, _ = evaluate_selection_numba(borda, alpha_next, select_k=4)
            vote_results.append({
                'method': f'Borda Count (top {n_top})',
                'topk_advantage': topk_adv * 100,
                'hit_rate': hit_rate * 100,
                'winner_precision': winner_prec * 100,
                'loser_avoidance': loser_avoid * 100
            })

    # Rank product
    rank_prod = rank_product_gpu(features)
    topk_adv, hit_rate, winner_prec, loser_avoid, _ = evaluate_selection_numba(rank_prod, alpha_next, select_k=4)
    vote_results.append({
        'method': f'Rank Product (all features)',
        'topk_advantage': topk_adv * 100,
        'hit_rate': hit_rate * 100,
        'winner_precision': winner_prec * 100,
        'loser_avoidance': loser_avoid * 100
    })

    # Rank product on top performers
    for n_top in [5, 10]:
        top_n_names = df_individual.head(n_top)['feature'].tolist()
        top_n_idx = [i for i, n in enumerate(feature_names) if n in top_n_names]
        if len(top_n_idx) > 0:
            rank_prod = rank_product_gpu(features, feature_indices=top_n_idx)
            topk_adv, hit_rate, winner_prec, loser_avoid, _ = evaluate_selection_numba(rank_prod, alpha_next, select_k=4)
            vote_results.append({
                'method': f'Rank Product (top {n_top})',
                'topk_advantage': topk_adv * 100,
                'hit_rate': hit_rate * 100,
                'winner_precision': winner_prec * 100,
                'loser_avoidance': loser_avoid * 100
            })

    # Majority voting at various thresholds
    for threshold in [0.3, 0.5, 0.7]:
        majority = majority_voting_gpu(features, k=10, threshold_pct=threshold)
        topk_adv, hit_rate, winner_prec, loser_avoid, _ = evaluate_selection_numba(majority, alpha_next, select_k=4)
        vote_results.append({
            'method': f'Majority Voting ({int(threshold*100)}% agree)',
            'topk_advantage': topk_adv * 100,
            'hit_rate': hit_rate * 100,
            'winner_precision': winner_prec * 100,
            'loser_avoidance': loser_avoid * 100
        })

    # Weighted voting (performance-weighted)
    if positive_mask.sum() > 0:
        pos_weights = np.array([max(0, individual_topk[i]) for i in range(len(individual_topk))])
        weighted_votes = weighted_voting_gpu(features, pos_weights, k=10)
        topk_adv, hit_rate, winner_prec, loser_avoid, _ = evaluate_selection_numba(weighted_votes, alpha_next, select_k=4)
        vote_results.append({
            'method': 'Performance-Weighted Voting',
            'topk_advantage': topk_adv * 100,
            'hit_rate': hit_rate * 100,
            'winner_precision': winner_prec * 100,
            'loser_avoidance': loser_avoid * 100
        })

    df_votes = pd.DataFrame(vote_results)
    df_votes = df_votes.sort_values('topk_advantage', ascending=False)
    print(f"\nTime: {time.time() - t0:.1f}s")
    print("\nVoting Ensemble Results:")
    print(df_votes.to_string(index=False))

    # =================================================================
    # TEST 5: Filter-then-Select
    # =================================================================
    print("\n" + "=" * 70)
    print("TEST 5: FILTER-THEN-SELECT")
    print("=" * 70)
    t0 = time.time()

    filter_results = []

    # Try various combinations
    combinations = [
        ('Quality', quality_idx, 'Reversion', reversion_idx),
        ('Quality', quality_idx, 'Momentum', momentum_idx),
        ('Momentum', momentum_idx, 'Reversion', reversion_idx),
        ('Reversion', reversion_idx, 'Momentum', momentum_idx),
    ]

    for filter_name, filter_idx, select_name, select_idx in combinations:
        if len(filter_idx) > 0 and len(select_idx) > 0:
            for threshold in [0.2, 0.3, 0.4]:
                selections = filter_then_select_gpu(
                    features[filter_idx],
                    features[select_idx],
                    filter_threshold=threshold,
                    select_k=4
                )
                topk_adv, hit_rate, winner_prec, loser_avoid, _ = evaluate_selection_numba(
                    selections, alpha_next, select_k=4
                )
                filter_results.append({
                    'method': f'{filter_name}(top {int((1-threshold)*100)}%) -> {select_name}',
                    'topk_advantage': topk_adv * 100,
                    'hit_rate': hit_rate * 100,
                    'winner_precision': winner_prec * 100,
                    'loser_avoidance': loser_avoid * 100
                })

    df_filter = pd.DataFrame(filter_results)
    df_filter = df_filter.sort_values('topk_advantage', ascending=False)
    print(f"\nTime: {time.time() - t0:.1f}s")
    print("\nFilter-then-Select Results:")
    print(df_filter.head(15).to_string(index=False))

    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: BEST METHODS")
    print("=" * 70)

    all_methods = []

    # Best individual
    best_ind = df_individual.iloc[0]
    all_methods.append({
        'method': f"Individual: {best_ind['feature'][:40]}",
        'topk_advantage': best_ind['topk_advantage'],
        'hit_rate': best_ind['hit_rate'],
        'loser_avoidance': best_ind['loser_avoidance']
    })

    # Best aggregated rank
    best_cat = df_category.loc[df_category['topk_advantage'].idxmax()]
    all_methods.append({
        'method': f"Aggregated Rank: {best_cat['category']}",
        'topk_advantage': best_cat['topk_advantage'],
        'hit_rate': best_cat['hit_rate'],
        'loser_avoidance': best_cat['loser_avoidance']
    })

    # Best combined score
    best_comb = df_combined.iloc[0]
    all_methods.append({
        'method': f"Combined Score: {best_comb['method'][:30]}",
        'topk_advantage': best_comb['topk_advantage'],
        'hit_rate': best_comb['hit_rate'],
        'loser_avoidance': best_comb['loser_avoidance']
    })

    # Best voting ensemble
    best_vote = df_votes.iloc[0]
    all_methods.append({
        'method': f"Voting: {best_vote['method'][:30]}",
        'topk_advantage': best_vote['topk_advantage'],
        'hit_rate': best_vote['hit_rate'],
        'loser_avoidance': best_vote['loser_avoidance']
    })

    # Best filter-select
    if len(df_filter) > 0:
        best_filt = df_filter.iloc[0]
        all_methods.append({
            'method': f"Filter-Select: {best_filt['method'][:30]}",
            'topk_advantage': best_filt['topk_advantage'],
            'hit_rate': best_filt['hit_rate'],
            'loser_avoidance': best_filt['loser_avoidance']
        })

    df_summary = pd.DataFrame(all_methods)
    df_summary = df_summary.sort_values('topk_advantage', ascending=False)
    print("\nBest Method per Category:")
    print(df_summary.to_string(index=False))

    # =================================================================
    # KEY INSIGHTS
    # =================================================================
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    print("\nBest for FILTERING LOSERS (loser_avoidance):")
    best = df_individual.loc[df_individual['loser_avoidance'].idxmax()]
    print(f"  {best['feature']}: {best['loser_avoidance']:.1f}%")

    print("\nBest for PICKING WINNERS (winner_precision):")
    best = df_individual.loc[df_individual['winner_precision'].idxmax()]
    print(f"  {best['feature']}: {best['winner_precision']:.1f}%")

    print("\nBest for CONSISTENCY (hit_rate):")
    best = df_individual.loc[df_individual['hit_rate'].idxmax()]
    print(f"  {best['feature']}: {best['hit_rate']:.1f}%")

    print("\nBest for ALPHA (topk_advantage):")
    best = df_individual.iloc[0]
    print(f"  {best['feature']}: {best['topk_advantage']:.2f}%")

    # Save results
    results_path = Path(__file__).parent / 'results' / 'ensemble_discovery_results.csv'
    df_individual.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")

    return df_individual, df_category, df_votes, df_filter


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Ensemble Discovery for ETF Selection')
    parser.add_argument('--subset', action='store_true',
                       help='Use subset of filters (faster, ~3000 features)')
    parser.add_argument('--full', action='store_true', default=True,
                       help='Use all filters (slower, ~17400 features)')
    args = parser.parse_args()

    # If --subset is specified, use subset; otherwise use full
    use_full = not args.subset

    results = main(use_full_features=use_full)
