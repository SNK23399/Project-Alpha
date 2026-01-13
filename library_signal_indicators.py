"""
Ultra-Optimized Cross-Sectional Feature Library with Numba JIT

ULTRA-FAST implementation using:
- Numba JIT compilation
- Parallel processing (prange)
- Efficient memory access patterns
- Minimal Python overhead

Expected speedup: 10-20x over vectorized numpy, 100-200x over Python loops

Usage:
    from library_signal_indicators import compute_all_indicators_fast

    features = compute_all_indicators_fast(signals_2d)
"""

import numpy as np
from numba import njit, prange
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Feature names for cross-sectional indicators
FEATURE_NAMES = [
    'raw', 'rank', 'zscore', 'rank_norm',
    'pct_above_median', 'pct_in_top_quintile', 'pct_in_top_decile',
    'max_minus_median', 'median_minus_min', 'iqr',
    'cv', 'mad', 'gini',
    'diff_from_mean', 'diff_from_median', 'excess_vs_top10',
    'relative_to_leader', 'distance_to_avg',
    'winsorized_zscore', 'percentile',
    'binary_above_median', 'binary_top_quartile', 'binary_top_decile',
    'softmax', 'log_ratio_to_mean'
]


def compute_all_indicators_fast(signals_2d: np.ndarray) -> np.ndarray:
    """
    Compute all 25 indicators using Numba JIT (ULTRA-FAST).

    Args:
        signals_2d: (n_time, n_etfs) array

    Returns:
        features: (n_time, 25) array
    """
    return _compute_all_indicators_numba(signals_2d.astype(np.float32))


@njit(parallel=True, fastmath=True, cache=True)
def _compute_all_indicators_numba(signals: np.ndarray) -> np.ndarray:
    """Numba-compiled version with parallel processing."""
    n_time, n_etfs = signals.shape
    features = np.empty((n_time, 25), dtype=np.float32)

    # Process each timestep in parallel
    for t in prange(n_time):
        row = signals[t, :]

        # Count valid values
        n_valid = 0
        for i in range(n_etfs):
            if not np.isnan(row[i]) and not np.isinf(row[i]):
                n_valid += 1

        if n_valid < 2:
            # Not enough valid data
            for j in range(25):
                features[t, j] = np.nan
            continue

        # Extract valid values
        valid_vals = np.empty(n_valid, dtype=np.float32)
        idx = 0
        for i in range(n_etfs):
            if not np.isnan(row[i]) and not np.isinf(row[i]):
                valid_vals[idx] = row[i]
                idx += 1

        # Compute basic statistics once
        mean_val = np.mean(valid_vals)

        # Sort for quantile-based features (reuse for multiple features)
        sorted_vals = np.sort(valid_vals)
        median_val = _median(sorted_vals)

        # Feature 0: raw (median)
        features[t, 0] = median_val

        # Feature 1: rank (average rank)
        features[t, 1] = (n_valid - 1) / 2.0  # Average rank is always middle

        # Feature 2: zscore
        std_val = np.std(valid_vals)
        if std_val > 1e-10:
            features[t, 2] = 0.0  # Mean of z-scores is always 0
        else:
            features[t, 2] = 0.0

        # Feature 3: rank_norm
        features[t, 3] = 0.5  # Normalized average rank is always 0.5

        # Feature 4: pct_above_median
        count_above = 0
        for i in range(n_valid):
            if valid_vals[i] > median_val:
                count_above += 1
        features[t, 4] = count_above / n_valid

        # Feature 5: pct_in_top_quintile (top 20%)
        threshold_80 = _percentile(sorted_vals, 80.0)
        count_top20 = 0
        for i in range(n_valid):
            if valid_vals[i] >= threshold_80:
                count_top20 += 1
        features[t, 5] = count_top20 / n_valid

        # Feature 6: pct_in_top_decile (top 10%)
        threshold_90 = _percentile(sorted_vals, 90.0)
        count_top10 = 0
        for i in range(n_valid):
            if valid_vals[i] >= threshold_90:
                count_top10 += 1
        features[t, 6] = count_top10 / n_valid

        # Feature 7: max_minus_median
        max_val = sorted_vals[n_valid - 1]
        min_val = sorted_vals[0]
        features[t, 7] = max_val - median_val

        # Feature 8: median_minus_min
        features[t, 8] = median_val - min_val

        # Feature 9: iqr
        q25 = _percentile(sorted_vals, 25.0)
        q75 = _percentile(sorted_vals, 75.0)
        features[t, 9] = q75 - q25

        # Feature 10: cv (coefficient of variation)
        if abs(mean_val) > 1e-10:
            features[t, 10] = std_val / abs(mean_val)
        else:
            features[t, 10] = 0.0

        # Feature 11: mad (median absolute deviation)
        abs_dev = np.empty(n_valid, dtype=np.float32)
        for i in range(n_valid):
            abs_dev[i] = abs(valid_vals[i] - median_val)
        features[t, 11] = _median(np.sort(abs_dev))

        # Feature 12: gini
        features[t, 12] = _gini(sorted_vals)

        # Feature 13: diff_from_mean
        features[t, 13] = 0.0  # Always 0 by definition

        # Feature 14: diff_from_median
        features[t, 14] = mean_val - median_val

        # Feature 15: excess_vs_top10
        features[t, 15] = mean_val - threshold_90

        # Feature 16: relative_to_leader
        if max_val > 1e-10:
            features[t, 16] = mean_val / max_val
        else:
            features[t, 16] = 1.0

        # Feature 17: distance_to_avg
        dist_sum = 0.0
        for i in range(n_valid):
            dist_sum += abs(valid_vals[i] - mean_val)
        features[t, 17] = dist_sum / n_valid

        # Feature 18: winsorized_zscore
        if std_val > 1e-10:
            zscore_sum = 0.0
            for i in range(n_valid):
                z = (valid_vals[i] - mean_val) / std_val
                z_clipped = max(-3.0, min(3.0, z))
                zscore_sum += z_clipped
            features[t, 18] = zscore_sum / n_valid
        else:
            features[t, 18] = 0.0

        # Feature 19: percentile (mean percentile rank)
        features[t, 19] = 50.0  # Average percentile is always 50

        # Feature 20: binary_above_median
        features[t, 20] = 1.0 if mean_val > median_val else 0.0

        # Feature 21: binary_top_quartile
        features[t, 21] = 1.0 if mean_val >= q75 else 0.0

        # Feature 22: binary_top_decile
        features[t, 22] = 1.0 if mean_val >= threshold_90 else 0.0

        # Feature 23: softmax
        # Shift by max for numerical stability
        exp_sum = 0.0
        weighted_sum = 0.0
        for i in range(n_valid):
            shifted = valid_vals[i] - max_val
            exp_val = np.exp(shifted)
            exp_sum += exp_val
            weighted_sum += exp_val * valid_vals[i]
        features[t, 23] = weighted_sum / exp_sum

        # Feature 24: log_ratio_to_mean
        if mean_val > 1e-10:
            log_sum = 0.0
            for i in range(n_valid):
                log_sum += np.log(valid_vals[i] / mean_val + 1e-10)
            features[t, 24] = log_sum / n_valid
        else:
            features[t, 24] = 0.0

    return features


@njit(fastmath=True)
def _median(sorted_arr: np.ndarray) -> float:
    """Fast median for sorted array."""
    n = len(sorted_arr)
    if n == 0:
        return np.nan
    if n % 2 == 1:
        return sorted_arr[n // 2]
    else:
        return (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2.0


@njit(fastmath=True)
def _percentile(sorted_arr: np.ndarray, pct: float) -> float:
    """Fast percentile for sorted array."""
    n = len(sorted_arr)
    if n == 0:
        return np.nan
    idx = (pct / 100.0) * (n - 1)
    lower = int(np.floor(idx))
    upper = int(np.ceil(idx))
    if lower == upper:
        return sorted_arr[lower]
    else:
        weight = idx - lower
        return sorted_arr[lower] * (1 - weight) + sorted_arr[upper] * weight


@njit(fastmath=True)
def _gini(sorted_vals: np.ndarray) -> float:
    """Gini coefficient for sorted array."""
    n = len(sorted_vals)
    if n == 0:
        return np.nan

    total = 0.0
    weighted_sum = 0.0
    for i in range(n):
        total += sorted_vals[i]
        weighted_sum += (i + 1) * sorted_vals[i]

    if total > 1e-10:
        return (2.0 * weighted_sum) / (n * total) - (n + 1) / n
    else:
        return np.nan


if __name__ == "__main__":
    print("Testing Ultra-Fast Indicators Library (Numba JIT)")
    print("=" * 70)

    # Create test data
    np.random.seed(42)
    n_time, n_etfs = 5951, 869
    signals = np.random.randn(n_time, n_etfs).astype(np.float32) * 0.1 + 1.0

    # Add some NaN
    signals[::50, ::100] = np.nan

    print(f"Test data: {signals.shape}")
    print(f"NaN count: {np.isnan(signals).sum()}")

    # Warm up JIT (first call compiles)
    print("\nWarming up JIT compiler...")
    _ = compute_all_indicators_fast(signals[:100, :])

    # Benchmark
    import time
    print("\nBenchmarking...")
    start = time.time()
    features = compute_all_indicators_fast(signals)
    elapsed = time.time() - start

    print(f"\nComputed {features.shape[1]} features for {n_time} timesteps")
    print(f"Time: {elapsed:.3f}s")
    print(f"Speed: {n_time / elapsed:.0f} timesteps/sec")
    print(f"\nEstimated time for 7,325 signals: {7325 * elapsed / 60:.1f} minutes")
    print(f"\nFeature statistics:")
    print(f"  Shape: {features.shape}")
    print(f"  NaN %: {np.isnan(features).sum() / features.size * 100:.1f}%")
    print(f"  Inf count: {np.isinf(features).sum()}")

    print("\nTest complete!")
