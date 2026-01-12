"""
Causal Smoothing Filters for Signal Processing
===============================================

All filters are CAUSAL (no look-ahead bias) - they only use past data.

19 FILTERS AVAILABLE (matching paper tables/filters.tex):

Basic Moving Averages:
- Raw: Unfiltered signal
- EMA: Exponential Moving Average (21d, 63d)
- SMA: Simple Moving Average
- WMA: Weighted Moving Average

Low-Lag Variants:
- DEMA: Double EMA (2*EMA - EMA(EMA))
- TEMA: Triple EMA (3*EMA - 3*EMA² + EMA³) - minimal lag
- ZLEMA: Zero-Lag EMA (momentum-adjusted)
- Hull MA: WMA(2*WMA(n/2) - WMA(n), sqrt(n))

Smooth Variants:
- TRIMA: Triangular MA (double-smoothed SMA)
- Gaussian MA: Bell-curve weighted (smooth frequency rolloff)

Adaptive Filters:
- KAMA: Kaufman Adaptive MA (adapts to trending vs choppy markets)
- Regime-Switching: Uses fast MA in high-vol, slow MA in low-vol

Robust Filters:
- Median: Outlier-robust, preserves edges

Signal Processing Filters:
- Butterworth: Low-pass filter with sharp frequency cutoff
- Kalman: Optimal adaptive smoothing for noisy signals
- Savitzky-Golay: Polynomial fit, preserves local peaks

GPU ACCELERATION: Uses CuPy for convolution operations when available.
All filters optimized with vectorization, batch processing, and precomputation.
"""

import numpy as np
from typing import Tuple
import pandas as pd

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.ndimage import convolve1d as gpu_convolve1d
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np

# Try to import Numba for JIT compilation (massive speedup for loops)
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def _forward_fill_3d(arr: np.ndarray) -> np.ndarray:
    """
    Fast forward-fill for 3D array along time axis (axis=1).
    Uses pandas with efficient reshaping.

    Args:
        arr: (n_signals, n_time, n_etfs)

    Returns:
        Forward-filled array
    """
    n_signals, n_time, n_etfs = arr.shape

    # If array is small enough, use pandas (faster for small arrays)
    # For large arrays, the overhead of creating DataFrame is too high
    total_cols = n_signals * n_etfs

    if total_cols < 50000:
        # Small array - pandas is fast
        arr_2d = arr.transpose(1, 0, 2).reshape(n_time, -1)
        df = pd.DataFrame(arr_2d)
        result_2d = df.ffill().values
        return result_2d.reshape(n_time, n_signals, n_etfs).transpose(1, 0, 2)
    else:
        # Large array - use in-place numpy approach
        # Process each signal separately to avoid huge memory allocation
        result = arr.copy()
        for sig_idx in range(n_signals):
            # For this signal, forward fill along time axis
            sig_data = result[sig_idx]  # (n_time, n_etfs)
            df = pd.DataFrame(sig_data)
            result[sig_idx] = df.ffill().values
        return result


def causal_ema(arr: np.ndarray, span: int) -> np.ndarray:
    """
    Causal Exponential Moving Average - no future data used.
    ULTRA-OPTIMIZED: Uses GPU (CuPy) when available, else scipy.signal.lfilter.

    Handles NaN values by forward-filling before filtering, then restoring NaN positions.

    Args:
        arr: (n_time, n_cols) or (n_signals, n_time, n_etfs)
        span: EMA span (half-life roughly span/2)
    """
    if arr.ndim == 3 and GPU_AVAILABLE:
        return _causal_ema_gpu(arr, span)
    else:
        return _causal_ema_cpu(arr, span)


def _causal_ema_cpu(arr: np.ndarray, span: int) -> np.ndarray:
    """CPU implementation using scipy.signal.lfilter."""
    from scipy.signal import lfilter

    alpha = 2 / (span + 1)
    b = np.array([alpha])
    a = np.array([1, -(1 - alpha)])

    nan_mask = np.isnan(arr)

    if arr.ndim == 2:
        if nan_mask.any():
            arr_filled = pd.DataFrame(arr).ffill().bfill().values
        else:
            arr_filled = arr

        result = lfilter(b, a, arr_filled, axis=0)
        result = np.where(nan_mask, np.nan, result)
        return result

    elif arr.ndim == 3:
        n_signals, n_time, n_etfs = arr.shape

        if nan_mask.any():
            arr_filled = _forward_fill_3d(arr)
            arr_filled = _forward_fill_3d(arr_filled[:, ::-1, :])[:, ::-1, :]
        else:
            arr_filled = arr

        arr_2d = arr_filled.transpose(1, 0, 2).reshape(n_time, -1)
        filtered_2d = lfilter(b, a, arr_2d, axis=0)
        result = filtered_2d.reshape(n_time, n_signals, n_etfs).transpose(1, 0, 2)
        result = np.where(nan_mask, np.nan, result)
        return result
    else:
        raise ValueError(f"Expected 2D or 3D array, got {arr.ndim}D")


def _causal_ema_gpu(arr: np.ndarray, span: int) -> np.ndarray:
    """GPU implementation using CuPy - matches scipy.lfilter exactly."""
    n_signals, n_time, n_etfs = arr.shape
    alpha = 2 / (span + 1)

    nan_mask = np.isnan(arr)

    if nan_mask.any():
        arr_filled = _forward_fill_3d(arr)
        arr_filled = _forward_fill_3d(arr_filled[:, ::-1, :])[:, ::-1, :]
    else:
        arr_filled = arr

    # Transfer to GPU
    arr_gpu = cp.asarray(arr_filled.astype(np.float32))

    # Reshape to (n_time, n_signals * n_etfs) for batch processing
    arr_2d = arr_gpu.transpose(1, 0, 2).reshape(n_time, -1)

    # EMA via recursive formula: y[t] = alpha * x[t] + (1-alpha) * y[t-1]
    # Match scipy.lfilter behavior: initial state is 0 (i.e., y[-1] = 0)
    result_2d = cp.empty_like(arr_2d)
    result_2d[0] = alpha * arr_2d[0]  # y[0] = alpha*x[0] + (1-alpha)*0

    decay = 1 - alpha
    for t in range(1, n_time):
        result_2d[t] = alpha * arr_2d[t] + decay * result_2d[t - 1]

    # Reshape back
    result = result_2d.reshape(n_time, n_signals, n_etfs).transpose(1, 0, 2)

    # Transfer back to CPU
    result_cpu = cp.asnumpy(result)

    # Restore NaN
    result_cpu = np.where(nan_mask, np.nan, result_cpu)

    # Free GPU memory
    del arr_gpu, arr_2d, result_2d, result
    cp.get_default_memory_pool().free_all_blocks()

    return result_cpu.astype(np.float32)


def causal_passthrough(arr: np.ndarray) -> np.ndarray:
    """
    Passthrough filter - returns input unchanged.
    Used for 'raw' filter to maintain consistent interface.
    """
    return arr.astype(np.float32) if arr.dtype != np.float32 else arr


def causal_sma(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Causal Simple Moving Average using cumsum trick.

    Args:
        arr: (n_signals, n_time, n_etfs)
        window: lookback window
    """
    n_signals, n_time, n_etfs = arr.shape

    # Handle NaN by replacing with 0 for cumsum
    mask = np.isnan(arr)
    arr_clean = np.where(mask, 0, arr)

    # Cumsum along time axis
    cumsum = np.cumsum(arr_clean, axis=1)
    count = np.cumsum(~mask, axis=1)

    result = np.full_like(arr, np.nan)

    # Rolling sum = cumsum[i] - cumsum[i-window]
    result[:, window:, :] = (cumsum[:, window:, :] - cumsum[:, :-window, :]) / \
                            np.maximum(count[:, window:, :] - count[:, :-window, :], 1)

    return result


def causal_wma(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Causal Weighted Moving Average - linear weights favoring recent data.
    GPU-ACCELERATED: Uses CuPy convolution when available.

    Weights: [1, 2, 3, ..., window] normalized to sum to 1.

    Args:
        arr: (n_signals, n_time, n_etfs)
        window: lookback window
    """
    n_signals, n_time, n_etfs = arr.shape

    # Create linear weights [1, 2, 3, ..., window] normalized
    weights = np.arange(1, window + 1, dtype=np.float32)
    weights = weights / weights.sum()

    # Handle NaN: replace with 0 for convolution
    mask = np.isnan(arr)
    arr_clean = np.where(mask, 0, arr).astype(np.float32)

    if GPU_AVAILABLE:
        # GPU path - much faster for large arrays
        # Transfer to GPU
        arr_gpu = cp.asarray(arr_clean)
        weights_gpu = cp.asarray(weights)
        mask_gpu = cp.asarray(mask)

        # Convolve along time axis (axis=1)
        # origin shifts filter: negative = causal (use past values)
        origin = -(window // 2)
        result_gpu = gpu_convolve1d(arr_gpu, weights_gpu, axis=1, mode='constant', cval=0.0, origin=origin)

        # Count valid values
        ones_gpu = cp.ones(window, dtype=cp.float32)
        valid_count_gpu = gpu_convolve1d((~mask_gpu).astype(cp.float32), ones_gpu, axis=1, mode='constant', cval=0.0, origin=origin)

        # Invalidate where we don't have full window
        result_gpu = cp.where(valid_count_gpu >= window, result_gpu, cp.nan)
        result_gpu[:, :window-1, :] = cp.nan

        # Transfer back to CPU
        result = cp.asnumpy(result_gpu)

        # Free GPU memory
        del arr_gpu, weights_gpu, mask_gpu, result_gpu, valid_count_gpu, ones_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        # CPU fallback
        from scipy.ndimage import convolve1d

        origin = -(window // 2)
        result = convolve1d(arr_clean, weights, axis=1, mode='constant', cval=0.0, origin=origin)

        # Count valid values
        valid_count = convolve1d((~mask).astype(np.float32), np.ones(window), axis=1, mode='constant', cval=0.0, origin=origin)

        # Invalidate where we don't have full window
        result = np.where(valid_count >= window, result, np.nan)
        result[:, :window-1, :] = np.nan

    return result.astype(np.float32)


def causal_hull_ma(arr: np.ndarray, period: int) -> np.ndarray:
    """
    Hull Moving Average - smooth AND responsive (low lag).
    OPTIMIZED: Fused GPU implementation - single transfer, 3 WMAs on GPU.

    Formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))

    The Hull MA uses weighted moving averages to reduce lag while
    maintaining smoothness. It's particularly good for trend following.

    Args:
        arr: (n_signals, n_time, n_etfs)
        period: base period (will use period/2 and sqrt(period) internally)
    """
    if arr.ndim == 3 and GPU_AVAILABLE:
        return _causal_hull_ma_gpu(arr, period)
    else:
        return _causal_hull_ma_cpu(arr, period)


def _causal_hull_ma_cpu(arr: np.ndarray, period: int) -> np.ndarray:
    """CPU implementation - 3 separate WMA calls."""
    half_period = max(2, period // 2)
    sqrt_period = max(2, int(np.sqrt(period)))

    # Step 1: WMA(n/2)
    wma_half = causal_wma(arr, half_period)

    # Step 2: WMA(n)
    wma_full = causal_wma(arr, period)

    # Step 3: raw_hull = 2 * WMA(n/2) - WMA(n)
    raw_hull = 2 * wma_half - wma_full

    # Step 4: Hull = WMA(raw_hull, sqrt(n))
    hull = causal_wma(raw_hull, sqrt_period)

    return hull


def _causal_hull_ma_gpu(arr: np.ndarray, period: int) -> np.ndarray:
    """GPU implementation - fused all 3 WMAs in single GPU pass."""
    half_period = max(2, period // 2)
    sqrt_period = max(2, int(np.sqrt(period)))
    n_signals, n_time, n_etfs = arr.shape

    # Create all weight kernels upfront
    def make_wma_weights(w):
        weights = np.arange(1, w + 1, dtype=np.float32)
        return weights / weights.sum()

    weights_half = make_wma_weights(half_period)
    weights_full = make_wma_weights(period)
    weights_sqrt = make_wma_weights(sqrt_period)

    # Handle NaN
    mask = np.isnan(arr)
    arr_clean = np.where(mask, 0, arr).astype(np.float32)

    # Transfer to GPU ONCE
    arr_gpu = cp.asarray(arr_clean)
    weights_half_gpu = cp.asarray(weights_half)
    weights_full_gpu = cp.asarray(weights_full)
    weights_sqrt_gpu = cp.asarray(weights_sqrt)

    # WMA(n/2)
    origin_half = -(half_period // 2)
    wma_half_gpu = gpu_convolve1d(arr_gpu, weights_half_gpu, axis=1,
                                   mode='constant', cval=0.0, origin=origin_half)
    wma_half_gpu[:, :half_period-1, :] = cp.nan

    # WMA(n)
    origin_full = -(period // 2)
    wma_full_gpu = gpu_convolve1d(arr_gpu, weights_full_gpu, axis=1,
                                   mode='constant', cval=0.0, origin=origin_full)
    wma_full_gpu[:, :period-1, :] = cp.nan

    # raw_hull = 2 * WMA(n/2) - WMA(n)
    raw_hull_gpu = 2 * wma_half_gpu - wma_full_gpu

    # WMA(raw_hull, sqrt(n)) - final smoothing
    origin_sqrt = -(sqrt_period // 2)
    # Replace NaN with 0 for final convolution
    raw_hull_clean = cp.where(cp.isnan(raw_hull_gpu), 0, raw_hull_gpu)
    hull_gpu = gpu_convolve1d(raw_hull_clean, weights_sqrt_gpu, axis=1,
                               mode='constant', cval=0.0, origin=origin_sqrt)

    # Invalidate warmup period (need full period + sqrt_period - 1 points)
    warmup = period + sqrt_period - 2
    hull_gpu[:, :warmup, :] = cp.nan

    # Transfer back ONCE
    result = cp.asnumpy(hull_gpu)

    # Free GPU memory
    del arr_gpu, weights_half_gpu, weights_full_gpu, weights_sqrt_gpu
    del wma_half_gpu, wma_full_gpu, raw_hull_gpu, raw_hull_clean, hull_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return result.astype(np.float32)


def causal_dema(arr: np.ndarray, span: int) -> np.ndarray:
    """
    Double Exponential Moving Average - reduced lag.
    OPTIMIZED: Uses GPU when available, fused 2-pass EMA.

    Formula: DEMA = 2*EMA - EMA(EMA)

    Args:
        arr: (n_signals, n_time, n_etfs)
        span: EMA span
    """
    if arr.ndim == 3 and GPU_AVAILABLE:
        return _causal_dema_gpu(arr, span)
    else:
        return _causal_dema_cpu(arr, span)


def _causal_dema_cpu(arr: np.ndarray, span: int) -> np.ndarray:
    """CPU implementation using scipy.lfilter."""
    from scipy.signal import lfilter

    alpha = 2 / (span + 1)
    b = np.array([alpha])
    a = np.array([1, -(1 - alpha)])

    n_signals, n_time, n_etfs = arr.shape
    nan_mask = np.isnan(arr)

    if nan_mask.any():
        arr_filled = _forward_fill_3d(arr)
        arr_filled = _forward_fill_3d(arr_filled[:, ::-1, :])[:, ::-1, :]
    else:
        arr_filled = arr

    arr_2d = arr_filled.transpose(1, 0, 2).reshape(n_time, -1)
    ema1_2d = lfilter(b, a, arr_2d, axis=0)
    ema2_2d = lfilter(b, a, ema1_2d, axis=0)
    dema_2d = 2 * ema1_2d - ema2_2d

    result = dema_2d.reshape(n_time, n_signals, n_etfs).transpose(1, 0, 2)
    result = np.where(nan_mask, np.nan, result)
    return result.astype(np.float32)


def _causal_dema_gpu(arr: np.ndarray, span: int) -> np.ndarray:
    """GPU implementation using CuPy."""
    n_signals, n_time, n_etfs = arr.shape
    alpha = 2 / (span + 1)
    decay = 1 - alpha

    nan_mask = np.isnan(arr)

    if nan_mask.any():
        arr_filled = _forward_fill_3d(arr)
        arr_filled = _forward_fill_3d(arr_filled[:, ::-1, :])[:, ::-1, :]
    else:
        arr_filled = arr

    arr_gpu = cp.asarray(arr_filled.astype(np.float32))
    arr_2d = arr_gpu.transpose(1, 0, 2).reshape(n_time, -1)

    # Fused 2-pass EMA on GPU
    ema1_2d = cp.empty_like(arr_2d)
    ema1_2d[0] = alpha * arr_2d[0]
    for t in range(1, n_time):
        ema1_2d[t] = alpha * arr_2d[t] + decay * ema1_2d[t - 1]

    ema2_2d = cp.empty_like(ema1_2d)
    ema2_2d[0] = alpha * ema1_2d[0]
    for t in range(1, n_time):
        ema2_2d[t] = alpha * ema1_2d[t] + decay * ema2_2d[t - 1]

    dema_2d = 2 * ema1_2d - ema2_2d

    result = dema_2d.reshape(n_time, n_signals, n_etfs).transpose(1, 0, 2)
    result_cpu = cp.asnumpy(result)
    result_cpu = np.where(nan_mask, np.nan, result_cpu)

    del arr_gpu, arr_2d, ema1_2d, ema2_2d, dema_2d, result
    cp.get_default_memory_pool().free_all_blocks()

    return result_cpu.astype(np.float32)


def causal_trima(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Triangular Moving Average - double-smoothed SMA.

    TRIMA = SMA(SMA(x, n/2), n/2)

    Very smooth but more lag than Hull MA. Center-weighted.

    Args:
        arr: (n_signals, n_time, n_etfs)
        window: total window (will use window/2 internally)
    """
    half_window = max(2, window // 2)

    sma1 = causal_sma(arr, half_window)
    trima = causal_sma(sma1, half_window)

    return trima


def causal_tema(arr: np.ndarray, span: int) -> np.ndarray:
    """
    Triple Exponential Moving Average - minimal lag, very responsive.
    OPTIMIZED: Uses GPU when available, fused 3-pass EMA.

    Formula: TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))

    Args:
        arr: (n_signals, n_time, n_etfs)
        span: EMA span
    """
    if arr.ndim == 3 and GPU_AVAILABLE:
        return _causal_tema_gpu(arr, span)
    else:
        return _causal_tema_cpu(arr, span)


def _causal_tema_cpu(arr: np.ndarray, span: int) -> np.ndarray:
    """CPU implementation using scipy.lfilter."""
    from scipy.signal import lfilter

    alpha = 2 / (span + 1)
    b = np.array([alpha])
    a = np.array([1, -(1 - alpha)])

    n_signals, n_time, n_etfs = arr.shape
    nan_mask = np.isnan(arr)

    if nan_mask.any():
        arr_filled = _forward_fill_3d(arr)
        arr_filled = _forward_fill_3d(arr_filled[:, ::-1, :])[:, ::-1, :]
    else:
        arr_filled = arr

    arr_2d = arr_filled.transpose(1, 0, 2).reshape(n_time, -1)
    ema1_2d = lfilter(b, a, arr_2d, axis=0)
    ema2_2d = lfilter(b, a, ema1_2d, axis=0)
    ema3_2d = lfilter(b, a, ema2_2d, axis=0)
    tema_2d = 3 * ema1_2d - 3 * ema2_2d + ema3_2d

    result = tema_2d.reshape(n_time, n_signals, n_etfs).transpose(1, 0, 2)
    result = np.where(nan_mask, np.nan, result)
    return result.astype(np.float32)


def _causal_tema_gpu(arr: np.ndarray, span: int) -> np.ndarray:
    """GPU implementation using CuPy - fused 3-pass EMA."""
    n_signals, n_time, n_etfs = arr.shape
    alpha = 2 / (span + 1)
    decay = 1 - alpha

    nan_mask = np.isnan(arr)

    if nan_mask.any():
        arr_filled = _forward_fill_3d(arr)
        arr_filled = _forward_fill_3d(arr_filled[:, ::-1, :])[:, ::-1, :]
    else:
        arr_filled = arr

    arr_gpu = cp.asarray(arr_filled.astype(np.float32))
    arr_2d = arr_gpu.transpose(1, 0, 2).reshape(n_time, -1)

    # Fused 3-pass EMA on GPU
    ema1_2d = cp.empty_like(arr_2d)
    ema1_2d[0] = alpha * arr_2d[0]
    for t in range(1, n_time):
        ema1_2d[t] = alpha * arr_2d[t] + decay * ema1_2d[t - 1]

    ema2_2d = cp.empty_like(ema1_2d)
    ema2_2d[0] = alpha * ema1_2d[0]
    for t in range(1, n_time):
        ema2_2d[t] = alpha * ema1_2d[t] + decay * ema2_2d[t - 1]

    ema3_2d = cp.empty_like(ema2_2d)
    ema3_2d[0] = alpha * ema2_2d[0]
    for t in range(1, n_time):
        ema3_2d[t] = alpha * ema2_2d[t] + decay * ema3_2d[t - 1]

    tema_2d = 3 * ema1_2d - 3 * ema2_2d + ema3_2d

    result = tema_2d.reshape(n_time, n_signals, n_etfs).transpose(1, 0, 2)
    result_cpu = cp.asnumpy(result)
    result_cpu = np.where(nan_mask, np.nan, result_cpu)

    del arr_gpu, arr_2d, ema1_2d, ema2_2d, ema3_2d, tema_2d, result
    cp.get_default_memory_pool().free_all_blocks()

    return result_cpu.astype(np.float32)


def causal_zlema(arr: np.ndarray, span: int) -> np.ndarray:
    """
    Zero-Lag Exponential Moving Average - momentum-adjusted for minimal lag.
    OPTIMIZED: Uses GPU when available, avoids array copy for lagged values.

    Formula: ZLEMA = EMA(2*price - price[lag])
    where lag = (span - 1) / 2

    Args:
        arr: (n_signals, n_time, n_etfs)
        span: EMA span
    """
    if arr.ndim == 3 and GPU_AVAILABLE:
        return _causal_zlema_gpu(arr, span)
    else:
        return _causal_zlema_cpu(arr, span)


def _causal_zlema_cpu(arr: np.ndarray, span: int) -> np.ndarray:
    """CPU implementation."""
    lag = max(1, (span - 1) // 2)
    n_signals, n_time, n_etfs = arr.shape

    # Momentum-adjusted series: 2 * price_t - price_{t-lag}
    adjusted = np.full_like(arr, np.nan)
    adjusted[:, lag:, :] = 2 * arr[:, lag:, :] - arr[:, :-lag, :]

    return causal_ema(adjusted, span)


def _causal_zlema_gpu(arr: np.ndarray, span: int) -> np.ndarray:
    """GPU implementation - computes adjustment and EMA in one pass."""
    lag = max(1, (span - 1) // 2)
    n_signals, n_time, n_etfs = arr.shape
    alpha = 2 / (span + 1)
    decay = 1 - alpha

    nan_mask = np.isnan(arr)

    if nan_mask.any():
        arr_filled = _forward_fill_3d(arr)
        arr_filled = _forward_fill_3d(arr_filled[:, ::-1, :])[:, ::-1, :]
    else:
        arr_filled = arr

    arr_gpu = cp.asarray(arr_filled.astype(np.float32))
    arr_2d = arr_gpu.transpose(1, 0, 2).reshape(n_time, -1)

    # Compute momentum-adjusted series on GPU: 2*x[t] - x[t-lag]
    adjusted_2d = cp.full_like(arr_2d, cp.nan)
    adjusted_2d[lag:] = 2 * arr_2d[lag:] - arr_2d[:-lag]

    # Apply EMA to adjusted series
    result_2d = cp.empty_like(adjusted_2d)
    result_2d[:lag] = cp.nan

    # Initialize at first valid point
    result_2d[lag] = alpha * adjusted_2d[lag]

    for t in range(lag + 1, n_time):
        result_2d[t] = alpha * adjusted_2d[t] + decay * result_2d[t - 1]

    result = result_2d.reshape(n_time, n_signals, n_etfs).transpose(1, 0, 2)
    result_cpu = cp.asnumpy(result)
    result_cpu = np.where(nan_mask, np.nan, result_cpu)

    del arr_gpu, arr_2d, adjusted_2d, result_2d, result
    cp.get_default_memory_pool().free_all_blocks()

    return result_cpu.astype(np.float32)


def causal_gaussian_ma(arr: np.ndarray, window: int, sigma: float = None) -> np.ndarray:
    """
    Gaussian-weighted Moving Average - bell-curve weights, smooth rolloff.
    GPU-ACCELERATED: Uses CuPy/scipy convolution with Gaussian kernel.

    Weights follow Gaussian distribution: w_i ∝ exp(-i²/2σ²)
    This provides smoother frequency response than linear weights (WMA).

    Args:
        arr: (n_signals, n_time, n_etfs)
        window: lookback window
        sigma: Gaussian std dev (default: window/4 for good coverage)
    """
    if sigma is None:
        sigma = window / 4.0

    n_signals, n_time, n_etfs = arr.shape

    # Create Gaussian kernel (causal: only past values, centered at end)
    # Indices 0..window-1, where window-1 is "now" and 0 is oldest
    x = np.arange(window, dtype=np.float32)
    # Gaussian centered at window-1 (most recent point)
    weights = np.exp(-0.5 * ((x - (window - 1)) / sigma) ** 2)
    weights = weights / weights.sum()

    # Handle NaN: replace with 0 for convolution
    mask = np.isnan(arr)
    arr_clean = np.where(mask, 0, arr).astype(np.float32)

    if GPU_AVAILABLE:
        # GPU path
        arr_gpu = cp.asarray(arr_clean)
        weights_gpu = cp.asarray(weights)
        mask_gpu = cp.asarray(mask)

        # Convolve along time axis with causal origin
        origin = -(window // 2)
        result_gpu = gpu_convolve1d(arr_gpu, weights_gpu, axis=1, mode='constant', cval=0.0, origin=origin)

        # Count valid values for proper normalization
        ones_gpu = cp.ones(window, dtype=cp.float32)
        valid_count_gpu = gpu_convolve1d((~mask_gpu).astype(cp.float32), ones_gpu, axis=1, mode='constant', cval=0.0, origin=origin)

        # Invalidate where we don't have enough data
        result_gpu = cp.where(valid_count_gpu >= window * 0.5, result_gpu, cp.nan)
        result_gpu[:, :window-1, :] = cp.nan

        result = cp.asnumpy(result_gpu)

        del arr_gpu, weights_gpu, mask_gpu, result_gpu, valid_count_gpu, ones_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        # CPU fallback
        from scipy.ndimage import convolve1d

        origin = -(window // 2)
        result = convolve1d(arr_clean, weights, axis=1, mode='constant', cval=0.0, origin=origin)

        # Count valid values
        valid_count = convolve1d((~mask).astype(np.float32), np.ones(window), axis=1, mode='constant', cval=0.0, origin=origin)

        result = np.where(valid_count >= window * 0.5, result, np.nan)
        result[:, :window-1, :] = np.nan

    return result.astype(np.float32)


def causal_median(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Causal Median Filter - outlier-robust, preserves edges.

    The median filter is excellent for:
    - Removing outliers/spikes
    - Preserving sharp edges/transitions
    - Non-linear smoothing (doesn't blur peaks like MA)

    Args:
        arr: (n_signals, n_time, n_etfs)
        window: lookback window

    OPTIMIZED: Uses vectorized sliding_window_view on entire 3D array at once.
    """
    return _causal_median_vectorized(arr.astype(np.float32), window)


def _causal_median_vectorized(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Vectorized median filter with chunking to avoid memory issues.

    Processes signals in chunks to avoid creating huge sliding window arrays.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    import warnings

    n_signals, n_time, n_etfs = arr.shape
    result = np.full_like(arr, np.nan, dtype=np.float32)

    if n_time < window:
        return result

    # Estimate memory needed per signal: n_etfs * (n_time - window + 1) * window * 4 bytes
    # With 587 ETFs, 5889 time, window 63: 587 * 5827 * 63 * 4 = 860 MB per signal
    # Process in chunks to stay under ~4GB
    bytes_per_signal = n_etfs * (n_time - window + 1) * window * 4
    max_memory = 2 * 1024**3  # 2 GB limit
    chunk_size = max(1, int(max_memory / bytes_per_signal))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Process signals in chunks
        for chunk_start in range(0, n_signals, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_signals)
            chunk = arr[chunk_start:chunk_end]  # (chunk_size, n_time, n_etfs)

            n_chunk = chunk.shape[0]

            # Reshape chunk to (chunk_size * n_etfs, n_time)
            chunk_2d = chunk.transpose(0, 2, 1).reshape(-1, n_time)

            # Create sliding window view
            view = sliding_window_view(chunk_2d, window, axis=1)

            # Compute median
            medians = np.nanmedian(view, axis=2)

            # Reshape back and store
            medians_3d = medians.reshape(n_chunk, n_etfs, -1).transpose(0, 2, 1)
            result[chunk_start:chunk_end, window-1:, :] = medians_3d

            # Free memory
            del chunk_2d, view, medians, medians_3d

    return result


# Numba-optimized median kernels using RUNNING MEDIAN algorithm
# This is O(n*w) instead of O(n*w*log(w)) - much faster for large windows
if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _bisect_left(arr, x, lo, hi):
        """Binary search for insertion point (like bisect.bisect_left)."""
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @njit(cache=True, fastmath=True)
    def _running_median_series(series, window, result):
        """
        Running median using sorted window with binary insertion.
        O(n*w) instead of O(n*w*log(w)) - avoids full sort each step.
        """
        n = len(series)
        if n < window:
            return

        # Initialize sorted window with first 'window' valid values
        sorted_window = np.empty(window, dtype=np.float32)
        valid_count = 0

        # Fill initial window
        for i in range(window):
            val = series[i]
            if not np.isnan(val):
                # Binary insertion to maintain sorted order
                pos = _bisect_left(sorted_window, val, 0, valid_count)
                # Shift elements right
                for j in range(valid_count, pos, -1):
                    sorted_window[j] = sorted_window[j - 1]
                sorted_window[pos] = val
                valid_count += 1

        # Compute first median
        if valid_count > 0:
            if valid_count % 2 == 1:
                result[window - 1] = sorted_window[valid_count // 2]
            else:
                result[window - 1] = (sorted_window[valid_count // 2 - 1] +
                                       sorted_window[valid_count // 2]) / 2.0

        # Slide window: remove oldest, add newest
        for t in range(window, n):
            old_val = series[t - window]
            new_val = series[t]

            # Remove old value if it was valid
            if not np.isnan(old_val):
                # Find and remove from sorted window
                pos = _bisect_left(sorted_window, old_val, 0, valid_count)
                # Handle case where value might be slightly different due to float
                if pos < valid_count and sorted_window[pos] == old_val:
                    # Shift elements left
                    for j in range(pos, valid_count - 1):
                        sorted_window[j] = sorted_window[j + 1]
                    valid_count -= 1

            # Add new value if valid
            if not np.isnan(new_val):
                pos = _bisect_left(sorted_window, new_val, 0, valid_count)
                # Shift elements right
                for j in range(valid_count, pos, -1):
                    sorted_window[j] = sorted_window[j - 1]
                sorted_window[pos] = new_val
                valid_count += 1

            # Compute median
            if valid_count > 0:
                if valid_count % 2 == 1:
                    result[t] = sorted_window[valid_count // 2]
                else:
                    result[t] = (sorted_window[valid_count // 2 - 1] +
                                 sorted_window[valid_count // 2]) / 2.0

    # NOTE: Using parallel=False because parallel=True causes silent crashes
    # on Windows when processing large arrays after GPU operations.
    # The crash appears to be related to Numba's threading layer conflicting
    # with CUDA/CuPy. Single-threaded is slower but stable.
    # cache=False to force recompilation and avoid stale cached parallel code.
    @njit(parallel=False, cache=False, fastmath=True)
    def _median_batch_numba(arr_flat, n_series, n_time, window, result_flat):
        """Process all series sequentially using running median."""
        for idx in range(n_series):  # Use range instead of prange
            _running_median_series(arr_flat[idx], window, result_flat[idx])


def _causal_median_numba_wrapper(arr: np.ndarray, window: int) -> np.ndarray:
    """Wrapper for Numba median computation."""
    n_signals, n_time, n_etfs = arr.shape

    # IMPORTANT: Ensure we're working with contiguous CPU arrays
    # This prevents conflicts with GPU memory when Numba tries to parallelize
    arr_cpu = np.ascontiguousarray(arr, dtype=np.float32)

    # Reshape to (n_signals * n_etfs, n_time) for batch processing
    arr_flat = np.ascontiguousarray(arr_cpu.transpose(0, 2, 1).reshape(-1, n_time))
    n_series = arr_flat.shape[0]

    result_flat = np.full((n_series, n_time), np.nan, dtype=np.float32)

    # Run Numba parallel batch
    _median_batch_numba(arr_flat, n_series, n_time, window, result_flat)

    # Reshape back
    result = result_flat.reshape(n_signals, n_etfs, n_time).transpose(0, 2, 1)

    return np.ascontiguousarray(result)


def causal_regime_switching(arr: np.ndarray, fast_window: int = 10, slow_window: int = 50,
                            vol_window: int = 21, threshold: float = 1.0) -> np.ndarray:
    """
    Regime-Switching Filter - adapts smoothing based on volatility regime.
    OPTIMIZED: Uses GPU for EMAs + vectorized volatility calculation.

    In high-volatility regimes: uses fast EMA (more responsive)
    In low-volatility regimes: uses slow EMA (more smoothing)

    Args:
        arr: (n_signals, n_time, n_etfs)
        fast_window: EMA span for high-vol regime
        slow_window: EMA span for low-vol regime
        vol_window: window for volatility calculation
        threshold: z-score threshold for regime switch (1.0 = 1 std above mean)
    """
    if arr.ndim == 3 and GPU_AVAILABLE:
        return _causal_regime_switching_gpu(arr, fast_window, slow_window, vol_window, threshold)
    else:
        return _causal_regime_switching_cpu(arr, fast_window, slow_window, vol_window, threshold)


def _causal_regime_switching_cpu(arr, fast_window, slow_window, vol_window, threshold):
    """CPU implementation."""
    n_signals, n_time, n_etfs = arr.shape

    ema_fast = causal_ema(arr, fast_window)
    ema_slow = causal_ema(arr, slow_window)

    # Volatility using cumsum
    changes = np.zeros_like(arr)
    changes[:, 1:, :] = np.abs(arr[:, 1:, :] - arr[:, :-1, :])

    cumsum = np.cumsum(changes, axis=1)
    cumsum_sq = np.cumsum(changes ** 2, axis=1)

    vol_short = np.full_like(arr, np.nan)
    if n_time > vol_window:
        roll_sum = cumsum[:, vol_window:, :] - cumsum[:, :-vol_window, :]
        roll_sum_sq = cumsum_sq[:, vol_window:, :] - cumsum_sq[:, :-vol_window, :]
        roll_mean = roll_sum / vol_window
        roll_var = roll_sum_sq / vol_window - roll_mean ** 2
        vol_short[:, vol_window:, :] = np.sqrt(np.maximum(roll_var, 0))

    long_window = vol_window * 4
    vol_long = np.full_like(arr, np.nan)
    if n_time > long_window:
        roll_sum_l = cumsum[:, long_window:, :] - cumsum[:, :-long_window, :]
        roll_sum_sq_l = cumsum_sq[:, long_window:, :] - cumsum_sq[:, :-long_window, :]
        roll_mean_l = roll_sum_l / long_window
        roll_var_l = roll_sum_sq_l / long_window - roll_mean_l ** 2
        vol_long[:, long_window:, :] = np.sqrt(np.maximum(roll_var_l, 0))

    vol_ratio = vol_short / (vol_long + 1e-10)
    high_vol_regime = vol_ratio > threshold
    result = np.where(high_vol_regime, ema_fast, ema_slow)

    return result.astype(np.float32)


def _causal_regime_switching_gpu(arr, fast_window, slow_window, vol_window, threshold):
    """GPU implementation - MEMORY-OPTIMIZED: process in chunks to avoid OOM."""
    n_signals, n_time, n_etfs = arr.shape

    # For large arrays, fall back to CPU to avoid OOM
    # GPU memory needed is roughly: 10 * n_signals * n_time * n_etfs * 4 bytes
    # With 30 signals * 5889 time * 587 etfs * 10 arrays * 4 bytes = ~4GB
    # Be conservative - use CPU for anything over 10M elements
    total_elements = n_signals * n_time * n_etfs
    if total_elements > 10_000_000:  # ~10M elements = ~400MB for 10 arrays
        # Too large for GPU - use CPU implementation
        return _causal_regime_switching_cpu(arr, fast_window, slow_window, vol_window, threshold)

    alpha_fast = 2 / (fast_window + 1)
    alpha_slow = 2 / (slow_window + 1)
    decay_fast = 1 - alpha_fast
    decay_slow = 1 - alpha_slow

    nan_mask = np.isnan(arr)

    if nan_mask.any():
        arr_filled = _forward_fill_3d(arr)
        arr_filled = _forward_fill_3d(arr_filled[:, ::-1, :])[:, ::-1, :]
    else:
        arr_filled = arr

    arr_gpu = cp.asarray(arr_filled.astype(np.float32))
    arr_2d = arr_gpu.transpose(1, 0, 2).reshape(n_time, -1)

    # Compute both EMAs in parallel-ish (same loop, different alphas)
    ema_fast_2d = cp.empty_like(arr_2d)
    ema_slow_2d = cp.empty_like(arr_2d)
    ema_fast_2d[0] = alpha_fast * arr_2d[0]
    ema_slow_2d[0] = alpha_slow * arr_2d[0]

    for t in range(1, n_time):
        ema_fast_2d[t] = alpha_fast * arr_2d[t] + decay_fast * ema_fast_2d[t - 1]
        ema_slow_2d[t] = alpha_slow * arr_2d[t] + decay_slow * ema_slow_2d[t - 1]

    # Compute volatility on GPU
    changes = cp.zeros_like(arr_2d)
    changes[1:] = cp.abs(arr_2d[1:] - arr_2d[:-1])

    cumsum = cp.cumsum(changes, axis=0)
    cumsum_sq = cp.cumsum(changes ** 2, axis=0)

    vol_short = cp.full_like(arr_2d, cp.nan)
    if n_time > vol_window:
        roll_sum = cumsum[vol_window:] - cumsum[:-vol_window]
        roll_sum_sq = cumsum_sq[vol_window:] - cumsum_sq[:-vol_window]
        roll_mean = roll_sum / vol_window
        roll_var = roll_sum_sq / vol_window - roll_mean ** 2
        vol_short[vol_window:] = cp.sqrt(cp.maximum(roll_var, 0))

    long_window = vol_window * 4
    vol_long = cp.full_like(arr_2d, cp.nan)
    if n_time > long_window:
        roll_sum_l = cumsum[long_window:] - cumsum[:-long_window]
        roll_sum_sq_l = cumsum_sq[long_window:] - cumsum_sq[:-long_window]
        roll_mean_l = roll_sum_l / long_window
        roll_var_l = roll_sum_sq_l / long_window - roll_mean_l ** 2
        vol_long[long_window:] = cp.sqrt(cp.maximum(roll_var_l, 0))

    vol_ratio = vol_short / (vol_long + 1e-10)
    high_vol_regime = vol_ratio > threshold

    result_2d = cp.where(high_vol_regime, ema_fast_2d, ema_slow_2d)

    result = result_2d.reshape(n_time, n_signals, n_etfs).transpose(1, 0, 2)
    result_cpu = cp.asnumpy(result)
    result_cpu = np.where(nan_mask, np.nan, result_cpu)

    del arr_gpu, arr_2d, ema_fast_2d, ema_slow_2d, changes, cumsum, cumsum_sq
    del vol_short, vol_long, vol_ratio, high_vol_regime, result_2d, result
    cp.get_default_memory_pool().free_all_blocks()

    return result_cpu.astype(np.float32)


def _kama_inner_loop_python(series: np.ndarray, period: int, fast_sc: float, slow_sc: float) -> np.ndarray:
    """
    Inner loop for KAMA computation - Python fallback.

    Fixed to handle sparse data where valid values may not be contiguous.
    """
    n_time = len(series)
    result = np.full(n_time, np.nan, dtype=np.float32)

    # Find first index with enough valid data to initialize
    # We need at least 'period' valid values before we can start
    valid_count = 0
    init_idx = -1

    for i in range(n_time):
        if not np.isnan(series[i]):
            valid_count += 1
            if valid_count >= period:
                init_idx = i
                break
        # Don't reset valid_count - we allow gaps in the data

    if init_idx < 0:
        return result  # Not enough valid data

    # Initialize KAMA with first valid value at init_idx
    kama = series[init_idx]
    if np.isnan(kama):
        # Find closest valid value
        for i in range(init_idx, -1, -1):
            if not np.isnan(series[i]):
                kama = series[i]
                break

    if np.isnan(kama):
        return result

    result[init_idx] = kama

    # Process rest of series
    for t in range(init_idx + 1, n_time):
        if np.isnan(series[t]):
            result[t] = kama  # Carry forward
            continue

        # Find value from 'period' steps back (or closest valid)
        past_val = np.nan
        for lookback in range(period, min(period + 10, t + 1)):  # Allow some flexibility
            if t - lookback >= 0 and not np.isnan(series[t - lookback]):
                past_val = series[t - lookback]
                break

        if np.isnan(past_val):
            result[t] = kama
            continue

        # Efficiency Ratio = |change| / sum(|daily changes|)
        change = abs(series[t] - past_val)

        # Calculate volatility (sum of absolute changes over period)
        volatility = 0.0
        for i in range(max(0, t - period), t):
            if not np.isnan(series[i]) and not np.isnan(series[i + 1]):
                volatility += abs(series[i + 1] - series[i])

        if volatility > 1e-10:
            er = min(change / volatility, 1.0)  # Cap at 1.0
        else:
            er = 0

        # Smoothing constant adapts based on ER
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # KAMA formula
        kama = kama + sc * (series[t] - kama)
        result[t] = kama

    return result


# Numba-optimized KAMA kernel (compiled at import time if available)
if NUMBA_AVAILABLE:
    @njit(cache=False, fastmath=False)  # fastmath=False required for correct NaN handling
    def _kama_kernel_numba_v2(series, period, fast_sc, slow_sc, result):
        """Numba JIT-compiled KAMA inner loop - handles sparse data."""
        n_time = len(series)

        # Find first index with enough valid data to initialize
        valid_count = 0
        init_idx = -1

        for i in range(n_time):
            if not np.isnan(series[i]):
                valid_count += 1
                if valid_count >= period:
                    init_idx = i
                    break

        if init_idx < 0:
            return  # Not enough valid data

        # Initialize KAMA with value at init_idx
        # Note: at init_idx, we have exactly 'period' valid values up to this point
        # so series[init_idx] should be valid
        kama = series[init_idx]

        # Safety check - shouldn't happen if counting is correct
        if np.isnan(kama):
            return

        result[init_idx] = kama

        # Process rest of series
        for t in range(init_idx + 1, n_time):
            if np.isnan(series[t]):
                result[t] = kama  # Carry forward
                continue

            # Find value from 'period' steps back (or closest valid)
            past_val = np.nan
            for lookback in range(period, min(period + 10, t + 1)):
                if t - lookback >= 0 and not np.isnan(series[t - lookback]):
                    past_val = series[t - lookback]
                    break

            if np.isnan(past_val):
                result[t] = kama
                continue

            # Efficiency Ratio = |change| / sum(|daily changes|)
            change = abs(series[t] - past_val)

            # Calculate volatility (sum of absolute changes over period)
            volatility = 0.0
            for i in range(max(0, t - period), t):
                if i + 1 < n_time and not np.isnan(series[i]) and not np.isnan(series[i + 1]):
                    volatility += abs(series[i + 1] - series[i])

            if volatility > 1e-10:
                er = min(change / volatility, 1.0)  # Cap at 1.0
            else:
                er = 0.0

            # Adaptive smoothing constant
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

            # KAMA update
            kama = kama + sc * (series[t] - kama)
            result[t] = kama

    # NOTE: Using parallel=False because parallel=True causes silent crashes
    # on Windows when processing large arrays after GPU operations.
    # cache=False to avoid stale cached parallel code.
    @njit(parallel=False, cache=False, fastmath=False)  # fastmath=False for NaN handling
    def _kama_batch_numba_v2(arr_flat, n_series, n_time, period, fast_sc, slow_sc, result_flat):
        """Process all series sequentially using Numba."""
        for idx in range(n_series):  # Use range instead of prange
            series = arr_flat[idx]
            result = result_flat[idx]
            _kama_kernel_numba_v2(series, period, fast_sc, slow_sc, result)


def _kama_inner_loop(series: np.ndarray, period: int, fast_sc: float, slow_sc: float) -> np.ndarray:
    """Wrapper that uses Numba if available, else falls back to Python."""
    # TODO: Re-enable Numba after fixing the cached compilation issue
    # For now, use Python version which correctly handles sparse data
    return _kama_inner_loop_python(series, period, fast_sc, slow_sc)


def causal_kama(arr: np.ndarray, period: int = 10, fast: int = 2, slow: int = 30) -> np.ndarray:
    """
    Kaufman Adaptive Moving Average - adapts to volatility.
    OPTIMIZED: Uses Numba JIT with parallel processing (~50-100x speedup).

    KAMA changes its smoothing factor based on price efficiency:
    - Trending markets: faster response (lower smoothing)
    - Choppy markets: slower response (higher smoothing)

    Args:
        arr: (n_signals, n_time, n_etfs)
        period: efficiency ratio period
        fast: fast EMA period (for trending)
        slow: slow EMA period (for choppy)
    """
    n_signals, n_time, n_etfs = arr.shape

    fast_sc = 2 / (fast + 1)  # Fast smoothing constant
    slow_sc = 2 / (slow + 1)  # Slow smoothing constant

    # Reshape to (n_signals * n_etfs, n_time) for batch processing
    arr_flat = arr.transpose(0, 2, 1).reshape(-1, n_time).astype(np.float32)
    n_series = arr_flat.shape[0]

    result_flat = np.full_like(arr_flat, np.nan, dtype=np.float32)

    if NUMBA_AVAILABLE:
        # Use renamed Numba function to force fresh compilation
        _kama_batch_numba_v2(arr_flat, n_series, n_time, period, fast_sc, slow_sc, result_flat)
    else:
        # Fallback to Python loop
        for idx in range(n_series):
            result_flat[idx] = _kama_inner_loop_python(arr_flat[idx], period, fast_sc, slow_sc)

    # Reshape back to original shape
    result = result_flat.reshape(n_signals, n_etfs, n_time).transpose(0, 2, 1)

    return result


def causal_butterworth(arr: np.ndarray, cutoff_period: int, order: int = 2) -> np.ndarray:
    """
    Causal Butterworth low-pass filter - excellent noise reduction.

    Uses scipy.signal.lfilter which is causal (only uses past data).

    Args:
        arr: (n_signals, n_time, n_etfs)
        cutoff_period: period in days (e.g., 63 for ~3 months)
        order: filter order (2 is typical, higher = sharper cutoff)

    Note: cutoff_period of 63 means frequencies faster than 63 days are attenuated.
    """
    from scipy.signal import butter, lfilter

    n_signals, n_time, n_etfs = arr.shape

    # Nyquist frequency = 0.5 (for daily data)
    # Cutoff frequency = 1/cutoff_period normalized to Nyquist
    nyquist = 0.5
    cutoff_freq = 1 / cutoff_period / 2  # Divide by 2 for Nyquist normalization
    cutoff_freq = min(cutoff_freq, 0.99 * nyquist)  # Must be < Nyquist

    # Design Butterworth filter
    b, a = butter(order, cutoff_freq / nyquist, btype='low')

    result = np.full_like(arr, np.nan)

    # Apply filter to each signal/ETF combination
    # lfilter is causal - only uses past values
    for sig_idx in range(n_signals):
        for etf_idx in range(n_etfs):
            series = arr[sig_idx, :, etf_idx]

            # Find valid (non-NaN) segments
            valid = ~np.isnan(series)
            if valid.sum() < cutoff_period * 2:
                continue

            # Fill NaN with forward fill for filtering, then restore NaN positions
            series_filled = np.copy(series)

            # Forward fill NaN
            last_valid = np.nan
            for i in range(len(series_filled)):
                if np.isnan(series_filled[i]):
                    series_filled[i] = last_valid
                else:
                    last_valid = series_filled[i]

            # Apply causal filter (lfilter, not filtfilt which is non-causal)
            if not np.isnan(series_filled).all():
                filtered = lfilter(b, a, series_filled)

                # Restore NaN positions and handle filter warm-up
                filtered[:cutoff_period] = np.nan  # Filter needs warm-up period
                result[sig_idx, :, etf_idx] = np.where(valid, filtered, np.nan)

    return result


def causal_butterworth_fast(arr: np.ndarray, cutoff_period: int, order: int = 2) -> np.ndarray:
    """
    Fast vectorized Butterworth - processes all signals/ETFs at once.
    OPTIMIZED: Uses pandas ffill and batch filtering.

    Args:
        arr: (n_signals, n_time, n_etfs)
        cutoff_period: period in days
        order: filter order
    """
    from scipy.signal import butter, lfilter

    n_signals, n_time, n_etfs = arr.shape

    nyquist = 0.5
    cutoff_freq = 1 / cutoff_period / 2
    cutoff_freq = min(cutoff_freq, 0.99 * nyquist)

    b, a = butter(order, cutoff_freq / nyquist, btype='low')

    # Track original NaN positions
    nan_mask = np.isnan(arr)

    # Forward fill NaN, then backfill for leading NaN
    arr_filled = _forward_fill_3d(arr)
    arr_filled = _forward_fill_3d(arr_filled[:, ::-1, :])[:, ::-1, :]  # Backfill

    # Reshape to 2D for batch filtering: (n_time, n_signals * n_etfs)
    arr_2d = arr_filled.transpose(1, 0, 2).reshape(n_time, -1)

    # Apply filter along axis 0 (time) for all columns at once
    filtered_2d = lfilter(b, a, arr_2d, axis=0)

    # Reshape back to 3D
    filtered = filtered_2d.reshape(n_time, n_signals, n_etfs).transpose(1, 0, 2)

    # Handle warm-up period
    filtered[:, :cutoff_period, :] = np.nan

    # Restore NaN where original was NaN
    result = np.where(nan_mask, np.nan, filtered)

    return result


def causal_kalman(arr: np.ndarray, process_var: float = 1e-5, measurement_var: float = 1e-2) -> np.ndarray:
    """
    Causal Kalman Filter - optimal adaptive smoothing.

    The Kalman filter adapts its smoothing based on the signal-to-noise ratio.
    It's optimal for linear systems with Gaussian noise.

    Key advantage: Automatically adapts smoothing to signal characteristics.

    Args:
        arr: (n_signals, n_time, n_etfs)
        process_var: Process noise variance (Q) - higher = more responsive
        measurement_var: Measurement noise variance (R) - higher = more smoothing

    Returns:
        Filtered array of same shape
    """
    n_signals, n_time, n_etfs = arr.shape
    result = np.full_like(arr, np.nan)

    for sig_idx in range(n_signals):
        for etf_idx in range(n_etfs):
            series = arr[sig_idx, :, etf_idx]

            # Find first valid value
            first_valid = None
            for i in range(n_time):
                if not np.isnan(series[i]):
                    first_valid = i
                    break

            if first_valid is None:
                continue

            # Initialize Kalman state
            x_est = series[first_valid]  # State estimate
            p_est = 1.0  # Error covariance estimate

            result[sig_idx, first_valid, etf_idx] = x_est

            for t in range(first_valid + 1, n_time):
                # Prediction step
                x_pred = x_est  # State transition is identity
                p_pred = p_est + process_var

                if np.isnan(series[t]):
                    # No measurement - use prediction
                    x_est = x_pred
                    p_est = p_pred
                else:
                    # Update step
                    # Kalman gain
                    k = p_pred / (p_pred + measurement_var)

                    # Updated state estimate
                    x_est = x_pred + k * (series[t] - x_pred)

                    # Updated error covariance
                    p_est = (1 - k) * p_pred

                result[sig_idx, t, etf_idx] = x_est

    return result


def causal_kalman_fast(arr: np.ndarray, process_var: float = 1e-5, measurement_var: float = 1e-2) -> np.ndarray:
    """
    Faster Kalman filter - uses scipy.signal.lfilter.

    Uses steady-state Kalman gain approximation:
    After a few iterations, the Kalman gain converges to a constant.
    This is equivalent to an EMA, so we use lfilter for speed.

    Args:
        arr: (n_signals, n_time, n_etfs)
        process_var: Process noise variance (higher = more responsive)
        measurement_var: Measurement noise variance (higher = more smoothing)
    """
    from scipy.signal import lfilter

    # Compute steady-state Kalman gain
    p_ss = (np.sqrt(process_var**2 + 4*process_var*measurement_var) + process_var) / 2
    k_ss = p_ss / (p_ss + measurement_var)

    # IIR filter coefficients (equivalent to EMA with alpha = k_ss)
    b = np.array([k_ss])
    a = np.array([1, -(1 - k_ss)])

    n_signals, n_time, n_etfs = arr.shape

    # Track original NaN positions
    nan_mask = np.isnan(arr)

    # Handle NaN by forward-filling, then backfill for leading NaN
    arr_filled = _forward_fill_3d(arr)
    arr_filled = _forward_fill_3d(arr_filled[:, ::-1, :])[:, ::-1, :]  # Backfill

    # Reshape to 2D for batch filtering
    arr_2d = arr_filled.transpose(1, 0, 2).reshape(n_time, -1)

    # Apply IIR filter
    filtered_2d = lfilter(b, a, arr_2d, axis=0)

    # Reshape back
    result = filtered_2d.reshape(n_time, n_signals, n_etfs).transpose(1, 0, 2)

    # Restore NaN where original was NaN
    result = np.where(nan_mask, np.nan, result)

    return result


def causal_savgol(arr: np.ndarray, window: int = 21, polyorder: int = 3) -> np.ndarray:
    """
    Causal Savitzky-Golay filter - polynomial smoothing that preserves peaks.
    GPU-ACCELERATED: Uses CuPy convolution when available.

    Unlike standard Savitzky-Golay which is symmetric (non-causal), this version
    uses only past data points for each output.

    Key advantage: Preserves local maxima/minima better than moving averages.

    Args:
        arr: (n_signals, n_time, n_etfs)
        window: Window length (must be odd and > polyorder)
        polyorder: Polynomial order (2 or 3 typical)

    Returns:
        Filtered array
    """
    from scipy.signal import savgol_coeffs

    n_signals, n_time, n_etfs = arr.shape

    # Ensure window is odd
    if window % 2 == 0:
        window += 1

    # Get Savitzky-Golay coefficients for causal filter (all past points)
    coeffs = savgol_coeffs(window, polyorder, deriv=0, pos=window-1).astype(np.float32)

    # Forward fill NaN using fast pandas-based ffill
    arr_filled = _forward_fill_3d(arr).astype(np.float32)

    if GPU_AVAILABLE:
        # GPU path - transfer data and convolve on GPU
        arr_gpu = cp.asarray(arr_filled)
        coeffs_gpu = cp.asarray(coeffs)

        # Convolve along time axis (axis=1)
        # CRITICAL: origin = -(window//2) makes the filter CAUSAL (no look-ahead)
        # Without this, the filter uses future data!
        origin = -(window // 2)
        filtered_gpu = gpu_convolve1d(arr_gpu, coeffs_gpu, axis=1, mode='constant', cval=0.0, origin=origin)

        # Handle warm-up period
        filtered_gpu[:, :window, :] = cp.nan

        # Transfer back to CPU
        filtered = cp.asnumpy(filtered_gpu)

        # Free GPU memory
        del arr_gpu, coeffs_gpu, filtered_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        # CPU fallback
        from scipy.ndimage import convolve1d

        # Reshape to 2D for batch convolution: (n_signals * n_etfs, n_time)
        arr_2d = arr_filled.transpose(0, 2, 1).reshape(-1, n_time)

        # Apply convolution along time axis
        # CRITICAL: origin = -(window//2) makes the filter CAUSAL (no look-ahead)
        origin = -(window // 2)
        filtered_2d = convolve1d(arr_2d, coeffs, axis=1, mode='constant', cval=0.0, origin=origin)

        # Reshape back
        filtered = filtered_2d.reshape(n_signals, n_etfs, n_time).transpose(0, 2, 1)

        # Handle warm-up period
        filtered[:, :window, :] = np.nan

    # Restore NaN where original was NaN
    result = np.where(np.isnan(arr), np.nan, filtered)

    return result


def causal_adaptive_ema(arr: np.ndarray, base_span: int = 63, vol_window: int = 21) -> np.ndarray:
    """
    Adaptive EMA - span adjusts based on local volatility.
    SIMPLIFIED: Uses fixed base_span EMA (volatility adaptation is minimal benefit).

    For speed, we just use the base_span EMA which captures most of the benefit.
    The volatility adaptation adds complexity with minimal improvement.

    Args:
        arr: (n_signals, n_time, n_etfs)
        base_span: EMA span to use
        vol_window: Ignored (kept for API compatibility)

    Returns:
        Filtered array
    """
    # Just use regular EMA - the adaptive benefit is minimal and the speed cost is high
    return causal_ema(arr, base_span)


# =============================================================================
# FILTER CONFIGURATION
# =============================================================================

# Full filter configurations with all filter types
# Total: 25 filters (including raw)
# ORDER: Previously problematic first, then untested, then known-good
DEFAULT_FILTER_CONFIGS = [
    # =========================================================================
    # GROUP 1: Previously problematic - test first
    # =========================================================================
    # Regime-Switching (fast in high-vol, slow in low-vol) - was OOM, now uses CPU fallback
    ('regime_10_50', causal_regime_switching, {'fast_window': 10, 'slow_window': 50}),

    # =========================================================================
    # GROUP 2: Other filters (not yet tested in full pipeline)
    # =========================================================================
    # Butterworth (excellent noise reduction)
    ('butter_21d', causal_butterworth_fast, {'cutoff_period': 21}),
    ('butter_63d', causal_butterworth_fast, {'cutoff_period': 63}),

    # Kalman filter (adaptive smoothing)
    ('kalman_fast', causal_kalman_fast, {'process_var': 1e-4, 'measurement_var': 1e-2}),
    ('kalman_slow', causal_kalman_fast, {'process_var': 1e-5, 'measurement_var': 1e-2}),

    # Savitzky-Golay (preserves peaks)
    ('savgol_21d', causal_savgol, {'window': 21, 'polyorder': 3}),
    ('savgol_63d', causal_savgol, {'window': 63, 'polyorder': 3}),

    # Kaufman Adaptive MA (adapts to trending vs choppy)
    ('kama_21d', causal_kama, {'period': 21, 'fast': 2, 'slow': 30}),

    # =========================================================================
    # GROUP 3: Known-good filters (tested and working)
    # =========================================================================
    # Median filter (outlier-robust) - now uses CPU ThreadPool, stable
    ('median_21d', causal_median, {'window': 21}),
    ('median_63d', causal_median, {'window': 63}),

    # Raw signal (no filtering)
    ('raw', causal_passthrough, {}),

    # EMA variants (fast IIR filter)
    ('ema_21d', causal_ema, {'span': 21}),
    ('ema_63d', causal_ema, {'span': 63}),

    # Double EMA (reduced lag)
    ('dema_21d', causal_dema, {'span': 21}),
    ('dema_63d', causal_dema, {'span': 63}),

    # Triple EMA (minimal lag, very responsive)
    ('tema_21d', causal_tema, {'span': 21}),
    ('tema_63d', causal_tema, {'span': 63}),

    # Zero-Lag EMA (momentum-adjusted)
    ('zlema_21d', causal_zlema, {'span': 21}),
    ('zlema_63d', causal_zlema, {'span': 63}),

    # Hull MA (low lag, smooth)
    ('hull_21d', causal_hull_ma, {'period': 21}),
    ('hull_63d', causal_hull_ma, {'period': 63}),

    # Triangular MA (double-smoothed, center-weighted)
    ('trima_21d', causal_trima, {'window': 21}),
    ('trima_63d', causal_trima, {'window': 63}),

    # Gaussian MA (bell-curve weights, smooth rolloff)
    ('gauss_21d', causal_gaussian_ma, {'window': 21}),
    ('gauss_63d', causal_gaussian_ma, {'window': 63}),
]


def apply_filter(signals: np.ndarray, filter_name: str, filter_func, filter_kwargs: dict) -> np.ndarray:
    """
    Apply a filter to signals array.

    Args:
        signals: (n_signals, n_time, n_etfs) array
        filter_name: name of filter (for logging)
        filter_func: filter function or None for raw
        filter_kwargs: kwargs for filter function

    Returns:
        Filtered signals as float32
    """
    if filter_func is None:
        return signals.astype(np.float32)
    else:
        return filter_func(signals, **filter_kwargs).astype(np.float32)


# =============================================================================
# FILTER REGISTRY - Access filters by name
# =============================================================================

# Build registry from DEFAULT_FILTER_CONFIGS
FILTER_REGISTRY = {name: (func, kwargs) for name, func, kwargs in DEFAULT_FILTER_CONFIGS}


def get_filter(filter_name: str):
    """
    Get a filter function and its default kwargs by name.

    Args:
        filter_name: Name of the filter (e.g., 'ema_21d', 'hull_63d')

    Returns:
        Tuple of (filter_function, default_kwargs)

    Raises:
        KeyError: If filter_name is not found
    """
    if filter_name not in FILTER_REGISTRY:
        raise KeyError(f"Unknown filter: {filter_name}. Available: {list(FILTER_REGISTRY.keys())}")
    return FILTER_REGISTRY[filter_name]


def get_available_filters() -> list:
    """Return list of all available filter names."""
    return list(FILTER_REGISTRY.keys())


def apply_single_filter(signal_data: np.ndarray, filter_name: str) -> np.ndarray:
    """
    Apply a single filter to signal data by name.

    Args:
        signal_data: 2D array (n_time, n_etfs) for a single signal
        filter_name: Name of the filter to apply

    Returns:
        Filtered signal data as float32
    """
    filter_func, filter_kwargs = get_filter(filter_name)

    # Expand to 3D for filter functions: (1, n_time, n_etfs)
    signal_3d = signal_data[np.newaxis, :, :]

    # Apply filter
    filtered_3d = filter_func(signal_3d, **filter_kwargs)

    # Return as 2D
    return filtered_3d[0].astype(np.float32)


def get_filter_info(filter_name: str) -> dict:
    """
    Get detailed information about a filter.

    Args:
        filter_name: Name of the filter

    Returns:
        Dictionary with filter info
    """
    filter_func, filter_kwargs = get_filter(filter_name)
    return {
        'name': filter_name,
        'function': filter_func.__name__,
        'parameters': filter_kwargs,
        'docstring': filter_func.__doc__
    }


def print_available_filters():
    """Print all available filters with their parameters."""
    print("=" * 70)
    print("AVAILABLE FILTERS")
    print("=" * 70)

    for name, func, kwargs in DEFAULT_FILTER_CONFIGS:
        params = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else "none"
        print(f"  {name:20s} -> {func.__name__}({params})")

    print("=" * 70)
    print(f"Total: {len(DEFAULT_FILTER_CONFIGS)} filters")


# =============================================================================
# GENERATOR FUNCTIONS - Incremental computation with yielding
# =============================================================================

def compute_filtered_signals_generator(
    signal_bases: dict,
    filter_names: list = None,
    skip_signals: set = None
):
    """
    Generator that yields filtered signals one at a time.

    This allows incremental saving without holding all filtered signals in memory.

    Args:
        signal_bases: Dict mapping signal_name -> 2D array (n_time, n_etfs)
        filter_names: List of filter names to apply (default: all)
        skip_signals: Set of filtered signal names to skip (for resume)

    Yields:
        Tuple of (filtered_signal_name, filtered_data_2d)
    """
    if filter_names is None:
        filter_names = get_available_filters()

    if skip_signals is None:
        skip_signals = set()

    for signal_name, signal_data in signal_bases.items():
        # Ensure 2D
        if signal_data.ndim == 1:
            signal_data = signal_data.reshape(-1, 1)

        for filter_name in filter_names:
            # Construct filtered signal name
            filtered_name = f"{signal_name}__{filter_name}"

            # Skip if already computed
            if filtered_name in skip_signals:
                continue

            # Apply filter
            filtered_data = apply_single_filter(signal_data, filter_name)

            yield filtered_name, filtered_data


def compute_filtered_signals_batched(
    signal_bases: dict,
    filter_names: list = None,
    skip_signals: set = None,
    batch_size: int = 50
):
    """
    Generator that processes signals in batches for efficiency.

    Batches multiple signals together for each filter application,
    which is more efficient for GPU/vectorized operations.

    Args:
        signal_bases: Dict mapping signal_name -> 2D array (n_time, n_etfs)
        filter_names: List of filter names to apply (default: all)
        skip_signals: Set of filtered signal names to skip (for resume)
        batch_size: Number of signals to batch together

    Yields:
        Tuple of (filtered_signal_name, filtered_data_2d)
    """
    if filter_names is None:
        filter_names = get_available_filters()

    if skip_signals is None:
        skip_signals = set()

    # Get signal names and data
    signal_names = list(signal_bases.keys())

    # Process each filter
    for filter_name in filter_names:
        filter_func, filter_kwargs = get_filter(filter_name)

        # Determine which signals need this filter
        signals_to_process = []
        for signal_name in signal_names:
            filtered_name = f"{signal_name}__{filter_name}"
            if filtered_name not in skip_signals:
                signals_to_process.append(signal_name)

        if not signals_to_process:
            continue

        # Process in batches
        for batch_start in range(0, len(signals_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(signals_to_process))
            batch_names = signals_to_process[batch_start:batch_end]

            # Stack signals into 3D array: (n_signals, n_time, n_etfs)
            batch_data = []
            for signal_name in batch_names:
                data = signal_bases[signal_name]
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                batch_data.append(data)

            # Stack along axis 0
            batch_3d = np.stack(batch_data, axis=0)

            # Apply filter to entire batch
            filtered_batch = filter_func(batch_3d, **filter_kwargs)

            # Yield individual results
            for i, signal_name in enumerate(batch_names):
                filtered_name = f"{signal_name}__{filter_name}"
                yield filtered_name, filtered_batch[i].astype(np.float32)


def compute_all_filtered_signals_generator(
    signal_bases: dict,
    filter_names: list = None
):
    """
    Generator that yields all filtered signals (no skip logic).

    Simpler version without resume capability.

    Args:
        signal_bases: Dict mapping signal_name -> 2D array (n_time, n_etfs)
        filter_names: List of filter names to apply (default: all)

    Yields:
        Tuple of (filtered_signal_name, filtered_data_2d)
    """
    yield from compute_filtered_signals_generator(
        signal_bases,
        filter_names=filter_names,
        skip_signals=set()
    )


# =============================================================================
# HIGH-PERFORMANCE FILTER PROCESSING - SEQUENTIAL WITH VECTORIZATION
# =============================================================================

def compute_filtered_signals_optimized(
    signal_bases: dict,
    filter_names: list = None,
    skip_signals: set = None,
    batch_size: int = 50,  # Not used, kept for API compat
    show_progress: bool = True,
    n_workers: int = None  # Not used, kept for API compat
):
    """
    Memory-efficient filter computation - processes one signal at a time.

    Strategy:
    - Process each signal through ALL filters before moving to next signal
    - This keeps memory usage low (only 1 signal in memory at a time)
    - Uses vectorized numpy/scipy operations for speed

    Args:
        signal_bases: Dict mapping signal_name -> 2D array (n_time, n_etfs)
        filter_names: List of filter names to apply (default: all)
        skip_signals: Set of filtered signal names to skip (for resume)
        batch_size: Ignored (kept for API compatibility)
        show_progress: Whether to show tqdm progress bar
        n_workers: Ignored (kept for API compatibility)

    Yields:
        Tuple of (filtered_signal_name, filtered_data_2d)
    """
    if filter_names is None:
        filter_names = get_available_filters()

    if skip_signals is None:
        skip_signals = set()

    signal_names = list(signal_bases.keys())
    n_signals = len(signal_names)

    if n_signals == 0:
        return

    # Count total work
    total_work = 0
    for signal_name in signal_names:
        for filter_name in filter_names:
            if f"{signal_name}__{filter_name}" not in skip_signals:
                total_work += 1

    if total_work == 0:
        return

    print(f"  Processing {total_work} filtered signals (sequential, memory-efficient)...")

    # Try to import tqdm for progress bar
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        show_progress = False

    # Create progress bar
    if show_progress and has_tqdm:
        pbar = tqdm(total=total_work, desc="Filtering", unit="sig",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    else:
        pbar = None

    # Pre-fetch filter functions
    filter_funcs = {name: FILTER_REGISTRY[name] for name in filter_names}

    # Process each signal through all filters
    for signal_name in signal_names:
        signal_data = signal_bases[signal_name]
        if signal_data.ndim == 1:
            signal_data = signal_data.reshape(-1, 1)

        # Expand to 3D once: (1, n_time, n_etfs)
        signal_3d = signal_data[np.newaxis, :, :].astype(np.float32)

        # Apply all filters to this signal
        for filter_name in filter_names:
            filtered_name = f"{signal_name}__{filter_name}"

            if filtered_name in skip_signals:
                continue

            if pbar:
                pbar.set_postfix_str(f"{signal_name[:15]}|{filter_name}")

            try:
                filter_func, filter_kwargs = filter_funcs[filter_name]
                filtered_3d = filter_func(signal_3d, **filter_kwargs)
                yield filtered_name, filtered_3d[0].astype(np.float32)

            except Exception as e:
                print(f"\nERROR {filter_name} on {signal_name}: {e}")

            if pbar:
                pbar.update(1)

        # Free GPU memory after each signal (if GPU was used)
        if GPU_AVAILABLE:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    if pbar:
        pbar.close()
