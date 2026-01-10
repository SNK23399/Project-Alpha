"""
Signal Indicator Transformations - GPU ACCELERATED
===================================================

Indicators transform filtered signals into predictive features.
OPTIMIZED: Uses CuPy for GPU acceleration with efficient memory management.

Indicator Categories:
- Level & Momentum: raw, momentum, velocity, acceleration, curvature
- Statistical Normalization: z-score, cross-sectional, percentile
- Trend: MA crossovers, divergence
- Mean Reversion: reversion, envelope
- Breakout & Range: distance to high/low, drawdown, range position, ratio to peak
- Volatility: vol ratio, relative vol, signal-to-noise, roughness
- Higher Moments: skewness, kurtosis
- Temporal Persistence: autocorrelation, days since cross, streak, zero-cross rate
- Regime: above mean
- Information Theory: entropy
- Cross-Sectional Dynamics: dispersion, convergence
"""

import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np

# Try to import Numba for CPU fallbacks
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

CACHE_DIR = Path(__file__).parent / "cache"


def to_gpu(arr: np.ndarray):
    """Move array to GPU if available."""
    if GPU_AVAILABLE:
        return cp.asarray(arr)
    return arr


def to_cpu(arr) -> np.ndarray:
    """Move array back to CPU."""
    if GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


# =============================================================================
# GPU-Accelerated Rolling Statistics
# =============================================================================

def gpu_rolling_mean(arr, window: int):
    """GPU-accelerated rolling mean using cumsum trick."""
    xp = cp if GPU_AVAILABLE else np

    mask = xp.isnan(arr)
    arr_clean = xp.where(mask, 0, arr)

    cumsum = xp.cumsum(arr_clean, axis=0)
    count = xp.cumsum(~mask, axis=0)

    result = xp.full_like(arr, xp.nan)
    result[window:] = (cumsum[window:] - cumsum[:-window]) / xp.maximum(count[window:] - count[:-window], 1)

    return result


def gpu_rolling_std(arr, window: int):
    """GPU-accelerated rolling std using cumsum trick."""
    xp = cp if GPU_AVAILABLE else np

    mean = gpu_rolling_mean(arr, window)
    mean_sq = gpu_rolling_mean(arr ** 2, window)
    variance = mean_sq - mean ** 2
    variance = xp.maximum(variance, 0)

    return xp.sqrt(variance)


def gpu_rolling_sum(arr, window: int):
    """GPU-accelerated rolling sum."""
    xp = cp if GPU_AVAILABLE else np

    mask = xp.isnan(arr)
    arr_clean = xp.where(mask, 0, arr)

    cumsum = xp.cumsum(arr_clean, axis=0)

    result = xp.full_like(arr, xp.nan)
    result[window:] = cumsum[window:] - cumsum[:-window]

    return result


def _sliding_window_view_gpu(arr, window: int):
    """Create a sliding window view on GPU array along axis 0."""
    # Uses as_strided for efficient memory-sharing view
    n_time, n_cols = arr.shape
    new_shape = (n_time - window + 1, n_cols, window)
    new_strides = (arr.strides[0], arr.strides[1], arr.strides[0])
    return cp.lib.stride_tricks.as_strided(arr, shape=new_shape, strides=new_strides)


def gpu_rolling_max(arr, window: int):
    """GPU-accelerated rolling max using sliding window approach."""
    xp = cp if GPU_AVAILABLE else np
    n_time = arr.shape[0]

    result = xp.full_like(arr, xp.nan)

    if not GPU_AVAILABLE:
        from numpy.lib.stride_tricks import sliding_window_view
        windowed = sliding_window_view(arr, window, axis=0)
        result[window-1:] = np.nanmax(windowed, axis=2)
        return result

    # For GPU: use strided view which is memory-efficient (no copy)
    arr_clean = xp.where(xp.isnan(arr), -xp.inf, arr)

    # Create sliding window view and compute max
    windowed = _sliding_window_view_gpu(arr_clean, window)
    result[window-1:] = xp.max(windowed, axis=2)
    result = xp.where(result == -xp.inf, xp.nan, result)

    return result


def gpu_rolling_min(arr, window: int):
    """GPU-accelerated rolling min using sliding window approach."""
    xp = cp if GPU_AVAILABLE else np
    n_time = arr.shape[0]

    result = xp.full_like(arr, xp.nan)

    if not GPU_AVAILABLE:
        from numpy.lib.stride_tricks import sliding_window_view
        windowed = sliding_window_view(arr, window, axis=0)
        result[window-1:] = np.nanmin(windowed, axis=2)
        return result

    # For GPU: use strided view which is memory-efficient (no copy)
    arr_clean = xp.where(xp.isnan(arr), xp.inf, arr)

    # Create sliding window view and compute min
    windowed = _sliding_window_view_gpu(arr_clean, window)
    result[window-1:] = xp.min(windowed, axis=2)
    result = xp.where(result == xp.inf, xp.nan, result)

    return result


def gpu_rolling_skewness(arr, window: int):
    """GPU-accelerated rolling skewness."""
    xp = cp if GPU_AVAILABLE else np

    mean = gpu_rolling_mean(arr, window)
    std = gpu_rolling_std(arr, window)

    # Third central moment
    diff = arr - mean
    m3 = gpu_rolling_mean(diff ** 3, window)

    skew = m3 / (std ** 3 + 1e-10)
    return skew


def gpu_rolling_kurtosis(arr, window: int):
    """GPU-accelerated rolling excess kurtosis."""
    xp = cp if GPU_AVAILABLE else np

    mean = gpu_rolling_mean(arr, window)
    std = gpu_rolling_std(arr, window)

    # Fourth central moment
    diff = arr - mean
    m4 = gpu_rolling_mean(diff ** 4, window)

    kurt = m4 / (std ** 4 + 1e-10) - 3  # Excess kurtosis
    return kurt


def gpu_rolling_autocorr(arr, window: int, lag: int):
    """GPU-accelerated rolling autocorrelation at given lag."""
    xp = cp if GPU_AVAILABLE else np
    n_time = arr.shape[0]

    result = xp.full_like(arr, xp.nan)

    if lag >= window:
        return result

    # Compute correlation between arr[t] and arr[t-lag] over window
    arr_lagged = xp.full_like(arr, xp.nan)
    arr_lagged[lag:] = arr[:-lag]

    # Rolling correlation
    mean_x = gpu_rolling_mean(arr, window)
    mean_y = gpu_rolling_mean(arr_lagged, window)

    cov = gpu_rolling_mean(arr * arr_lagged, window) - mean_x * mean_y
    std_x = gpu_rolling_std(arr, window)
    std_y = gpu_rolling_std(arr_lagged, window)

    result = cov / (std_x * std_y + 1e-10)
    return result


# =============================================================================
# Compute Rolling Stats for All Required Windows
# =============================================================================

def compute_rolling_stats(
    signals_np: np.ndarray,
    n_time: int,
    keep_on_gpu: bool = False,
    verbose: bool = False,
    fast_mode: bool = True,
    use_fp16: bool = True
) -> dict:
    """
    Compute rolling statistics needed for indicator computation.

    OPTIMIZED: Uses fp16 for speed, keeps data on GPU during computation.

    Args:
        signals_np: (n_signals, n_time, n_etfs) array
        n_time: number of time steps
        keep_on_gpu: if True, return GPU arrays
        verbose: if True, print progress for each stat type
        fast_mode: if True, skip expensive stats (skew/kurt/autocorr) - ~3x faster
        use_fp16: if True, use float16 for ~2x speed (sufficient for feature discovery)

    Returns:
        dict with keys:
            'mean': {window: array} for windows [5, 21, 63, 126, 252]
            'std': {window: array} for windows [5, 21, 63, 126, 252]
            'max': {window: array} for windows [21, 63, 126]
            'min': {window: array} for windows [21, 63, 126]
            'skew': {window: array} for windows [21, 63, 126] (if not fast_mode)
            'kurt': {window: array} for windows [21, 63, 126] (if not fast_mode)
            'autocorr': {lag: array} for lags [1, 5, 21] (if not fast_mode)
    """
    xp = cp if GPU_AVAILABLE else np
    n_signals, _, n_etfs = signals_np.shape

    # Use fp16 for speed (halves memory, doubles throughput)
    dtype = np.float16 if use_fp16 else np.float32
    signals = to_gpu(signals_np.astype(dtype))
    signals_flat = signals.transpose(1, 0, 2).reshape(n_time, -1)

    def reshape_to_3d(arr_flat):
        return arr_flat.reshape(n_time, n_signals, n_etfs).transpose(1, 0, 2)

    def log(msg):
        if verbose:
            print(msg, end="", flush=True)

    # Keep results on GPU during computation, transfer all at end
    stats_gpu = {
        'mean': {},
        'std': {},
        'max': {},
        'min': {},
        'skew': {},
        'kurt': {},
        'autocorr': {},
    }

    # Mean and std at all windows (fast - uses cumsum trick)
    log("ms")  # mean/std
    for w in [5, 21, 63, 126, 252]:
        if w < n_time:
            mean_flat = gpu_rolling_mean(signals_flat, w)
            std_flat = gpu_rolling_std(signals_flat, w)
            stats_gpu['mean'][w] = reshape_to_3d(mean_flat)
            stats_gpu['std'][w] = reshape_to_3d(std_flat)
            del mean_flat, std_flat
            log(".")

    # Max, min at breakout windows
    log(" mm")
    for w in [21, 63, 126]:
        if w < n_time:
            max_flat = gpu_rolling_max(signals_flat, w)
            min_flat = gpu_rolling_min(signals_flat, w)
            stats_gpu['max'][w] = reshape_to_3d(max_flat)
            stats_gpu['min'][w] = reshape_to_3d(min_flat)
            del max_flat, min_flat
            log(".")

    # Skewness, kurtosis, autocorr - skip in fast mode (rarely best predictors)
    if not fast_mode:
        log(" sk")
        for w in [21, 63, 126]:
            if w < n_time:
                skew_flat = gpu_rolling_skewness(signals_flat, w)
                kurt_flat = gpu_rolling_kurtosis(signals_flat, w)
                stats_gpu['skew'][w] = reshape_to_3d(skew_flat)
                stats_gpu['kurt'][w] = reshape_to_3d(kurt_flat)
                del skew_flat, kurt_flat
                log(".")

        log(" ac")
        if 63 < n_time:
            for lag in [1, 5, 21]:
                autocorr_flat = gpu_rolling_autocorr(signals_flat, 63, lag)
                stats_gpu['autocorr'][lag] = reshape_to_3d(autocorr_flat)
                del autocorr_flat
                log(".")

    # Transfer all results to CPU at once (if not keeping on GPU)
    log(" →cpu")
    if keep_on_gpu and GPU_AVAILABLE:
        return stats_gpu

    stats = {
        'mean': {},
        'std': {},
        'max': {},
        'min': {},
        'skew': {},
        'kurt': {},
        'autocorr': {},
    }

    # Keep as fp16 on CPU too for consistency
    out_dtype = np.float16 if use_fp16 else np.float32
    for key in ['mean', 'std', 'max', 'min', 'skew', 'kurt']:
        for w, arr in stats_gpu[key].items():
            stats[key][w] = to_cpu(arr).astype(out_dtype)
    for lag, arr in stats_gpu['autocorr'].items():
        stats['autocorr'][lag] = to_cpu(arr).astype(out_dtype)

    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()

    return stats


# =============================================================================
# Main Indicator Computation
# =============================================================================

def compute_indicators_chunked(
    signals_np: np.ndarray,
    rolling_stats: dict,
    n_time: int,
    n_etfs: int,
    use_fp16: bool = True,
    chunk_size: int = 10
):
    """
    Generator that yields indicator chunks to avoid memory exhaustion.

    Yields:
        (chunk_features, chunk_names): tuple of (n_signals, n_chunk_indicators, n_time, n_etfs), list
    """
    xp = cp if GPU_AVAILABLE else np
    n_signals = signals_np.shape[0]
    dtype = np.float16 if use_fp16 else np.float32

    stats = rolling_stats
    signals = to_gpu(signals_np.astype(dtype))

    def flatten(arr):
        return arr.transpose(1, 0, 2).reshape(n_time, -1)

    def unflatten(arr_flat):
        return arr_flat.reshape(n_time, n_signals, n_etfs).transpose(1, 0, 2)

    def get_ma(w):
        return to_gpu(stats['mean'][w].astype(dtype))

    def get_std(w):
        return to_gpu(stats['std'][w].astype(dtype))

    def get_max(w):
        if w in stats['max']:
            return to_gpu(stats['max'][w].astype(dtype))
        return None

    def get_min(w):
        if w in stats['min']:
            return to_gpu(stats['min'][w].astype(dtype))
        return None

    # Accumulate indicators in chunks
    chunk_names = []
    chunk_data = []

    # Store chunks that are ready to be yielded
    ready_chunks = []

    def add_indicator(name, data):
        """Add indicator, transfer to CPU, check if chunk ready."""
        nonlocal chunk_names, chunk_data
        chunk_names.append(name)
        # Transfer to CPU immediately to free GPU memory
        cpu_data = to_cpu(data)
        chunk_data.append(cpu_data)

        # Free GPU memory after each indicator
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()

        # Check if chunk is full - if so, prepare it for yielding
        if len(chunk_data) >= chunk_size:
            features = np.stack(chunk_data, axis=1)
            ready_chunks.append((features, chunk_names.copy()))
            chunk_names.clear()
            chunk_data.clear()

    # =========================================================================
    # LEVEL & MOMENTUM
    # =========================================================================

    # Level
    add_indicator('level', signals)

    # Momentum
    for h in [5, 21, 63, 126]:
        shifted = xp.full_like(signals, xp.nan)
        shifted[:, h:, :] = signals[:, :-h, :]
        mom = (signals - shifted) / (xp.abs(shifted) + 1e-10)
        add_indicator(f'mom_{h}d', mom)
        del shifted, mom

    # Yield any ready chunks
    for chunk in ready_chunks:
        yield chunk
    ready_chunks.clear()

    # Momentum Acceleration
    for h in [5, 21, 63]:
        shifted = xp.full_like(signals, xp.nan)
        shifted[:, h:, :] = signals[:, :-h, :]
        mom = (signals - shifted) / (xp.abs(shifted) + 1e-10)
        mom_accel = xp.diff(mom, axis=1, prepend=xp.float32(xp.nan))
        add_indicator(f'mom_accel_{h}d', mom_accel)
        del shifted, mom, mom_accel

    # Velocity
    velocity = xp.diff(signals, axis=1, prepend=xp.float32(xp.nan))
    for w in [5, 21, 63]:
        std_w = get_std(w)
        add_indicator(f'velocity_{w}d', velocity / (std_w + 1e-10))
        del std_w

    for chunk in ready_chunks:
        yield chunk
    ready_chunks.clear()

    # Acceleration
    accel = xp.diff(velocity, axis=1, prepend=xp.float32(xp.nan))
    for w in [5, 21, 63]:
        std_w = get_std(w)
        add_indicator(f'accel_{w}d', accel / (std_w + 1e-10))
        del std_w
    del velocity, accel

    for chunk in ready_chunks:
        yield chunk
    ready_chunks.clear()

    # =========================================================================
    # STATISTICAL NORMALIZATION
    # =========================================================================

    # Z-Score
    for w in [21, 63, 126, 252]:
        ma_w = get_ma(w)
        std_w = get_std(w)
        zscore = (signals - ma_w) / (std_w + 1e-10)
        add_indicator(f'zscore_{w}d', zscore)
        del ma_w, std_w, zscore

    for chunk in ready_chunks:
        yield chunk
    ready_chunks.clear()

    # Cross-sectional Z-Score and Rank
    xs_mean = xp.nanmean(signals, axis=2, keepdims=True)
    xs_std = xp.nanstd(signals, axis=2, keepdims=True)
    xs_zscore = (signals - xs_mean) / (xs_std + 1e-10)
    add_indicator('xs_zscore', xs_zscore)
    xs_rank = 1 / (1 + xp.exp(-xs_zscore))
    add_indicator('xs_rank', xs_rank)
    del xs_mean, xs_std, xs_zscore, xs_rank

    # Percentile
    for w in [63, 126, 252]:
        ma_w = get_ma(w)
        std_w = get_std(w)
        pct = (signals - ma_w) / (2 * std_w + 1e-10)
        add_indicator(f'percentile_{w}d', pct)
        del ma_w, std_w, pct

    for chunk in ready_chunks:
        yield chunk
    ready_chunks.clear()

    # =========================================================================
    # TREND INDICATORS
    # =========================================================================

    trend_pairs = [(5, 21, 'short'), (21, 63, 'med'), (63, 126, 'long'), (126, 252, 'ext')]
    for fast, slow, name in trend_pairs:
        ma_fast = get_ma(fast)
        ma_slow = get_ma(slow)
        trend = (ma_fast - ma_slow) / (xp.abs(ma_slow) + 1e-10)
        add_indicator(f'trend_{name}', trend)
        del ma_fast, ma_slow, trend

    # Divergence
    for w in [21, 63]:
        ma_w = get_ma(w)
        div = signals - ma_w
        add_indicator(f'divergence_{w}d', div)
        del ma_w, div

    for chunk in ready_chunks:
        yield chunk
    ready_chunks.clear()

    # =========================================================================
    # MEAN REVERSION
    # =========================================================================

    for w in [21, 63, 126, 252]:
        ma_w = get_ma(w)
        revert = (signals - ma_w) / (xp.abs(ma_w) + 1e-10)
        add_indicator(f'revert_{w}d', revert)
        del ma_w, revert

    # Envelope
    for w in [21, 63]:
        ma_w = get_ma(w)
        std_w = get_std(w)
        envelope = (signals - ma_w) / (2 * std_w + 1e-10)
        add_indicator(f'envelope_{w}d', envelope)
        del ma_w, std_w, envelope

    for chunk in ready_chunks:
        yield chunk
    ready_chunks.clear()

    # =========================================================================
    # BREAKOUT & RANGE
    # =========================================================================

    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()

    for w in [21, 63, 126]:
        rolling_max = get_max(w)
        rolling_min = get_min(w)

        if rolling_max is None or rolling_min is None:
            signals_flat = flatten(signals)
            rolling_max = unflatten(gpu_rolling_max(signals_flat, w))
            rolling_min = unflatten(gpu_rolling_min(signals_flat, w))

        std_w = get_std(w)
        range_val = rolling_max - rolling_min

        add_indicator(f'dist_high_{w}d', (signals - rolling_max) / (std_w + 1e-10))
        add_indicator(f'dist_low_{w}d', (signals - rolling_min) / (std_w + 1e-10))
        add_indicator(f'drawdown_{w}d', (signals - rolling_max) / (xp.abs(rolling_max) + 1e-10))
        add_indicator(f'range_pos_{w}d', (signals - rolling_min) / (range_val + 1e-10))
        add_indicator(f'ratio_peak_{w}d', signals / (rolling_max + 1e-10))

        del rolling_max, rolling_min, std_w, range_val

        for chunk in ready_chunks:
            yield chunk
        ready_chunks.clear()

    # =========================================================================
    # VOLATILITY METRICS
    # =========================================================================

    vol_pairs = [(5, 21, '5_21'), (21, 63, '21_63'), (63, 126, '63_126')]
    for short, long, name in vol_pairs:
        std_short = get_std(short)
        std_long = get_std(long)
        vol_ratio = std_short / (std_long + 1e-10)
        add_indicator(f'vol_ratio_{name}', vol_ratio)
        del std_short, std_long, vol_ratio

    for short in [21, 63]:
        std_short = get_std(short)
        std_252 = get_std(252)
        rel_vol = std_short / (std_252 + 1e-10)
        add_indicator(f'rel_vol_{short}_252', rel_vol)
        del std_short, std_252, rel_vol

    # Signal-to-Noise
    for w in [21, 63, 126]:
        ma_w = get_ma(w)
        std_w = get_std(w)
        snr = xp.abs(ma_w) / (std_w + 1e-10)
        add_indicator(f'snr_{w}d', snr)
        del ma_w, std_w, snr

    # Roughness: path length / net move (reuses rolling sum via cumsum)
    abs_diff = xp.abs(xp.diff(signals, axis=1, prepend=xp.float16(xp.nan)))
    for w in [21, 63, 126]:
        # Rolling sum of |delta| using cumsum trick
        cumsum_abs = xp.cumsum(xp.where(xp.isnan(abs_diff), 0, abs_diff), axis=1)
        path_len = xp.full_like(signals, xp.nan)
        path_len[:, w:, :] = cumsum_abs[:, w:, :] - cumsum_abs[:, :-w, :]
        # Net move over window
        net_move = signals - xp.roll(signals, w, axis=1)
        net_move[:, :w, :] = xp.nan
        roughness = path_len / (xp.abs(net_move) + 1e-10)
        add_indicator(f'roughness_{w}d', roughness)
        del path_len, net_move, roughness
    del abs_diff, cumsum_abs

    for chunk in ready_chunks:
        yield chunk
    ready_chunks.clear()

    # =========================================================================
    # HIGHER MOMENTS (cheap: reuse mean/std, just need rolling mean of powers)
    # =========================================================================

    signals_flat = flatten(signals)
    for w in [21, 63, 126]:
        ma_w = get_ma(w)
        std_w = get_std(w)
        diff = signals - ma_w

        # Skewness: E[(x-mu)^3] / std^3
        diff_flat = flatten(diff)
        m3_flat = gpu_rolling_mean(diff_flat ** 3, w)
        m3 = unflatten(m3_flat)
        skew = m3 / (std_w ** 3 + 1e-10)
        add_indicator(f'skew_{w}d', skew)
        del m3_flat, m3, skew

        # Kurtosis: E[(x-mu)^4] / std^4 - 3
        m4_flat = gpu_rolling_mean(diff_flat ** 4, w)
        m4 = unflatten(m4_flat)
        kurt = m4 / (std_w ** 4 + 1e-10) - 3
        add_indicator(f'kurt_{w}d', kurt)
        del m4_flat, m4, kurt, diff_flat, diff, ma_w, std_w

    for chunk in ready_chunks:
        yield chunk
    ready_chunks.clear()

    # =========================================================================
    # CURVATURE (cheap: reuses velocity/accel already computed conceptually)
    # =========================================================================

    velocity = xp.diff(signals, axis=1, prepend=xp.float16(xp.nan))
    accel = xp.diff(velocity, axis=1, prepend=xp.float16(xp.nan))
    for w in [5, 21, 63]:
        # Curvature = acceleration / |velocity|
        std_w = get_std(w)
        curv = accel / (xp.abs(velocity) + 1e-10)
        add_indicator(f'curvature_{w}d', curv)
    del velocity, accel

    for chunk in ready_chunks:
        yield chunk
    ready_chunks.clear()

    # =========================================================================
    # REGIME INDICATORS
    # =========================================================================

    for w in [21, 63, 126]:
        ma_w = get_ma(w)
        above = (signals > ma_w).astype(dtype)
        add_indicator(f'above_mean_{w}d', above)
        del ma_w, above

    # =========================================================================
    # CROSS-SECTIONAL DYNAMICS
    # =========================================================================

    xs_std = xp.nanstd(signals, axis=2, keepdims=True)
    xs_std_broadcast = xp.broadcast_to(xs_std, signals.shape)
    add_indicator('dispersion', xs_std_broadcast.copy())

    for w in [21, 63]:
        xs_std_flat = xs_std[:, :, 0]
        xs_std_shifted = xp.full_like(xs_std_flat, xp.nan)
        xs_std_shifted[:, w:] = xs_std_flat[:, :-w]
        convergence = xs_std_flat - xs_std_shifted
        convergence_broadcast = xp.broadcast_to(convergence[:, :, None], signals.shape)
        add_indicator(f'convergence_{w}d', convergence_broadcast.copy())
        del xs_std_shifted, convergence, convergence_broadcast

    del xs_std, xs_std_broadcast

    # Yield any pending ready chunks
    for chunk in ready_chunks:
        yield chunk
    ready_chunks.clear()

    # Yield remaining indicators that didn't fill a complete chunk
    if chunk_data:
        features = np.stack(chunk_data, axis=1)
        yield (features, chunk_names.copy())

    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()


def compute_indicators(
    signals_np: np.ndarray,
    rolling_stats: dict,
    n_time: int,
    n_etfs: int,
    filter_name: str = "unknown",
    use_cache: bool = False,
    use_fp16: bool = True,
    chunk_size: int = 10
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute indicators for filtered signals.

    GPU-optimized with fp16 for speed. Uses chunked computation to save memory.

    Args:
        signals_np: (n_signals, n_time, n_etfs) filtered signals
        rolling_stats: dict from compute_rolling_stats() with keys:
                       'mean', 'std', 'max', 'min', 'skew', 'kurt'
        n_time: number of time steps
        n_etfs: number of ETFs
        filter_name: name for caching
        use_cache: whether to use disk cache
        use_fp16: if True, use float16 for ~2x speed
        chunk_size: number of indicators per chunk

    Returns:
        features: (n_signals, n_indicators, n_time, n_etfs)
        indicator_names: list of indicator names
    """
    # Collect all chunks
    all_features = []
    all_names = []

    for chunk_features, chunk_names in compute_indicators_chunked(
        signals_np, rolling_stats, n_time, n_etfs, use_fp16, chunk_size
    ):
        all_features.append(chunk_features)
        all_names.extend(chunk_names)

    # Concatenate along indicator axis
    features = np.concatenate(all_features, axis=1)
    return features, all_names


# For backwards compatibility
INDICATOR_NAMES = None  # Will be set after first computation

def compute_indicators_streaming(*args, **kwargs):
    """Alias for compute_indicators."""
    return compute_indicators(*args, **kwargs)


def get_indicator_count():
    """Return the number of indicators that will be computed.

    Current implementation (optimized - reuses precomputed rolling stats):
    - Level & Momentum: 14 + 3 (curvature) = 17
    - Statistical Normalization: 9
    - Trend: 6
    - Mean Reversion: 6
    - Breakout & Range: 15
    - Volatility Metrics: 8 + 3 (roughness) = 11
    - Higher Moments: 6 (skew + kurt)
    - Regime: 3
    - Cross-Sectional Dynamics: 3

    Total: 76 indicators

    Still excluded (truly expensive - require sequential scan or histograms):
    - Autocorrelation (3) - needs rolling correlation with lagged self
    - Days Since Cross (2) - sequential scan
    - Streak (1) - sequential scan
    - Zero-Cross Rate (2) - could add with cumsum trick
    - Entropy (2) - histogram binning per window
    """
    count = 0
    # Level & Momentum
    count += 1   # level
    count += 4   # mom 5,21,63,126
    count += 3   # mom_accel 5,21,63
    count += 3   # velocity 5,21,63
    count += 3   # accel 5,21,63
    count += 3   # curvature 5,21,63
    # Statistical Normalization
    count += 4   # zscore 21,63,126,252
    count += 2   # xs_zscore, xs_rank
    count += 3   # percentile 63,126,252
    # Trend
    count += 4   # trend short,med,long,ext
    count += 2   # divergence 21,63
    # Mean Reversion
    count += 4   # revert 21,63,126,252
    count += 2   # envelope 21,63
    # Breakout & Range (5 indicators × 3 windows)
    count += 15  # dist_high, dist_low, drawdown, range_pos, ratio_peak × [21,63,126]
    # Volatility Metrics
    count += 3   # vol_ratio 5_21, 21_63, 63_126
    count += 2   # rel_vol 21_252, 63_252
    count += 3   # snr 21,63,126
    count += 3   # roughness 21,63,126
    # Higher Moments
    count += 3   # skew 21,63,126
    count += 3   # kurt 21,63,126
    # Regime
    count += 3   # above_mean 21,63,126
    # Cross-Sectional Dynamics
    count += 1   # dispersion
    count += 2   # convergence 21,63
    return count  # Total: 76


N_INDICATORS = get_indicator_count()
