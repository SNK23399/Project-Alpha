"""
DPO Enhanced Variants Library
==============================

Implements Detrended Price Oscillator (DPO) with various moving average types.

Standard DPO formula:
  shift = period // 2 + 1
  dpo = price[t + shift] - MA[t, period]

This library tests different MA types to find optimal DPO variants:

ORIGINAL MA TYPES (6):
- SMA: Simple Moving Average (baseline)
- EMA: Exponential Moving Average (responsive)
- DEMA: Double Exponential Moving Average (smooth + responsive)
- TEMA: Triple Exponential Moving Average (smoothest)
- ZLEMA: Zero-Lag EMA (no lag)
- Hull MA: Weighted MA (reduced lag)

EXTENDED MA TYPES (6):
- WMA: Weighted Moving Average (linear weights)
- TRIMA: Triangular Moving Average (double-smoothed SMA)
- KAMA: Kaufman Adaptive Moving Average (responds to market regime)
- Median: Outlier-robust filter (preserves edges)
- Gaussian MA: Bell-curve weighted smoothing
- Adaptive EMA: Volatility-responsive exponential

Each variant is computed for DPO windows: 40d through 70d (in steps of 1d = 31 periods)
Total: 31 periods × 12 MA types = 372 variants

OPTIMIZATION: Uses GPU-accelerated moving average implementations from signal_filters
- All MA types: GPU-accelerated with CuPy and scipy.signal optimizations
- Batch processing: Vectorized computations across all ETFs simultaneously
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple

# Import optimized, GPU-accelerated MA implementations from signal_filters
from library.signal_filters import (
    causal_sma as _causal_sma,
    causal_ema as _causal_ema,
    causal_dema as _causal_dema,
    causal_tema as _causal_tema,
    causal_zlema as _causal_zlema,
    causal_hull_ma as _causal_hull_ma,
    causal_wma as _causal_wma,
    causal_trima as _causal_trima,
    causal_kama as _causal_kama,
    causal_median as _causal_median,
    causal_gaussian_ma as _causal_gaussian_ma,
    causal_adaptive_ema as _causal_adaptive_ema,
)


def causal_sma(prices_2d: np.ndarray, window: int) -> np.ndarray:
    """Wrapper for causal_sma that handles 2D input (n_time, n_etfs)."""
    # Reshape to 3D: (1, n_time, n_etfs)
    prices_3d = prices_2d[np.newaxis, :, :]
    result_3d = _causal_sma(prices_3d, window)
    # Reshape back to 2D
    return result_3d[0, :, :]


def causal_ema(prices_2d: np.ndarray, span: int) -> np.ndarray:
    """Wrapper for causal_ema that handles 2D input (n_time, n_etfs)."""
    prices_3d = prices_2d[np.newaxis, :, :]
    result_3d = _causal_ema(prices_3d, span)
    return result_3d[0, :, :]


def causal_dema(prices_2d: np.ndarray, span: int) -> np.ndarray:
    """Wrapper for causal_dema that handles 2D input (n_time, n_etfs)."""
    prices_3d = prices_2d[np.newaxis, :, :]
    result_3d = _causal_dema(prices_3d, span)
    return result_3d[0, :, :]


def causal_tema(prices_2d: np.ndarray, span: int) -> np.ndarray:
    """Wrapper for causal_tema that handles 2D input (n_time, n_etfs)."""
    prices_3d = prices_2d[np.newaxis, :, :]
    result_3d = _causal_tema(prices_3d, span)
    return result_3d[0, :, :]


def causal_zlema(prices_2d: np.ndarray, span: int) -> np.ndarray:
    """Wrapper for causal_zlema that handles 2D input (n_time, n_etfs)."""
    prices_3d = prices_2d[np.newaxis, :, :]
    result_3d = _causal_zlema(prices_3d, span)
    return result_3d[0, :, :]


def causal_hull_ma(prices_2d: np.ndarray, period: int) -> np.ndarray:
    """Wrapper for causal_hull_ma that handles 2D input (n_time, n_etfs)."""
    prices_3d = prices_2d[np.newaxis, :, :]
    result_3d = _causal_hull_ma(prices_3d, period)
    return result_3d[0, :, :]


def causal_wma(prices_2d: np.ndarray, window: int) -> np.ndarray:
    """Wrapper for causal_wma that handles 2D input (n_time, n_etfs)."""
    prices_3d = prices_2d[np.newaxis, :, :]
    result_3d = _causal_wma(prices_3d, window)
    return result_3d[0, :, :]


def causal_trima(prices_2d: np.ndarray, window: int) -> np.ndarray:
    """Wrapper for causal_trima that handles 2D input (n_time, n_etfs)."""
    prices_3d = prices_2d[np.newaxis, :, :]
    result_3d = _causal_trima(prices_3d, window)
    return result_3d[0, :, :]


def causal_kama(prices_2d: np.ndarray, period: int) -> np.ndarray:
    """Wrapper for causal_kama that handles 2D input (n_time, n_etfs).

    Kaufman Adaptive Moving Average - adapts to trending vs choppy markets.
    Uses period as the main parameter (fast=2, slow=30 are optimized defaults).
    """
    prices_3d = prices_2d[np.newaxis, :, :]
    result_3d = _causal_kama(prices_3d, period=period, fast=2, slow=30)
    return result_3d[0, :, :]


def causal_median(prices_2d: np.ndarray, window: int) -> np.ndarray:
    """Wrapper for causal_median that handles 2D input (n_time, n_etfs).

    Robust to outliers while preserving edges (unlike averaging smooths).
    """
    prices_3d = prices_2d[np.newaxis, :, :]
    result_3d = _causal_median(prices_3d, window)
    return result_3d[0, :, :]


def causal_gaussian_ma(prices_2d: np.ndarray, window: int) -> np.ndarray:
    """Wrapper for causal_gaussian_ma that handles 2D input (n_time, n_etfs).

    Gaussian-weighted moving average with smooth frequency rolloff.
    Uses sigma = window / 4 for balanced smoothing.
    """
    prices_3d = prices_2d[np.newaxis, :, :]
    sigma = window / 4.0  # Reasonable default for smooth bell curve
    result_3d = _causal_gaussian_ma(prices_3d, window, sigma=sigma)
    return result_3d[0, :, :]


def causal_adaptive_ema(prices_2d: np.ndarray, span: int) -> np.ndarray:
    """Wrapper for causal_adaptive_ema that handles 2D input (n_time, n_etfs).

    Adaptive EMA that adjusts smoothing based on volatility.
    Uses base_span = span parameter, vol_window = span / 3 for adaptation.
    """
    prices_3d = prices_2d[np.newaxis, :, :]
    vol_window = max(5, span // 3)  # At least 5 bars for volatility window
    result_3d = _causal_adaptive_ema(prices_3d, base_span=span, vol_window=vol_window)
    return result_3d[0, :, :]


def compute_dpo_variants_generator(
    etf_prices: pd.DataFrame,
    core_prices: pd.Series
) -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Generator that yields enhanced DPO variants.

    For each DPO period (40d through 70d in steps of 1d), computes with each MA type:

    Original 6 types:
    - sma (baseline simple average)
    - ema (responsive exponential)
    - dema (smooth + responsive double exponential)
    - tema (smoothest triple exponential)
    - zlema (zero-lag exponential)
    - hull (reduced lag weighted MA)

    Extended 6 types:
    - wma (weighted moving average)
    - trima (triangular double-smoothed)
    - kama (Kaufman adaptive - responds to market regime)
    - median (outlier-robust, preserves edges)
    - gaussian (smooth bell-curve weighted)
    - adaptive_ema (volatility-responsive exponential)

    Yields:
        (signal_name, signal_2d_array)

    Total: 31 periods × 12 MA types = 372 variants

    Example signals:
        dpo_40d__sma, dpo_41d__sma, ..., dpo_70d__sma
        dpo_40d__ema, dpo_41d__ema, ..., dpo_70d__ema
        dpo_40d__tema, ..., dpo_60d__tema (best), ..., dpo_70d__tema
        dpo_40d__kama, ..., dpo_60d__kama (possibly best adaptive), ..., dpo_70d__kama
        dpo_40d__median, dpo_40d__gaussian, dpo_40d__adaptive_ema, etc.
    """

    prices_arr = etf_prices.values
    core_prices_arr = core_prices.values

    # DPO windows: fine-grained exploration from 40d to 70d
    # 40d:  ~2 months
    # 50d:  3 months - historically good
    # 60d:  3+ months - has high IR variants (BEST)
    # 70d:  3.5 months - upper bound
    # Test intermediate values for fine-grained optimization
    dpo_periods = list(range(40, 71))  # 40, 41, 42, ..., 70 (31 periods)
    
    # MA types to test - using optimized, GPU-accelerated implementations from signal_filters
    ma_types = {
        # Original 6 types
        'sma': causal_sma,              # Baseline (simple average)
        'ema': causal_ema,              # Standard exponential (responsive)
        'dema': causal_dema,            # Double exponential (smooth + responsive)
        'tema': causal_tema,            # Triple exponential (smoothest)
        'zlema': causal_zlema,          # Zero-lag exponential (no lag)
        'hull': causal_hull_ma,         # Hull MA (reduced lag)
        # Additional 6 types
        'wma': causal_wma,              # Weighted MA (linear weights)
        'trima': causal_trima,          # Triangular MA (double-smoothed)
        'kama': causal_kama,            # Kaufman Adaptive (adapts to trending/choppy)
        'median': causal_median,        # Median (outlier-robust)
        'gaussian': causal_gaussian_ma, # Gaussian MA (smooth bell curve)
        'adaptive_ema': causal_adaptive_ema,  # Adaptive EMA (vol-responsive)
    }
    
    for dpo_period in dpo_periods:
        dpo_shift = dpo_period // 2 + 1
        
        for ma_name, ma_func in ma_types.items():
            # Create signal name upfront for error reporting
            signal_name = f'dpo_{dpo_period}d__{ma_name}'

            try:
                # Compute MA
                ma = ma_func(prices_arr, dpo_period)

                # Compute DPO: price[t + shift] - MA[t]
                dpo = np.empty_like(prices_arr)
                dpo[:-dpo_shift, :] = prices_arr[dpo_shift:, :] - ma[:-dpo_shift, :]
                dpo[-dpo_shift:, :] = np.nan

                yield signal_name, dpo

            except Exception as e:
                print(f"  Warning: Failed to compute {signal_name}: {e}")
                continue


def count_dpo_variants() -> int:
    """Count total DPO variants that will be generated."""
    dpo_periods = list(range(40, 71))  # 40 to 70 inclusive (31 periods)
    ma_types = ['sma', 'ema', 'dema', 'tema', 'zlema', 'hull',
                'wma', 'trima', 'kama', 'median', 'gaussian', 'adaptive_ema']
    return len(dpo_periods) * len(ma_types)


if __name__ == "__main__":
    print(f"DPO Enhanced Variants Library - Extended MA Types")
    print(f"Total variants: {count_dpo_variants()}")
    print(f"  Periods: 40d to 70d in steps of 1 (31 total)")
    print(f"  Original MA types: sma, ema, dema, tema, zlema, hull (6)")
    print(f"  Extended MA types: wma, trima, kama, median, gaussian, adaptive_ema (6)")
    print(f"  Total MA types: 12")
    print(f"  Total: {count_dpo_variants()} variants (31 × 12)")
