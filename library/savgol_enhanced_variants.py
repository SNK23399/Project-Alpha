"""
Savitzky-Golay Enhanced Variants Library - Window & Polynomial Order Optimization
====================================================================================

Implements Savitzky-Golay causal smoothing with optimized window and polynomial order parameters.

ENSEMBLE-VALIDATED APPROACH:
After testing 19 filter types (EMA, median, Kalman, Butterworth, etc.),
ensemble optimization showed Savitzky-Golay dominates (45-57% selection at N=3-5).
This library uses ONLY Savitzky-Golay with multiple window/polyorder combinations.

SAVITZKY-GOLAY - THE CHAMPION:
- Polynomial fit-based smoother that preserves local peaks/troughs
- Minimal lag (only looks at past data - fully causal)
- 45-57% selection rate in ensemble backtests (best-performing ensemble sizes N=4-5)
- Outperforms moving averages due to better edge preservation

SAVITZKY-GOLAY PARAMETER VARIANTS:

Window Sizes (odd numbers, trading days):
- savgol_15d: Small window (1.5 weeks) - responsive, more noise
- savgol_17d:
- savgol_19d:
- savgol_21d: Standard (1 month) - balanced, MOST SELECTED in backtests
- savgol_23d:
- savgol_25d:
- savgol_27d: Large window (1.4 weeks) - smooth, less responsive
- savgol_31d: Very smooth (1.5 months)
- savgol_35d: Aggressive smoothing (1.75 months)

Polynomial Order:
- polyorder=2: Quadratic fit - EMPIRICALLY OPTIMAL (83-87% selection rate in ensembles)
- Cubic (polyorder=3) eliminated: Shows overfitting tendency, polyorder_2 consistently outperforms

Windows: 15, 17, 19, 21, 23, 25, 27, 31, 35 (9 windows)
Polynomial Order: 2 ONLY (1 variant - optimized based on backtest analysis)
Total: 9 windows × 1 polyorder = 9 variants

OPTIMIZATION: Uses GPU-accelerated Savitzky-Golay from signal_filters
- Causal filtering: No look-ahead bias, only uses past data
- GPU acceleration: CuPy convolution when available, scipy fallback
- Batch processing: Vectorized across all ETFs simultaneously
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple

# Import optimized, GPU-accelerated Savitzky-Golay implementation from signal_filters
from library.signal_filters import (
    causal_savgol as _causal_savgol,
)


def causal_savgol(prices_2d: np.ndarray, window: int = 21, polyorder: int = 2) -> np.ndarray:
    """
    Wrapper for causal_savgol that handles 2D input (n_time, n_etfs).

    Args:
        prices_2d: (n_time, n_etfs) price array
        window: Window length (odd number, trading days)
        polyorder: Polynomial order (DEFAULT: 2 - quadratic, empirically optimal)

    Returns:
        (n_time, n_etfs) smoothed array
    """
    prices_3d = prices_2d[np.newaxis, :, :]
    result_3d = _causal_savgol(prices_3d, window=window, polyorder=polyorder)
    return result_3d[0, :, :]


def compute_savgol_variants_generator(
    etf_prices: pd.DataFrame,
    core_prices: pd.Series
) -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Generator that yields Savitzky-Golay variants with window exploration only.

    For each window size, computes Savitzky-Golay with FIXED polyorder=2 (empirically optimal).

    SAVITZKY-GOLAY PARAMETERS (9 total variants):

    Windows (9 variants, odd trading days):
    - savgol_15d  (1.5 weeks - most responsive)
    - savgol_17d
    - savgol_19d
    - savgol_21d  (1 month - MOST SELECTED in backtests)
    - savgol_23d
    - savgol_25d
    - savgol_27d  (1.35 weeks)
    - savgol_31d
    - savgol_35d  (1.75 months - most smooth)

    Polynomial Order:
    - polyorder=2  (FIXED - quadratic polynomial, 83-87% selection rate in ensemble backtests)
    - polyorder=3 ELIMINATED: Cubic overfits, polyorder_2 consistently superior

    Total: 9 windows × 1 polyorder = 9 variants

    Yields:
        (signal_name, signal_2d_array)

    Example signals:
        savgol_15d__polyorder_2
        savgol_21d__polyorder_2  (MOST COMMON)
        savgol_35d__polyorder_2
        etc.
    """

    prices_arr = etf_prices.values

    # Window sizes: odd numbers, centered around 21d which is most common in backtests
    # 15d:  1.5 weeks - very responsive
    # 17d:  mid-point
    # 19d:  mid-point
    # 21d:  ~1 month - STANDARD, most selected in ensemble (35% for N=3)
    # 23d:  mid-point
    # 25d:  mid-point
    # 27d:  1.35 weeks
    # 31d:  1.55 weeks - smoother
    # 35d:  1.75 weeks - very smooth, less responsive
    savgol_windows = [15, 17, 19, 21, 23, 25, 27, 31, 35]

    # FIXED polynomial order: polyorder=2 only
    # Backtest analysis showed polyorder=2 selected 83-87% of the time across all ensemble sizes
    # polyorder=3 (cubic) was eliminated as it showed overfitting and worse generalization
    polyorders = [2]

    for window in savgol_windows:
        for polyorder in polyorders:
            signal_name = f'savgol_{window}d__polyorder_{polyorder}'

            try:
                # Apply Savitzky-Golay filter
                filtered = causal_savgol(prices_arr, window=window, polyorder=polyorder)
                yield signal_name, filtered

            except Exception as e:
                print(f"  Warning: Failed to compute {signal_name}: {e}")
                continue


def count_savgol_variants() -> int:
    """Count total Savitzky-Golay variants that will be generated."""
    savgol_windows = [15, 17, 19, 21, 23, 25, 27, 31, 35]  # 9 windows
    polyorders = [2]  # 1 polynomial order (polyorder_2 ONLY)
    return len(savgol_windows) * len(polyorders)


if __name__ == "__main__":
    print(f"Savitzky-Golay Enhanced Variants Library - Window Optimization (polyorder_2 LOCKED)")
    print(f"Total variants: {count_savgol_variants()}")
    print(f"  Windows: 15d, 17d, 19d, 21d, 23d, 25d, 27d, 31d, 35d (9 total)")
    print(f"  Polynomial order: 2 (quadratic) ONLY (1 total)")
    print(f"  Total: {count_savgol_variants()} variants (9 × 1)")
    print()
    print(f"Empirical validation: polyorder_2 selected 83-87% of the time across")
    print(f"all ensemble sizes (N=1-5). Cubic fitting (polyorder_3) eliminated as suboptimal.")
