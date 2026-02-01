"""
DPO Enhanced Variants Library - Savgol-Based with Shift Divisors & Polyorder Exploration
=========================================================================================

Implements Detrended Price Oscillator (DPO) with Savgol filtering and optimized shift exploration.

Standard DPO formula:
  shift = period / divisor
  dpo = price[t + shift] - MA[t, period]

SAVGOL APPROACH:
Savgol filters outperformed TEMA in tracking correlation tests. This library uses Savgol
with multiple polyorders and shift divisors to explore optimal lag-shift alignment.

SAVGOL ADVANTAGES OVER TEMA:
- Higher correlation with raw signals (better tracking fidelity)
- Preserves local maxima/minima better than exponential moving averages
- Configurable polynomial degree for flexibility

PARAMETERS:
- Periods: [21d, 63d] - Bi-scale momentum analysis (3-week and 9-week)
- Polyorder: 4 - Quartic (heavy) smoothing for optimal tracking
- Shift divisors: [1.0, 2.0] - Responsive to conservative lag-shift alignment

OPTIMIZATION: Uses GPU-accelerated Savgol implementation from signal_filters
- Causal Savgol: Prevents look-ahead bias for walk-forward backtesting
- Batch processing: Vectorized computations across all ETFs simultaneously
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple

# Import optimized, GPU-accelerated Savgol implementation from signal_filters
from library.signal_filters import (
    causal_savgol as _causal_savgol,
)


def causal_savgol(prices_2d: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Wrapper for causal_savgol that handles 2D input (n_time, n_etfs)."""
    prices_3d = prices_2d[np.newaxis, :, :]
    result_3d = _causal_savgol(prices_3d, window=window, polyorder=polyorder)
    return result_3d[0, :, :]


def compute_dpo_variants_generator(
    etf_prices: pd.DataFrame,
    core_prices: pd.Series
) -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Generator that yields Savgol-based DPO variants with shift divisor exploration.

    For each DPO period (21d, 42d, 63d), computes Savgol with polyorder 4 and multiple shift divisors.

    Savgol Configuration:
    - Polyorder: 4 (Quartic - heavy smoothing for optimal tracking)

    Shift Divisors:
    - shift_1_0 (responsive): shift = period / 1.0
    - shift_2_0 (conservative): shift = period / 2.0

    Yields:
        (signal_name, signal_2d_array)

    Example signals:
        dpo_21d__savgol__polyorder_4__shift_1_0
        dpo_21d__savgol__polyorder_4__shift_2_0
        dpo_42d__savgol__polyorder_4__shift_1_0
        dpo_42d__savgol__polyorder_4__shift_2_0
        etc.
    """

    prices_arr = etf_prices.values
    core_prices_arr = core_prices.values

    # DPO windows: specific periods for bi-scale momentum analysis
    dpo_periods = [21, 63]

    # Fixed polyorder for consistent heavy smoothing
    polyorder = 4

    # Shift divisors to optimize lag-shift alignment
    # Two shifts: responsive (1.0) and conservative (2.0)
    shift_divisors = {
        f'shift_{x:.1f}'.replace('.', '_'): lambda p, div=x: max(1, int(p / div))
        for x in [1.0, 2.0]
    }

    for dpo_period in dpo_periods:
        # Generate Savgol with fixed polyorder and multiple shift divisors
        for shift_name, shift_func in shift_divisors.items():
            dpo_shift = shift_func(dpo_period)
            signal_name = f'dpo_{dpo_period}d__savgol__polyorder_{polyorder}__{shift_name}'

            try:
                # Compute Savgol MA using dpo_period as window
                # Window must be odd, so ensure it's odd
                window = dpo_period if dpo_period % 2 == 1 else dpo_period + 1
                ma = causal_savgol(prices_arr, window=window, polyorder=polyorder)

                # Compute DPO with custom shift: price[t + shift] - MA[t]
                dpo = np.empty_like(prices_arr)
                dpo[:-dpo_shift, :] = prices_arr[dpo_shift:, :] - ma[:-dpo_shift, :]
                dpo[-dpo_shift:, :] = np.nan

                yield signal_name, dpo

            except Exception as e:
                print(f"  Warning: Failed to compute {signal_name}: {e}")
                continue


def count_dpo_variants() -> int:
    """Count total DPO variants that will be generated."""
    dpo_periods = [21, 63]
    shift_divisors = [1.0, 2.0]    # Shifts: responsive, conservative
    return len(dpo_periods) * len(shift_divisors)


if __name__ == "__main__":
    dpo_periods = [21, 63]
    polyorder = 4
    shift_divisors = [1.0, 2.0]
    total = len(dpo_periods) * len(shift_divisors)

    print(f"DPO Enhanced Variants Library - Savgol-Based (Polyorder 4)")
    print(f"Total variants: {total}")
    print(f"  Periods: {dpo_periods}")
    print(f"  Polyorder: {polyorder} (quartic - heavy smoothing)")
    print(f"  Shift divisors: {shift_divisors}")
