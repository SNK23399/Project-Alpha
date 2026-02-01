"""
DPO Enhanced Variants Library - TEMA-Only with Shift Divisors
=============================================================

Implements Detrended Price Oscillator (DPO) with TEMA and optimized shift exploration.

Standard DPO formula:
  shift = period // 2 + 1
  dpo = price[t + shift] - MA[t, period]

ENSEMBLE-VALIDATED APPROACH:
After testing 12 MA types, ensemble optimization showed TEMA dominates (94-100% selection).
This library uses ONLY TEMA with multiple shift divisors to explore optimal lag-shift alignment.

TEMA - THE CHAMPION:
- Triple Exponential Moving Average (period/4-5 lag)
- Optimal balance of responsiveness and smoothness
- 98-100% selection rate in ensemble backtests

TEMA SHIFT DIVISORS:
Explores optimal lag-shift alignment with full spectrum (responsive to conservative)
- Example: 60d period → shifts range from 60 (1.0 divisor) to 20 (3.0 divisor)

Window selection strategy: Broad range with step spacing to explore momentum cycles
at multiple timeframes (2-week minimum through 6-month maximum)

OPTIMIZATION: Uses GPU-accelerated TEMA implementation from signal_filters
- TEMA: GPU-accelerated with scipy.signal optimizations
- Batch processing: Vectorized computations across all ETFs simultaneously
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple

# Import optimized, GPU-accelerated TEMA implementation from signal_filters
from library.signal_filters import (
    causal_tema as _causal_tema,
)


def causal_tema(prices_2d: np.ndarray, span: int) -> np.ndarray:
    """Wrapper for causal_tema that handles 2D input (n_time, n_etfs)."""
    prices_3d = prices_2d[np.newaxis, :, :]
    result_3d = _causal_tema(prices_3d, span)
    return result_3d[0, :, :]


def compute_dpo_variants_generator(
    etf_prices: pd.DataFrame,
    core_prices: pd.Series
) -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Generator that yields TEMA-only DPO variants with broad exploratory scope.

    For each DPO period (21d, 42d, 63d), computes TEMA with 2 shift divisors.

    TEMA Shift Divisors:
    - tema__shift_1_0 (responsive): shift = period / 1.0
    - tema__shift_2_0 (conservative): shift = period / 2.0

    Yields:
        (signal_name, signal_2d_array)

    Example signals:
        dpo_10d__tema__shift_1_0
        dpo_60d__tema__shift_2_0
        dpo_100d__tema__shift_3_0
        etc.
    """

    prices_arr = etf_prices.values
    core_prices_arr = core_prices.values

    # DPO windows: specific periods for multi-scale momentum analysis
    dpo_periods = [21, 42, 63]

    # TEMA shift divisors to optimize lag-alignment
    # TEMA's lower lag (period/4-5) vs standard shift (period/2+1) creates misalignment
    # Two shifts: responsive (1.0) and conservative (2.0)
    tema_shift_divisors = {
        f'tema__shift_{x:.1f}'.replace('.', '_'): lambda p, div=x: max(1, int(p / div))
        for x in [1.0, 2.0]
    }

    for dpo_period in dpo_periods:
        # Generate TEMA with multiple shift divisors
        for shift_name, shift_func in tema_shift_divisors.items():
            dpo_shift = shift_func(dpo_period)
            signal_name = f'dpo_{dpo_period}d__{shift_name}'

            try:
                # Compute TEMA
                ma = causal_tema(prices_arr, dpo_period)

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
    dpo_periods = [21, 42, 63]
    tema_shifts = 2  # Shifts: 1.0 and 2.0
    return len(dpo_periods) * tema_shifts


if __name__ == "__main__":
    dpo_periods = [21, 42, 63]
    tema_shifts = [1.0, 2.0]
    total = len(dpo_periods) * len(tema_shifts)

    print(f"DPO Enhanced Variants Library")
    print(f"Total variants: {total}")
    print(f"  Periods: {dpo_periods}")
    print(f"  TEMA shift divisors: {tema_shifts}")
