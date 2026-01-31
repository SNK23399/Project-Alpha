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

TEMA SHIFT DIVISORS (11 variants - broad exploratory range):
Explores optimal lag-shift alignment with full spectrum (1.0 to 3.0):
- shift = period / 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0 (11 shifts, step 0.2)
- Example: 60d period → shifts: 60, 50, 43, 37, 33, 30, 27, 25, 23, 21, 20

Windows: 10d through 100d (46 periods with 2-day granularity)
Total: 46 periods × 11 TEMA shift divisors = 506 variants

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

    For each DPO period (10d through 100d with 2-day step), computes TEMA with 11 shift divisors:

    TEMA Shift Divisors (11 variants spanning responsive to conservative):
    - tema__shift_1_0  (shift = period / 1.0: 60d → 60) - most responsive
    - tema__shift_1_2  (shift = period / 1.2: 60d → 50)
    - tema__shift_1_4  (shift = period / 1.4: 60d → 43)
    - ... [continues with 0.2 increments] ...
    - tema__shift_3_0  (shift = period / 3.0: 60d → 20) - most conservative

    Total: 46 periods × 11 TEMA shifts = 506 variants

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

    # DPO windows: 10d to 100d range with step of 2 days (46 periods)
    # Broad exploratory range for momentum cycle discovery
    # 10d:  ~2 weeks - lower bound (very short-term)
    # 100d: ~5 months - upper bound (longer-term)
    # 10-100d range with step 2 (46 periods) - comprehensive exploration
    dpo_periods = list(range(10, 101, 2))  # 10, 12, 14, ..., 100 (46 periods)

    # TEMA shift divisors to optimize lag-alignment
    # TEMA's lower lag (period/4-5) vs standard shift (period/2+1) creates misalignment
    # Multiple shifts let ensemble find optimal lag-alignment for each window
    # Broad range: 1.0 to 3.0 (responsive to conservative)
    # Step 0.2 increments (11 shifts total: 1.0, 1.2, 1.4, ..., 3.0)
    tema_shift_divisors = {
        f'tema__shift_{x:.1f}'.replace('.', '_'): lambda p, div=x: max(1, int(p / div))
        for x in np.arange(1.0, 3.1, 0.2)
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
    dpo_periods = list(range(10, 101, 2))  # 10 to 100 inclusive, step 2 (46 periods)
    tema_shifts = len(list(np.arange(1.0, 3.1, 0.2)))  # 1.0 to 3.0 step 0.2 (11 shifts)
    return len(dpo_periods) * tema_shifts


if __name__ == "__main__":
    print(f"DPO Enhanced Variants Library - Broad Exploratory Space")
    print(f"Total variants: {count_dpo_variants()}")
    print(f"  Periods: 10d to 100d in steps of 2 days (46 total)")
    print(f"  TEMA shift divisors: 1.0 to 3.0 in steps of 0.2 (11 total)")
    print(f"  Total: {count_dpo_variants()} variants (46 × 11)")
