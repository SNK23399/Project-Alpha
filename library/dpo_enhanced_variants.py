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

TEMA SHIFT DIVISORS (5 variants - conservative regime):
Explores optimal lag-shift alignment with conservative shifts (1.8 to 2.2):
- shift = period / 1.8, 1.9, 2.0, 2.1, 2.2 (5 shifts, step 0.1)
- Example: 60d period → shifts: 33, 32, 30, 29, 27

Windows: 50d through 70d (21 periods with 1-day granularity)
Total: 21 periods × 5 TEMA shift divisors = 105 variants

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
    Generator that yields TEMA-only DPO variants optimized for N=4 (longer-term regime).

    For each DPO period (50d through 70d with 1-day step), computes TEMA with 5 shift divisors:

    TEMA Shift Divisors (5 conservative variants):
    - tema__shift_1_8  (shift = period / 1.8: 60d → 33)
    - tema__shift_1_9  (shift = period / 1.9: 60d → 32)
    - tema__shift_2_0  (shift = period / 2.0: 60d → 30)
    - tema__shift_2_1  (shift = period / 2.1: 60d → 29)
    - tema__shift_2_2  (shift = period / 2.2: 60d → 27)

    Total: 21 periods × 5 TEMA shifts = 105 variants

    Yields:
        (signal_name, signal_2d_array)

    Example signals:
        dpo_50d__tema__shift_1_8
        dpo_60d__tema__shift_2_0
        dpo_70d__tema__shift_2_2
        etc.
    """

    prices_arr = etf_prices.values
    core_prices_arr = core_prices.values

    # DPO windows: 50d to 70d range with step of 1 day (21 periods)
    # Optimized for longer-term momentum (N=4 regime)
    # 50d:  ~2.5 months - lower bound
    # 70d:  ~3.5 months - upper bound
    # 50-70d range with step 1 (21 periods) - concentrated on proven range
    dpo_periods = list(range(50, 71))  # 50, 51, 52, ..., 70 (21 periods)

    # TEMA shift divisors to optimize lag-alignment
    # TEMA's lower lag (period/4-5) vs standard shift (period/2+1) creates misalignment
    # Multiple shifts let ensemble find optimal lag-alignment for each window
    # Conservative shifts: 1.8 to 2.2 (larger shifts for stability)
    # Step 0.1 increments (5 shifts total: 1.8, 1.9, 2.0, 2.1, 2.2)
    tema_shift_divisors = {
        f'tema__shift_{x:.1f}'.replace('.', '_'): lambda p, div=x: max(1, int(p / div))
        for x in np.arange(1.8, 2.3, 0.1)
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
    dpo_periods = list(range(50, 71))  # 50 to 70 inclusive, step 1 (21 periods)
    tema_shifts = len(list(np.arange(1.8, 2.3, 0.1)))  # 1.8 to 2.2 step 0.1 (5 shifts)
    return len(dpo_periods) * tema_shifts


if __name__ == "__main__":
    print(f"DPO Enhanced Variants Library - TEMA-Optimized for N=4 Regime")
    print(f"Total variants: {count_dpo_variants()}")
    print(f"  Periods: 50d to 70d in steps of 1 day (21 total)")
    print(f"  TEMA shift divisors: 1.8 to 2.2 in steps of 0.1 (5 total)")
    print(f"  Total: {count_dpo_variants()} variants (21 × 5)")
