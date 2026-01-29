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

TEMA SHIFT DIVISORS (7 variants):
Explores optimal lag-shift alignment by varying the shift divisor:
- shift = period / 1.1  (smallest shift: 45d → 41, 60d → 55)
- shift = period / 1.2  (45d → 38, 60d → 50)
- shift = period / 1.3  (45d → 35, 60d → 46)
- shift = period / 1.4  (45d → 32, 60d → 43)
- shift = period / 1.5  (45d → 30, 60d → 40) - previously "conservative"
- shift = period / 1.6  (45d → 28, 60d → 38)
- shift = period / 1.7  (largest shift: 45d → 26, 60d → 35)

Windows: 30d through 50d (21 periods)
Total: 21 periods × 7 TEMA shift divisors = 147 variants

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
    Generator that yields TEMA-only DPO variants with shift divisor exploration.

    For each DPO period (30d through 70d in steps of 1d), computes TEMA with 7 shift divisors:

    TEMA Shift Divisors (7 variants to optimize lag-shift alignment):
    - tema__shift_1_1  (shift = period / 1.1, smallest shift: 45d → 41)
    - tema__shift_1_2  (shift = period / 1.2: 45d → 38)
    - tema__shift_1_3  (shift = period / 1.3: 45d → 35)
    - tema__shift_1_4  (shift = period / 1.4: 45d → 32)
    - tema__shift_1_5  (shift = period / 1.5: 45d → 30, previously "conservative")
    - tema__shift_1_6  (shift = period / 1.6: 45d → 28)
    - tema__shift_1_7  (shift = period / 1.7, largest shift: 45d → 26)

    Total: 41 periods × 7 TEMA shifts = 287 variants

    Yields:
        (signal_name, signal_2d_array)

    Example signals:
        dpo_40d__tema__shift_1_1
        dpo_45d__tema__shift_1_5
        dpo_60d__tema__shift_1_7
        dpo_70d__tema__shift_1_3
        etc.
    """

    prices_arr = etf_prices.values
    core_prices_arr = core_prices.values

    # DPO windows: optimized range from 30d to 50d
    # Backtest analysis showed 33-41d is the sweet spot (where most selections occur)
    # 30d:  ~1.5 months - lower bound
    # 35d:  ~1.75 months - frequently selected
    # 40d:  ~2 months - frequently selected
    # 45d:  ~2.25 months - historically good
    # 50d:  ~2.5 months - upper bound
    # Removed 51-70d (larger windows showed worse performance and clutter)
    dpo_periods = list(range(30, 51))  # 30, 31, 32, ..., 50 (21 periods)

    # TEMA shift divisors to optimize lag-alignment
    # TEMA's lower lag (period/4-5) vs standard shift (period/2+1) creates misalignment
    # Multiple shifts let ensemble find optimal lag-alignment for each window
    # Shifts range from 1.1 (small shift) to 1.7 (large shift)
    tema_shift_divisors = {
        'tema__shift_1_1': lambda p: max(1, int(p / 1.1)),
        'tema__shift_1_2': lambda p: max(1, int(p / 1.2)),
        'tema__shift_1_3': lambda p: max(1, int(p / 1.3)),
        'tema__shift_1_4': lambda p: max(1, int(p / 1.4)),
        'tema__shift_1_5': lambda p: max(1, int(p / 1.5)),
        'tema__shift_1_6': lambda p: max(1, int(p / 1.6)),
        'tema__shift_1_7': lambda p: max(1, int(p / 1.7)),
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
    dpo_periods = list(range(30, 51))  # 30 to 50 inclusive (21 periods)
    tema_shifts = 7        # shift_1_1, shift_1_2, shift_1_3, shift_1_4, shift_1_5, shift_1_6, shift_1_7
    return len(dpo_periods) * tema_shifts


if __name__ == "__main__":
    print(f"DPO Enhanced Variants Library - TEMA-Optimized with Shift Divisors")
    print(f"Total variants: {count_dpo_variants()}")
    print(f"  Periods: 30d to 50d in steps of 1 (21 total)")
    print(f"  TEMA shift divisors: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7 (7)")
    print(f"  Total: {count_dpo_variants()} variants (21 × 7)")
