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

TEMA SHIFT DIVISORS (3 variants - ensemble-validated):
Explores optimal lag-shift alignment by varying the shift divisor:
- shift = period / 1.0  (smallest shift: 45d → 45, 60d → 60) - 77.8% usage
- shift = period / 1.5  (45d → 30, 60d → 40) - 20.9% usage
- shift = period / 2.0  (45d → 23, 60d → 30) - 1.3% usage
[Removed 2.5, 3.0, 3.5, 4.0 - not used in ensemble selection]

Windows: 30d through 69d (14 periods)
Total: 14 periods × 3 TEMA shift divisors = 42 variants

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

    For each DPO period (30d through 69d in steps of 3d), computes TEMA with 3 shift divisors:

    TEMA Shift Divisors (3 variants - ensemble-validated only):
    - tema__shift_1_0  (shift = period / 1.0, smallest shift: 45d → 45) - 77.8% usage
    - tema__shift_1_5  (shift = period / 1.5: 45d → 30) - 20.9% usage
    - tema__shift_2_0  (shift = period / 2.0: 45d → 23) - 1.3% usage

    Total: 14 periods × 3 TEMA shifts = 42 variants

    Yields:
        (signal_name, signal_2d_array)

    Example signals:
        dpo_30d__tema__shift_1_0
        dpo_45d__tema__shift_2_0
        dpo_60d__tema__shift_1_5
        dpo_69d__tema__shift_1_0
        etc.
    """

    prices_arr = etf_prices.values
    core_prices_arr = core_prices.values

    # DPO windows: 21d to 70d range with step of 3 (17 periods)
    # Backtest analysis showed all DPO > 69d are never selected
    # 21d:  ~1 month - lower bound (for faster momentum)
    # 33d:  ~1.65 months - most frequently selected in backtest
    # 36d:  ~1.8 months - second most frequent
    # 69d:  ~3.45 months - upper bound (last used value)
    # 21-70d range with step of 3 (17 periods) - optimized based on backtest analysis
    dpo_periods = list(range(30, 72, 3))  # 30, 33, 36, ..., 69 (14 periods)

    # TEMA shift divisors to optimize lag-alignment
    # TEMA's lower lag (period/4-5) vs standard shift (period/2+1) creates misalignment
    # Multiple shifts let ensemble find optimal lag-alignment for each window
    # Shifts range from 1.0 (maximum forward lookahead) to 2.0 (conservative)
    # Ensemble analysis shows only 1.0, 1.5, 2.0 are used in practice
    tema_shift_divisors = {
        'tema__shift_1_0': lambda p: max(1, int(p / 1.0)),
        'tema__shift_1_5': lambda p: max(1, int(p / 1.5)),
        'tema__shift_2_0': lambda p: max(1, int(p / 2.0)),
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
    dpo_periods = list(range(30, 72, 3))  # 30 to 69 inclusive, step 3 (14 periods)
    tema_shifts = 3       # shift_1_0, shift_1_5, shift_2_0 (ensemble-validated only)
    return len(dpo_periods) * tema_shifts


if __name__ == "__main__":
    print(f"DPO Enhanced Variants Library - TEMA-Optimized with Shift Divisors")
    print(f"Total variants: {count_dpo_variants()}")
    print(f"  Periods: 30d to 69d in steps of 3 (14 total)")
    print(f"  TEMA shift divisors: 1.0, 1.5, 2.0 (3 ensemble-validated)")
    print(f"  Total: {count_dpo_variants()} variants (14 × 3)")
