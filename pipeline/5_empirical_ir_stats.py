"""
Step 5: Empirical Information Ratio Statistics (Deterministic)
===============================================================

DETERMINISTIC PIPELINE - Step 5 (Replaces MC Simulation)

Instead of Monte Carlo sampling, calculates empirical IR statistics
directly from feature_ir data using expanding historical windows.

For each feature at each test date:
1. Load feature_ir values from step 4 (n_dates, n_features, n_satellites)
2. Use expanding window (all data before test date)
3. Calculate:
   - Mean IR (average Information Ratio)
   - Std IR (uncertainty in IR)
   - Hit rate (% positive months)
4. Save as empirical priors (fully deterministic)

Key Benefits:
- No randomness (deterministic)
- No MC sampling overhead (10x faster)
- Transparent (can see exact statistics)
- Consistent (same input = same output)

Output:
    data/empirical_ir_stats_1month.npz
        ir_mean: (n_test_dates, n_signals) - mean IR
        ir_std: (n_test_dates, n_signals) - std IR
        hit_rate: (n_test_dates, n_signals) - % positive
        dates: test dates
        features: signal names
        n_satellites: [3, 4] ensemble sizes

Usage:
    python 5_empirical_ir_stats.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================
# CONFIGURATION
# ============================================================

HOLDING_MONTHS = 1
N_SATELLITES_TO_ANALYZE = [3, 4]
MIN_TRAINING_MONTHS = 12  # Require at least this much history

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_empirical_statistics(feature_ir_values):
    """
    Calculate empirical statistics from feature_ir values.

    Args:
        feature_ir_values: 1D array of IR values for a single signal

    Returns:
        mean_ir, std_ir, hit_rate
    """
    # Remove NaNs
    valid = feature_ir_values[~np.isnan(feature_ir_values)]

    if len(valid) == 0:
        return np.nan, np.nan, np.nan

    mean_ir = np.mean(valid)
    std_ir = np.std(valid) if len(valid) > 1 else 0.0
    hit_rate = np.mean(valid > 0)  # % positive

    return mean_ir, std_ir, hit_rate


def main():
    """Main entry point for empirical statistics calculation."""

    print("=" * 120)
    print("STEP 5: EMPIRICAL IR STATISTICS (DETERMINISTIC)")
    print("=" * 120)

    # Load feature_ir from step 4
    feature_ir_file = OUTPUT_DIR / 'feature_ir_1month.npz'

    if not feature_ir_file.exists():
        print(f"ERROR: {feature_ir_file} not found!")
        print("Please run step 4 first: python 4_precompute_feature_ir.py")
        return 1

    print(f"\nLoading feature_ir from {feature_ir_file.name}...")
    data = np.load(feature_ir_file, allow_pickle=True)

    feature_ir = data['feature_ir']  # (n_dates, n_signals, 10)
    dates = data['dates']
    features = data['features']

    n_dates, n_signals, max_n_satellites = feature_ir.shape

    print(f"  Loaded: {n_dates} dates × {n_signals} signals × {max_n_satellites} N values")
    print(f"  Date range: {dates[0]} to {dates[-1]}")

    # Determine test start index (require MIN_TRAINING_MONTHS of history)
    test_start_idx = MIN_TRAINING_MONTHS
    n_test_dates = n_dates - test_start_idx

    if n_test_dates <= 0:
        print(f"ERROR: Not enough data ({n_dates} dates) for {MIN_TRAINING_MONTHS} month minimum")
        return 1

    print(f"\n  Training period: {dates[0]} to {dates[test_start_idx-1]} ({test_start_idx} months)")
    print(f"  Test period: {dates[test_start_idx]} to {dates[-1]} ({n_test_dates} test dates)")

    # Calculate empirical statistics for each N
    results = {}

    for n_satellites in N_SATELLITES_TO_ANALYZE:
        print(f"\n  Processing N={n_satellites}...")

        if n_satellites > max_n_satellites:
            print(f"    Skipping N={n_satellites} (only have {max_n_satellites})")
            continue

        # Arrays for statistics
        ir_means = np.full((n_test_dates, n_signals), np.nan, dtype=np.float32)
        ir_stds = np.full((n_test_dates, n_signals), np.nan, dtype=np.float32)
        hit_rates = np.full((n_test_dates, n_signals), np.nan, dtype=np.float32)

        # For each test date, calculate empirical statistics
        for test_offset, test_idx in enumerate(tqdm(
            range(test_start_idx, n_dates),
            desc=f"    Calculating stats",
            unit="date",
            ncols=120,
            leave=False
        )):
            # For each signal, use expanding window (all data before test_idx)
            for feat_idx in range(n_signals):
                # Get historical feature_ir values up to this test date
                historical_irs = feature_ir[:test_idx, feat_idx, n_satellites - 1]

                # Calculate statistics
                mean_ir, std_ir, hit_rate = calculate_empirical_statistics(historical_irs)

                ir_means[test_offset, feat_idx] = mean_ir
                ir_stds[test_offset, feat_idx] = std_ir
                hit_rates[test_offset, feat_idx] = hit_rate

        results[n_satellites] = {
            'ir_mean': ir_means,
            'ir_std': ir_stds,
            'hit_rate': hit_rates
        }

        print(f"    Mean IR across signals: {np.nanmean(ir_means):.6f}")
        print(f"    Avg std IR: {np.nanmean(ir_stds):.6f}")
        print(f"    Avg hit rate: {np.nanmean(hit_rates):.1%}")

    # Save empirical statistics
    print(f"\nSaving empirical statistics...")

    output_file = OUTPUT_DIR / 'empirical_ir_stats_1month.npz'

    np.savez_compressed(
        output_file,
        **{
            f'ir_mean_N{n}': results[n]['ir_mean']
            for n in N_SATELLITES_TO_ANALYZE
            if n in results
        },
        **{
            f'ir_std_N{n}': results[n]['ir_std']
            for n in N_SATELLITES_TO_ANALYZE
            if n in results
        },
        **{
            f'hit_rate_N{n}': results[n]['hit_rate']
            for n in N_SATELLITES_TO_ANALYZE
            if n in results
        },
        test_dates=dates[test_start_idx:],
        features=features,
        n_satellites_analyzed=np.array(list(results.keys()))
    )

    print(f"  [SAVED] {output_file}")
    print(f"  Shape per N: ({n_test_dates}, {n_signals})")

    # Summary
    print("\n" + "=" * 120)
    print("EMPIRICAL STATISTICS CALCULATION COMPLETE")
    print("=" * 120)
    print(f"  Test dates: {n_test_dates}")
    print(f"  Signals: {n_signals}")
    print(f"  Ensemble sizes analyzed: {sorted(results.keys())}")
    print(f"  Deterministic: Yes (no randomness)")
    print("=" * 120)

    return 0


if __name__ == '__main__':
    start = time.time()
    exit_code = main()
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    sys.exit(exit_code)
