"""
Analyze Parameter Selection Frequency from Backtest Results
============================================================

Extracts selected features from backtest results and counts how often
each parameter (DPO period, TEMA shift, Savgol window) gets selected.

Shows ALL possible values (used and unused).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Backtest results directory
DATA_DIR = Path(__file__).parent / 'data' / 'backtest_results'


def parse_feature_name(feature_name: str) -> dict:
    """Parse feature name into components.

    Example: dpo_30d__tema__shift_1_0__savgol_15d_polyorder_2
    Returns: {'dpo_period': 30, 'tema_shift': '1_0', 'savgol_window': 15}
    """
    parts = feature_name.split('__')

    result = {}

    # DPO period (e.g., "dpo_30d")
    if len(parts) > 0 and parts[0].startswith('dpo_'):
        dpo_str = parts[0].replace('dpo_', '').replace('d', '')
        try:
            result['dpo_period'] = int(dpo_str)
        except:
            pass

    # TEMA shift (e.g., "tema", "shift_1_0" -> convert to "1.0")
    if len(parts) >= 3 and parts[1] == 'tema' and parts[2].startswith('shift_'):
        shift_str = parts[2].replace('shift_', '')  # "1_0" or "1_5" or "2_0"
        # Convert underscore format to decimal: "1_0" -> "1.0", "1_5" -> "1.5", etc
        shift_decimal = shift_str.replace('_', '.')
        result['tema_shift'] = shift_decimal

    # Savgol window (e.g., "savgol_15d_polyorder_2")
    if len(parts) >= 4 and parts[3].startswith('savgol_'):
        # Extract number before 'd_polyorder'
        savgol_part = parts[3].replace('savgol_', '')  # "15d_polyorder_2"
        window_str = savgol_part.split('d_')[0]  # "15"
        try:
            result['savgol_window'] = int(window_str)
        except:
            pass

    return result


def analyze_parameters(results_df: pd.DataFrame) -> dict:
    """Count parameter selections from feature names."""

    dpo_counts = defaultdict(int)
    tema_counts = defaultdict(float)  # Use float to match decimal format
    savgol_counts = defaultdict(int)

    for _, row in results_df.iterrows():
        if pd.isna(row['selected_features']):
            continue

        features = row['selected_features'].split('|')

        for feature in features:
            feature = feature.strip()
            params = parse_feature_name(feature)

            if 'dpo_period' in params:
                dpo_counts[params['dpo_period']] += 1
            if 'tema_shift' in params:
                # Convert to float for matching
                try:
                    shift_key = float(params['tema_shift'])
                    tema_counts[shift_key] += 1
                except:
                    pass
            if 'savgol_window' in params:
                savgol_counts[params['savgol_window']] += 1

    return {
        'dpo': dict(dpo_counts),
        'tema': dict(tema_counts),
        'savgol': dict(savgol_counts)
    }


def main():
    """Analyze parameter selection and print tables."""

    print("=" * 120)
    print("PARAMETER SELECTION FREQUENCY ANALYSIS")
    print("=" * 120)

    # Load backtest results
    n3_file = DATA_DIR / 'bayesian_backtest_N3.csv'
    n4_file = DATA_DIR / 'bayesian_backtest_N4.csv'

    if not n3_file.exists() or not n4_file.exists():
        print(f"ERROR: Backtest results not found!")
        print(f"  Expected: {n3_file}")
        print(f"  Expected: {n4_file}")
        return 1

    df_n3 = pd.read_csv(n3_file)
    df_n4 = pd.read_csv(n4_file)

    print(f"\nLoaded N=3 results: {len(df_n3)} months")
    print(f"Loaded N=4 results: {len(df_n4)} months")

    # Analyze each
    params_n3 = analyze_parameters(df_n3)
    params_n4 = analyze_parameters(df_n4)

    # Define all possible parameters (fine granularity)
    dpo_periods = list(range(30, 70))  # 30, 31, 32, ..., 69 (40 total, step 1)
    tema_shifts = [f'{x:.1f}' for x in np.arange(1.0, 2.1, 0.1)]  # 1.0, 1.1, 1.2, ..., 2.0 (11 total, step 0.1)
    savgol_windows = list(range(15, 55))  # 15, 16, 17, ..., 54 (40 total, step 1)

    # ========================================================================
    # DPO PERIODS TABLE
    # ========================================================================
    print("\n" + "=" * 120)
    print("DPO PERIODS (days)")
    print("=" * 120)
    print(f"{'Period':>10} {'N=3':>8} {'N=4':>8} {'Total':>8}")
    print("-" * 120)

    for period in dpo_periods:
        count_n3 = params_n3['dpo'].get(period, 0)
        count_n4 = params_n4['dpo'].get(period, 0)
        total = count_n3 + count_n4

        used = "✓" if total > 0 else " "
        print(f"{period:>9}d {count_n3:>8} {count_n4:>8} {total:>8}")

    # Summary
    print("-" * 120)
    used_count = sum(1 for p in dpo_periods if params_n3['dpo'].get(p, 0) + params_n4['dpo'].get(p, 0) > 0)
    print(f"{'SUMMARY':>10} - Used: {used_count}/{len(dpo_periods)}, Unused: {len(dpo_periods) - used_count}/{len(dpo_periods)}")

    # ========================================================================
    # TEMA SHIFTS TABLE
    # ========================================================================
    print("\n" + "=" * 120)
    print("TEMA SHIFTS (divisors, range order, step 0.1)")
    print("=" * 120)
    print(f"{'Shift':>10} {'N=3':>8} {'N=4':>8} {'Total':>8}")
    print("-" * 120)

    for shift_str in tema_shifts:
        shift_val = float(shift_str)
        count_n3 = int(params_n3['tema'].get(shift_val, 0))
        count_n4 = int(params_n4['tema'].get(shift_val, 0))
        total = count_n3 + count_n4

        print(f"{shift_str:>10} {count_n3:>8} {count_n4:>8} {total:>8}")

    # Summary
    print("-" * 120)
    used_count = sum(1 for s_str in tema_shifts if int(params_n3['tema'].get(float(s_str), 0)) + int(params_n4['tema'].get(float(s_str), 0)) > 0)
    print(f"{'SUMMARY':>10} - Used: {used_count}/{len(tema_shifts)}, Unused: {len(tema_shifts) - used_count}/{len(tema_shifts)}")

    # ========================================================================
    # SAVGOL WINDOWS TABLE
    # ========================================================================
    print("\n" + "=" * 120)
    print("SAVGOL WINDOWS (days)")
    print("=" * 120)
    print(f"{'Window':>10} {'N=3':>8} {'N=4':>8} {'Total':>8}")
    print("-" * 120)

    for window in savgol_windows:
        count_n3 = params_n3['savgol'].get(window, 0)
        count_n4 = params_n4['savgol'].get(window, 0)
        total = count_n3 + count_n4

        print(f"{window:>9}d {count_n3:>8} {count_n4:>8} {total:>8}")

    # Summary
    print("-" * 120)
    used_count = sum(1 for w in savgol_windows if params_n3['savgol'].get(w, 0) + params_n4['savgol'].get(w, 0) > 0)
    print(f"{'SUMMARY':>10} - Used: {used_count}/{len(savgol_windows)}, Unused: {len(savgol_windows) - used_count}/{len(savgol_windows)}")

    print("\n" + "=" * 120)

    return 0


if __name__ == '__main__':
    exit(main())
