"""
Analyze Parameter Selection Frequency from Backtest Results
============================================================

Extracts selected features from backtest results and counts how often
each parameter (DPO period, TEMA shift, Savgol window) gets selected.

Shows ALL possible values (used and unused).

Automatically extracts parameter ranges from the actual pipeline code.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

# Backtest results directory
DATA_DIR = Path(__file__).parent / 'data' / 'backtest_results'

# Add library path to import DPO variants
LIB_DIR = Path(__file__).parent.parent / 'library'
sys.path.insert(0, str(LIB_DIR))


# ============================================================
# PARAMETER EXTRACTION FROM ACTUAL PIPELINE CODE
# ============================================================

def get_dpo_periods() -> list:
    """Extract DPO periods from dpo_enhanced_variants.py (handles split ranges and hardcoded lists)"""
    try:
        dpo_file = LIB_DIR / 'dpo_enhanced_variants.py'
        if dpo_file.exists():
            with open(dpo_file, 'r') as f:
                content = f.read()
            import re

            # First try to match split ranges with + operator (range objects)
            matches = re.findall(r'list\(range\((\d+),\s*(\d+)(?:,\s*(\d+))?\)\)', content)

            if matches:
                periods = []
                for match in matches:
                    start, end = int(match[0]), int(match[1])
                    step = int(match[2]) if match[2] else 1
                    periods.extend(list(range(start, end, step)))

                # Remove duplicates while preserving order
                seen = set()
                result = []
                for p in periods:
                    if p not in seen:
                        result.append(p)
                        seen.add(p)

                return sorted(result)

            # If no ranges found, try to match hardcoded list like [31, 62]
            list_match = re.search(r'dpo_periods\s*=\s*\[([^\]]+)\]', content)
            if list_match:
                list_content = list_match.group(1)
                # Extract all numbers from the list
                numbers = re.findall(r'\d+', list_content)
                if numbers:
                    return sorted([int(n) for n in numbers])

    except Exception as e:
        print(f"  [DEBUG] DPO extraction error: {e}")

    return None


def get_tema_shifts() -> list:
    """Extract TEMA shift divisors from dpo_enhanced_variants.py (handles np.arange or hardcoded lists)"""
    try:
        dpo_file = LIB_DIR / 'dpo_enhanced_variants.py'
        if dpo_file.exists():
            with open(dpo_file, 'r') as f:
                content = f.read()
            import re

            # First try to match np.arange(...) pattern
            match = re.search(r'for x in np\.arange\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)', content)
            if match:
                start, end, step = float(match.group(1)), float(match.group(2)), float(match.group(3))
                shifts = [f'{x:.1f}' for x in np.arange(start, end, step)]
                return shifts

            # If no np.arange found, try to match hardcoded list like [1.0, 2.0]
            list_match = re.search(r'for x in (\[[\d.,\s]+\])', content)
            if list_match:
                list_str = list_match.group(1)
                # Extract all numbers from the list
                numbers = re.findall(r'[\d.]+', list_str)
                if numbers:
                    return [f'{float(n):.1f}' for n in numbers]
    except Exception as e:
        print(f"  [DEBUG] TEMA extraction error: {e}")

    return None


def get_dpo_shifts() -> list:
    """Extract DPO shift divisors from dpo_enhanced_variants.py (new Savgol-based format)"""
    try:
        dpo_file = LIB_DIR / 'dpo_enhanced_variants.py'
        if dpo_file.exists():
            with open(dpo_file, 'r') as f:
                content = f.read()
            import re

            # Match hardcoded list like [1.0, 2.0] in shift_divisors
            list_match = re.search(r'for x in (\[[\d.,\s]+\])', content)
            if list_match:
                list_str = list_match.group(1)
                numbers = re.findall(r'[\d.]+', list_str)
                if numbers:
                    return sorted([f'{float(n):.1f}' for n in numbers])
    except Exception as e:
        print(f"  [DEBUG] DPO shift extraction error: {e}")

    return None


def get_dpo_polyorder() -> int:
    """Extract DPO polyorder from dpo_enhanced_variants.py (new Savgol-based format)"""
    try:
        dpo_file = LIB_DIR / 'dpo_enhanced_variants.py'
        if dpo_file.exists():
            with open(dpo_file, 'r') as f:
                content = f.read()
            import re

            # Match polyorder assignment like: polyorder = 4
            match = re.search(r'polyorder\s*=\s*(\d+)', content)
            if match:
                return int(match.group(1))
    except Exception as e:
        print(f"  [DEBUG] DPO polyorder extraction error: {e}")

    return None


def get_savgol_windows() -> list:
    """Extract Savgol windows from 3_apply_filters.py (handles split ranges, + operator, or hardcoded lists)"""
    try:
        filters_file = Path(__file__).parent / '3_apply_filters.py'
        if filters_file.exists():
            with open(filters_file, 'r') as f:
                content = f.read()
            import re

            # First try to match split ranges with + operator
            matches = re.findall(r'list\(range\((\d+),\s*(\d+)(?:,\s*(\d+))?\)\)', content)

            if matches:
                windows = []
                for match in matches:
                    start, end = int(match[0]), int(match[1])
                    step = int(match[2]) if match[2] else 1
                    windows.extend(list(range(start, end, step)))

                # Remove duplicates while preserving order
                seen = set()
                result = []
                for w in windows:
                    if w not in seen:
                        result.append(w)
                        seen.add(w)

                return sorted(result)

            # If no ranges found, try to match hardcoded list like [7, 14, 21, 28]
            list_match = re.search(r'savgol_windows\s*=\s*(\[[\d.,\s]+\])', content)
            if list_match:
                list_str = list_match.group(1)
                # Extract all numbers from the list
                numbers = re.findall(r'\d+', list_str)
                if numbers:
                    return sorted([int(n) for n in numbers])
    except Exception as e:
        print(f"  [DEBUG] Savgol extraction error: {e}")

    return None


def parse_feature_name(feature_name: str) -> dict:
    """Parse feature name into components.

    New DPO format with post-processing (Savgol-based):
    Example: dpo_21d__savgol__polyorder_4__shift_1_0__savgol_6d_polyorder_4
    Returns: {'dpo_period': 21, 'polyorder': 4, 'shift': '1.0', 'savgol_window': 6}

    New DPO format without post-processing (base signal):
    Example: dpo_21d__savgol__polyorder_4__shift_1_0
    Returns: {'dpo_period': 21, 'polyorder': 4, 'shift': '1.0'}

    Old DPO format (TEMA-based, for backward compatibility):
    Example: dpo_30d__tema__shift_1_0__savgol_15d_polyorder_2
    Returns: {'dpo_period': 30, 'tema_shift': '1.0', 'savgol_window': 15, 'polyorder': 2}
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

    # Check for new Savgol-based DPO format: dpo_Xd__savgol__polyorder_Y__shift_Z_Z[__savgol_Wd_polyorder_P]
    if len(parts) >= 4 and parts[1] == 'savgol':
        # Extract polyorder from DPO (e.g., "polyorder_4" -> 4)
        if parts[2].startswith('polyorder_'):
            polyorder_str = parts[2].replace('polyorder_', '')
            try:
                result['polyorder'] = int(polyorder_str)
            except:
                pass

        # Extract shift (e.g., "shift_1_0" -> "1.0")
        if parts[3].startswith('shift_'):
            shift_str = parts[3].replace('shift_', '')  # "1_0" or "2_0"
            shift_decimal = shift_str.replace('_', '.')  # "1.0" or "2.0"
            result['shift'] = shift_decimal

        # Check for post-processing Savgol filter (e.g., "savgol_6d_polyorder_4")
        if len(parts) >= 5 and parts[4].startswith('savgol_'):
            savgol_part = parts[4].replace('savgol_', '')  # "6d_polyorder_4"
            window_str = savgol_part.split('d_')[0]  # "6"
            try:
                result['savgol_window'] = int(window_str)
            except:
                pass

            # Extract post-processing polyorder if different from DPO polyorder
            if 'polyorder_' in savgol_part:
                polyorder_str = savgol_part.split('polyorder_')[1]  # "4"
                try:
                    result['savgol_polyorder'] = int(polyorder_str)
                except:
                    pass

    # Check for old TEMA-based DPO format (backward compatibility)
    elif len(parts) >= 3 and parts[1] == 'tema' and parts[2].startswith('shift_'):
        shift_str = parts[2].replace('shift_', '')  # "1_0" or "1_5" or "2_0"
        shift_decimal = shift_str.replace('_', '.')
        result['tema_shift'] = shift_decimal

        # Old format also has Savgol filter applied afterward
        if len(parts) >= 4 and parts[3].startswith('savgol_'):
            savgol_part = parts[3].replace('savgol_', '')  # "15d_polyorder_2"
            window_str = savgol_part.split('d_')[0]  # "15"
            try:
                result['savgol_window'] = int(window_str)
            except:
                pass

            if 'polyorder_' in savgol_part:
                polyorder_str = savgol_part.split('polyorder_')[1]  # "2"
                try:
                    result['polyorder'] = int(polyorder_str)
                except:
                    pass

    return result


def analyze_parameters(results_df: pd.DataFrame) -> dict:
    """Count parameter selections from feature names."""

    dpo_counts = defaultdict(int)
    tema_counts = defaultdict(float)     # Old format: tema shifts
    shift_counts = defaultdict(float)    # New format: DPO shifts
    savgol_counts = defaultdict(int)
    polyorder_counts = defaultdict(int)

    for _, row in results_df.iterrows():
        if pd.isna(row['selected_features']):
            continue

        features = row['selected_features'].split('|')

        for feature in features:
            feature = feature.strip()
            params = parse_feature_name(feature)

            if 'dpo_period' in params:
                dpo_counts[params['dpo_period']] += 1

            # Count shifts from both old and new formats
            if 'tema_shift' in params:
                # Old format: count tema shifts
                try:
                    shift_key = float(params['tema_shift'])
                    tema_counts[shift_key] += 1
                except:
                    pass
            if 'shift' in params:
                # New format: count DPO shifts
                try:
                    shift_key = float(params['shift'])
                    shift_counts[shift_key] += 1
                except:
                    pass

            if 'savgol_window' in params:
                savgol_counts[params['savgol_window']] += 1
            if 'polyorder' in params:
                polyorder_counts[params['polyorder']] += 1

    return {
        'dpo': dict(dpo_counts),
        'tema': dict(tema_counts),
        'shift': dict(shift_counts),
        'savgol': dict(savgol_counts),
        'polyorder': dict(polyorder_counts)
    }


def main():
    """Analyze parameter selection and print tables."""

    print("=" * 120)
    print("PARAMETER SELECTION FREQUENCY ANALYSIS")
    print("=" * 120)

    # Dynamically find all backtest result files
    import re
    backtest_files = sorted(DATA_DIR.glob('bayesian_backtest_N*.csv'))

    if not backtest_files:
        print(f"ERROR: No backtest results found!")
        print(f"  Expected pattern: {DATA_DIR}/bayesian_backtest_N*.csv")
        return 1

    # Extract N values and load files
    params_by_n = {}
    for filepath in backtest_files:
        match = re.search(r'N(\d+)', filepath.name)
        if match:
            n_value = int(match.group(1))
            df = pd.read_csv(filepath)
            print(f"Loaded N={n_value} results: {len(df)} months")
            params_by_n[n_value] = analyze_parameters(df)

    if not params_by_n:
        print("ERROR: Could not extract N values from backtest files")
        return 1

    # Get sorted list of N values
    n_values = sorted(params_by_n.keys())

    # Extract parameter ranges from actual pipeline code
    print("\nExtracting parameter ranges from pipeline code...")
    dpo_periods = get_dpo_periods()
    dpo_shifts = get_dpo_shifts()
    dpo_polyorder = get_dpo_polyorder()
    savgol_windows = get_savgol_windows()

    # Validate extraction
    if not dpo_periods:
        print("ERROR: Could not extract DPO periods from code")
        return 1
    if not dpo_shifts:
        print("ERROR: Could not extract DPO shifts from code")
        return 1
    if dpo_polyorder is None:
        print("ERROR: Could not extract DPO polyorder from code")
        return 1
    if not savgol_windows:
        print("ERROR: Could not extract Savgol windows from code")
        return 1

    print(f"  DPO periods: {dpo_periods[0]}d to {dpo_periods[-1]}d ({len(dpo_periods)} total)")
    print(f"  DPO shifts: {dpo_shifts[0]} to {dpo_shifts[-1]} ({len(dpo_shifts)} total)")
    print(f"  DPO polyorder: {dpo_polyorder} (quartic)")
    print(f"  Savgol windows: {savgol_windows[0]}d to {savgol_windows[-1]}d ({len(savgol_windows)} total)")

    # ========================================================================
    # DPO PERIODS TABLE (with split range detection)
    # ========================================================================
    print("\n" + "=" * 120)
    print("DPO PERIODS (days)")
    print("=" * 120)

    # Detect splits in ranges (gaps > step size indicate separate ranges)
    splits = []
    if len(dpo_periods) > 1:
        # Calculate expected step from first consecutive pair
        step = dpo_periods[1] - dpo_periods[0]

        # Find gaps (where difference > step)
        for i in range(len(dpo_periods) - 1):
            if dpo_periods[i + 1] - dpo_periods[i] > step:
                splits.append((dpo_periods[:i + 1], dpo_periods[i + 1:]))
                break

    # Build dynamic header based on available N values
    header = f"{'Period':>10}"
    for n in n_values:
        header += f" {'N=' + str(n):>8}"
    header += f" {'Total':>8}"

    # Print splits separately if detected, otherwise print as one
    if splits:
        range1, range2 = splits[0]

        # Range 1
        print(f"SHORT-TERM: {range1[0]}d to {range1[-1]}d")
        print(header)
        print("-" * 120)
        for period in range1:
            row = f"{period:>9}d"
            counts = []
            for n in n_values:
                count = params_by_n[n]['dpo'].get(period, 0)
                row += f" {count:>8}"
                counts.append(count)
            total = sum(counts)
            row += f" {total:>8}"
            print(row)

        used_count_1 = sum(1 for p in range1 if sum(params_by_n[n]['dpo'].get(p, 0) for n in n_values) > 0)
        print("-" * 120)
        print(f"{'SUMMARY':>10} - Used: {used_count_1}/{len(range1)}, Unused: {len(range1) - used_count_1}/{len(range1)}")

        # Range 2
        print(f"\nMEDIUM-TERM: {range2[0]}d to {range2[-1]}d")
        print(header)
        print("-" * 120)
        for period in range2:
            row = f"{period:>9}d"
            counts = []
            for n in n_values:
                count = params_by_n[n]['dpo'].get(period, 0)
                row += f" {count:>8}"
                counts.append(count)
            total = sum(counts)
            row += f" {total:>8}"
            print(row)

        used_count_2 = sum(1 for p in range2 if sum(params_by_n[n]['dpo'].get(p, 0) for n in n_values) > 0)
        print("-" * 120)
        print(f"{'SUMMARY':>10} - Used: {used_count_2}/{len(range2)}, Unused: {len(range2) - used_count_2}/{len(range2)}")

        # Combined summary
        print("-" * 120)
        total_used = used_count_1 + used_count_2
        total_periods = len(dpo_periods)
        print(f"{'COMBINED':>10} - Used: {total_used}/{total_periods}, Unused: {total_periods - total_used}/{total_periods}")
    else:
        # No splits, print normally
        print(header)
        print("-" * 120)

        for period in dpo_periods:
            row = f"{period:>9}d"
            counts = []
            for n in n_values:
                count = params_by_n[n]['dpo'].get(period, 0)
                row += f" {count:>8}"
                counts.append(count)
            total = sum(counts)
            row += f" {total:>8}"
            print(row)

        # Summary
        print("-" * 120)
        used_count = sum(1 for p in dpo_periods if sum(params_by_n[n]['dpo'].get(p, 0) for n in n_values) > 0)
        print(f"{'SUMMARY':>10} - Used: {used_count}/{len(dpo_periods)}, Unused: {len(dpo_periods) - used_count}/{len(dpo_periods)}")

    # ========================================================================
    # TEMA SHIFTS TABLE
    # ========================================================================
    print("\n" + "=" * 120)
    print("DPO SHIFTS (divisors)")
    print("=" * 120)

    # Build dynamic header
    shift_header = f"{'Shift':>10}"
    for n in n_values:
        shift_header += f" {'N=' + str(n):>8}"
    shift_header += f" {'Total':>8}"
    print(shift_header)
    print("-" * 120)

    for shift_str in dpo_shifts:
        shift_val = float(shift_str)
        row = f"{shift_str:>10}"
        counts = []
        for n in n_values:
            count = int(params_by_n[n]['shift'].get(shift_val, 0))
            row += f" {count:>8}"
            counts.append(count)
        total = sum(counts)
        row += f" {total:>8}"
        print(row)

    # Summary
    print("-" * 120)
    used_count = sum(1 for s_str in dpo_shifts if sum(int(params_by_n[n]['shift'].get(float(s_str), 0)) for n in n_values) > 0)
    print(f"{'SUMMARY':>10} - Used: {used_count}/{len(dpo_shifts)}, Unused: {len(dpo_shifts) - used_count}/{len(dpo_shifts)}")

    # ========================================================================
    # DPO POLYORDER
    # ========================================================================
    print("\n" + "=" * 120)
    print(f"DPO POLYORDER (Savgol polynomial degree)")
    print("=" * 120)
    print(f"Polyorder: {dpo_polyorder} (quartic - heavy smoothing)")
    print(f"All {len(dpo_periods)} × {len(dpo_shifts)} = {len(dpo_periods) * len(dpo_shifts)} DPO variants use polyorder {dpo_polyorder}")

    # ========================================================================
    # SAVGOL WINDOWS TABLE (with split range detection)
    # ========================================================================
    print("\n" + "=" * 120)
    print("SAVGOL WINDOWS (days)")
    print("=" * 120)

    # Detect splits in ranges (gaps > step size indicate separate ranges)
    splits = []
    if len(savgol_windows) > 1:
        # Calculate expected step from first consecutive pair
        step = savgol_windows[1] - savgol_windows[0]

        # Find gaps (where difference > step)
        for i in range(len(savgol_windows) - 1):
            if savgol_windows[i + 1] - savgol_windows[i] > step:
                splits.append((savgol_windows[:i + 1], savgol_windows[i + 1:]))
                break

    # Build dynamic header
    savgol_header = f"{'Window':>10}"
    for n in n_values:
        savgol_header += f" {'N=' + str(n):>8}"
    savgol_header += f" {'Total':>8}"

    # Print splits separately if detected, otherwise print as one
    if splits:
        range1, range2 = splits[0]

        # Range 1
        print(f"SHORT-TERM: {range1[0]}d to {range1[-1]}d")
        print(savgol_header)
        print("-" * 120)
        for window in range1:
            row = f"{window:>9}d"
            counts = []
            for n in n_values:
                count = params_by_n[n]['savgol'].get(window, 0)
                row += f" {count:>8}"
                counts.append(count)
            total = sum(counts)
            row += f" {total:>8}"
            print(row)

        used_count_1 = sum(1 for w in range1 if sum(params_by_n[n]['savgol'].get(w, 0) for n in n_values) > 0)
        print("-" * 120)
        print(f"{'SUMMARY':>10} - Used: {used_count_1}/{len(range1)}, Unused: {len(range1) - used_count_1}/{len(range1)}")

        # Range 2
        print(f"\nMEDIUM-TERM: {range2[0]}d to {range2[-1]}d")
        print(savgol_header)
        print("-" * 120)
        for window in range2:
            row = f"{window:>9}d"
            counts = []
            for n in n_values:
                count = params_by_n[n]['savgol'].get(window, 0)
                row += f" {count:>8}"
                counts.append(count)
            total = sum(counts)
            row += f" {total:>8}"
            print(row)

        used_count_2 = sum(1 for w in range2 if sum(params_by_n[n]['savgol'].get(w, 0) for n in n_values) > 0)
        print("-" * 120)
        print(f"{'SUMMARY':>10} - Used: {used_count_2}/{len(range2)}, Unused: {len(range2) - used_count_2}/{len(range2)}")

        # Combined summary
        print("-" * 120)
        total_used = used_count_1 + used_count_2
        total_windows = len(savgol_windows)
        print(f"{'COMBINED':>10} - Used: {total_used}/{total_windows}, Unused: {total_windows - total_used}/{total_windows}")
    else:
        # No splits, print normally
        print(savgol_header)
        print("-" * 120)

        for window in savgol_windows:
            row = f"{window:>9}d"
            counts = []
            for n in n_values:
                count = params_by_n[n]['savgol'].get(window, 0)
                row += f" {count:>8}"
                counts.append(count)
            total = sum(counts)
            row += f" {total:>8}"
            print(row)

        # Summary
        print("-" * 120)
        used_count = sum(1 for w in savgol_windows if sum(params_by_n[n]['savgol'].get(w, 0) for n in n_values) > 0)
        print(f"{'SUMMARY':>10} - Used: {used_count}/{len(savgol_windows)}, Unused: {len(savgol_windows) - used_count}/{len(savgol_windows)}")

    # ========================================================================
    # POLYORDER TABLE
    # ========================================================================
    print("\n" + "=" * 120)
    print("POLYORDER (Savgol filter polynomial degree)")
    print("=" * 120)

    # Build dynamic header
    poly_header = f"{'Order':>10}"
    for n in n_values:
        poly_header += f" {'N=' + str(n):>8}"
    poly_header += f" {'Total':>8}"
    print(poly_header)
    print("-" * 120)

    # Get all polyorders found across all N values
    all_polyorders = set()
    for n in n_values:
        all_polyorders.update(params_by_n[n]['polyorder'].keys())

    polyorders_sorted = sorted(all_polyorders)

    for order in polyorders_sorted:
        row = f"{int(order):>10}"
        counts = []
        for n in n_values:
            count = int(params_by_n[n]['polyorder'].get(order, 0))
            row += f" {count:>8}"
            counts.append(count)
        total = sum(counts)
        row += f" {total:>8}"
        print(row)

    # Summary
    print("-" * 120)
    used_count = sum(1 for o in polyorders_sorted if sum(int(params_by_n[n]['polyorder'].get(o, 0)) for n in n_values) > 0)
    print(f"{'SUMMARY':>10} - Used: {used_count}/{len(polyorders_sorted)}, Unused: {len(polyorders_sorted) - used_count}/{len(polyorders_sorted)}")

    print("\n" + "=" * 120)

    return 0


if __name__ == '__main__':
    exit(main())
