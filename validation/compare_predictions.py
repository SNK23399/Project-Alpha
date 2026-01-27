#!/usr/bin/env python3
"""
Compare Historical Walk-Forward Predictions with Current Results
=================================================================

This script compares satellite selections made during walk-forward validation
with current results to detect any systematic future bias.

Usage:
    python compare_predictions.py /path/to/historical_predictions

Output:
    - Satellite selection accuracy metrics
    - Detection of systematic look-ahead bias
    - Monthly and aggregate comparison tables
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# ============================================================
# CONFIGURATION
# ============================================================

CURRENT_RESULTS_DIR = project_root / 'pipeline' / 'data' / 'backtest_results'
N_VALUES = [3, 4, 5, 6, 7]

# ============================================================
# FUNCTIONS
# ============================================================

def load_current_results():
    """Load all current backtest results from pipeline."""
    current = {}

    for n in N_VALUES:
        filepath = CURRENT_RESULTS_DIR / f'bayesian_backtest_N{n}.csv'
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            current[n] = df.set_index('date')
        else:
            print(f"  Warning: {filepath} not found")

    return current

def load_historical_predictions(predictions_dir):
    """Load all historical predictions from walk-forward validation."""
    historical = defaultdict(dict)

    predictions_path = Path(predictions_dir)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    # Iterate through date directories
    for date_dir in sorted(predictions_path.iterdir()):
        if not date_dir.is_dir():
            continue

        date_str = date_dir.name
        try:
            date = pd.to_datetime(date_str)
        except:
            print(f"  Warning: Invalid date directory: {date_str}")
            continue

        # Load each N value for this date
        for n in N_VALUES:
            filepath = date_dir / f'bayesian_backtest_N{n}.csv'
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    if len(df) > 0:
                        # Get the last row (most recent prediction for this date)
                        latest = df.iloc[-1]
                        historical[n][date] = {
                            'selected_isins': latest['selected_isins'],
                            'avg_alpha': latest['avg_alpha'],
                            'learned_decay': latest['learned_decay'],
                            'learned_prior_strength': latest['learned_prior_strength']
                        }
                except Exception as e:
                    print(f"  Warning: Error reading {filepath}: {e}")

    return historical

def parse_isins(isin_str):
    """Parse comma-separated ISIN string into set."""
    if pd.isna(isin_str) or not isin_str:
        return set()
    return set(isin.strip() for isin in str(isin_str).split(','))

def calculate_match_metrics(predicted_isins, current_isins):
    """Calculate match metrics between predicted and current selections."""
    predicted = parse_isins(predicted_isins)
    current = parse_isins(current_isins)

    if len(predicted) == 0 and len(current) == 0:
        return {
            'match_count': 0,
            'match_pct': 100.0,
            'precision': np.nan,
            'recall': np.nan,
            'f1': np.nan
        }

    matches = len(predicted & current)

    # Calculate metrics
    match_pct = (matches / len(predicted) * 100) if len(predicted) > 0 else 0
    precision = (matches / len(current)) if len(current) > 0 else 0
    recall = (matches / len(predicted)) if len(predicted) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'match_count': matches,
        'match_pct': match_pct,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predicted_isins': predicted,
        'current_isins': current
    }

def print_header(text):
    """Print formatted header."""
    print()
    print("=" * 100)
    print(text)
    print("=" * 100)

def generate_comparison_report(current_results, historical_predictions):
    """Generate comprehensive comparison report."""

    print_header("WALK-FORWARD VALIDATION COMPARISON REPORT")

    print(f"\nCurrent Results Directory: {CURRENT_RESULTS_DIR}")
    print(f"Comparison Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Summary statistics by N
    print_header("OVERALL STATISTICS BY N")

    for n in N_VALUES:
        if n not in historical_predictions or len(historical_predictions[n]) == 0:
            print(f"\nN={n}: No historical predictions found")
            continue

        print(f"\nN={n}:")
        print(f"  Historical prediction dates: {len(historical_predictions[n])}")

        # Get current results for this N
        if n not in current_results:
            print(f"  Current results: NOT FOUND")
            continue

        current_df = current_results[n]
        print(f"  Current backtest dates: {len(current_df)}")

        # Find overlapping dates
        hist_dates = set(historical_predictions[n].keys())
        current_dates = set(current_df.index)
        overlapping = hist_dates & current_dates

        print(f"  Overlapping dates: {len(overlapping)}")

        if len(overlapping) == 0:
            print("  WARNING: No overlapping dates between historical and current results")
            continue

        # Calculate match metrics for all overlapping dates
        matches = []
        alpha_diffs = []

        for date in sorted(overlapping):
            hist_data = historical_predictions[n][date]
            current_row = current_df.loc[date]

            metrics = calculate_match_metrics(
                hist_data['selected_isins'],
                current_row['selected_isins']
            )
            matches.append(metrics['match_pct'])

            # Calculate alpha difference
            alpha_diff = current_row['avg_alpha'] - hist_data['avg_alpha']
            alpha_diffs.append(alpha_diff)

        if matches:
            print(f"  Satellite Match %: {np.mean(matches):.1f}% ± {np.std(matches):.1f}%")
            print(f"    Min: {np.min(matches):.1f}%, Max: {np.max(matches):.1f}%")
            print(f"  Avg Alpha Difference: {np.mean(alpha_diffs):.6f} ± {np.std(alpha_diffs):.6f}")
            print(f"    Min: {np.min(alpha_diffs):.6f}, Max: {np.max(alpha_diffs):.6f}")

    # Detailed comparison table
    print_header("DETAILED COMPARISON BY DATE AND N")

    for n in sorted(N_VALUES):
        if n not in historical_predictions or len(historical_predictions[n]) == 0:
            continue
        if n not in current_results:
            continue

        current_df = current_results[n]
        hist_dates = set(historical_predictions[n].keys())
        current_dates = set(current_df.index)
        overlapping = sorted(hist_dates & current_dates)

        if len(overlapping) == 0:
            continue

        print(f"\nN={n} - Satellite Match Accuracy")
        print("-" * 100)
        print(f"{'Date':<12} {'Predicted':<20} {'Current':<20} {'Match':<8} {'Match%':<8} {'Δ Alpha':<12}")
        print("-" * 100)

        for date in overlapping:
            hist_data = historical_predictions[n][date]
            current_row = current_df.loc[date]

            metrics = calculate_match_metrics(
                hist_data['selected_isins'],
                current_row['selected_isins']
            )

            alpha_diff = current_row['avg_alpha'] - hist_data['avg_alpha']

            pred_isins = ','.join(sorted(metrics['predicted_isins']))[:18]
            curr_isins = ','.join(sorted(metrics['current_isins']))[:18]

            print(f"{date.strftime('%Y-%m-%d'):<12} {pred_isins:<20} {curr_isins:<20} "
                  f"{metrics['match_count']:<8} {metrics['match_pct']:<7.1f}% {alpha_diff:<12.6f}")

    # Look-ahead bias detection
    print_header("LOOK-AHEAD BIAS ANALYSIS")

    has_bias = False

    for n in sorted(N_VALUES):
        if n not in historical_predictions or len(historical_predictions[n]) == 0:
            continue
        if n not in current_results:
            continue

        current_df = current_results[n]
        hist_dates = set(historical_predictions[n].keys())
        current_dates = set(current_df.index)
        overlapping = hist_dates & current_dates

        if len(overlapping) < 2:
            continue

        alpha_diffs = []
        for date in overlapping:
            hist_data = historical_predictions[n][date]
            current_row = current_df.loc[date]
            alpha_diff = current_row['avg_alpha'] - hist_data['avg_alpha']
            alpha_diffs.append(alpha_diff)

        alpha_diffs = np.array(alpha_diffs)

        # Test if differences are systematically positive (indicates look-ahead bias)
        mean_diff = np.mean(alpha_diffs)

        print(f"\nN={n}:")
        print(f"  Historical Avg Alpha (at time): {np.mean([historical_predictions[n][d]['avg_alpha'] for d in overlapping]):.6f}")
        print(f"  Current Avg Alpha (with all data): {current_df.loc[list(overlapping), 'avg_alpha'].mean():.6f}")
        print(f"  Mean Difference: {mean_diff:+.6f}")

        if mean_diff > 0.001:  # Threshold for "significant" positive bias
            print(f"  ⚠️  POSSIBLE LOOK-AHEAD BIAS DETECTED (positive difference)")
            has_bias = True
        elif mean_diff < -0.001:  # Negative bias
            print(f"  ⚠️  NEGATIVE BIAS (current results worse than predictions)")
            has_bias = True
        else:
            print(f"  ✓ No systematic bias detected")

    print_header("CONCLUSION")

    if has_bias:
        print("\n⚠️  WARNING: Potential systematic bias detected in validation results.")
        print("\nThis could indicate:")
        print("  1. Look-ahead bias in feature engineering (e.g., forward returns used in features)")
        print("  2. Information leakage from future data in signal calculation")
        print("  3. Database contains future data that wasn't properly excluded")
        print("\nRecommended Action: Review signal computation and database truncation logic")
    else:
        print("\n✓ No systematic look-ahead bias detected in walk-forward validation.")
        print("\nHistorical predictions align with current results, suggesting:")
        print("  - No significant information leakage from future data")
        print("  - Validation methodology is sound")
        print("  - Backtesting results are reliable for strategy evaluation")

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python compare_predictions.py /path/to/historical_predictions")
        print("\nExample:")
        print("  python compare_predictions.py ./historical_predictions")
        sys.exit(1)

    predictions_dir = sys.argv[1]

    try:
        print("\n" + "=" * 100)
        print("LOADING DATA")
        print("=" * 100)

        print("\nLoading current backtest results...")
        current_results = load_current_results()
        print(f"  Loaded results for N: {list(current_results.keys())}")

        print(f"\nLoading historical predictions from: {predictions_dir}")
        historical = load_historical_predictions(predictions_dir)
        print(f"  Loaded predictions for N: {list(historical.keys())}")

        # Generate report
        generate_comparison_report(current_results, historical)

        print("\n" + "=" * 100)
        print("COMPARISON COMPLETE")
        print("=" * 100 + "\n")

        return 0

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
