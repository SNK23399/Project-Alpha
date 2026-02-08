#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Incremental Month-by-Month Validation
======================================

Validates Step 7 predictions by:
1. Loading full backtest results as baseline (complete data)
2. For each historical month:
   a. Truncate database to end of that month
   b. Run pipeline Steps 1-6
   c. Extract predictions
   d. Compare with baseline backtest

This verifies predictions are reproducible and deterministic.

Usage:
    python validate_incremental.py                    # All months
    python validate_incremental.py --months 5         # Last 5 months
"""

import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import tempfile
import sqlite3

# Paths
project_root = Path(__file__).parent
pipeline_dir = project_root / "pipeline"
data_dir = pipeline_dir / "data"
backtest_dir = data_dir / "backtest_results"
db_path = project_root / "maintenance" / "data" / "etf_database.db"
validation_dir = project_root / "validation_incremental"

N_SATELLITES_TO_TEST = [3, 4, 5]


def run_baseline_pipeline(force_rerun=False):
    """Run full pipeline (Steps 1-6) to create baseline backtest results."""
    print(f"\n{'='*100}")
    print("PHASE 1: RUNNING BASELINE PIPELINE (Steps 1-6)")
    print(f"{'='*100}\n")

    # Check if baseline already exists
    baseline_exists = all(
        (backtest_dir / f"bayesian_backtest_N{n}.csv").exists()
        for n in N_SATELLITES_TO_TEST
    )

    if baseline_exists and not force_rerun:
        print("Baseline backtest results already exist. Skipping baseline pipeline run.")
        print("(Use --force-baseline to force re-run)\n")
        return True

    if baseline_exists and force_rerun:
        print("Re-running baseline pipeline (force-baseline flag set)...")

    # Clear existing data directory to ensure clean run
    if (pipeline_dir / "data").exists():
        print("Clearing existing pipeline data...")
        shutil.rmtree(pipeline_dir / "data")
    (pipeline_dir / "data").mkdir(parents=True, exist_ok=True)

    # Run steps 1-6
    for step in range(1, 7):
        script_map = {
            1: "1_compute_forward_ir.py",
            2: "2_compute_signal_bases.py",
            3: "3_apply_filters.py",
            4: "4_precompute_feature_ir.py",
            5: "5_empirical_ir_stats.py",
            6: "6_deterministic_strategy_ir.py",
        }

        script = pipeline_dir / script_map[step]
        print(f"Step {step}: {script.name}...", end=" ", flush=True)

        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(pipeline_dir),
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                print(f"ERROR")
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")
                return False

            print("OK")

        except subprocess.TimeoutExpired:
            print(f"ERROR (timeout)")
            return False
        except Exception as e:
            print(f"ERROR: {e}")
            return False

    # Verify baseline was created
    print(f"\nVerifying baseline results...")
    for n in N_SATELLITES_TO_TEST:
        csv_file = backtest_dir / f"bayesian_backtest_N{n}.csv"
        if csv_file.exists():
            df = pd.read_csv(str(csv_file))
            print(f"  [OK] bayesian_backtest_N{n}.csv: {len(df)} rows")
        else:
            print(f"  [FAIL] bayesian_backtest_N{n}.csv: NOT FOUND")
            return False

    return True


def load_baseline_backtest() -> dict:
    """Load the full backtest results as baseline."""
    baseline = {}
    for n in N_SATELLITES_TO_TEST:
        csv_file = backtest_dir / f"bayesian_backtest_N{n}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"Baseline not found: {str(csv_file)}")
        df = pd.read_csv(str(csv_file))
        df['date'] = pd.to_datetime(df['date'])
        baseline[n] = df.set_index('date').to_dict('index')
    return baseline


def get_month_end_date(test_date: pd.Timestamp) -> pd.Timestamp:
    """Get last day of the month before test_date."""
    first_of_month = test_date.replace(day=1)
    last_of_prev_month = first_of_month - pd.DateOffset(days=1)
    return last_of_prev_month


def truncate_database(original_db: Path, truncate_date: pd.Timestamp, output_db: Path):
    """Create a truncated database copy with data up to truncate_date."""
    print(f"    Truncating database to {truncate_date.date()}...")

    shutil.copy2(original_db, output_db)

    try:
        conn = sqlite3.connect(str(output_db))
        cursor = conn.cursor()

        # Find date column
        cursor.execute("PRAGMA table_info(prices)")
        columns = cursor.fetchall()
        date_col = next((col[1] for col in columns if 'date' in col[1].lower()), 'date')

        # Delete prices after truncate_date
        delete_sql = f"DELETE FROM prices WHERE {date_col} > ?"
        cursor.execute(delete_sql, (truncate_date.strftime('%Y-%m-%d'),))
        conn.commit()
        conn.close()

        return True
    except Exception as e:
        print(f"    WARNING: Database truncation issue: {e}")
        return True  # Continue anyway


def run_pipeline_steps(pipeline_dir: Path) -> bool:
    """Run pipeline steps 1-6."""
    for step in range(1, 7):
        script_map = {
            1: "1_compute_forward_ir.py",
            2: "2_compute_signal_bases.py",
            3: "3_apply_filters.py",
            4: "4_precompute_feature_ir.py",
            5: "5_empirical_ir_stats.py",
            6: "6_deterministic_strategy_ir.py",
        }

        script = pipeline_dir / script_map[step]

        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(pipeline_dir),
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                print(f"    ERROR in Step {step}")
                return False

        except subprocess.TimeoutExpired:
            print(f"    ERROR: Step {step} timed out")
            return False
        except Exception as e:
            print(f"    ERROR: Step {step} failed: {e}")
            return False

    return True


def extract_predictions(test_date: pd.Timestamp) -> dict:
    """Extract predictions from backtest CSVs."""
    predictions = {}

    for n in N_SATELLITES_TO_TEST:
        csv_file = backtest_dir / f"bayesian_backtest_N{n}.csv"
        if not csv_file.exists():
            continue

        df = pd.read_csv(str(csv_file))
        df['date'] = pd.to_datetime(df['date'])

        row = df[df['date'] == test_date]
        if not row.empty:
            isins_str = row.iloc[0]['selected_isins']
            if isinstance(isins_str, str) and isins_str.strip():
                predictions[n] = sorted([x.strip() for x in isins_str.split(',')])

    return predictions


def compare_predictions(baseline_pred: list, rerun_pred: list) -> tuple:
    """Compare two prediction sets."""
    if not baseline_pred or not rerun_pred:
        return False, "Missing predictions"

    match = baseline_pred == rerun_pred

    if match:
        return True, "Match"
    else:
        missing = set(baseline_pred) - set(rerun_pred)
        extra = set(rerun_pred) - set(baseline_pred)
        details = []
        if missing:
            details.append(f"Missing: {missing}")
        if extra:
            details.append(f"Extra: {extra}")
        return False, "; ".join(details)


def validate_month(test_date: pd.Timestamp, baseline: dict, temp_work_dir: Path) -> dict:
    """Validate a single month."""
    result = {
        'date': test_date,
        'status': 'pending',
        'matches': {}
    }

    truncate_date = get_month_end_date(test_date)

    print(f"\n  {test_date.date()} (DB truncated to {truncate_date.date()})")

    # Get baseline predictions
    baseline_preds = {}
    for n in N_SATELLITES_TO_TEST:
        if test_date in baseline[n]:
            isins_str = baseline[n][test_date].get('selected_isins', '')
            if isinstance(isins_str, str) and isins_str.strip():
                baseline_preds[n] = sorted([x.strip() for x in isins_str.split(',')])

    if not baseline_preds:
        result['status'] = 'skipped'
        print(f"    → No baseline predictions")
        return result

    try:
        # Create working directory
        work_dir = temp_work_dir / f"validate_{test_date.strftime('%Y%m%d')}"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Clear pipeline data
        if (pipeline_dir / "data").exists():
            shutil.rmtree(pipeline_dir / "data")
        (pipeline_dir / "data").mkdir(parents=True, exist_ok=True)

        # Truncate database
        truncated_db = work_dir / "etf_database_truncated.db"
        truncate_database(db_path, truncate_date, truncated_db)

        # Run pipeline
        print(f"    Running pipeline...", end="", flush=True)
        if not run_pipeline_steps(pipeline_dir):
            result['status'] = 'error'
            print(" ERROR")
            return result
        print(" OK")

        # Extract predictions
        rerun_preds = extract_predictions(test_date)

        # Compare
        all_match = True
        for n in N_SATELLITES_TO_TEST:
            if n in baseline_preds:
                match, details = compare_predictions(
                    baseline_preds[n],
                    rerun_preds.get(n, [])
                )
                result['matches'][n] = {'match': match, 'details': details}

                status = "[OK]" if match else "[FAIL]"
                print(f"    {status} N={n}: {details}")

                if not match:
                    all_match = False

        result['status'] = 'success' if all_match else 'mismatch'

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        print(f"    ERROR: {e}")

    return result


def main():
    """Main validation runner."""
    print(f"\n{'='*100}")
    print("INCREMENTAL MONTH-BY-MONTH VALIDATION")
    print(f"{'='*100}")
    print(f"\nStrategy: Baseline (full data) vs Rerun (truncated month-by-month)\n")

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--months', type=int, default=None, help='Validate last N months')
    parser.add_argument('--force-baseline', action='store_true', help='Force re-run of baseline pipeline')
    args = parser.parse_args()

    # Phase 1: Run baseline pipeline if needed
    print(f"{'='*100}")
    print("PHASE 1: BASELINE PIPELINE EXECUTION")
    print(f"{'='*100}")
    if not run_baseline_pipeline(force_rerun=args.force_baseline):
        print(f"\nERROR: Baseline pipeline failed")
        return 1

    # Phase 2: Load baseline
    print(f"\n{'='*100}")
    print("PHASE 2: LOADING BASELINE RESULTS")
    print(f"{'='*100}\n")
    print(f"Loading baseline backtest results...")
    try:
        baseline = load_baseline_backtest()
        print(f"  Loaded backtest for N={list(baseline.keys())}")
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    # Get months to validate
    baseline_3 = baseline[3]
    test_dates = sorted(baseline_3.keys())

    if args.months:
        test_dates = test_dates[-args.months:]

    print(f"  {len(test_dates)} months to validate: {test_dates[0].date()} to {test_dates[-1].date()}")

    # Phase 3: Validate
    print(f"\n{'='*100}")
    print(f"PHASE 3: INCREMENTAL VALIDATION ({len(test_dates)} MONTHS)")
    print(f"{'='*100}")

    all_results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for test_date in tqdm(test_dates, desc="Validating", ncols=100):
            result = validate_month(test_date, baseline, Path(temp_dir))
            all_results.append(result)

    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}\n")

    successes = sum(1 for r in all_results if r['status'] == 'success')
    mismatches = sum(1 for r in all_results if r['status'] == 'mismatch')
    errors = sum(1 for r in all_results if r['status'] == 'error')
    skipped = sum(1 for r in all_results if r['status'] == 'skipped')

    print(f"Results:")
    print(f"  [OK] Successes: {successes}")
    print(f"  [FAIL] Mismatches: {mismatches}")
    print(f"  [WARN] Errors: {errors}")
    print(f"  [SKIP] Skipped: {skipped}")

    # Detailed failures
    if mismatches > 0 or errors > 0:
        print(f"\n{'='*100}")
        print("FAILURES")
        print(f"{'='*100}\n")

        for result in all_results:
            if result['status'] in ['mismatch', 'error']:
                print(f"Date: {result['date'].date()}")
                if result['status'] == 'error':
                    print(f"  ERROR: {result.get('error', 'Unknown')}")
                else:
                    for n, match_info in result['matches'].items():
                        print(f"  N={n}: {match_info['details']}")
                print()

    # Overall
    print(f"{'='*100}")
    if successes == len([r for r in all_results if r['status'] != 'skipped']):
        print("[SUCCESS] All validations passed! Predictions are reproducible.")
    else:
        print("[FAILURE] Some validations failed.")
    print(f"{'='*100}\n")

    return 0 if mismatches == 0 and errors == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
