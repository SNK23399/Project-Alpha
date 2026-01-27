#!/usr/bin/env python3
"""
Run Pipeline Against Truncated Databases (Validation Period Only)
==================================================================

For each month-end database in the validation period (2018-02-28 to 2025-11-30):
1. Temporarily swap in the truncated database
2. Run pipeline stages 1-6 (with that month's data only)
3. Save backtest results to historical_predictions/{date}/
4. Restore original database

Note: Only tests periods that were actually tested in the original pipeline.
The first ~36 months (2009-09-30 to 2018-01-31) are skipped as they were
not included in the original backtest results.

RESUME SUPPORT:
This script automatically detects already-completed months and skips them.
You can interrupt the script at any time and resume later - it will pick up
from where you left off.

This tests the pipeline as if it were deployed at each historical date,
with only data available at that time, and compares against current results.

Usage:
    python run_pipeline_walk_forward.py    # Run/resume validation
    python run_pipeline_walk_forward.py    # Run again to continue from last completed month

Output:
    historical_predictions/
    ├── 2018-02-28/    ← First validation date
    │   ├── bayesian_backtest_N3.csv
    │   ├── bayesian_backtest_N4.csv
    │   ├── bayesian_backtest_N5.csv
    │   ├── bayesian_backtest_N6.csv
    │   ├── bayesian_backtest_N7.csv
    │   └── metadata.txt
    ├── 2018-03-31/
    └── ... (through 2025-11-30)
"""

import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import time

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not installed. Install with: pip install tqdm")
    print("Progress bar will be basic without tqdm.\n")

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
VALIDATION_DIR = SCRIPT_DIR
DATABASES_DIR = VALIDATION_DIR / 'databases'
PREDICTIONS_DIR = VALIDATION_DIR / 'historical_predictions'
MAIN_DB_PATH = PROJECT_ROOT / 'maintenance' / 'data' / 'etf_database.db'
PIPELINE_SCRIPT = PROJECT_ROOT / 'main.py'
BACKTEST_OUTPUT_DIR = PROJECT_ROOT / 'pipeline' / 'data' / 'backtest_results'

# Only test periods that were actually tested in the pipeline
# Current backtest results: 2018-02-28 to 2025-11-30
# Skip first ~36 months (2009-09-30 to 2018-01-31) that weren't tested
VALIDATION_START_DATE = '2018-02-28'
VALIDATION_END_DATE = '2025-11-30'

# ============================================================
# FUNCTIONS
# ============================================================

def load_database_dates():
    """Load list of month-end dates from summary.csv."""
    summary_file = DATABASES_DIR / 'summary.csv'

    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    df = pd.read_csv(summary_file)
    dates = pd.to_datetime(df['date']).tolist()

    return sorted(dates)

def get_completed_dates():
    """Get list of dates that have already been processed."""
    if not PREDICTIONS_DIR.exists():
        return set()

    completed = set()
    for date_dir in PREDICTIONS_DIR.iterdir():
        if date_dir.is_dir() and (date_dir / 'bayesian_backtest_N5.csv').exists():
            try:
                # Parse date from directory name (YYYY-MM-DD format)
                date_str = date_dir.name
                date_obj = pd.to_datetime(date_str)
                completed.add(date_obj)
            except (ValueError, TypeError):
                # Skip directories that don't match date format
                pass

    return completed

def backup_main_database():
    """Create backup of main database."""
    if not MAIN_DB_PATH.exists():
        raise FileNotFoundError(f"Main database not found: {MAIN_DB_PATH}")

    backup_path = MAIN_DB_PATH.parent / f'{MAIN_DB_PATH.name}.backup'
    shutil.copy2(MAIN_DB_PATH, backup_path)

    return backup_path

def swap_database(source_db):
    """Swap in truncated database as main database."""
    # Remove current main database
    if MAIN_DB_PATH.exists():
        MAIN_DB_PATH.unlink()

    # Copy truncated database to main location
    shutil.copy2(source_db, MAIN_DB_PATH)

def restore_database(backup_path):
    """Restore main database from backup."""
    if MAIN_DB_PATH.exists():
        MAIN_DB_PATH.unlink()

    shutil.copy2(backup_path, MAIN_DB_PATH)

def run_pipeline():
    """Run pipeline stages 1-6."""
    cmd = [
        'python',
        str(PIPELINE_SCRIPT),
        '--steps', '1,2,3,4,5,6'
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True
        )

        return result.returncode == 0, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return False, '', 'Pipeline execution timed out'
    except Exception as e:
        return False, '', str(e)

def save_backtest_results(target_date):
    """Copy backtest results to historical_predictions/{date}/."""
    date_str = target_date.strftime('%Y-%m-%d')
    output_dir = PREDICTIONS_DIR / date_str
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy all backtest CSV files
    for csv_file in BACKTEST_OUTPUT_DIR.glob('bayesian_backtest_N*.csv'):
        if csv_file.is_file():
            shutil.copy2(csv_file, output_dir / csv_file.name)

    # Write metadata
    metadata_file = output_dir / 'metadata.txt'
    with open(metadata_file, 'w') as f:
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Target Date: {date_str}\n")
        f.write(f"Data Available Through: {date_str}\n")

def print_header(text):
    """Print formatted header."""
    print()
    print("=" * 100)
    print(text)
    print("=" * 100)

def main():
    """Main entry point."""

    print_header("WALK-FORWARD PIPELINE VALIDATION")

    try:
        # Validate prerequisites
        print(f"\nValidating setup...")
        print(f"  Project root: {PROJECT_ROOT}")
        print(f"  Pipeline script: {PIPELINE_SCRIPT}")
        print(f"  Main database: {MAIN_DB_PATH}")
        print(f"  Databases directory: {DATABASES_DIR}")
        print(f"  Predictions output: {PREDICTIONS_DIR}")

        if not PIPELINE_SCRIPT.exists():
            raise FileNotFoundError(f"Pipeline script not found: {PIPELINE_SCRIPT}")

        if not DATABASES_DIR.exists():
            raise FileNotFoundError(f"Databases directory not found: {DATABASES_DIR}")

        # Load database dates
        print(f"\nLoading database dates...")
        all_dates = load_database_dates()
        print(f"  Found {len(all_dates)} total month-end databases")

        # Filter to only validated period (when pipeline was tested)
        validation_start = pd.to_datetime(VALIDATION_START_DATE)
        validation_end = pd.to_datetime(VALIDATION_END_DATE)
        dates = [d for d in all_dates if validation_start <= d <= validation_end]

        print(f"  Filtered to validation period: {len(dates)} databases")
        print(f"  Start: {dates[0].strftime('%Y-%m-%d')} (pipeline testing began)")
        print(f"  End:   {dates[-1].strftime('%Y-%m-%d')} (current backtest results)")

        # Check for already-completed dates (resume support)
        print(f"\nChecking for previously completed months...")
        completed_dates = get_completed_dates()
        if completed_dates:
            print(f"  Found {len(completed_dates)} already completed months")
            dates_to_process = [d for d in dates if d not in completed_dates]
            skipped = len(dates) - len(dates_to_process)
            if dates_to_process:
                print(f"  Skipping: {dates[0].strftime('%Y-%m-%d')} through {[d for d in dates if d in completed_dates][-1].strftime('%Y-%m-%d')}")
                print(f"  Resuming from: {dates_to_process[0].strftime('%Y-%m-%d')}")
                print(f"  Remaining: {len(dates_to_process)} months to process")
                dates = dates_to_process
            else:
                print(f"  All {skipped} months are already completed!")
                print(f"  No new work to do. Run compare_predictions.py to analyze results.")
                PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
                return 0
        else:
            print(f"  No completed months found. Starting fresh.")

        # Create backup of main database
        print(f"\nCreating backup of main database...")
        backup_path = backup_main_database()
        print(f"  Backup: {backup_path}")

        # Output directory
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

        # Process each date
        print(f"\nRunning pipeline for each historical month...")
        print("-" * 100)

        successful = 0
        failed = 0
        failed_dates = []
        start_time = time.time()
        total_remaining = len(dates)

        # Use tqdm for progress bar if available
        if HAS_TQDM:
            date_iterator = tqdm(
                enumerate(dates, 1),
                total=total_remaining,
                desc="Pipeline validation",
                unit="month",
                bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%] - {postfix}",
                ncols=120
            )
        else:
            date_iterator = enumerate(dates, 1)

        for idx, target_date in date_iterator:
            date_str = target_date.strftime('%Y-%m-%d')

            # Get truncated database
            truncated_db = DATABASES_DIR / date_str / 'etf_database.db'

            if not truncated_db.exists():
                status = "Database not found"
                if HAS_TQDM:
                    date_iterator.set_postfix_str(f"[{idx:3d}/{len(dates)}] {date_str} - ERROR")
                else:
                    print(f"[{idx:3d}/{len(dates)}] {date_str} - ERROR: {status}")
                failed += 1
                failed_dates.append((date_str, status))
                continue

            try:
                # Swap in truncated database
                swap_database(truncated_db)

                # Run pipeline
                success, stdout, stderr = run_pipeline()

                if success:
                    # Save results
                    save_backtest_results(target_date)
                    if HAS_TQDM:
                        date_iterator.set_postfix_str(f"[{idx:3d}/{len(dates)}] {date_str} - OK")
                    successful += 1
                else:
                    status = "Pipeline failed"
                    if HAS_TQDM:
                        date_iterator.set_postfix_str(f"[{idx:3d}/{len(dates)}] {date_str} - FAILED")
                    else:
                        print(f"[{idx:3d}/{len(dates)}] {date_str} - FAILED", flush=True)
                        if stderr:
                            print(f"  Error: {stderr[:100]}", flush=True)
                    failed += 1
                    failed_dates.append((date_str, status))

            except Exception as e:
                status = str(e)[:50]
                if HAS_TQDM:
                    date_iterator.set_postfix_str(f"[{idx:3d}/{len(dates)}] {date_str} - ERROR")
                else:
                    print(f"[{idx:3d}/{len(dates)}] {date_str} - ERROR: {status}", flush=True)
                failed += 1
                failed_dates.append((date_str, status))

        # Close progress bar if using tqdm
        if HAS_TQDM and hasattr(date_iterator, 'close'):
            date_iterator.close()

        # Restore original database
        print("\n" + "-" * 100)
        print(f"\nRestoring original database...")
        restore_database(backup_path)
        print(f"  Original database restored")

        # Print summary
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        print_header("PIPELINE EXECUTION COMPLETE")

        print(f"\nResults (this run):")
        print(f"  Processed: {successful + failed}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Success rate: {(successful / (successful + failed) * 100):.1f}% (this run)" if (successful + failed) > 0 else "  Success rate: N/A")
        print(f"  Elapsed time: {minutes}m {seconds}s")

        # Get final total stats
        final_completed = get_completed_dates()
        print(f"\nCumulative Results (all runs):")
        print(f"  Total completed: {len(final_completed)}/94")
        print(f"  Remaining: {94 - len(final_completed)}")
        if len(final_completed) == 94:
            print(f"  Status: VALIDATION COMPLETE")

        if failed_dates:
            print(f"\nFailed dates:")
            for date_str, error in failed_dates[:10]:  # Show first 10
                print(f"  {date_str}: {error}")
            if len(failed_dates) > 10:
                print(f"  ... and {len(failed_dates) - 10} more")

        print(f"\nHistorical predictions saved to:")
        print(f"  {PREDICTIONS_DIR}")

        if len(final_completed) == 94:
            print(f"\nNext step:")
            print(f"  python compare_predictions.py ./historical_predictions")
        else:
            print(f"\nTo continue where you left off, run:")
            print(f"  python run_pipeline_walk_forward.py")

        print()

        return 0 if failed == 0 else 1

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
