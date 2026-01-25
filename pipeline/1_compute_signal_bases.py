"""
Step 1: Compute Signal Bases
============================

This script:
1. Loads ETF prices from the database
2. Computes all 293 signal bases (momentum, volatility, technical indicators, etc.)
3. Saves them to parquet files (each signal saved immediately)

Features:
- FULL recomputation: Always recomputes from scratch to ensure fresh data
- Price corrections captured: Any database updates are reflected
- Fresh rolling windows: All calculations use current data
- No look-ahead bias: Signal computation only uses historical price data

Outputs:
- pipeline/data/signals/    Contains 293 parquet files (one per signal base)

Usage:
  python 1_compute_signal_bases.py                     # All data (2009-09-25 to today)
  python 1_compute_signal_bases.py 2020-01-01 2024-12-31  # Custom date range
"""

import sys
import time
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from support.etf_database import ETFDatabase
from support.signal_database import SignalDatabase
from library_signal_bases import compute_signal_bases_generator, count_total_signals

# Output directories
PIPELINE_DIR = Path(__file__).parent
SIGNAL_OUTPUT_DIR = PIPELINE_DIR / 'data' / 'signals'
BACKUP_DIR = PIPELINE_DIR / 'backup' / '1_compute_signal_bases'


def backup_signal_bases() -> str:
    """
    Backup current signal bases to dated folder.

    Creates: backup/1_compute_signal_bases/YYYY_MM_DD/

    Returns:
        Path to backup directory, or None if no signals to backup
    """
    if not SIGNAL_OUTPUT_DIR.exists():
        return None

    # Check if there are any parquet files
    parquet_files = list(SIGNAL_OUTPUT_DIR.glob('*.parquet'))
    if not parquet_files:
        return None

    # Create dated backup folder: YYYY_MM_DD
    timestamp = datetime.now().strftime('%Y_%m_%d')
    backup_date_dir = BACKUP_DIR / timestamp

    # Ensure backup directory exists
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # If same day folder exists, append time to make unique
    if backup_date_dir.exists():
        timestamp_with_time = datetime.now().strftime('%Y_%m_%d_%H%M%S')
        backup_date_dir = BACKUP_DIR / timestamp_with_time

    # Copy all signal files
    backup_date_dir.mkdir(parents=True, exist_ok=True)
    for parquet_file in parquet_files:
        shutil.copy2(parquet_file, backup_date_dir / parquet_file.name)

    print(f"  Backed up {len(parquet_files)} signal files to {backup_date_dir.relative_to(PIPELINE_DIR)}/")

    return str(backup_date_dir)


def compute_and_save_signal_bases(
    start_date: str = None,
    end_date: str = None,
    isins: list = None,
    backup: bool = True
):
    """
    Compute signal bases and save to parquet files.

    Recomputes ALL signal bases from scratch to ensure:
    - Price corrections are captured
    - Rolling window calculations are fresh
    - Data consistency across all signals

    Args:
        start_date: Start date (YYYY-MM-DD). If None, use 2009-09-25 (core inception)
        end_date: End date (YYYY-MM-DD). If None, use today
        isins: List of ISINs to compute (None = all in database)
        backup: If True, backup existing signals before recomputing (default True)

    Returns:
        Dict with computation statistics
    """
    print("=" * 120)
    print("STEP 1: COMPUTE SIGNAL BASES")
    print("=" * 120)
    print(f"  Output:  {SIGNAL_OUTPUT_DIR}/")

    # Initialize databases (database is in maintenance/data folder)
    db_path = project_root / "maintenance" / "data" / "etf_database.db"
    etf_db = ETFDatabase(str(db_path))
    signal_db = SignalDatabase(SIGNAL_OUTPUT_DIR)

    # Determine date range
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if start_date is None:
        # Default: use all available data (from core ETF inception: 2009-09-25)
        start_date = '2009-09-25'

    print(f"\nComputation date range: {start_date} to {end_date}")

    # Load price data
    print("\n" + "-" * 120)
    print("Loading price data...")
    print("-" * 120)

    # Need extra lookback for rolling calculations
    # 400 days ensures full warmup for: 252-day signals + 63-day filters + margin
    lookback_start = (pd.Timestamp(start_date) - timedelta(days=400)).strftime('%Y-%m-%d')

    etf_prices = etf_db.load_all_prices(isins=isins, start_date=lookback_start, end_date=end_date)

    # Load core (MSCI World) prices
    core_isin = 'IE00B4L5Y983'  # iShares Core MSCI World
    if core_isin not in etf_prices.columns:
        print(f"WARNING: Core ETF ({core_isin}) not in price data, loading separately...")
        core_prices = etf_db.load_prices(core_isin, start_date=lookback_start, end_date=end_date)
    else:
        core_prices = etf_prices[core_isin]

    if len(etf_prices) == 0:
        print("ERROR: No price data found")
        return None

    print(f"  Loaded: {len(etf_prices)} days × {len(etf_prices.columns)} ETFs")
    print(f"  Date range: {etf_prices.index[0]} to {etf_prices.index[-1]}")
    print(f"  Core ETF: {core_isin}")

    # Will save all loaded dates
    save_dates = etf_prices.index
    print(f"\n  Will save all {len(save_dates)} days")

    # Backup existing signals before recomputation
    print("\n  Backing up existing signals...")
    backup_signal_bases()

    # Clear existing signals for fresh full recomputation
    print("\n  Clearing existing signals for fresh computation...")
    signal_db.clear_all_signal_bases()

    # Compute and save signals incrementally (TRUE incremental: compute one, save one)
    print("\n" + "-" * 120)
    print("Computing and saving signal bases (parquet format)...")
    print("-" * 120)

    isin_list = list(etf_prices.columns)
    signal_start = time.time()
    n_records = 0
    n_signals = 0

    # Get total signal count upfront for accurate progress bar
    total_signals = count_total_signals()
    print(f"  Total signals to compute: {total_signals}")

    # Compute signals one at a time and save immediately
    # Using tqdm to show progress with accurate total count
    pbar = tqdm(
        compute_signal_bases_generator(etf_prices, core_prices),
        desc="Computing signal bases",
        unit="signal",
        ncols=120,
        total=total_signals
    )

    for signal_name, signal_2d in pbar:
        n_signals += 1

        # Truncate signal name to 30 chars with ... if needed, then pad to fixed width
        max_name_len = 30
        display_name = signal_name if len(signal_name) <= max_name_len else signal_name[:max_name_len-3] + "..."
        # Pad to fixed width so progress bar doesn't jump around
        display_name = display_name.ljust(max_name_len)
        pbar.set_description(f"Computing: {display_name}")

        # Save to parquet (all dates)
        records = signal_db.save_signal_base_from_array(
            signal_name, signal_2d, save_dates, isin_list
        )
        n_records += records

    pbar.close()

    signal_time = time.time() - signal_start

    print(f"\n  Computed and saved {n_signals} signals in {signal_time:.1f}s")
    print(f"  Saved {n_records:,} records")
    if signal_time > 0:
        print(f"  Signals per second: {n_signals / signal_time:.1f}")

    # Log computation
    signal_db.log_computation(
        computation_type='signal_bases',
        start_date=str(save_dates[0].date()),
        end_date=str(save_dates[-1].date()),
        n_etfs=len(etf_prices.columns),
        n_signals=n_signals,
        computation_time=signal_time,
        notes='Full recomputation'
    )

    # Summary
    print("\n" + "=" * 120)
    print("COMPUTATION COMPLETE")
    print("=" * 120)
    print(f"  Date range: {save_dates[0].date()} to {save_dates[-1].date()}")
    print(f"  Days computed: {len(save_dates)}")
    print(f"  ETFs: {len(etf_prices.columns)}")
    print(f"  Signal bases: {n_signals}")
    print(f"  Records saved: {n_records:,}")
    print(f"  Total time: {signal_time:.1f}s ({signal_time/60:.1f} min)")
    print("=" * 120)

    # Show storage stats
    stats = signal_db.get_stats()
    print(f"\nStorage: {SIGNAL_OUTPUT_DIR}/")
    print(f"  Total size: {stats['total_size_mb']:.1f} MB")
    print(f"  Signal files: {stats['unique_signal_bases']}")
    if stats['signal_date_range']:
        print(f"  Date range: {stats['signal_date_range'][0]} to {stats['signal_date_range'][1]}")

    # Return computation statistics
    return {
        'n_days': len(save_dates),
        'n_etfs': len(etf_prices.columns),
        'n_signals': n_signals,
        'start_date': str(save_dates[0].date()),
        'end_date': str(save_dates[-1].date()),
        'total_time': signal_time,
        'records_saved': n_records
    }


if __name__ == "__main__":
    # Show configuration
    print("\n" + "=" * 120)
    print("SIGNAL BASE COMPUTATION")
    print("=" * 120)
    print(f"Output:  {SIGNAL_OUTPUT_DIR}/")
    print("=" * 120)

    print("\nRunning in FULL recomputation mode (all data from scratch)\n")

    # Parse arguments
    backup_flag = '--backup' in sys.argv
    date_args = [a for a in sys.argv[1:] if not a.startswith('-')]

    if len(date_args) >= 2:
        start_date = date_args[0]
        end_date = date_args[1]
        print(f"Custom date range: {start_date} to {end_date}\n")
        stats = compute_and_save_signal_bases(start_date=start_date, end_date=end_date, backup=backup_flag)
    else:
        print("Computing all available data (from 2009-09-25 to today)\n")
        stats = compute_and_save_signal_bases(backup=backup_flag)

    if stats:
        print("\nSignal bases computed and saved successfully!")
        print("Next step: Run 2_apply_filters.py to filter the signals")
