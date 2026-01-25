#!/usr/bin/env python
"""
Step 2: Apply Filters to Signal Bases
======================================

This script applies smoothing filters to the computed signal bases.
Each base signal is filtered with multiple filter types (EMA, Hull MA, etc.)
to create a comprehensive set of filtered signals for downstream analysis.

Features:
- FULL recomputation: Always recomputes from scratch to ensure fresh data
- Automatic backup: Old filtered signals are backed up to backup/2_apply_filters/YYYY_MM_DD/
- GPU acceleration: Uses GPU if available for filter calculations

Output:
    Filtered signals are saved to data/signals/filtered_signals/ as parquet files.
    Each file is named {base_signal}__{filter_name}.parquet

Usage:
  python 2_apply_filters.py                # All filters, full recomputation with automatic backup
  python 2_apply_filters.py --filters ema_21d,ema_63d  # Specific filters with automatic backup
"""

import sys
import time
import argparse
import warnings
import shutil
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread, Lock
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from tqdm import tqdm

# Suppress expected warnings from NaN handling in filters and correlation
warnings.filterwarnings('ignore', message='All-NaN slice encountered')
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Degrees of freedom')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning)  # Suppress all numpy RuntimeWarnings

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from support.signal_database import SignalDatabase
from library_signal_filters import (
    compute_filtered_signals_optimized,
    get_available_filters,
    GPU_AVAILABLE
)

# Output directories
PIPELINE_DIR = Path(__file__).parent
SIGNAL_OUTPUT_DIR = PIPELINE_DIR / 'data' / 'signals'
BACKUP_DIR = PIPELINE_DIR / 'backup' / '2_apply_filters'
DATA_DIR = PIPELINE_DIR / 'data'

# Configuration
N_CORES = max(1, cpu_count() - 1)
CORRELATION_THRESHOLD = 0.1


def backup_filtered_signals() -> str:
    """
    Backup current filtered signals to dated folder.

    Creates: backup/2_apply_filters/YYYY_MM_DD/

    Returns:
        Path to backup directory, or None if no signals to backup
    """
    if not SIGNAL_OUTPUT_DIR.exists():
        return None

    filtered_dir = SIGNAL_OUTPUT_DIR / 'filtered_signals'
    if not filtered_dir.exists():
        return None

    # Check if there are any parquet files
    parquet_files = list(filtered_dir.glob('*.parquet'))
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

    # Copy all filtered signal files
    backup_date_dir.mkdir(parents=True, exist_ok=True)
    for parquet_file in parquet_files:
        shutil.copy2(parquet_file, backup_date_dir / parquet_file.name)

    print(f"  Backed up {len(parquet_files)} filtered signal files to {backup_date_dir.relative_to(PIPELINE_DIR)}/")

    return str(backup_date_dir)



def compute_signal_correlation_and_rankings(filtered_name, signal_data, alpha_df, dates, isins,
                                            date_to_idx, isin_to_idx, rankings, kept_signal_names):
    """
    Compute correlation with forward IR and rankings for a single filtered signal.

    Args:
        filtered_name: Name of filtered signal
        signal_data: Numpy array of signal values (dates × ISINs)
        alpha_df: Forward IR DataFrame
        dates: Date index
        isins: ISIN list
        date_to_idx: Date to index mapping
        isin_to_idx: ISIN to index mapping
        rankings: Ranking matrix (mutable - updated in place)
        kept_signal_names: List of kept signal names (mutable - appended to)

    Returns:
        True if signal passed correlation filter, False otherwise
    """
    # Convert to DataFrame
    df_signal = pd.DataFrame(signal_data, index=dates, columns=isins)

    # Get daily IR values
    daily_ir = alpha_df.groupby('date')['forward_ir'].mean()
    daily_ir.index = pd.to_datetime(daily_ir.index)

    # Find common dates
    common_dates = pd.Index(df_signal.index).intersection(pd.Index(daily_ir.index))

    if len(common_dates) < 2:
        return False

    # Check correlation with forward IR
    signal_vals = df_signal.loc[common_dates].values.flatten()
    ir_vals = np.repeat(daily_ir.loc[common_dates].values, len(isins))
    mask = ~(np.isnan(signal_vals) | np.isnan(ir_vals))

    if mask.sum() < 2:
        return False

    correlation = float(np.corrcoef(signal_vals[mask], ir_vals[mask])[0, 1])

    if abs(correlation) < CORRELATION_THRESHOLD:
        return False

    # Signal passed filter - compute and add rankings
    feat_idx = len(kept_signal_names)  # Current feature index

    for date in common_dates:
        if date not in date_to_idx:
            continue

        date_idx = date_to_idx[date]
        values = df_signal.loc[date].values

        # Z-score normalization
        if len(values) > 1:
            mean_val = np.nanmean(values)
            std_val = np.nanstd(values)
            if std_val > 0:
                z_scores = (values - mean_val) / std_val
                rank_vals = pd.Series(z_scores).rank(pct=True).values
            else:
                rank_vals = pd.Series(values).rank(pct=True).values
        else:
            rank_vals = pd.Series(values).rank(pct=True).values

        # Fill matrix
        for isin, rank_val in zip(isins, rank_vals):
            if isin in isin_to_idx and not np.isnan(rank_val):
                isin_idx = isin_to_idx[isin]
                rankings[date_idx, isin_idx, feat_idx] = rank_val

    kept_signal_names.append(filtered_name)
    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply smoothing filters to signal bases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--filters',
        type=str,
        default=None,
        help='Comma-separated list of specific filters (default: all). Example: ema_21d,ema_63d'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=30,
        help='Number of signals to batch together (default: 30)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel I/O workers (default: 8)'
    )

    return parser.parse_args()

def save_worker(save_queue: Queue, signal_db: SignalDatabase, dates, isins, stats: dict,
                alpha_df=None, date_to_idx=None, isin_to_idx=None, rankings=None,
                kept_signal_names=None, stats_lock=None):
    """
    Worker thread for parallel parquet saving with inline correlation computation.

    Reads (filtered_name, filtered_data) from queue, saves to disk, and computes
    correlation + rankings inline (while signal is in memory).
    """
    while True:
        item = save_queue.get()
        if item is None:  # Poison pill
            save_queue.task_done()
            break

        filtered_name, filtered_data = item
        try:
            # Save to disk
            records = signal_db.save_filtered_signal_from_array(
                filtered_name,
                filtered_data,
                dates,
                isins
            )

            # INLINE: Check correlation with forward IR and build rankings (while signal in memory)
            if alpha_df is not None and rankings is not None:
                compute_signal_correlation_and_rankings(
                    filtered_name, filtered_data, alpha_df, dates, isins,
                    date_to_idx, isin_to_idx, rankings, kept_signal_names
                )

            # Thread-safe stats update
            if stats_lock:
                with stats_lock:
                    stats['records'] += records
                    stats['count'] += 1
            else:
                stats['records'] += records
                stats['count'] += 1

        except Exception as err:
            if stats_lock:
                with stats_lock:
                    stats['errors'].append((filtered_name, str(err)))
            else:
                stats['errors'].append((filtered_name, str(err)))
        finally:
            save_queue.task_done()


def main():
    """Main entry point for filter application."""
    args = parse_args()

    print("=" * 80)
    print("STEP 2: APPLY FILTERS TO SIGNAL BASES")
    print("=" * 80)

    # Initialize database
    signal_db = SignalDatabase(SIGNAL_OUTPUT_DIR)

    # Get base signals (sorted for deterministic ordering)
    base_signal_names = sorted(signal_db.get_completed_signals())

    if not base_signal_names:
        print("\nERROR: No base signals found!")
        print("Please run step 1 first: python 1_compute_signal_bases.py")
        return 1

    print(f"\nFound {len(base_signal_names)} base signals")

    # Get filter names
    if args.filters:
        filter_names = [f.strip() for f in args.filters.split(',')]
        available = get_available_filters()
        invalid = [f for f in filter_names if f not in available]
        if invalid:
            print(f"\nERROR: Invalid filter names: {invalid}")
            print(f"Available filters: {available}")
            return 1
    else:
        filter_names = get_available_filters()

    print(f"Applying {len(filter_names)} filters")

    # Backup existing filtered signals before recomputation
    print("\nBacking up and clearing existing filtered signals...")
    backup_filtered_signals()

    # Always recompute all filtered signals from scratch
    filtered_dir = signal_db.filtered_dir
    if filtered_dir.exists():
        shutil.rmtree(filtered_dir)
    filtered_dir.mkdir(parents=True, exist_ok=True)

    # Calculate totals
    total_combinations = len(base_signal_names) * len(filter_names)

    print(f"\nConfiguration: {total_combinations} combinations ({len(base_signal_names)} signals × {len(filter_names)} filters)")
    print(f"  Batch size: {args.batch_size} | Workers: {args.workers} | GPU: {GPU_AVAILABLE}")

    # Load all base signals into memory (they're needed for batching)
    print("\nLoading base signals into memory...")

    load_start = time.time()
    signal_bases = {}
    dates = None
    isins = None

    for i, signal_name in enumerate(base_signal_names):
        signal_df = signal_db.load_signal_base(signal_name)
        if signal_df is None or len(signal_df) == 0:
            print(f"  WARNING: Could not load {signal_name}, skipping")
            continue

        signal_bases[signal_name] = signal_df.values.astype(np.float32)

        # Store dates and ISINs from first signal (all should be same)
        if dates is None:
            dates = signal_df.index
            isins = signal_df.columns.tolist()

        if (i + 1) % 50 == 0:
            print(f"  Loaded {i+1}/{len(base_signal_names)} signals...")

    load_time = time.time() - load_start
    memory_gb = sum(arr.nbytes for arr in signal_bases.values()) / 1e9
    print(f"  Loaded {len(signal_bases)} signals in {load_time:.1f}s ({memory_gb:.2f} GB in memory)")

    # Initialize ranking matrix components BEFORE starting parallel workers
    print("\nPreparing ranking matrix components...")
    alpha_df = None
    date_to_idx = None
    isin_to_idx = None
    rankings = None
    kept_signal_names = []
    stats_lock = None

    alpha_file = DATA_DIR / 'forward_alpha_1month.parquet'
    if alpha_file.exists():
        alpha_df = pd.read_parquet(alpha_file)
        alpha_df['date'] = pd.to_datetime(alpha_df['date'])

        # Create mappings for ranking matrix
        target_dates = sorted(alpha_df['date'].unique())
        target_dates = pd.to_datetime(target_dates)
        date_to_idx = {date: idx for idx, date in enumerate(target_dates)}
        isin_list_unique = sorted(alpha_df['isin'].unique())
        isin_to_idx = {isin: idx for idx, isin in enumerate(isin_list_unique)}

        # Initialize ranking matrix (will trim after loop)
        # Estimate: using total_combinations as upper bound, will trim after filtering
        rankings = np.full((len(target_dates), len(isin_list_unique), total_combinations), np.nan, dtype=np.float32)
        print(f"  Ranking matrix initialized: {rankings.shape} (will trim after filtering)")
        stats_lock = Lock()  # For thread-safe list operations
    else:
        print(f"  WARNING: Forward IR not found at {alpha_file}")
        print("  Skipping ranking matrix creation. Run Step 0 first if needed.")

    # Set up parallel saving
    print("\nApplying filters with batch processing...")

    save_queue = Queue(maxsize=args.workers * 2)
    stats = {'count': 0, 'records': 0, 'errors': []}

    # Start save workers
    save_threads = []
    for _ in range(args.workers):
        t = Thread(target=save_worker, args=(save_queue, signal_db, dates, isins, stats,
                                             alpha_df, date_to_idx, isin_to_idx, rankings,
                                             kept_signal_names, stats_lock))
        t.daemon = True
        t.start()
        save_threads.append(t)

    # Process with optimized batching
    start_time = time.time()
    print(f"Computing and saving {total_combinations} filtered signals...")

    for filtered_name, filtered_data in compute_filtered_signals_optimized(
        signal_bases,
        filter_names=filter_names,
        skip_signals=set(),  # Always compute all
        batch_size=args.batch_size,
        show_progress=True
    ):
        # Queue for parallel saving
        save_queue.put((filtered_name, filtered_data))

    # Signal workers to stop
    for _ in save_threads:
        save_queue.put(None)

    # Wait for all saves to complete
    save_queue.join()

    elapsed = time.time() - start_time

    # Report errors
    if stats['errors']:
        print(f"\nWARNING: {len(stats['errors'])} errors occurred during saving:")
        for name, err in stats['errors'][:5]:
            print(f"  {name}: {err}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")

    print("\n" + "=" * 80)
    print("FILTER APPLICATION COMPLETE")
    print("=" * 80)
    print(f"  Signals: {stats['count']} | Records: {stats['records']:,}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min) | Throughput: {stats['count']/elapsed:.1f} signals/sec")

    # Show storage summary
    filtered_dir = signal_db.filtered_dir
    if filtered_dir.exists():
        total_size = sum(f.stat().st_size for f in filtered_dir.glob("*.parquet"))
        total_files = len(list(filtered_dir.glob("*.parquet")))
        print(f"  Storage: {total_size/1e9:.2f} GB ({total_files} files)")

    print("=" * 80)

    # Save ranking matrix (computed inline during filter application)
    if rankings is not None and len(kept_signal_names) > 0:
        print("\nSaving ranking matrix...")

        # Trim rankings to only kept signals
        rankings = rankings[:, :, :len(kept_signal_names)]

        # Save ranking matrix
        matrix_file = DATA_DIR / 'rankings_matrix_filtered_1month.npz'
        target_dates = sorted(alpha_df['date'].unique())
        target_dates = pd.to_datetime(target_dates)
        np.savez_compressed(
            matrix_file,
            rankings=rankings,
            dates=np.array(target_dates),
            isins=np.array(sorted(alpha_df['isin'].unique())),
            features=np.array(kept_signal_names),
            n_filtered=len(kept_signal_names)
        )

        print(f"  Ranking matrix: {len(kept_signal_names)} signals passed correlation filter (from {stats['count']})")
        print(f"  Shape: {rankings.shape}")
        print(f"  [SAVED] {matrix_file}")
    elif rankings is None:
        print("\nSkipping ranking matrix (forward IR not available)")
    elif len(kept_signal_names) == 0:
        print("\nNo signals passed correlation filter for ranking matrix")

    print("=" * 80)
    print("STEP 2 COMPLETE")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
