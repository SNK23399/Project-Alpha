#!/usr/bin/env python
"""
Step 2: Apply Filters to Signal Bases (Walk-Forward Pipeline - ALL FEATURES VERSION)
====================================================================================

This script applies smoothing filters to the computed signal bases.
Each base signal is filtered with multiple filter types (EMA, Hull MA, etc.)
to create a comprehensive set of filtered signals for downstream analysis.

NOTE: This script is SHARED with the filtered version - filter application
has no look-ahead bias (just transforms signal data). Outputs go to:
    walk forward backtest all features/data/signals/filtered_signals/

This generates ALL 7,911 possible filter combinations (293 signals × 27 filters).
The filtered version in 'walk forward backtest/' pre-selects the best 500 by
momentum alpha in step 4, whereas this ALL FEATURES version uses all of them.

Output:
    Filtered signals are saved to data/signals/filtered_signals/ as parquet files.
    Each file is named {base_signal}__{filter_name}.parquet
"""

import sys
import time
import argparse
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread
import numpy as np

# Suppress expected warnings from NaN handling in filters
warnings.filterwarnings('ignore', message='All-NaN slice encountered')
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Degrees of freedom')

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from support.signal_database import SignalDatabase
from library_signal_filters import (
    compute_filtered_signals_optimized,
    get_available_filters,
    print_available_filters,
    GPU_AVAILABLE
)

# Output directory (relative to this script's location)
SIGNAL_OUTPUT_DIR = Path(__file__).parent / 'data' / 'signals'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply smoothing filters to signal bases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'mode',
        nargs='?',
        default='full',
        choices=['full', 'incremental'],
        help='Processing mode (default: full)'
    )

    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Ignore existing filtered signals, recompute everything'
    )

    parser.add_argument(
        '--filters',
        type=str,
        default=None,
        help='Comma-separated list of filters to apply (default: all)'
    )

    parser.add_argument(
        '--signals',
        type=str,
        default=None,
        help='Comma-separated list of base signals to filter (default: all)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=30,  # Smaller batches to avoid GPU OOM
        help='Number of signals to batch together (default: 30)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel I/O workers (default: 8)'
    )

    parser.add_argument(
        '--list-filters',
        action='store_true',
        help='List all available filters and exit'
    )

    return parser.parse_args()


def save_worker(save_queue: Queue, signal_db: SignalDatabase, dates, isins, stats: dict):
    """
    Worker thread for parallel parquet saving.

    Reads (filtered_name, filtered_data) from queue and saves to parquet.
    """
    while True:
        item = save_queue.get()
        if item is None:  # Poison pill
            save_queue.task_done()
            break

        filtered_name, filtered_data = item
        try:
            records = signal_db.save_filtered_signal_from_array(
                filtered_name,
                filtered_data,
                dates,
                isins
            )
            stats['records'] += records
            stats['count'] += 1
        except Exception as e:
            stats['errors'].append((filtered_name, str(e)))
        finally:
            save_queue.task_done()


def main():
    """Main entry point for filter application."""
    args = parse_args()

    # If user wants to see available filters
    if args.list_filters:
        print_available_filters()
        return 0

    print("=" * 80)
    print("STEP 2: APPLY FILTERS TO SIGNAL BASES (OPTIMIZED)")
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

    # Filter to specific signals if requested
    if args.signals:
        requested_signals = [s.strip() for s in args.signals.split(',')]
        base_signal_names = [s for s in base_signal_names if s in requested_signals]
        print(f"Filtering to {len(base_signal_names)} requested signals")

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

    # Always recompute all filtered signals (no resume)
    print("\n(Recomputing all filtered signals)")
    # Clear existing filtered signals
    filtered_dir = signal_db.filtered_dir
    if filtered_dir.exists():
        import shutil
        shutil.rmtree(filtered_dir)
        filtered_dir.mkdir(parents=True, exist_ok=True)
    completed_filtered = set()

    # Calculate totals
    total_combinations = len(base_signal_names) * len(filter_names)
    to_skip = len([s for s in completed_filtered
                   if any(s.startswith(f"{bn}__") for bn in base_signal_names)])
    to_compute = total_combinations - to_skip

    print(f"\nConfiguration:")
    print(f"  Total combinations: {len(base_signal_names)} signals × {len(filter_names)} filters = {total_combinations}")
    print(f"  Already computed: {to_skip}")
    print(f"  To compute: {to_compute}")
    print(f"  Batch size: {args.batch_size} signals")
    print(f"  I/O workers: {args.workers}")
    print(f"  GPU available: {GPU_AVAILABLE}")

    if to_compute == 0:
        print("\nAll filtered signals already computed. Nothing to do.")
        print("Use --no-resume to force recomputation.")
        return 0

    # Load all base signals into memory (they're needed for batching)
    print("\n" + "-" * 80)
    print("Loading base signals into memory...")
    print("-" * 80)

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

    # Set up parallel saving
    print("\n" + "-" * 80)
    print("Applying filters with batch processing...")
    print("-" * 80)

    save_queue = Queue(maxsize=args.workers * 2)  # Limit queue size to control memory
    stats = {'count': 0, 'records': 0, 'errors': []}

    # Start save workers
    save_threads = []
    for _ in range(args.workers):
        t = Thread(target=save_worker, args=(save_queue, signal_db, dates, isins, stats))
        t.daemon = True
        t.start()
        save_threads.append(t)

    # Process with optimized batching (tqdm progress bar is built-in)
    start_time = time.time()

    print(f"\n  Processing {to_compute} filtered signals...")

    for filtered_name, filtered_data in compute_filtered_signals_optimized(
        signal_bases,
        filter_names=filter_names,
        skip_signals=completed_filtered,
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
    print(f"  Filtered signals computed: {stats['count']}")
    print(f"  Total records saved: {stats['records']:,}")
    print(f"  Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Throughput: {stats['count']/elapsed:.1f} signals/sec")

    # Show storage summary
    filtered_dir = signal_db.filtered_dir
    if filtered_dir.exists():
        total_size = sum(f.stat().st_size for f in filtered_dir.glob("*.parquet"))
        total_files = len(list(filtered_dir.glob("*.parquet")))
        print(f"\nStorage: {filtered_dir}")
        print(f"  Total size: {total_size/1e9:.2f} GB")
        print(f"  Total files: {total_files}")

    print("=" * 80)
    print(f"\nFiltered signals saved to {SIGNAL_OUTPUT_DIR / 'filtered_signals'}/")
    print("Next step: Run 3_backtesting.py to evaluate strategies")

    return 0


if __name__ == '__main__':
    sys.exit(main())
