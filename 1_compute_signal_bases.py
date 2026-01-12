"""
Step 1: Compute Signal Bases

This script:
1. Loads ETF prices from the database
2. Computes all 293 signal bases
3. Saves them incrementally to parquet files (each signal saved immediately)

Run this FIRST before filtering.

Incremental saving benefits:
- Reduced peak memory usage (don't need to hold all signals in memory)
- Resume capability if interrupted (already-saved signals persist)
- Real-time progress feedback
- ~10-50x faster writes than SQLite
"""

import time
from datetime import datetime, timedelta
import pandas as pd

from support.etf_database import ETFDatabase
from support.signal_database import SignalDatabase
from library_signal_bases import compute_signal_bases_generator

SIGNAL_OUTPUT_DIR = "data/signals"


def compute_and_save_signal_bases(
    start_date: str = None,
    end_date: str = None,
    isins: list = None,
    incremental: bool = False,
    resume: bool = True
):
    """
    Compute signal bases and save to parquet files incrementally.

    Args:
        start_date: Start date (YYYY-MM-DD). If None, use last computed date + 1
        end_date: End date (YYYY-MM-DD). If None, use today
        isins: List of ISINs to compute (None = all in database)
        incremental: If True, only compute new dates. If False, replace all.
        resume: If True, skip signals already saved (allows resuming interrupted runs)

    Returns:
        Dict with computation statistics
    """
    print("=" * 80)
    print("STEP 1: COMPUTE SIGNAL BASES")
    print("=" * 80)
    print(f"  Output:  {SIGNAL_OUTPUT_DIR}/")

    # Initialize databases
    etf_db = ETFDatabase("data/etf_database.db")
    signal_db = SignalDatabase(SIGNAL_OUTPUT_DIR)

    # Check for existing signals if resuming
    completed_signals = set()
    if resume and not incremental:
        completed_signals = signal_db.get_completed_signals()
        if completed_signals:
            print(f"\nResume mode: Found {len(completed_signals)} already-computed signals")

    # Determine date range
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if incremental and start_date is None:
        # Check last computed date from any signal
        available = signal_db.get_available_signals()
        if available:
            date_range = signal_db.get_signal_date_range(available[0])
            if date_range:
                last_date = date_range[1]
                start_date = (pd.Timestamp(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')
                print(f"\nIncremental mode: Last computed date = {last_date}")
                print(f"                  Computing from {start_date} to {end_date}")

        if start_date is None:
            print("\nNo existing signals found - performing full computation")
            incremental = False

    if start_date is None:
        # Default: use all available data (from core ETF inception: 2009-09-25)
        start_date = '2009-09-25'
        print(f"\nFull computation mode: {start_date} to {end_date}")

    # Load price data
    print("\n" + "-" * 80)
    print("Loading price data...")
    print("-" * 80)

    # Need extra lookback for rolling calculations (252 days = 1 year)
    lookback_start = (pd.Timestamp(start_date) - timedelta(days=300)).strftime('%Y-%m-%d')

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

    # Determine what dates to save
    if incremental:
        new_date_mask = etf_prices.index >= start_date
        save_dates = etf_prices.index[new_date_mask]
        print(f"\n  Incremental mode: Will save {len(save_dates)} new days")
    else:
        save_dates = etf_prices.index
        print(f"\n  Full mode: Will save all {len(save_dates)} days")

    # Clear existing signals if not resuming and not incremental
    if not resume and not incremental:
        print("\n  Clearing existing signals for fresh computation...")
        signal_db.clear_all_signal_bases()
        completed_signals = set()

    # Compute and save signals incrementally (TRUE incremental: compute one, save one)
    print("\n" + "-" * 80)
    print("Computing and saving signal bases (parquet format)...")
    print("-" * 80)

    isin_list = list(etf_prices.columns)
    signal_start = time.time()
    n_records = 0
    n_signals = 0
    n_skipped = len(completed_signals)
    signal_names = []

    # Use generator to compute signals one at a time
    # Pass skip_signals to avoid recomputing already-saved signals
    for signal_name, signal_2d in compute_signal_bases_generator(
        etf_prices, core_prices, skip_signals=completed_signals
    ):
        signal_names.append(signal_name)
        n_signals += 1

        # For incremental mode, slice to only new dates
        if incremental:
            new_date_mask = etf_prices.index >= start_date
            signal_2d = signal_2d[new_date_mask, :]
            signal_dates = save_dates
        else:
            signal_dates = etf_prices.index

        # Save to parquet immediately (much faster than SQLite!)
        records = signal_db.save_signal_base_from_array(
            signal_name, signal_2d, signal_dates, isin_list
        )
        n_records += records

    signal_time = time.time() - signal_start

    print(f"\n  Computed and saved {n_signals} signals in {signal_time:.1f}s")
    print(f"  Saved {n_records:,} records")
    if n_skipped > 0:
        print(f"  Skipped {n_skipped} already-computed signals (resume mode)")
    if signal_time > 0:
        print(f"  Signals per second: {n_signals / signal_time:.1f}")

    # Log computation
    signal_db.log_computation(
        computation_type='signal_bases',
        start_date=str(save_dates[0].date()),
        end_date=str(save_dates[-1].date()),
        n_etfs=len(etf_prices.columns),
        n_signals=n_signals + n_skipped,
        computation_time=signal_time,
        notes=f"{'Incremental' if incremental else 'Full'}{' (resumed)' if n_skipped > 0 else ''}"
    )

    # Summary
    print("\n" + "=" * 80)
    print("COMPUTATION COMPLETE")
    print("=" * 80)
    print(f"  Mode: {'Incremental' if incremental else 'Full'}{' (resumed)' if n_skipped > 0 else ''}")
    print(f"  Date range: {save_dates[0].date()} to {save_dates[-1].date()}")
    print(f"  Days computed: {len(save_dates)}")
    print(f"  ETFs: {len(etf_prices.columns)}")
    print(f"  Signal bases: {n_signals + n_skipped}")
    print(f"  Signals computed+saved: {n_signals}")
    print(f"  Signals skipped (resume): {n_skipped}")
    print(f"  Records saved: {n_records:,}")
    print(f"  Total time: {signal_time:.1f}s ({signal_time/60:.1f} min)")
    print("=" * 80)

    # Show storage stats
    stats = signal_db.get_stats()
    print(f"\nStorage: {SIGNAL_OUTPUT_DIR}/")
    print(f"  Total size: {stats['total_size_mb']:.1f} MB")
    print(f"  Signal files: {stats['unique_signal_bases']}")
    if stats['signal_date_range']:
        print(f"  Date range: {stats['signal_date_range'][0]} to {stats['signal_date_range'][1]}")

    return {
        'mode': 'incremental' if incremental else 'full',
        'resumed': n_skipped > 0,
        'start_date': str(save_dates[0].date()),
        'end_date': str(save_dates[-1].date()),
        'n_days': len(save_dates),
        'n_etfs': len(etf_prices.columns),
        'n_signals': n_signals + n_skipped,
        'n_signals_saved': n_signals,
        'n_skipped': n_skipped,
        'total_time': signal_time,
        'records_saved': n_records
    }


if __name__ == "__main__":
    import sys

    # Show configuration
    print("\n" + "=" * 80)
    print("SIGNAL BASE COMPUTATION")
    print("=" * 80)
    print(f"Output:  {SIGNAL_OUTPUT_DIR}/")
    print("=" * 80)

    # Parse command line arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'

    if mode == 'incremental':
        print("\nRunning in INCREMENTAL mode (compute only new dates)\n")
        stats = compute_and_save_signal_bases(incremental=True)
    elif mode == 'full':
        print("\nRunning in FULL mode (recompute all data)\n")

        # Check for --no-resume flag
        no_resume = '--no-resume' in sys.argv

        # Optional: specify date range
        date_args = [a for a in sys.argv[2:] if not a.startswith('--')]
        if len(date_args) >= 2:
            start_date = date_args[0]
            end_date = date_args[1]
            print(f"Custom date range: {start_date} to {end_date}\n")
            stats = compute_and_save_signal_bases(
                start_date=start_date,
                end_date=end_date,
                incremental=False,
                resume=not no_resume
            )
        else:
            print("Computing all available data\n")
            if no_resume:
                print("(--no-resume: will recompute all signals from scratch)\n")
            stats = compute_and_save_signal_bases(
                incremental=False,
                resume=not no_resume
            )
    elif mode == 'resume':
        print("\nRunning in RESUME mode (continue from last saved signal)\n")
        stats = compute_and_save_signal_bases(incremental=False, resume=True)
    else:
        print("Usage:")
        print("  python 1_compute_signal_bases.py full              # Full computation (all data)")
        print("  python 1_compute_signal_bases.py full --no-resume  # Full, recompute all signals")
        print("  python 1_compute_signal_bases.py full 2020-01-01 2024-12-31  # Custom range")
        print("  python 1_compute_signal_bases.py incremental       # Only new dates")
        print("  python 1_compute_signal_bases.py resume            # Continue interrupted run")
        sys.exit(1)

    if stats:
        print("\n Signal bases computed and saved successfully!")
        print("\nNext step: Run 2_apply_filters.py to filter the signals")
