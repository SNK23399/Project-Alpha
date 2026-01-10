"""
Step 2: Apply Filters to Signal Bases

This script:
1. Loads signal bases from the database
2. Applies all 27 filters
3. Saves filtered signals to the database

Run this AFTER computing signal bases (step 1).
"""

import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from support.etf_database import ETFDatabase
from support.signal_database import SignalDatabase
from signal_filters import DEFAULT_FILTER_CONFIGS


def apply_and_save_filters(
    signal_names: list = None,
    filter_names: list = None,
    start_date: str = None,
    end_date: str = None,
    incremental: bool = False,
    batch_size: int = 10
):
    """
    Apply filters to signal bases and save to database.

    Args:
        signal_names: List of signal names to filter (None = all)
        filter_names: List of filter names to apply (None = all 27 filters)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        incremental: If True, only filter new dates
        batch_size: Number of signals to process at once (memory management)

    Returns:
        Dict with filtering statistics
    """
    print("=" * 80)
    print("STEP 2: APPLY FILTERS TO SIGNAL BASES")
    print("=" * 80)

    # Initialize databases
    db = ETFDatabase("data/etf_database.db")
    signal_db = SignalDatabase("data/etf_database.db")

    # Get available signal bases
    stats = signal_db.get_stats()
    if stats['signal_base_records'] == 0:
        print("\nERROR: No signal bases found in database!")
        print("Please run 1_compute_signal_bases.py first")
        return None

    print(f"\nDatabase state:")
    print(f"  Signal base records: {stats['signal_base_records']:,}")
    print(f"  Unique signals: {stats['unique_signal_bases']}")
    print(f"  Date range: {stats['signal_date_range']}")

    # Determine which signals to filter
    if signal_names is None:
        # Get list of available signals from database
        conn = signal_db._get_connection()
        try:
            rows = conn.execute("SELECT DISTINCT signal_name FROM signal_bases").fetchall()
            signal_names = [row[0] for row in rows]
        finally:
            conn.close()

    print(f"\n  Signals to filter: {len(signal_names)}")

    # Determine which filters to apply
    if filter_names is None:
        filter_configs = DEFAULT_FILTER_CONFIGS
    else:
        filter_configs = [
            (fname, func, kwargs)
            for fname, func, kwargs in DEFAULT_FILTER_CONFIGS
            if fname in filter_names
        ]

    print(f"  Filters to apply: {len(filter_configs)}")

    # Determine date range
    if end_date is None:
        end_date = stats['signal_date_range'][1] if stats['signal_date_range'] else None

    if incremental and start_date is None:
        # Check last filtered date
        # TODO: Could track this per filter, but for simplicity use signal_bases range
        existing_range = signal_db.get_signal_base_date_range(signal_names[0])
        if existing_range:
            # Filter last month of data (to be safe with overlapping windows)
            start_date = (pd.Timestamp(existing_range[1]) - timedelta(days=30)).strftime('%Y-%m-%d')
            print(f"\nIncremental mode: Filtering from {start_date}")
        else:
            incremental = False

    if start_date is None:
        start_date = stats['signal_date_range'][0] if stats['signal_date_range'] else None

    print(f"  Date range: {start_date} to {end_date}")

    # Apply filters
    print("\n" + "-" * 80)
    print("Applying filters...")
    print("-" * 80)

    total_start = time.time()
    total_records = 0
    filter_times = {}

    for filter_idx, (filter_name, filter_func, filter_kwargs) in enumerate(filter_configs, 1):
        print(f"\n[{filter_idx:2d}/{len(filter_configs)}] Filter: {filter_name}")
        print("-" * 80)

        filter_start = time.time()
        filter_records = 0

        # Process signals in batches to manage memory
        for batch_start in range(0, len(signal_names), batch_size):
            batch_end = min(batch_start + batch_size, len(signal_names))
            batch_signals = signal_names[batch_start:batch_end]

            print(f"  Processing signals {batch_start+1}-{batch_end}/{len(signal_names)}...", end=" ")

            # Load signal bases for this batch
            signals_3d, loaded_names, dates, isins = signal_db.load_signal_bases(
                batch_signals,
                start_date=start_date,
                end_date=end_date,
                as_3d_array=True
            )

            if len(loaded_names) == 0:
                print("(no data)")
                continue

            # Need lookback for filter warm-up
            # Load additional history if needed
            if start_date:
                lookback_start = (pd.Timestamp(start_date) - timedelta(days=100)).strftime('%Y-%m-%d')
                signals_full_3d, _, dates_full, _ = signal_db.load_signal_bases(
                    batch_signals,
                    start_date=lookback_start,
                    end_date=end_date,
                    as_3d_array=True
                )

                if len(dates_full) > len(dates):
                    # Use full data for filtering (includes warm-up)
                    signals_3d = signals_full_3d
                    dates = dates_full

            # Apply filter to each signal in batch
            for sig_idx, sig_name in enumerate(loaded_names):
                signal_data = signals_3d[sig_idx:sig_idx+1, :, :]  # (1, n_time, n_etfs)

                # Apply filter
                filtered = filter_func(signal_data, **filter_kwargs)

                # If incremental, only save new dates
                if incremental and start_date:
                    save_mask = dates >= start_date
                    save_dates = dates[save_mask]
                    save_filtered = filtered[:, save_mask, :]
                else:
                    save_dates = dates
                    save_filtered = filtered

                # Save to database
                n_records = signal_db.update_filtered_signals(
                    sig_name,
                    filter_name,
                    save_filtered,
                    save_dates,
                    isins,
                    replace=(not incremental)
                )

                filter_records += n_records

            print(f"({filter_records:,} records)")

        filter_time = time.time() - filter_start
        filter_times[filter_name] = filter_time
        total_records += filter_records

        print(f"  Filter completed in {filter_time:.1f}s ({filter_records:,} records)")

    total_time = time.time() - total_start

    # Log computation
    signal_db.log_computation(
        computation_type="filtering_incremental" if incremental else "filtering_full",
        start_date=start_date,
        end_date=end_date,
        n_etfs=len(isins),
        n_signals=len(signal_names) * len(filter_configs),
        computation_time=total_time,
        notes=f"Applied {len(filter_configs)} filters to {len(signal_names)} signals"
    )

    # Summary
    print("\n" + "=" * 80)
    print("FILTERING COMPLETE")
    print("=" * 80)
    print(f"  Mode: {'Incremental' if incremental else 'Full'}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Signal bases: {len(signal_names)}")
    print(f"  Filters: {len(filter_configs)}")
    print(f"  Total signal variants: {len(signal_names) * len(filter_configs)}")
    print(f"  Records saved: {total_records:,}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Records per second: {total_records / total_time:,.0f}")
    print("=" * 80)

    # Show slowest and fastest filters
    sorted_filters = sorted(filter_times.items(), key=lambda x: x[1], reverse=True)
    print(f"\nSlowest 5 filters:")
    for i, (fname, ftime) in enumerate(sorted_filters[:5], 1):
        print(f"  {i}. {fname:25s} {ftime:6.1f}s")

    print(f"\nFastest 5 filters:")
    for i, (fname, ftime) in enumerate(sorted_filters[-5:], 1):
        print(f"  {i}. {fname:25s} {ftime:6.1f}s")

    # Show database stats
    stats = signal_db.get_stats()
    print(f"\nDatabase state after filtering:")
    print(f"  Signal base records: {stats['signal_base_records']:,}")
    print(f"  Filtered signal records: {stats['filtered_signal_records']:,}")
    print(f"  Unique filters: {stats['unique_filters']}")
    print(f"  Database size: {stats['db_size_mb']:.1f} MB")

    return {
        'mode': 'incremental' if incremental else 'full',
        'start_date': start_date,
        'end_date': end_date,
        'n_signals': len(signal_names),
        'n_filters': len(filter_configs),
        'total_variants': len(signal_names) * len(filter_configs),
        'total_time': total_time,
        'records_saved': total_records,
        'filter_times': filter_times
    }


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'

    if mode == 'incremental':
        print("\nRunning in INCREMENTAL mode (filter only new dates)\n")
        stats = apply_and_save_filters(incremental=True)
    elif mode == 'full':
        print("\nRunning in FULL mode (filter all data)\n")
        stats = apply_and_save_filters(incremental=False)
    elif mode == 'test':
        print("\nRunning in TEST mode (single signal, single filter)\n")
        # Test with just one signal and one filter
        stats = apply_and_save_filters(
            signal_names=['ret_1d'],
            filter_names=['ema_21d'],
            incremental=False
        )
    else:
        print("Usage:")
        print("  python 2_apply_filters.py full         # Filter all signal bases")
        print("  python 2_apply_filters.py incremental  # Filter only new dates")
        print("  python 2_apply_filters.py test         # Test with one signal/filter")
        sys.exit(1)

    if stats:
        print("\n✓ Filters applied and saved successfully!")
        print("\nNext step: Run 3_compute_features.py to compute final features")
