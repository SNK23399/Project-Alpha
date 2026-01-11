"""
Step 1: Compute Signal Bases

This script:
1. Loads ETF prices from the database
2. Computes all 293 signal bases
3. Saves them to the signal_bases table

Run this FIRST before filtering.
"""

import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from support.etf_database import ETFDatabase
from signal_bases_optimized_full import compute_all_signal_bases

# Note: Using optimized version with parallelization (5-8x faster)
# Computes ALL 293 signals using industry-standard libraries
# Parallelizes expensive operations (Ulcer, CVaR, Sortino, Calmar) across ETFs


def compute_and_save_signal_bases(
    start_date: str = None,
    end_date: str = None,
    isins: list = None,
    incremental: bool = False
):
    """
    Compute signal bases and save to database.

    Args:
        start_date: Start date (YYYY-MM-DD). If None, use last computed date + 1
        end_date: End date (YYYY-MM-DD). If None, use today
        isins: List of ISINs to compute (None = all in database)
        incremental: If True, only compute new dates. If False, replace all.

    Returns:
        Dict with computation statistics
    """
    print("=" * 80)
    print("STEP 1: COMPUTE SIGNAL BASES")
    print("=" * 80)

    # Initialize ETF database (for reading price data)
    db = ETFDatabase("data/etf_database.db")
    # Note: Signals saved to separate database: data/signal_database.db

    # Determine date range
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if incremental and start_date is None:
        # Check last computed date
        existing_range = signal_db.get_signal_base_date_range('ret_1d')
        if existing_range:
            last_date = existing_range[1]
            start_date = (pd.Timestamp(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"\nIncremental mode: Last computed date = {last_date}")
            print(f"                  Computing from {start_date} to {end_date}")
        else:
            print("\nNo existing signals found - performing full computation")
            incremental = False

    if start_date is None:
        # Default: last 5 years
        start_date = (pd.Timestamp(end_date) - timedelta(days=1260)).strftime('%Y-%m-%d')
        print(f"\nFull computation mode: {start_date} to {end_date}")

    # Load price data
    print("\n" + "-" * 80)
    print("Loading price data...")
    print("-" * 80)

    # Need extra lookback for rolling calculations (252 days = 1 year)
    lookback_start = (pd.Timestamp(start_date) - timedelta(days=300)).strftime('%Y-%m-%d')

    etf_prices = db.load_all_prices(isins=isins, start_date=lookback_start, end_date=end_date)

    # Load core (MSCI World) prices
    core_isin = 'IE00B4L5Y983'  # iShares Core MSCI World
    if core_isin not in etf_prices.columns:
        print(f"WARNING: Core ETF ({core_isin}) not in price data, loading separately...")
        core_prices = db.load_prices(core_isin, start_date=lookback_start, end_date=end_date)
    else:
        core_prices = etf_prices[core_isin]

    if len(etf_prices) == 0:
        print("ERROR: No price data found")
        return None

    print(f"  Loaded: {len(etf_prices)} days × {len(etf_prices.columns)} ETFs")
    print(f"  Date range: {etf_prices.index[0]} to {etf_prices.index[-1]}")
    print(f"  Core ETF: {core_isin}")

    # Compute signal bases
    print("\n" + "-" * 80)
    print("Computing signal bases...")
    print("-" * 80)

    signal_start = time.time()
    signals_3d, signal_names = compute_all_signal_bases(etf_prices, core_prices)
    signal_time = time.time() - signal_start

    print(f"\n  Computed {len(signal_names)} signals in {signal_time:.1f}s")
    print(f"  Signal array shape: {signals_3d.shape}")
    print(f"  Signal array size: {signals_3d.nbytes / 1024**3:.2f} GB")

    # Determine what to save
    if incremental:
        new_date_mask = etf_prices.index >= start_date
        save_dates = etf_prices.index[new_date_mask]
        save_signals_3d = signals_3d[:, new_date_mask, :]
        print(f"\n  Incremental mode: Saving {len(save_dates)} new days")
    else:
        save_dates = etf_prices.index
        save_signals_3d = signals_3d
        print(f"\n  Full mode: Saving all {len(save_dates)} days")

    # Save to database - ONE SIGNAL AT A TIME using fast pandas method
    print("\n" + "-" * 80)
    print("Saving to database (fast per-signal method)...")
    print("-" * 80)

    import sqlite3

    save_start = time.time()
    n_records = 0
    n_signals = len(signal_names)
    isins = list(etf_prices.columns)

    # Connect directly for fast pandas to_sql
    conn = sqlite3.connect("data/signal_database.db")

    # Create table if not exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_bases (
            signal_name TEXT,
            isin TEXT,
            date TEXT,
            value REAL,
            PRIMARY KEY (signal_name, isin, date)
        )
    """)

    # Delete existing if replacing
    if not incremental:
        print("  Clearing existing signal data...")
        conn.execute("DELETE FROM signal_bases")
        conn.commit()

    # Save each signal using fast pandas method
    dates_str = save_dates.strftime('%Y-%m-%d')

    for sig_idx, signal_name in enumerate(signal_names):
        # Get this signal as DataFrame
        signal_2d = save_signals_3d[sig_idx]  # (n_time, n_etfs)

        # Convert to long format DataFrame (fast with pandas melt)
        df = pd.DataFrame(signal_2d, index=dates_str, columns=isins)
        df.index.name = 'date'
        df_long = df.reset_index().melt(id_vars='date', var_name='isin', value_name='value')
        df_long['signal_name'] = signal_name

        # Drop NaN values
        df_long = df_long.dropna(subset=['value'])

        # Reorder columns
        df_long = df_long[['signal_name', 'isin', 'date', 'value']]

        # Fast insert using pandas (much faster than executemany)
        df_long.to_sql('signal_bases', conn, if_exists='append', index=False, method='multi')

        n_records += len(df_long)

        # Progress update
        if (sig_idx + 1) % 10 == 0:
            elapsed = time.time() - save_start
            rate = (sig_idx + 1) / elapsed
            eta = (n_signals - sig_idx - 1) / rate if rate > 0 else 0
            print(f"    [Saved {sig_idx + 1}/{n_signals} signals... ETA: {eta:.0f}s]")

    # Create index for fast queries
    print("  Creating index...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_name_date ON signal_bases(signal_name, date)")
    conn.commit()
    conn.close()

    save_time = time.time() - save_start
    print(f"  Saved {n_records:,} records in {save_time:.1f}s")
    if save_time > 0:
        print(f"  Records per second: {n_records / save_time:,.0f}")

    # Log computation
    total_time = time.time() - signal_start

    # Summary
    print("\n" + "=" * 80)
    print("COMPUTATION COMPLETE")
    print("=" * 80)
    print(f"  Mode: {'Incremental' if incremental else 'Full'}")
    print(f"  Date range: {save_dates[0].date()} to {save_dates[-1].date()}")
    print(f"  Days computed: {len(save_dates)}")
    print(f"  ETFs: {len(etf_prices.columns)}")
    print(f"  Signal bases: {len(signal_names)}")
    print(f"  Records saved: {n_records:,}")
    print(f"  Computation time: {signal_time:.1f}s ({signal_time/60:.1f} min)")
    print(f"  Database save time: {save_time:.1f}s ({save_time/60:.1f} min)")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 80)

    # Show database stats
    import os
    db_size = os.path.getsize("data/signal_database.db") / (1024 * 1024)
    print(f"\nDatabase: data/signal_database.db")
    print(f"  Size: {db_size:.1f} MB")
    print(f"  Signals: {len(signal_names)}")
    print(f"  Records: {n_records:,}")

    return {
        'mode': 'incremental' if incremental else 'full',
        'start_date': str(save_dates[0].date()),
        'end_date': str(save_dates[-1].date()),
        'n_days': len(save_dates),
        'n_etfs': len(etf_prices.columns),
        'n_signals': len(signal_names),
        'computation_time': signal_time,
        'save_time': save_time,
        'total_time': total_time,
        'records_saved': n_records
    }


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'

    if mode == 'incremental':
        print("\nRunning in INCREMENTAL mode (compute only new dates)\n")
        stats = compute_and_save_signal_bases(incremental=True)
    elif mode == 'full':
        print("\nRunning in FULL mode (recompute all data)\n")

        # Optional: specify date range
        if len(sys.argv) >= 4:
            start_date = sys.argv[2]
            end_date = sys.argv[3]
            print(f"Custom date range: {start_date} to {end_date}\n")
            stats = compute_and_save_signal_bases(
                start_date=start_date,
                end_date=end_date,
                incremental=False
            )
        else:
            print("Computing last 5 years of data\n")
            stats = compute_and_save_signal_bases(incremental=False)
    else:
        print("Usage:")
        print("  python 1_compute_signal_bases.py full              # Full computation (5 years)")
        print("  python 1_compute_signal_bases.py full 2020-01-01 2024-12-31  # Custom range")
        print("  python 1_compute_signal_bases.py incremental      # Only new dates")
        sys.exit(1)

    if stats:
        print("\n✓ Signal bases computed and saved successfully!")
        print("\nNext step: Run 2_apply_filters.py to filter the signals")
