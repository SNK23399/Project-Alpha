"""
Step 2: Compute Signal Bases
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
  python 2_compute_signal_bases.py                     # All data (2009-09-25 to today)
  python 2_compute_signal_bases.py 2020-01-01 2024-12-31  # Custom date range
"""

import sys
import time
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from support.etf_database import ETFDatabase
from support.signal_database import SignalDatabase
from library.signal_bases import compute_signal_bases_generator, count_total_signals

# Output directories
PIPELINE_DIR = Path(__file__).parent
SIGNAL_OUTPUT_DIR = PIPELINE_DIR / 'data' / 'signals'
BACKUP_DIR = PIPELINE_DIR / 'backup' / '2_compute_signal_bases'
DATA_DIR = PIPELINE_DIR / 'data'

# Configuration
N_CORES = max(1, cpu_count() - 1)
CORRELATION_THRESHOLD = 0.05


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


def compute_signal_correlation_and_rankings(signal_name, signal_data, alpha_df, dates, isins,
                                            date_to_idx, isin_to_idx, rankings, kept_signal_names):
    """
    Compute correlation with forward IR and rankings for a single signal base.

    Args:
        signal_name: Name of signal
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

    kept_signal_names.append(signal_name)
    return True


def compute_and_save_signal_bases(
    start_date: str = None,
    end_date: str = None,
    isins: list = None
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
    print("STEP 2: COMPUTE SIGNAL BASES")
    print("=" * 120)

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
    print("\nLoading price data...")

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
    print("\nComputing and saving signal bases (parquet format)...")

    isin_list = list(etf_prices.columns)
    signal_start = time.time()
    n_records = 0
    n_signals = 0

    # Get total signal count upfront for accurate progress bar
    total_signals = count_total_signals()
    print(f"  Total signals to compute: {total_signals}")

    # Initialize ranking matrix components BEFORE main loop (for inline computation)
    print("\n  Preparing ranking matrix components...")
    alpha_file = DATA_DIR / 'forward_alpha_1month.parquet'
    rankings = None
    kept_signal_names = []
    date_to_idx = None
    isin_to_idx = None
    target_dates = None
    alpha_df = None

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
        rankings = np.full((len(target_dates), len(isin_list_unique), total_signals), np.nan, dtype=np.float32)
        print(f"  Ranking matrix initialized: {rankings.shape} (will trim after filtering)")
    else:
        print(f"  WARNING: Forward IR not found at {alpha_file}")
        print("  Skipping ranking matrix creation. Run Step 1 first if needed.")

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

        # Save to parquet (all dates) - save ALL signal bases
        records = signal_db.save_signal_base_from_array(
            signal_name, signal_2d, save_dates, isin_list
        )
        n_records += records

        # INLINE: Check correlation with forward IR and build rankings (while signal in memory)
        if alpha_df is not None and rankings is not None:
            compute_signal_correlation_and_rankings(
                signal_name, signal_2d, alpha_df, save_dates, isin_list,
                date_to_idx, isin_to_idx, rankings, kept_signal_names
            )

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
    print("SIGNAL BASES COMPUTATION COMPLETE")
    print("=" * 120)
    print(f"  Date range: {save_dates[0].date()} to {save_dates[-1].date()}")
    print(f"  Days: {len(save_dates)} | ETFs: {len(etf_prices.columns)} | Signals: {n_signals}")
    print(f"  Records: {n_records:,} | Time: {signal_time:.1f}s ({signal_time/60:.1f} min)")

    stats = signal_db.get_stats()
    print(f"  Storage: {stats['total_size_mb']:.1f} MB | Files: {stats['unique_signal_bases']}")
    print("=" * 120)

    # Save ranking matrix (computed inline during signal generation)
    if rankings is not None and len(kept_signal_names) > 0:
        print("\nSaving ranking matrix...")

        # Trim rankings to only kept signals
        rankings = rankings[:, :, :len(kept_signal_names)]

        # Save ranking matrix
        matrix_file = DATA_DIR / 'rankings_matrix_signal_bases_1month.npz'
        np.savez_compressed(
            matrix_file,
            rankings=rankings,
            dates=np.array(target_dates),
            isins=np.array(sorted(alpha_df['isin'].unique())),
            features=np.array(kept_signal_names),
            n_filtered=len(kept_signal_names)
        )

        print(f"  Ranking matrix: {len(kept_signal_names)} signals passed correlation filter (from {n_signals})")
        print(f"  Shape: {rankings.shape}")
        print(f"  [SAVED] {matrix_file}")
    elif rankings is None:
        print("\nSkipping ranking matrix (forward IR not available)")
    elif len(kept_signal_names) == 0:
        print("\nNo signals passed correlation filter for ranking matrix")

    print("=" * 120)

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
    # Parse arguments
    date_args = [a for a in sys.argv[1:] if not a.startswith('-')]

    if len(date_args) >= 2:
        start_date = date_args[0]
        end_date = date_args[1]
        print(f"Custom date range: {start_date} to {end_date}\n")
        stats = compute_and_save_signal_bases(start_date=start_date, end_date=end_date)
    else:
        print("Computing all available data (from 2009-09-25 to today)\n")
        stats = compute_and_save_signal_bases()

    if stats:
        print("\nSignal bases computed successfully!")
