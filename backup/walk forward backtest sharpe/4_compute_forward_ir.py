"""
Phase 1: Compute Forward Alpha and Information Ratio
=====================================================

INFORMATION RATIO OPTIMIZATION PROJECT - Step 4

Computes walk-forward data for IR-based backtesting:

1. **Forward Alpha**: ETF return - Benchmark return for each month
2. **Forward IR**: forward_alpha / realized_volatility_during_that_month
   - Realized volatility = std(daily alphas) within the month
   - True IR metric that consolidates alpha, consistency, and risk
3. **Rankings Matrix**: Z-score normalized feature signals
   - Cross-sectional z-score ranking (volatility-adjusted)
   - 7,618 features total (filtered + raw)
   - Used by Phase 2-4 to evaluate feature quality

Output:
    data/forward_alpha_1month.parquet         # Contains forward_alpha, forward_ir, forward_return, core_return
    data/rankings_matrix_ir_1month.npz        # 3D matrix: (dates, isins, features)

Usage:
    python 4_compute_forward_ir.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from support.etf_database import ETFDatabase
from support.signal_database import SignalDatabase


# ============================================================
# CONFIGURATION
# ============================================================

# Core ETF (benchmark)
CORE_ISIN = 'IE00B4L5Y983'  # iShares Core MSCI World

# Holding period (in months) - 1-month horizon is optimal
HOLDING_MONTHS = 1

# Minimum price history required (1 year)
MIN_HISTORY_DAYS = 252

# Number of CPU cores for parallel processing
N_CORES = max(1, cpu_count() - 1)

# Number of threads for I/O operations
N_THREADS = min(32, cpu_count() * 4)

# Force recompute (set to True to regenerate files)
FORCE_RECOMPUTE = False

# Output directory (relative to this script's location)
OUTPUT_DIR = Path(__file__).parent / 'data'

# Signal directory (where scripts 1-3 store signals)
SIGNAL_DIR = Path(__file__).parent / 'data' / 'signals'


# ============================================================
# PHASE 1A: COMPUTE FORWARD ALPHA AND INFORMATION RATIO
# ============================================================

def get_monthly_dates(start_date=None, end_date=None):
    """
    Generate end-of-month dates using pandas.

    If start_date/end_date not provided, uses reasonable defaults.
    Dates can be datetime objects, strings, or year integers.
    """
    # Handle defaults
    if start_date is None:
        start_date = '2015-01-01'
    elif isinstance(start_date, int):
        start_date = f'{start_date}-01-01'

    if end_date is None:
        # Default to current date
        end_date = pd.Timestamp.now()
    elif isinstance(end_date, int):
        end_date = f'{end_date}-12-31'

    return pd.date_range(
        start=start_date,
        end=end_date,
        freq='ME'  # Month End
    ).tolist()


def load_single_etf(args):
    """Load a single ETF's price data (for parallel loading)."""
    isin, db_path = args
    try:
        db = ETFDatabase(db_path)
        data = db.load_prices(isin)
        if data is None or len(data) == 0:
            return isin, None

        if isinstance(data, pd.Series):
            df = data.to_frame(name='price')
        else:
            df = data

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        if 'price' in df.columns:
            return isin, df['price']
        elif 'close' in df.columns:
            return isin, df['close']
        return isin, None
    except Exception:
        return isin, None


def parse_horizon(horizon):
    """Parse horizon specification into (value, unit) tuple."""
    if isinstance(horizon, int):
        return (horizon, 'month')
    elif isinstance(horizon, str) and horizon.endswith('d'):
        return (int(horizon[:-1]), 'days')
    else:
        raise ValueError(f"Invalid horizon format: {horizon}. Use int for months only.")


def get_horizon_label(horizon):
    """Get a label string for a horizon (for filenames and display)."""
    value, unit = parse_horizon(horizon)
    if unit == 'month':
        return f"{value}month"
    else:
        return f"{value}days"


def compute_forward_alpha(horizon):
    """
    Compute forward alpha and forward IR (Information Ratio) for all ETFs.

    Returns both:
    - forward_alpha: ETF return - Benchmark return
    - forward_ir: forward_alpha / realized_volatility_during_that_month

    Uses vectorized operations for efficiency.
    """
    horizon_value, horizon_unit = parse_horizon(horizon)
    horizon_label = get_horizon_label(horizon)

    print(f"\n{'='*60}")
    print(f"PHASE 1A: COMPUTE FORWARD ALPHA & IR ({horizon_label.upper()} HORIZON)")
    print(f"{'='*60}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    alpha_file = OUTPUT_DIR / f'forward_alpha_{horizon_label}.parquet'

    if alpha_file.exists() and not FORCE_RECOMPUTE:
        print(f"\n[LOADING] Pre-computed forward alpha from {alpha_file}")
        alpha_df = pd.read_parquet(alpha_file)
        alpha_df['date'] = pd.to_datetime(alpha_df['date'])
        print(f"  Observations: {len(alpha_df):,}")
        print(f"  Dates: {alpha_df['date'].nunique()}")
        print(f"  ISINs: {alpha_df['isin'].nunique()}")
        print(f"  Mean alpha: {alpha_df['forward_alpha'].mean():.4f}")
        return alpha_df

    print(f"\n[COMPUTING] Forward alpha (optimized)...")

    # Load ETF prices in parallel FIRST (to determine date range)
    # Database is in maintenance/data folder
    db_path = str(project_root / "maintenance" / "data" / "etf_database.db")
    db = ETFDatabase(db_path)
    universe_df = db.load_universe()
    etf_list = universe_df['isin'].tolist()

    print(f"\nLoading price data for {len(etf_list)} ETFs with {N_THREADS} threads...")

    prices = {}
    args_list = [(isin, db_path) for isin in etf_list]

    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = {executor.submit(load_single_etf, args): args[0] for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading prices"):
            isin, price_series = future.result()
            if price_series is not None:
                prices[isin] = price_series

    print(f"Successfully loaded {len(prices)} ETFs")

    if CORE_ISIN not in prices:
        raise ValueError(f"Core ISIN {CORE_ISIN} not found in price data")

    # Build combined price DataFrame for vectorized computation
    print(f"\nBuilding combined price matrix...")
    price_df = pd.DataFrame(prices)
    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index()

    # Generate monthly dates ADAPTIVELY based on actual data range
    data_start = price_df.index.min()
    data_end = price_df.index.max()
    print(f"\nData range: {data_start.date()} to {data_end.date()}")

    # Start from 2015 or data start (whichever is later), end at data end
    start_date = max(pd.Timestamp('2015-01-01'), data_start)
    monthly_dates = get_monthly_dates(start_date=start_date, end_date=data_end)
    monthly_dates_arr = np.array(monthly_dates, dtype='datetime64[ns]')
    print(f"Generated {len(monthly_dates)} monthly dates ({monthly_dates[0].date()} to {monthly_dates[-1].date()})")

    # Get trading dates
    trading_dates = price_df.index.values

    print(f"  Price matrix: {price_df.shape[0]} days × {price_df.shape[1]} ETFs")
    print(f"  Date range: {price_df.index[0].date()} to {price_df.index[-1].date()}")

    # Map monthly test dates to actual trading dates
    print(f"\nMapping monthly dates to trading dates...")
    start_indices = np.searchsorted(trading_dates, monthly_dates_arr[:-horizon_value])
    end_indices = np.searchsorted(trading_dates, monthly_dates_arr[horizon_value:])

    # Filter valid indices (within bounds)
    valid_mask = (start_indices < len(trading_dates)) & (end_indices < len(trading_dates))
    valid_start_idx = start_indices[valid_mask]
    valid_end_idx = end_indices[valid_mask]

    n_periods = len(valid_start_idx)
    print(f"  Valid periods: {n_periods}")

    # Get prices at start and end dates (vectorized)
    print(f"\nComputing returns (vectorized)...")

    start_dates = trading_dates[valid_start_idx]
    end_dates = trading_dates[valid_end_idx]

    # Extract price matrices at start and end dates
    start_prices = price_df.iloc[valid_start_idx].values  # (n_periods, n_etfs)
    end_prices = price_df.iloc[valid_end_idx].values      # (n_periods, n_etfs)

    # Compute returns: (end - start) / start
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = (end_prices / start_prices) - 1  # (n_periods, n_etfs)

    n_etfs = returns.shape[1]

    # Get core returns
    core_col_idx = price_df.columns.get_loc(CORE_ISIN)
    core_returns = returns[:, core_col_idx]  # (n_periods,)

    # Compute alpha: ETF return - core return
    alpha = returns - core_returns[:, np.newaxis]  # Broadcasting

    # Compute realized (forward) volatility per ISIN for each month
    # This is the actual volatility realized DURING that month
    # True forward IR = forward_alpha / realized_volatility_during_that_month
    print(f"\nComputing realized volatility per ISIN for each month...")

    forward_volatility = np.full((n_periods, n_etfs), np.nan, dtype=np.float32)

    for period in range(n_periods):
        # Get date range for this period
        period_start_date = start_dates[period]
        period_end_date = end_dates[period]

        # Find all daily dates within this month's period
        mask = (price_df.index >= period_start_date) & (price_df.index <= period_end_date)
        daily_dates_in_period = price_df.index[mask]

        if len(daily_dates_in_period) >= 5:  # Need at least 5 trading days in the month
            # Get start and end indices for this period in the full daily price data
            start_idx = price_df.index.get_loc(daily_dates_in_period[0])
            end_idx = price_df.index.get_loc(daily_dates_in_period[-1])

            # Compute daily returns within this period
            daily_period_prices = price_df.iloc[start_idx:end_idx + 1].values  # (n_days, n_etfs)
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_period_returns = (daily_period_prices[1:] / daily_period_prices[:-1]) - 1

            # Get benchmark returns for this period
            benchmark_col = price_df.columns.get_loc(CORE_ISIN)
            daily_benchmark_returns = daily_period_returns[:, benchmark_col]

            # Compute daily alphas within this period
            daily_period_alphas = daily_period_returns - daily_benchmark_returns[:, np.newaxis]

            # Realized volatility = std of daily alphas during the month
            forward_volatility[period, :] = np.std(daily_period_alphas, axis=0)

    # Floor to avoid division by zero
    forward_volatility = np.maximum(forward_volatility, 1e-8)
    forward_volatility = np.nan_to_num(forward_volatility, nan=1e-8)

    print(f"  Mean realized volatility: {np.nanmean(forward_volatility):.6f}")
    print(f"  Min: {np.nanmin(forward_volatility):.6f}, Max: {np.nanmax(forward_volatility):.6f}")

    # Compute true forward IR: forward_alpha / realized_volatility_during_that_month
    # This is the actual Information Ratio achieved in that month
    with np.errstate(divide='ignore', invalid='ignore'):
        forward_ir = alpha / forward_volatility  # Element-wise division (n_periods, n_etfs)

    # Check history requirement (vectorized)
    # For each ETF, find first valid date
    print(f"\nChecking history requirements...")
    first_valid_idx = np.argmax(~np.isnan(price_df.values), axis=0)  # First non-NaN per ETF
    first_valid_dates = trading_dates[first_valid_idx]

    # Compute required history start for each period
    history_threshold = pd.Timedelta(days=int(MIN_HISTORY_DAYS * 1.5))

    # Create mask for sufficient history
    # ETF has sufficient history if first_valid_date <= start_date - history_threshold
    start_dates_pd = pd.to_datetime(start_dates)
    required_start = start_dates_pd - history_threshold

    # Build results using vectorized operations
    print(f"\nBuilding results DataFrame...")

    isins = price_df.columns.tolist()
    n_etfs = len(isins)

    # Flatten arrays
    flat_returns = returns.flatten()
    flat_alpha = alpha.flatten()
    flat_ir = forward_ir.flatten()
    flat_core_returns = np.repeat(core_returns, n_etfs)

    # Create date and isin arrays
    flat_dates = np.repeat(start_dates, n_etfs)
    flat_isins = np.tile(isins, n_periods)

    # Create validity mask
    # 1. Not NaN
    valid = ~np.isnan(flat_alpha)
    # 2. Not core ETF
    valid &= (flat_isins != CORE_ISIN)
    # 3. Sufficient history
    first_valid_tiled = np.tile(first_valid_dates, n_periods)
    required_start_tiled = np.repeat(required_start.values, n_etfs)
    valid &= (first_valid_tiled <= required_start_tiled)

    # Build DataFrame from valid entries only
    alpha_df = pd.DataFrame({
        'date': pd.to_datetime(flat_dates[valid]),
        'isin': np.array(flat_isins)[valid],
        'forward_return': flat_returns[valid],
        'core_return': flat_core_returns[valid],
        'forward_alpha': flat_alpha[valid],
        'forward_ir': flat_ir[valid]
    })

    # Save
    alpha_df.to_parquet(alpha_file, index=False)

    print(f"\n{horizon_label} forward alpha and IR:")
    print(f"  Observations: {len(alpha_df):,}")
    print(f"  Dates: {alpha_df['date'].nunique()}")
    print(f"  ISINs: {alpha_df['isin'].nunique()}")
    print(f"  Mean alpha: {alpha_df['forward_alpha'].mean():.4f}")
    print(f"  Mean IR: {alpha_df['forward_ir'].mean():.4f}")
    print(f"  IR range: {alpha_df['forward_ir'].min():.4f} to {alpha_df['forward_ir'].max():.4f}")
    print(f"\n[SAVED] {alpha_file}")

    return alpha_df


# ============================================================
# PHASE 1B: CREATE RANKINGS MATRIX (Z-SCORE NORMALIZED)
# ============================================================

def process_single_feature(args):
    """
    Process a single feature and return rankings.

    OPTIMIZED: Uses vectorized operations instead of iterrows().
    Uses searchsorted for fast date matching.
    """
    feature_name, is_filtered, target_dates_arr, date_to_idx, isin_to_idx = args

    db = SignalDatabase(SIGNAL_DIR)

    try:
        if is_filtered:
            data = db.load_filtered_signal_by_name(feature_name)
        else:
            data = db.load_signal_base(feature_name)

        if data is None:
            return None

        # Convert to DataFrame with date index and ISIN columns
        if isinstance(data, pd.Series):
            df = data.reset_index()
            df.columns = ['date', 'isin', 'value']
        elif isinstance(data, pd.DataFrame):
            df = data.stack().reset_index()
            df.columns = ['date', 'isin', 'value']
        else:
            return None

        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['value'])

        if len(df) == 0:
            return None

        # Get available signal dates as sorted array
        signal_dates = np.sort(df['date'].unique())

        # Use searchsorted to find closest signal date on or before each target date
        # searchsorted with side='right' gives us the index where target would be inserted
        # So index-1 gives us the largest signal_date <= target_date
        insert_positions = np.searchsorted(signal_dates, target_dates_arr, side='right')

        # Build date mapping: target_date -> signal_date (only where valid)
        valid_positions = insert_positions > 0  # Must have at least one signal date before

        # Prepare output list using numpy operations
        rankings = []

        # Group by signal date for efficient processing
        # Map each target date to its corresponding signal date
        target_to_signal = {}
        for i, (target_date, pos, is_valid) in enumerate(zip(target_dates_arr, insert_positions, valid_positions)):
            if is_valid and target_date in date_to_idx:
                signal_date = signal_dates[pos - 1]
                if signal_date not in target_to_signal:
                    target_to_signal[signal_date] = []
                target_to_signal[signal_date].append((target_date, date_to_idx[target_date]))

        # Process each unique signal date once
        for signal_date, target_info_list in target_to_signal.items():
            # Get data for this signal date
            mask = df['date'] == signal_date
            group = df.loc[mask, ['isin', 'value']].copy()

            if len(group) == 0:
                continue

            # PHASE 1 CHANGE: Add z-score normalization before percentile ranking
            # Z-score accounts for signal volatility, not just magnitude
            if len(group) > 1:
                group['zscore'] = (group['value'] - group['value'].mean()) / group['value'].std()
                # Handle case where std is 0 (all values identical)
                if np.isnan(group['zscore']).any() or np.isinf(group['zscore']).any():
                    # Fall back to raw values if z-score fails
                    group['rank'] = group['value'].rank(pct=True)
                else:
                    # Rank by z-score (volatility-adjusted)
                    group['rank'] = group['zscore'].rank(pct=True)
            else:
                # Single value - just rank by value
                group['rank'] = group['value'].rank(pct=True)

            # Filter to ISINs we care about and map to indices (vectorized)
            isin_mask = group['isin'].isin(isin_to_idx)
            valid_group = group[isin_mask]

            if len(valid_group) == 0:
                continue

            # Map ISINs to indices
            isin_indices = valid_group['isin'].map(isin_to_idx).values
            rank_values = valid_group['rank'].values

            # Add rankings for all target dates that map to this signal date
            for target_date, date_idx in target_info_list:
                for isin_idx, rank in zip(isin_indices, rank_values):
                    rankings.append((date_idx, isin_idx, rank))

        return rankings
    except Exception:
        return None


def load_feature_metrics(holding_months):
    """Load pre-computed feature metrics."""
    # Try walk_forward directory first, then feature_analysis
    metrics_file = OUTPUT_DIR / f'feature_metrics_{holding_months}month.csv'
    if not metrics_file.exists():
        metrics_file = Path('data/feature_analysis') / f'feature_metrics_{holding_months}month.csv'

    if metrics_file.exists():
        return pd.read_csv(metrics_file)

    # If no metrics file, compute from scratch
    print("\nWARNING: No feature metrics file found. Running feature evaluation...")
    return None


def create_rankings_matrix(alpha_df, horizon):
    """
    Create a 3D matrix of rankings for all features.

    OPTIMIZED: Uses vectorized matrix filling instead of Python loops.
    """
    horizon_label = get_horizon_label(horizon)

    print(f"\n{'='*60}")
    print(f"PHASE 1B: CREATE RANKINGS MATRIX")
    print(f"{'='*60}")

    matrix_file = OUTPUT_DIR / f'rankings_matrix_ir_{horizon_label}.npz'

    if matrix_file.exists() and not FORCE_RECOMPUTE:
        print(f"\n[LOADING] Pre-computed rankings matrix from {matrix_file}")
        data = np.load(matrix_file, allow_pickle=True)
        rankings_data = {
            'rankings': data['rankings'],
            'dates': pd.to_datetime(data['dates']),
            'isins': data['isins'],
            'features': data['features'].tolist() if isinstance(data['features'], np.ndarray) else data['features'],
            'n_filtered': int(data['n_filtered'])
        }
        print(f"  Matrix shape: {rankings_data['rankings'].shape}")
        return rankings_data

    print(f"\n[COMPUTING] Rankings matrix (optimized)...")

    # Load feature metrics (use 1-month metrics as base for feature selection)
    feature_metrics = load_feature_metrics(1)  # Always use 1-month for feature selection

    if feature_metrics is None:
        # Load all signals from database
        db = SignalDatabase(SIGNAL_DIR)
        filtered_signals = sorted(db.get_completed_filtered_signals())
        raw_signals = db.get_available_signals()

        filtered_features = pd.DataFrame({
            'feature_name': filtered_signals,
            'feature_type': 'filtered'
        })
        raw_signals_df = pd.DataFrame({
            'feature_name': raw_signals,
            'feature_type': 'raw'
        })
    else:
        # Use all features from pre-computed metrics
        df_filtered = feature_metrics[feature_metrics['feature_type'] == 'filtered'].copy()
        df_raw = feature_metrics[feature_metrics['feature_type'] == 'raw'].copy()
        filtered_features = df_filtered
        raw_signals_df = df_raw

    print(f"\nSelected {len(filtered_features)} filtered features")
    print(f"Selected {len(raw_signals_df)} raw signals")

    # Get dimensions - target dates are the monthly test dates
    target_dates = sorted(alpha_df['date'].unique())
    target_dates_arr = np.array(target_dates, dtype='datetime64[ns]')
    dates = target_dates
    isins = sorted(alpha_df['isin'].unique())

    n_dates = len(dates)
    n_isins = len(isins)
    n_filtered = len(filtered_features)
    n_raw = len(raw_signals_df)
    n_features = n_filtered + n_raw

    print(f"\nTarget dates (monthly):")
    print(f"  First: {dates[0].date()}")
    print(f"  Last: {dates[-1].date()}")
    print(f"  Count: {n_dates}")
    print(f"  (Signal data will be matched to closest available date on or before)")

    print(f"\nMatrix dimensions:")
    print(f"  Dates: {n_dates}")
    print(f"  ISINs: {n_isins}")
    print(f"  Features: {n_features} ({n_filtered} filtered + {n_raw} raw)")

    # Initialize matrix
    rankings = np.full((n_dates, n_isins, n_features), np.nan, dtype=np.float32)

    # Create lookup dicts
    date_to_idx = {date: idx for idx, date in enumerate(dates)}
    isin_to_idx = {isin: idx for idx, isin in enumerate(isins)}

    # Prepare arguments for parallel processing
    # Use list comprehension instead of iterrows for speed
    filtered_names = filtered_features['feature_name'].tolist()
    raw_names = raw_signals_df['feature_name'].tolist()

    filtered_args = [
        (name, True, target_dates_arr, date_to_idx, isin_to_idx)
        for name in filtered_names
    ]

    raw_args = [
        (name, False, target_dates_arr, date_to_idx, isin_to_idx)
        for name in raw_names
    ]

    # Process filtered features
    print(f"\nLoading {len(filtered_args)} filtered features with {N_CORES} cores...")
    with Pool(N_CORES) as pool:
        filtered_results = list(tqdm(
            pool.imap(process_single_feature, filtered_args),
            total=len(filtered_args),
            desc="Filtered"
        ))

    # Process raw features
    print(f"\nLoading {len(raw_args)} raw features with {N_CORES} cores...")
    with Pool(N_CORES) as pool:
        raw_results = list(tqdm(
            pool.imap(process_single_feature, raw_args),
            total=len(raw_args),
            desc="Raw"
        ))

    # Fill matrix using vectorized operations where possible
    print(f"\nFilling rankings matrix...")

    for feat_idx, result in enumerate(tqdm(filtered_results, desc="Filtered matrix")):
        if result is not None and len(result) > 0:
            # Convert to numpy arrays for vectorized indexing
            result_arr = np.array(result, dtype=np.float32)
            date_indices = result_arr[:, 0].astype(np.int32)
            isin_indices = result_arr[:, 1].astype(np.int32)
            rank_values = result_arr[:, 2]
            rankings[date_indices, isin_indices, feat_idx] = rank_values

    for feat_idx, result in enumerate(tqdm(raw_results, desc="Raw matrix")):
        if result is not None and len(result) > 0:
            result_arr = np.array(result, dtype=np.float32)
            date_indices = result_arr[:, 0].astype(np.int32)
            isin_indices = result_arr[:, 1].astype(np.int32)
            rank_values = result_arr[:, 2]
            rankings[date_indices, isin_indices, n_filtered + feat_idx] = rank_values

    # Create feature names list
    feature_names = filtered_names + raw_names

    print(f"\nMatrix statistics:")
    print(f"  Non-NaN values: {(~np.isnan(rankings)).sum():,}")
    print(f"  Coverage: {(~np.isnan(rankings)).sum() / rankings.size * 100:.1f}%")

    # Save
    np.savez_compressed(
        matrix_file,
        rankings=rankings,
        dates=dates,
        isins=np.array(isins),
        features=feature_names,
        n_filtered=n_filtered
    )
    print(f"\n[SAVED] {matrix_file}")

    return {
        'rankings': rankings,
        'dates': pd.to_datetime(dates),
        'isins': np.array(isins),
        'features': feature_names,
        'n_filtered': n_filtered
    }


# ============================================================
# MAIN
# ============================================================

def main(horizon=HOLDING_MONTHS):
    """Run Phase 1: Compute forward alpha, IR, and rankings matrix."""
    horizon_label = get_horizon_label(horizon)

    print("=" * 60)
    print(f"PHASE 1: WALK-FORWARD DATA PREPARATION ({horizon_label.upper()})")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1A: Compute forward alpha and IR
    alpha_df = compute_forward_alpha(horizon)

    # Phase 1B: Create rankings matrix
    rankings_data = create_rankings_matrix(alpha_df, horizon)

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print(f"  • forward_alpha_{horizon_label}.parquet")
    print(f"    Columns: date, isin, forward_return, core_return, forward_alpha, forward_ir")
    print(f"  • rankings_matrix_ir_{horizon_label}.npz")
    print(f"    3D matrix: ({len(rankings_data['dates'])} dates × {len(rankings_data['isins'])} ISINs × {len(rankings_data['features'])} features)")
    print(f"\nNext step: Run Phase 2 - 5_precompute_feature_ir.py")

    return alpha_df, rankings_data


if __name__ == '__main__':
    main(HOLDING_MONTHS)
