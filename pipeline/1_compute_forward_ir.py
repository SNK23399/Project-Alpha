"""
STEP 1: Compute Forward Alpha and Information Ratio
====================================================

FIRST STEP IN PIPELINE - Runs independently from signal computation

Computes forward IR directly from ETF prices:

1. **Forward Alpha**: ETF return - Benchmark return for each month
2. **Forward IR**: forward_alpha / realized_volatility_during_that_month
   - Realized volatility = std(daily alphas) within the month
   - True IR metric that consolidates alpha, consistency, and risk

This is the target variable used by subsequent steps:
- Steps 2-3: Generate signals (signal bases, filters)
- Step 4: Precomputes feature IR (for each feature, what IR would be achieved)
- Step 5: Computes MC statistics (Bayesian priors from historical IR distribution)
- Step 6: Bayesian satellite selection using the learned patterns

Output:
    data/forward_alpha_1month.parquet
        Columns: date, isin, forward_return, core_return, forward_alpha, forward_ir

Usage:
    python 1_compute_forward_ir.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from support.etf_database import ETFDatabase


# ============================================================
# CONFIGURATION
# ============================================================

# Core ETF (benchmark)
CORE_ISIN = 'IE00B4L5Y983'  # iShares Core MSCI World

# Holding period (in months) - 1-month horizon is optimal
HOLDING_MONTHS = 1

# Minimum price history required (1 year)
MIN_HISTORY_DAYS = 252

# Minimum AUM (Fund Size) to reduce survivor bias
# ETFs below this threshold are more likely to be delisted
# This filters to only ETFs with stable funding and realistic trading universe
MIN_AUM_MILLIONS = 75  # EUR millions

# Number of threads for I/O operations
N_THREADS = min(32, 4 * 4)

# Force recompute (set to True to regenerate files)
FORCE_RECOMPUTE = True

# Output directory (relative to this script's location)
OUTPUT_DIR = Path(__file__).parent / 'data'


# ============================================================
# HELPER FUNCTIONS
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


# ============================================================
# MAIN COMPUTATION
# ============================================================

def compute_forward_ir(horizon=HOLDING_MONTHS):
    """
    Compute forward alpha and forward IR (Information Ratio) for all ETFs.

    Returns:
    - forward_alpha: ETF return - Benchmark return
    - forward_ir: forward_alpha / realized_volatility_during_that_month

    Uses vectorized operations for efficiency.
    """
    horizon_value, horizon_unit = parse_horizon(horizon)
    horizon_label = get_horizon_label(horizon)

    print(f"\n{'='*120}")
    print(f"STEP 1: COMPUTE FORWARD IR ({horizon_label.upper()} HORIZON)")
    print(f"{'='*120}")
    print(f"\nTarget variable for feature correlation filtering and backtesting")

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
        print(f"  Mean IR: {alpha_df['forward_ir'].mean():.4f}")
        return alpha_df

    print(f"\n[COMPUTING] Forward alpha and IR (optimized)...")

    # Load ETF prices in parallel FIRST (to determine date range)
    # Database is in maintenance/data folder
    db_path = str(project_root / "maintenance" / "data" / "etf_database.db")
    db = ETFDatabase(db_path)
    universe_df = db.load_universe()

    # Filter by minimum AUM to reduce survivor bias
    # Smaller ETFs are more likely to be delisted
    initial_count = len(universe_df)

    # Keep original for core ISIN verification
    universe_original = universe_df.copy()

    universe_df = universe_df[
        (universe_df['fund_size'].notna()) &
        (universe_df['fund_size'] >= MIN_AUM_MILLIONS)
    ].copy()
    filtered_count = initial_count - len(universe_df)

    print(f"\nAUM Filter: {initial_count} total ETFs → {len(universe_df)} with AUM >= {MIN_AUM_MILLIONS}M EUR")
    if filtered_count > 0:
        print(f"  Excluded {filtered_count} ETFs with lower AUM (reduces survivor bias)")

    etf_list = universe_df['isin'].tolist()

    # Verify core ETF passed AUM filter
    if CORE_ISIN not in etf_list:
        core_data = universe_original[universe_original['isin'] == CORE_ISIN]
        if len(core_data) == 0:
            raise ValueError(f"Core ISIN {CORE_ISIN} not found in database")
        core_aum = core_data['fund_size'].values[0]
        raise ValueError(
            f"Core ISIN {CORE_ISIN} has AUM {core_aum:.0f}M EUR, "
            f"below minimum threshold of {MIN_AUM_MILLIONS}M EUR. "
            f"Increase MIN_AUM_MILLIONS or use different core ETF."
        )

    print(f"\nLoading price data for {len(etf_list)} ETFs with {N_THREADS} threads...")

    prices = {}
    args_list = [(isin, db_path) for isin in etf_list]

    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = {executor.submit(load_single_etf, args): args[0] for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading prices", ncols=120):
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
    print(f"  Data range: {data_start.date()} to {data_end.date()}")

    # Start from 2015 or data start (whichever is later), end at data end
    start_date = max(pd.Timestamp('2015-01-01'), data_start)
    monthly_dates = get_monthly_dates(start_date=start_date, end_date=data_end)
    monthly_dates_arr = np.array(monthly_dates, dtype='datetime64[ns]')
    print(f"  Generated {len(monthly_dates)} monthly dates ({monthly_dates[0].date()} to {monthly_dates[-1].date()})")

    # Get trading dates
    trading_dates = price_df.index.values

    print(f"\n  Price matrix: {price_df.shape[0]} days × {price_df.shape[1]} ETFs")
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

    print(f"\n{horizon_label} forward IR statistics:")
    print(f"  Observations: {len(alpha_df):,}")
    print(f"  Dates: {alpha_df['date'].nunique()}")
    print(f"  ISINs: {alpha_df['isin'].nunique()}")
    print(f"  Mean alpha: {alpha_df['forward_alpha'].mean():.4f}")
    print(f"  Mean IR: {alpha_df['forward_ir'].mean():.4f}")
    print(f"  IR range: {alpha_df['forward_ir'].min():.4f} to {alpha_df['forward_ir'].max():.4f}")
    print(f"\n[SAVED] {alpha_file}")

    return alpha_df


# ============================================================
# MAIN
# ============================================================

def main():
    """Run Step 1: Compute forward IR."""
    print("=" * 120)
    print("STEP 1: FORWARD IR COMPUTATION")
    print("=" * 120)
    print("\nThis is the FIRST STEP in the pipeline.")
    print("It computes forward IR directly from ETF prices.")
    print("Subsequent steps will use this as the target variable.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Compute forward IR
    alpha_df = compute_forward_ir(HOLDING_MONTHS)

    print("\n" + "=" * 120)
    print("STEP 1 COMPLETE")
    print("=" * 120)


if __name__ == '__main__':
    main()
