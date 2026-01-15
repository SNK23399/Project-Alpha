"""
Step 4: Feature Analysis Pipeline

This script performs feature selection using greedy forward selection to find
the optimal ensemble of features for predicting ETF outperformance.

For a given holding period (1, 2, or 3 months), it:
1. Computes forward alpha for all ETFs at monthly intervals
2. Evaluates all features for their predictive power
3. Creates a rankings matrix with all feature values
4. Uses greedy forward selection to build an optimal feature ensemble
5. Saves the ensemble for use in backtesting

The greedy selection iteratively adds features that maximize the improvement
in average alpha and hit rate of the portfolio.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from support.etf_database import ETFDatabase
from support.signal_database import SignalDatabase
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================

# Core ETF (benchmark)
CORE_ISIN = 'IE00B4L5Y983'  # iShares Core MSCI World

# Holding period(s) (in months) - can be a single int or list of ints
# Examples: HOLDING_MONTHS = 1  OR  HOLDING_MONTHS = [1, 2, 3]
HOLDING_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Minimum price history required (1 year)
MIN_HISTORY_DAYS = 252

# Number of CPU cores for parallel processing
N_CORES = cpu_count()

# Portfolio configuration
N_SATELLITES = 4
MIN_ETFS_PER_PERIOD = 50

# Feature pre-filtering thresholds
TOP_N_FILTERED_FEATURES = 300  # Load top 300 filtered features by momentum alpha
TOP_N_RAW_SIGNALS = 100        # Load top 100 raw signal bases
MIN_ALPHA = 0.005              # Minimum 0.5% alpha per period
MIN_HIT_RATE = 0.45            # Minimum 45% hit rate
MIN_IC_IR = 0.2                # Minimum IC IR for consistency

# Ensemble search configuration
MAX_ENSEMBLE_SIZE = 20
MIN_IMPROVEMENT = 0.00001  # Very small threshold (0.001%) for greedy


# ============================================================
# STEP 1: COMPUTE FORWARD ALPHA (MONTHLY DATES)
# ============================================================

def get_monthly_dates(start_year=2015, end_year=2024):
    """Generate end-of-month dates for all months."""
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if month == 12:
                date = datetime(year, month, 31)
            elif month in [4, 6, 9, 11]:
                date = datetime(year, month, 30)
            elif month == 2:
                # Handle leap years
                if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                    date = datetime(year, month, 29)
                else:
                    date = datetime(year, month, 28)
            else:
                date = datetime(year, month, 31)
            dates.append(date)
    return dates


def compute_forward_alpha(holding_months):
    """
    Compute forward alpha for all ETFs at monthly intervals.

    Key change: ALWAYS uses monthly evaluation dates, regardless of holding period.
    The holding_months parameter only affects the forward return horizon.
    """
    print(f"\n{'='*60}")
    print(f"STEP 1: COMPUTE FORWARD ALPHA ({holding_months}-MONTH HORIZON)")
    print(f"{'='*60}")

    # Generate monthly dates
    monthly_dates = get_monthly_dates()
    print(f"\nGenerated {len(monthly_dates)} monthly dates")
    print(f"  First: {monthly_dates[0].date()}")
    print(f"  Last: {monthly_dates[-1].date()}")

    # Load ETF prices
    db = ETFDatabase()
    universe_df = db.load_universe()
    etf_list = universe_df['isin'].tolist()

    print(f"\nLoading price data for {len(etf_list)} ETFs...")
    prices = {}

    for isin in tqdm(etf_list, desc="Loading prices"):
        try:
            data = db.load_prices(isin)
            if data is None or len(data) == 0:
                continue

            if isinstance(data, pd.Series):
                df = data.to_frame(name='price')
            else:
                df = data

            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            if 'price' in df.columns:
                prices[isin] = df[['price']].copy()
            elif 'close' in df.columns:
                prices[isin] = df[['close']].rename(columns={'close': 'price'})
        except Exception:
            pass

    print(f"Successfully loaded {len(prices)} ETFs")

    # Get core benchmark
    if CORE_ISIN not in prices:
        raise ValueError(f"Core ISIN {CORE_ISIN} not found in price data")

    core_prices = prices[CORE_ISIN]

    # Compute forward alpha
    results = []

    print(f"\nComputing {holding_months}-month forward alpha at monthly intervals...")

    # Iterate through monthly dates, excluding last N months
    for i, current_date in enumerate(tqdm(monthly_dates[:-holding_months], desc="Processing dates")):
        # Find actual trading date
        try:
            current_actual = core_prices.index[core_prices.index >= current_date][0]
        except IndexError:
            continue

        # Find end date (holding_months ahead)
        end_date = monthly_dates[i + holding_months]

        try:
            end_actual = core_prices.index[core_prices.index >= end_date][0]
        except IndexError:
            continue

        # Get core returns
        try:
            core_price_start = core_prices.loc[current_actual, 'price']
            core_price_end = core_prices.loc[end_actual, 'price']
            core_return = (core_price_end / core_price_start) - 1
        except (KeyError, IndexError):
            continue

        # Compute for all ETFs
        for isin, etf_prices in prices.items():
            if isin == CORE_ISIN:
                continue

            try:
                # Check sufficient history
                history_start = current_actual - pd.Timedelta(days=MIN_HISTORY_DAYS * 1.5)
                history_available = etf_prices[etf_prices.index >= history_start]

                if len(history_available) < MIN_HISTORY_DAYS:
                    continue

                # Get ETF prices
                etf_price_start = etf_prices.loc[current_actual, 'price']
                etf_price_end = etf_prices.loc[end_actual, 'price']

                etf_return = (etf_price_end / etf_price_start) - 1
                forward_alpha = etf_return - core_return

                results.append({
                    'date': current_actual,
                    'isin': isin,
                    'forward_return': etf_return,
                    'core_return': core_return,
                    'forward_alpha': forward_alpha
                })
            except (KeyError, IndexError):
                pass

    # Convert to DataFrame
    alpha_df = pd.DataFrame(results)

    print(f"\n{holding_months}-month forward alpha:")
    print(f"  Observations: {len(alpha_df):,}")
    print(f"  Dates: {alpha_df['date'].nunique()}")
    print(f"  ISINs: {alpha_df['isin'].nunique()}")
    print(f"  Mean alpha: {alpha_df['forward_alpha'].mean():.4f}")

    return alpha_df


# ============================================================
# STEP 2: EVALUATE FEATURES (PARALLELIZED)
# ============================================================

def evaluate_single_feature(args):
    """Evaluate a single feature."""
    signal_name, is_filtered, alpha_df = args

    db = SignalDatabase()

    try:
        # Load signal
        if is_filtered:
            data = db.load_filtered_signal_by_name(signal_name)
        else:
            data = db.load_signal_base(signal_name)

        if data is None:
            return None

        # Convert to long format
        if isinstance(data, pd.Series):
            df = data.reset_index()
            df.columns = ['date', 'isin', 'value']
        elif isinstance(data, pd.DataFrame):
            df = data.stack().reset_index()
            df.columns = ['date', 'isin', 'value']
        else:
            return None

        df['date'] = pd.to_datetime(df['date'])

        # Merge with alpha
        merged = pd.merge(alpha_df, df, on=['date', 'isin'], how='inner')

        if len(merged) < 100:
            return None

        # Calculate metrics
        ic = merged[['value', 'forward_alpha']].corr().iloc[0, 1]

        # Rank correlation
        merged['value_rank'] = merged.groupby('date')['value'].rank(pct=True)
        merged['alpha_rank'] = merged.groupby('date')['forward_alpha'].rank(pct=True)
        ic_rank = merged[['value_rank', 'alpha_rank']].corr().iloc[0, 1]

        # Hit rate
        merged['signal_positive'] = merged['value_rank'] > 0.5
        merged['alpha_positive'] = merged['forward_alpha'] > 0
        hit_rate = (merged['signal_positive'] == merged['alpha_positive']).mean()

        # Momentum alpha (top quartile vs bottom quartile)
        merged['quartile'] = pd.qcut(merged['value'], q=4, labels=False, duplicates='drop')
        momentum_alpha = merged[merged['quartile'] == 3]['forward_alpha'].mean() - \
                       merged[merged['quartile'] == 0]['forward_alpha'].mean()

        return {
            'feature_name': signal_name,
            'feature_type': 'filtered' if is_filtered else 'raw',
            'ic': ic,
            'ic_rank': ic_rank,
            'hit_rate': hit_rate,
            'momentum_alpha': momentum_alpha,
            'n_obs': len(merged)
        }
    except Exception:
        return None


def evaluate_features(alpha_df, holding_months, force_recompute=False):
    """Evaluate all features for their predictive power (load from pre-computed file)."""
    print(f"\n{'='*60}")
    print(f"STEP 2: LOAD FEATURE METRICS")
    print(f"{'='*60}")

    # Try to load existing feature metrics CSV
    metrics_file = Path('data/feature_analysis') / f'feature_metrics_{holding_months}month.csv'

    if metrics_file.exists() and not force_recompute:
        print(f"\n[LOADING] Pre-computed feature metrics from {metrics_file}")
        df = pd.read_csv(metrics_file)
        print(f"Loaded {len(df)} features")
        return df

    # Check for cached results
    cache_file = Path('data/feature_analysis') / f'feature_metrics_{holding_months}month_cache.parquet'

    if cache_file.exists() and not force_recompute:
        print(f"\n[CACHE HIT] Loading cached feature metrics from {cache_file}")
        df = pd.read_parquet(cache_file)
        print(f"Loaded {len(df)} features from cache")
        return df

    print(f"\n[COMPUTING] Feature metrics (this will be saved)...")

    db = SignalDatabase()

    # Get all filtered signals
    filtered_signals = list(db.get_completed_filtered_signals())
    print(f"\nFound {len(filtered_signals)} filtered signals")

    # Get all raw signal bases
    raw_signals = db.get_available_signals()
    print(f"Found {len(raw_signals)} raw signal bases")

    # Prepare arguments
    filtered_args = [(name, True, alpha_df) for name in filtered_signals]
    raw_args = [(name, False, alpha_df) for name in raw_signals]

    all_features = []

    # Evaluate filtered signals in parallel
    print(f"\nEvaluating {len(filtered_signals)} filtered signals with {N_CORES} cores...")
    with Pool(N_CORES) as pool:
        filtered_results = list(tqdm(
            pool.imap(evaluate_single_feature, filtered_args),
            total=len(filtered_args),
            desc="Filtered"
        ))

    # Add non-None results
    all_features.extend([r for r in filtered_results if r is not None])

    # Evaluate raw signals in parallel
    print(f"\nEvaluating {len(raw_signals)} raw signals with {N_CORES} cores...")
    with Pool(N_CORES) as pool:
        raw_results = list(tqdm(
            pool.imap(evaluate_single_feature, raw_args),
            total=len(raw_args),
            desc="Raw"
        ))

    # Add non-None results
    all_features.extend([r for r in raw_results if r is not None])

    df = pd.DataFrame(all_features)
    print(f"\nEvaluated {len(df)} features total")

    # Cache the results
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file, index=False)
    print(f"\n[CACHED] Saved feature metrics to {cache_file}")

    return df


# ============================================================
# STEP 3: LOAD AND FILTER FEATURES
# ============================================================

def load_and_filter_features(feature_metrics, alpha_df):
    """Load top features and create a subset for ensemble search."""
    print(f"\n{'='*60}")
    print(f"STEP 3: LOAD AND FILTER TOP FEATURES")
    print(f"{'='*60}")

    # Split by feature type
    df_filtered = feature_metrics[feature_metrics['feature_type'] == 'filtered'].copy()
    df_raw = feature_metrics[feature_metrics['feature_type'] == 'raw'].copy()

    print(f"\nFiltered signals: {len(df_filtered)}")
    print(f"Raw signals: {len(df_raw)}")

    # Take top N by momentum alpha
    df_filtered_top = df_filtered.nlargest(TOP_N_FILTERED_FEATURES, 'momentum_alpha')
    df_raw_top = df_raw.nlargest(TOP_N_RAW_SIGNALS, 'momentum_alpha')

    print(f"\nSelected top {len(df_filtered_top)} filtered signals")
    print(f"Selected top {len(df_raw_top)} raw signals")

    # Get unique dates from alpha_df for this holding period
    period_dates = set(alpha_df['date'].unique())
    print(f"\nPeriod dates: {len(period_dates)}")

    return df_filtered_top, df_raw_top, period_dates


# ============================================================
# STEP 4: CREATE RANKINGS MATRIX (PARALLELIZED)
# ============================================================

def process_single_feature(args):
    """Process a single feature and return rankings."""
    feature_name, is_filtered, period_dates, date_to_idx, isin_to_idx = args

    db = SignalDatabase()

    try:
        # Load signal
        if is_filtered:
            data = db.load_filtered_signal_by_name(feature_name)
        else:
            data = db.load_signal_base(feature_name)

        if data is None:
            return None

        # Convert to long format
        if isinstance(data, pd.Series):
            df = data.reset_index()
            df.columns = ['date', 'isin', 'value']
        elif isinstance(data, pd.DataFrame):
            df = data.stack().reset_index()
            df.columns = ['date', 'isin', 'value']
        else:
            return None

        df['date'] = pd.to_datetime(df['date'])

        # Filter to period dates
        df = df[df['date'].isin(period_dates)]
        df = df.dropna(subset=['value'])

        # Compute ranks
        rankings = []
        for date, group in df.groupby('date'):
            if date not in date_to_idx:
                continue

            date_idx = date_to_idx[date]
            group = group.copy()
            group['rank'] = group['value'].rank(pct=True)

            for _, row in group.iterrows():
                if row['isin'] in isin_to_idx:
                    isin_idx = isin_to_idx[row['isin']]
                    rankings.append((date_idx, isin_idx, row['rank']))

        return rankings
    except Exception:
        return None


def create_rankings_matrix(filtered_features, raw_signals, alpha_df, period_dates):
    """Create a 3D matrix of rankings for all features (parallelized)."""
    print(f"\n{'='*60}")
    print(f"STEP 4: CREATE RANKINGS MATRIX (PARALLEL)")
    print(f"{'='*60}")

    # Get dimensions
    dates = sorted(period_dates)
    isins = sorted(alpha_df['isin'].unique())

    n_dates = len(dates)
    n_isins = len(isins)
    n_filtered = len(filtered_features)
    n_raw = len(raw_signals)
    n_features = n_filtered + n_raw

    print(f"\nMatrix dimensions:")
    print(f"  Dates: {n_dates}")
    print(f"  ISINs: {n_isins}")
    print(f"  Features: {n_features} ({n_filtered} filtered + {n_raw} raw)")
    print(f"  Total elements: {n_dates * n_isins * n_features:,}")

    # Initialize matrix
    rankings = np.full((n_dates, n_isins, n_features), np.nan, dtype=np.float32)

    # Create lookup dicts
    date_to_idx = {date: idx for idx, date in enumerate(dates)}
    isin_to_idx = {isin: idx for idx, isin in enumerate(isins)}

    # Prepare arguments for parallel processing
    filtered_args = [
        (row['feature_name'], True, period_dates, date_to_idx, isin_to_idx)
        for _, row in filtered_features.iterrows()
    ]

    raw_args = [
        (row['feature_name'], False, period_dates, date_to_idx, isin_to_idx)
        for _, row in raw_signals.iterrows()
    ]

    # Process filtered features in parallel
    print(f"\nLoading {len(filtered_args)} filtered features with {N_CORES} cores...")
    with Pool(N_CORES) as pool:
        filtered_results = list(tqdm(
            pool.imap(process_single_feature, filtered_args),
            total=len(filtered_args),
            desc="Filtered"
        ))

    # Process raw features in parallel
    print(f"\nLoading {len(raw_args)} raw features with {N_CORES} cores...")
    with Pool(N_CORES) as pool:
        raw_results = list(tqdm(
            pool.imap(process_single_feature, raw_args),
            total=len(raw_args),
            desc="Raw"
        ))

    # Fill matrix with filtered features
    for feat_idx, result in enumerate(filtered_results):
        if result is not None:
            for date_idx, isin_idx, rank in result:
                rankings[date_idx, isin_idx, feat_idx] = rank

    # Fill matrix with raw features
    for feat_idx, result in enumerate(raw_results):
        if result is not None:
            for date_idx, isin_idx, rank in result:
                rankings[date_idx, isin_idx, n_filtered + feat_idx] = rank

    # Create feature names list
    feature_names = filtered_features['feature_name'].tolist() + raw_signals['feature_name'].tolist()

    print(f"\nMatrix statistics:")
    print(f"  Non-NaN values: {(~np.isnan(rankings)).sum():,}")
    print(f"  Coverage: {(~np.isnan(rankings)).sum() / rankings.size * 100:.1f}%")

    return {
        'rankings': rankings,
        'dates': pd.to_datetime(dates),
        'isins': np.array(isins),
        'features': feature_names,
        'n_filtered': n_filtered
    }


# ============================================================
# STEP 5: GREEDY ENSEMBLE SEARCH
# ============================================================

def evaluate_ensemble(rankings, dates, isins, feature_indices, alpha_df):
    """Evaluate an ensemble of features."""
    n_satellites = N_SATELLITES

    # Average rankings across selected features
    ensemble_scores = np.nanmean(rankings[:, :, feature_indices], axis=2)

    # Pre-build alpha lookup for speed
    alpha_lookup = {}
    for _, row in alpha_df.iterrows():
        key = (row['date'], row['isin'])
        alpha_lookup[key] = row['forward_alpha']

    period_alphas = []
    hit_count = 0

    for date_idx, date in enumerate(dates):
        scores = ensemble_scores[date_idx, :]
        valid_mask = ~np.isnan(scores)

        if valid_mask.sum() < n_satellites:
            continue

        # Get top N satellites
        valid_scores = scores[valid_mask]
        valid_isins_arr = isins[valid_mask]
        top_indices = np.argsort(valid_scores)[-n_satellites:]
        selected_isins = valid_isins_arr[top_indices]

        # Lookup alphas
        alphas = []
        for isin in selected_isins:
            key = (date, isin)
            if key in alpha_lookup:
                alphas.append(alpha_lookup[key])

        if len(alphas) == 0:
            continue

        avg_alpha = np.mean(alphas)
        period_alphas.append(avg_alpha)

        if avg_alpha > 0:
            hit_count += 1

    if len(period_alphas) == 0:
        return None

    return {
        'avg_alpha': np.mean(period_alphas),
        'std_alpha': np.std(period_alphas),
        'hit_rate': hit_count / len(period_alphas),
        'n_periods': len(period_alphas)
    }


def evaluate_single_feature_for_filtering(args):
    """Evaluate a single feature for pre-filtering (parallelizable)."""
    feat_idx, rankings, dates, isins, alpha_df = args
    perf = evaluate_ensemble(rankings, dates, isins, [feat_idx], alpha_df)

    if perf is None:
        return None

    # Check if it's a positive or negative predictor
    if perf['avg_alpha'] >= MIN_ALPHA and perf['hit_rate'] >= MIN_HIT_RATE:
        return (feat_idx, 'positive', perf['avg_alpha'])
    elif perf['avg_alpha'] <= -MIN_ALPHA and perf['hit_rate'] >= MIN_HIT_RATE:
        return (feat_idx, 'negative', perf['avg_alpha'])
    else:
        return None


def check_inversion_needed(args):
    """Check if a feature needs inversion (parallelizable)."""
    feat_idx, rankings, dates, isins, alpha_df = args
    perf = evaluate_ensemble(rankings, dates, isins, [feat_idx], alpha_df)
    if perf is not None and perf['avg_alpha'] < 0:
        return feat_idx
    return None


def evaluate_candidate_addition(args):
    """Evaluate adding one candidate to current ensemble (parallelizable)."""
    feat_idx, current_indices, rankings, dates, isins, alpha_df, best_perf = args

    # Evaluate with this feature added
    test_indices = current_indices + [feat_idx]
    perf = evaluate_ensemble(rankings, dates, isins, test_indices, alpha_df)

    if perf is None:
        return None

    # Calculate improvement (weighted: 50% alpha, 50% hit rate)
    if best_perf is None:
        improvement = perf['avg_alpha'] * 0.5 + perf['hit_rate'] * 0.5
    else:
        improvement = (perf['avg_alpha'] - best_perf['avg_alpha']) * 0.5 + \
                    (perf['hit_rate'] - best_perf['hit_rate']) * 0.5

    return (feat_idx, improvement, perf)


def greedy_ensemble_search(rankings_data, alpha_df, holding_months):
    """Greedy forward selection for optimal ensemble."""
    print(f"\n{'='*60}")
    print(f"STEP 5: GREEDY ENSEMBLE SEARCH")
    print(f"{'='*60}")

    rankings = rankings_data['rankings'].copy()  # Make a copy to allow modifications
    dates = rankings_data['dates']
    isins = rankings_data['isins']
    feature_names = rankings_data['features']
    n_filtered = rankings_data['n_filtered']

    # Pre-filter features (PARALLELIZED)
    print("\nPre-filtering features...")
    print("Accepting features that either:")
    print("  1. Predict HIGH alpha (good at finding winners)")
    print("  2. Predict LOW alpha (good at finding losers to avoid)")

    # Prepare arguments for parallel processing
    filter_args = [
        (feat_idx, rankings, dates, isins, alpha_df)
        for feat_idx in range(len(feature_names))
    ]

    # Evaluate all features in parallel
    print(f"\nEvaluating {len(feature_names)} features with {N_CORES} cores...")
    with Pool(N_CORES) as pool:
        filter_results = list(tqdm(
            pool.imap(evaluate_single_feature_for_filtering, filter_args),
            total=len(filter_args),
            desc="Filtering"
        ))

    # Collect results
    candidate_indices = []
    positive_predictors = 0
    negative_predictors = 0

    for result in filter_results:
        if result is not None:
            feat_idx, pred_type, avg_alpha = result
            candidate_indices.append(feat_idx)
            if pred_type == 'positive':
                positive_predictors += 1
            else:
                negative_predictors += 1

    print(f"\nPre-filtered to {len(candidate_indices)} candidates")
    print(f"  Positive predictors (find winners): {positive_predictors}")
    print(f"  Negative predictors (find losers): {negative_predictors}")

    if len(candidate_indices) == 0:
        print("\nNo features passed pre-filtering!")
        return [], None

    # Invert rankings for negative predictors (parallelized)
    print("\nChecking which features need inversion...")

    inversion_args = [
        (feat_idx, rankings, dates, isins, alpha_df)
        for feat_idx in candidate_indices
    ]

    with Pool(N_CORES) as pool:
        inversion_results = list(tqdm(
            pool.imap(check_inversion_needed, inversion_args),
            total=len(inversion_args),
            desc="Checking"
        ))

    # Apply inversions
    features_to_invert = [idx for idx in inversion_results if idx is not None]
    print(f"\nInverting {len(features_to_invert)} negative predictors...")
    for feat_idx in features_to_invert:
        rankings[:, :, feat_idx] = 1.0 - rankings[:, :, feat_idx]
        print(f"  Inverted: {feature_names[feat_idx]}")

    # Greedy forward selection
    print("\nGreedy forward selection...")
    selected_features = []
    best_perf = None

    for iteration in range(MAX_ENSEMBLE_SIZE):
        print(f"\n--- Iteration {iteration + 1} ---")

        # Current ensemble indices
        current_indices = [f['idx'] for f in selected_features]

        # Remaining candidates
        remaining = [idx for idx in candidate_indices if idx not in current_indices]

        if len(remaining) == 0:
            print("No more candidates to test!")
            break

        # Prepare arguments for parallel evaluation
        eval_args = [
            (feat_idx, current_indices, rankings, dates, isins, alpha_df, best_perf)
            for feat_idx in remaining
        ]

        # Evaluate all candidates in parallel
        print(f"Testing {len(remaining)} candidates with {N_CORES} cores...")
        with Pool(N_CORES) as pool:
            eval_results = list(tqdm(
                pool.imap(evaluate_candidate_addition, eval_args),
                total=len(eval_args),
                desc="Candidates"
            ))

        # Find best addition
        best_addition = None
        best_addition_perf = None
        best_improvement = 0

        for result in eval_results:
            if result is not None:
                feat_idx, improvement, perf = result
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_addition = feat_idx
                    best_addition_perf = perf

        # Check if we found an improvement
        if best_addition is None or best_improvement < MIN_IMPROVEMENT:
            print(f"No improvement found (best improvement: {best_improvement:.6f})")
            break

        # Add the best feature
        feat_type = 'raw' if best_addition >= n_filtered else 'filtered'
        selected_features.append({
            'idx': best_addition,
            'name': feature_names[best_addition],
            'type': feat_type
        })
        best_perf = best_addition_perf

        print(f"Added: {feature_names[best_addition]} ({feat_type})")
        print(f"  Alpha: {best_perf['avg_alpha']:.4f} ({best_perf['avg_alpha']*100:.2f}%)")
        print(f"  Hit rate: {best_perf['hit_rate']:.2%}")
        print(f"  Improvement: {best_improvement:.6f}")

    if len(selected_features) == 0:
        print("\nNo ensemble found!")
        return [], None

    print(f"\n{'='*60}")
    print(f"FINAL ENSEMBLE ({len(selected_features)} features)")
    print(f"{'='*60}")

    for i, feat in enumerate(selected_features, 1):
        print(f"{i}. [{feat['type']:8s}] {feat['name']}")

    return selected_features, best_perf


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_single_period(holding_months):
    """Run the complete pipeline for a single holding period."""
    print("="*60)
    print(f"FEATURE ANALYSIS PIPELINE - {holding_months} MONTH HOLDING PERIOD")
    print("="*60)

    output_dir = Path('data/feature_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Compute forward alpha
    alpha_df = compute_forward_alpha(holding_months)

    alpha_file = output_dir / f'forward_alpha_{holding_months}month.parquet'
    alpha_df.to_parquet(alpha_file, index=False)
    print(f"\n[SAVED] {alpha_file}")

    # Step 2: Evaluate features
    feature_metrics = evaluate_features(alpha_df, holding_months)

    metrics_file = output_dir / f'feature_metrics_{holding_months}month.csv'
    feature_metrics.to_csv(metrics_file, index=False)
    print(f"\n[SAVED] {metrics_file}")

    # Step 3: Pre-filter features and load raw signals
    filtered_features, raw_signals, period_dates = load_and_filter_features(
        feature_metrics, alpha_df
    )

    # Step 4: Create rankings matrix
    rankings_data = create_rankings_matrix(
        filtered_features, raw_signals, alpha_df, period_dates
    )

    matrix_file = output_dir / f'rankings_matrix_{holding_months}month.npz'
    np.savez_compressed(
        matrix_file,
        rankings=rankings_data['rankings'],
        dates=rankings_data['dates'],
        isins=rankings_data['isins'],
        features=rankings_data['features'],
        n_filtered=rankings_data['n_filtered']
    )
    print(f"\n[SAVED] {matrix_file}")

    # Step 5: Greedy ensemble search
    selected_features, final_perf = greedy_ensemble_search(
        rankings_data, alpha_df, holding_months
    )

    # Check if ensemble was found
    if final_perf is None:
        print(f"\nWARNING: No ensemble found for {holding_months}-month holding period")
        return None

    # Save ensemble (convert from idx/name/type to feature_name/feature_type/selection_order)
    ensemble_data = []
    for i, feat in enumerate(selected_features, 1):
        ensemble_data.append({
            'feature_name': feat['name'],
            'feature_type': feat['type'],
            'selection_order': i
        })
    ensemble_df = pd.DataFrame(ensemble_data)

    ensemble_file = output_dir / f'ensemble_{holding_months}month.csv'
    ensemble_df.to_csv(ensemble_file, index=False)
    print(f"\n[SAVED] {ensemble_file}")

    # Save performance metrics
    perf_df = pd.DataFrame([{
        'holding_months': holding_months,
        'n_features': len(selected_features),
        'avg_alpha': final_perf['avg_alpha'],
        'std_alpha': final_perf['std_alpha'],
        'hit_rate': final_perf['hit_rate'],
        'n_periods': final_perf['n_periods'],
        'sharpe': final_perf['avg_alpha'] / final_perf['std_alpha'] if final_perf['std_alpha'] > 0 else 0
    }])

    perf_file = output_dir / f'ensemble_performance_{holding_months}month.csv'
    perf_df.to_csv(perf_file, index=False)
    print(f"\n[SAVED] {perf_file}")

    print(f"\n{'='*60}")
    print(f"[COMPLETE] {holding_months}-MONTH PIPELINE")
    print(f"{'='*60}")
    print(f"\nEnsemble: {len(selected_features)} features")
    print(f"Alpha: {final_perf['avg_alpha']:.4f} ({final_perf['avg_alpha']*100:.2f}%)")
    print(f"Hit rate: {final_perf['hit_rate']:.2%}")
    print(f"Sharpe: {perf_df['sharpe'].iloc[0]:.3f}")

    return perf_df


def main():
    """Run the complete pipeline for one or more holding periods."""
    # Convert HOLDING_MONTHS to list if it's a single integer
    holding_periods = HOLDING_MONTHS if isinstance(HOLDING_MONTHS, list) else [HOLDING_MONTHS]

    print("\n" + "="*60)
    print(f"RUNNING FEATURE ANALYSIS FOR {len(holding_periods)} HOLDING PERIOD(S): {holding_periods}")
    print("="*60 + "\n")

    # Run pipeline for each holding period
    all_results = []
    for i, months in enumerate(holding_periods, 1):
        print(f"\n{'#'*60}")
        print(f"# PERIOD {i}/{len(holding_periods)}: {months} MONTH(S)")
        print(f"{'#'*60}\n")

        result = run_single_period(months)
        if result is not None:
            all_results.append(result)

        # Add separator between runs
        if i < len(holding_periods):
            print("\n" + "="*60 + "\n")

    # Save combined summary
    if len(all_results) > 0:
        summary_df = pd.concat(all_results, ignore_index=True)
        summary_file = Path('data/feature_analysis') / 'all_periods_summary.csv'
        summary_df.to_csv(summary_file, index=False)

        print("\n" + "="*60)
        print("SUMMARY OF ALL HOLDING PERIODS")
        print("="*60)
        print(summary_df.to_string(index=False))
        print(f"\n[SAVED] {summary_file}")


if __name__ == '__main__':
    main()
