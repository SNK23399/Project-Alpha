"""
Step 4: Analyze Feature Predictive Power

This script analyzes each feature individually to identify which features
have genuine predictive power for future ETF outperformance.

Key Metrics:
1. Hit Rate: % of times feature correctly predicts outperformance vs core
2. Expected Alpha: Average alpha when following the signal
3. Sharpe Ratio: Risk-adjusted return of the signal
4. Information Ratio: Alpha / tracking error
5. Win/Loss Ratio: Average winner / average loser
6. Correlation with Future Returns: Predictive correlation

Strategy: For each feature, we test a simple rule:
  - If feature value is HIGH (top quartile), expect ETF to OUTPERFORM
  - If feature value is LOW (bottom quartile), expect ETF to UNDERPERFORM

This walk-forward analysis prevents overfitting by:
  - Only using historical data for decisions
  - Testing one feature at a time (no combinations)
  - Using simple quartile thresholds (no optimization)
  - Measuring out-of-sample performance

Output: Ranked list of features by predictive power
"""

import time
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from support.signal_database import SignalDatabase
from support.etf_database import ETFDatabase


def load_core_returns(etf_db: ETFDatabase, core_isin: str) -> pd.Series:
    """Load core ETF returns for comparison."""
    print(f"Loading core ETF returns ({core_isin})...")

    # Get core prices (returns as Series)
    core_prices = etf_db.load_prices(core_isin)
    if core_prices is None or len(core_prices) == 0:
        raise ValueError(f"Core ETF {core_isin} not found in database")

    # Calculate daily returns
    core_returns = core_prices.pct_change()

    print(f"  Loaded {len(core_returns)} days of core returns")
    return core_returns


def load_all_etf_returns(etf_db: ETFDatabase) -> pd.DataFrame:
    """Load returns for all ETFs in database."""
    print("Loading all ETF returns...")

    # Get all ISINs from universe
    universe_df = etf_db.load_universe()
    all_isins = universe_df['isin'].tolist()
    print(f"  Found {len(all_isins)} ETFs")

    returns_dict = {}
    for isin in tqdm(all_isins, desc="Loading returns", ncols=100):
        prices = etf_db.load_prices(isin)
        if prices is not None and len(prices) > 0:
            returns_dict[isin] = prices.pct_change()

    returns_df = pd.DataFrame(returns_dict)
    print(f"  Loaded returns for {len(returns_df.columns)} ETFs")
    print(f"  Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")

    return returns_df


def analyze_feature_predictive_power(
    feature_file: Path,
    signal_db: SignalDatabase,
    etf_returns: pd.DataFrame,
    core_returns: pd.Series,
    forward_periods: int = 21,
    min_observations: int = 252
) -> list:
    """
    Analyze predictive power of ALL indicators for a single signal+filter.

    Strategy for each indicator:
    1. For each date, get feature value (cross-sectional statistic)
    2. Check if HIGH feature value predicts HIGH avg alpha across universe
    3. Measure: correlation, hit rate, information ratio

    Args:
        feature_file: Path to feature parquet file
        signal_db: SignalDatabase instance
        etf_returns: DataFrame of ETF returns (dates × ISINs)
        core_returns: Series of core ETF returns
        forward_periods: Days to hold position (default 21 = 1 month)
        min_observations: Minimum history required before testing

    Returns:
        list of dicts with performance metrics (one per indicator)
    """
    # Parse signal + filter name from filename
    signal_filter_name = feature_file.stem

    # Load the FILTERED SIGNAL (not the features yet)
    # This is the (time × ETFs) array we need to compute forward alphas
    try:
        signal_df = signal_db.load_filtered_signal_by_name(signal_filter_name)
    except Exception as e:
        return [{'error': f'Failed to load signal: {str(e)}'}]

    if len(signal_df) == 0:
        return [{'error': 'Empty signal file'}]

    # Load feature data (cross-sectional indicators)
    try:
        feature_df = pd.read_parquet(feature_file)
    except Exception as e:
        return [{'error': f'Failed to load features: {str(e)}'}]

    # Get all indicator columns
    indicator_cols = feature_df.columns.tolist()

    # Align dates
    common_dates = (signal_df.index
                    .intersection(etf_returns.index)
                    .intersection(feature_df.index))

    if len(common_dates) < min_observations + forward_periods:
        return [{'error': f'Insufficient data: {len(common_dates)} days'}]

    signal_df = signal_df.loc[common_dates]
    feature_df = feature_df.loc[common_dates]
    etf_returns_aligned = etf_returns.loc[common_dates]
    core_returns_aligned = core_returns.loc[common_dates]

    # Align columns (ISINs)
    common_isins = signal_df.columns.intersection(etf_returns_aligned.columns)
    signal_df = signal_df[common_isins]
    etf_returns_aligned = etf_returns_aligned[common_isins]

    # Calculate forward returns for each date
    # For efficiency, pre-compute forward alphas for all dates
    forward_alphas = []

    for i in range(len(common_dates) - forward_periods):
        current_date = common_dates[i]
        future_slice = slice(i + 1, i + 1 + forward_periods)

        # Forward returns for each ETF
        fwd_rets = etf_returns_aligned.iloc[future_slice].sum(axis=0)

        # Core forward return
        core_fwd_ret = core_returns_aligned.iloc[future_slice].sum()

        # Alpha vs core
        alphas = fwd_rets - core_fwd_ret

        forward_alphas.append({
            'date': current_date,
            'mean_alpha': alphas.mean(),
            'median_alpha': alphas.median(),
            'std_alpha': alphas.std(),
            'hit_rate': (alphas > 0).mean(),  # % of ETFs with positive alpha
            'core_return': core_fwd_ret
        })

    forward_alphas_df = pd.DataFrame(forward_alphas).set_index('date')

    # Now analyze each indicator
    all_results = []

    for indicator_col in indicator_cols:
        feature_series = feature_df[indicator_col].iloc[:-forward_periods]  # Align with forward alphas

        # Skip if too many NaN
        if feature_series.isna().sum() / len(feature_series) > 0.5:
            continue

        # Align with forward alphas
        aligned_dates = forward_alphas_df.index.intersection(feature_series.index)
        if len(aligned_dates) < min_observations:
            continue

        feature_vals = feature_series.loc[aligned_dates]
        alpha_vals = forward_alphas_df.loc[aligned_dates, 'mean_alpha']

        # Calculate correlation
        corr = feature_vals.corr(alpha_vals)

        if pd.isna(corr):
            continue

        # Strategy: Buy when feature is HIGH (top 25%)
        threshold_high = feature_vals.quantile(0.75)
        threshold_low = feature_vals.quantile(0.25)

        high_signal = feature_vals >= threshold_high
        low_signal = feature_vals <= threshold_low

        if high_signal.sum() > 0:
            alpha_when_high = alpha_vals[high_signal].mean()
            hit_rate_when_high = (alpha_vals[high_signal] > 0).mean()
        else:
            alpha_when_high = 0
            hit_rate_when_high = 0

        if low_signal.sum() > 0:
            alpha_when_low = alpha_vals[low_signal].mean()
            hit_rate_when_low = (alpha_vals[low_signal] > 0).mean()
        else:
            alpha_when_low = 0
            hit_rate_when_low = 0

        # Long-short: go LONG when high, SHORT when low
        long_short_alpha = alpha_when_high - alpha_when_low

        all_results.append({
            'signal_filter': signal_filter_name,
            'indicator': indicator_col.split('__')[-1],  # Extract indicator name
            'n_observations': len(aligned_dates),
            'correlation': corr,
            'alpha_when_high': alpha_when_high,
            'alpha_when_low': alpha_when_low,
            'long_short_alpha': long_short_alpha,
            'hit_rate_when_high': hit_rate_when_high,
            'hit_rate_when_low': hit_rate_when_low,
            'avg_alpha_universe': alpha_vals.mean(),
            'std_alpha_universe': alpha_vals.std(),
            'information_ratio': alpha_vals.mean() / alpha_vals.std() if alpha_vals.std() > 0 else 0
        })

    return all_results


def analyze_all_features(
    forward_periods: int = 21,
    min_observations: int = 252,
    max_features: int = None
):
    """
    Analyze predictive power of ALL features.

    Args:
        forward_periods: Days to hold position (21 = monthly, 63 = quarterly)
        min_observations: Minimum history before testing
        max_features: Maximum number of features to test (None = all)
    """
    print("=" * 80)
    print("STEP 4: FEATURE PREDICTIVE POWER ANALYSIS")
    print("=" * 80)
    print(f"\nForward period: {forward_periods} days (~{forward_periods/21:.1f} months)")
    print(f"Minimum history: {min_observations} days (~{min_observations/252:.1f} years)")

    # Initialize databases
    signal_db = SignalDatabase("data/signals")
    etf_db = ETFDatabase("data/etf_database.db")

    # Load core returns
    core_isin = 'IE00B4L5Y983'  # iShares Core MSCI World
    core_returns = load_core_returns(etf_db, core_isin)

    # Load all ETF returns
    etf_returns = load_all_etf_returns(etf_db)

    # Get all feature files
    feature_dir = Path("data/features")
    feature_files = sorted(feature_dir.glob("*.parquet"))

    print(f"\nFound {len(feature_files)} feature files")

    if max_features:
        feature_files = feature_files[:max_features]
        print(f"Limiting to first {max_features} features for testing")

    # Analyze each feature
    print(f"\nAnalyzing features...")
    print("=" * 80)

    results = []
    errors = []

    pbar = tqdm(feature_files, desc="Analyzing features", unit="feature", ncols=100)

    for feature_file in pbar:
        signal_filter_name = feature_file.stem
        pbar.set_description(f"Analyzing {signal_filter_name[:30]:30s}")

        try:
            feature_metrics = analyze_feature_predictive_power(
                feature_file,
                signal_db,
                etf_returns,
                core_returns,
                forward_periods=forward_periods,
                min_observations=min_observations
            )

            # feature_metrics is a list of dicts (one per indicator)
            for metrics in feature_metrics:
                if 'error' in metrics:
                    errors.append((signal_filter_name, metrics['error']))
                else:
                    results.append(metrics)

        except Exception as e:
            errors.append((signal_filter_name, str(e)))
            pbar.write(f"  ERROR: {signal_filter_name} - {str(e)[:50]}")

    pbar.close()

    # Convert to DataFrame and rank
    if len(results) == 0:
        print("\nNo valid results!")
        return None

    results_df = pd.DataFrame(results)

    # Rank by multiple criteria
    results_df['rank_alpha'] = results_df['alpha_when_high'].rank(ascending=False)
    results_df['rank_long_short'] = results_df['long_short_alpha'].rank(ascending=False)
    results_df['rank_corr'] = results_df['correlation'].abs().rank(ascending=False)
    results_df['rank_hit_rate'] = results_df['hit_rate_when_high'].rank(ascending=False)

    # Combined rank (lower is better)
    results_df['combined_rank'] = (
        results_df['rank_alpha'] * 0.3 +           # 30% weight: alpha when signal is high
        results_df['rank_long_short'] * 0.3 +      # 30% weight: long-short alpha
        results_df['rank_corr'] * 0.2 +            # 20% weight: correlation strength
        results_df['rank_hit_rate'] * 0.2          # 20% weight: hit rate
    )

    results_df = results_df.sort_values('combined_rank')

    # Save results
    output_dir = Path("data/feature_analysis")
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / f"predictive_power_forward_{forward_periods}d.parquet"
    results_df.to_parquet(output_file)

    # Also save as CSV for easy viewing
    csv_file = output_dir / f"predictive_power_forward_{forward_periods}d.csv"
    results_df.to_csv(csv_file, index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAnalyzed: {len(results_df)} features")
    print(f"Errors: {len(errors)}")
    print(f"\nResults saved to:")
    print(f"  {output_file}")
    print(f"  {csv_file}")

    print("\n" + "=" * 80)
    print("TOP 30 FEATURES BY PREDICTIVE POWER")
    print("=" * 80)

    top_features = results_df.head(30)[[
        'signal_filter', 'indicator', 'correlation', 'alpha_when_high',
        'long_short_alpha', 'hit_rate_when_high', 'n_observations'
    ]]

    # Format for better readability
    display_df = top_features.copy()
    display_df['correlation'] = display_df['correlation'].apply(lambda x: f"{x:+.3f}")
    display_df['alpha_when_high'] = display_df['alpha_when_high'].apply(lambda x: f"{x:+.4f}")
    display_df['long_short_alpha'] = display_df['long_short_alpha'].apply(lambda x: f"{x:+.4f}")
    display_df['hit_rate_when_high'] = display_df['hit_rate_when_high'].apply(lambda x: f"{x:.1%}")

    print(display_df.to_string(index=False))

    if errors:
        print(f"\n{len(errors)} features had errors:")
        for name, error in errors[:10]:
            print(f"  - {name}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    return results_df


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    forward_periods = 21  # Default: 1 month
    max_features = None   # Default: all features

    if len(sys.argv) > 1:
        forward_periods = int(sys.argv[1])

    if len(sys.argv) > 2:
        max_features = int(sys.argv[2])

    # Run analysis
    start_time = time.time()

    results_df = analyze_all_features(
        forward_periods=forward_periods,
        max_features=max_features
    )

    elapsed = time.time() - start_time

    print(f"\nTotal time: {elapsed / 60:.1f} minutes")

    print("\nUsage:")
    print("  python 4_analyze_feature_predictive_power.py [forward_periods] [max_features]")
    print("\nExamples:")
    print("  python 4_analyze_feature_predictive_power.py          # 21-day forward, all features")
    print("  python 4_analyze_feature_predictive_power.py 63       # 63-day (quarterly) forward")
    print("  python 4_analyze_feature_predictive_power.py 21 100   # 21-day, first 100 features only")
