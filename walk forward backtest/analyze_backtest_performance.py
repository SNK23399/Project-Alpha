"""
Backtest Performance Analysis
=============================

Analyzes when the strategy works well and when it doesn't.
Aggregates patterns across ALL N values for robust insights.

Examines:
1. Hit vs Miss months - what's different?
2. High alpha vs Low alpha months - what drives performance?
3. Market conditions during good/bad periods
4. Selected ETF characteristics
5. Time patterns and seasonality
6. Market regime analysis

Usage:
    python analyze_backtest_performance.py

Analyzes all available N values (1-10) to find patterns that generalize.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = DATA_DIR / 'backtest_results'


def load_backtest_results(n_satellites: int = 5) -> pd.DataFrame:
    """Load backtest results for a specific N."""
    results_file = RESULTS_DIR / f'decay_backtest_N{n_satellites}.csv'
    if not results_file.exists():
        return None

    df = pd.read_csv(results_file)
    df['date'] = pd.to_datetime(df['date'])
    df['n_satellites'] = n_satellites
    df = df.set_index('date')
    return df


def load_all_backtest_results() -> pd.DataFrame:
    """Load backtest results for all available N values."""
    all_results = []

    for n in range(1, 11):
        df = load_backtest_results(n)
        if df is not None:
            all_results.append(df)

    if not all_results:
        raise FileNotFoundError("No backtest results found. Run 6_backtest_strategy.py first.")

    combined = pd.concat(all_results)
    return combined


def load_market_data() -> pd.DataFrame:
    """Load core ETF (ACWI) prices for market context."""
    from support.etf_database import ETFDatabase

    etf_db = ETFDatabase("data/etf_database.db")
    core_isin = 'IE00B4L5Y983'  # MSCI ACWI

    prices = etf_db.load_prices(core_isin)
    if prices is None or len(prices) == 0:
        return None

    # Compute market metrics
    market = pd.DataFrame(index=prices.index)
    market['price'] = prices
    market['return_1m'] = prices.pct_change(21)
    market['return_3m'] = prices.pct_change(63)
    market['return_6m'] = prices.pct_change(126)
    market['return_12m'] = prices.pct_change(252)
    market['volatility_21d'] = prices.pct_change().rolling(21).std() * np.sqrt(252)
    market['volatility_63d'] = prices.pct_change().rolling(63).std() * np.sqrt(252)

    # Drawdown
    rolling_max = prices.expanding().max()
    market['drawdown'] = (prices - rolling_max) / rolling_max

    # Trend indicators
    market['above_sma_50'] = (prices > prices.rolling(50).mean()).astype(int)
    market['above_sma_200'] = (prices > prices.rolling(200).mean()).astype(int)

    # Resample to monthly (end of month)
    market_monthly = market.resample('ME').last()

    return market_monthly


def analyze_hits_vs_misses(results: pd.DataFrame, market: pd.DataFrame = None):
    """Analyze what's different between hit months (alpha > 0) and miss months.

    Aggregates across all N values for robust patterns.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 1: HIT vs MISS MONTHS (Aggregated Across All N)")
    print("=" * 80)

    hits = results[results['avg_alpha'] > 0]
    misses = results[results['avg_alpha'] <= 0]

    n_values = results['n_satellites'].unique()
    print(f"\nAnalyzing N values: {sorted(n_values)}")
    print(f"Total observations: {len(results)} ({len(hits)} hits, {len(misses)} misses)")

    # Per-N breakdown
    print(f"\n{'N':<5} {'Hits':>8} {'Misses':>8} {'Hit Rate':>10} {'Avg Alpha':>12}")
    print("-" * 50)

    for n in sorted(n_values):
        n_data = results[results['n_satellites'] == n]
        n_hits = (n_data['avg_alpha'] > 0).sum()
        n_misses = len(n_data) - n_hits
        hit_rate = n_hits / len(n_data)
        avg_alpha = n_data['avg_alpha'].mean()
        print(f"{n:<5} {n_hits:>8} {n_misses:>8} {hit_rate:>9.1%} {avg_alpha*100:>11.2f}%")

    print(f"\n{'Metric':<30} {'Hits':>15} {'Misses':>15} {'Difference':>15}")
    print("-" * 80)

    # Alpha statistics
    print(f"{'Mean Alpha':<30} {hits['avg_alpha'].mean()*100:>14.2f}% {misses['avg_alpha'].mean()*100:>14.2f}% {(hits['avg_alpha'].mean() - misses['avg_alpha'].mean())*100:>14.2f}%")
    print(f"{'Median Alpha':<30} {hits['avg_alpha'].median()*100:>14.2f}% {misses['avg_alpha'].median()*100:>14.2f}% {(hits['avg_alpha'].median() - misses['avg_alpha'].median())*100:>14.2f}%")
    print(f"{'Std Alpha':<30} {hits['avg_alpha'].std()*100:>14.2f}% {misses['avg_alpha'].std()*100:>14.2f}%")

    # If we have market data, analyze market conditions
    if market is not None:
        print("\n" + "-" * 80)
        print("Market Conditions (averaged across all N):")
        print("-" * 80)

        # Get unique dates for hits and misses
        hit_dates = hits.index.unique().intersection(market.index)
        miss_dates = misses.index.unique().intersection(market.index)

        if len(hit_dates) > 0 and len(miss_dates) > 0:
            hit_market = market.loc[hit_dates]
            miss_market = market.loc[miss_dates]

            metrics = ['return_1m', 'return_3m', 'volatility_21d', 'drawdown', 'above_sma_200']

            for metric in metrics:
                if metric in hit_market.columns:
                    hit_val = hit_market[metric].mean()
                    miss_val = miss_market[metric].mean()

                    if metric.startswith('return') or metric == 'drawdown':
                        print(f"{metric:<30} {hit_val*100:>14.2f}% {miss_val*100:>14.2f}% {(hit_val-miss_val)*100:>14.2f}%")
                    elif metric.startswith('volatility'):
                        print(f"{metric:<30} {hit_val*100:>14.1f}% {miss_val*100:>14.1f}% {(hit_val-miss_val)*100:>14.1f}%")
                    else:
                        print(f"{metric:<30} {hit_val:>15.2f} {miss_val:>15.2f} {hit_val-miss_val:>15.2f}")

    # Analyze selected ETFs in hits vs misses
    print("\n" + "-" * 80)
    print("Selected ETF Analysis (across all N):")
    print("-" * 80)

    # Count ETF frequency in hits vs misses
    hit_etfs = []
    miss_etfs = []

    for _, row in hits.iterrows():
        if pd.notna(row.get('selected_isins')):
            hit_etfs.extend(row['selected_isins'].split(','))

    for _, row in misses.iterrows():
        if pd.notna(row.get('selected_isins')):
            miss_etfs.extend(row['selected_isins'].split(','))

    hit_counts = pd.Series(hit_etfs).value_counts()
    miss_counts = pd.Series(miss_etfs).value_counts()

    # ETFs that appear more in hits
    all_etfs = set(hit_counts.index) | set(miss_counts.index)
    etf_analysis = []

    total_hits = len(hits)
    total_misses = len(misses)

    for etf in all_etfs:
        hit_count = hit_counts.get(etf, 0)
        miss_count = miss_counts.get(etf, 0)
        # Normalize by total observations
        hit_freq = hit_count / total_hits if total_hits > 0 else 0
        miss_freq = miss_count / total_misses if total_misses > 0 else 0
        etf_analysis.append({
            'isin': etf,
            'hit_count': hit_count,
            'miss_count': miss_count,
            'hit_freq': hit_freq,
            'miss_freq': miss_freq,
            'diff': hit_freq - miss_freq
        })

    etf_df = pd.DataFrame(etf_analysis).sort_values('diff', ascending=False)

    print("\nETFs appearing MORE in HIT months (top 10):")
    for _, row in etf_df.head(10).iterrows():
        print(f"  {row['isin']}: {row['hit_freq']:.1%} in hits ({int(row['hit_count'])}x), {row['miss_freq']:.1%} in misses ({int(row['miss_count'])}x)")

    print("\nETFs appearing MORE in MISS months (top 10):")
    for _, row in etf_df.tail(10).iterrows():
        print(f"  {row['isin']}: {row['hit_freq']:.1%} in hits ({int(row['hit_count'])}x), {row['miss_freq']:.1%} in misses ({int(row['miss_count'])}x)")

    return hits, misses, etf_df


def analyze_alpha_quartiles(results: pd.DataFrame, market: pd.DataFrame = None):
    """Analyze characteristics of high vs low alpha months.

    Aggregates across all N values for robust patterns.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 2: HIGH vs LOW ALPHA MONTHS (Quartiles, All N)")
    print("=" * 80)

    results = results.copy()

    # Split into quartiles
    results['alpha_quartile'] = pd.qcut(results['avg_alpha'], 4, labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4 (Best)'])

    print(f"\n{'Quartile':<15} {'Count':>8} {'Mean Alpha':>12} {'Min Alpha':>12} {'Max Alpha':>12}")
    print("-" * 60)

    for q in ['Q1 (Worst)', 'Q2', 'Q3', 'Q4 (Best)']:
        subset = results[results['alpha_quartile'] == q]
        print(f"{q:<15} {len(subset):>8} {subset['avg_alpha'].mean()*100:>11.2f}% {subset['avg_alpha'].min()*100:>11.2f}% {subset['avg_alpha'].max()*100:>11.2f}%")

    # N distribution by quartile
    print("\n" + "-" * 80)
    print("N Distribution by Quartile (which N values end up in each quartile):")
    print("-" * 80)

    print(f"\n{'N':<5}", end="")
    for q in ['Q1 (Worst)', 'Q2', 'Q3', 'Q4 (Best)']:
        print(f"{q:>12}", end="")
    print()
    print("-" * 55)

    for n in sorted(results['n_satellites'].unique()):
        n_data = results[results['n_satellites'] == n]
        print(f"{n:<5}", end="")
        for q in ['Q1 (Worst)', 'Q2', 'Q3', 'Q4 (Best)']:
            count = len(n_data[n_data['alpha_quartile'] == q])
            pct = count / len(n_data) * 100
            print(f"{pct:>11.0f}%", end="")
        print()

    # Compare Q1 (worst) vs Q4 (best)
    q1 = results[results['alpha_quartile'] == 'Q1 (Worst)']
    q4 = results[results['alpha_quartile'] == 'Q4 (Best)']

    if market is not None:
        print("\n" + "-" * 80)
        print("Market Conditions by Quartile:")
        print("-" * 80)

        print(f"\n{'Metric':<25} {'Q1 (Worst)':>15} {'Q4 (Best)':>15} {'Difference':>15}")
        print("-" * 70)

        q1_dates = q1.index.unique().intersection(market.index)
        q4_dates = q4.index.unique().intersection(market.index)

        if len(q1_dates) > 0 and len(q4_dates) > 0:
            q1_market = market.loc[q1_dates]
            q4_market = market.loc[q4_dates]

            metrics = ['return_1m', 'return_3m', 'return_6m', 'volatility_21d', 'volatility_63d', 'drawdown', 'above_sma_200']

            for metric in metrics:
                if metric in q1_market.columns:
                    q1_val = q1_market[metric].mean()
                    q4_val = q4_market[metric].mean()

                    if metric.startswith('return') or metric == 'drawdown':
                        print(f"{metric:<25} {q1_val*100:>14.2f}% {q4_val*100:>14.2f}% {(q4_val-q1_val)*100:>14.2f}%")
                    elif metric.startswith('volatility'):
                        print(f"{metric:<25} {q1_val*100:>14.1f}% {q4_val*100:>14.1f}% {(q4_val-q1_val)*100:>14.1f}%")
                    else:
                        print(f"{metric:<25} {q1_val:>15.2f} {q4_val:>15.2f} {q4_val-q1_val:>15.2f}")

    return q1, q4


def analyze_time_patterns(results: pd.DataFrame):
    """Analyze performance patterns over time.

    Aggregates across all N values for robust patterns.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 3: TIME PATTERNS (Aggregated Across All N)")
    print("=" * 80)

    results = results.copy()
    results['year'] = results.index.year
    results['month'] = results.index.month
    results['quarter'] = results.index.quarter

    # By year - average across all N values
    print("\nPerformance by Year (averaged across all N):")
    print("-" * 80)
    print(f"{'Year':<8} {'Obs':>8} {'Avg Alpha':>12} {'Hit Rate':>12} {'Consistency':>12}")
    print("-" * 80)

    for year in sorted(results['year'].unique()):
        year_data = results[results['year'] == year]
        hit_rate = (year_data['avg_alpha'] > 0).mean()
        # Consistency = how many N values have positive avg alpha this year
        n_positive = 0
        for n in year_data['n_satellites'].unique():
            n_year = year_data[year_data['n_satellites'] == n]
            if n_year['avg_alpha'].mean() > 0:
                n_positive += 1
        consistency = n_positive / len(year_data['n_satellites'].unique())
        print(f"{year:<8} {len(year_data):>8} {year_data['avg_alpha'].mean()*100:>11.2f}% {hit_rate:>11.1%} {consistency:>11.0%}")

    # By month of year
    print("\nPerformance by Month of Year:")
    print("-" * 60)
    print(f"{'Month':<12} {'Avg Alpha':>12} {'Hit Rate':>12} {'Count':>8}")
    print("-" * 60)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for month in range(1, 13):
        month_data = results[results['month'] == month]
        if len(month_data) > 0:
            hit_rate = (month_data['avg_alpha'] > 0).mean()
            print(f"{month_names[month-1]:<12} {month_data['avg_alpha'].mean()*100:>11.2f}% {hit_rate:>11.1%} {len(month_data):>8}")

    # By quarter
    print("\nPerformance by Quarter:")
    print("-" * 60)
    print(f"{'Quarter':<12} {'Avg Alpha':>12} {'Hit Rate':>12} {'Count':>8}")
    print("-" * 60)

    for q in range(1, 5):
        q_data = results[results['quarter'] == q]
        if len(q_data) > 0:
            hit_rate = (q_data['avg_alpha'] > 0).mean()
            print(f"Q{q:<11} {q_data['avg_alpha'].mean()*100:>11.2f}% {hit_rate:>11.1%} {len(q_data):>8}")

    # Consecutive patterns
    print("\n" + "-" * 80)
    print("Consecutive Hit/Miss Streaks:")
    print("-" * 80)

    results['is_hit'] = results['avg_alpha'] > 0

    # Find streaks
    streaks = []
    current_streak = 1
    current_type = results['is_hit'].iloc[0]

    for i in range(1, len(results)):
        if results['is_hit'].iloc[i] == current_type:
            current_streak += 1
        else:
            streaks.append((current_type, current_streak))
            current_streak = 1
            current_type = results['is_hit'].iloc[i]
    streaks.append((current_type, current_streak))

    hit_streaks = [s[1] for s in streaks if s[0]]
    miss_streaks = [s[1] for s in streaks if not s[0]]

    print(f"Hit streaks:  max={max(hit_streaks) if hit_streaks else 0}, avg={np.mean(hit_streaks) if hit_streaks else 0:.1f}, count={len(hit_streaks)}")
    print(f"Miss streaks: max={max(miss_streaks) if miss_streaks else 0}, avg={np.mean(miss_streaks) if miss_streaks else 0:.1f}, count={len(miss_streaks)}")


def analyze_etf_performance(results: pd.DataFrame):
    """Analyze which ETFs contribute most to performance.

    Aggregates across all N values for robust patterns.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 4: ETF-LEVEL PERFORMANCE (Across All N)")
    print("=" * 80)

    # Load forward alpha data to get individual ETF alphas
    alpha_file = DATA_DIR / 'forward_alpha_1month.parquet'
    if not alpha_file.exists():
        print("Forward alpha file not found, skipping ETF-level analysis")
        return

    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])

    # Build ETF performance summary
    etf_stats = []

    for _, row in results.iterrows():
        date = row.name
        n_sat = row.get('n_satellites', 0)
        selected = row.get('selected_isins', '')
        if pd.isna(selected) or selected == '':
            continue

        selected_isins = selected.split(',')
        date_alpha = alpha_df[alpha_df['date'] == date]

        for isin in selected_isins:
            isin_alpha = date_alpha[date_alpha['isin'] == isin]
            if len(isin_alpha) > 0:
                etf_stats.append({
                    'date': date,
                    'n_satellites': n_sat,
                    'isin': isin,
                    'alpha': isin_alpha['forward_alpha'].values[0],
                    'portfolio_alpha': row['avg_alpha']
                })

    if len(etf_stats) == 0:
        print("No ETF-level data available")
        return

    etf_df = pd.DataFrame(etf_stats)

    # Aggregate by ETF
    etf_summary = etf_df.groupby('isin').agg({
        'alpha': ['mean', 'std', 'count', lambda x: (x > 0).mean()]
    }).round(4)
    etf_summary.columns = ['avg_alpha', 'std_alpha', 'times_selected', 'hit_rate']
    etf_summary = etf_summary.sort_values('avg_alpha', ascending=False)

    # Count unique N values each ETF appears in
    etf_n_counts = etf_df.groupby('isin')['n_satellites'].nunique()
    etf_summary['n_values'] = etf_n_counts

    print(f"\nTotal unique ETFs selected: {len(etf_summary)}")
    print(f"Total selections: {len(etf_df)}")

    print("\nTop 15 ETFs by Average Alpha (when selected):")
    print("-" * 90)
    print(f"{'ISIN':<20} {'Avg Alpha':>12} {'Std Alpha':>12} {'Hit Rate':>10} {'Selected':>10} {'N vals':>8}")
    print("-" * 90)

    for isin, row in etf_summary.head(15).iterrows():
        print(f"{isin:<20} {row['avg_alpha']*100:>11.2f}% {row['std_alpha']*100:>11.2f}% {row['hit_rate']:>9.1%} {int(row['times_selected']):>10} {int(row['n_values']):>8}")

    print("\nBottom 15 ETFs by Average Alpha (when selected):")
    print("-" * 90)
    print(f"{'ISIN':<20} {'Avg Alpha':>12} {'Std Alpha':>12} {'Hit Rate':>10} {'Selected':>10} {'N vals':>8}")
    print("-" * 90)

    for isin, row in etf_summary.tail(15).iterrows():
        print(f"{isin:<20} {row['avg_alpha']*100:>11.2f}% {row['std_alpha']*100:>11.2f}% {row['hit_rate']:>9.1%} {int(row['times_selected']):>10} {int(row['n_values']):>8}")

    # Most frequently selected (appears across many N values = robust signal)
    print("\nMost Frequently Selected ETFs (robust across N):")
    print("-" * 90)
    most_selected = etf_summary.sort_values(['n_values', 'times_selected'], ascending=[False, False]).head(15)

    for isin, row in most_selected.iterrows():
        print(f"{isin:<20} In {int(row['n_values'])} N values, {int(row['times_selected']):>4}x total, Alpha: {row['avg_alpha']*100:>6.2f}%, Hit: {row['hit_rate']:.0%}")

    return etf_summary


def analyze_regime_performance(results: pd.DataFrame, market: pd.DataFrame):
    """Analyze performance in different market regimes.

    Aggregates across all N values for robust patterns.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 5: MARKET REGIME ANALYSIS (Across All N)")
    print("=" * 80)

    if market is None:
        print("No market data available")
        return

    # For regime analysis, we need to match dates
    # Since results has multiple rows per date (one per N), we need unique dates
    unique_dates = results.index.unique().intersection(market.index)

    results_aligned = results.loc[results.index.isin(unique_dates)].copy()

    # Add market data to results
    for col in ['above_sma_200', 'volatility_63d', 'drawdown', 'return_3m']:
        if col in market.columns:
            results_aligned[col] = results_aligned.index.map(lambda d: market.loc[d, col] if d in market.index else np.nan)

    # Define regimes
    print("\nPerformance by Market Trend (SMA 200):")
    print("-" * 80)

    bull = results_aligned[results_aligned['above_sma_200'] == 1]
    bear = results_aligned[results_aligned['above_sma_200'] == 0]

    print(f"{'Regime':<25} {'Obs':>8} {'Avg Alpha':>12} {'Hit Rate':>12} {'Sharpe':>10}")
    print("-" * 80)

    if len(bull) > 0:
        bull_sharpe = bull['avg_alpha'].mean() / bull['avg_alpha'].std() if bull['avg_alpha'].std() > 0 else 0
        print(f"{'Bull (above SMA200)':<25} {len(bull):>8} {bull['avg_alpha'].mean()*100:>11.2f}% {(bull['avg_alpha']>0).mean():>11.1%} {bull_sharpe:>10.2f}")

    if len(bear) > 0:
        bear_sharpe = bear['avg_alpha'].mean() / bear['avg_alpha'].std() if bear['avg_alpha'].std() > 0 else 0
        print(f"{'Bear (below SMA200)':<25} {len(bear):>8} {bear['avg_alpha'].mean()*100:>11.2f}% {(bear['avg_alpha']>0).mean():>11.1%} {bear_sharpe:>10.2f}")

    # Breakdown by N within each regime
    print("\n  Bull market by N:")
    for n in sorted(results_aligned['n_satellites'].unique()):
        n_bull = bull[bull['n_satellites'] == n]
        if len(n_bull) > 0:
            print(f"    N={n}: {n_bull['avg_alpha'].mean()*100:>6.2f}% alpha, {(n_bull['avg_alpha']>0).mean():>5.0%} hit rate")

    print("\n  Bear market by N:")
    for n in sorted(results_aligned['n_satellites'].unique()):
        n_bear = bear[bear['n_satellites'] == n]
        if len(n_bear) > 0:
            print(f"    N={n}: {n_bear['avg_alpha'].mean()*100:>6.2f}% alpha, {(n_bear['avg_alpha']>0).mean():>5.0%} hit rate")

    # By volatility regime
    print("\nPerformance by Volatility Regime:")
    print("-" * 80)

    vol_median = results_aligned['volatility_63d'].median()
    high_vol = results_aligned[results_aligned['volatility_63d'] > vol_median]
    low_vol = results_aligned[results_aligned['volatility_63d'] <= vol_median]

    if len(high_vol) > 0:
        hv_sharpe = high_vol['avg_alpha'].mean() / high_vol['avg_alpha'].std() if high_vol['avg_alpha'].std() > 0 else 0
        print(f"{'High Volatility':<25} {len(high_vol):>8} {high_vol['avg_alpha'].mean()*100:>11.2f}% {(high_vol['avg_alpha']>0).mean():>11.1%} {hv_sharpe:>10.2f}")

    if len(low_vol) > 0:
        lv_sharpe = low_vol['avg_alpha'].mean() / low_vol['avg_alpha'].std() if low_vol['avg_alpha'].std() > 0 else 0
        print(f"{'Low Volatility':<25} {len(low_vol):>8} {low_vol['avg_alpha'].mean()*100:>11.2f}% {(low_vol['avg_alpha']>0).mean():>11.1%} {lv_sharpe:>10.2f}")

    # By drawdown regime
    print("\nPerformance by Drawdown Regime:")
    print("-" * 80)

    no_dd = results_aligned[results_aligned['drawdown'] > -0.05]
    mild_dd = results_aligned[(results_aligned['drawdown'] <= -0.05) & (results_aligned['drawdown'] > -0.15)]
    severe_dd = results_aligned[results_aligned['drawdown'] <= -0.15]

    for name, subset in [('No drawdown (>-5%)', no_dd), ('Mild DD (-5% to -15%)', mild_dd), ('Severe DD (<-15%)', severe_dd)]:
        if len(subset) > 0:
            sharpe = subset['avg_alpha'].mean() / subset['avg_alpha'].std() if subset['avg_alpha'].std() > 0 else 0
            print(f"{name:<25} {len(subset):>8} {subset['avg_alpha'].mean()*100:>11.2f}% {(subset['avg_alpha']>0).mean():>11.1%} {sharpe:>10.2f}")

    # By market return regime
    print("\nPerformance by Prior 3-Month Market Return:")
    print("-" * 80)

    try:
        results_aligned['return_tercile'] = pd.qcut(results_aligned['return_3m'], 3, labels=['Down', 'Flat', 'Up'])

        for regime in ['Down', 'Flat', 'Up']:
            subset = results_aligned[results_aligned['return_tercile'] == regime]
            if len(subset) > 0:
                sharpe = subset['avg_alpha'].mean() / subset['avg_alpha'].std() if subset['avg_alpha'].std() > 0 else 0
                print(f"{'Market ' + regime:<25} {len(subset):>8} {subset['avg_alpha'].mean()*100:>11.2f}% {(subset['avg_alpha']>0).mean():>11.1%} {sharpe:>10.2f}")
    except Exception as e:
        print(f"  Could not compute terciles: {e}")


def analyze_rolling_performance(results: pd.DataFrame):
    """Analyze rolling performance metrics.

    Computes rolling metrics per N value for fair comparison.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 6: ROLLING PERFORMANCE (Per N)")
    print("=" * 80)

    n_values = sorted(results['n_satellites'].unique())

    # Compute rolling metrics for each N
    print("\nRolling 12-Month Performance by N:")
    print("-" * 90)
    print(f"{'N':<5} {'Min Alpha':>12} {'Avg Alpha':>12} {'Max Alpha':>12} {'Min Hit':>10} {'Avg Hit':>10}")
    print("-" * 90)

    for n in n_values:
        n_data = results[results['n_satellites'] == n].copy().sort_index()

        # Rolling metrics
        n_data['rolling_12m_alpha'] = n_data['avg_alpha'].rolling(12).mean() * 12
        n_data['rolling_12m_hit'] = n_data['avg_alpha'].rolling(12).apply(lambda x: (x > 0).mean())

        rolling_data = n_data.dropna(subset=['rolling_12m_alpha'])

        if len(rolling_data) > 0:
            print(f"{n:<5} {rolling_data['rolling_12m_alpha'].min()*100:>11.1f}% {rolling_data['rolling_12m_alpha'].mean()*100:>11.1f}% {rolling_data['rolling_12m_alpha'].max()*100:>11.1f}% {rolling_data['rolling_12m_hit'].min():>9.0%} {rolling_data['rolling_12m_hit'].mean():>9.0%}")

    # Find worst and best periods across all N (aggregated)
    print("\n" + "-" * 90)
    print("Worst Calendar Months (averaged across all N):")
    print("-" * 90)

    # Group by date and average across N
    date_avg = results.groupby(results.index).agg({
        'avg_alpha': 'mean',
        'n_satellites': 'count'
    })
    date_avg.columns = ['avg_alpha', 'n_count']

    worst_months = date_avg.nsmallest(10, 'avg_alpha')
    for date, row in worst_months.iterrows():
        print(f"  {date.strftime('%Y-%m')}: Avg Alpha = {row['avg_alpha']*100:>6.2f}% (across {int(row['n_count'])} N values)")

    print("\nBest Calendar Months (averaged across all N):")
    print("-" * 90)

    best_months = date_avg.nlargest(10, 'avg_alpha')
    for date, row in best_months.iterrows():
        print(f"  {date.strftime('%Y-%m')}: Avg Alpha = {row['avg_alpha']*100:>6.2f}% (across {int(row['n_count'])} N values)")

    # Correlation of performance across N values
    print("\n" + "-" * 90)
    print("Correlation of Monthly Alpha Across N Values:")
    print("-" * 90)
    print("(High correlation = all N values move together, strategy is consistent)")

    # Pivot to get N values as columns
    pivot_df = results.pivot_table(index=results.index, columns='n_satellites', values='avg_alpha')

    if len(pivot_df.columns) > 1:
        corr_matrix = pivot_df.corr()
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        print(f"\n  Average pairwise correlation: {avg_corr:.3f}")

        # Show correlation between adjacent N values
        print("\n  Adjacent N correlations:")
        for i in range(len(n_values) - 1):
            n1, n2 = n_values[i], n_values[i+1]
            if n1 in corr_matrix.columns and n2 in corr_matrix.columns:
                corr = corr_matrix.loc[n1, n2]
                print(f"    N={n1} vs N={n2}: {corr:.3f}")


def main():
    """Run comprehensive backtest analysis across all N values."""
    print("=" * 80)
    print("BACKTEST PERFORMANCE ANALYSIS - ALL N VALUES")
    print("=" * 80)
    print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data for all N values
    print("\nLoading data...")
    try:
        results = load_all_backtest_results()
        n_values = sorted(results['n_satellites'].unique())
        print(f"  Loaded {len(results)} total observations")
        print(f"  N values: {n_values}")
        print(f"  Period: {results.index.min().strftime('%Y-%m')} to {results.index.max().strftime('%Y-%m')}")
        print(f"  Unique months: {len(results.index.unique())}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    # Load market data
    try:
        market = load_market_data()
        if market is not None:
            print(f"  Loaded market data: {len(market)} months")
    except Exception as e:
        print(f"  Warning: Could not load market data: {e}")
        market = None

    # Summary stats per N
    print("\n" + "-" * 80)
    print("OVERALL SUMMARY BY N:")
    print("-" * 80)
    print(f"{'N':<5} {'Months':>8} {'Avg Alpha':>12} {'Annual':>10} {'Hit Rate':>10} {'Sharpe':>8}")
    print("-" * 60)

    for n in n_values:
        n_data = results[results['n_satellites'] == n]
        avg_alpha = n_data['avg_alpha'].mean()
        std_alpha = n_data['avg_alpha'].std()
        hit_rate = (n_data['avg_alpha'] > 0).mean()
        sharpe = avg_alpha / std_alpha if std_alpha > 0 else 0
        print(f"{n:<5} {len(n_data):>8} {avg_alpha*100:>11.2f}% {avg_alpha*12*100:>9.1f}% {hit_rate:>9.1%} {sharpe:>8.3f}")

    # Aggregate stats
    print("-" * 60)
    avg_alpha = results['avg_alpha'].mean()
    std_alpha = results['avg_alpha'].std()
    hit_rate = (results['avg_alpha'] > 0).mean()
    sharpe = avg_alpha / std_alpha if std_alpha > 0 else 0
    print(f"{'ALL':<5} {len(results):>8} {avg_alpha*100:>11.2f}% {avg_alpha*12*100:>9.1f}% {hit_rate:>9.1%} {sharpe:>8.3f}")

    # Run analyses
    analyze_hits_vs_misses(results, market)
    analyze_alpha_quartiles(results, market)
    analyze_time_patterns(results)
    analyze_etf_performance(results)

    if market is not None:
        analyze_regime_performance(results, market)

    analyze_rolling_performance(results)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
