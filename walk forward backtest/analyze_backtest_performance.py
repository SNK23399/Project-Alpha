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
7. Feature/Signal performance analysis
8. Feature stability and predictive power
9. Feature correlation analysis
10. Feature selection patterns (hits vs misses)
11. Miss analysis deep dive - WHY we miss
12. Prediction confidence analysis - unanimous vs spread rankings
13. Regime-conditional feature performance - bull vs bear features
14. Feature turnover analysis - how often features change

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

# ============================================================
# CONFIGURATION - Choose which backtest to analyze
# ============================================================
# Options: 'decay' (script 6) or 'filtered' (script 7 with MC filtering)
BACKTEST_SOURCE = 'decay'  # Change to 'decay' to analyze script 6 results


def load_backtest_results(n_satellites: int = 5, source: str = None) -> pd.DataFrame:
    """Load backtest results for a specific N.

    Args:
        n_satellites: Number of satellites (1-10)
        source: 'decay' for script 6, 'filtered' for script 7
    """
    if source is None:
        source = BACKTEST_SOURCE

    results_file = RESULTS_DIR / f'{source}_backtest_N{n_satellites}.csv'
    if not results_file.exists():
        return None

    df = pd.read_csv(results_file)
    df['date'] = pd.to_datetime(df['date'])
    df['n_satellites'] = n_satellites
    df = df.set_index('date')
    return df


def load_all_backtest_results(source: str = None) -> pd.DataFrame:
    """Load backtest results for all available N values.

    Args:
        source: 'decay' for script 6, 'filtered' for script 7
    """
    if source is None:
        source = BACKTEST_SOURCE

    all_results = []

    for n in range(1, 11):
        df = load_backtest_results(n, source=source)
        if df is not None:
            all_results.append(df)

    if not all_results:
        raise FileNotFoundError(
            f"No backtest results found for source='{source}'. "
            f"Run {'6_backtest_strategy.py' if source == 'decay' else '7_backtest_strategy_filtered.py'} first."
        )

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


def load_feature_data():
    """Load feature-level data for analysis."""
    # Load rankings matrix
    rankings_file = DATA_DIR / 'rankings_matrix_1month.npz'
    if not rankings_file.exists():
        return None

    npz_data = np.load(rankings_file, allow_pickle=True)
    rankings = npz_data['rankings']
    dates = pd.to_datetime(npz_data['dates'])
    isins = list(npz_data['isins'])
    feature_names = list(npz_data['features'])

    # Load feature-alpha matrix
    feature_alpha_file = DATA_DIR / 'feature_alpha_1month.npz'
    if not feature_alpha_file.exists():
        return None

    fa_data = np.load(feature_alpha_file, allow_pickle=True)
    feature_alpha = fa_data['feature_alpha']
    feature_hit = fa_data['feature_hit']

    # Load forward alpha
    alpha_file = DATA_DIR / 'forward_alpha_1month.parquet'
    if not alpha_file.exists():
        return None

    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])

    return {
        'rankings': rankings,
        'dates': dates,
        'isins': isins,
        'feature_names': feature_names,
        'feature_alpha': feature_alpha,
        'feature_hit': feature_hit,
        'alpha_df': alpha_df
    }


def analyze_feature_performance(feature_data):
    """
    ANALYSIS 7: Feature/Signal Performance Analysis

    Examines which features consistently predict alpha and which don't.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 7: FEATURE/SIGNAL PERFORMANCE")
    print("=" * 80)

    if feature_data is None:
        print("No feature data available")
        return

    feature_names = feature_data['feature_names']
    feature_alpha = feature_data['feature_alpha']
    feature_hit = feature_data['feature_hit']
    dates = feature_data['dates']

    n_dates, n_features, n_max_satellites = feature_alpha.shape

    print(f"\nTotal features: {n_features}")
    print(f"Date range: {dates[0].date()} to {dates[-1].date()} ({n_dates} months)")

    # Analyze feature performance for N=5 (our focus)
    n_sat = 5
    print(f"\n--- Feature Performance for N={n_sat} ---")

    feature_stats = []
    for feat_idx, feat_name in enumerate(feature_names):
        alphas = feature_alpha[:, feat_idx, n_sat - 1]
        hits = feature_hit[:, feat_idx, n_sat - 1]

        valid_mask = ~np.isnan(alphas)
        if valid_mask.sum() < 12:  # Need at least 12 months
            continue

        valid_alphas = alphas[valid_mask]
        valid_hits = hits[valid_mask]

        avg_alpha = np.mean(valid_alphas)
        std_alpha = np.std(valid_alphas)
        hit_rate = np.mean(valid_hits)
        sharpe = avg_alpha / std_alpha if std_alpha > 0 else 0

        # Recent performance (last 24 months)
        recent_mask = valid_mask.copy()
        recent_mask[:max(0, n_dates - 24)] = False
        if recent_mask.sum() >= 6:
            recent_alpha = np.mean(alphas[recent_mask])
            recent_hit = np.mean(hits[recent_mask])
        else:
            recent_alpha = np.nan
            recent_hit = np.nan

        # Early performance (first 24 months of available data)
        early_indices = np.where(valid_mask)[0][:24]
        if len(early_indices) >= 6:
            early_alpha = np.mean(alphas[early_indices])
            early_hit = np.mean(hits[early_indices])
        else:
            early_alpha = np.nan
            early_hit = np.nan

        feature_stats.append({
            'feature': feat_name,
            'avg_alpha': avg_alpha,
            'std_alpha': std_alpha,
            'hit_rate': hit_rate,
            'sharpe': sharpe,
            'n_valid': valid_mask.sum(),
            'recent_alpha': recent_alpha,
            'recent_hit': recent_hit,
            'early_alpha': early_alpha,
            'early_hit': early_hit,
            'alpha_decay': recent_alpha - early_alpha if not np.isnan(recent_alpha) and not np.isnan(early_alpha) else np.nan
        })

    feature_df = pd.DataFrame(feature_stats)

    # Top features by average alpha
    print(f"\nTop 20 Features by Average Alpha (N={n_sat}):")
    print("-" * 100)
    print(f"{'Feature':<35} {'Avg Alpha':>10} {'Hit Rate':>10} {'Sharpe':>8} {'Recent':>10} {'Decay':>10}")
    print("-" * 100)

    top_features = feature_df.nlargest(20, 'avg_alpha')
    for _, row in top_features.iterrows():
        decay_str = f"{row['alpha_decay']*100:>+9.2f}%" if not np.isnan(row['alpha_decay']) else "N/A"
        recent_str = f"{row['recent_alpha']*100:>9.2f}%" if not np.isnan(row['recent_alpha']) else "N/A"
        print(f"{row['feature']:<35} {row['avg_alpha']*100:>9.2f}% {row['hit_rate']:>9.1%} {row['sharpe']:>8.2f} {recent_str:>10} {decay_str:>10}")

    # Bottom features (negative alpha)
    print(f"\nBottom 20 Features by Average Alpha (N={n_sat}):")
    print("-" * 100)
    print(f"{'Feature':<35} {'Avg Alpha':>10} {'Hit Rate':>10} {'Sharpe':>8} {'Recent':>10} {'Decay':>10}")
    print("-" * 100)

    bottom_features = feature_df.nsmallest(20, 'avg_alpha')
    for _, row in bottom_features.iterrows():
        decay_str = f"{row['alpha_decay']*100:>+9.2f}%" if not np.isnan(row['alpha_decay']) else "N/A"
        recent_str = f"{row['recent_alpha']*100:>9.2f}%" if not np.isnan(row['recent_alpha']) else "N/A"
        print(f"{row['feature']:<35} {row['avg_alpha']*100:>9.2f}% {row['hit_rate']:>9.1%} {row['sharpe']:>8.2f} {recent_str:>10} {decay_str:>10}")

    # Features with best Sharpe ratio (most consistent)
    print(f"\nTop 20 Features by Sharpe Ratio (Consistency):")
    print("-" * 100)
    print(f"{'Feature':<35} {'Sharpe':>8} {'Avg Alpha':>10} {'Hit Rate':>10} {'Std Alpha':>10}")
    print("-" * 100)

    best_sharpe = feature_df.nlargest(20, 'sharpe')
    for _, row in best_sharpe.iterrows():
        print(f"{row['feature']:<35} {row['sharpe']:>8.2f} {row['avg_alpha']*100:>9.2f}% {row['hit_rate']:>9.1%} {row['std_alpha']*100:>9.2f}%")

    # Features showing decay (worked before, not now)
    print(f"\nFeatures with Largest Performance Decay (early vs recent):")
    print("-" * 100)
    print(f"{'Feature':<35} {'Early Alpha':>12} {'Recent Alpha':>12} {'Decay':>10} {'Early Hit':>10} {'Recent Hit':>10}")
    print("-" * 100)

    decaying = feature_df.dropna(subset=['alpha_decay']).nsmallest(15, 'alpha_decay')
    for _, row in decaying.iterrows():
        print(f"{row['feature']:<35} {row['early_alpha']*100:>11.2f}% {row['recent_alpha']*100:>11.2f}% {row['alpha_decay']*100:>+9.2f}% {row['early_hit']:>9.1%} {row['recent_hit']:>9.1%}")

    # Features improving over time
    print(f"\nFeatures with Largest Performance Improvement (recent better than early):")
    print("-" * 100)

    improving = feature_df.dropna(subset=['alpha_decay']).nlargest(15, 'alpha_decay')
    for _, row in improving.iterrows():
        print(f"{row['feature']:<35} {row['early_alpha']*100:>11.2f}% {row['recent_alpha']*100:>11.2f}% {row['alpha_decay']*100:>+9.2f}% {row['early_hit']:>9.1%} {row['recent_hit']:>9.1%}")

    # Feature category analysis
    print("\n" + "-" * 80)
    print("Feature Performance by Category:")
    print("-" * 80)

    # Parse feature categories from names
    categories = {}
    for _, row in feature_df.iterrows():
        feat = row['feature']
        # Extract category from feature name (e.g., "momentum_12m" -> "momentum")
        parts = feat.split('_')
        if len(parts) >= 1:
            cat = parts[0]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(row)

    cat_stats = []
    for cat, rows in categories.items():
        if len(rows) >= 2:
            avg_alpha = np.mean([r['avg_alpha'] for r in rows])
            avg_hit = np.mean([r['hit_rate'] for r in rows])
            avg_sharpe = np.mean([r['sharpe'] for r in rows])
            cat_stats.append({
                'category': cat,
                'n_features': len(rows),
                'avg_alpha': avg_alpha,
                'avg_hit_rate': avg_hit,
                'avg_sharpe': avg_sharpe
            })

    cat_df = pd.DataFrame(cat_stats).sort_values('avg_alpha', ascending=False)

    print(f"\n{'Category':<20} {'# Features':>12} {'Avg Alpha':>12} {'Avg Hit Rate':>12} {'Avg Sharpe':>10}")
    print("-" * 70)
    for _, row in cat_df.iterrows():
        print(f"{row['category']:<20} {row['n_features']:>12} {row['avg_alpha']*100:>11.2f}% {row['avg_hit_rate']:>11.1%} {row['avg_sharpe']:>10.2f}")

    return feature_df


def analyze_feature_stability(feature_data):
    """
    ANALYSIS 8: Feature Stability Over Time

    Examines whether features maintain their predictive power over time.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 8: FEATURE STABILITY OVER TIME")
    print("=" * 80)

    if feature_data is None:
        print("No feature data available")
        return

    feature_names = feature_data['feature_names']
    feature_alpha = feature_data['feature_alpha']
    feature_hit = feature_data['feature_hit']
    dates = feature_data['dates']

    n_dates, n_features, n_max_satellites = feature_alpha.shape
    n_sat = 5  # Focus on N=5

    # Split data into time periods
    n_periods = 4
    period_size = n_dates // n_periods

    print(f"\nAnalyzing feature stability across {n_periods} time periods")
    print(f"Period size: ~{period_size} months each")

    # Calculate feature performance per period
    period_results = []

    for period_idx in range(n_periods):
        start_idx = period_idx * period_size
        end_idx = start_idx + period_size if period_idx < n_periods - 1 else n_dates

        period_start = dates[start_idx]
        period_end = dates[min(end_idx - 1, n_dates - 1)]

        for feat_idx, feat_name in enumerate(feature_names):
            alphas = feature_alpha[start_idx:end_idx, feat_idx, n_sat - 1]
            hits = feature_hit[start_idx:end_idx, feat_idx, n_sat - 1]

            valid_mask = ~np.isnan(alphas)
            if valid_mask.sum() < 6:
                continue

            avg_alpha = np.mean(alphas[valid_mask])
            hit_rate = np.mean(hits[valid_mask])

            period_results.append({
                'feature': feat_name,
                'period': period_idx + 1,
                'period_start': period_start,
                'period_end': period_end,
                'avg_alpha': avg_alpha,
                'hit_rate': hit_rate
            })

    period_df = pd.DataFrame(period_results)

    # Calculate feature stability (std of alpha across periods)
    stability_stats = []
    for feat_name in feature_names:
        feat_periods = period_df[period_df['feature'] == feat_name]
        if len(feat_periods) >= 3:
            alphas = feat_periods['avg_alpha'].values
            hits = feat_periods['hit_rate'].values

            stability_stats.append({
                'feature': feat_name,
                'mean_alpha': np.mean(alphas),
                'std_alpha': np.std(alphas),
                'min_alpha': np.min(alphas),
                'max_alpha': np.max(alphas),
                'alpha_range': np.max(alphas) - np.min(alphas),
                'mean_hit': np.mean(hits),
                'n_positive_periods': (alphas > 0).sum(),
                'consistency': (alphas > 0).sum() / len(alphas)  # % of periods with positive alpha
            })

    stability_df = pd.DataFrame(stability_stats)

    # Most stable features (low std, high consistency)
    print(f"\nMost STABLE Features (consistent across time periods):")
    print("-" * 100)
    print(f"{'Feature':<35} {'Mean Alpha':>10} {'Std Alpha':>10} {'Range':>10} {'Consistency':>12} {'Mean Hit':>10}")
    print("-" * 100)

    # Filter to positive alpha features, sort by consistency then std
    stable = stability_df[stability_df['mean_alpha'] > 0].sort_values(
        ['consistency', 'std_alpha'], ascending=[False, True]
    ).head(20)

    for _, row in stable.iterrows():
        print(f"{row['feature']:<35} {row['mean_alpha']*100:>9.2f}% {row['std_alpha']*100:>9.2f}% {row['alpha_range']*100:>9.2f}% {row['consistency']:>11.0%} {row['mean_hit']:>9.1%}")

    # Most unstable features (high variance)
    print(f"\nMost UNSTABLE Features (high variance across time periods):")
    print("-" * 100)

    unstable = stability_df.nlargest(20, 'std_alpha')
    for _, row in unstable.iterrows():
        print(f"{row['feature']:<35} {row['mean_alpha']*100:>9.2f}% {row['std_alpha']*100:>9.2f}% {row['alpha_range']*100:>9.2f}% {row['consistency']:>11.0%} {row['mean_hit']:>9.1%}")

    # Features that are positive in ALL periods
    print(f"\nFeatures POSITIVE in ALL {n_periods} Periods (most reliable):")
    print("-" * 80)

    always_positive = stability_df[stability_df['n_positive_periods'] == n_periods].sort_values('mean_alpha', ascending=False)
    print(f"Found {len(always_positive)} features positive in all periods")

    for _, row in always_positive.head(15).iterrows():
        print(f"  {row['feature']:<35} Mean: {row['mean_alpha']*100:>6.2f}%, Min: {row['min_alpha']*100:>6.2f}%, Max: {row['max_alpha']*100:>6.2f}%")

    # Period-by-period breakdown for top features
    print("\n" + "-" * 80)
    print("Period-by-Period Performance for Top 10 Stable Features:")
    print("-" * 80)

    top_stable_features = stable.head(10)['feature'].tolist()

    # Header
    print(f"\n{'Feature':<30}", end="")
    for p in range(1, n_periods + 1):
        print(f"{'P' + str(p):>10}", end="")
    print(f"{'Std':>10}")
    print("-" * (30 + 10 * n_periods + 10))

    for feat in top_stable_features:
        feat_data = period_df[period_df['feature'] == feat].sort_values('period')
        feat_stab = stability_df[stability_df['feature'] == feat].iloc[0]

        print(f"{feat:<30}", end="")
        for _, row in feat_data.iterrows():
            print(f"{row['avg_alpha']*100:>9.2f}%", end="")
        print(f"{feat_stab['std_alpha']*100:>9.2f}%")

    return stability_df


def analyze_feature_correlation(feature_data):
    """
    ANALYSIS 9: Feature Correlation Analysis

    Identifies redundant features and potential diversification opportunities.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 9: FEATURE CORRELATION ANALYSIS")
    print("=" * 80)

    if feature_data is None:
        print("No feature data available")
        return

    feature_names = feature_data['feature_names']
    feature_alpha = feature_data['feature_alpha']
    dates = feature_data['dates']

    n_dates, n_features, n_max_satellites = feature_alpha.shape
    n_sat = 5

    # Build feature alpha time series matrix
    alpha_series = {}
    for feat_idx, feat_name in enumerate(feature_names):
        alphas = feature_alpha[:, feat_idx, n_sat - 1]
        if np.sum(~np.isnan(alphas)) >= 24:  # Need sufficient data
            alpha_series[feat_name] = alphas

    if len(alpha_series) < 10:
        print("Not enough features with sufficient data")
        return

    # Create DataFrame and compute correlation
    alpha_df = pd.DataFrame(alpha_series, index=dates)

    # Only use features with positive average alpha
    positive_features = [f for f in alpha_df.columns if alpha_df[f].mean() > 0]
    print(f"\nAnalyzing {len(positive_features)} features with positive average alpha")

    if len(positive_features) < 5:
        print("Not enough positive features for correlation analysis")
        return

    corr_matrix = alpha_df[positive_features].corr()

    # Find highly correlated feature pairs
    print("\nHighly Correlated Feature Pairs (r > 0.7):")
    print("-" * 80)
    print("(These features are redundant - picking both adds little diversification)")

    high_corr_pairs = []
    for i, f1 in enumerate(positive_features):
        for j, f2 in enumerate(positive_features):
            if i < j:
                corr = corr_matrix.loc[f1, f2]
                if corr > 0.7:
                    high_corr_pairs.append((f1, f2, corr))

    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

    for f1, f2, corr in high_corr_pairs[:20]:
        print(f"  r={corr:.3f}: {f1} <-> {f2}")

    # Find uncorrelated features (good for diversification)
    print("\nLow Correlation Feature Pairs (|r| < 0.2):")
    print("-" * 80)
    print("(These features provide diversification - combining them may reduce variance)")

    low_corr_pairs = []
    for i, f1 in enumerate(positive_features):
        for j, f2 in enumerate(positive_features):
            if i < j:
                corr = corr_matrix.loc[f1, f2]
                if abs(corr) < 0.2:
                    # Get average alpha for both
                    alpha1 = alpha_df[f1].mean()
                    alpha2 = alpha_df[f2].mean()
                    low_corr_pairs.append((f1, f2, corr, alpha1, alpha2))

    low_corr_pairs.sort(key=lambda x: (x[3] + x[4]) / 2, reverse=True)

    for f1, f2, corr, a1, a2 in low_corr_pairs[:20]:
        print(f"  r={corr:+.3f}: {f1} ({a1*100:.2f}%) + {f2} ({a2*100:.2f}%)")

    # Feature clusters (groups of highly correlated features)
    print("\n" + "-" * 80)
    print("Feature Clusters (groups of correlated features):")
    print("-" * 80)

    # Simple clustering: find features with avg correlation > 0.5 to each other
    clusters = []
    assigned = set()

    for f1 in positive_features:
        if f1 in assigned:
            continue

        cluster = [f1]
        assigned.add(f1)

        for f2 in positive_features:
            if f2 not in assigned:
                avg_corr = np.mean([corr_matrix.loc[f2, c] for c in cluster])
                if avg_corr > 0.5:
                    cluster.append(f2)
                    assigned.add(f2)

        if len(cluster) >= 2:
            avg_alpha = np.mean([alpha_df[f].mean() for f in cluster])
            clusters.append((cluster, avg_alpha))

    clusters.sort(key=lambda x: x[1], reverse=True)

    for i, (cluster, avg_alpha) in enumerate(clusters[:10], 1):
        print(f"\nCluster {i} (avg alpha: {avg_alpha*100:.2f}%):")
        for f in cluster:
            f_alpha = alpha_df[f].mean()
            print(f"  - {f} ({f_alpha*100:.2f}%)")

    return corr_matrix


def analyze_selection_patterns(feature_data, results):
    """
    ANALYSIS 10: What Features Get Selected in Hits vs Misses

    Looks at which features the greedy search tends to select
    and whether there are patterns in hit vs miss months.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 10: FEATURE SELECTION PATTERNS (Hits vs Misses)")
    print("=" * 80)

    # This would require saving which features were selected each month
    # For now, we can analyze based on feature performance in hit vs miss months

    if feature_data is None:
        print("No feature data available")
        return

    feature_names = feature_data['feature_names']
    feature_alpha = feature_data['feature_alpha']
    dates = feature_data['dates']

    # Get hit and miss dates from results
    hits = results[results['avg_alpha'] > 0]
    misses = results[results['avg_alpha'] <= 0]

    hit_dates = hits.index.unique()
    miss_dates = misses.index.unique()

    # Map to date indices
    date_to_idx = {d: i for i, d in enumerate(dates)}

    hit_indices = [date_to_idx[d] for d in hit_dates if d in date_to_idx]
    miss_indices = [date_to_idx[d] for d in miss_dates if d in date_to_idx]

    n_sat = 5

    print(f"\nComparing feature performance in {len(hit_indices)} hit months vs {len(miss_indices)} miss months")

    feature_comparison = []
    for feat_idx, feat_name in enumerate(feature_names):
        hit_alphas = feature_alpha[hit_indices, feat_idx, n_sat - 1]
        miss_alphas = feature_alpha[miss_indices, feat_idx, n_sat - 1]

        hit_valid = hit_alphas[~np.isnan(hit_alphas)]
        miss_valid = miss_alphas[~np.isnan(miss_alphas)]

        if len(hit_valid) < 10 or len(miss_valid) < 10:
            continue

        hit_avg = np.mean(hit_valid)
        miss_avg = np.mean(miss_valid)
        diff = hit_avg - miss_avg

        feature_comparison.append({
            'feature': feat_name,
            'hit_alpha': hit_avg,
            'miss_alpha': miss_avg,
            'diff': diff,
            'hit_better': diff > 0
        })

    comp_df = pd.DataFrame(feature_comparison)

    # Features that perform much better in hit months
    print("\nFeatures with MUCH BETTER Alpha in HIT Months:")
    print("-" * 90)
    print(f"{'Feature':<35} {'Hit Alpha':>12} {'Miss Alpha':>12} {'Difference':>12}")
    print("-" * 90)

    better_in_hits = comp_df.nlargest(15, 'diff')
    for _, row in better_in_hits.iterrows():
        print(f"{row['feature']:<35} {row['hit_alpha']*100:>11.2f}% {row['miss_alpha']*100:>11.2f}% {row['diff']*100:>+11.2f}%")

    # Features that perform better in miss months (contrarian?)
    print("\nFeatures with BETTER Alpha in MISS Months (contrarian indicators?):")
    print("-" * 90)

    better_in_misses = comp_df.nsmallest(15, 'diff')
    for _, row in better_in_misses.iterrows():
        print(f"{row['feature']:<35} {row['hit_alpha']*100:>11.2f}% {row['miss_alpha']*100:>11.2f}% {row['diff']*100:>+11.2f}%")

    # Summary
    n_hit_better = (comp_df['diff'] > 0).sum()
    n_total = len(comp_df)
    print(f"\nSummary: {n_hit_better}/{n_total} ({n_hit_better/n_total:.1%}) features have higher alpha in hit months")

    return comp_df


def analyze_miss_deep_dive(results: pd.DataFrame, feature_data, market: pd.DataFrame = None):
    """
    ANALYSIS 11: Miss Analysis Deep Dive

    Deep dive into WHY we miss - is it wrong ETFs, wrong features, or bad timing?
    Categorizes misses and looks for actionable patterns.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 11: MISS ANALYSIS DEEP DIVE")
    print("=" * 80)

    # Focus on N=5 for detailed analysis
    n5_results = results[results['n_satellites'] == 5].copy()
    misses = n5_results[n5_results['avg_alpha'] <= 0].copy()

    if len(misses) == 0:
        print("No misses found!")
        return

    print(f"\nAnalyzing {len(misses)} miss months for N=5")

    # Load forward alpha for detailed ETF analysis
    alpha_file = DATA_DIR / 'forward_alpha_1month.parquet'
    if alpha_file.exists():
        alpha_df = pd.read_parquet(alpha_file)
        alpha_df['date'] = pd.to_datetime(alpha_df['date'])
    else:
        alpha_df = None

    # Categorize misses by severity
    misses['severity'] = pd.cut(
        misses['avg_alpha'],
        bins=[-np.inf, -0.03, -0.01, 0],
        labels=['Severe (<-3%)', 'Moderate (-1% to -3%)', 'Minor (0 to -1%)']
    )

    print("\nMiss Severity Distribution:")
    print("-" * 50)
    severity_counts = misses['severity'].value_counts()
    for sev, count in severity_counts.items():
        pct = count / len(misses) * 100
        avg_alpha = misses[misses['severity'] == sev]['avg_alpha'].mean()
        print(f"  {sev}: {count} months ({pct:.1f}%), avg alpha: {avg_alpha*100:.2f}%")

    # Worst misses detailed analysis
    print("\n" + "-" * 80)
    print("TOP 15 WORST MISSES - Detailed Breakdown:")
    print("-" * 80)

    worst_misses = misses.nsmallest(15, 'avg_alpha')

    for date, row in worst_misses.iterrows():
        print(f"\n{date.strftime('%Y-%m')}: Alpha = {row['avg_alpha']*100:.2f}%")

        # Get selected ETFs for this month
        selected_str = row.get('selected_isins', '')
        if pd.notna(selected_str) and selected_str:
            selected = selected_str.split(',')
            print(f"  Selected ETFs: {len(selected)}")

            # Get individual ETF performance if available
            if alpha_df is not None:
                date_alpha = alpha_df[alpha_df['date'] == date]
                if len(date_alpha) > 0:
                    # Check individual ETF performance
                    etf_details = []
                    for isin in selected:
                        etf_alpha_row = date_alpha[date_alpha['isin'] == isin]
                        if len(etf_alpha_row) > 0:
                            etf_alpha = etf_alpha_row['forward_alpha'].values[0]
                            etf_details.append((isin, etf_alpha))

                    etf_details.sort(key=lambda x: x[1])
                    n_positive = sum(1 for _, a in etf_details if a > 0)
                    print(f"  ETF breakdown ({n_positive}/{len(etf_details)} positive):")
                    for isin, alpha in etf_details:
                        sign = "+" if alpha > 0 else ""
                        print(f"    {isin}: {sign}{alpha*100:.2f}%")

                    # What would best ETFs have been?
                    top_5 = date_alpha.nlargest(5, 'forward_alpha')
                    print(f"  Best available ETFs that month:")
                    for _, best_row in top_5.iterrows():
                        print(f"    {best_row['isin']}: +{best_row['forward_alpha']*100:.2f}%")

        # Market context if available
        if market is not None and date in market.index:
            mkt = market.loc[date]
            trend = "Bull" if mkt.get('above_sma_200', 0) == 1 else "Bear"
            ret_1m = mkt.get('return_1m', np.nan)
            vol = mkt.get('volatility_63d', np.nan)
            dd = mkt.get('drawdown', np.nan)
            print(f"  Market: {trend}, 1m return: {ret_1m*100:.1f}%, vol: {vol*100:.0f}%, drawdown: {dd*100:.1f}%")

    # Miss patterns analysis
    print("\n" + "-" * 80)
    print("MISS PATTERN ANALYSIS:")
    print("-" * 80)

    if alpha_df is not None:
        # Analyze why we picked wrong ETFs
        miss_reasons = {
            'all_negative': 0,       # All our picks were negative
            'some_negative': 0,      # Some picks were negative
            'positive_but_weak': 0,  # Picks were positive but below avg
            'unlucky': 0             # Picks were good, just bad luck
        }

        for date, row in misses.iterrows():
            selected_str = row.get('selected_isins', '')
            if not pd.notna(selected_str) or not selected_str:
                continue

            selected = selected_str.split(',')
            date_alpha = alpha_df[alpha_df['date'] == date]

            if len(date_alpha) == 0:
                continue

            # Get selected ETF alphas
            selected_alphas = []
            for isin in selected:
                etf_row = date_alpha[date_alpha['isin'] == isin]
                if len(etf_row) > 0:
                    selected_alphas.append(etf_row['forward_alpha'].values[0])

            if not selected_alphas:
                continue

            n_positive = sum(1 for a in selected_alphas if a > 0)
            avg_selected = np.mean(selected_alphas)
            avg_all = date_alpha['forward_alpha'].mean()

            if n_positive == 0:
                miss_reasons['all_negative'] += 1
            elif n_positive < len(selected_alphas):
                miss_reasons['some_negative'] += 1
            elif avg_selected < avg_all:
                miss_reasons['positive_but_weak'] += 1
            else:
                miss_reasons['unlucky'] += 1

        total = sum(miss_reasons.values())
        print("\nWhy did we miss?")
        for reason, count in miss_reasons.items():
            pct = count / total * 100 if total > 0 else 0
            desc = {
                'all_negative': 'All selected ETFs had negative alpha',
                'some_negative': 'Some selected ETFs had negative alpha',
                'positive_but_weak': 'Picks were positive but below market avg',
                'unlucky': 'Good picks but core outperformed'
            }
            print(f"  {desc[reason]}: {count} ({pct:.1f}%)")

    # Consecutive miss patterns
    print("\n" + "-" * 80)
    print("Consecutive Miss Streaks:")
    print("-" * 80)

    n5_results_sorted = n5_results.sort_index()
    n5_results_sorted['is_miss'] = n5_results_sorted['avg_alpha'] <= 0

    # Find streaks
    miss_streaks = []
    current_streak_start = None
    current_streak_len = 0

    for date, row in n5_results_sorted.iterrows():
        if row['is_miss']:
            if current_streak_start is None:
                current_streak_start = date
            current_streak_len += 1
        else:
            if current_streak_len >= 2:
                streak_data = n5_results_sorted.loc[current_streak_start:date]
                streak_data = streak_data[streak_data['is_miss']]
                miss_streaks.append({
                    'start': current_streak_start,
                    'length': current_streak_len,
                    'total_alpha': streak_data['avg_alpha'].sum(),
                    'avg_alpha': streak_data['avg_alpha'].mean()
                })
            current_streak_start = None
            current_streak_len = 0

    # Handle final streak
    if current_streak_len >= 2:
        miss_streaks.append({
            'start': current_streak_start,
            'length': current_streak_len,
            'total_alpha': n5_results_sorted.loc[current_streak_start:]['avg_alpha'].sum(),
            'avg_alpha': n5_results_sorted.loc[current_streak_start:]['avg_alpha'].mean()
        })

    miss_streaks.sort(key=lambda x: x['length'], reverse=True)

    print(f"\nLongest consecutive miss streaks:")
    for streak in miss_streaks[:10]:
        print(f"  {streak['start'].strftime('%Y-%m')}: {streak['length']} months, total alpha: {streak['total_alpha']*100:.2f}%, avg: {streak['avg_alpha']*100:.2f}%")

    return misses


def analyze_prediction_confidence(feature_data, results):
    """
    ANALYSIS 12: Prediction Confidence Analysis

    Examines whether we perform better when our rankings are unanimous (high confidence)
    vs when they are spread out (low confidence).
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 12: PREDICTION CONFIDENCE ANALYSIS")
    print("=" * 80)

    if feature_data is None:
        print("No feature data available")
        return

    rankings = feature_data['rankings']
    dates = feature_data['dates']
    isins = feature_data['isins']
    feature_names = feature_data['feature_names']

    n_dates, n_isins, n_features = rankings.shape

    # Focus on N=5
    n_sat = 5
    n5_results = results[results['n_satellites'] == 5].copy()

    print(f"\nRankings shape: {n_dates} dates x {n_isins} ETFs x {n_features} features")

    confidence_data = []

    for date_idx, date in enumerate(dates):
        if date not in n5_results.index:
            continue

        month_result = n5_results.loc[date]
        if isinstance(month_result, pd.DataFrame):
            month_result = month_result.iloc[0]

        alpha = month_result['avg_alpha']

        # Get rankings for this date
        date_rankings = rankings[date_idx, :, :]  # n_isins x n_features

        # For each ETF, calculate how consistently it's ranked in top N
        top_n_counts = np.zeros(n_isins)
        valid_features = 0

        for feat_idx in range(n_features):
            feat_rankings = date_rankings[:, feat_idx]
            if np.all(np.isnan(feat_rankings)):
                continue
            valid_features += 1

            # Count ETFs ranked in top N
            top_n_mask = feat_rankings <= n_sat
            top_n_counts += top_n_mask.astype(float)

        if valid_features == 0:
            continue

        # Normalize by number of features
        top_n_freq = top_n_counts / valid_features

        # Confidence metrics
        # 1. Max agreement: highest frequency any ETF appears in top N
        max_agreement = np.max(top_n_freq)

        # 2. Top ETF spread: std of top N frequencies among top-ranked ETFs
        top_etfs = np.argsort(top_n_freq)[-n_sat:]
        top_etf_freq = top_n_freq[top_etfs]
        top_etf_spread = np.std(top_etf_freq)

        # 3. Unanimity score: how many ETFs have >70% agreement
        high_agreement_count = np.sum(top_n_freq > 0.7)

        # 4. Entropy of ranking distribution (lower = more confident)
        freq_normalized = top_n_freq / np.sum(top_n_freq) if np.sum(top_n_freq) > 0 else top_n_freq
        freq_normalized = freq_normalized[freq_normalized > 0]
        entropy = -np.sum(freq_normalized * np.log(freq_normalized + 1e-10))

        confidence_data.append({
            'date': date,
            'alpha': alpha,
            'max_agreement': max_agreement,
            'top_etf_spread': top_etf_spread,
            'high_agreement_count': high_agreement_count,
            'entropy': entropy,
            'valid_features': valid_features,
            'is_hit': alpha > 0
        })

    conf_df = pd.DataFrame(confidence_data)

    if len(conf_df) == 0:
        print("No confidence data available")
        return

    print(f"\nAnalyzed {len(conf_df)} months with confidence metrics")

    # Correlation between confidence and performance
    print("\nCorrelation: Confidence Metrics vs Alpha:")
    print("-" * 60)
    for metric in ['max_agreement', 'top_etf_spread', 'high_agreement_count', 'entropy']:
        corr = conf_df[metric].corr(conf_df['alpha'])
        print(f"  {metric:<25}: r = {corr:+.3f}")

    # Split by confidence levels
    print("\n" + "-" * 80)
    print("Performance by Confidence Terciles:")
    print("-" * 80)

    # By max agreement
    conf_df['agreement_tercile'] = pd.qcut(conf_df['max_agreement'], 3, labels=['Low', 'Medium', 'High'])

    print(f"\n{'Max Agreement':<15} {'Count':>8} {'Avg Alpha':>12} {'Hit Rate':>10} {'Avg Entropy':>12}")
    print("-" * 60)

    for tercile in ['Low', 'Medium', 'High']:
        subset = conf_df[conf_df['agreement_tercile'] == tercile]
        print(f"{tercile:<15} {len(subset):>8} {subset['alpha'].mean()*100:>11.2f}% {subset['is_hit'].mean():>9.1%} {subset['entropy'].mean():>12.2f}")

    # By entropy (inverse confidence)
    conf_df['entropy_tercile'] = pd.qcut(conf_df['entropy'], 3, labels=['Low (Confident)', 'Medium', 'High (Uncertain)'])

    print(f"\n{'Entropy Level':<20} {'Count':>8} {'Avg Alpha':>12} {'Hit Rate':>10}")
    print("-" * 60)

    for tercile in ['Low (Confident)', 'Medium', 'High (Uncertain)']:
        subset = conf_df[conf_df['entropy_tercile'] == tercile]
        print(f"{tercile:<20} {len(subset):>8} {subset['alpha'].mean()*100:>11.2f}% {subset['is_hit'].mean():>9.1%}")

    # Identify high-confidence misses and low-confidence hits
    print("\n" + "-" * 80)
    print("Anomalies Analysis:")
    print("-" * 80)

    # High confidence misses (we were sure but wrong)
    high_conf_misses = conf_df[(conf_df['max_agreement'] > conf_df['max_agreement'].quantile(0.75)) &
                               (~conf_df['is_hit'])]
    print(f"\nHigh Confidence MISSES (top 25% agreement but missed): {len(high_conf_misses)}")
    if len(high_conf_misses) > 0:
        print(f"  Average alpha: {high_conf_misses['alpha'].mean()*100:.2f}%")
        print(f"  Worst: {high_conf_misses['alpha'].min()*100:.2f}%")
        print("  Dates:", [d.strftime('%Y-%m') for d in high_conf_misses.nsmallest(5, 'alpha')['date']])

    # Low confidence hits (uncertain but right)
    low_conf_hits = conf_df[(conf_df['max_agreement'] < conf_df['max_agreement'].quantile(0.25)) &
                            (conf_df['is_hit'])]
    print(f"\nLow Confidence HITS (bottom 25% agreement but hit): {len(low_conf_hits)}")
    if len(low_conf_hits) > 0:
        print(f"  Average alpha: {low_conf_hits['alpha'].mean()*100:.2f}%")
        print(f"  Best: {low_conf_hits['alpha'].max()*100:.2f}%")

    # Summary insight
    print("\n" + "-" * 80)
    print("KEY INSIGHT:")
    print("-" * 80)

    high_conf = conf_df[conf_df['max_agreement'] > conf_df['max_agreement'].median()]
    low_conf = conf_df[conf_df['max_agreement'] <= conf_df['max_agreement'].median()]

    high_conf_perf = high_conf['alpha'].mean()
    low_conf_perf = low_conf['alpha'].mean()

    if high_conf_perf > low_conf_perf:
        print(f"Higher confidence predictions perform BETTER:")
        print(f"  High confidence avg alpha: {high_conf_perf*100:.2f}%")
        print(f"  Low confidence avg alpha:  {low_conf_perf*100:.2f}%")
        print(f"  Difference: {(high_conf_perf - low_conf_perf)*100:+.2f}%")
        print("\n  ACTIONABLE: Consider scaling position size with confidence!")
    else:
        print(f"Higher confidence predictions perform WORSE:")
        print(f"  High confidence avg alpha: {high_conf_perf*100:.2f}%")
        print(f"  Low confidence avg alpha:  {low_conf_perf*100:.2f}%")
        print(f"  Difference: {(high_conf_perf - low_conf_perf)*100:+.2f}%")
        print("\n  WARNING: Overconfidence may be hurting performance!")

    return conf_df


def analyze_regime_conditional_features(feature_data, market: pd.DataFrame):
    """
    ANALYSIS 13: Regime-Conditional Feature Performance

    Analyzes which features work best in bull vs bear markets,
    high vs low volatility, etc.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 13: REGIME-CONDITIONAL FEATURE PERFORMANCE")
    print("=" * 80)

    if feature_data is None:
        print("No feature data available")
        return

    if market is None:
        print("No market data available")
        return

    feature_names = feature_data['feature_names']
    feature_alpha = feature_data['feature_alpha']
    dates = feature_data['dates']

    n_dates, n_features, n_max_satellites = feature_alpha.shape
    n_sat = 5

    # Map dates to market data
    date_to_idx = {d: i for i, d in enumerate(dates)}

    # Classify each date by regime
    regime_classifications = []
    for date in dates:
        if date not in market.index:
            regime_classifications.append(None)
            continue

        mkt = market.loc[date]

        regime = {
            'date': date,
            'trend': 'Bull' if mkt.get('above_sma_200', 0) == 1 else 'Bear',
            'volatility': 'High' if mkt.get('volatility_63d', 0) > market['volatility_63d'].median() else 'Low',
            'drawdown': 'Severe' if mkt.get('drawdown', 0) < -0.10 else 'Normal'
        }

        # Recent return tercile
        ret_3m = mkt.get('return_3m', 0)
        if ret_3m > market['return_3m'].quantile(0.66):
            regime['momentum'] = 'Up'
        elif ret_3m < market['return_3m'].quantile(0.33):
            regime['momentum'] = 'Down'
        else:
            regime['momentum'] = 'Flat'

        regime_classifications.append(regime)

    # Analyze feature performance by regime
    print("\n1. TREND REGIME (Bull vs Bear Market):")
    print("-" * 100)

    bull_indices = [i for i, r in enumerate(regime_classifications) if r and r['trend'] == 'Bull']
    bear_indices = [i for i, r in enumerate(regime_classifications) if r and r['trend'] == 'Bear']

    print(f"  Bull periods: {len(bull_indices)} months, Bear periods: {len(bear_indices)} months")

    trend_comparison = []
    for feat_idx, feat_name in enumerate(feature_names):
        bull_alphas = feature_alpha[bull_indices, feat_idx, n_sat - 1]
        bear_alphas = feature_alpha[bear_indices, feat_idx, n_sat - 1]

        bull_valid = bull_alphas[~np.isnan(bull_alphas)]
        bear_valid = bear_alphas[~np.isnan(bear_alphas)]

        if len(bull_valid) < 10 or len(bear_valid) < 10:
            continue

        bull_avg = np.mean(bull_valid)
        bear_avg = np.mean(bear_valid)

        trend_comparison.append({
            'feature': feat_name,
            'bull_alpha': bull_avg,
            'bear_alpha': bear_avg,
            'bull_hit': np.mean(bull_valid > 0),
            'bear_hit': np.mean(bear_valid > 0),
            'diff': bull_avg - bear_avg
        })

    trend_df = pd.DataFrame(trend_comparison)

    # Features that work MUCH better in bull markets
    print(f"\nFeatures that work MUCH BETTER in BULL markets:")
    print(f"{'Feature':<35} {'Bull Alpha':>10} {'Bear Alpha':>10} {'Difference':>10}")
    print("-" * 70)

    bull_lovers = trend_df.nlargest(10, 'diff')
    for _, row in bull_lovers.iterrows():
        print(f"{row['feature']:<35} {row['bull_alpha']*100:>9.2f}% {row['bear_alpha']*100:>9.2f}% {row['diff']*100:>+9.2f}%")

    # Features that work MUCH better in bear markets
    print(f"\nFeatures that work MUCH BETTER in BEAR markets:")
    print(f"{'Feature':<35} {'Bear Alpha':>10} {'Bull Alpha':>10} {'Difference':>10}")
    print("-" * 70)

    bear_lovers = trend_df.nsmallest(10, 'diff')
    for _, row in bear_lovers.iterrows():
        print(f"{row['feature']:<35} {row['bear_alpha']*100:>9.2f}% {row['bull_alpha']*100:>9.2f}% {-row['diff']*100:>+9.2f}%")

    # Features that work in BOTH regimes (regime-agnostic)
    print(f"\nFeatures that work in BOTH regimes (robust):")
    trend_df['min_alpha'] = trend_df[['bull_alpha', 'bear_alpha']].min(axis=1)
    robust = trend_df[trend_df['min_alpha'] > 0].sort_values('min_alpha', ascending=False).head(10)

    for _, row in robust.iterrows():
        print(f"  {row['feature']:<35} Bull: {row['bull_alpha']*100:>5.2f}%, Bear: {row['bear_alpha']*100:>5.2f}%")

    # 2. VOLATILITY REGIME
    print("\n" + "-" * 80)
    print("2. VOLATILITY REGIME (High vs Low Vol):")
    print("-" * 80)

    high_vol_indices = [i for i, r in enumerate(regime_classifications) if r and r['volatility'] == 'High']
    low_vol_indices = [i for i, r in enumerate(regime_classifications) if r and r['volatility'] == 'Low']

    print(f"  High vol periods: {len(high_vol_indices)} months, Low vol periods: {len(low_vol_indices)} months")

    vol_comparison = []
    for feat_idx, feat_name in enumerate(feature_names):
        high_alphas = feature_alpha[high_vol_indices, feat_idx, n_sat - 1]
        low_alphas = feature_alpha[low_vol_indices, feat_idx, n_sat - 1]

        high_valid = high_alphas[~np.isnan(high_alphas)]
        low_valid = low_alphas[~np.isnan(low_alphas)]

        if len(high_valid) < 10 or len(low_valid) < 10:
            continue

        vol_comparison.append({
            'feature': feat_name,
            'high_vol_alpha': np.mean(high_valid),
            'low_vol_alpha': np.mean(low_valid),
            'diff': np.mean(high_valid) - np.mean(low_valid)
        })

    vol_df = pd.DataFrame(vol_comparison)

    print(f"\nFeatures that work MUCH BETTER in HIGH VOLATILITY:")
    high_vol_lovers = vol_df.nlargest(10, 'diff')
    for _, row in high_vol_lovers.iterrows():
        print(f"  {row['feature']:<35} High Vol: {row['high_vol_alpha']*100:>5.2f}%, Low Vol: {row['low_vol_alpha']*100:>5.2f}%")

    print(f"\nFeatures that work MUCH BETTER in LOW VOLATILITY:")
    low_vol_lovers = vol_df.nsmallest(10, 'diff')
    for _, row in low_vol_lovers.iterrows():
        print(f"  {row['feature']:<35} Low Vol: {row['low_vol_alpha']*100:>5.2f}%, High Vol: {row['high_vol_alpha']*100:>5.2f}%")

    # 3. MOMENTUM REGIME
    print("\n" + "-" * 80)
    print("3. MARKET MOMENTUM REGIME (Recent 3-month return):")
    print("-" * 80)

    up_indices = [i for i, r in enumerate(regime_classifications) if r and r['momentum'] == 'Up']
    down_indices = [i for i, r in enumerate(regime_classifications) if r and r['momentum'] == 'Down']
    flat_indices = [i for i, r in enumerate(regime_classifications) if r and r['momentum'] == 'Flat']

    print(f"  Up periods: {len(up_indices)}, Flat periods: {len(flat_indices)}, Down periods: {len(down_indices)} months")

    # Best features in each regime
    for regime_name, indices in [('UP', up_indices), ('DOWN', down_indices)]:
        if len(indices) < 10:
            continue

        print(f"\nTop 10 features in {regime_name} momentum regime:")
        regime_stats = []
        for feat_idx, feat_name in enumerate(feature_names):
            alphas = feature_alpha[indices, feat_idx, n_sat - 1]
            valid = alphas[~np.isnan(alphas)]
            if len(valid) >= 5:
                regime_stats.append({
                    'feature': feat_name,
                    'alpha': np.mean(valid),
                    'hit_rate': np.mean(valid > 0)
                })

        regime_df = pd.DataFrame(regime_stats).nlargest(10, 'alpha')
        for _, row in regime_df.iterrows():
            print(f"  {row['feature']:<35} Alpha: {row['alpha']*100:>5.2f}%, Hit: {row['hit_rate']:.0%}")

    # Summary: recommended regime-conditional strategy
    print("\n" + "-" * 80)
    print("REGIME-CONDITIONAL STRATEGY RECOMMENDATIONS:")
    print("-" * 80)

    if len(trend_df) > 0:
        # Best bull-market features
        bull_best = trend_df.nlargest(3, 'bull_alpha')['feature'].tolist()
        bear_best = trend_df.nlargest(3, 'bear_alpha')['feature'].tolist()

        print(f"\nIn BULL markets, prioritize: {', '.join(bull_best)}")
        print(f"In BEAR markets, prioritize: {', '.join(bear_best)}")

        # All-weather features
        if len(robust) > 0:
            robust_features = robust.head(3)['feature'].tolist()
            print(f"All-weather (regime-agnostic): {', '.join(robust_features)}")

    return trend_df, vol_df


def analyze_feature_turnover(feature_data, results):
    """
    ANALYSIS 14: Feature Turnover Analysis

    Examines how often the selected features change month-to-month.
    High turnover may indicate noise-fitting.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 14: FEATURE TURNOVER ANALYSIS")
    print("=" * 80)

    if feature_data is None:
        print("No feature data available")
        return

    feature_names = feature_data['feature_names']
    feature_alpha = feature_data['feature_alpha']
    feature_hit = feature_data['feature_hit']
    dates = feature_data['dates']

    n_dates, n_features, n_max_satellites = feature_alpha.shape
    n_sat = 5

    # Simulate which features would be selected each month
    # (top K features by alpha * hit_rate)
    K = 5  # Number of features typically used

    selected_features_by_month = []

    for date_idx in range(n_dates):
        # Score features by alpha and hit rate
        feature_scores = []
        for feat_idx, feat_name in enumerate(feature_names):
            alpha = feature_alpha[date_idx, feat_idx, n_sat - 1]
            hit = feature_hit[date_idx, feat_idx, n_sat - 1]

            if np.isnan(alpha) or np.isnan(hit):
                continue

            # Simple scoring: alpha * hit_rate
            score = alpha * hit
            feature_scores.append((feat_name, score, alpha, hit))

        if not feature_scores:
            selected_features_by_month.append(set())
            continue

        # Select top K features
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = [f[0] for f in feature_scores[:K]]
        selected_features_by_month.append(set(top_k))

    # Calculate turnover statistics
    print(f"\nSimulating feature selection with top {K} features per month")
    print(f"Total months analyzed: {len(selected_features_by_month)}")

    turnovers = []
    for i in range(1, len(selected_features_by_month)):
        prev_set = selected_features_by_month[i-1]
        curr_set = selected_features_by_month[i]

        if len(prev_set) == 0 or len(curr_set) == 0:
            continue

        # Turnover = % of features that changed
        overlap = len(prev_set & curr_set)
        turnover = 1 - (overlap / K)
        turnovers.append({
            'date': dates[i],
            'turnover': turnover,
            'overlap': overlap,
            'new_features': list(curr_set - prev_set),
            'dropped_features': list(prev_set - curr_set)
        })

    turnover_df = pd.DataFrame(turnovers)

    if len(turnover_df) == 0:
        print("Not enough data for turnover analysis")
        return

    # Summary statistics
    print("\nFeature Turnover Statistics:")
    print("-" * 60)
    print(f"  Average monthly turnover: {turnover_df['turnover'].mean():.1%}")
    print(f"  Median monthly turnover:  {turnover_df['turnover'].median():.1%}")
    print(f"  Min turnover:            {turnover_df['turnover'].min():.1%}")
    print(f"  Max turnover:            {turnover_df['turnover'].max():.1%}")

    # Turnover distribution
    print("\nTurnover Distribution:")
    print(f"  0% (no change):    {(turnover_df['turnover'] == 0).sum():>4} months ({(turnover_df['turnover'] == 0).mean():.1%})")
    print(f"  1-20% (1 change):  {((turnover_df['turnover'] > 0) & (turnover_df['turnover'] <= 0.2)).sum():>4} months")
    print(f"  21-40% (2 changes):{((turnover_df['turnover'] > 0.2) & (turnover_df['turnover'] <= 0.4)).sum():>4} months")
    print(f"  41-60% (3 changes):{((turnover_df['turnover'] > 0.4) & (turnover_df['turnover'] <= 0.6)).sum():>4} months")
    print(f"  61-80% (4 changes):{((turnover_df['turnover'] > 0.6) & (turnover_df['turnover'] <= 0.8)).sum():>4} months")
    print(f"  81-100% (all new): {(turnover_df['turnover'] > 0.8).sum():>4} months ({(turnover_df['turnover'] > 0.8).mean():.1%})")

    # Correlate turnover with performance
    n5_results = results[results['n_satellites'] == n_sat].copy()
    turnover_df['date_match'] = turnover_df['date']

    merged = turnover_df.merge(
        n5_results[['avg_alpha']].reset_index(),
        left_on='date_match',
        right_on='date',
        how='inner'
    )

    if len(merged) > 20:
        corr = merged['turnover'].corr(merged['avg_alpha'])
        print(f"\nCorrelation: Feature Turnover vs Alpha: r = {corr:+.3f}")

        # Split by turnover level
        print("\nPerformance by Turnover Level:")
        print("-" * 60)

        merged['turnover_level'] = pd.cut(
            merged['turnover'],
            bins=[0, 0.2, 0.5, 1.0],
            labels=['Low (0-20%)', 'Medium (20-50%)', 'High (50%+)'],
            include_lowest=True
        )

        for level in ['Low (0-20%)', 'Medium (20-50%)', 'High (50%+)']:
            subset = merged[merged['turnover_level'] == level]
            if len(subset) > 0:
                hit_rate = (subset['avg_alpha'] > 0).mean()
                print(f"  {level:<18}: {len(subset):>4} months, avg alpha: {subset['avg_alpha'].mean()*100:>6.2f}%, hit rate: {hit_rate:.1%}")

    # Most stable features (appear most often)
    print("\n" + "-" * 80)
    print("Feature Persistence (how often each feature is selected):")
    print("-" * 80)

    all_selected = []
    for feat_set in selected_features_by_month:
        all_selected.extend(feat_set)

    feature_counts = pd.Series(all_selected).value_counts()
    n_months = len(selected_features_by_month)

    print(f"\nMost frequently selected features (out of {n_months} months):")
    for feat, count in feature_counts.head(15).items():
        pct = count / n_months * 100
        print(f"  {feat:<35}: {count:>3} months ({pct:.1f}%)")

    # Features that rarely get selected
    print(f"\nLeast frequently selected features (minimum 5 selections):")
    rare_features = feature_counts[(feature_counts >= 5) & (feature_counts <= 15)]
    for feat, count in rare_features.head(10).items():
        pct = count / n_months * 100
        print(f"  {feat:<35}: {count:>3} months ({pct:.1f}%)")

    # Month-over-month patterns
    print("\n" + "-" * 80)
    print("High Turnover Months (>60% feature change):")
    print("-" * 80)

    high_turnover_months = turnover_df[turnover_df['turnover'] > 0.6].sort_values('turnover', ascending=False)

    for _, row in high_turnover_months.head(10).iterrows():
        date_str = row['date'].strftime('%Y-%m')
        dropped = ', '.join(row['dropped_features'][:3]) if row['dropped_features'] else 'none'
        new = ', '.join(row['new_features'][:3]) if row['new_features'] else 'none'
        print(f"  {date_str}: {row['turnover']:.0%} turnover")
        print(f"    Dropped: {dropped}")
        print(f"    Added:   {new}")

    return turnover_df


class TeeOutput:
    """Write output to both console and file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def main():
    """Run comprehensive backtest analysis across all N values."""

    # Set up output to both console and file
    output_file = RESULTS_DIR / 'analysis_report.txt'
    tee = TeeOutput(output_file)
    old_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 80)
        print("BACKTEST PERFORMANCE ANALYSIS - ALL N VALUES")
        print("=" * 80)
        print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Source: {BACKTEST_SOURCE} ({'Script 6 - no MC filter' if BACKTEST_SOURCE == 'decay' else 'Script 7 - MC filtered'})")

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

        summary_data = []
        for n in n_values:
            n_data = results[results['n_satellites'] == n]
            avg_alpha = n_data['avg_alpha'].mean()
            std_alpha = n_data['avg_alpha'].std()
            hit_rate = (n_data['avg_alpha'] > 0).mean()
            sharpe = avg_alpha / std_alpha if std_alpha > 0 else 0
            print(f"{n:<5} {len(n_data):>8} {avg_alpha*100:>11.2f}% {avg_alpha*12*100:>9.1f}% {hit_rate:>9.1%} {sharpe:>8.3f}")
            summary_data.append({
                'n_satellites': n,
                'months': len(n_data),
                'avg_alpha_monthly': avg_alpha,
                'avg_alpha_annual': avg_alpha * 12,
                'hit_rate': hit_rate,
                'sharpe': sharpe
            })

        # Aggregate stats
        print("-" * 60)
        avg_alpha = results['avg_alpha'].mean()
        std_alpha = results['avg_alpha'].std()
        hit_rate = (results['avg_alpha'] > 0).mean()
        sharpe = avg_alpha / std_alpha if std_alpha > 0 else 0
        print(f"{'ALL':<5} {len(results):>8} {avg_alpha*100:>11.2f}% {avg_alpha*12*100:>9.1f}% {hit_rate:>9.1%} {sharpe:>8.3f}")

        # Save summary to CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(RESULTS_DIR / 'analysis_summary_by_n.csv', index=False)

        # Run analyses
        hits, misses, etf_df = analyze_hits_vs_misses(results, market)
        if etf_df is not None:
            etf_df.to_csv(RESULTS_DIR / 'analysis_etf_hit_miss.csv')

        analyze_alpha_quartiles(results, market)
        analyze_time_patterns(results)
        etf_summary = analyze_etf_performance(results)
        if etf_summary is not None:
            etf_summary.to_csv(RESULTS_DIR / 'analysis_etf_performance.csv')

        if market is not None:
            analyze_regime_performance(results, market)

        analyze_rolling_performance(results)

        # Feature-level analyses
        print("\n" + "=" * 80)
        print("FEATURE-LEVEL ANALYSES")
        print("=" * 80)

        feature_data = None
        feature_df = None
        stability_df = None
        confidence_df = None
        turnover_df = None

        try:
            feature_data = load_feature_data()
            if feature_data is not None:
                print(f"  Loaded feature data: {len(feature_data['feature_names'])} features")

                # Feature performance analysis
                feature_df = analyze_feature_performance(feature_data)
                if feature_df is not None:
                    feature_df.to_csv(RESULTS_DIR / 'analysis_feature_performance.csv', index=False)

                # Feature stability over time
                stability_df = analyze_feature_stability(feature_data)
                if stability_df is not None:
                    stability_df.to_csv(RESULTS_DIR / 'analysis_feature_stability.csv', index=False)

                # Feature correlation analysis
                corr_matrix = analyze_feature_correlation(feature_data)
                if corr_matrix is not None:
                    corr_matrix.to_csv(RESULTS_DIR / 'analysis_feature_correlation.csv')

                # Selection patterns in hits vs misses
                selection_df = analyze_selection_patterns(feature_data, results)
                if selection_df is not None:
                    selection_df.to_csv(RESULTS_DIR / 'analysis_selection_patterns.csv', index=False)

                # Deep dive analyses
                print("\n" + "=" * 80)
                print("DEEP DIVE ANALYSES")
                print("=" * 80)

                # Miss analysis deep dive
                miss_analysis = analyze_miss_deep_dive(results, feature_data, market)
                if miss_analysis is not None:
                    miss_analysis.to_csv(RESULTS_DIR / 'analysis_miss_deep_dive.csv')

                # Prediction confidence analysis
                confidence_df = analyze_prediction_confidence(feature_data, results)
                if confidence_df is not None:
                    confidence_df.to_csv(RESULTS_DIR / 'analysis_prediction_confidence.csv', index=False)

                # Regime-conditional feature performance
                if market is not None:
                    regime_results = analyze_regime_conditional_features(feature_data, market)
                    if regime_results is not None:
                        trend_df, vol_df = regime_results
                        if trend_df is not None:
                            trend_df.to_csv(RESULTS_DIR / 'analysis_regime_trend.csv', index=False)
                        if vol_df is not None:
                            vol_df.to_csv(RESULTS_DIR / 'analysis_regime_volatility.csv', index=False)

                # Feature turnover analysis
                turnover_df = analyze_feature_turnover(feature_data, results)
                if turnover_df is not None:
                    turnover_df.to_csv(RESULTS_DIR / 'analysis_feature_turnover.csv', index=False)

            else:
                print("  Feature data not available, skipping feature analyses")
        except Exception as e:
            print(f"  Warning: Could not run feature analyses: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {RESULTS_DIR}")
        print("Files generated:")
        print("  - analysis_report.txt (this report)")
        print("  - analysis_summary_by_n.csv")
        print("  - analysis_etf_hit_miss.csv")
        print("  - analysis_etf_performance.csv")
        print("  - analysis_feature_performance.csv")
        print("  - analysis_feature_stability.csv")
        print("  - analysis_feature_correlation.csv")
        print("  - analysis_selection_patterns.csv")
        print("  - analysis_miss_deep_dive.csv")
        print("  - analysis_prediction_confidence.csv")
        print("  - analysis_regime_trend.csv")
        print("  - analysis_regime_volatility.csv")
        print("  - analysis_feature_turnover.csv")

        return 0

    finally:
        # Restore stdout and close file
        sys.stdout = old_stdout
        tee.close()
        print(f"\nAnalysis complete! Report saved to: {output_file}")


if __name__ == '__main__':
    sys.exit(main())
