"""
Robustness Analysis: Check if results are driven by early lucky periods
========================================================================

This script analyzes whether the strategy improvements are consistent
across time, or if they're just compounding from early lucky periods.

Tests:
1. Year-by-year performance breakdown
2. Rolling alpha consistency (are we beating baseline throughout?)
3. Sub-period analysis (early vs mid vs late)
4. Worst-period analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

DATA_DIR = Path(__file__).parent / 'data'

def load_strategy_results():
    """Load all strategy result files."""
    results = {}

    # From combination_tests
    combo_dir = DATA_DIR / 'combination_tests'
    for f in combo_dir.glob('*_results.csv'):
        name = f.stem.replace('_results', '')
        df = pd.read_csv(f)
        df['date'] = pd.to_datetime(df['date'])
        results[name] = df

    # From dynamic_n_variants
    dyn_dir = DATA_DIR / 'dynamic_n_variants'
    if dyn_dir.exists():
        for f in dyn_dir.glob('*_results.csv'):
            name = 'dyn_' + f.stem.replace('_results', '')
            df = pd.read_csv(f)
            df['date'] = pd.to_datetime(df['date'])
            results[name] = df

    return results


def yearly_breakdown(df, name):
    """Analyze performance by year."""
    df = df.copy()
    df['year'] = df['date'].dt.year

    yearly = df.groupby('year').agg({
        'avg_alpha': ['mean', 'std', 'count'],
    }).round(4)
    yearly.columns = ['mean_alpha', 'std_alpha', 'n_months']
    yearly['hit_rate'] = df.groupby('year')['avg_alpha'].apply(lambda x: (x > 0).mean())
    yearly['annual_alpha'] = yearly['mean_alpha'] * 12

    return yearly


def compare_yearly(baseline_df, strategy_df, strategy_name):
    """Compare strategy vs baseline year by year."""
    # Merge on date
    merged = pd.merge(
        baseline_df[['date', 'avg_alpha']],
        strategy_df[['date', 'avg_alpha']],
        on='date',
        suffixes=('_base', '_strat')
    )
    merged['diff'] = merged['avg_alpha_strat'] - merged['avg_alpha_base']
    merged['year'] = merged['date'].dt.year

    yearly = merged.groupby('year').agg({
        'avg_alpha_base': 'mean',
        'avg_alpha_strat': 'mean',
        'diff': ['mean', 'std'],
    })
    yearly.columns = ['base_alpha', 'strat_alpha', 'diff_mean', 'diff_std']
    yearly['wins_baseline'] = merged.groupby('year')['diff'].apply(lambda x: (x > 0).mean())
    yearly['annual_diff'] = yearly['diff_mean'] * 12

    return yearly


def subperiod_analysis(baseline_df, strategy_df, n_periods=3):
    """Split into sub-periods and analyze each."""
    merged = pd.merge(
        baseline_df[['date', 'avg_alpha']],
        strategy_df[['date', 'avg_alpha']],
        on='date',
        suffixes=('_base', '_strat')
    )
    merged['diff'] = merged['avg_alpha_strat'] - merged['avg_alpha_base']
    merged = merged.sort_values('date')

    # Split into n equal periods
    n = len(merged)
    period_size = n // n_periods

    results = []
    for i in range(n_periods):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < n_periods - 1 else n

        subset = merged.iloc[start_idx:end_idx]

        results.append({
            'period': i + 1,
            'start_date': subset['date'].min(),
            'end_date': subset['date'].max(),
            'n_months': len(subset),
            'base_alpha': subset['avg_alpha_base'].mean() * 12,
            'strat_alpha': subset['avg_alpha_strat'].mean() * 12,
            'diff': subset['diff'].mean() * 12,
            'hit_rate_base': (subset['avg_alpha_base'] > 0).mean(),
            'hit_rate_strat': (subset['avg_alpha_strat'] > 0).mean(),
            'wins_baseline': (subset['diff'] > 0).mean(),
        })

    return pd.DataFrame(results)


def rolling_alpha_analysis(baseline_df, strategy_df, window=12):
    """Analyze rolling alpha difference."""
    merged = pd.merge(
        baseline_df[['date', 'avg_alpha']],
        strategy_df[['date', 'avg_alpha']],
        on='date',
        suffixes=('_base', '_strat')
    )
    merged['diff'] = merged['avg_alpha_strat'] - merged['avg_alpha_base']
    merged = merged.sort_values('date')

    # Rolling statistics
    merged['rolling_diff'] = merged['diff'].rolling(window).mean()
    merged['rolling_positive'] = (merged['diff'] > 0).rolling(window).mean()

    # Summary stats
    valid = merged.dropna()

    return {
        'pct_rolling_positive': (valid['rolling_diff'] > 0).mean(),
        'avg_rolling_diff': valid['rolling_diff'].mean() * 12,
        'min_rolling_diff': valid['rolling_diff'].min() * 12,
        'max_rolling_diff': valid['rolling_diff'].max() * 12,
        'rolling_series': merged[['date', 'rolling_diff', 'rolling_positive']].dropna()
    }


def worst_periods_analysis(baseline_df, strategy_df, n_worst=5):
    """Find the worst months and check if strategy still helps."""
    merged = pd.merge(
        baseline_df[['date', 'avg_alpha']],
        strategy_df[['date', 'avg_alpha']],
        on='date',
        suffixes=('_base', '_strat')
    )
    merged['diff'] = merged['avg_alpha_strat'] - merged['avg_alpha_base']

    # Worst months for baseline
    worst_base = merged.nsmallest(n_worst, 'avg_alpha_base')

    # Worst months for strategy
    worst_strat = merged.nsmallest(n_worst, 'avg_alpha_strat')

    return {
        'worst_baseline_months': worst_base[['date', 'avg_alpha_base', 'avg_alpha_strat', 'diff']],
        'worst_strategy_months': worst_strat[['date', 'avg_alpha_base', 'avg_alpha_strat', 'diff']],
        'strategy_helps_in_worst': (worst_base['diff'] > 0).mean(),
    }


def main():
    print("=" * 90)
    print("ROBUSTNESS ANALYSIS: Checking for early luck vs consistent improvement")
    print("=" * 90)

    results = load_strategy_results()

    # Get baseline
    baseline = results.get('baseline')
    if baseline is None:
        print("ERROR: No baseline found")
        return

    # Key strategies to analyze
    strategies_to_check = [
        ('ic', 'IC Weighting'),
        ('ic_dynn', 'IC + Dynamic N'),
        ('dyn_ic___dynn_floor_2', 'IC + DynN Floor-2'),
        ('dyn_ic___dynn_floor_3', 'IC + DynN Floor-3'),
        ('dyn_ic___dynn_veryconserv', 'IC + DynN VeryConserv'),
    ]

    for key, name in strategies_to_check:
        if key not in results:
            continue

        strat = results[key]

        print(f"\n{'='*90}")
        print(f"STRATEGY: {name}")
        print("=" * 90)

        # 1. Year-by-year comparison
        print("\n1. YEAR-BY-YEAR PERFORMANCE vs BASELINE")
        print("-" * 70)
        yearly = compare_yearly(baseline, strat, name)

        print(f"{'Year':<8} {'Base Alpha':>12} {'Strat Alpha':>12} {'Diff':>12} {'Wins Base':>12}")
        print("-" * 70)

        all_positive = True
        for year, row in yearly.iterrows():
            diff_str = f"{row['annual_diff']*100:+.2f}%"
            wins_str = f"{row['wins_baseline']*100:.0f}%"
            base_str = f"{row['base_alpha']*12*100:.2f}%"
            strat_str = f"{row['strat_alpha']*12*100:.2f}%"

            marker = " <-- WORSE" if row['annual_diff'] < 0 else ""
            if row['annual_diff'] < 0:
                all_positive = False

            print(f"{year:<8} {base_str:>12} {strat_str:>12} {diff_str:>12} {wins_str:>12}{marker}")

        if all_positive:
            print("\n  [OK] Strategy beats baseline in ALL years")
        else:
            n_worse = (yearly['annual_diff'] < 0).sum()
            print(f"\n  [!] Strategy underperforms in {n_worse}/{len(yearly)} years")

        # 2. Sub-period analysis (thirds)
        print("\n2. SUB-PERIOD ANALYSIS (Early / Middle / Late)")
        print("-" * 70)
        subperiods = subperiod_analysis(baseline, strat, n_periods=3)

        period_names = ['Early', 'Middle', 'Late']
        for _, row in subperiods.iterrows():
            pname = period_names[int(row['period']) - 1]
            print(f"\n  {pname} ({row['start_date'].strftime('%Y-%m')} to {row['end_date'].strftime('%Y-%m')}):")
            print(f"    Baseline alpha: {row['base_alpha']*100:.2f}%")
            print(f"    Strategy alpha: {row['strat_alpha']*100:.2f}%")
            print(f"    Difference:     {row['diff']*100:+.2f}%")
            print(f"    Beats baseline: {row['wins_baseline']*100:.0f}% of months")

        # Check consistency
        all_periods_positive = (subperiods['diff'] > 0).all()
        if all_periods_positive:
            print("\n  [OK] Strategy outperforms in ALL sub-periods - NOT just early luck!")
        else:
            worse_periods = subperiods[subperiods['diff'] < 0]['period'].tolist()
            print(f"\n  [!] Strategy underperforms in period(s): {worse_periods}")

        # 3. Rolling alpha
        print("\n3. ROLLING 12-MONTH ALPHA ANALYSIS")
        print("-" * 70)
        rolling = rolling_alpha_analysis(baseline, strat, window=12)

        print(f"  % of time rolling alpha > 0:  {rolling['pct_rolling_positive']*100:.1f}%")
        print(f"  Average rolling alpha diff:   {rolling['avg_rolling_diff']*100:+.2f}%")
        print(f"  Worst rolling 12m diff:       {rolling['min_rolling_diff']*100:+.2f}%")
        print(f"  Best rolling 12m diff:        {rolling['max_rolling_diff']*100:+.2f}%")

        if rolling['pct_rolling_positive'] > 0.8:
            print("  [OK] Strategy consistently beats baseline (>80% of rolling windows)")
        elif rolling['pct_rolling_positive'] > 0.6:
            print("  [~] Strategy usually beats baseline (60-80% of rolling windows)")
        else:
            print("  [!] Strategy inconsistently beats baseline (<60% of rolling windows)")

        # 4. Worst periods
        print("\n4. WORST MONTHS ANALYSIS")
        print("-" * 70)
        worst = worst_periods_analysis(baseline, strat)

        print("  5 worst months for BASELINE:")
        for _, row in worst['worst_baseline_months'].iterrows():
            marker = "HELPED" if row['diff'] > 0 else "HURT"
            print(f"    {row['date'].strftime('%Y-%m')}: Base={row['avg_alpha_base']*100:+.1f}%, "
                  f"Strat={row['avg_alpha_strat']*100:+.1f}% [{marker}]")

        print(f"\n  Strategy helps in {worst['strategy_helps_in_worst']*100:.0f}% of baseline's worst months")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print("""
Key questions answered:
1. Is the improvement consistent across years? (Check year-by-year)
2. Is it just early luck compounding? (Check sub-periods)
3. Does it work throughout the backtest? (Check rolling alpha)
4. Does it help or hurt in bad times? (Check worst months)

A ROBUST strategy should:
- Outperform in most/all years
- Outperform in early, middle, AND late periods
- Have positive rolling alpha >70% of the time
- Not dramatically underperform in crisis months
""")


if __name__ == '__main__':
    main()
