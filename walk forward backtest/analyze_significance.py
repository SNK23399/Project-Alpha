"""
Statistical Significance Analysis for Strategy Improvements
============================================================

Tests whether strategy improvements are statistically significant vs baseline.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data' / 'combination_tests'

def load_results():
    """Load all strategy results."""
    results = {}
    for f in DATA_DIR.glob('*_results.csv'):
        name = f.stem.replace('_results', '')
        df = pd.read_csv(f)
        df['date'] = pd.to_datetime(df['date'])
        results[name] = df
    return results

def paired_ttest(baseline, strategy):
    """Paired t-test for matched samples."""
    # Align on dates
    merged = pd.merge(
        baseline[['date', 'avg_alpha']],
        strategy[['date', 'avg_alpha']],
        on='date',
        suffixes=('_base', '_strat')
    )

    diff = merged['avg_alpha_strat'] - merged['avg_alpha_base']
    t_stat, p_value = stats.ttest_rel(merged['avg_alpha_strat'], merged['avg_alpha_base'])

    return {
        'n_periods': len(merged),
        'mean_diff': diff.mean(),
        'std_diff': diff.std(),
        't_stat': t_stat,
        'p_value': p_value,
        'significant_5pct': p_value < 0.05,
        'significant_10pct': p_value < 0.10,
    }

def wilcoxon_test(baseline, strategy):
    """Wilcoxon signed-rank test (non-parametric alternative)."""
    merged = pd.merge(
        baseline[['date', 'avg_alpha']],
        strategy[['date', 'avg_alpha']],
        on='date',
        suffixes=('_base', '_strat')
    )

    diff = merged['avg_alpha_strat'] - merged['avg_alpha_base']

    # Remove zeros (Wilcoxon can't handle them)
    diff_nonzero = diff[diff != 0]

    if len(diff_nonzero) < 10:
        return {'p_value': np.nan, 'significant_5pct': False}

    stat, p_value = stats.wilcoxon(diff_nonzero)

    return {
        'w_stat': stat,
        'p_value': p_value,
        'significant_5pct': p_value < 0.05,
        'significant_10pct': p_value < 0.10,
    }

def bootstrap_ci(baseline, strategy, n_bootstrap=10000):
    """Bootstrap confidence interval for mean difference."""
    merged = pd.merge(
        baseline[['date', 'avg_alpha']],
        strategy[['date', 'avg_alpha']],
        on='date',
        suffixes=('_base', '_strat')
    )

    diff = (merged['avg_alpha_strat'] - merged['avg_alpha_base']).values
    n = len(diff)

    # Bootstrap
    np.random.seed(42)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(diff, size=n, replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)

    # Significant if CI doesn't include 0
    significant = (ci_lower > 0) or (ci_upper < 0)

    return {
        'mean_diff': np.mean(diff),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant_95ci': significant,
    }

def main():
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 80)

    results = load_results()
    baseline = results['baseline']

    # Strategies to test
    strategies_to_test = [
        'ic',
        'stab',
        'ic_stab',
        'dynn',
        'ic_dynn',
        'ic_stab_dynn',
        'ic_stab_ts_dynn',
    ]

    print("\n" + "-" * 80)
    print("PAIRED T-TEST (H0: no difference vs baseline)")
    print("-" * 80)
    print(f"{'Strategy':<25} {'Mean Diff':>10} {'t-stat':>10} {'p-value':>10} {'Sig?':>8}")
    print("-" * 80)

    significance_results = []

    for strat_name in strategies_to_test:
        if strat_name not in results:
            continue

        strat = results[strat_name]

        # T-test
        ttest = paired_ttest(baseline, strat)

        # Wilcoxon
        wilcox = wilcoxon_test(baseline, strat)

        # Bootstrap
        boot = bootstrap_ci(baseline, strat)

        sig_marker = ""
        if ttest['significant_5pct']:
            sig_marker = "**"
        elif ttest['significant_10pct']:
            sig_marker = "*"

        print(f"{strat_name:<25} {ttest['mean_diff']*100:>9.2f}% {ttest['t_stat']:>10.3f} "
              f"{ttest['p_value']:>10.4f} {sig_marker:>8}")

        significance_results.append({
            'strategy': strat_name,
            'mean_diff_monthly': ttest['mean_diff'],
            'mean_diff_annual': ttest['mean_diff'] * 12,
            't_stat': ttest['t_stat'],
            'p_value_ttest': ttest['p_value'],
            'p_value_wilcoxon': wilcox['p_value'],
            'ci_lower': boot['ci_lower'],
            'ci_upper': boot['ci_upper'],
            'significant_ttest_5pct': ttest['significant_5pct'],
            'significant_ttest_10pct': ttest['significant_10pct'],
            'significant_wilcoxon_5pct': wilcox['significant_5pct'],
            'significant_bootstrap_95ci': boot['significant_95ci'],
        })

    print("\nLegend: ** = p < 0.05, * = p < 0.10")

    # Summary table
    print("\n" + "=" * 80)
    print("FULL SIGNIFICANCE SUMMARY")
    print("=" * 80)

    df_sig = pd.DataFrame(significance_results)

    print(f"\n{'Strategy':<20} {'Ann.Diff':>10} {'p(t-test)':>10} {'p(Wilcox)':>10} {'95% CI':>20}")
    print("-" * 80)

    for _, row in df_sig.iterrows():
        ci_str = f"[{row['ci_lower']*100:.2f}%, {row['ci_upper']*100:.2f}%]"
        print(f"{row['strategy']:<20} {row['mean_diff_annual']*100:>9.2f}% "
              f"{row['p_value_ttest']:>10.4f} {row['p_value_wilcoxon']:>10.4f} {ci_str:>20}")

    # Save results
    df_sig.to_csv(DATA_DIR / 'significance_analysis.csv', index=False)
    print(f"\n[SAVED] Results to {DATA_DIR / 'significance_analysis.csv'}")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    for _, row in df_sig.iterrows():
        strat = row['strategy']
        p = row['p_value_ttest']
        diff = row['mean_diff_annual'] * 100

        if p < 0.05:
            verdict = f"SIGNIFICANT at 5% level (p={p:.4f})"
        elif p < 0.10:
            verdict = f"MARGINALLY significant at 10% level (p={p:.4f})"
        else:
            verdict = f"NOT significant (p={p:.4f})"

        direction = "improvement" if diff > 0 else "worse"
        print(f"\n{strat}:")
        print(f"  -> {diff:+.2f}% annual alpha {direction}")
        print(f"  -> {verdict}")

        # Check if CI includes zero
        if row['ci_lower'] > 0:
            print(f"  -> 95% CI entirely above zero: [{row['ci_lower']*100:.2f}%, {row['ci_upper']*100:.2f}%]")
        elif row['ci_upper'] < 0:
            print(f"  -> 95% CI entirely below zero: [{row['ci_lower']*100:.2f}%, {row['ci_upper']*100:.2f}%]")
        else:
            print(f"  -> 95% CI includes zero: [{row['ci_lower']*100:.2f}%, {row['ci_upper']*100:.2f}%]")

if __name__ == '__main__':
    main()
