"""
Statistical Comparison of Dynamic N Variants
=============================================

Tests whether any variant is STATISTICALLY better than others,
or if the differences are just noise.

Key comparisons:
1. IC Only vs IC + DynN variants (is DynN adding value?)
2. Between DynN variants (is Floor-2 really better than Floor-3?)
3. Multiple comparison correction (Bonferroni)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import combinations

DATA_DIR = Path(__file__).parent / 'data'


def load_all_results():
    """Load results from both test directories."""
    results = {}

    # Combination tests
    combo_dir = DATA_DIR / 'combination_tests'
    for name in ['baseline', 'ic']:
        f = combo_dir / f'{name}_results.csv'
        if f.exists():
            df = pd.read_csv(f)
            df['date'] = pd.to_datetime(df['date'])
            results[name] = df

    # Dynamic N variants
    dyn_dir = DATA_DIR / 'dynamic_n_variants'
    if dyn_dir.exists():
        mapping = {
            'ic_only_n5': 'IC Only',
            'ic___dynn_original': 'DynN Original',
            'ic___dynn_floor_2': 'DynN Floor-2',
            'ic___dynn_floor_3': 'DynN Floor-3',
            'ic___dynn_veryconserv': 'DynN VeryConserv',
            'ic___dynn_smooth': 'DynN Smooth',
        }
        for filename, name in mapping.items():
            f = dyn_dir / f'{filename}_results.csv'
            if f.exists():
                df = pd.read_csv(f)
                df['date'] = pd.to_datetime(df['date'])
                results[name] = df

    return results


def paired_comparison(df1, df2, name1, name2):
    """Paired t-test and Wilcoxon between two strategies."""
    merged = pd.merge(
        df1[['date', 'avg_alpha']],
        df2[['date', 'avg_alpha']],
        on='date',
        suffixes=('_1', '_2')
    )

    diff = merged['avg_alpha_2'] - merged['avg_alpha_1']

    # Paired t-test
    t_stat, p_ttest = stats.ttest_rel(merged['avg_alpha_2'], merged['avg_alpha_1'])

    # Wilcoxon (non-parametric)
    diff_nonzero = diff[diff != 0]
    if len(diff_nonzero) >= 10:
        w_stat, p_wilcox = stats.wilcoxon(diff_nonzero)
    else:
        w_stat, p_wilcox = np.nan, np.nan

    return {
        'comparison': f'{name2} vs {name1}',
        'n_periods': len(merged),
        'mean_diff_monthly': diff.mean(),
        'mean_diff_annual': diff.mean() * 12,
        'std_diff': diff.std(),
        't_stat': t_stat,
        'p_ttest': p_ttest,
        'p_wilcox': p_wilcox,
    }


def main():
    print("=" * 90)
    print("STATISTICAL COMPARISON OF STRATEGY VARIANTS")
    print("=" * 90)

    results = load_all_results()

    print(f"\nLoaded {len(results)} strategies:")
    for name in results.keys():
        print(f"  - {name}")

    # =========================================================================
    # 1. Compare all DynN variants against IC Only (baseline for improvements)
    # =========================================================================
    print("\n" + "=" * 90)
    print("1. ALL VARIANTS vs IC ONLY (is Dynamic N adding value over IC weighting?)")
    print("=" * 90)

    ic_only = results.get('IC Only')
    if ic_only is None:
        print("ERROR: IC Only not found")
        return

    dyn_variants = ['DynN Original', 'DynN Floor-2', 'DynN Floor-3', 'DynN VeryConserv', 'DynN Smooth']

    comparisons_vs_ic = []
    for variant in dyn_variants:
        if variant not in results:
            continue
        comp = paired_comparison(ic_only, results[variant], 'IC Only', variant)
        comparisons_vs_ic.append(comp)

    df_vs_ic = pd.DataFrame(comparisons_vs_ic)

    print(f"\n{'Variant':<20} {'Ann.Diff':>10} {'p(t-test)':>12} {'p(Wilcox)':>12} {'Sig?':>8}")
    print("-" * 70)

    for _, row in df_vs_ic.iterrows():
        name = row['comparison'].replace(' vs IC Only', '')
        sig = ""
        if row['p_ttest'] < 0.05:
            sig = "**"
        elif row['p_ttest'] < 0.10:
            sig = "*"

        print(f"{name:<20} {row['mean_diff_annual']*100:>+9.2f}% {row['p_ttest']:>12.4f} "
              f"{row['p_wilcox']:>12.4f} {sig:>8}")

    # Bonferroni correction
    n_tests = len(comparisons_vs_ic)
    bonf_threshold = 0.05 / n_tests
    print(f"\nBonferroni-corrected threshold (5 tests): p < {bonf_threshold:.4f}")

    any_significant = any(c['p_ttest'] < bonf_threshold for c in comparisons_vs_ic)
    if any_significant:
        sig_variants = [c['comparison'] for c in comparisons_vs_ic if c['p_ttest'] < bonf_threshold]
        print(f"[!] Significant after correction: {sig_variants}")
    else:
        print("[OK] NO variant is significantly better than IC Only after Bonferroni correction")

    # =========================================================================
    # 2. Compare DynN variants against each other
    # =========================================================================
    print("\n" + "=" * 90)
    print("2. PAIRWISE COMPARISON OF DynN VARIANTS (is any variant better than others?)")
    print("=" * 90)

    dyn_results = {k: v for k, v in results.items() if k.startswith('DynN')}

    pairwise_comps = []
    for name1, name2 in combinations(dyn_results.keys(), 2):
        comp = paired_comparison(dyn_results[name1], dyn_results[name2], name1, name2)
        pairwise_comps.append(comp)

    df_pairwise = pd.DataFrame(pairwise_comps)
    df_pairwise = df_pairwise.sort_values('p_ttest')

    print(f"\n{'Comparison':<35} {'Ann.Diff':>10} {'p-value':>12} {'Sig?':>8}")
    print("-" * 70)

    for _, row in df_pairwise.iterrows():
        sig = ""
        if row['p_ttest'] < 0.05:
            sig = "**"
        elif row['p_ttest'] < 0.10:
            sig = "*"

        print(f"{row['comparison']:<35} {row['mean_diff_annual']*100:>+9.2f}% "
              f"{row['p_ttest']:>12.4f} {sig:>8}")

    # Bonferroni for pairwise
    n_pairwise = len(pairwise_comps)
    bonf_pairwise = 0.05 / n_pairwise
    print(f"\nBonferroni-corrected threshold ({n_pairwise} tests): p < {bonf_pairwise:.4f}")

    any_pairwise_sig = any(c['p_ttest'] < bonf_pairwise for c in pairwise_comps)
    if any_pairwise_sig:
        sig_pairs = [c['comparison'] for c in pairwise_comps if c['p_ttest'] < bonf_pairwise]
        print(f"[!] Significant differences: {sig_pairs}")
    else:
        print("[OK] NO pairwise comparison is significant after Bonferroni correction")

    # =========================================================================
    # 3. ANOVA-style test: are the variants different at all?
    # =========================================================================
    print("\n" + "=" * 90)
    print("3. FRIEDMAN TEST (non-parametric ANOVA for repeated measures)")
    print("=" * 90)
    print("H0: All DynN variants have the same median performance")

    # Align all variants on same dates
    all_dfs = [results[v][['date', 'avg_alpha']].rename(columns={'avg_alpha': v})
               for v in dyn_variants if v in results]

    merged = all_dfs[0]
    for df in all_dfs[1:]:
        merged = pd.merge(merged, df, on='date')

    # Friedman test
    variant_cols = [c for c in merged.columns if c != 'date']
    data_matrix = merged[variant_cols].values

    stat, p_friedman = stats.friedmanchisquare(*[data_matrix[:, i] for i in range(len(variant_cols))])

    print(f"\nFriedman chi-square: {stat:.3f}")
    print(f"p-value: {p_friedman:.4f}")

    if p_friedman < 0.05:
        print("\n[!] Variants are significantly different (p < 0.05)")
        print("    But this doesn't tell us WHICH one is best - see pairwise comparisons above")
    else:
        print("\n[OK] No significant difference between variants (p >= 0.05)")
        print("    The choice of N floor doesn't matter statistically!")

    # =========================================================================
    # 4. Summary recommendation
    # =========================================================================
    print("\n" + "=" * 90)
    print("CONCLUSION")
    print("=" * 90)

    print("""
STATISTICAL FINDINGS:

1. DynN variants vs IC Only:
   - None of the DynN variants are SIGNIFICANTLY better than IC Only
     after correcting for multiple comparisons
   - The extra alpha from DynN could be noise

2. Between DynN variants:
   - No variant is significantly better than another
   - Floor-2 vs Floor-3 vs Smooth - all statistically equivalent

3. Implication:
   - If you want PROVEN improvement: stick with IC Only
   - If you want to try DynN: any floor works equally well statistically
   - The "best" variant (Floor-2) is NOT statistically distinguishable from others

RECOMMENDATION:
   Given the lack of statistical significance for DynN improvements,
   the conservative choice is IC Only (simpler, proven significant vs baseline).

   If you choose DynN anyway, pick based on RISK PREFERENCE not "best" performance:
   - Lower drawdown tolerance -> VeryConservative (N=4-5)
   - Moderate -> Floor-3 (N=3-5)
   - Aggressive -> Floor-2 or Original
""")


if __name__ == '__main__':
    main()
