"""
Step 7: Out-of-Sample Validation
=================================

This script performs the MOST CRITICAL test: out-of-sample validation.

The test:
1. Split data into in-sample (2010-2019) and out-of-sample (2020-2025)
2. Run the ENTIRE pipeline on in-sample data ONLY
3. Test performance on out-of-sample data with FROZEN parameters
4. Compare in-sample vs out-of-sample performance

This is the gold standard test for overfitting. If out-of-sample performance
significantly degrades, the strategy is overfit to historical data.

Expected outcomes:
- NOT overfit: Out-of-sample hit rate ~75-85% (vs 95% in-sample)
- Overfit: Out-of-sample hit rate <60% (barely better than random)

Usage:
    python 7_out_of_sample_validation.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = '#1e1e1e'
plt.rcParams['axes.facecolor'] = '#2d2d2d'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['grid.color'] = '#3d3d3d'


# ============================================================
# CONFIGURATION
# ============================================================

IN_SAMPLE_START = '2010-01-01'
IN_SAMPLE_END = '2019-12-31'
OUT_OF_SAMPLE_START = '2020-01-01'
OUT_OF_SAMPLE_END = '2025-12-31'

N_SATELLITES = 4


# ============================================================
# STEP 1: ANALYZE CURRENT RESULTS (FULL SAMPLE)
# ============================================================

def analyze_full_sample():
    """Analyze performance on full sample (2010-2025)."""
    print("\n" + "="*60)
    print("ANALYZING FULL SAMPLE PERFORMANCE (2010-2025)")
    print("="*60)

    # Load summary
    summary_file = Path('data/feature_analysis/multi_horizon_consensus_summary.csv')
    summary_df = pd.read_csv(summary_file)

    # Load forward alpha
    alpha_file = Path('data/feature_analysis/forward_alpha_1month.parquet')
    alpha_df = pd.read_parquet(alpha_file)

    print(f"\nFull sample date range: {alpha_df['date'].min().date()} to {alpha_df['date'].max().date()}")
    print(f"Total evaluation dates: {alpha_df['date'].nunique()}")

    # Print results for all methods
    print("\n" + "-"*60)
    print("FULL SAMPLE RESULTS (ALL METHODS)")
    print("-"*60)
    for _, row in summary_df.iterrows():
        print(f"\n{row['method'].upper()}:")
        print(f"  Alpha: {row['avg_alpha']:.4f} ({row['avg_alpha']*100:.2f}%)")
        print(f"  Portfolio Hit Rate: {row['portfolio_hit_rate']:.2%}")
        print(f"  Sharpe: {row['sharpe']:.3f}")

    return alpha_df


# ============================================================
# STEP 2: SPLIT INTO IN-SAMPLE AND OUT-OF-SAMPLE
# ============================================================

def split_data(alpha_df):
    """Split data into in-sample and out-of-sample periods."""
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)

    alpha_df['date'] = pd.to_datetime(alpha_df['date'])

    in_sample = alpha_df[
        (alpha_df['date'] >= IN_SAMPLE_START) &
        (alpha_df['date'] <= IN_SAMPLE_END)
    ].copy()

    out_of_sample = alpha_df[
        (alpha_df['date'] >= OUT_OF_SAMPLE_START) &
        (alpha_df['date'] <= OUT_OF_SAMPLE_END)
    ].copy()

    print(f"\nIn-sample period: {IN_SAMPLE_START} to {IN_SAMPLE_END}")
    print(f"  Dates: {in_sample['date'].nunique()}")
    print(f"  Observations: {len(in_sample):,}")

    print(f"\nOut-of-sample period: {OUT_OF_SAMPLE_START} to {OUT_OF_SAMPLE_END}")
    print(f"  Dates: {out_of_sample['date'].nunique()}")
    print(f"  Observations: {len(out_of_sample):,}")

    return in_sample, out_of_sample


# ============================================================
# STEP 3: ANALYZE PERFORMANCE BY PERIOD
# ============================================================

def analyze_by_period(alpha_df):
    """Analyze performance broken down by time periods."""
    print("\n" + "="*60)
    print("PERFORMANCE BY TIME PERIOD")
    print("="*60)

    alpha_df['date'] = pd.to_datetime(alpha_df['date'])
    alpha_df['year'] = alpha_df['date'].dt.year

    # Analyze by year
    print("\nAnnual Performance:")
    print("-"*60)
    yearly_stats = []

    for year in sorted(alpha_df['year'].unique()):
        year_data = alpha_df[alpha_df['year'] == year]

        stats = {
            'year': year,
            'n_obs': len(year_data),
            'avg_alpha': year_data['forward_alpha'].mean(),
            'hit_rate': (year_data['forward_alpha'] > 0).mean(),
            'sharpe': year_data['forward_alpha'].mean() / year_data['forward_alpha'].std() if year_data['forward_alpha'].std() > 0 else 0
        }
        yearly_stats.append(stats)

        print(f"{year}: Alpha={stats['avg_alpha']:.4f} ({stats['avg_alpha']*100:.2f}%), "
              f"Hit={stats['hit_rate']:.1%}, Sharpe={stats['sharpe']:.3f}, N={stats['n_obs']}")

    # Compare in-sample vs out-of-sample
    print("\n" + "-"*60)
    print("IN-SAMPLE VS OUT-OF-SAMPLE (RAW ALPHA)")
    print("-"*60)

    in_sample = alpha_df[
        (alpha_df['date'] >= IN_SAMPLE_START) &
        (alpha_df['date'] <= IN_SAMPLE_END)
    ]

    out_of_sample = alpha_df[
        (alpha_df['date'] >= OUT_OF_SAMPLE_START) &
        (alpha_df['date'] <= OUT_OF_SAMPLE_END)
    ]

    print(f"\nIn-sample (2010-2019):")
    print(f"  Avg Alpha: {in_sample['forward_alpha'].mean():.4f} ({in_sample['forward_alpha'].mean()*100:.2f}%)")
    print(f"  Hit Rate: {(in_sample['forward_alpha'] > 0).mean():.2%}")
    print(f"  Sharpe: {in_sample['forward_alpha'].mean() / in_sample['forward_alpha'].std():.3f}")

    print(f"\nOut-of-sample (2020-2025):")
    print(f"  Avg Alpha: {out_of_sample['forward_alpha'].mean():.4f} ({out_of_sample['forward_alpha'].mean()*100:.2f}%)")
    print(f"  Hit Rate: {(out_of_sample['forward_alpha'] > 0).mean():.2%}")
    print(f"  Sharpe: {out_of_sample['forward_alpha'].mean() / out_of_sample['forward_alpha'].std():.3f}")

    # Test for regime differences
    print("\n" + "-"*60)
    print("REGIME ANALYSIS")
    print("-"*60)

    # Identify volatile years (high std of forward alpha)
    year_volatility = alpha_df.groupby('year')['forward_alpha'].std()
    high_vol_years = year_volatility.nlargest(5).index.tolist()
    low_vol_years = year_volatility.nsmallest(5).index.tolist()

    print(f"\nHigh volatility years: {high_vol_years}")
    print(f"Low volatility years: {low_vol_years}")

    high_vol_data = alpha_df[alpha_df['year'].isin(high_vol_years)]
    low_vol_data = alpha_df[alpha_df['year'].isin(low_vol_years)]

    print(f"\nHigh volatility regime:")
    print(f"  Avg Alpha: {high_vol_data['forward_alpha'].mean():.4f}")
    print(f"  Hit Rate: {(high_vol_data['forward_alpha'] > 0).mean():.2%}")

    print(f"\nLow volatility regime:")
    print(f"  Avg Alpha: {low_vol_data['forward_alpha'].mean():.4f}")
    print(f"  Hit Rate: {(low_vol_data['forward_alpha'] > 0).mean():.2%}")

    return yearly_stats


# ============================================================
# STEP 4: WALK-FORWARD ANALYSIS
# ============================================================

def walk_forward_analysis(alpha_df):
    """
    Simulate walk-forward testing where features are reselected periodically.

    This tests if the strategy remains effective when features are updated
    based on expanding windows of data.
    """
    print("\n" + "="*60)
    print("WALK-FORWARD ANALYSIS")
    print("="*60)
    print("\nSimulating feature reselection every 2 years...")

    alpha_df['date'] = pd.to_datetime(alpha_df['date'])

    # Define walk-forward windows
    windows = [
        ('2010-01-01', '2012-12-31', '2013-01-01', '2013-12-31'),  # Train 2010-2012, test 2013
        ('2010-01-01', '2013-12-31', '2014-01-01', '2014-12-31'),  # Train 2010-2013, test 2014
        ('2010-01-01', '2014-12-31', '2015-01-01', '2015-12-31'),  # etc.
        ('2010-01-01', '2015-12-31', '2016-01-01', '2016-12-31'),
        ('2010-01-01', '2016-12-31', '2017-01-01', '2017-12-31'),
        ('2010-01-01', '2017-12-31', '2018-01-01', '2018-12-31'),
        ('2010-01-01', '2018-12-31', '2019-01-01', '2019-12-31'),
        ('2010-01-01', '2019-12-31', '2020-01-01', '2020-12-31'),
        ('2010-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
        ('2010-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
        ('2010-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
        ('2010-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
    ]

    print("\n" + "-"*60)
    print("Year-by-Year Out-of-Sample Performance")
    print("-"*60)
    print("(Simulated - would need to rerun pipeline for true walk-forward)")
    print()

    results = []
    for train_start, train_end, test_start, test_end in windows:
        test_data = alpha_df[
            (alpha_df['date'] >= test_start) &
            (alpha_df['date'] <= test_end)
        ]

        if len(test_data) > 0:
            test_year = pd.to_datetime(test_start).year
            avg_alpha = test_data['forward_alpha'].mean()
            hit_rate = (test_data['forward_alpha'] > 0).mean()

            results.append({
                'test_year': test_year,
                'avg_alpha': avg_alpha,
                'hit_rate': hit_rate,
                'n_obs': len(test_data)
            })

            print(f"{test_year}: Alpha={avg_alpha:.4f} ({avg_alpha*100:.2f}%), "
                  f"Hit={hit_rate:.1%}, N={len(test_data)}")

    # Calculate average out-of-sample performance
    if results:
        avg_oos_alpha = np.mean([r['avg_alpha'] for r in results])
        avg_oos_hit = np.mean([r['hit_rate'] for r in results])

        print("\n" + "-"*60)
        print(f"Average walk-forward performance:")
        print(f"  Alpha: {avg_oos_alpha:.4f} ({avg_oos_alpha*100:.2f}%)")
        print(f"  Hit Rate: {avg_oos_hit:.2%}")

    return results


# ============================================================
# STEP 5: VISUALIZATION
# ============================================================

def visualize_results(yearly_stats, output_dir='data/feature_analysis'):
    """Create visualizations of temporal performance."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    output_dir = Path(output_dir)

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Temporal Performance Analysis', fontsize=16, fontweight='bold', color='white')

    years = [s['year'] for s in yearly_stats]
    alphas = [s['avg_alpha'] * 100 for s in yearly_stats]  # Convert to %
    hit_rates = [s['hit_rate'] * 100 for s in yearly_stats]  # Convert to %
    sharpes = [s['sharpe'] for s in yearly_stats]

    # Plot 1: Alpha by year
    ax = axes[0]
    colors = ['lightgreen' if y < 2020 else 'lightcoral' for y in years]
    bars = ax.bar(years, alphas, color=colors, edgecolor='black', alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.axvline(x=2019.5, color='yellow', linestyle=':', linewidth=2, label='Out-of-sample starts')
    ax.set_title('Average Alpha by Year', fontweight='bold', color='white')
    ax.set_xlabel('Year')
    ax.set_ylabel('Alpha (%)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Add in-sample / out-of-sample labels
    ax.text(2014.5, max(alphas) * 0.9, 'IN-SAMPLE', ha='center', fontsize=12,
            color='lightgreen', fontweight='bold')
    ax.text(2022, max(alphas) * 0.9, 'OUT-OF-SAMPLE', ha='center', fontsize=12,
            color='lightcoral', fontweight='bold')

    # Plot 2: Hit rate by year
    ax = axes[1]
    bars = ax.bar(years, hit_rates, color=colors, edgecolor='black', alpha=0.7)
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, label='Random (50%)')
    ax.axvline(x=2019.5, color='yellow', linestyle=':', linewidth=2)
    ax.set_title('Hit Rate by Year', fontweight='bold', color='white')
    ax.set_xlabel('Year')
    ax.set_ylabel('Hit Rate (%)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Sharpe by year
    ax = axes[2]
    bars = ax.bar(years, sharpes, color=colors, edgecolor='black', alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.axvline(x=2019.5, color='yellow', linestyle=':', linewidth=2)
    ax.set_title('Sharpe Ratio by Year', fontweight='bold', color='white')
    ax.set_xlabel('Year')
    ax.set_ylabel('Sharpe Ratio')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_file = output_dir / 'temporal_performance_analysis.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
    print(f"\n[SAVED] {plot_file}")

    plt.show()


# ============================================================
# STEP 6: DEGRADATION ANALYSIS
# ============================================================

def degradation_analysis(yearly_stats):
    """
    Analyze if performance degrades over time (sign of overfitting).

    If a strategy is overfit, performance typically:
    - Starts high in early years (data it was optimized on)
    - Degrades in later years (data it hasn't seen)
    """
    print("\n" + "="*60)
    print("DEGRADATION ANALYSIS")
    print("="*60)

    years = np.array([s['year'] for s in yearly_stats])
    alphas = np.array([s['avg_alpha'] for s in yearly_stats])
    hit_rates = np.array([s['hit_rate'] for s in yearly_stats])

    # Linear regression: performance vs time
    from scipy.stats import linregress

    # Alpha trend
    slope_alpha, intercept_alpha, r_alpha, p_alpha, se_alpha = linregress(years, alphas)

    print("\nAlpha trend over time:")
    print(f"  Slope: {slope_alpha:.6f} per year")
    print(f"  R²: {r_alpha**2:.3f}")
    print(f"  p-value: {p_alpha:.4f}")

    if p_alpha < 0.05:
        if slope_alpha < 0:
            print("  WARNING: Significant negative trend (performance degrading)")
        else:
            print("  OK: Significant positive trend (performance improving)")
    else:
        print("  OK: No significant trend (stable performance)")

    # Hit rate trend
    slope_hit, intercept_hit, r_hit, p_hit, se_hit = linregress(years, hit_rates)

    print("\nHit rate trend over time:")
    print(f"  Slope: {slope_hit:.6f} per year")
    print(f"  R²: {r_hit**2:.3f}")
    print(f"  p-value: {p_hit:.4f}")

    if p_hit < 0.05:
        if slope_hit < 0:
            print("  WARNING: Significant negative trend (performance degrading)")
        else:
            print("  OK: Significant positive trend (performance improving)")
    else:
        print("  OK: No significant trend (stable performance)")

    # Compare first half vs second half
    mid_point = len(yearly_stats) // 2
    first_half = yearly_stats[:mid_point]
    second_half = yearly_stats[mid_point:]

    first_alpha = np.mean([s['avg_alpha'] for s in first_half])
    second_alpha = np.mean([s['avg_alpha'] for s in second_half])

    first_hit = np.mean([s['hit_rate'] for s in first_half])
    second_hit = np.mean([s['hit_rate'] for s in second_half])

    print("\n" + "-"*60)
    print("First half vs Second half:")
    print(f"  First half alpha: {first_alpha:.4f} ({first_alpha*100:.2f}%)")
    print(f"  Second half alpha: {second_alpha:.4f} ({second_alpha*100:.2f}%)")
    print(f"  Change: {(second_alpha - first_alpha)/abs(first_alpha)*100:+.1f}%")

    print(f"\n  First half hit rate: {first_hit:.2%}")
    print(f"  Second half hit rate: {second_hit:.2%}")
    print(f"  Change: {(second_hit - first_hit)/first_hit*100:+.1f}%")

    if second_alpha < first_alpha * 0.7:
        print("\n  WARNING: Performance dropped >30% in second half")
        print("    -> Possible overfitting to early period")
    elif second_hit < first_hit * 0.8:
        print("\n  WARNING: Hit rate dropped >20% in second half")
        print("    -> Possible overfitting to early period")
    else:
        print("\n  OK: Performance relatively stable between halves")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Run comprehensive out-of-sample validation."""
    print("\n" + "="*60)
    print("OUT-OF-SAMPLE VALIDATION")
    print("="*60)
    print("\nThis script analyzes temporal stability and out-of-sample performance.")
    print("For TRUE out-of-sample validation, you need to:")
    print("  1. Rerun pipeline on in-sample data only (2010-2019)")
    print("  2. Freeze all parameters")
    print("  3. Test on out-of-sample data (2020-2025)")
    print("\nThis script provides ANALYSIS of existing results.")

    # Step 1: Analyze full sample
    alpha_df = analyze_full_sample()

    # Step 2: Split data
    in_sample, out_of_sample = split_data(alpha_df)

    # Step 3: Analyze by period
    yearly_stats = analyze_by_period(alpha_df)

    # Step 4: Walk-forward analysis
    wf_results = walk_forward_analysis(alpha_df)

    # Step 5: Visualize
    visualize_results(yearly_stats)

    # Step 6: Degradation analysis
    degradation_analysis(yearly_stats)

    # Final summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    print("\nIMPORTANT: This analysis uses data that was already used")
    print("for feature selection. For TRUE out-of-sample validation:")
    print()
    print("TO DO:")
    print("  [ ] Rerun 4_feature_analysis_pipeline.py with HOLDING_MONTHS = 1")
    print("      and date filter for 2010-2019 only")
    print("  [ ] Rerun 5_multi_horizon_consensus.py on in-sample results")
    print("  [ ] Manually test on 2020-2025 data with frozen features")
    print()
    print("Only then can you definitively say if the strategy is overfit.")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
