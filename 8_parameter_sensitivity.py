"""
Step 8: Parameter Sensitivity Analysis
=======================================

This script tests how sensitive your results are to hyperparameter choices.

If performance is highly sensitive to specific parameter values, it suggests
overfitting (you've tuned parameters to maximize backtest performance).

Parameters tested:
- N_SATELLITES: 2, 3, 4, 5, 6
- MIN_ALPHA thresholds
- MIN_HIT_RATE thresholds
- Consensus method variations

Expected outcome if NOT overfit:
- Performance should be relatively stable across reasonable parameter ranges
- No "cliff" where changing N_SATELLITES from 4 to 3 causes collapse

Expected outcome if overfit:
- Performance is optimal only at specific values
- Small changes cause large performance drops

Usage:
    python 8_parameter_sensitivity.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

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

# Parameters to test
N_SATELLITES_RANGE = [2, 3, 4, 5, 6, 8, 10]
CONSENSUS_METHODS = ['unanimous', 'primary_only', 'weighted_avg', 'primary_veto', 'majority']


# ============================================================
# ANALYZE EXISTING RESULTS
# ============================================================

def analyze_consensus_methods():
    """Analyze how performance varies across consensus methods."""
    print("\n" + "="*60)
    print("CONSENSUS METHOD SENSITIVITY")
    print("="*60)

    summary_file = Path('data/feature_analysis/multi_horizon_consensus_summary.csv')
    if not summary_file.exists():
        print(f"ERROR: {summary_file} not found!")
        print("Please run 5_multi_horizon_consensus.py first.")
        return None

    summary_df = pd.read_csv(summary_file)

    print("\nPerformance across different consensus methods:")
    print("-"*60)

    for _, row in summary_df.iterrows():
        print(f"\n{row['method'].upper()}:")
        print(f"  Alpha: {row['avg_alpha']:.4f} ({row['avg_alpha']*100:.2f}%)")
        print(f"  Portfolio Hit Rate: {row['portfolio_hit_rate']:.2%}")
        print(f"  Sharpe: {row['sharpe']:.3f}")

    # Calculate variability
    alpha_std = summary_df['avg_alpha'].std()
    alpha_range = summary_df['avg_alpha'].max() - summary_df['avg_alpha'].min()
    hit_std = summary_df['portfolio_hit_rate'].std()
    hit_range = summary_df['portfolio_hit_rate'].max() - summary_df['portfolio_hit_rate'].min()

    print("\n" + "-"*60)
    print("Sensitivity metrics:")
    print(f"  Alpha std: {alpha_std:.4f}")
    print(f"  Alpha range: {alpha_range:.4f} ({alpha_range*100:.2f}%)")
    print(f"  Hit rate std: {hit_std:.4f}")
    print(f"  Hit rate range: {hit_range:.4f} ({hit_range*100:.2f}%)")

    print("\nInterpretation:")
    if hit_range > 0.30:  # 30% range
        print("  WARNING: HIGH SENSITIVITY - Performance varies greatly by method")
        print("    -> Strategy may be overfit to 'unanimous' method")
    elif hit_range > 0.15:  # 15% range
        print("  WARNING: MODERATE SENSITIVITY - Some variation across methods")
        print("    -> Consider testing robustness further")
    else:
        print("  OK: LOW SENSITIVITY - Performance stable across methods")
        print("    -> Good sign of robustness")

    return summary_df


# ============================================================
# SIMULATE N_SATELLITES SENSITIVITY
# ============================================================

def simulate_n_satellites_sensitivity():
    """
    Simulate how performance changes with different N_SATELLITES.

    Note: This is a simplified simulation based on top-N selection.
    True validation would require rerunning the entire pipeline.
    """
    print("\n" + "="*60)
    print("N_SATELLITES SENSITIVITY (SIMULATED)")
    print("="*60)

    print("\nLoading forward alpha data...")
    alpha_file = Path('data/feature_analysis/forward_alpha_1month.parquet')
    if not alpha_file.exists():
        print(f"ERROR: {alpha_file} not found!")
        return None

    alpha_df = pd.read_parquet(alpha_file)

    # Load 1-month ensemble rankings
    print("Loading ensemble rankings...")
    matrix_file = Path('data/feature_analysis/rankings_matrix_1month.npz')
    if not matrix_file.exists():
        print(f"ERROR: {matrix_file} not found!")
        return None

    data = np.load(matrix_file, allow_pickle=True)
    rankings = data['rankings']
    dates = pd.to_datetime(data['dates'])
    isins = data['isins']
    features = data['features']

    # Load ensemble
    ensemble_file = Path('data/feature_analysis/ensemble_1month.csv')
    if not ensemble_file.exists():
        print(f"ERROR: {ensemble_file} not found!")
        return None

    ensemble_df = pd.read_csv(ensemble_file)

    # Get feature indices
    feature_indices = []
    for feat_name in ensemble_df['feature_name']:
        try:
            idx = list(features).index(feat_name)
            feature_indices.append(idx)
        except ValueError:
            pass

    if len(feature_indices) == 0:
        print("ERROR: No ensemble features found!")
        return None

    # Compute ensemble scores
    ensemble_rankings = rankings[:, :, feature_indices]
    ensemble_scores = np.nanmean(ensemble_rankings, axis=2)

    print(f"\nTesting N_SATELLITES: {N_SATELLITES_RANGE}")
    print("-"*60)

    results = []

    for n_sats in N_SATELLITES_RANGE:
        # Select top N for each date
        selections = []
        for date_idx, date in enumerate(dates):
            scores_at_date = ensemble_scores[date_idx, :]
            valid_mask = ~np.isnan(scores_at_date)

            if valid_mask.sum() >= n_sats:
                top_indices = np.argsort(scores_at_date)[-n_sats:]
                for isin_idx in top_indices:
                    selections.append({
                        'date': date,
                        'isin': isins[isin_idx]
                    })

        selections_df = pd.DataFrame(selections)

        # Evaluate performance
        merged = pd.merge(selections_df, alpha_df, on=['date', 'isin'], how='inner')

        if len(merged) > 0:
            avg_alpha = merged['forward_alpha'].mean()
            hit_rate = (merged['forward_alpha'] > 0).mean()
            date_alphas = merged.groupby('date')['forward_alpha'].mean()
            portfolio_hit_rate = (date_alphas > 0).mean()
            sharpe = avg_alpha / merged['forward_alpha'].std() if merged['forward_alpha'].std() > 0 else 0

            results.append({
                'n_satellites': n_sats,
                'avg_alpha': avg_alpha,
                'hit_rate': hit_rate,
                'portfolio_hit_rate': portfolio_hit_rate,
                'sharpe': sharpe
            })

            print(f"N={n_sats:2d}: Alpha={avg_alpha:.4f} ({avg_alpha*100:.2f}%), "
                  f"Portfolio Hit={portfolio_hit_rate:.1%}, Sharpe={sharpe:.3f}")

    results_df = pd.DataFrame(results)

    # Analyze sensitivity
    print("\n" + "-"*60)
    print("Sensitivity analysis:")

    alpha_std = results_df['avg_alpha'].std()
    hit_std = results_df['portfolio_hit_rate'].std()

    alpha_range = results_df['avg_alpha'].max() - results_df['avg_alpha'].min()
    hit_range = results_df['portfolio_hit_rate'].max() - results_df['portfolio_hit_rate'].min()

    print(f"  Alpha range: {alpha_range:.4f} ({alpha_range*100:.2f}%)")
    print(f"  Hit rate range: {hit_range:.4f} ({hit_range*100:.2f}%)")

    # Check for "cliff" effect
    best_idx = results_df['portfolio_hit_rate'].argmax()
    best_n = results_df.iloc[best_idx]['n_satellites']
    best_hit = results_df.iloc[best_idx]['portfolio_hit_rate']

    print(f"\n  Best N_SATELLITES: {best_n:.0f} (hit rate: {best_hit:.2%})")

    # Check neighbors
    neighbors = results_df[
        (results_df['n_satellites'] >= best_n - 1) &
        (results_df['n_satellites'] <= best_n + 1)
    ]

    if len(neighbors) > 1:
        neighbor_hits = neighbors['portfolio_hit_rate'].values
        max_drop = best_hit - neighbor_hits.min()

        print(f"  Max drop from best: {max_drop:.2%}")

        if max_drop > 0.20:  # 20% drop
            print("  WARNING: CLIFF EFFECT - Performance drops >20% at nearby N")
            print("    -> Suggests overfitting to N_SATELLITES=4")
        elif max_drop > 0.10:
            print("  WARNING: MODERATE SENSITIVITY - Performance varies 10-20%")
        else:
            print("  OK: STABLE - Performance similar across N values")

    return results_df


# ============================================================
# HORIZON WEIGHT SENSITIVITY
# ============================================================

def horizon_weight_sensitivity():
    """Test if results are sensitive to horizon weight configuration."""
    print("\n" + "="*60)
    print("HORIZON WEIGHT SENSITIVITY")
    print("="*60)

    print("\nComparing 'unanimous' (all horizons equal) vs 'weighted_avg' (exponential decay):")

    summary_file = Path('data/feature_analysis/multi_horizon_consensus_summary.csv')
    summary_df = pd.read_csv(summary_file)

    unanimous = summary_df[summary_df['method'] == 'unanimous'].iloc[0]
    weighted = summary_df[summary_df['method'] == 'weighted_avg'].iloc[0]

    print(f"\nUnanimous (equal weight):")
    print(f"  Portfolio Hit Rate: {unanimous['portfolio_hit_rate']:.2%}")
    print(f"  Alpha: {unanimous['avg_alpha']:.4f}")

    print(f"\nWeighted Average (exponential decay):")
    print(f"  Portfolio Hit Rate: {weighted['portfolio_hit_rate']:.2%}")
    print(f"  Alpha: {weighted['avg_alpha']:.4f}")

    hit_diff = abs(unanimous['portfolio_hit_rate'] - weighted['portfolio_hit_rate'])

    print(f"\nDifference: {hit_diff:.2%}")

    if hit_diff > 0.15:
        print("  WARNING: HIGH SENSITIVITY - Weighting scheme matters a lot")
        print("    -> Strategy may be overfit to weighting choice")
    else:
        print("  OK: LOW SENSITIVITY - Weighting scheme doesn't matter much")
        print("    -> Robust to weight configuration")


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_sensitivity(n_sats_results, consensus_results, output_dir='data/feature_analysis'):
    """Create visualizations of parameter sensitivity."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    output_dir = Path(output_dir)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold', color='white')

    # Plot 1: N_SATELLITES vs Alpha
    if n_sats_results is not None:
        ax = axes[0, 0]
        ax.plot(n_sats_results['n_satellites'], n_sats_results['avg_alpha'] * 100,
                marker='o', linewidth=2, markersize=8, color='skyblue')
        ax.axvline(x=4, color='red', linestyle='--', alpha=0.5, label='Current (N=4)')
        ax.set_title('Alpha vs N_SATELLITES', fontweight='bold', color='white')
        ax.set_xlabel('Number of Satellites')
        ax.set_ylabel('Alpha (%)')
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 2: N_SATELLITES vs Hit Rate
        ax = axes[0, 1]
        ax.plot(n_sats_results['n_satellites'], n_sats_results['portfolio_hit_rate'] * 100,
                marker='o', linewidth=2, markersize=8, color='lightcoral')
        ax.axvline(x=4, color='red', linestyle='--', alpha=0.5, label='Current (N=4)')
        ax.axhline(y=50, color='yellow', linestyle=':', alpha=0.5, label='Random (50%)')
        ax.set_title('Portfolio Hit Rate vs N_SATELLITES', fontweight='bold', color='white')
        ax.set_xlabel('Number of Satellites')
        ax.set_ylabel('Hit Rate (%)')
        ax.legend()
        ax.grid(alpha=0.3)

    # Plot 3: Consensus Method Comparison (Alpha)
    if consensus_results is not None:
        ax = axes[1, 0]
        methods = consensus_results['method'].values
        alphas = consensus_results['avg_alpha'].values * 100

        colors = ['green' if m == 'unanimous' else 'skyblue' for m in methods]
        bars = ax.bar(range(len(methods)), alphas, color=colors, edgecolor='black', alpha=0.7)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_title('Alpha by Consensus Method', fontweight='bold', color='white')
        ax.set_ylabel('Alpha (%)')
        ax.grid(axis='y', alpha=0.3)

        # Plot 4: Consensus Method Comparison (Hit Rate)
        ax = axes[1, 1]
        hit_rates = consensus_results['portfolio_hit_rate'].values * 100

        colors = ['green' if m == 'unanimous' else 'lightcoral' for m in methods]
        bars = ax.bar(range(len(methods)), hit_rates, color=colors, edgecolor='black', alpha=0.7)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.axhline(y=50, color='yellow', linestyle=':', linewidth=2, label='Random')
        ax.set_title('Portfolio Hit Rate by Consensus Method', fontweight='bold', color='white')
        ax.set_ylabel('Hit Rate (%)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_file = output_dir / 'parameter_sensitivity.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
    print(f"\n[SAVED] {plot_file}")

    plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Run parameter sensitivity analysis."""
    print("\n" + "="*60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*60)

    # Test 1: Consensus method sensitivity
    consensus_results = analyze_consensus_methods()

    # Test 2: N_SATELLITES sensitivity
    n_sats_results = simulate_n_satellites_sensitivity()

    # Test 3: Horizon weight sensitivity
    horizon_weight_sensitivity()

    # Visualize
    if n_sats_results is not None and consensus_results is not None:
        visualize_sensitivity(n_sats_results, consensus_results)

    # Final summary
    print("\n" + "="*60)
    print("SENSITIVITY SUMMARY")
    print("="*60)

    print("\nKey findings:")
    print("  1. Consensus method: Check if 'unanimous' significantly outperforms others")
    print("  2. N_SATELLITES: Check if performance is stable around N=4")
    print("  3. Horizon weights: Check if weighting scheme matters")

    print("\nWARNING: If performance is ONLY good at specific parameter values,")
    print("this suggests overfitting to those parameters.")

    print("\nGOOD SIGN: If performance is relatively stable across reasonable")
    print("parameter ranges, the strategy is more robust.")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
