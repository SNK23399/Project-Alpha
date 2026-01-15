"""
Multi-Horizon Consensus Strategy

This script tests whether combining signals from multiple holding periods
can enhance the 1-month strategy's performance by:
1. Using 1-month ensemble as primary signal
2. Requiring confirmation from longer-term horizons (3, 6, 12 months)
3. Testing different consensus mechanisms (unanimous, majority, weighted)

Goal: Boost 1-month strategy's already excellent 96% hit rate and 44% annualized alpha.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Import our support modules
import sys
sys.path.append('support')
from etf_database import ETFDatabase
from signal_database import SignalDatabase


# ============================================================
# CONFIGURATION
# ============================================================

# Core ETF (benchmark)
CORE_ISIN = 'IE00B4L5Y983'  # iShares Core MSCI World

# Horizons to use for consensus
PRIMARY_HORIZON = 1  # Primary signal (1 month)
CONFIRMATION_HORIZONS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Longer-term confirmation

# Number of satellites to select
N_SATELLITES = 4

# Consensus mechanisms to test
CONSENSUS_METHODS = [
    'primary_only',      # Baseline: just use 1-month
    'unanimous',         # All horizons must agree
    'majority',          # At least 50% of horizons must agree
    'weighted_avg',      # Weighted average of rankings
    'primary_veto',      # Use 1-month but veto if 2+ longer horizons disagree
]

# Weights for weighted average (if using weighted_avg method)
# Exponentially decaying weights: shorter horizons get more weight
HORIZON_WEIGHTS = {
    1: 0.30,   # 30% weight on 1-month (strongest signal)
    2: 0.20,   # 20% weight on 2-month
    3: 0.15,   # 15% weight on 3-month
    4: 0.10,   # 10% weight on 4-month
    5: 0.07,   # 7% weight on 5-month
    6: 0.05,   # 5% weight on 6-month
    7: 0.04,   # 4% weight on 7-month
    8: 0.03,   # 3% weight on 8-month
    9: 0.02,   # 2% weight on 9-month
    10: 0.015, # 1.5% weight on 10-month
    11: 0.015, # 1.5% weight on 11-month
    12: 0.02,  # 2% weight on 12-month
}  # Total = 100%


# ============================================================
# STEP 1: LOAD ENSEMBLE RANKINGS FOR ALL HORIZONS
# ============================================================

def load_ensemble_rankings(holding_months):
    """
    Load the ensemble features and compute rankings for all ETFs.

    Returns:
        DataFrame with columns: date, isin, ensemble_score (0-1)
    """
    print(f"\nLoading {holding_months}-month ensemble...")

    # Load ensemble features
    ensemble_file = Path('data/feature_analysis') / f'ensemble_{holding_months}month.csv'
    if not ensemble_file.exists():
        print(f"  WARNING: Ensemble file not found: {ensemble_file}")
        return None

    ensemble_df = pd.read_csv(ensemble_file)
    feature_names = ensemble_df['feature_name'].tolist()
    print(f"  Features: {len(feature_names)}")

    # Load rankings matrix
    matrix_file = Path('data/feature_analysis') / f'rankings_matrix_{holding_months}month.npz'
    if not matrix_file.exists():
        print(f"  WARNING: Rankings matrix not found: {matrix_file}")
        return None

    data = np.load(matrix_file, allow_pickle=True)
    rankings = data['rankings']
    dates = pd.to_datetime(data['dates'])
    isins = data['isins']
    all_features = data['features']

    # Find indices of selected features
    feature_indices = []
    for feat_name in feature_names:
        try:
            idx = list(all_features).index(feat_name)
            feature_indices.append(idx)
        except ValueError:
            print(f"  WARNING: Feature not found: {feat_name}")

    if len(feature_indices) == 0:
        print(f"  ERROR: No features found!")
        return None

    # Compute ensemble score (average of selected features)
    ensemble_rankings = rankings[:, :, feature_indices]
    ensemble_scores = np.nanmean(ensemble_rankings, axis=2)

    # Convert to DataFrame
    results = []
    for date_idx, date in enumerate(dates):
        for isin_idx, isin in enumerate(isins):
            score = ensemble_scores[date_idx, isin_idx]
            if not np.isnan(score):
                results.append({
                    'date': date,
                    'isin': isin,
                    f'score_{holding_months}m': score
                })

    df = pd.DataFrame(results)
    print(f"  Loaded {len(df)} rankings")

    return df


def load_all_horizons():
    """Load ensemble rankings for all horizons."""
    print("="*60)
    print("LOADING ENSEMBLE RANKINGS FOR ALL HORIZONS")
    print("="*60)

    horizons = [PRIMARY_HORIZON] + CONFIRMATION_HORIZONS
    all_dfs = []

    for horizon in horizons:
        df = load_ensemble_rankings(horizon)
        if df is not None:
            all_dfs.append(df)

    if len(all_dfs) == 0:
        raise ValueError("No ensemble rankings loaded!")

    # Merge all horizons
    print("\nMerging all horizons...")
    merged = all_dfs[0]
    for df in all_dfs[1:]:
        merged = pd.merge(merged, df, on=['date', 'isin'], how='inner')

    print(f"Merged data: {len(merged)} rows")
    print(f"Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
    print(f"Unique ISINs: {merged['isin'].nunique()}")

    return merged


# ============================================================
# STEP 2: APPLY CONSENSUS MECHANISMS
# ============================================================

def apply_consensus(df, method):
    """
    Apply consensus mechanism to select top N satellites.

    Args:
        df: DataFrame with scores for all horizons
        method: Consensus method name

    Returns:
        DataFrame with selected ISINs per date
    """
    print(f"\nApplying consensus method: {method}")

    results = []

    for date in df['date'].unique():
        date_df = df[df['date'] == date].copy()

        if method == 'primary_only':
            # Just use 1-month rankings
            date_df = date_df.sort_values(f'score_{PRIMARY_HORIZON}m', ascending=False)
            selected = date_df.head(N_SATELLITES)['isin'].tolist()

        elif method == 'unanimous':
            # All horizons must agree (top N in all horizons)
            # Find ETFs that are top N in ALL horizons
            top_sets = []
            for horizon in [PRIMARY_HORIZON] + CONFIRMATION_HORIZONS:
                col = f'score_{horizon}m'
                if col in date_df.columns:
                    top_n = date_df.nlargest(N_SATELLITES * 2, col)['isin'].tolist()
                    top_sets.append(set(top_n))

            # Intersection of all sets
            if len(top_sets) > 0:
                unanimous = set.intersection(*top_sets)
                selected = list(unanimous)[:N_SATELLITES]

                # If not enough unanimous, fill with primary
                if len(selected) < N_SATELLITES:
                    primary_top = date_df.nlargest(N_SATELLITES, f'score_{PRIMARY_HORIZON}m')
                    for isin in primary_top['isin']:
                        if isin not in selected:
                            selected.append(isin)
                            if len(selected) >= N_SATELLITES:
                                break
            else:
                selected = []

        elif method == 'majority':
            # Count how many horizons have each ETF in top N
            vote_counts = {}
            for horizon in [PRIMARY_HORIZON] + CONFIRMATION_HORIZONS:
                col = f'score_{horizon}m'
                if col in date_df.columns:
                    top_n = date_df.nlargest(N_SATELLITES * 2, col)['isin'].tolist()
                    for isin in top_n:
                        vote_counts[isin] = vote_counts.get(isin, 0) + 1

            # Sort by vote count, then by primary score
            date_df['votes'] = date_df['isin'].map(vote_counts).fillna(0)
            date_df = date_df.sort_values(
                ['votes', f'score_{PRIMARY_HORIZON}m'],
                ascending=[False, False]
            )
            selected = date_df.head(N_SATELLITES)['isin'].tolist()

        elif method == 'weighted_avg':
            # Weighted average of all horizon scores
            date_df['weighted_score'] = 0
            for horizon, weight in HORIZON_WEIGHTS.items():
                col = f'score_{horizon}m'
                if col in date_df.columns:
                    date_df['weighted_score'] += date_df[col] * weight

            date_df = date_df.sort_values('weighted_score', ascending=False)
            selected = date_df.head(N_SATELLITES)['isin'].tolist()

        elif method == 'primary_veto':
            # Use primary, but veto if 2+ longer horizons strongly disagree
            date_df_sorted = date_df.sort_values(f'score_{PRIMARY_HORIZON}m', ascending=False)
            selected = []

            for isin in date_df_sorted['isin']:
                # Check if this ETF is in bottom half of 2+ longer horizons
                disagreements = 0
                for horizon in CONFIRMATION_HORIZONS:
                    col = f'score_{horizon}m'
                    if col in date_df.columns:
                        rank_pct = date_df[col].rank(pct=True)
                        etf_rank = rank_pct[date_df['isin'] == isin].iloc[0]
                        if etf_rank < 0.5:  # Bottom half
                            disagreements += 1

                # If 2+ horizons disagree, veto
                if disagreements < 2:
                    selected.append(isin)
                    if len(selected) >= N_SATELLITES:
                        break

            # If not enough after vetoes, fill with primary
            if len(selected) < N_SATELLITES:
                for isin in date_df_sorted['isin']:
                    if isin not in selected:
                        selected.append(isin)
                        if len(selected) >= N_SATELLITES:
                            break

        else:
            raise ValueError(f"Unknown consensus method: {method}")

        # Store results
        for isin in selected:
            results.append({
                'date': date,
                'isin': isin,
                'selected': True
            })

    selection_df = pd.DataFrame(results)
    print(f"  Selected {len(selection_df)} total (ETF, date) pairs")
    print(f"  Average per date: {len(selection_df) / df['date'].nunique():.1f}")

    return selection_df


# ============================================================
# STEP 3: EVALUATE PERFORMANCE
# ============================================================

def evaluate_performance(selections_df, alpha_df, method_name):
    """
    Evaluate performance of selected satellites.

    Args:
        selections_df: DataFrame with selected (date, isin) pairs
        alpha_df: Forward alpha data
        method_name: Name of consensus method

    Returns:
        Dict with performance metrics
    """
    # Merge selections with alpha
    merged = pd.merge(
        selections_df,
        alpha_df,
        on=['date', 'isin'],
        how='inner'
    )

    if len(merged) == 0:
        print(f"  WARNING: No matches found for {method_name}")
        return None

    # Calculate metrics
    avg_alpha = merged['forward_alpha'].mean()
    std_alpha = merged['forward_alpha'].std()

    # Hit rate (% positive alpha)
    hit_rate = (merged['forward_alpha'] > 0).mean()

    # Group by date to get portfolio-level metrics
    date_alphas = merged.groupby('date')['forward_alpha'].mean()

    # Portfolio hit rate (% of dates with positive avg alpha)
    portfolio_hit_rate = (date_alphas > 0).mean()

    sharpe = avg_alpha / std_alpha if std_alpha > 0 else 0

    return {
        'method': method_name,
        'n_selections': len(merged),
        'n_periods': len(date_alphas),
        'avg_alpha': avg_alpha,
        'std_alpha': std_alpha,
        'sharpe': sharpe,
        'hit_rate': hit_rate,
        'portfolio_hit_rate': portfolio_hit_rate,
        'date_alphas': date_alphas
    }


# ============================================================
# STEP 4: COMPARE ALL METHODS
# ============================================================

def compare_all_methods(rankings_df):
    """Compare all consensus methods."""
    print("\n" + "="*60)
    print("COMPARING CONSENSUS METHODS")
    print("="*60)

    # Load 1-month forward alpha for evaluation
    alpha_file = Path('data/feature_analysis') / 'forward_alpha_1month.parquet'
    alpha_df = pd.read_parquet(alpha_file)

    # Test each method
    results = []

    for method in CONSENSUS_METHODS:
        print(f"\n{'='*60}")
        print(f"METHOD: {method}")
        print(f"{'='*60}")

        # Apply consensus
        selections = apply_consensus(rankings_df, method)

        # Evaluate performance
        perf = evaluate_performance(selections, alpha_df, method)

        if perf is not None:
            results.append(perf)

            # Print summary
            print(f"\n{method.upper()} RESULTS:")
            print(f"  Avg Alpha: {perf['avg_alpha']:.4f} ({perf['avg_alpha']*100:.2f}%)")
            print(f"  Hit Rate (individual): {perf['hit_rate']:.2%}")
            print(f"  Hit Rate (portfolio): {perf['portfolio_hit_rate']:.2%}")
            print(f"  Sharpe: {perf['sharpe']:.3f}")
            print(f"  Std: {perf['std_alpha']:.4f}")
            print(f"  Periods: {perf['n_periods']}")

    return results


# ============================================================
# STEP 5: VISUALIZE RESULTS
# ============================================================

def visualize_comparison(results):
    """Create comparison visualizations."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    # Create summary DataFrame
    summary_df = pd.DataFrame([{
        'method': r['method'],
        'avg_alpha': r['avg_alpha'],
        'hit_rate': r['hit_rate'],
        'portfolio_hit_rate': r['portfolio_hit_rate'],
        'sharpe': r['sharpe']
    } for r in results])

    # Sort by alpha
    summary_df = summary_df.sort_values('avg_alpha', ascending=False)

    print("\nSUMMARY TABLE:")
    print(summary_df.to_string(index=False))

    # Save summary
    output_dir = Path('data/feature_analysis')
    summary_file = output_dir / 'multi_horizon_consensus_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\n[SAVED] {summary_file}")

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-Horizon Consensus Comparison', fontsize=16, fontweight='bold')

    methods = summary_df['method'].tolist()

    # Plot 1: Average Alpha
    ax = axes[0, 0]
    bars = ax.bar(methods, summary_df['avg_alpha'] * 100)
    ax.set_title('Average Alpha (Monthly %)', fontweight='bold')
    ax.set_ylabel('Alpha (%)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Highlight best
    best_idx = summary_df['avg_alpha'].argmax()
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(0.7)

    # Plot 2: Hit Rate
    ax = axes[0, 1]
    bars = ax.bar(methods, summary_df['hit_rate'] * 100)
    ax.set_title('Hit Rate (Individual Selections)', fontweight='bold')
    ax.set_ylabel('Hit Rate (%)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    ax.legend()

    # Highlight best
    best_idx = summary_df['hit_rate'].argmax()
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(0.7)

    # Plot 3: Portfolio Hit Rate
    ax = axes[1, 0]
    bars = ax.bar(methods, summary_df['portfolio_hit_rate'] * 100)
    ax.set_title('Portfolio Hit Rate (Avg per Date)', fontweight='bold')
    ax.set_ylabel('Hit Rate (%)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    ax.legend()

    # Highlight best
    best_idx = summary_df['portfolio_hit_rate'].argmax()
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(0.7)

    # Plot 4: Sharpe Ratio
    ax = axes[1, 1]
    bars = ax.bar(methods, summary_df['sharpe'])
    ax.set_title('Sharpe Ratio', fontweight='bold')
    ax.set_ylabel('Sharpe')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Highlight best
    best_idx = summary_df['sharpe'].argmax()
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(0.7)

    plt.tight_layout()

    # Save plot
    plot_file = output_dir / 'multi_horizon_consensus_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {plot_file}")

    plt.show()

    # Plot cumulative alpha over time for each method
    fig, ax = plt.subplots(figsize=(14, 7))

    for result in results:
        date_alphas = result['date_alphas']
        cumulative = (1 + date_alphas).cumprod() - 1
        ax.plot(cumulative.index, cumulative.values * 100,
                label=result['method'], linewidth=2, alpha=0.8)

    ax.set_title('Cumulative Alpha Over Time (1-Month Horizon)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Alpha (%)')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_file = output_dir / 'multi_horizon_consensus_cumulative.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {plot_file}")

    plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Run the multi-horizon consensus analysis."""
    print("\n" + "="*60)
    print("MULTI-HORIZON CONSENSUS STRATEGY")
    print("="*60)
    print(f"\nPrimary Horizon: {PRIMARY_HORIZON} month")
    print(f"Confirmation Horizons: {CONFIRMATION_HORIZONS}")
    print(f"Number of Satellites: {N_SATELLITES}")
    print(f"Consensus Methods: {len(CONSENSUS_METHODS)}")

    # Step 1: Load ensemble rankings
    rankings_df = load_all_horizons()

    # Step 2-3: Compare all methods
    results = compare_all_methods(rankings_df)

    # Step 4: Visualize
    if len(results) > 0:
        visualize_comparison(results)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    # Print final recommendation
    if len(results) > 0:
        best_alpha = max(results, key=lambda x: x['avg_alpha'])
        best_hit_rate = max(results, key=lambda x: x['hit_rate'])
        best_sharpe = max(results, key=lambda x: x['sharpe'])

        print("\nRECOMMENDATIONS:")
        print(f"  Best Alpha: {best_alpha['method']} ({best_alpha['avg_alpha']*100:.2f}%)")
        print(f"  Best Hit Rate: {best_hit_rate['method']} ({best_hit_rate['hit_rate']:.2%})")
        print(f"  Best Sharpe: {best_sharpe['method']} ({best_sharpe['sharpe']:.3f})")


if __name__ == '__main__':
    main()
