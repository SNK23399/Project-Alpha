"""
Step 10: Temporal Stability Check for UNANIMOUS CONSENSUS
==========================================================

Script 7 analyzed RAW alpha (all ETFs), which is misleading.
This script analyzes the ACTUAL unanimous consensus selections over time.
"""

import pandas as pd
import numpy as np
from pathlib import Path
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


def load_unanimous_selections():
    """Load the actual unanimous consensus selections from Script 5."""
    # We need to reconstruct this from the multi-horizon data
    import numpy as np

    def load_ensemble_rankings(holding_months, output_dir='data/feature_analysis'):
        """Load ensemble rankings for a specific horizon."""
        file = Path(output_dir) / f'rankings_matrix_{holding_months}month.npz'

        if not file.exists():
            return None

        data = np.load(file, allow_pickle=True)
        rankings = data['rankings']
        dates = pd.to_datetime(data['dates'])
        isins = data['isins']
        features = data['features']

        # Load ensemble
        ensemble_file = Path(output_dir) / f'ensemble_{holding_months}month.csv'
        if not ensemble_file.exists():
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
            return None

        # Extract ensemble rankings
        ensemble_rankings = rankings[:, :, feature_indices]

        # Average across ensemble features to get score
        ensemble_scores = np.nanmean(ensemble_rankings, axis=2)

        # Convert to DataFrame
        rows = []
        for date_idx, date in enumerate(dates):
            for isin_idx, isin in enumerate(isins):
                score = ensemble_scores[date_idx, isin_idx]
                if not np.isnan(score):
                    rows.append({
                        'date': date,
                        'isin': isin,
                        f'score_{holding_months}m': score
                    })

        return pd.DataFrame(rows)

    def merge_horizons(horizons):
        """Merge rankings from multiple horizons."""
        all_dfs = []

        for horizon in horizons:
            df = load_ensemble_rankings(horizon)
            if df is not None:
                all_dfs.append(df)

        if len(all_dfs) == 0:
            return None

        # Merge all horizons
        merged = all_dfs[0]
        for df in all_dfs[1:]:
            merged = pd.merge(merged, df, on=['date', 'isin'], how='inner')

        return merged

    def select_unanimous(date_df, n_satellites, horizons):
        """Select ETFs using unanimous consensus."""
        # All horizons must agree (top N in all horizons)
        top_sets = []
        for horizon in horizons:
            col = f'score_{horizon}m'
            if col in date_df.columns:
                top_n = date_df.nlargest(n_satellites * 2, col)['isin'].tolist()
                top_sets.append(set(top_n))

        # Intersection of all sets
        if len(top_sets) > 0:
            unanimous = set.intersection(*top_sets)
            selected = list(unanimous)[:n_satellites]

            # If not enough unanimous, fill with primary (1-month)
            if len(selected) < n_satellites:
                primary_top = date_df.nlargest(n_satellites, 'score_1m')
                for isin in primary_top['isin']:
                    if isin not in selected:
                        selected.append(isin)
                        if len(selected) >= n_satellites:
                            break
        else:
            selected = []

        return selected

    print("Loading unanimous consensus selections...")
    horizons = list(range(1, 13))
    merged_df = merge_horizons(horizons)

    if merged_df is None:
        return None

    n_satellites = 4
    selections = []

    for date in merged_df['date'].unique():
        date_df = merged_df[merged_df['date'] == date].copy()
        selected = select_unanimous(date_df, n_satellites, horizons)

        for isin in selected:
            selections.append({
                'date': date,
                'isin': isin
            })

    return pd.DataFrame(selections)


def analyze_temporal_stability():
    """Analyze temporal stability of unanimous consensus."""
    print("\n" + "="*60)
    print("TEMPORAL STABILITY ANALYSIS - UNANIMOUS CONSENSUS")
    print("="*60)

    # Load selections
    selections_df = load_unanimous_selections()

    if selections_df is None:
        print("ERROR: Could not load selections")
        return

    # Load forward alpha
    alpha_file = Path('data/feature_analysis/forward_alpha_1month.parquet')
    alpha_df = pd.read_parquet(alpha_file)

    # Merge
    merged = pd.merge(selections_df, alpha_df, on=['date', 'isin'], how='inner')

    print(f"\nTotal selections: {len(merged)}")
    print(f"Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")

    # Overall performance
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Alpha: {merged['forward_alpha'].mean():.4f} ({merged['forward_alpha'].mean()*100:.2f}%)")
    print(f"  Hit rate: {(merged['forward_alpha'] > 0).mean():.2%}")

    date_alphas = merged.groupby('date')['forward_alpha'].mean()
    portfolio_hit_rate = (date_alphas > 0).mean()
    print(f"  Portfolio hit rate: {portfolio_hit_rate:.2%}")

    # Year-by-year analysis
    print(f"\n" + "-"*60)
    print("YEAR-BY-YEAR PERFORMANCE")
    print("-"*60)

    merged['year'] = merged['date'].dt.year

    yearly_stats = []
    for year in sorted(merged['year'].unique()):
        year_data = merged[merged['year'] == year]

        alpha = year_data['forward_alpha'].mean()
        hit_rate = (year_data['forward_alpha'] > 0).mean()

        year_date_alphas = year_data.groupby('date')['forward_alpha'].mean()
        portfolio_hit = (year_date_alphas > 0).mean()

        yearly_stats.append({
            'year': year,
            'alpha': alpha,
            'hit_rate': hit_rate,
            'portfolio_hit_rate': portfolio_hit,
            'n_selections': len(year_data),
            'n_dates': year_data['date'].nunique()
        })

        print(f"{year}: Alpha={alpha:+.4f} ({alpha*100:+.2f}%), "
              f"Portfolio Hit={portfolio_hit:.1%}, N={len(year_data)}")

    yearly_df = pd.DataFrame(yearly_stats)

    # First half vs second half
    print(f"\n" + "-"*60)
    print("FIRST HALF VS SECOND HALF")
    print("-"*60)

    dates = sorted(merged['date'].unique())
    mid_point = len(dates) // 2

    first_half_dates = dates[:mid_point]
    second_half_dates = dates[mid_point:]

    first_half = merged[merged['date'].isin(first_half_dates)]
    second_half = merged[merged['date'].isin(second_half_dates)]

    print(f"\nFirst half ({first_half_dates[0].date()} to {first_half_dates[-1].date()}):")
    print(f"  Alpha: {first_half['forward_alpha'].mean():.4f} ({first_half['forward_alpha'].mean()*100:.2f}%)")
    print(f"  Hit rate: {(first_half['forward_alpha'] > 0).mean():.2%}")

    first_date_alphas = first_half.groupby('date')['forward_alpha'].mean()
    first_portfolio_hit = (first_date_alphas > 0).mean()
    print(f"  Portfolio hit rate: {first_portfolio_hit:.2%}")

    print(f"\nSecond half ({second_half_dates[0].date()} to {second_half_dates[-1].date()}):")
    print(f"  Alpha: {second_half['forward_alpha'].mean():.4f} ({second_half['forward_alpha'].mean()*100:.2f}%)")
    print(f"  Hit rate: {(second_half['forward_alpha'] > 0).mean():.2%}")

    second_date_alphas = second_half.groupby('date')['forward_alpha'].mean()
    second_portfolio_hit = (second_date_alphas > 0).mean()
    print(f"  Portfolio hit rate: {second_portfolio_hit:.2%}")

    # Calculate change
    alpha_change = (second_half['forward_alpha'].mean() - first_half['forward_alpha'].mean()) / first_half['forward_alpha'].mean() * 100
    hit_change = (second_portfolio_hit - first_portfolio_hit) / first_portfolio_hit * 100

    print(f"\nChange:")
    print(f"  Alpha: {alpha_change:+.1f}%")
    print(f"  Portfolio hit rate: {hit_change:+.1f}%")

    # Interpretation
    print(f"\n" + "-"*60)
    print("INTERPRETATION")
    print("-"*60)

    if abs(alpha_change) < 20 and abs(hit_change) < 20:
        print("\nOK: Performance is STABLE between halves")
        print("  -> No evidence of degradation or overfitting")
    elif alpha_change < -30 or hit_change < -20:
        print("\nWARNING: Performance DEGRADED in second half")
        print("  -> Possible overfitting to early period")
    else:
        print("\nMODERATE: Some performance variation between halves")
        print("  -> Monitor but not alarming")

    # Trend analysis
    from scipy.stats import linregress

    years = yearly_df['year'].values
    alphas = yearly_df['alpha'].values
    hits = yearly_df['portfolio_hit_rate'].values

    slope_alpha, intercept_alpha, r_alpha, p_alpha, se_alpha = linregress(years, alphas)
    slope_hit, intercept_hit, r_hit, p_hit, se_hit = linregress(years, hits)

    print(f"\n" + "-"*60)
    print("TREND ANALYSIS")
    print("-"*60)

    print(f"\nAlpha trend:")
    print(f"  Slope: {slope_alpha:.6f} per year")
    print(f"  p-value: {p_alpha:.4f}")

    if p_alpha < 0.05:
        if slope_alpha < 0:
            print("  WARNING: Significant negative trend")
        else:
            print("  OK: Significant positive trend")
    else:
        print("  OK: No significant trend (stable)")

    print(f"\nPortfolio hit rate trend:")
    print(f"  Slope: {slope_hit:.6f} per year")
    print(f"  p-value: {p_hit:.4f}")

    if p_hit < 0.05:
        if slope_hit < 0:
            print("  WARNING: Significant negative trend")
        else:
            print("  OK: Significant positive trend")
    else:
        print("  OK: No significant trend (stable)")

    # Visualization
    print(f"\n" + "-"*60)
    print("CREATING VISUALIZATION")
    print("-"*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Temporal Stability - Unanimous Consensus',
                 fontsize=16, fontweight='bold', color='white')

    # Plot 1: Alpha over time
    ax = axes[0, 0]
    ax.plot(yearly_df['year'], yearly_df['alpha'] * 100,
            marker='o', linewidth=2, markersize=8, color='skyblue')
    ax.axhline(y=0, color='yellow', linestyle='--', alpha=0.5)
    ax.set_title('Alpha by Year', fontweight='bold', color='white')
    ax.set_xlabel('Year')
    ax.set_ylabel('Alpha (%)')
    ax.grid(alpha=0.3)

    # Plot 2: Portfolio hit rate over time
    ax = axes[0, 1]
    ax.plot(yearly_df['year'], yearly_df['portfolio_hit_rate'] * 100,
            marker='o', linewidth=2, markersize=8, color='lightcoral')
    ax.axhline(y=50, color='yellow', linestyle='--', alpha=0.5, label='Random (50%)')
    ax.set_title('Portfolio Hit Rate by Year', fontweight='bold', color='white')
    ax.set_xlabel('Year')
    ax.set_ylabel('Hit Rate (%)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Rolling 12-month alpha
    ax = axes[1, 0]
    rolling_alpha = merged.set_index('date')['forward_alpha'].rolling(window=12).mean()
    ax.plot(rolling_alpha.index, rolling_alpha * 100, linewidth=2, color='skyblue')
    ax.axhline(y=0, color='yellow', linestyle='--', alpha=0.5)
    ax.set_title('Rolling 12-Month Alpha', fontweight='bold', color='white')
    ax.set_xlabel('Date')
    ax.set_ylabel('Alpha (%)')
    ax.grid(alpha=0.3)

    # Plot 4: Cumulative alpha
    ax = axes[1, 1]
    date_alphas_sorted = date_alphas.sort_index()
    cumulative = (1 + date_alphas_sorted).cumprod()
    ax.plot(cumulative.index, (cumulative - 1) * 100, linewidth=2, color='lightgreen')
    ax.axhline(y=0, color='yellow', linestyle='--', alpha=0.5)
    ax.set_title('Cumulative Alpha', fontweight='bold', color='white')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Alpha (%)')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    output_file = Path('data/feature_analysis/temporal_stability_unanimous.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
    print(f"\n[SAVED] {output_file}")

    plt.show()

    # Save yearly stats
    yearly_file = Path('data/feature_analysis/yearly_performance_unanimous.csv')
    yearly_df.to_csv(yearly_file, index=False)
    print(f"[SAVED] {yearly_file}")

    return yearly_df


def main():
    """Run temporal stability analysis."""
    yearly_stats = analyze_temporal_stability()

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)

    print("\nThis analysis uses ACTUAL unanimous consensus selections,")
    print("not raw alpha of all ETFs (which Script 7 mistakenly analyzed).")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
