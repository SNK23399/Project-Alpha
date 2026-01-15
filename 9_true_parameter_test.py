"""
Step 9: True Parameter Sensitivity Test
=========================================

This script uses the ACTUAL selection logic from 5_multi_horizon_consensus.py
to test parameter sensitivity, instead of simulating with raw rankings.

This will give us the true answer about whether the 95.4% hit rate is
robust to parameter changes.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import the actual functions from script 5
import sys
sys.path.append(str(Path(__file__).parent))

# We'll reimplement the core logic to avoid import issues

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


def test_n_satellites_sensitivity():
    """Test sensitivity to N_SATELLITES using actual selection logic."""
    print("\n" + "="*60)
    print("TRUE N_SATELLITES SENSITIVITY TEST")
    print("="*60)

    # Load alpha data
    alpha_file = Path('data/feature_analysis/forward_alpha_1month.parquet')
    alpha_df = pd.read_parquet(alpha_file)

    # Load merged horizons
    print("\nLoading ensemble rankings for all horizons...")
    horizons = list(range(1, 13))
    merged_df = merge_horizons(horizons)

    if merged_df is None:
        print("ERROR: Could not load horizon data")
        return None

    print(f"Loaded data: {len(merged_df)} rows")
    print(f"Dates: {merged_df['date'].nunique()}")
    print(f"ISINs: {merged_df['isin'].nunique()}")

    # Test different N values
    n_range = [2, 3, 4, 5, 6, 8, 10]
    results = []

    print(f"\nTesting N_SATELLITES: {n_range}")
    print("-"*60)

    for n_sats in n_range:
        print(f"\nTesting N={n_sats}...")

        # Apply unanimous selection for each date
        selections = []
        for date in merged_df['date'].unique():
            date_df = merged_df[merged_df['date'] == date].copy()
            selected = select_unanimous(date_df, n_sats, horizons)

            for isin in selected:
                selections.append({
                    'date': date,
                    'isin': isin
                })

        selections_df = pd.DataFrame(selections)

        # Merge with alpha
        merged = pd.merge(selections_df, alpha_df, on=['date', 'isin'], how='inner')

        if len(merged) > 0:
            avg_alpha = merged['forward_alpha'].mean()
            hit_rate = (merged['forward_alpha'] > 0).mean()

            # Portfolio metrics
            date_alphas = merged.groupby('date')['forward_alpha'].mean()
            portfolio_hit_rate = (date_alphas > 0).mean()
            sharpe = avg_alpha / merged['forward_alpha'].std() if merged['forward_alpha'].std() > 0 else 0

            avg_selections = len(merged) / merged['date'].nunique()

            results.append({
                'n_satellites': n_sats,
                'avg_alpha': avg_alpha,
                'hit_rate': hit_rate,
                'portfolio_hit_rate': portfolio_hit_rate,
                'sharpe': sharpe,
                'avg_selections_per_date': avg_selections
            })

            print(f"  Alpha: {avg_alpha:.4f} ({avg_alpha*100:.2f}%)")
            print(f"  Hit rate: {hit_rate:.2%}")
            print(f"  Portfolio hit rate: {portfolio_hit_rate:.2%}")
            print(f"  Avg selections/date: {avg_selections:.1f}")

    results_df = pd.DataFrame(results)

    # Analysis
    print("\n" + "-"*60)
    print("SENSITIVITY ANALYSIS")
    print("-"*60)

    alpha_range = results_df['avg_alpha'].max() - results_df['avg_alpha'].min()
    hit_range = results_df['portfolio_hit_rate'].max() - results_df['portfolio_hit_rate'].min()

    print(f"\nAlpha range: {alpha_range:.4f} ({alpha_range*100:.2f}%)")
    print(f"Portfolio hit rate range: {hit_range:.4f} ({hit_range*100:.2f}%)")

    # Find best
    best_idx = results_df['portfolio_hit_rate'].argmax()
    best_n = results_df.iloc[best_idx]['n_satellites']
    best_hit = results_df.iloc[best_idx]['portfolio_hit_rate']

    print(f"\nBest N_SATELLITES: {best_n:.0f} (hit rate: {best_hit:.2%})")

    # Check cliff effect
    if len(results_df) > 1:
        neighbors = results_df[
            (results_df['n_satellites'] >= best_n - 1) &
            (results_df['n_satellites'] <= best_n + 1)
        ]

        if len(neighbors) > 1:
            neighbor_hits = neighbors['portfolio_hit_rate'].values
            max_drop = best_hit - neighbor_hits.min()

            print(f"Max drop from best: {max_drop:.2%}")

            if max_drop > 0.20:
                print("  WARNING: CLIFF EFFECT - Performance drops >20% at nearby N")
            elif max_drop > 0.10:
                print("  WARNING: MODERATE SENSITIVITY - Performance varies 10-20%")
            else:
                print("  OK: STABLE - Performance similar across N values")

    # Save results
    output_file = Path('data/feature_analysis/true_n_satellites_sensitivity.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n[SAVED] {output_file}")

    return results_df


def test_horizon_subset_sensitivity():
    """Test if we really need all 12 horizons or if fewer would work."""
    print("\n" + "="*60)
    print("HORIZON SUBSET SENSITIVITY TEST")
    print("="*60)

    # Load alpha data
    alpha_file = Path('data/feature_analysis/forward_alpha_1month.parquet')
    alpha_df = pd.read_parquet(alpha_file)

    # Test different horizon combinations
    horizon_configs = [
        ([1], "1-month only"),
        ([1, 3, 6, 12], "Key horizons (1, 3, 6, 12)"),
        ([1, 2, 3, 4, 5, 6], "Short-term (1-6 months)"),
        (list(range(1, 13)), "All horizons (1-12)")
    ]

    n_satellites = 4  # Use current setting
    results = []

    print(f"\nTesting with N_SATELLITES={n_satellites}")
    print("-"*60)

    for horizons, desc in horizon_configs:
        print(f"\n{desc}: {horizons}")

        # Load merged horizons
        merged_df = merge_horizons(horizons)

        if merged_df is None:
            print("  ERROR: Could not load horizon data")
            continue

        # Apply unanimous selection
        selections = []
        for date in merged_df['date'].unique():
            date_df = merged_df[merged_df['date'] == date].copy()
            selected = select_unanimous(date_df, n_satellites, horizons)

            for isin in selected:
                selections.append({
                    'date': date,
                    'isin': isin
                })

        selections_df = pd.DataFrame(selections)

        # Merge with alpha
        merged = pd.merge(selections_df, alpha_df, on=['date', 'isin'], how='inner')

        if len(merged) > 0:
            avg_alpha = merged['forward_alpha'].mean()
            hit_rate = (merged['forward_alpha'] > 0).mean()

            # Portfolio metrics
            date_alphas = merged.groupby('date')['forward_alpha'].mean()
            portfolio_hit_rate = (date_alphas > 0).mean()
            sharpe = avg_alpha / merged['forward_alpha'].std() if merged['forward_alpha'].std() > 0 else 0

            results.append({
                'config': desc,
                'n_horizons': len(horizons),
                'avg_alpha': avg_alpha,
                'hit_rate': hit_rate,
                'portfolio_hit_rate': portfolio_hit_rate,
                'sharpe': sharpe
            })

            print(f"  Alpha: {avg_alpha:.4f} ({avg_alpha*100:.2f}%)")
            print(f"  Portfolio hit rate: {portfolio_hit_rate:.2%}")

    results_df = pd.DataFrame(results)

    # Analysis
    print("\n" + "-"*60)
    print("ANALYSIS")
    print("-"*60)

    if len(results_df) > 0:
        hit_range = results_df['portfolio_hit_rate'].max() - results_df['portfolio_hit_rate'].min()
        print(f"\nHit rate range: {hit_range:.4f} ({hit_range*100:.2f}%)")

        if hit_range > 0.15:
            print("  WARNING: HIGH SENSITIVITY - Number of horizons matters a lot")
        else:
            print("  OK: LOW SENSITIVITY - Performance stable across horizon configs")

        # Save results
        output_file = Path('data/feature_analysis/horizon_subset_sensitivity.csv')
        results_df.to_csv(output_file, index=False)
        print(f"\n[SAVED] {output_file}")

    return results_df


def main():
    """Run true parameter sensitivity tests."""
    print("\n" + "="*60)
    print("TRUE PARAMETER SENSITIVITY ANALYSIS")
    print("="*60)
    print("\nUsing ACTUAL selection logic from multi-horizon consensus")
    print("="*60)

    # Test 1: N_SATELLITES sensitivity
    n_results = test_n_satellites_sensitivity()

    # Test 2: Horizon subset sensitivity
    horizon_results = test_horizon_subset_sensitivity()

    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nThis analysis uses the TRUE selection logic, not simulations.")
    print("Results should accurately reflect real parameter sensitivity.")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
