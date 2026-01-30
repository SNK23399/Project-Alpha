"""
Analyze optimal ensemble size from MC precomputed data.

Uses the MC statistics (mc_ir_mean_1month.npz) for each N value (3, 4, 5)
to estimate which ensemble size gives the best expected IR performance.

Approach:
1. Load MC IR means for each N value
2. Compute mean expected IR across all features for each N
3. Find the N with highest mean IR
4. Recommend MIN_ENSEMBLE_SIZE (at peak or slightly before)
5. Recommend MAX_ENSEMBLE_SIZE (reasonable upper bound)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent / 'data'

# Current ensemble sizes computed in MC step 5
N_VALUES = [3, 4, 5]


def load_mc_data():
    """Load MC IR means and n_satellites info."""
    try:
        npz_file = DATA_DIR / 'mc_ir_mean_1month.npz'
        if npz_file.exists():
            data = np.load(npz_file, allow_pickle=False)
            return data['mc_ir_mean'], data['n_satellites']  # Returns (n_sat, test_dates, features) and (n_sat,)
        else:
            print(f"Warning: Could not find {npz_file}")
            return None, None
    except Exception as e:
        print(f"Warning: Error loading MC data: {e}")
        return None, None


def analyze_ensemble_size():
    """Analyze ensemble size performance from MC data."""

    print("\n" + "="*70)
    print("ENSEMBLE SIZE OPTIMIZATION ANALYSIS")
    print("="*70)

    # Load all MC data at once
    print(f"\nLoading MC data from {DATA_DIR / 'mc_ir_mean_1month.npz'}...")
    mc_ir_all, n_satellites_array = load_mc_data()

    if mc_ir_all is None:
        print("[ERROR] No MC data available for analysis")
        return None

    n_satellites_list = n_satellites_array.tolist()
    print(f"  [OK] Loaded N values: {n_satellites_list}")
    print(f"  Data shape: {mc_ir_all.shape} (n_satellites, test_dates, features)")

    results = {}

    # Analyze each N value
    for idx, n in enumerate(n_satellites_list):
        print(f"\nAnalyzing N={n} (index {idx})...")
        mc_ir = mc_ir_all[idx, :, :]  # (test_dates, features)

        # Compute statistics
        valid_mask = ~np.isnan(mc_ir)

        mean_ir = np.nanmean(mc_ir)  # Average across all dates and features
        median_ir = np.nanmedian(mc_ir)
        std_ir = np.nanstd(mc_ir)

        # Also compute mean of TOP features per test date
        # (since we only select top features, this is more relevant)
        top_features_per_date = []
        for test_idx in range(mc_ir.shape[0]):
            test_irs = mc_ir[test_idx, :]
            top_k = np.nanmax(test_irs)  # Max IR for this test date
            top_features_per_date.append(top_k)

        mean_top_ir = np.nanmean(top_features_per_date)

        results[n] = {
            'mean_ir': mean_ir,
            'median_ir': median_ir,
            'std_ir': std_ir,
            'mean_top_ir': mean_top_ir,
            'n_valid': np.sum(valid_mask),
            'n_total': mc_ir.size
        }

        print(f"  [OK] N={n}:")
        print(f"    Mean IR (all): {mean_ir:.6f}")
        print(f"    Median IR: {median_ir:.6f}")
        print(f"    Std IR: {std_ir:.6f}")
        print(f"    Mean of top IR per date: {mean_top_ir:.6f}")
        print(f"    Valid samples: {np.sum(valid_mask):,} / {mc_ir.size:,}")

    if not results:
        print("\n[ERROR] No MC data available for analysis")
        return None

    # Analyze results
    print("\n" + "-"*70)
    print("ANALYSIS RESULTS")
    print("-"*70)

    # Find best N by mean IR
    best_n_by_mean = max(results.keys(), key=lambda n: results[n]['mean_ir'])
    best_n_by_top = max(results.keys(), key=lambda n: results[n]['mean_top_ir'])

    print(f"\nBest N by overall mean IR: {best_n_by_mean} (IR={results[best_n_by_mean]['mean_ir']:.6f})")
    print(f"Best N by top feature IR: {best_n_by_top} (IR={results[best_n_by_top]['mean_top_ir']:.6f})")

    # Ranking
    print(f"\nRanking by mean IR:")
    for rank, n in enumerate(sorted(results.keys(),
                                    key=lambda x: results[x]['mean_ir'],
                                    reverse=True), 1):
        ir = results[n]['mean_ir']
        change_vs_best = ((ir - results[best_n_by_mean]['mean_ir']) /
                         abs(results[best_n_by_mean]['mean_ir']) * 100)
        print(f"  {rank}. N={n}: IR={ir:.6f} ({change_vs_best:+.2f}% vs best)")

    # Recommendations
    print(f"\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if len(results) < 3:
        print("\nNote: Currently have {} N values. For robust recommendations, consider:".format(len(results)))
        print("  - Expand N_SATELLITES_TO_PRECOMPUTE to include more values")
        print("  - Example: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] for full range analysis")

    # Conservative recommendation: use peak N as MIN
    min_ensemble_rec = best_n_by_mean
    max_ensemble_rec = best_n_by_mean + 2  # Allow some diversification beyond peak

    print(f"\nBased on current data (N={list(results.keys())}):")
    print(f"\n  MIN_ENSEMBLE_SIZE: {min_ensemble_rec}")
    print(f"    Rationale: Peak expected IR found at N={best_n_by_mean}")
    print(f"    This forces selection of features near the optimal diversity level")

    print(f"\n  MAX_ENSEMBLE_SIZE: {max_ensemble_rec}")
    print(f"    Rationale: Allow up to N+2 for additional diversification")
    print(f"    Greedy selection can stop early if diminishing returns")

    print(f"\nNote: These are data-driven recommendations.")
    print(f"You can validate by running full backtest with different settings.")

    # Return as dict for potential programmatic use
    return {
        'recommended_min': min_ensemble_rec,
        'recommended_max': max_ensemble_rec,
        'best_n_by_mean': best_n_by_mean,
        'best_n_by_top': best_n_by_top,
        'results': results
    }


if __name__ == '__main__':
    recommendations = analyze_ensemble_size()

    if recommendations:
        print("\n" + "="*70)
        print("To implement these recommendations, update pipeline_copy/6_bayesian_strategy_ir.py:")
        print("-"*70)
        print(f"MIN_ENSEMBLE_SIZE = {recommendations['recommended_min']}")
        print(f"MAX_ENSEMBLE_SIZE = {recommendations['recommended_max']}")
        print("="*70 + "\n")
