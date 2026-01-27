#!/usr/bin/env python3
"""
Monte Carlo Test: Random Selection at Different AUM Thresholds
================================================================

Compare performance of RANDOM satellite selection across different AUM filters.
Uses vectorized numpy operations and multiprocessing for speed.

Usage:
    python mc_random_aum_comparison.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from support.etf_database import ETFDatabase

# Configuration
N_SATELLITES = [3, 4, 5, 6, 7]
AUM_THRESHOLDS = list(range(0, 105, 5))  # [0, 5, 10, 15, ..., 95, 100]
MC_ITERATIONS = 1000000  # Large number now that we're vectorized
CORE_ISIN = 'IE00B4L5Y983'
N_WORKERS = cpu_count() - 1  # All but one CPU core


def load_data():
    """Load forward alpha data and universe."""
    alpha_file = Path(__file__).parent.parent / 'pipeline' / 'data' / 'forward_alpha_1month.parquet'
    if not alpha_file.exists():
        raise FileNotFoundError(f"Forward alpha file not found: {alpha_file}")

    df = pd.read_parquet(alpha_file)
    df['date'] = pd.to_datetime(df['date'])

    db_path = str(Path(__file__).parent.parent / "maintenance" / "data" / "etf_database.db")
    db = ETFDatabase(db_path)
    universe_df = db.load_universe()

    return df.sort_values('date'), universe_df


def mc_random_selection_vectorized(alpha_df, dates, aum_threshold, n_satellites, universe_df):
    """
    Vectorized MC: all random samples generated at once for all months.
    Returns monthly IR and Alpha averages.
    """
    # Filter universe by AUM
    valid_isins = set(
        universe_df[
            (universe_df['fund_size'].notna()) &
            (universe_df['fund_size'] >= aum_threshold)
        ]['isin'].values
    )

    monthly_irs = []
    monthly_alphas = []

    for date in dates:
        # Get IRs and Alphas for this date
        date_data = alpha_df[alpha_df['date'] == date]
        available_data = date_data[date_data['isin'].isin(valid_isins)]

        available_irs = available_data['forward_ir'].values
        available_alphas = available_data['forward_alpha'].values

        if len(available_irs) < n_satellites:
            continue

        n_available = len(available_irs)

        # Skip if too few ETFs available
        if n_available < n_satellites:
            continue

        # Vectorized: generate all MC samples at once using numpy
        random_indices = np.random.choice(
            n_available,
            size=(MC_ITERATIONS, n_satellites),
            replace=True
        )

        # Select IRs and Alphas for all samples (vectorized)
        selected_irs = available_irs[random_indices]  # (MC_ITERATIONS, n_satellites)
        selected_alphas = available_alphas[random_indices]  # (MC_ITERATIONS, n_satellites)

        # Compute mean IR and Alpha for each sample (vectorized)
        mean_irs = np.mean(selected_irs, axis=1)  # (MC_ITERATIONS,)
        mean_alphas = np.mean(selected_alphas, axis=1)  # (MC_ITERATIONS,)

        # Average across all MC iterations
        monthly_irs.append(np.mean(mean_irs))
        monthly_alphas.append(np.mean(mean_alphas))

    return np.array(monthly_irs), np.array(monthly_alphas)


def calculate_stats(monthly_irs, monthly_alphas):
    """Calculate performance statistics for both IR and Alpha."""
    if len(monthly_irs) == 0 or len(monthly_alphas) == 0:
        return None

    monthly_irs = np.array(monthly_irs)
    monthly_alphas = np.array(monthly_alphas)

    # IR statistics
    monthly_returns_ir = monthly_irs / np.sqrt(12)
    annual_return_ir = np.mean(monthly_returns_ir) * 12
    hit_rate_ir = (monthly_irs > 0).sum() / len(monthly_irs) * 100

    # Alpha statistics (monthly alpha directly)
    annual_alpha = np.mean(monthly_alphas) * 12
    hit_rate_alpha = (monthly_alphas > 0).sum() / len(monthly_alphas) * 100

    return {
        # IR metrics
        'mean_ir': np.mean(monthly_irs),
        'annual_return_ir': annual_return_ir * 100,
        'hit_rate_ir': hit_rate_ir,

        # Alpha metrics
        'mean_alpha_monthly': np.mean(monthly_alphas) * 100,
        'annual_alpha': annual_alpha * 100,
        'hit_rate_alpha': hit_rate_alpha,

        # Combined
        'std_ir': np.std(monthly_irs),
    }


def test_one_combination(args):
    """Test single (n_sat, aum_threshold) combination. For parallel processing."""
    n_sat, aum_threshold, alpha_df, dates, universe_df = args

    monthly_irs, monthly_alphas = mc_random_selection_vectorized(alpha_df, dates, aum_threshold, n_sat, universe_df)
    stats = calculate_stats(monthly_irs, monthly_alphas)

    return (n_sat, aum_threshold, stats)


def main():
    """Main MC comparison with parallelization."""

    print("\n" + "="*100)
    print("MONTE CARLO: RANDOM SELECTION AT DIFFERENT AUM FILTERS")
    print("="*100)
    print(f"\nTest: {MC_ITERATIONS:,} random satellite selections per month (vectorized + parallel)")
    print(f"Workers: {N_WORKERS} CPU cores")
    print(f"Period: Walk-forward backtest (~94 months)")
    print()

    # Load data
    print("Loading data...")
    alpha_df, universe_df = load_data()
    dates = sorted(alpha_df['date'].unique())
    print(f"  Loaded {len(dates)} months of data")

    # Create all combinations to test
    combinations = []
    for n_sat in N_SATELLITES:
        for aum_threshold in AUM_THRESHOLDS:
            combinations.append((n_sat, aum_threshold, alpha_df, dates, universe_df))

    # Run all combinations in parallel
    print(f"\nRunning {len(combinations)} tests in parallel...")
    with Pool(processes=N_WORKERS) as pool:
        results_list = pool.map(test_one_combination, combinations)

    # Organize results
    results = {}
    for n_sat, aum_threshold, stats in results_list:
        if n_sat not in results:
            results[n_sat] = {}
        results[n_sat][aum_threshold] = stats

    # Print results
    print("\n" + "="*120)
    print("RESULTS: RANDOM SELECTION PERFORMANCE BY AUM FILTER")
    print("="*120)

    for n_sat in N_SATELLITES:
        print(f"\n{'N=' + str(n_sat):^120}")
        print("="*120)
        print("INFORMATION RATIO (IR) METRICS:")
        print("-"*120)
        print(f"{'AUM':>10} {'Annual IR':>15} {'Hit Rate IR':>15} {'Mean IR':>15} {'Std IR':>15}")
        print("-"*120)

        for aum_threshold in AUM_THRESHOLDS:
            if aum_threshold in results[n_sat]:
                s = results[n_sat][aum_threshold]
                if s:
                    print(
                        f"{aum_threshold:>9}M "
                        f"{s['annual_return_ir']:>14.1f}% "
                        f"{s['hit_rate_ir']:>14.1f}% "
                        f"{s['mean_ir']:>14.3f} "
                        f"{s['std_ir']:>14.3f}"
                    )
        print("-"*120)
        print("ALPHA (ABSOLUTE RETURN) METRICS:")
        print("-"*120)
        print(f"{'AUM':>10} {'Monthly Alpha':>15} {'Annual Alpha':>15} {'Hit Rate Alpha':>15}")
        print("-"*120)

        for aum_threshold in AUM_THRESHOLDS:
            if aum_threshold in results[n_sat]:
                s = results[n_sat][aum_threshold]
                if s:
                    print(
                        f"{aum_threshold:>9}M "
                        f"{s['mean_alpha_monthly']:>14.2f}% "
                        f"{s['annual_alpha']:>14.1f}% "
                        f"{s['hit_rate_alpha']:>14.1f}%"
                    )
        print("="*120)

    # Analysis
    print("\n" + "="*120)
    print("TREND ANALYSIS")
    print("="*120)

    for n_sat in N_SATELLITES:
        print(f"\nN={n_sat}:")

        aums = sorted(results[n_sat].keys())

        # IR trends
        ir_returns = [results[n_sat][aum]['annual_return_ir'] for aum in aums if results[n_sat][aum]]
        if len(ir_returns) >= 2:
            ir_trend = ir_returns[-1] - ir_returns[0]
            ir_direction = "Higher AUM better" if ir_trend > 0 else "Lower AUM better"
            print(f"  IR Trend: {ir_direction} (100M vs 0M: {abs(ir_trend):+.1f}%)")

        # Alpha trends
        alpha_returns = [results[n_sat][aum]['annual_alpha'] for aum in aums if results[n_sat][aum]]
        if len(alpha_returns) >= 2:
            alpha_trend = alpha_returns[-1] - alpha_returns[0]
            alpha_direction = "Higher AUM better" if alpha_trend > 0 else "Lower AUM better"
            print(f"  Alpha Trend: {alpha_direction} (100M vs 0M: {abs(alpha_trend):+.1f}%)")

        # Hit rates
        hit_rate_ir = [results[n_sat][aum]['hit_rate_ir'] for aum in aums if results[n_sat][aum]]
        hit_rate_alpha = [results[n_sat][aum]['hit_rate_alpha'] for aum in aums if results[n_sat][aum]]

        if len(hit_rate_ir) >= 2:
            print(f"  Hit Rate IR: {hit_rate_ir[0]:.1f}% (0M) → {hit_rate_ir[-1]:.1f}% (100M)")
        if len(hit_rate_alpha) >= 2:
            print(f"  Hit Rate Alpha: {hit_rate_alpha[0]:.1f}% (0M) → {hit_rate_alpha[-1]:.1f}% (100M)")

    # Export to CSV
    print("\n" + "="*120)
    print("EXPORTING RESULTS TO CSV")
    print("="*120)

    csv_data = []
    for n_sat in N_SATELLITES:
        for aum_threshold in AUM_THRESHOLDS:
            if aum_threshold in results[n_sat] and results[n_sat][aum_threshold]:
                s = results[n_sat][aum_threshold]
                csv_data.append({
                    'N_satellites': n_sat,
                    'AUM_threshold_M': aum_threshold,
                    'mean_ir': s['mean_ir'],
                    'annual_return_ir': s['annual_return_ir'],
                    'hit_rate_ir': s['hit_rate_ir'],
                    'mean_alpha_monthly': s['mean_alpha_monthly'],
                    'annual_alpha': s['annual_alpha'],
                    'hit_rate_alpha': s['hit_rate_alpha'],
                    'std_ir': s['std_ir'],
                })

    csv_df = pd.DataFrame(csv_data)
    csv_path = Path(__file__).parent / 'mc_aum_comparison_results.csv'
    csv_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    print(f"Total rows: {len(csv_df)}")

    # Summary
    print("\n" + "="*120)
    print("INTERPRETATION")
    print("="*120)
    print("""
With RANDOM satellite selection (vectorized, 1M iterations per month):
- If lower AUM performs worse -> Survivor bias is real
- If lower AUM performs better -> Survivor bias is NOT dominant
- If trend is noisy -> Both effects are similar magnitude

Key insight: This isolates the INHERENT AUM effect without signal confounding.

Metrics explained:
- Annual Return IR: IR converted to annual % (for reference)
- Hit Rate IR: % of months where IR > 0
- Annual Alpha: Actual monthly alpha averaged and annualized (%)
- Hit Rate Alpha: % of months where alpha > 0
- Mean Alpha Monthly: Average monthly alpha in percentage points
""")

    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
