"""
Monte Carlo Correlation Threshold × AUM Filter Comparison
===========================================================

Tests the combined effect of:
1. Correlation thresholds (0.0 to 0.2 in steps of 0.01) - signal quality filter
2. AUM thresholds (0 to 100M in steps of 5M) - universe size filter

Uses Monte Carlo random satellite selection (1M iterations per month).
Signals are filtered by correlation with forward IR, then random subsets
are selected to isolate the effect of correlation filtering.

Results show how signal filtering affects portfolio performance in combination
with universe filtering.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy import stats

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from support.etf_database import ETFDatabase

# Configuration
AUM_THRESHOLDS = list(range(0, 105, 5))  # [0, 5, 10, ..., 100]
CORRELATION_THRESHOLDS = np.arange(0.0, 0.21, 0.01)  # [0.0, 0.01, ..., 0.2]
MC_ITERATIONS = 1000000  # Large MC sample for accurate estimates
N_SATELLITES = 5  # Fixed number for satellite selection
CORE_ISIN = 'IE00B4L5Y983'
N_WORKERS = cpu_count() - 1


def _compute_one_correlation(signal_idx, ranking_matrix, ir_flat):
    """Compute correlation for a single signal (module-level for pickling)."""
    signal_rankings = ranking_matrix[:, :, signal_idx].flatten()
    valid_mask = ~np.isnan(signal_rankings) & ~np.isnan(ir_flat)

    if valid_mask.sum() > 1:
        corr, _ = stats.spearmanr(signal_rankings[valid_mask], ir_flat[valid_mask])
        return abs(corr) if not np.isnan(corr) else 0.0
    return 0.0


def load_data():
    """Load forward alpha data, universe, and ranking matrix."""
    pipeline_dir = Path(__file__).parent.parent / 'pipeline'

    # Load forward IR/alpha
    alpha_file = pipeline_dir / 'data' / 'forward_alpha_1month.parquet'
    if not alpha_file.exists():
        raise FileNotFoundError(f"Missing {alpha_file}")
    df = pd.read_parquet(alpha_file)

    # Load ranking matrix (from correlation threshold 0.0 run)
    rankings_base_file = pipeline_dir / 'data' / 'rankings_matrix_signal_bases_1month.npz'
    if not rankings_base_file.exists():
        raise FileNotFoundError(f"Missing {rankings_base_file}. Run step 2 with CORRELATION_THRESHOLD = 0.0 first")

    rankings_base = np.load(rankings_base_file)
    ranking_matrix_base = rankings_base['rankings']  # Shape: (dates, isins, features)
    signal_names_base = rankings_base['features']  # Signal names/indices
    dates_list = rankings_base['dates']
    isins_list = rankings_base['isins']

    print(f"Loaded ranking matrix: {ranking_matrix_base.shape} (dates × ISINs × signals)")
    print(f"Signals: {len(signal_names_base)}, ISINs: {len(isins_list)}, Dates: {len(dates_list)}")

    # Load universe
    db_path = Path(__file__).parent.parent / 'maintenance' / 'data' / 'etf_database.db'
    db = ETFDatabase(str(db_path), readonly=True)
    universe_df = db.load_universe()

    return df.sort_values('date'), universe_df, ranking_matrix_base, signal_names_base, dates_list, isins_list


def compute_signal_correlations(signal_names_base, ranking_matrix_base, alpha_df, dates_list, isins_list):
    """
    Vectorized and parallelized signal correlation computation.

    Step 1: Build IR matrix once (vectorized)
    Step 2: Parallelize correlation computation across signals
    """
    print("\nComputing signal correlations (vectorized & parallelized)...")

    n_dates, n_isins, n_signals = ranking_matrix_base.shape

    # Step 1: Build IR matrix once (vectorized)
    ir_matrix = np.full((n_dates, n_isins), np.nan, dtype=np.float32)

    for date_idx, date in enumerate(dates_list):
        date_data = alpha_df[alpha_df['date'] == date]
        for isin_idx, isin in enumerate(isins_list):
            isin_data = date_data[date_data['isin'] == isin]
            if len(isin_data) > 0:
                ir_matrix[date_idx, isin_idx] = isin_data['forward_ir'].values[0]

    ir_flat = ir_matrix.flatten()

    # Step 2: Parallelize correlation computation across signals
    compute_corr_partial = partial(_compute_one_correlation, ranking_matrix=ranking_matrix_base, ir_flat=ir_flat)

    with Pool(N_WORKERS) as pool:
        correlations_list = pool.map(compute_corr_partial, range(n_signals))

    correlations = np.array(correlations_list)

    print(f"Signal correlations: mean={correlations.mean():.3f}, median={np.median(correlations):.3f}, std={correlations.std():.3f}")
    return correlations


def mc_random_selection_with_signal_filter(alpha_df, dates_list, isins_list, ranking_matrix_base,
                                            signal_correlations, corr_threshold, aum_threshold,
                                            universe_df):
    """
    Monte Carlo satellite selection with filtered signals and AUM.

    Returns: (monthly_irs, monthly_alphas) arrays
    """
    # Filter signals by correlation
    signal_mask = signal_correlations >= corr_threshold
    signal_indices = np.where(signal_mask)[0]

    if len(signal_indices) < 1:
        return np.array([]), np.array([])

    # Filter ranking matrix to only include passing signals
    ranking_matrix_filtered = ranking_matrix_base[:, :, signal_indices]

    # Filter ISINs by AUM
    valid_isins = set(
        universe_df[
            (universe_df['fund_size'].notna()) &
            (universe_df['fund_size'] >= aum_threshold)
        ]['isin'].values
    )

    isin_mask = np.array([isin in valid_isins for isin in isins_list])
    isin_indices = np.where(isin_mask)[0]

    if len(isin_indices) < N_SATELLITES:
        return np.array([]), np.array([])

    # Filter ranking matrix and ISINs
    ranking_matrix_final = ranking_matrix_filtered[:, isin_indices, :]
    isins_filtered = np.array(isins_list)[isin_indices]

    monthly_irs = []
    monthly_alphas = []

    # Process each unique month
    for date_idx, date in enumerate(dates_list):
        date_data = alpha_df[alpha_df['date'] == date]
        if len(date_data) == 0:
            continue

        # Get forward IR and alpha for ISINs in this month
        available_data = date_data[date_data['isin'].isin(isins_filtered)]
        if len(available_data) < N_SATELLITES:
            continue

        available_irs = available_data['forward_ir'].values
        available_alphas = available_data['forward_alpha'].values

        n_available = len(available_irs)

        # Monte Carlo: generate all random samples at once (vectorized)
        random_indices = np.random.choice(
            n_available,
            size=(MC_ITERATIONS, N_SATELLITES),
            replace=True
        )

        # Select IRs and Alphas for all samples (vectorized)
        selected_irs = available_irs[random_indices]  # (MC_ITERATIONS, N_SATELLITES)
        selected_alphas = available_alphas[random_indices]  # (MC_ITERATIONS, N_SATELLITES)

        # Compute mean IR and Alpha for each sample (vectorized)
        mean_irs = np.mean(selected_irs, axis=1)  # (MC_ITERATIONS,)
        mean_alphas = np.mean(selected_alphas, axis=1)  # (MC_ITERATIONS,)

        # Average across all MC iterations
        monthly_irs.append(np.mean(mean_irs))
        monthly_alphas.append(np.mean(mean_alphas))

    return np.array(monthly_irs), np.array(monthly_alphas)


def calculate_stats(monthly_irs, monthly_alphas):
    """Calculate performance statistics."""
    if len(monthly_irs) == 0 or len(monthly_alphas) == 0:
        return None

    monthly_irs = np.array(monthly_irs)
    monthly_alphas = np.array(monthly_alphas)

    # IR statistics
    monthly_returns_ir = monthly_irs / np.sqrt(12)
    annual_return_ir = np.mean(monthly_returns_ir) * 12
    hit_rate_ir = (monthly_irs > 0).sum() / len(monthly_irs) * 100

    # Alpha statistics
    annual_alpha = np.mean(monthly_alphas) * 12
    hit_rate_alpha = (monthly_alphas > 0).sum() / len(monthly_alphas) * 100

    return {
        'mean_ir': np.mean(monthly_irs),
        'annual_return_ir': annual_return_ir * 100,
        'hit_rate_ir': hit_rate_ir,
        'mean_alpha_monthly': np.mean(monthly_alphas) * 100,
        'annual_alpha': annual_alpha * 100,
        'hit_rate_alpha': hit_rate_alpha,
        'std_ir': np.std(monthly_irs),
        'n_months': len(monthly_irs),
    }


def test_one_combination(args):
    """Test single (corr_threshold, aum_threshold) combination."""
    (corr_threshold, aum_threshold, alpha_df, dates_list, isins_list,
     ranking_matrix_base, signal_correlations, universe_df) = args

    monthly_irs, monthly_alphas = mc_random_selection_with_signal_filter(
        alpha_df, dates_list, isins_list, ranking_matrix_base,
        signal_correlations, corr_threshold, aum_threshold, universe_df
    )
    stats = calculate_stats(monthly_irs, monthly_alphas)

    return (corr_threshold, aum_threshold, stats)


def main():
    """Main test with parallelization."""
    print("\n" + "="*140)
    print("MONTE CARLO: CORRELATION THRESHOLD × AUM FILTER COMPARISON")
    print("="*140)

    # Load data
    print("\nLoading data...")
    alpha_df, universe_df, ranking_matrix_base, signal_names_base, dates_list, isins_list = load_data()

    # Compute signal correlations (once, used for all combinations)
    signal_correlations = compute_signal_correlations(signal_names_base, ranking_matrix_base, alpha_df, dates_list, isins_list)

    # Prepare test combinations
    combinations = [
        (corr_thresh, aum_thresh, alpha_df, dates_list, isins_list, ranking_matrix_base, signal_correlations, universe_df)
        for corr_thresh in CORRELATION_THRESHOLDS
        for aum_thresh in AUM_THRESHOLDS
    ]

    total_combos = len(combinations)
    print(f"\nTesting {total_combos} combinations (21 AUM × 21 correlation thresholds)")
    print(f"MC iterations: {MC_ITERATIONS:,} per month")
    print(f"N satellites: {N_SATELLITES}")
    print(f"Using {N_WORKERS} workers for parallelization\n")

    # Run parallel tests
    with Pool(N_WORKERS) as pool:
        results_list = pool.map(test_one_combination, combinations)

    # Organize results
    results = {}
    for corr_thresh, aum_thresh, stats in results_list:
        if corr_thresh not in results:
            results[corr_thresh] = {}
        results[corr_thresh][aum_thresh] = stats

    # Print results by AUM threshold
    print("\n" + "="*150)
    print("RESULTS BY AUM THRESHOLD (Annual IR, Hit Rate IR, Annual Alpha)")
    print("="*150)

    for aum_thresh in AUM_THRESHOLDS:
        print(f"\nAUM >= {aum_thresh}M:")
        print("-"*150)
        print(f"{'Corr':>8} {'Ann IR':>10} {'Hit IR':>8} {'Ann Alpha':>12} {'Hit Alpha':>10} {'Months':>8}")
        print("-"*150)

        for corr_thresh in CORRELATION_THRESHOLDS:
            if corr_thresh in results and aum_thresh in results[corr_thresh]:
                s = results[corr_thresh][aum_thresh]
                if s:
                    print(
                        f"{corr_thresh:>7.2f} "
                        f"{s['annual_return_ir']:>9.1f}% "
                        f"{s['hit_rate_ir']:>7.1f}% "
                        f"{s['annual_alpha']:>11.1f}% "
                        f"{s['hit_rate_alpha']:>9.1f}% "
                        f"{s['n_months']:>8.0f}"
                    )

    # Print results by Correlation threshold
    print("\n" + "="*150)
    print("RESULTS BY CORRELATION THRESHOLD (Annual IR, Hit Rate IR, Annual Alpha)")
    print("="*150)

    for corr_thresh in CORRELATION_THRESHOLDS:
        print(f"\nCorr >= {corr_thresh:.2f}:")
        print("-"*150)
        print(f"{'AUM(M)':>8} {'Ann IR':>10} {'Hit IR':>8} {'Ann Alpha':>12} {'Hit Alpha':>10} {'Months':>8}")
        print("-"*150)

        for aum_thresh in AUM_THRESHOLDS:
            if corr_thresh in results and aum_thresh in results[corr_thresh]:
                s = results[corr_thresh][aum_thresh]
                if s:
                    print(
                        f"{aum_thresh:>7.0f} "
                        f"{s['annual_return_ir']:>9.1f}% "
                        f"{s['hit_rate_ir']:>7.1f}% "
                        f"{s['annual_alpha']:>11.1f}% "
                        f"{s['hit_rate_alpha']:>9.1f}% "
                        f"{s['n_months']:>8.0f}"
                    )

    # Trend analysis
    print("\n" + "="*150)
    print("TREND ANALYSIS")
    print("="*150)

    # Effect of correlation threshold (averaged across AUM)
    print("\nCorrelation Threshold Effect (averaging across all AUM filters):")
    print("-"*150)
    print(f"{'Corr':>8} {'Avg Ann IR':>12} {'Avg Hit IR':>12} {'Avg Ann Alpha':>15}")
    print("-"*150)

    for corr_thresh in CORRELATION_THRESHOLDS:
        valid_stats = [results[corr_thresh][aum] for aum in AUM_THRESHOLDS
                      if aum in results[corr_thresh] and results[corr_thresh][aum]]
        if valid_stats:
            avg_ir = np.mean([s['annual_return_ir'] for s in valid_stats])
            avg_hit_ir = np.mean([s['hit_rate_ir'] for s in valid_stats])
            avg_alpha = np.mean([s['annual_alpha'] for s in valid_stats])
            print(f"{corr_thresh:>7.2f} {avg_ir:>11.1f}% {avg_hit_ir:>11.1f}% {avg_alpha:>14.1f}%")

    # Effect of AUM threshold (averaged across correlation)
    print("\nAUM Threshold Effect (averaging across all correlation filters):")
    print("-"*150)
    print(f"{'AUM(M)':>8} {'Avg Ann IR':>12} {'Avg Hit IR':>12} {'Avg Ann Alpha':>15}")
    print("-"*150)

    for aum_thresh in AUM_THRESHOLDS:
        valid_stats = [results[corr][aum_thresh] for corr in CORRELATION_THRESHOLDS
                      if corr in results and aum_thresh in results[corr] and results[corr][aum_thresh]]
        if valid_stats:
            avg_ir = np.mean([s['annual_return_ir'] for s in valid_stats])
            avg_hit_ir = np.mean([s['hit_rate_ir'] for s in valid_stats])
            avg_alpha = np.mean([s['annual_alpha'] for s in valid_stats])
            print(f"{aum_thresh:>7.0f} {avg_ir:>11.1f}% {avg_hit_ir:>11.1f}% {avg_alpha:>14.1f}%")

    # Export to CSV
    print("\n" + "="*150)
    print("EXPORTING RESULTS TO CSV")
    print("="*150)

    csv_data = []
    for corr_thresh in CORRELATION_THRESHOLDS:
        for aum_thresh in AUM_THRESHOLDS:
            if corr_thresh in results and aum_thresh in results[corr_thresh] and results[corr_thresh][aum_thresh]:
                s = results[corr_thresh][aum_thresh]
                csv_data.append({
                    'correlation_threshold': corr_thresh,
                    'AUM_threshold_M': aum_thresh,
                    'mean_ir': s['mean_ir'],
                    'annual_return_ir': s['annual_return_ir'],
                    'hit_rate_ir': s['hit_rate_ir'],
                    'mean_alpha_monthly': s['mean_alpha_monthly'],
                    'annual_alpha': s['annual_alpha'],
                    'hit_rate_alpha': s['hit_rate_alpha'],
                    'std_ir': s['std_ir'],
                    'n_months': s['n_months'],
                })

    csv_df = pd.DataFrame(csv_data)
    csv_path = Path(__file__).parent / 'mc_correlation_aum_comparison_results.csv'
    csv_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    print(f"Total rows: {len(csv_df)}")

    print("\n" + "="*150)
    print("INTERPRETATION")
    print("="*150)
    print("""
With MONTE CARLO random satellite selection (1M iterations per month):

Correlation Threshold Effect:
- If stricter correlation (0.1-0.2) performs worse -> Low-correlation signals are valuable
- If stricter correlation performs better -> Current 0.1 threshold might be too permissive
- If no clear trend -> Correlation filtering doesn't significantly impact signal quality

AUM Threshold Effect (should match previous AUM-only test):
- If higher AUM performs better -> Survivor bias is real
- If lower AUM performs better -> Survivor bias is NOT dominant

Combined Effects:
- If effects interact -> Some correlation ranges work better with certain AUM filters
- If independent -> Can optimize correlation and AUM thresholds separately

Key metrics:
- Annual Return IR: Reference metric only (IR normalized as %)
- Hit Rate IR: % of months with positive signal quality
- Annual Alpha: Actual monthly returns averaged and annualized
- Hit Rate Alpha: % of months with positive returns
- N Months: Number of valid test periods (should be consistent ~130)
""")

    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
