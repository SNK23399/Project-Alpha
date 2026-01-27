#!/usr/bin/env python3
"""
Correlation Threshold Test: Bayesian Ranking-Based Satellite Selection
========================================================================

Compare performance of CORRELATION FILTERING using deterministic ranking-based
satellite selection (like the actual Step 6 strategy). Tests whether stricter
correlation thresholds improve satellite quality.

Uses vectorized numpy operations and multiprocessing for speed.

Usage:
    python mc_correlation_bayesian_selection.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from multiprocessing import Pool, cpu_count
from scipy import stats
from functools import partial

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from support.etf_database import ETFDatabase

# Configuration
CORRELATION_THRESHOLDS = np.arange(0.0, 0.21, 0.01)  # [0.0, 0.01, 0.02, ..., 0.20]
AUM_THRESHOLDS = list(range(0, 105, 5))  # [0, 5, 10, 15, ..., 95, 100]
N_SATELLITES = 5  # Test with single N value for clarity
CORE_ISIN = 'IE00B4L5Y983'
N_WORKERS = cpu_count() - 1


def load_data():
    """Load ranking matrix, forward alpha, and universe."""
    print("Loading data...")

    # Load full ranking matrix (all signals, correlation threshold 0.0)
    ranking_file = Path(__file__).parent.parent / 'pipeline' / 'data' / 'rankings_matrix_signal_bases_1month.npz'
    if not ranking_file.exists():
        raise FileNotFoundError(f"Ranking matrix not found: {ranking_file}")

    ranking_data = np.load(ranking_file, allow_pickle=True)
    ranking_matrix = ranking_data['rankings']  # (n_dates, n_isins, n_signals)
    isins_array = ranking_data['isins']
    dates_array = ranking_data['dates']

    # Load forward alpha
    alpha_file = Path(__file__).parent.parent / 'pipeline' / 'data' / 'forward_alpha_1month.parquet'
    if not alpha_file.exists():
        raise FileNotFoundError(f"Forward alpha file not found: {alpha_file}")

    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])

    # Load universe
    db_path = str(Path(__file__).parent.parent / "maintenance" / "data" / "etf_database.db")
    db = ETFDatabase(db_path)
    universe_df = db.load_universe()

    return ranking_matrix, isins_array, dates_array, alpha_df, universe_df


def compute_signal_correlations_with_ir(ranking_matrix, isins_array, dates_array, alpha_df):
    """
    Compute Spearman correlation between each signal and forward IR.
    Returns: correlations array of shape (n_signals,)
    """
    print("Computing signal correlations with forward IR...")

    n_dates, n_isins, n_signals = ranking_matrix.shape
    isins_list = list(isins_array)
    dates_list = list(dates_array)

    # Build IR matrix once (vectorized)
    ir_matrix = np.full((n_dates, n_isins), np.nan, dtype=np.float32)

    for date_idx, date in enumerate(dates_list):
        date_data = alpha_df[alpha_df['date'] == date]
        for isin_idx, isin in enumerate(isins_list):
            isin_data = date_data[date_data['isin'] == isin]
            if len(isin_data) > 0:
                ir_matrix[date_idx, isin_idx] = isin_data['forward_ir'].values[0]

    ir_flat = ir_matrix.flatten()

    # Compute correlation for each signal (vectorized)
    correlations = np.zeros(n_signals, dtype=np.float32)

    for sig_idx in range(n_signals):
        signal_rankings = ranking_matrix[:, :, sig_idx].flatten()
        valid_mask = ~np.isnan(signal_rankings) & ~np.isnan(ir_flat)

        if valid_mask.sum() > 1:
            corr, _ = stats.spearmanr(signal_rankings[valid_mask], ir_flat[valid_mask])
            correlations[sig_idx] = abs(corr) if not np.isnan(corr) else 0.0

    print(f"  Computed correlations for {n_signals} signals")
    return correlations


def select_satellites_by_ranking(date_idx, ranking_matrix, isins_array, alpha_df, dates_array,
                                  filtered_signal_indices, aum_threshold, universe_df, n_satellites,
                                  all_correlations):
    """
    Select satellites for a single date using deterministic ranking-based selection.

    Process:
    1. Get valid ISINs for this AUM threshold
    2. Get valid ISINs with alpha data for this date
    3. For each valid ISIN, compute weighted score using filtered signals
    4. Select top N by score

    Returns: (monthly_ir, monthly_alpha) or (nan, nan) if insufficient data
    """
    date = dates_array[date_idx]

    # Get valid ISINs by AUM
    valid_isins_aum = set(
        universe_df[
            (universe_df['fund_size'].notna()) &
            (universe_df['fund_size'] >= aum_threshold)
        ]['isin'].values
    )

    # Get data for this date
    date_data = alpha_df[alpha_df['date'] == date]

    # Get ISINs with valid alpha and in valid AUM set
    valid_isins = set(date_data[date_data['isin'].isin(valid_isins_aum)]['isin'].values)

    if len(valid_isins) < n_satellites:
        return np.nan, np.nan

    # Map ISINs to indices
    isin_to_idx = {isin: idx for idx, isin in enumerate(isins_array)}
    valid_indices = [isin_to_idx[isin] for isin in valid_isins if isin in isin_to_idx]

    if len(valid_indices) < n_satellites:
        return np.nan, np.nan

    # Get rankings for this date, filtered signals only
    rankings_slice = ranking_matrix[date_idx, valid_indices, :][:, filtered_signal_indices]

    # Skip if all NaN
    if np.all(np.isnan(rankings_slice)):
        return np.nan, np.nan

    # Compute weighted score for each ISIN
    # Weight = correlation value (signals with higher correlation to IR get higher weight)
    correlations = all_correlations[filtered_signal_indices]

    scores = np.zeros(len(valid_indices))
    for i, isin_idx in enumerate(valid_indices):
        # Average ranking across filtered signals, weighted by correlation
        isin_rankings = rankings_slice[i, :]  # Shape: (n_filtered_signals,)

        # Remove NaN values
        valid_mask = ~np.isnan(isin_rankings)
        if valid_mask.sum() == 0:
            scores[i] = -np.inf
            continue

        # Weighted average: sum(ranking * correlation) / sum(correlation)
        valid_rankings = isin_rankings[valid_mask]
        valid_corrs = correlations[valid_mask]

        if valid_corrs.sum() > 0:
            scores[i] = np.average(valid_rankings, weights=valid_corrs)
        else:
            # If all correlations are 0 (shouldn't happen), use simple average
            scores[i] = np.mean(valid_rankings)

    # Select top N by score
    if np.all(np.isinf(scores)):
        return np.nan, np.nan

    top_indices = np.argsort(scores)[-n_satellites:]
    selected_isins = [list(valid_isins)[idx] for idx in top_indices]

    # Compute monthly IR and Alpha
    selected_data = date_data[date_data['isin'].isin(selected_isins)]

    if len(selected_data) == 0:
        return np.nan, np.nan

    monthly_ir = selected_data['forward_ir'].mean()
    monthly_alpha = selected_data['forward_alpha'].mean()

    return monthly_ir, monthly_alpha


def test_one_combination(args):
    """Test single (corr_threshold, aum_threshold) combination. For parallel processing."""
    corr_threshold, aum_threshold, ranking_matrix, isins_array, dates_array, \
        alpha_df, universe_df, all_correlations = args

    # Filter signals by correlation threshold
    filtered_signal_indices = np.where(all_correlations >= corr_threshold)[0]

    if len(filtered_signal_indices) == 0:
        # No signals pass threshold
        return (corr_threshold, aum_threshold, None)

    # Process each date
    monthly_irs = []
    monthly_alphas = []

    for date_idx in range(len(dates_array)):
        ir, alpha = select_satellites_by_ranking(
            date_idx, ranking_matrix, isins_array, alpha_df, dates_array,
            filtered_signal_indices, aum_threshold, universe_df, N_SATELLITES,
            all_correlations
        )

        if not np.isnan(ir) and not np.isnan(alpha):
            monthly_irs.append(ir)
            monthly_alphas.append(alpha)

    if len(monthly_irs) == 0:
        return (corr_threshold, aum_threshold, None)

    # Calculate stats
    monthly_irs = np.array(monthly_irs)
    monthly_alphas = np.array(monthly_alphas)

    annual_ir = np.mean(monthly_irs / np.sqrt(12)) * 12  # Convert to annual return
    hit_rate_ir = (monthly_irs > 0).sum() / len(monthly_irs) * 100
    annual_alpha = np.mean(monthly_alphas) * 12
    hit_rate_alpha = (monthly_alphas > 0).sum() / len(monthly_alphas) * 100

    stats_dict = {
        'mean_ir': np.mean(monthly_irs),
        'annual_return_ir': annual_ir * 100,
        'hit_rate_ir': hit_rate_ir,
        'mean_alpha_monthly': np.mean(monthly_alphas) * 100,
        'annual_alpha': annual_alpha * 100,
        'hit_rate_alpha': hit_rate_alpha,
        'std_ir': np.std(monthly_irs),
        'n_months': len(monthly_irs),
        'n_signals': len(filtered_signal_indices),
    }

    return (corr_threshold, aum_threshold, stats_dict)


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


def main():
    """Main correlation threshold test with Bayesian ranking-based selection."""

    print("\n" + "="*120)
    print("CORRELATION FILTERING TEST: DETERMINISTIC BAYESIAN RANKING-BASED SATELLITE SELECTION")
    print("="*120)
    print(f"\nTest: Compare satellite quality across correlation thresholds using ranking-based selection")
    print(f"Correlation Thresholds: {len(CORRELATION_THRESHOLDS)} values ({CORRELATION_THRESHOLDS[0]:.2f} to {CORRELATION_THRESHOLDS[-1]:.2f})")
    print(f"AUM Thresholds: {len(AUM_THRESHOLDS)} values (0M to 100M)")
    print(f"N Satellites: {N_SATELLITES}")
    print(f"Workers: {N_WORKERS} CPU cores")
    print()

    # Load data
    print("Loading data...")
    ranking_matrix, isins_array, dates_array, alpha_df, universe_df = load_data()
    dates_array = pd.to_datetime(dates_array)

    print(f"  Ranking matrix: {ranking_matrix.shape} (dates, isins, signals)")
    print(f"  Unique ISINs: {len(isins_array)}")
    print(f"  Date range: {dates_array[0]} to {dates_array[-1]}")

    # Compute correlations once
    print("\nComputing signal correlations...")
    all_correlations = compute_signal_correlations_with_ir(
        ranking_matrix, isins_array, dates_array, alpha_df
    )

    # Create all combinations to test
    print("\nPreparing test combinations...")
    combinations = []
    for corr_threshold in CORRELATION_THRESHOLDS:
        for aum_threshold in AUM_THRESHOLDS:
            combinations.append((
                corr_threshold, aum_threshold, ranking_matrix, isins_array, dates_array,
                alpha_df, universe_df, all_correlations
            ))

    # Run all combinations in parallel
    print(f"Running {len(combinations)} tests in parallel...")
    with Pool(processes=N_WORKERS) as pool:
        results_list = pool.map(test_one_combination, combinations)

    # Organize results
    results = {}
    for corr_threshold, aum_threshold, stats in results_list:
        if corr_threshold not in results:
            results[corr_threshold] = {}
        results[corr_threshold][aum_threshold] = stats

    # Print results
    print("\n" + "="*140)
    print("RESULTS: CORRELATION THRESHOLD IMPACT ON RANKING-BASED SATELLITE SELECTION")
    print("="*140)

    print(f"\n{'Corr Threshold':>14} {'AUM':>8} {'Signals':>8} {'Months':>8} {'Annual IR':>12} {'Hit Rate IR':>12} {'Annual Alpha':>13} {'Hit Rate Alpha':>14}")
    print("-"*140)

    for corr_threshold in CORRELATION_THRESHOLDS:
        threshold_str = f"{corr_threshold:.2f}"
        for aum_threshold in AUM_THRESHOLDS:
            if corr_threshold in results and aum_threshold in results[corr_threshold]:
                s = results[corr_threshold][aum_threshold]
                if s:
                    print(
                        f"{threshold_str:>14} {aum_threshold:>7}M {s['n_signals']:>8} {s['n_months']:>8} "
                        f"{s['annual_return_ir']:>11.1f}% {s['hit_rate_ir']:>11.1f}% "
                        f"{s['annual_alpha']:>12.1f}% {s['hit_rate_alpha']:>13.1f}%"
                    )

    # Trend analysis
    print("\n" + "="*140)
    print("TREND ANALYSIS: DOES STRICTER CORRELATION FILTERING HELP?")
    print("="*140)

    for aum_threshold in [0, 25, 50, 75, 100]:
        if aum_threshold not in AUM_THRESHOLDS:
            continue

        print(f"\nAUM Threshold = {aum_threshold}M:")

        ir_returns = []
        alpha_returns = []
        thresholds_valid = []

        for corr_threshold in CORRELATION_THRESHOLDS:
            if corr_threshold in results and aum_threshold in results[corr_threshold]:
                s = results[corr_threshold][aum_threshold]
                if s:
                    ir_returns.append(s['annual_return_ir'])
                    alpha_returns.append(s['annual_alpha'])
                    thresholds_valid.append(corr_threshold)

        if len(ir_returns) >= 2:
            ir_trend = ir_returns[-1] - ir_returns[0]
            alpha_trend = alpha_returns[-1] - alpha_returns[0]

            ir_direction = "Stricter filter HELPS" if ir_trend > 0 else "Stricter filter HURTS"
            alpha_direction = "Stricter filter HELPS" if alpha_trend > 0 else "Stricter filter HURTS"

            print(f"  IR Trend: {ir_direction} ({ir_trend:+.2f}% from corr=0.0 to 0.20)")
            print(f"  Alpha Trend: {alpha_direction} ({alpha_trend:+.2f}% from corr=0.0 to 0.20)")
            print(f"  IR:    {ir_returns[0]:6.1f}% (corr=0.0) -> {ir_returns[-1]:6.1f}% (corr=0.20)")
            print(f"  Alpha: {alpha_returns[0]:6.1f}% (corr=0.0) -> {alpha_returns[-1]:6.1f}% (corr=0.20)")

    # Export to CSV
    print("\n" + "="*140)
    print("EXPORTING RESULTS TO CSV")
    print("="*140)

    csv_data = []
    for corr_threshold in CORRELATION_THRESHOLDS:
        for aum_threshold in AUM_THRESHOLDS:
            if corr_threshold in results and aum_threshold in results[corr_threshold]:
                s = results[corr_threshold][aum_threshold]
                if s:
                    csv_data.append({
                        'correlation_threshold': corr_threshold,
                        'aum_threshold_M': aum_threshold,
                        'n_signals': s['n_signals'],
                        'n_months': s['n_months'],
                        'mean_ir': s['mean_ir'],
                        'annual_return_ir': s['annual_return_ir'],
                        'hit_rate_ir': s['hit_rate_ir'],
                        'mean_alpha_monthly': s['mean_alpha_monthly'],
                        'annual_alpha': s['annual_alpha'],
                        'hit_rate_alpha': s['hit_rate_alpha'],
                        'std_ir': s['std_ir'],
                    })

    csv_df = pd.DataFrame(csv_data)
    csv_path = Path(__file__).parent / 'mc_correlation_bayesian_results.csv'
    csv_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    print(f"Total rows: {len(csv_df)}")

    # Summary
    print("\n" + "="*140)
    print("INTERPRETATION")
    print("="*140)
    print("""
With DETERMINISTIC RANKING-BASED satellite selection (using filtered signals):
- If stricter filters HELP -> Signal correlation is predictive of quality
- If stricter filters HURT -> Correlation filtering removes useful signals
- If trend is flat -> Correlation threshold has negligible effect on ranking quality

Key difference from random test:
- Random test: Couldn't show effect because it didn't use signal rankings at all
- This test: Uses actual ranking-based selection, so correlation threshold SHOULD matter
- Result interpretation: Shows whether correlation is a good quality filter for ranking signals

Metrics explained:
- Annual IR: IR converted to annual % (for reference)
- Hit Rate IR: % of months where IR > 0
- Annual Alpha: Actual monthly alpha averaged and annualized (%)
- Hit Rate Alpha: % of months where alpha > 0
""")

    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
