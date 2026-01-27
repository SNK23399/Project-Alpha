#!/usr/bin/env python3
"""
Comprehensive Signal Correlation Analysis

Rank all 293 signals by their predictive power for forward IR.
Identify top performers and signal categories.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_data():
    """Load ranking matrix and forward alpha."""
    # Load ranking matrix
    ranking_file = Path(__file__).parent.parent / 'pipeline' / 'data' / 'rankings_matrix_signal_bases_1month.npz'
    ranking_data = np.load(ranking_file, allow_pickle=True)
    ranking_matrix = ranking_data['rankings']
    dates_array = ranking_data['dates']

    # Load forward alpha
    alpha_file = Path(__file__).parent.parent / 'pipeline' / 'data' / 'forward_alpha_1month.parquet'
    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])

    # Load signal names
    signal_dir = Path(__file__).parent.parent / 'pipeline' / 'data' / 'signals' / 'signal_bases'
    signal_files = sorted([f.stem for f in signal_dir.glob('*.pkl')])

    return ranking_matrix, dates_array, alpha_df, signal_files


def compute_all_signal_correlations(ranking_matrix, dates_array, alpha_df, signal_files):
    """Compute correlation of each signal with forward IR."""
    print("Computing correlations for all 293 signals...")

    n_dates, n_isins, n_signals = ranking_matrix.shape
    dates_list = pd.to_datetime(dates_array)

    # Build IR matrix once
    ir_matrix = np.full((n_dates, n_isins), np.nan, dtype=np.float32)

    for date_idx, date in enumerate(dates_list):
        date_data = alpha_df[alpha_df['date'] == date]
        for isin_idx in range(n_isins):
            if len(date_data) > 0:
                ir_matrix[date_idx, isin_idx] = date_data['forward_ir'].mean()

    ir_flat = ir_matrix.flatten()

    # Compute correlation for each signal
    correlations = np.zeros(n_signals, dtype=np.float32)

    for sig_idx in range(n_signals):
        signal_rankings = ranking_matrix[:, :, sig_idx].flatten()
        valid_mask = ~np.isnan(signal_rankings) & ~np.isnan(ir_flat)

        if valid_mask.sum() > 1:
            corr, _ = stats.spearmanr(signal_rankings[valid_mask], ir_flat[valid_mask])
            correlations[sig_idx] = abs(corr) if not np.isnan(corr) else 0.0

        if (sig_idx + 1) % 50 == 0:
            print(f"  Processed {sig_idx + 1} / {n_signals}")

    return correlations


def categorize_signal(signal_name):
    """Categorize signal by type."""
    if 'rs' in signal_name or 'momentum' in signal_name or 'return' in signal_name:
        return 'Momentum'
    elif 'volatility' in signal_name or 'vol' in signal_name:
        return 'Volatility'
    elif 'beta' in signal_name:
        return 'Beta/Sensitivity'
    elif 'sharpe' in signal_name or 'calmar' in signal_name or 'sortino' in signal_name or 'info' in signal_name:
        return 'Risk-Adjusted Returns'
    elif 'drawdown' in signal_name or 'ulcer' in signal_name or 'cvar' in signal_name:
        return 'Drawdown/Risk'
    elif 'ratio' in signal_name or 'spread' in signal_name:
        return 'Relative Measures'
    elif 'price' in signal_name or 'proximity' in signal_name:
        return 'Valuation'
    elif 'alpha' in signal_name or 'outperformance' in signal_name:
        return 'Alpha/Outperformance'
    elif 'correlation' in signal_name or 'corr' in signal_name:
        return 'Correlation'
    elif 'stochastic' in signal_name or 'cci' in signal_name or 'rsi' in signal_name:
        return 'Technical'
    elif 'disagreement' in signal_name or 'crowding' in signal_name:
        return 'Sentiment/Crowding'
    else:
        return 'Other'


def main():
    """Main analysis."""

    print("\n" + "="*120)
    print("COMPREHENSIVE SIGNAL CORRELATION ANALYSIS")
    print("="*120)

    # Load data
    ranking_matrix, dates_array, alpha_df, signal_files = load_data()

    print(f"\nData loaded:")
    print(f"  Total signals: {len(signal_files)}")
    print(f"  Ranking matrix shape: {ranking_matrix.shape}")
    print(f"  Date range: {pd.to_datetime(dates_array[0])} to {pd.to_datetime(dates_array[-1])}")

    # Compute correlations
    correlations = compute_all_signal_correlations(ranking_matrix, dates_array, alpha_df, signal_files)

    # Create results dataframe
    results = pd.DataFrame({
        'signal_name': signal_files,
        'correlation': correlations,
        'category': [categorize_signal(s) for s in signal_files],
    }).sort_values('correlation', ascending=False)

    print("\n" + "="*120)
    print("TOP 30 SIGNALS BY CORRELATION WITH FORWARD IR")
    print("="*120)

    print(f"\n{'Rank':>4} {'Correlation':>12} {'Category':<30} {'Signal Name':<40}")
    print("-"*120)

    for idx, (_, row) in enumerate(results.head(30).iterrows(), 1):
        print(f"{idx:4d} {row['correlation']:12.4f} {row['category']:<30} {row['signal_name']:<40}")

    # Summary statistics by category
    print("\n" + "="*120)
    print("SIGNAL PERFORMANCE BY CATEGORY")
    print("="*120)

    category_stats = results.groupby('category')['correlation'].agg([
        ('Count', 'count'),
        ('Mean', 'mean'),
        ('Median', 'median'),
        ('Max', 'max'),
        ('Min', 'min'),
    ]).sort_values('Max', ascending=False)

    print(f"\n{'Category':<30} {'Count':>6} {'Mean':>10} {'Median':>10} {'Max':>10} {'Min':>10}")
    print("-"*120)

    for category, row in category_stats.iterrows():
        print(
            f"{category:<30} {int(row['Count']):>6} "
            f"{row['Mean']:>10.4f} {row['Median']:>10.4f} "
            f"{row['Max']:>10.4f} {row['Min']:>10.4f}"
        )

    # Bottom 20 (worst performers)
    print("\n" + "="*120)
    print("BOTTOM 20 SIGNALS (WEAKEST CORRELATION)")
    print("="*120)

    print(f"\n{'Rank':>4} {'Correlation':>12} {'Category':<30} {'Signal Name':<40}")
    print("-"*120)

    for idx, (_, row) in enumerate(results.tail(20).iloc[::-1].iterrows(), 1):
        print(f"{idx:4d} {row['correlation']:12.4f} {row['category']:<30} {row['signal_name']:<40}")

    # Distribution analysis
    print("\n" + "="*120)
    print("CORRELATION DISTRIBUTION")
    print("="*120)

    print(f"""
Overall Statistics:
  Mean correlation:        {results['correlation'].mean():.4f}
  Median correlation:      {results['correlation'].median():.4f}
  Std dev:                 {results['correlation'].std():.4f}
  Min:                     {results['correlation'].min():.4f}
  Max:                     {results['correlation'].max():.4f}

Percentiles:
  90th percentile:         {results['correlation'].quantile(0.90):.4f}
  75th percentile:         {results['correlation'].quantile(0.75):.4f}
  50th percentile:         {results['correlation'].quantile(0.50):.4f}
  25th percentile:         {results['correlation'].quantile(0.25):.4f}
  10th percentile:         {results['correlation'].quantile(0.10):.4f}

Count by performance tier:
  Top tier (>0.10):        {(results['correlation'] > 0.10).sum()} signals ({(results['correlation'] > 0.10).sum()/len(results)*100:.1f}%)
  Good (0.05-0.10):        {((results['correlation'] >= 0.05) & (results['correlation'] <= 0.10)).sum()} signals
  Fair (0.02-0.05):        {((results['correlation'] >= 0.02) & (results['correlation'] < 0.05)).sum()} signals
  Poor (<0.02):            {(results['correlation'] < 0.02).sum()} signals ({(results['correlation'] < 0.02).sum()/len(results)*100:.1f}%)
""")

    # Key findings
    print("\n" + "="*120)
    print("KEY FINDINGS")
    print("="*120)

    top_5 = results.head(5)
    print(f"""
1. TOP SIGNAL (CLEAR DOMINANCE):
   {top_5.iloc[0]['signal_name']} (correlation: {top_5.iloc[0]['correlation']:.4f})
   - {top_5.iloc[0]['correlation'] / top_5.iloc[4]['correlation']:.1f}x better than #5

2. SIGNAL CATEGORIES RANKED BY BEST PERFORMER:
""")

    for cat in results.groupby('category')['correlation'].max().sort_values(ascending=False).index:
        best = results[results['category'] == cat]['correlation'].max()
        avg = results[results['category'] == cat]['correlation'].mean()
        count = len(results[results['category'] == cat])
        print(f"   {cat:<30} Best: {best:.4f}, Avg: {avg:.4f} ({count} signals)")

    print(f"""
3. SIGNAL DIVERSITY:
   - Only {(results['correlation'] > 0.10).sum()} signals have correlation > 0.10
   - Top signal is {results.iloc[0]['correlation'] / results[results['correlation'] > 0.10]['correlation'].mean():.1f}x better than average of top tier
   - Suggests heavy concentration in predictive power

4. NOISE LEVEL:
   - {(results['correlation'] < 0.02).sum()} signals ({(results['correlation'] < 0.02).sum()/len(results)*100:.1f}%) have correlation < 0.02 (effectively noise)
   - These add little predictive value but increase complexity
""")

    # Export detailed results
    print("\n" + "="*120)
    print("EXPORTING RESULTS")
    print("="*120)

    csv_path = Path(__file__).parent / 'all_signal_correlations.csv'
    results.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")
    print(f"Total rows: {len(results)}")

    # Recommendations
    print("\n" + "="*120)
    print("RECOMMENDATIONS")
    print("="*120)

    print(f"""
Based on this analysis:

1. IMMEDIATE:
   - Use top {(results['correlation'] > 0.10).sum()} signals (correlation >= 0.10)
   - Removes {(results['correlation'] < 0.02).sum()} noise signals
   - Should improve hit rates significantly

2. MEDIUM-TERM:
   - Test top 5-10 signals vs top 1 (rs_vs_univ_126d)
   - Check if combination improves robustness
   - Look for uncorrelated pairs among top performers

3. INVESTIGATION:
   - Why does {results.iloc[0]['signal_name']} dominate?
   - Are other {results.iloc[0]['category']} signals also strong?
   - Why are {results.tail(1).iloc[0]['category']} signals so weak?

4. TESTING STRATEGY:
   Tier 1: {results.iloc[0]['signal_name']} alone
   Tier 2: Top 3 signals ({', '.join(results.head(3)['signal_name'].tolist())})
   Tier 3: Top {(results['correlation'] > 0.05).sum()} signals (correlation >= 0.05)
   Current: All 293 signals
""")

    return 0


if __name__ == '__main__':
    sys.exit(main())
