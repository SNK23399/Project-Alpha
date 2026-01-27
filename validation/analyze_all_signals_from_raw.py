#!/usr/bin/env python3
"""
Comprehensive Signal Analysis - All 293 Signals

Rank signals by correlation with forward IR.
Uses raw signal pickle files directly.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_signal(signal_name):
    """Load a signal from pickle file."""
    signal_file = Path(__file__).parent.parent / 'pipeline' / 'data' / 'signals' / 'signal_bases' / f'{signal_name}.pkl'
    with open(signal_file, 'rb') as f:
        return pickle.load(f)


def compute_signal_correlation(signal_df, alpha_df):
    """Compute correlation between signal and forward IR."""
    # Convert to numeric
    signal_numeric = signal_df.drop('date', axis=1, errors='ignore').apply(pd.to_numeric, errors='coerce')

    # Convert index to datetime for alignment
    signal_dates = pd.to_datetime(signal_numeric.index)

    # Align dates
    alpha_dates = alpha_df['date'].values
    common_dates = pd.DatetimeIndex(signal_dates).intersection(pd.DatetimeIndex(alpha_dates))

    if len(common_dates) == 0:
        return None

    # Get values for common dates
    signal_common = signal_numeric.loc[common_dates].values.flatten()
    alpha_common = alpha_df[alpha_df['date'].isin(common_dates)].sort_values('date')
    ir_values = alpha_common['forward_ir'].values

    # Repeat IR to match signal shape
    ir_values_expanded = np.repeat(ir_values, signal_numeric.shape[1])[:len(signal_common)]

    # Compute correlation
    valid_mask = ~np.isnan(signal_common) & ~np.isnan(ir_values_expanded)

    if valid_mask.sum() > 1:
        corr, pval = stats.spearmanr(signal_common[valid_mask], ir_values_expanded[valid_mask])
        return {
            'correlation': abs(corr) if not np.isnan(corr) else 0.0,
            'p_value': pval,
            'n_valid': valid_mask.sum(),
        }

    return None


def categorize_signal(signal_name):
    """Categorize signal by type."""
    if 'rs' in signal_name or 'momentum' in signal_name or 'return' in signal_name:
        return 'Momentum'
    elif 'volatility' in signal_name or 'vol' in signal_name:
        return 'Volatility'
    elif 'beta' in signal_name:
        return 'Beta'
    elif 'sharpe' in signal_name or 'calmar' in signal_name or 'sortino' in signal_name or 'info' in signal_name:
        return 'Risk-Adjusted'
    elif 'drawdown' in signal_name or 'ulcer' in signal_name or 'cvar' in signal_name:
        return 'Drawdown Risk'
    elif 'ratio' in signal_name or 'spread' in signal_name:
        return 'Relative Measures'
    elif 'price' in signal_name or 'proximity' in signal_name:
        return 'Valuation'
    elif 'alpha' in signal_name:
        return 'Alpha'
    elif 'correlation' in signal_name or 'corr' in signal_name:
        return 'Correlation'
    elif 'stochastic' in signal_name or 'cci' in signal_name or 'rsi' in signal_name or 'williams' in signal_name:
        return 'Technical'
    elif 'disagreement' in signal_name or 'crowding' in signal_name or 'crowded' in signal_name:
        return 'Sentiment'
    else:
        return 'Other'


def main():
    """Main analysis."""

    print("\n" + "="*130)
    print("COMPREHENSIVE ANALYSIS: ALL 293 SIGNALS")
    print("="*130)

    # Load forward alpha
    alpha_file = Path(__file__).parent.parent / 'pipeline' / 'data' / 'forward_alpha_1month.parquet'
    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])

    # Load signal names
    signal_dir = Path(__file__).parent.parent / 'pipeline' / 'data' / 'signals' / 'signal_bases'
    signal_files = sorted([f.stem for f in signal_dir.glob('*.pkl')])

    print(f"\nLoading and analyzing {len(signal_files)} signals...")
    print(f"Forward alpha data: {alpha_df.shape[0]} rows, date range {alpha_df['date'].min()} to {alpha_df['date'].max()}")

    # Compute correlations for all signals
    results_list = []

    for idx, signal_name in enumerate(signal_files):
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1} / {len(signal_files)} signals")

        try:
            signal = load_signal(signal_name)
            signal['date'] = pd.to_datetime(signal.index)

            corr_result = compute_signal_correlation(signal, alpha_df)

            if corr_result:
                results_list.append({
                    'signal_name': signal_name,
                    'correlation': corr_result['correlation'],
                    'p_value': corr_result['p_value'],
                    'n_valid': corr_result['n_valid'],
                    'category': categorize_signal(signal_name),
                })
        except Exception as e:
            print(f"  Error processing {signal_name}: {e}")

    results = pd.DataFrame(results_list).sort_values('correlation', ascending=False)

    print(f"\nSuccessfully analyzed {len(results)} / {len(signal_files)} signals")

    # Display top 30
    print("\n" + "="*130)
    print("TOP 30 SIGNALS BY CORRELATION WITH FORWARD IR")
    print("="*130)

    print(f"\n{'Rank':>4} {'Correlation':>12} {'P-Value':>12} {'Valid Pts':>10} {'Category':<20} {'Signal Name':<40}")
    print("-"*130)

    for idx, (_, row) in enumerate(results.head(30).iterrows(), 1):
        print(
            f"{idx:4d} {row['correlation']:12.4f} {row['p_value']:12.2e} {row['n_valid']:>10} "
            f"{row['category']:<20} {row['signal_name']:<40}"
        )

    # Performance by category
    print("\n" + "="*130)
    print("SIGNAL PERFORMANCE BY CATEGORY")
    print("="*130)

    category_stats = results.groupby('category')['correlation'].agg([
        ('Count', 'count'),
        ('Mean', 'mean'),
        ('Median', 'median'),
        ('Best', 'max'),
        ('Worst', 'min'),
    ]).sort_values('Best', ascending=False)

    print(f"\n{'Category':<20} {'Count':>6} {'Mean':>10} {'Median':>10} {'Best':>10} {'Worst':>10}")
    print("-"*130)

    for category, row in category_stats.iterrows():
        print(
            f"{category:<20} {int(row['Count']):>6} {row['Mean']:>10.4f} {row['Median']:>10.4f} "
            f"{row['Best']:>10.4f} {row['Worst']:>10.4f}"
        )

    # Bottom 20
    print("\n" + "="*130)
    print("BOTTOM 20 SIGNALS (WEAKEST CORRELATION)")
    print("="*130)

    print(f"\n{'Rank':>4} {'Correlation':>12} {'P-Value':>12} {'Category':<20} {'Signal Name':<40}")
    print("-"*130)

    for idx, (_, row) in enumerate(results.tail(20).iloc[::-1].iterrows(), 1):
        print(
            f"{idx:4d} {row['correlation']:12.4f} {row['p_value']:12.2e} "
            f"{row['category']:<20} {row['signal_name']:<40}"
        )

    # Statistics
    print("\n" + "="*130)
    print("OVERALL STATISTICS")
    print("="*130)

    print(f"""
Correlation Distribution:
  Mean:                    {results['correlation'].mean():.4f}
  Median:                  {results['correlation'].median():.4f}
  Std Dev:                 {results['correlation'].std():.4f}
  Min:                     {results['correlation'].min():.4f}
  Max:                     {results['correlation'].max():.4f}

Percentiles:
  90th percentile:         {results['correlation'].quantile(0.90):.4f}
  75th percentile:         {results['correlation'].quantile(0.75):.4f}
  50th percentile:         {results['correlation'].quantile(0.50):.4f}
  25th percentile:         {results['correlation'].quantile(0.25):.4f}
  10th percentile:         {results['correlation'].quantile(0.10):.4f}

Performance Tiers:
  Excellent (>0.15):       {(results['correlation'] > 0.15).sum():3d} signals ({(results['correlation'] > 0.15).sum()/len(results)*100:5.1f}%)
  Very Good (0.10-0.15):   {((results['correlation'] >= 0.10) & (results['correlation'] <= 0.15)).sum():3d} signals
  Good (0.05-0.10):        {((results['correlation'] >= 0.05) & (results['correlation'] < 0.10)).sum():3d} signals
  Fair (0.02-0.05):        {((results['correlation'] >= 0.02) & (results['correlation'] < 0.05)).sum():3d} signals
  Poor (<0.02):            {(results['correlation'] < 0.02).sum():3d} signals ({(results['correlation'] < 0.02).sum()/len(results)*100:5.1f}%)

Dominance Analysis:
  Best signal correlation: {results.iloc[0]['correlation']:.4f}
  Median correlation:      {results['correlation'].median():.4f}
  Top signal is {results.iloc[0]['correlation'] / results['correlation'].median():.1f}x better than median

  Top 5 signals:           {results.head(5)['correlation'].mean():.4f} (average)
  Top 10 signals:          {results.head(10)['correlation'].mean():.4f} (average)
  Top 30 signals:          {results.head(30)['correlation'].mean():.4f} (average)
  All signals:             {results['correlation'].mean():.4f} (average)
""")

    # Key findings
    print("\n" + "="*130)
    print("KEY FINDINGS")
    print("="*130)

    print(f"""
1. SIGNAL DOMINANCE:
   Top signal: {results.iloc[0]['signal_name']} (correlation: {results.iloc[0]['correlation']:.4f})

   Dominance ratios vs other top signals:
     vs 2nd:  {results.iloc[0]['correlation'] / results.iloc[1]['correlation']:.2f}x
     vs 5th:  {results.iloc[0]['correlation'] / results.iloc[4]['correlation']:.2f}x
     vs 10th: {results.iloc[0]['correlation'] / results.iloc[9]['correlation']:.2f}x

2. CATEGORY ANALYSIS:
   Strongest category:     {category_stats.index[0]} (best: {category_stats.iloc[0]['Best']:.4f})
   Weakest category:       {category_stats.index[-1]} (best: {category_stats.iloc[-1]['Best']:.4f})

3. NOISE REDUCTION OPPORTUNITY:
   {(results['correlation'] < 0.02).sum()} signals ({(results['correlation'] < 0.02).sum()/len(results)*100:.1f}%) have correlation < 0.02
   Removing these would cut signal complexity by {(results['correlation'] < 0.02).sum() / len(results) * 100:.0f}%

4. TIER ANALYSIS:
   - Top 1 signal explains ~{(results.head(1)['correlation'].sum() / results['correlation'].sum()) * 100:.1f}% of total correlation
   - Top 5 signals explain ~{(results.head(5)['correlation'].sum() / results['correlation'].sum()) * 100:.1f}% of total correlation
   - Top 10 signals explain ~{(results.head(10)['correlation'].sum() / results['correlation'].sum()) * 100:.1f}% of total correlation
   - Top 30 signals explain ~{(results.head(30)['correlation'].sum() / results['correlation'].sum()) * 100:.1f}% of total correlation
""")

    # Export results
    print("\n" + "="*130)
    print("EXPORTING RESULTS")
    print("="*130)

    csv_path = Path(__file__).parent / 'all_signal_correlations_detailed.csv'
    results.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")
    print(f"Total signals analyzed: {len(results)}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
