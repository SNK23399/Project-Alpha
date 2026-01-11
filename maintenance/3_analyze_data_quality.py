"""
Data Quality Analysis for ETF Price Data

This script analyzes the ETF price database to verify data quality before
computing signal bases. It checks for:
- Missing data / gaps
- Outliers and suspicious price movements
- Data coverage and completeness
- Pricing inconsistencies
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "support"))

import pandas as pd
import numpy as np
from etf_database import ETFDatabase


def analyze_data_quality(db_path="data/etf_database.db"):
    """Analyze ETF price data quality."""
    print("=" * 80)
    print(" " * 25 + "ETF DATA QUALITY ANALYSIS")
    print("=" * 80)
    print()

    db = ETFDatabase(db_path)

    # Load all prices
    print("Loading all ETF prices...")
    all_prices = db.load_all_prices()

    print(f"\nDatabase Overview:")
    print(f"  ETFs: {len(all_prices.columns)}")
    print(f"  Date range: {all_prices.index[0].date()} to {all_prices.index[-1].date()}")
    print(f"  Trading days: {len(all_prices)}")
    print(f"  Shape: {all_prices.shape}")
    print(f"  Memory: {all_prices.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # 1. Missing Data Analysis (from each ETF's actual start date)
    print("\n" + "=" * 80)
    print("1. MISSING DATA ANALYSIS")
    print("=" * 80)
    print("\n(Analyzing each ETF from its first available date, not from database start)")

    # Calculate missing % from each ETF's first non-null date
    missing_from_start = {}
    etf_start_dates = {}
    for isin in all_prices.columns:
        series = all_prices[isin].dropna()
        if len(series) > 0:
            start_date = series.index[0]
            etf_start_dates[isin] = start_date
            # Count trading days from start to end
            etf_data = all_prices.loc[start_date:, isin]
            total_days = len(etf_data)
            missing_days = etf_data.isna().sum()
            missing_from_start[isin] = (missing_days / total_days) * 100 if total_days > 0 else 0
        else:
            missing_from_start[isin] = 100.0

    missing_pct = pd.Series(missing_from_start)

    print(f"\nMissing data statistics (from each ETF's start date):")
    print(f"  Mean missing: {missing_pct.mean():.2f}%")
    print(f"  Median missing: {missing_pct.median():.2f}%")
    print(f"  Min missing: {missing_pct.min():.2f}%")
    print(f"  Max missing: {missing_pct.max():.2f}%")

    # ETFs with excessive missing data (>10% from their start date is concerning)
    high_missing = missing_pct[missing_pct > 10].sort_values(ascending=False)
    if len(high_missing) > 0:
        print(f"\n[WARN] {len(high_missing)} ETFs with >10% missing data (from their start date):")
        for isin, pct in high_missing.head(10).items():
            start = etf_start_dates.get(isin, 'N/A')
            start_str = start.strftime('%Y-%m-%d') if hasattr(start, 'strftime') else str(start)
            print(f"  {isin}: {pct:.1f}% missing (since {start_str})")
        if len(high_missing) > 10:
            print(f"  ... and {len(high_missing) - 10} more")
    else:
        print(f"\n[OK] No ETFs with >10% missing data from their start date")

    # Data coverage over time
    coverage_over_time = (~all_prices.isna()).sum(axis=1)
    print(f"\nData coverage over time:")
    print(f"  First date ({all_prices.index[0].date()}): {coverage_over_time.iloc[0]} ETFs")
    print(f"  Mid date ({all_prices.index[len(all_prices)//2].date()}): {coverage_over_time.iloc[len(all_prices)//2]} ETFs")
    print(f"  Last date ({all_prices.index[-1].date()}): {coverage_over_time.iloc[-1]} ETFs")

    # 2. Price Movement Analysis (outliers)
    print("\n" + "=" * 80)
    print("2. PRICE MOVEMENT ANALYSIS")
    print("=" * 80)

    # Calculate daily returns
    returns = all_prices.pct_change(fill_method=None)

    print(f"\nDaily return statistics:")
    print(f"  Mean: {returns.mean().mean():.4f}")
    print(f"  Median: {returns.median().median():.4f}")
    print(f"  Std: {returns.std().mean():.4f}")

    # Detect outliers (>10% daily moves)
    outlier_threshold = 0.10
    outliers = (returns.abs() > outlier_threshold)
    outlier_count = outliers.sum().sum()

    print(f"\nOutlier detection (|return| > {outlier_threshold*100:.0f}%):")
    print(f"  Total outlier days: {outlier_count}")
    print(f"  Percentage of all data: {outlier_count / returns.size * 100:.3f}%")

    # ETFs with most outliers
    outliers_per_etf = outliers.sum()
    top_outliers = outliers_per_etf[outliers_per_etf > 0].sort_values(ascending=False)
    if len(top_outliers) > 0:
        print(f"\nTop 10 ETFs with most outliers:")
        for isin, count in top_outliers.head(10).items():
            pct = count / (~all_prices[isin].isna()).sum() * 100
            print(f"  {isin}: {count} outliers ({pct:.2f}% of trading days)")
    else:
        print(f"\n[OK] No outliers detected")

    # 3. Data Availability by Time Period
    print("\n" + "=" * 80)
    print("3. DATA AVAILABILITY BY TIME PERIOD")
    print("=" * 80)

    # Count ETFs with data for different lookback periods
    lookback_periods = [21, 63, 126, 252, 504, 756, 1260]  # 1m, 3m, 6m, 1y, 2y, 3y, 5y

    print(f"\nETFs with sufficient history for signal windows:")
    for days in lookback_periods:
        # Count ETFs with at least 'days' of non-null data
        valid_counts = (~all_prices.tail(days * 2).isna()).sum()  # Look at last 2x days
        sufficient = (valid_counts >= days).sum()
        pct = sufficient / len(all_prices.columns) * 100

        years = days / 252
        print(f"  {days:4d} days ({years:4.1f}y): {sufficient:4d} ETFs ({pct:5.1f}%)")

    # 4. Price Level Analysis
    print("\n" + "=" * 80)
    print("4. PRICE LEVEL ANALYSIS")
    print("=" * 80)

    # Get most recent non-null prices
    latest_prices = all_prices.iloc[-1]
    latest_prices = latest_prices.dropna()

    print(f"\nLatest price statistics ({all_prices.index[-1].date()}):")
    print(f"  Mean: €{latest_prices.mean():.2f}")
    print(f"  Median: €{latest_prices.median():.2f}")
    print(f"  Min: €{latest_prices.min():.2f}")
    print(f"  Max: €{latest_prices.max():.2f}")

    # Very low prices (potential issues)
    low_price_threshold = 1.0
    low_prices = latest_prices[latest_prices < low_price_threshold]
    if len(low_prices) > 0:
        print(f"\n[WARN] {len(low_prices)} ETFs with price < €{low_price_threshold}:")
        for isin, price in low_prices.head(10).items():
            print(f"  {isin}: €{price:.4f}")
        if len(low_prices) > 10:
            print(f"  ... and {len(low_prices) - 10} more")
    else:
        print(f"\n[OK] No ETFs with price < €{low_price_threshold}")

    # Very high prices (potential errors)
    high_price_threshold = 1000.0
    high_prices = latest_prices[latest_prices > high_price_threshold]
    if len(high_prices) > 0:
        print(f"\n[WARN] {len(high_prices)} ETFs with price > €{high_price_threshold}:")
        for isin, price in high_prices.head(10).items():
            print(f"  {isin}: €{price:.2f}")
        if len(high_prices) > 10:
            print(f"  ... and {len(high_prices) - 10} more")
    else:
        print(f"\n[OK] No ETFs with price > €{high_price_threshold}")

    # 5. Core ETF Analysis (MSCI World)
    print("\n" + "=" * 80)
    print("5. CORE ETF ANALYSIS (MSCI WORLD)")
    print("=" * 80)

    core_isin = 'IE00B4L5Y983'  # iShares Core MSCI World
    if core_isin in all_prices.columns:
        core_prices = all_prices[core_isin].dropna()
        core_returns = core_prices.pct_change().dropna()

        print(f"\nCore ETF (IE00B4L5Y983 - iShares Core MSCI World):")
        print(f"  Data points: {len(core_prices)}")
        print(f"  Date range: {core_prices.index[0].date()} to {core_prices.index[-1].date()}")
        print(f"  Missing: {all_prices[core_isin].isna().sum()} days ({(all_prices[core_isin].isna().sum() / len(all_prices)) * 100:.2f}%)")
        print(f"  Latest price: €{core_prices.iloc[-1]:.2f}")
        print(f"  Ann. return: {((1 + core_returns.mean())**252 - 1) * 100:.2f}%")
        print(f"  Ann. volatility: {core_returns.std() * np.sqrt(252) * 100:.2f}%")
        print(f"  Sharpe ratio: {(core_returns.mean() / core_returns.std()) * np.sqrt(252):.2f}")

        print(f"\n[OK] Core ETF data looks good")
    else:
        print(f"\n[ERROR] Core ETF (IE00B4L5Y983) not found in database!")
        print(f"  This is required for alpha calculations!")

    # 6. Recommendations
    print("\n" + "=" * 80)
    print("6. RECOMMENDATIONS")
    print("=" * 80)

    recommendations = []

    # Check for high missing data
    if len(high_missing) > 0:
        recommendations.append(f"Consider excluding {len(high_missing)} ETFs with >50% missing data")

    # Check for outliers
    if outlier_count > 0:
        outliers_pct = outlier_count / returns.size * 100
        if outliers_pct > 0.1:
            recommendations.append(f"Review {outlier_count} outlier data points ({outliers_pct:.3f}% of data)")
        else:
            recommendations.append(f"Outliers present but within acceptable range ({outliers_pct:.3f}%)")

    # Check core ETF
    if core_isin not in all_prices.columns:
        recommendations.append(f"[CRITICAL] Core ETF (IE00B4L5Y983) missing - signal computation will fail!")
    elif all_prices[core_isin].isna().sum() / len(all_prices) > 0.1:
        recommendations.append(f"[WARN] Core ETF has {(all_prices[core_isin].isna().sum() / len(all_prices)) * 100:.1f}% missing data")

    # Check data coverage
    if coverage_over_time.iloc[-1] < len(all_prices.columns) * 0.5:
        recommendations.append(f"[WARN] Only {coverage_over_time.iloc[-1]} ETFs have recent data (expected {len(all_prices.columns)})")

    if len(recommendations) == 0:
        print("\n[OK] Data quality looks good! Ready to compute signals.")
    else:
        print("\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return {
        'n_etfs': len(all_prices.columns),
        'n_days': len(all_prices),
        'date_range': (all_prices.index[0], all_prices.index[-1]),
        'missing_pct': missing_pct,
        'high_missing': high_missing,
        'outlier_count': outlier_count,
        'core_available': core_isin in all_prices.columns,
        'recommendations': recommendations
    }


if __name__ == '__main__':
    results = analyze_data_quality()
