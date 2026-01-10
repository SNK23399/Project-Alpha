"""
Step 3: Compute Features from Filtered Signals

This script:
1. Loads filtered signals from the database
2. Computes 25 indicators (cross-sectional features)
3. Saves features for model training

Run this AFTER applying filters (step 2).

Note: Features are typically computed on-demand during model training
to avoid materializing the full 112,725 feature space (115 GB).
This script is for generating feature datasets for specific use cases.
"""

import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

from support.etf_database import ETFDatabase
from support.signal_database import SignalDatabase


def compute_cross_sectional_features(
    signals_2d: np.ndarray,
    feature_types: list = None
) -> np.ndarray:
    """
    Compute cross-sectional features (25 indicators) from filtered signals.

    Args:
        signals_2d: (n_time, n_etfs) array of filtered signal values
        feature_types: List of feature types to compute (None = all 25)

    Returns:
        features_2d: (n_time, n_features) array where n_features = len(feature_types)
    """
    n_time, n_etfs = signals_2d.shape

    if feature_types is None:
        feature_types = [
            'raw', 'rank', 'zscore', 'rank_norm',
            'pct_above_median', 'pct_in_top_quintile', 'pct_in_top_decile',
            'max_minus_median', 'median_minus_min', 'iqr',
            'cv', 'mad', 'gini',
            'diff_from_mean', 'diff_from_median', 'excess_vs_top10',
            'relative_to_leader', 'distance_to_avg',
            'winsorized_zscore', 'percentile',
            'binary_above_median', 'binary_top_quartile', 'binary_top_decile',
            'softmax', 'log_ratio_to_mean'
        ]

    features_list = []

    for time_idx in range(n_time):
        row = signals_2d[time_idx, :]

        # Handle NaN/inf
        valid_mask = np.isfinite(row)
        if valid_mask.sum() < 2:
            # Not enough valid data
            features_list.append(np.full(len(feature_types), np.nan))
            continue

        row_features = []

        for feat_type in feature_types:
            if feat_type == 'raw':
                # Raw signal value (use median as representative)
                feat = np.nanmedian(row)

            elif feat_type == 'rank':
                # Average rank across ETFs
                ranks = np.argsort(np.argsort(row))
                feat = np.mean(ranks[valid_mask])

            elif feat_type == 'zscore':
                # Z-score (standardized)
                mean_val = np.nanmean(row)
                std_val = np.nanstd(row)
                if std_val > 1e-10:
                    feat = np.nanmean((row - mean_val) / std_val)
                else:
                    feat = 0.0

            elif feat_type == 'rank_norm':
                # Normalized rank [0, 1]
                ranks = np.argsort(np.argsort(row))
                feat = np.mean(ranks[valid_mask] / (valid_mask.sum() - 1))

            elif feat_type == 'pct_above_median':
                # Percentage above median
                median_val = np.nanmedian(row)
                feat = np.mean(row[valid_mask] > median_val)

            elif feat_type == 'pct_in_top_quintile':
                # Percentage in top 20%
                threshold = np.nanpercentile(row, 80)
                feat = np.mean(row[valid_mask] >= threshold)

            elif feat_type == 'pct_in_top_decile':
                # Percentage in top 10%
                threshold = np.nanpercentile(row, 90)
                feat = np.mean(row[valid_mask] >= threshold)

            elif feat_type == 'max_minus_median':
                # Max - Median (spread)
                feat = np.nanmax(row) - np.nanmedian(row)

            elif feat_type == 'median_minus_min':
                # Median - Min (spread)
                feat = np.nanmedian(row) - np.nanmin(row)

            elif feat_type == 'iqr':
                # Interquartile range
                feat = np.nanpercentile(row, 75) - np.nanpercentile(row, 25)

            elif feat_type == 'cv':
                # Coefficient of variation
                mean_val = np.nanmean(row)
                std_val = np.nanstd(row)
                if abs(mean_val) > 1e-10:
                    feat = std_val / abs(mean_val)
                else:
                    feat = 0.0

            elif feat_type == 'mad':
                # Median absolute deviation
                median_val = np.nanmedian(row)
                feat = np.nanmedian(np.abs(row - median_val))

            elif feat_type == 'gini':
                # Gini coefficient (inequality measure)
                sorted_row = np.sort(row[valid_mask])
                n = len(sorted_row)
                index = np.arange(1, n + 1)
                feat = (2 * np.sum(index * sorted_row)) / (n * np.sum(sorted_row)) - (n + 1) / n

            elif feat_type == 'diff_from_mean':
                # Difference from mean
                feat = np.nanmean(row - np.nanmean(row))

            elif feat_type == 'diff_from_median':
                # Difference from median
                feat = np.nanmean(row - np.nanmedian(row))

            elif feat_type == 'excess_vs_top10':
                # Excess vs top 10%
                threshold = np.nanpercentile(row, 90)
                feat = np.nanmean(row) - threshold

            elif feat_type == 'relative_to_leader':
                # Ratio to max value
                max_val = np.nanmax(row)
                if max_val > 1e-10:
                    feat = np.nanmean(row) / max_val
                else:
                    feat = 1.0

            elif feat_type == 'distance_to_avg':
                # Average distance to mean
                mean_val = np.nanmean(row)
                feat = np.nanmean(np.abs(row - mean_val))

            elif feat_type == 'winsorized_zscore':
                # Winsorized z-score (clip at ±3σ)
                mean_val = np.nanmean(row)
                std_val = np.nanstd(row)
                if std_val > 1e-10:
                    zscore = (row - mean_val) / std_val
                    zscore_clipped = np.clip(zscore, -3, 3)
                    feat = np.nanmean(zscore_clipped)
                else:
                    feat = 0.0

            elif feat_type == 'percentile':
                # Mean percentile rank
                feat = np.nanmean([np.sum(row[valid_mask] <= val) / valid_mask.sum() * 100
                                  for val in row[valid_mask]])

            elif feat_type == 'binary_above_median':
                # Binary: above median
                median_val = np.nanmedian(row)
                feat = float(np.nanmean(row) > median_val)

            elif feat_type == 'binary_top_quartile':
                # Binary: in top quartile
                threshold = np.nanpercentile(row, 75)
                feat = float(np.nanmean(row) >= threshold)

            elif feat_type == 'binary_top_decile':
                # Binary: in top decile
                threshold = np.nanpercentile(row, 90)
                feat = float(np.nanmean(row) >= threshold)

            elif feat_type == 'softmax':
                # Softmax-weighted average
                row_shifted = row - np.nanmax(row)  # Numerical stability
                exp_vals = np.exp(row_shifted)
                exp_vals[~valid_mask] = 0
                softmax = exp_vals / np.sum(exp_vals)
                feat = np.sum(softmax * row)

            elif feat_type == 'log_ratio_to_mean':
                # Log ratio to mean
                mean_val = np.nanmean(row)
                if mean_val > 1e-10:
                    feat = np.nanmean(np.log(row / mean_val + 1e-10))
                else:
                    feat = 0.0

            else:
                feat = np.nan

            row_features.append(feat)

        features_list.append(row_features)

    return np.array(features_list)


def compute_and_save_features(
    signal_name: str,
    filter_name: str,
    start_date: str = None,
    end_date: str = None,
    output_format: str = 'parquet'
):
    """
    Compute features from a specific filtered signal and save to file.

    Args:
        signal_name: Signal base name (e.g., 'ret_1d')
        filter_name: Filter name (e.g., 'ema_21d')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_format: 'parquet' or 'csv'

    Returns:
        DataFrame with computed features
    """
    print("=" * 80)
    print(f"STEP 3: COMPUTE FEATURES - {signal_name} / {filter_name}")
    print("=" * 80)

    # Initialize database
    signal_db = SignalDatabase("data/etf_database.db")

    # Load filtered signal
    print("\nLoading filtered signal from database...")
    signals_2d, dates, isins = signal_db.load_filtered_signals(
        signal_name,
        filter_name,
        start_date=start_date,
        end_date=end_date,
        as_2d_array=True
    )

    if len(dates) == 0:
        print(f"ERROR: No data found for {signal_name}/{filter_name}")
        return None

    print(f"  Shape: {signals_2d.shape}")
    print(f"  Dates: {dates[0].date()} to {dates[-1].date()}")
    print(f"  ETFs: {len(isins)}")

    # Compute features
    print("\nComputing cross-sectional features...")
    feature_start = time.time()

    features_2d = compute_cross_sectional_features(signals_2d)

    feature_time = time.time() - feature_start

    print(f"  Computed {features_2d.shape[1]} features in {feature_time:.1f}s")

    # Create DataFrame
    feature_names = [
        'raw', 'rank', 'zscore', 'rank_norm',
        'pct_above_median', 'pct_in_top_quintile', 'pct_in_top_decile',
        'max_minus_median', 'median_minus_min', 'iqr',
        'cv', 'mad', 'gini',
        'diff_from_mean', 'diff_from_median', 'excess_vs_top10',
        'relative_to_leader', 'distance_to_avg',
        'winsorized_zscore', 'percentile',
        'binary_above_median', 'binary_top_quartile', 'binary_top_decile',
        'softmax', 'log_ratio_to_mean'
    ]

    df = pd.DataFrame(
        features_2d,
        index=dates,
        columns=[f"{signal_name}__{filter_name}__{feat}" for feat in feature_names]
    )

    # Save to file
    output_dir = Path("data/features")
    output_dir.mkdir(exist_ok=True, parents=True)

    if output_format == 'parquet':
        output_path = output_dir / f"{signal_name}__{filter_name}.parquet"
        df.to_parquet(output_path)
    elif output_format == 'csv':
        output_path = output_dir / f"{signal_name}__{filter_name}.csv"
        df.to_csv(output_path)
    else:
        raise ValueError(f"Unknown format: {output_format}")

    print(f"\n✓ Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024**2:.1f} MB")

    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python 3_compute_features.py <signal_name> <filter_name>")
        print("\nExample:")
        print("  python 3_compute_features.py ret_1d ema_21d")
        print("  python 3_compute_features.py vol_21d raw")
        print("\nNote: This computes features for a single signal/filter combination.")
        print("      Features are typically computed on-demand during model training")
        print("      to avoid materializing the full 112,725 feature space.")
        sys.exit(1)

    signal_name = sys.argv[1]
    filter_name = sys.argv[2]

    df = compute_and_save_features(signal_name, filter_name)

    if df is not None:
        print("\n" + "=" * 80)
        print("FEATURE COMPUTATION COMPLETE")
        print("=" * 80)
        print(f"\nFeature preview:")
        print(df.tail())
        print("\n✓ Features computed and saved successfully!")
