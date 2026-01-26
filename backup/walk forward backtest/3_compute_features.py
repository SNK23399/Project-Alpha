"""
Step 3: Compute Features from Filtered Signals (Walk-Forward Pipeline)
=======================================================================

This script:
1. Loads filtered signals from the database
2. Computes 25 indicators (cross-sectional features)
3. Saves features for model training

NOTE: This script is identical to the original - feature computation
has no look-ahead bias (just transforms filtered signal data).
"""

import sys
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from support.signal_database import SignalDatabase
from library_signal_indicators import compute_all_indicators_fast, FEATURE_NAMES

# Output directories (relative to this script's location)
SIGNAL_INPUT_DIR = Path(__file__).parent / 'data' / 'signals'
FEATURE_OUTPUT_DIR = Path(__file__).parent / 'data' / 'features'


def compute_cross_sectional_features(
    signals_2d: np.ndarray,
    feature_types: list = None
) -> np.ndarray:
    """
    Compute cross-sectional features (25 indicators) from filtered signals.

    ULTRA-OPTIMIZED: Uses Numba JIT for 650x speedup over Python loops.

    Args:
        signals_2d: (n_time, n_etfs) array of filtered signal values
        feature_types: List of feature types to compute (None = all 25)

    Returns:
        features_2d: (n_time, n_features) array where n_features = len(feature_types)
    """
    # Use ultra-fast Numba JIT implementation
    features = compute_all_indicators_fast(signals_2d)

    # If specific features requested, select subset
    if feature_types is not None:
        indices = [FEATURE_NAMES.index(ft) for ft in feature_types if ft in FEATURE_NAMES]
        features = features[:, indices]

    return features


def compute_and_save_features(
    signal_name: str,
    filter_name: str,
    start_date: str = None,
    end_date: str = None,
    output_format: str = 'parquet',
    verbose: bool = True
):
    """
    Compute features from a specific filtered signal and save to file.

    Args:
        signal_name: Signal base name (e.g., 'ret_1d')
        filter_name: Filter name (e.g., 'ema_21d')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_format: 'parquet' or 'csv'
        verbose: Print detailed progress messages

    Returns:
        DataFrame with computed features
    """
    if verbose:
        print("=" * 80)
        print(f"STEP 3: COMPUTE FEATURES - {signal_name} / {filter_name}")
        print("=" * 80)

    # Initialize database (parquet-based)
    signal_db = SignalDatabase(SIGNAL_INPUT_DIR)

    # Load filtered signal using the new API
    if verbose:
        print("\nLoading filtered signal from database...")
    filtered_name = f"{signal_name}__{filter_name}"

    # Check if filtered signal exists
    if not signal_db.filtered_signal_exists(filtered_name):
        if verbose:
            print(f"ERROR: Filtered signal not found: {filtered_name}")
            print(f"       Run 2_apply_filters.py first.")
        return None

    # Load as DataFrame
    df = signal_db.load_filtered_signal_by_name(
        filtered_name,
        start_date=start_date,
        end_date=end_date
    )

    if len(df) == 0:
        if verbose:
            print(f"ERROR: No data found for {filtered_name}")
        return None

    # Convert to 2D array
    signals_2d = df.values
    dates = df.index
    isins = df.columns.tolist()

    if verbose:
        print(f"  Shape: {signals_2d.shape}")
        print(f"  Dates: {dates[0].date()} to {dates[-1].date()}")
        print(f"  ETFs: {len(isins)}")
        print("\nComputing cross-sectional features...")

    # Compute features
    feature_start = time.time()
    features_2d = compute_cross_sectional_features(signals_2d)
    feature_time = time.time() - feature_start

    if verbose:
        print(f"  Computed {features_2d.shape[1]} features in {feature_time:.1f}s")

    # Create DataFrame (use FEATURE_NAMES from library)
    df = pd.DataFrame(
        features_2d,
        index=dates,
        columns=[f"{signal_name}__{filter_name}__{feat}" for feat in FEATURE_NAMES]
    )

    # Save to file
    output_dir = FEATURE_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True, parents=True)

    if output_format == 'parquet':
        output_path = output_dir / f"{signal_name}__{filter_name}.parquet"
        df.to_parquet(output_path)
    elif output_format == 'csv':
        output_path = output_dir / f"{signal_name}__{filter_name}.csv"
        df.to_csv(output_path)
    else:
        raise ValueError(f"Unknown format: {output_format}")

    if verbose:
        print(f"\nSaved to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024**2:.1f} MB")

    return df


def compute_all_features():
    """
    Compute features for ALL filtered signals.

    This processes all 7,325 filtered signals (293 signals × 25 filters)
    and generates ~5.1 GB of feature files.
    """
    print("=" * 80)
    print("BATCH FEATURE COMPUTATION - ALL FILTERED SIGNALS")
    print("=" * 80)

    # Initialize database
    signal_db = SignalDatabase(SIGNAL_INPUT_DIR)

    # Get all filtered signals
    filtered_signals = sorted(signal_db.get_completed_filtered_signals())

    print(f"\nFound {len(filtered_signals)} filtered signals")
    print(f"This will generate ~{len(filtered_signals) * 0.7:.1f} MB of feature files")

    # Check what's already done
    output_dir = FEATURE_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True, parents=True)

    existing = set(f.stem for f in output_dir.glob("*.parquet"))
    remaining = [s for s in filtered_signals if s not in existing]

    if existing:
        print(f"\nAlready computed: {len(existing)} signals")
    print(f"Remaining: {len(remaining)} signals")

    if not remaining:
        print("\nAll features already computed!")
        return

    # Process all remaining with progress bar
    print(f"\nStarting batch processing...")
    print("=" * 80)

    start_time = time.time()
    errors = []

    # Create progress bar
    pbar = tqdm(remaining, desc="Computing features", unit="signal",
                ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for filtered_name in pbar:
        # Parse signal and filter name
        if '__' not in filtered_name:
            pbar.write(f"Skipping invalid name: {filtered_name}")
            continue

        signal_name, filter_name = filtered_name.rsplit('__', 1)

        # Update progress bar description
        pbar.set_description(f"Processing {signal_name[:20]:20s}")

        try:
            df = compute_and_save_features(signal_name, filter_name, output_format='parquet', verbose=False)
            if df is None:
                errors.append((filtered_name, "Failed to compute"))
                pbar.write(f"  ERROR: {filtered_name} - Failed to compute")
        except Exception as e:
            errors.append((filtered_name, str(e)))
            pbar.write(f"  ERROR: {filtered_name} - {str(e)[:50]}")

    pbar.close()

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nTotal time: {total_time / 60:.1f} minutes")
    print(f"Processed: {len(remaining)} signals")
    print(f"Successful: {len(remaining) - len(errors)}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nErrors encountered:")
        for name, error in errors[:10]:
            print(f"  - {name}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    else:
        print("\nAll features computed successfully!")


if __name__ == "__main__":
    import sys

    # No arguments = compute all features
    if len(sys.argv) == 1:
        compute_all_features()
        sys.exit(0)

    # Two arguments = compute single signal/filter
    if len(sys.argv) == 3:
        signal_name = sys.argv[1]
        filter_name = sys.argv[2]

        df = compute_and_save_features(signal_name, filter_name)

        if df is not None:
            print("\n" + "=" * 80)
            print("FEATURE COMPUTATION COMPLETE")
            print("=" * 80)
            print(f"\nFeature preview:")
            print(df.tail())
            print("\nFeatures computed and saved successfully!")
        sys.exit(0)

    # Invalid usage
    print("Usage:")
    print("  python 3_compute_features.py")
    print("  python 3_compute_features.py <signal_name> <filter_name>")
    print("\nExamples:")
    print("  python 3_compute_features.py                    # Compute ALL 7,325 signals")
    print("  python 3_compute_features.py ret_1d ema_21d     # Single signal")
    print("  python 3_compute_features.py vol_21d raw        # Single signal")
    print("\nNote: No arguments = batch mode (all filtered signals, ~5.1 GB)")
    print("      Two arguments = single signal mode")
    sys.exit(1)
