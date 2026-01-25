"""
Step 3: Compute Features from Filtered Signals (Walk-Forward Pipeline - ALL FEATURES VERSION)
==============================================================================================

This script:
1. Loads filtered signals from the database
2. Computes 25 cross-sectional indicators for EACH filtered signal
3. Saves features for model training

NOTE: This script is SHARED with the filtered version - feature computation
has no look-ahead bias (just transforms filtered signal data). Outputs go to:
    walk forward backtest all features/data/features/

With 7,911 filtered signals (step 2), this generates 7,911 × 25 = 197,775 features
(though many will be duplicates due to signal overlap). These are all available
for selection in step 4 (unlike the filtered version which pre-selects 500).
"""

import sys
import time
import pickle
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from support.signal_database import SignalDatabase
from library_signal_indicators import compute_all_indicators_fast, compute_all_indicators_batch_gpu, FEATURE_NAMES, GPU_AVAILABLE

# Output directories (relative to this script's location)
SIGNAL_INPUT_DIR = Path(__file__).parent / 'data' / 'signals'
FEATURE_OUTPUT_DIR = Path(__file__).parent / 'data' / 'features'


def _save_feature_file(save_args):
    """
    Worker function for parallel pickle saving.

    Args:
        save_args: Tuple of (output_path, feature_data, filtered_name)

    Returns:
        (filtered_name, success, error_msg)
    """
    output_path, feature_data, filtered_name = save_args
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(feature_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return (filtered_name, True, None)
    except Exception as e:
        return (filtered_name, False, str(e))


def compute_cross_sectional_features(
    signals_2d: np.ndarray,
    feature_types: list = None
) -> np.ndarray:
    """
    Compute per-ISIN indicators from filtered signals.

    CORRECTED: Now computes per-ISIN transformations (not market-wide aggregates).
    ULTRA-OPTIMIZED: Uses Numba JIT for 650x speedup over Python loops.

    Args:
        signals_2d: (n_time, n_etfs) array of filtered signal values
        feature_types: List of feature types to compute (None = all 25)

    Returns:
        features_3d: (n_time, n_etfs, 25) array where each ISIN gets its own values
    """
    # Use ultra-fast Numba JIT implementation
    features = compute_all_indicators_fast(signals_2d)

    # If specific features requested, select subset
    if feature_types is not None:
        indices = [FEATURE_NAMES.index(ft) for ft in feature_types if ft in FEATURE_NAMES]
        features = features[:, :, indices]

    return features


def compute_and_save_features(
    signal_name: str,
    filter_name: str,
    start_date: str = None,
    end_date: str = None,
    output_format: str = 'parquet',
    verbose: bool = False
):
    """
    Compute features from a specific filtered signal and save to file.

    Args:
        signal_name: Signal base name (e.g., 'ret_1d')
        filter_name: Filter name (e.g., 'ema_21d')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_format: 'parquet' or 'csv' (for compatibility, not used)
        verbose: Print detailed progress messages

    Returns:
        Tuple of (filtered_name, feature_data, error_msg) or (filtered_name, None, error_msg) if failed
    """
    filtered_name = f"{signal_name}__{filter_name}"

    try:
        if verbose:
            print("\nLoading filtered signal from database...")

        # Initialize database (parquet-based)
        signal_db = SignalDatabase(SIGNAL_INPUT_DIR)

        # Check if filtered signal exists
        if not signal_db.filtered_signal_exists(filtered_name):
            msg = f"Filtered signal not found: {filtered_name}"
            if verbose:
                print(f"ERROR: {msg}")
            return (filtered_name, None, msg)

        # Load as DataFrame
        df = signal_db.load_filtered_signal_by_name(
            filtered_name,
            start_date=start_date,
            end_date=end_date
        )

        if len(df) == 0:
            msg = f"No data found for {filtered_name}"
            if verbose:
                print(f"ERROR: {msg}")
            return (filtered_name, None, msg)

        # Convert to 2D array
        signals_2d = df.values
        dates = df.index
        isins = df.columns.tolist()

        if verbose:
            print(f"  Shape: {signals_2d.shape}")
            print(f"  Dates: {dates[0].date()} to {dates[-1].date()}")
            print(f"  ETFs: {len(isins)}")
            print("\nComputing per-ISIN indicators...")

        # Compute features (CORRECTED: now returns (dates, isins, 25))
        feature_start = time.time()
        features_3d = compute_cross_sectional_features(signals_2d)
        feature_time = time.time() - feature_start

        if verbose:
            print(f"  Shape: {features_3d.shape}")
            print(f"  Computed {features_3d.shape[2]} indicators × {features_3d.shape[1]} ISINs in {feature_time:.1f}s")
            print(f"  Total: {features_3d.shape[1] * features_3d.shape[2]} features per date")

        # Save to file
        output_dir = FEATURE_OUTPUT_DIR
        output_dir.mkdir(exist_ok=True, parents=True)

        # Store as dictionary with 3D array + metadata
        # Step 4 will load this and expand it into the rankings matrix
        feature_data = {
            'dates': dates,
            'isins': isins,
            'features_3d': features_3d,  # Shape: (dates, isins, 25)
            'feature_names': FEATURE_NAMES
        }

        output_path = output_dir / f"{signal_name}__{filter_name}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(feature_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        if verbose:
            print(f"\nSaved to: {output_path}")
            print(f"  File size: {output_path.stat().st_size / 1024**2:.1f} MB")
            print(f"  Structure: dict with keys ['dates', 'isins', 'features_3d', 'feature_names']")

        return (filtered_name, feature_data, None)

    except Exception as e:
        msg = str(e)
        if verbose:
            print(f"ERROR: {msg}")
        return (filtered_name, None, msg)


def _process_single_signal(args):
    """
    Worker function for multiprocessing.

    Args:
        args: Tuple of (signal_name, filter_name)

    Returns:
        Tuple of (filtered_name, success, error_msg)
    """
    signal_name, filter_name = args
    filtered_name, feature_data, error_msg = compute_and_save_features(
        signal_name, filter_name, verbose=False
    )
    return (filtered_name, feature_data is not None, error_msg)


def compute_and_save_features_batch_gpu(signal_filter_pairs: list) -> list:
    """
    Process multiple signals on GPU simultaneously for maximum efficiency.

    This batches signals together so GPU can parallelize across multiple signals
    at once, achieving 5-12x speedup per signal compared to single-signal processing.

    Args:
        signal_filter_pairs: List of (signal_name, filter_name) tuples

    Returns:
        List of (filtered_name, success, error_msg) tuples
    """
    if not GPU_AVAILABLE:
        # Fallback to single-signal processing
        results = []
        for signal_name, filter_name in signal_filter_pairs:
            filtered_name, feature_data, error_msg = compute_and_save_features(
                signal_name, filter_name, verbose=False
            )
            results.append((filtered_name, feature_data is not None, error_msg))
        return results

    signal_db = SignalDatabase(SIGNAL_INPUT_DIR)
    output_dir = FEATURE_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True, parents=True)

    results = []
    BATCH_SIZE = 10  # Process 10 signals at a time on GPU

    # Progress tracking for overall process
    pbar = tqdm(total=len(signal_filter_pairs), desc="Computing features", unit="signal",
                ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for batch_start in range(0, len(signal_filter_pairs), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(signal_filter_pairs))
        batch_pairs = signal_filter_pairs[batch_start:batch_end]

        try:
            # Load all signals in batch
            signals_list = []
            dates_list = []
            isins_list = []
            pair_list = []

            for signal_name, filter_name in batch_pairs:
                filtered_name = f"{signal_name}__{filter_name}"

                try:
                    if not signal_db.filtered_signal_exists(filtered_name):
                        results.append((filtered_name, False, "Filtered signal not found"))
                        pbar.update(1)
                        continue

                    df = signal_db.load_filtered_signal_by_name(filtered_name)
                    if df is None or df.empty:
                        results.append((filtered_name, False, "Empty signal"))
                        pbar.update(1)
                        continue

                    # Get dates and ISIN values
                    dates = df.index.tolist()
                    isins = df.columns.tolist()

                    # Convert to 2D array
                    signals_2d = df.values.astype(np.float32)

                    signals_list.append(signals_2d)
                    dates_list.append(dates)
                    isins_list.append(isins)
                    pair_list.append((signal_name, filter_name, filtered_name))

                except Exception as e:
                    results.append((filtered_name, False, str(e)))
                    pbar.update(1)
                    continue

            if not signals_list:
                pbar.update(len(batch_pairs))
                continue

            # Stack signals into batch: (n_signals, n_time, n_etfs)
            # Pad to same size if needed
            max_time = max(s.shape[0] for s in signals_list)
            max_etfs = max(s.shape[1] for s in signals_list)

            signals_batch = np.full(
                (len(signals_list), max_time, max_etfs),
                np.nan,
                dtype=np.float32
            )

            for i, signals_2d in enumerate(signals_list):
                signals_batch[i, :signals_2d.shape[0], :signals_2d.shape[1]] = signals_2d

            # Process entire batch on GPU at once
            pbar.set_description(f"GPU batch [{batch_start//BATCH_SIZE + 1}] Computing features")
            features_batch = compute_all_indicators_batch_gpu(signals_batch)

            # Prepare save tasks for parallel execution
            save_tasks = []
            for i, (signal_name, filter_name, filtered_name) in enumerate(pair_list):
                dates = dates_list[i]
                isins = isins_list[i]
                n_time = len(dates)
                n_etfs = len(isins)

                # Extract features for this signal (trim padding)
                features_3d = features_batch[i, :n_time, :n_etfs, :]

                feature_data = {
                    'dates': dates,
                    'isins': isins,
                    'features_3d': features_3d,
                    'feature_names': FEATURE_NAMES
                }

                output_path = output_dir / f"{filtered_name}.pkl"
                save_tasks.append((output_path, feature_data, filtered_name))

            # Save all signals in parallel using optimal CPU cores
            # Cap at 8 to avoid file handle and disk I/O contention
            pbar.set_description(f"GPU batch [{batch_start//BATCH_SIZE + 1}] Saving {len(save_tasks)} files")
            n_save_processes = min(8, cpu_count())  # Cap at 8 cores to avoid resource contention
            try:
                with Pool(n_save_processes) as save_pool:
                    save_results = list(save_pool.map(_save_feature_file, save_tasks))
            except Exception as e:
                # If parallel saving fails, fall back to sequential
                pbar.set_description(f"GPU batch [{batch_start//BATCH_SIZE + 1}] Saving {len(save_tasks)} files (fallback)")
                save_results = [_save_feature_file(task) for task in save_tasks]

            # Collect results and update progress
            for filtered_name, success, error_msg in save_results:
                if success:
                    results.append((filtered_name, True, None))
                else:
                    results.append((filtered_name, False, error_msg))
                pbar.update(1)

        except Exception as e:
            # Batch processing failed, fallback to individual processing
            pbar.set_description(f"GPU batch failed, fallback to CPU")
            for signal_name, filter_name in batch_pairs:
                filtered_name, feature_data, error_msg = compute_and_save_features(
                    signal_name, filter_name, verbose=False
                )
                results.append((filtered_name, feature_data is not None, error_msg))
                pbar.update(1)

    pbar.close()
    return results


def compute_all_features():
    """
    Compute features for ALL filtered signals using multi-processing.

    This processes all 7,325 filtered signals (293 signals × 25 filters)
    and generates ~5.1 GB of feature files.

    Uses multi-processing to parallelize across CPU cores for 2-3x speedup.
    """
    print("=" * 80)
    print("BATCH FEATURE COMPUTATION - ALL FILTERED SIGNALS (MULTI-PROCESS)")
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

    existing = set(f.stem for f in output_dir.glob("*.pkl"))
    remaining = [s for s in filtered_signals if s not in existing]

    if existing:
        print(f"\nAlready computed: {len(existing)} signals")
    print(f"Remaining: {len(remaining)} signals")

    if not remaining:
        print("\nAll features already computed!")
        return

    # Parse signal/filter pairs
    signal_filter_pairs = []
    for filtered_name in remaining:
        if '__' not in filtered_name:
            print(f"Skipping invalid name: {filtered_name}")
            continue
        signal_name, filter_name = filtered_name.rsplit('__', 1)
        signal_filter_pairs.append((signal_name, filter_name))

    start_time = time.time()
    errors = []
    n_processes = None
    use_gpu = GPU_AVAILABLE

    # Use GPU batching if available, otherwise CPU multi-processing
    if use_gpu:
        print(f"\nStarting GPU BATCHED processing...")
        print(f"Batch size: 10 signals per GPU batch")
        print("=" * 80 + "\n")

        # Use GPU batched processing (includes progress bar)
        results_list = compute_and_save_features_batch_gpu(signal_filter_pairs)

        # Extract errors
        for filtered_name, success, error_msg in results_list:
            if not success:
                errors.append((filtered_name, error_msg))
    else:
        print(f"\nStarting multi-process batch processing ({cpu_count()} cores)...")
        print("=" * 80)

        # Use multiprocessing with progress bar
        n_processes = max(1, cpu_count() - 1)  # Leave 1 core free

        with Pool(n_processes) as pool:
            # Use imap_unordered for better performance + progress feedback
            results = tqdm(
                pool.imap_unordered(_process_single_signal, signal_filter_pairs),
                total=len(signal_filter_pairs),
                desc="Computing features",
                unit="signal",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

            # Process results as they complete
            for filtered_name, success, error_msg in results:
                if not success:
                    errors.append((filtered_name, error_msg))

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nTotal time: {total_time / 60:.1f} minutes")
    print(f"Time per signal: {total_time / len(signal_filter_pairs):.2f} seconds")
    print(f"Processed: {len(signal_filter_pairs)} signals")
    print(f"Successful: {len(signal_filter_pairs) - len(errors)}")
    print(f"Errors: {len(errors)}")

    if use_gpu:
        print(f"Processing: GPU batched (10 signals per batch)")
    else:
        print(f"Processing: CPU multi-process ({n_processes} cores)")

    if errors:
        print("\nErrors encountered:")
        for name, error in errors[:10]:
            print(f"  - {name}: {error[:60]}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    else:
        print("\nAll features computed successfully!")


if __name__ == "__main__":
    import sys

    # No arguments = compute all features (multi-process)
    if len(sys.argv) == 1:
        compute_all_features()
        sys.exit(0)

    # Two arguments = compute single signal/filter (verbose mode)
    if len(sys.argv) == 3:
        signal_name = sys.argv[1]
        filter_name = sys.argv[2]

        print("=" * 80)
        print(f"STEP 3: COMPUTE FEATURES - {signal_name} / {filter_name}")
        print("=" * 80)

        filtered_name, feature_data, error_msg = compute_and_save_features(
            signal_name, filter_name, verbose=True
        )

        if feature_data is not None:
            print("\n" + "=" * 80)
            print("FEATURE COMPUTATION COMPLETE")
            print("=" * 80)
            print(f"\nFeature data structure:")
            print(f"  - dates: {len(feature_data['dates'])} timestamps")
            print(f"  - isins: {len(feature_data['isins'])} ISINs")
            print(f"  - features_3d shape: {feature_data['features_3d'].shape}")
            print(f"  - feature_names: {len(feature_data['feature_names'])} metrics")
            print("\nFeatures computed and saved successfully!")
        else:
            print(f"\nERROR: {error_msg}")
        sys.exit(0)

    # Invalid usage
    print("Usage:")
    print("  python 3_compute_features.py")
    print("  python 3_compute_features.py <signal_name> <filter_name>")
    print("\nExamples:")
    print("  python 3_compute_features.py                    # Compute ALL 7,325 signals (multi-process)")
    print("  python 3_compute_features.py ret_1d ema_21d     # Single signal (verbose)")
    print("  python 3_compute_features.py vol_21d raw        # Single signal (verbose)")
    print("\nNote: No arguments = batch mode (all filtered signals, multi-process, ~5.1 GB)")
    print("      Two arguments = single signal mode (verbose output)")
    sys.exit(1)
