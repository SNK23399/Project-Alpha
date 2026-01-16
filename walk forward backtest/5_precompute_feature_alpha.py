"""
Step 5: Precompute Feature-Alpha Matrix (Walk-Forward Pipeline)
================================================================

OPTIMIZED VERSION with:
- 2D numpy alpha matrix for O(1) lookups (instead of pandas .loc)
- Numba JIT compilation for the hot loop (~100x speedup)
- Vectorized operations where possible
- Parallel processing across features
- Support for multiple horizons in a single run

This script precomputes the alpha that each feature would achieve at each
date for each N value. This allows the walk-forward backtest to run very
fast by avoiding repeated computations.

The output is a 3D matrix: (n_dates, n_features, n_satellites_max)
where each cell contains the average alpha achieved by selecting top-N
ETFs according to that feature at that date.

This is the key optimization that allows testing many N values quickly.

Output:
    data/feature_alpha_{holding_months}month.npz

Usage:
    python 5_precompute_feature_alpha.py [horizons]

Examples:
    python 5_precompute_feature_alpha.py              # Default: 1 month horizon
    python 5_precompute_feature_alpha.py 3            # Single: 3 month horizon
    python 5_precompute_feature_alpha.py 1,2,3,4,5,6  # Multiple horizons
    python 5_precompute_feature_alpha.py 1-6          # Range: 1 to 6 months
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import numba for JIT compilation
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: numba not installed. Install with: pip install numba")


# ============================================================
# CONFIGURATION
# ============================================================

# N values to precompute (1-10)
N_SATELLITES_MAX = 10

# Horizons to compute (list of months)
# Set to single value [1] for original behavior, or multiple [1,2,3,4,5,6] for multi-horizon
HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Default holding period (used if HORIZONS is empty or for backward compatibility)
DEFAULT_HOLDING_MONTHS = 1

# Force recompute
FORCE_RECOMPUTE = False

# Number of CPU cores for parallel processing
N_CORES = max(1, cpu_count() - 1)

# Output directory (relative to this script's location)
OUTPUT_DIR = Path(__file__).parent / 'data'


# ============================================================
# NUMBA-OPTIMIZED CORE COMPUTATION
# ============================================================

if HAS_NUMBA:
    @njit(cache=True)
    def _compute_feature_alpha_for_date_numba(
        rankings_date: np.ndarray,      # (n_isins, n_features)
        alpha_date: np.ndarray,         # (n_isins,) - alpha values for this date
        alpha_valid: np.ndarray,        # (n_isins,) - float32 (1.0=valid, 0.0=invalid)
        n_satellites_max: int
    ) -> tuple:
        """
        Numba-optimized computation for a single date.

        For each feature, sort ISINs by ranking and compute average alpha
        for top-1, top-2, ..., top-N selections.

        Returns:
            feature_alpha: (n_features, n_satellites_max)
            feature_hit: (n_features, n_satellites_max)
        """
        n_isins, n_features = rankings_date.shape

        feature_alpha = np.full((n_features, n_satellites_max), np.nan, dtype=np.float32)
        feature_hit = np.full((n_features, n_satellites_max), np.nan, dtype=np.float32)

        for feat_idx in range(n_features):
            feat_rankings = rankings_date[:, feat_idx]

            # Count valid rankings (not NaN)
            valid_count = 0
            for i in range(n_isins):
                if not np.isnan(feat_rankings[i]):
                    valid_count += 1

            if valid_count < n_satellites_max:
                continue

            # Create arrays of valid (ranking, isin_idx) pairs
            valid_rankings = np.empty(valid_count, dtype=np.float32)
            valid_indices = np.empty(valid_count, dtype=np.int64)
            idx = 0
            for i in range(n_isins):
                if not np.isnan(feat_rankings[i]):
                    valid_rankings[idx] = feat_rankings[i]
                    valid_indices[idx] = i
                    idx += 1

            # Sort by ranking descending (higher = better)
            sort_order = np.argsort(valid_rankings)[::-1]
            sorted_indices = valid_indices[sort_order]

            # Compute cumulative average alpha for top-1, top-2, ..., top-N
            alpha_sum = np.float64(0.0)
            alpha_count = 0

            for n in range(n_satellites_max):
                if n >= len(sorted_indices):
                    break

                isin_idx = sorted_indices[n]

                # Check if this ISIN has valid alpha (alpha_valid is 1.0 for valid)
                if alpha_valid[isin_idx] > 0.5:
                    alpha_sum += alpha_date[isin_idx]
                    alpha_count += 1

                if alpha_count > 0:
                    avg_alpha = alpha_sum / alpha_count
                    feature_alpha[feat_idx, n] = np.float32(avg_alpha)
                    feature_hit[feat_idx, n] = np.float32(1.0) if avg_alpha > 0 else np.float32(0.0)

        return feature_alpha, feature_hit


    @njit(parallel=True, cache=True)
    def _compute_all_dates_numba(
        rankings: np.ndarray,           # (n_dates, n_isins, n_features)
        alpha_matrix: np.ndarray,       # (n_dates, n_isins)
        alpha_valid_matrix: np.ndarray, # (n_dates, n_isins) - float32 (1.0=valid, 0.0=invalid)
        n_satellites_max: int
    ) -> tuple:
        """
        Numba-optimized parallel computation across all dates.

        NOTE: alpha_valid_matrix should be float32 (1.0 for valid, 0.0 for invalid)
        to avoid numba issues with boolean arrays.

        Returns:
            feature_alpha: (n_dates, n_features, n_satellites_max)
            feature_hit: (n_dates, n_features, n_satellites_max)
        """
        n_dates, n_isins, n_features = rankings.shape

        feature_alpha = np.full((n_dates, n_features, n_satellites_max), np.nan, dtype=np.float32)
        feature_hit = np.full((n_dates, n_features, n_satellites_max), np.nan, dtype=np.float32)

        # Parallel loop over dates
        for date_idx in prange(n_dates):
            # Get data for this date
            rankings_date = rankings[date_idx, :, :]
            alpha_date = alpha_matrix[date_idx, :]
            alpha_valid = alpha_valid_matrix[date_idx, :]

            # Process each feature
            for feat_idx in range(n_features):
                feat_rankings = rankings_date[:, feat_idx]

                # Count valid rankings
                valid_count = 0
                for i in range(n_isins):
                    if not np.isnan(feat_rankings[i]):
                        valid_count += 1

                if valid_count < n_satellites_max:
                    continue

                # Create arrays of valid (ranking, isin_idx) pairs
                valid_rankings = np.empty(valid_count, dtype=np.float32)
                valid_indices = np.empty(valid_count, dtype=np.int64)
                idx = 0
                for i in range(n_isins):
                    if not np.isnan(feat_rankings[i]):
                        valid_rankings[idx] = feat_rankings[i]
                        valid_indices[idx] = i
                        idx += 1

                # Sort by ranking descending
                sort_order = np.argsort(valid_rankings)[::-1]
                sorted_indices = valid_indices[sort_order]

                # Compute cumulative average alpha
                alpha_sum = np.float64(0.0)
                alpha_count = 0

                for n in range(n_satellites_max):
                    if n >= len(sorted_indices):
                        break

                    isin_idx = sorted_indices[n]

                    # Check if this ISIN has valid alpha (alpha_valid is 1.0 for valid)
                    if alpha_valid[isin_idx] > 0.5:
                        alpha_sum += alpha_date[isin_idx]
                        alpha_count += 1

                    if alpha_count > 0:
                        avg_alpha = alpha_sum / alpha_count
                        feature_alpha[date_idx, feat_idx, n] = np.float32(avg_alpha)
                        feature_hit[date_idx, feat_idx, n] = np.float32(1.0) if avg_alpha > 0 else np.float32(0.0)

        return feature_alpha, feature_hit

else:
    # Fallback without numba - vectorized numpy version
    def _compute_all_dates_numba(rankings, alpha_matrix, alpha_valid_matrix, n_satellites_max):
        """Fallback implementation without numba."""
        return _compute_all_dates_numpy(rankings, alpha_matrix, alpha_valid_matrix, n_satellites_max)


def _compute_all_dates_numpy(rankings, alpha_matrix, alpha_valid_matrix, n_satellites_max):
    """
    Numpy-vectorized fallback (used if numba not available).
    Still much faster than pure Python due to vectorized operations.
    """
    n_dates, n_isins, n_features = rankings.shape

    feature_alpha = np.full((n_dates, n_features, n_satellites_max), np.nan, dtype=np.float32)
    feature_hit = np.full((n_dates, n_features, n_satellites_max), np.nan, dtype=np.float32)

    for date_idx in tqdm(range(n_dates), desc="Processing dates"):
        rankings_date = rankings[date_idx]  # (n_isins, n_features)
        alpha_date = alpha_matrix[date_idx]  # (n_isins,)
        alpha_valid = alpha_valid_matrix[date_idx]  # (n_isins,)

        for feat_idx in range(n_features):
            feat_rankings = rankings_date[:, feat_idx]
            valid_mask = ~np.isnan(feat_rankings)

            if valid_mask.sum() < n_satellites_max:
                continue

            # Get valid indices sorted by ranking (descending)
            valid_indices = np.where(valid_mask)[0]
            valid_rankings = feat_rankings[valid_mask]
            sort_order = np.argsort(valid_rankings)[::-1]
            sorted_indices = valid_indices[sort_order]

            # Compute cumulative alpha for top-N
            alpha_sum = 0.0
            alpha_count = 0

            for n in range(min(n_satellites_max, len(sorted_indices))):
                isin_idx = sorted_indices[n]

                if alpha_valid[isin_idx]:
                    alpha_sum += alpha_date[isin_idx]
                    alpha_count += 1

                if alpha_count > 0:
                    avg_alpha = alpha_sum / alpha_count
                    feature_alpha[date_idx, feat_idx, n] = avg_alpha
                    feature_hit[date_idx, feat_idx, n] = 1.0 if avg_alpha > 0 else 0.0

    return feature_alpha, feature_hit


# ============================================================
# DATA LOADING AND PREPARATION
# ============================================================

def load_data(holding_months):
    """Load forward alpha and rankings matrix."""
    print(f"\n{'='*60}")
    print(f"LOADING DATA")
    print(f"{'='*60}")

    # Load forward alpha
    alpha_file = OUTPUT_DIR / f'forward_alpha_{holding_months}month.parquet'
    if not alpha_file.exists():
        raise FileNotFoundError(f"Forward alpha not found: {alpha_file}\nRun 4_compute_forward_alpha.py first.")

    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])

    print(f"\n[OK] Forward alpha loaded ({len(alpha_df):,} observations)")

    # Load rankings matrix
    rankings_file = OUTPUT_DIR / f'rankings_matrix_{holding_months}month.npz'
    if not rankings_file.exists():
        raise FileNotFoundError(f"Rankings matrix not found: {rankings_file}\nRun 4_compute_forward_alpha.py first.")

    npz_data = np.load(rankings_file, allow_pickle=True)
    rankings = npz_data['rankings'].copy()
    dates = pd.to_datetime(npz_data['dates'])
    isins = npz_data['isins']
    feature_names = npz_data['features']

    print(f"[OK] Rankings matrix loaded {rankings.shape}")
    print(f"     {len(dates)} dates, {len(isins)} ISINs, {len(feature_names)} features")

    return alpha_df, rankings, dates, isins, feature_names


def prepare_alpha_matrix(alpha_df, dates, isins):
    """
    Convert alpha DataFrame to 2D numpy arrays for fast O(1) lookups.

    Returns:
        alpha_matrix: (n_dates, n_isins) - alpha values
        alpha_valid: (n_dates, n_isins) - boolean mask of valid entries
    """
    print(f"\nPreparing alpha matrix for fast lookups...")

    n_dates = len(dates)
    n_isins = len(isins)

    # Create ISIN to index mapping
    isin_to_idx = {isin: idx for idx, isin in enumerate(isins)}

    # Create date to index mapping
    date_to_idx = {date: idx for idx, date in enumerate(dates)}

    # Initialize matrices
    alpha_matrix = np.full((n_dates, n_isins), np.nan, dtype=np.float32)

    # Fill matrix using vectorized operations where possible
    # Group by date for efficiency
    for date, group in tqdm(alpha_df.groupby('date'), desc="Building alpha matrix", total=len(dates)):
        if date not in date_to_idx:
            continue

        date_idx = date_to_idx[date]

        # Vectorized: map ISINs to indices and fill
        valid_isins = group['isin'].values
        valid_alphas = group['forward_alpha'].values

        for isin, alpha in zip(valid_isins, valid_alphas):
            if isin in isin_to_idx:
                isin_idx = isin_to_idx[isin]
                alpha_matrix[date_idx, isin_idx] = alpha

    # Create validity mask
    alpha_valid = ~np.isnan(alpha_matrix)

    print(f"  Alpha matrix shape: {alpha_matrix.shape}")
    print(f"  Valid entries: {alpha_valid.sum():,} ({alpha_valid.sum() / alpha_valid.size * 100:.1f}%)")

    return alpha_matrix, alpha_valid


# ============================================================
# MAIN COMPUTATION
# ============================================================

def precompute_feature_alpha(alpha_df, rankings, dates, isins, feature_names):
    """
    Precompute alpha for each (date, feature, N) combination.

    OPTIMIZED: Uses numba JIT and 2D alpha matrix for ~50-100x speedup.

    Returns:
        feature_alpha: (n_dates, n_features, N_SATELLITES_MAX) array
                      Each cell = average alpha of top-N ETFs by that feature
        feature_hit:   (n_dates, n_features, N_SATELLITES_MAX) array
                      Each cell = 1 if alpha > 0, else 0
    """
    print(f"\n{'='*60}")
    print(f"PRECOMPUTING FEATURE-ALPHA MATRIX")
    print(f"{'='*60}")

    n_dates = len(dates)
    n_features = len(feature_names)

    print(f"\nMatrix dimensions:")
    print(f"  Dates: {n_dates}")
    print(f"  Features: {n_features}")
    print(f"  N values: 1-{N_SATELLITES_MAX}")
    print(f"  Total cells: {n_dates * n_features * N_SATELLITES_MAX:,}")
    print(f"\nOptimization: {'Numba JIT (parallel)' if HAS_NUMBA else 'NumPy vectorized'}")

    # Prepare alpha matrix for fast lookups
    alpha_matrix, alpha_valid = prepare_alpha_matrix(alpha_df, dates, isins)

    # Run optimized computation
    print(f"\nComputing feature-alpha matrix...")

    if HAS_NUMBA:
        # Convert boolean alpha_valid to float32 (1.0 for True, 0.0 for False)
        # This avoids numba issues with boolean array indexing
        alpha_valid_float = alpha_valid.astype(np.float32)

        # Warm up JIT compiler
        print("  Warming up JIT compiler...")
        _ = _compute_all_dates_numba(
            rankings[:2].astype(np.float32),
            alpha_matrix[:2].astype(np.float32),
            alpha_valid_float[:2],
            N_SATELLITES_MAX
        )

        print("  Running parallel computation...")
        feature_alpha, feature_hit = _compute_all_dates_numba(
            rankings.astype(np.float32),
            alpha_matrix.astype(np.float32),
            alpha_valid_float,
            N_SATELLITES_MAX
        )
    else:
        feature_alpha, feature_hit = _compute_all_dates_numpy(
            rankings,
            alpha_matrix,
            alpha_valid,
            N_SATELLITES_MAX
        )

    # Statistics
    non_nan = (~np.isnan(feature_alpha)).sum()
    total = feature_alpha.size
    coverage = non_nan / total * 100

    print(f"\nMatrix statistics:")
    print(f"  Non-NaN cells: {non_nan:,} / {total:,} ({coverage:.1f}%)")
    print(f"  Mean alpha (all): {np.nanmean(feature_alpha):.4f}")
    print(f"  Hit rate (all): {np.nanmean(feature_hit):.2%}")

    return feature_alpha, feature_hit


def save_feature_alpha(feature_alpha, feature_hit, dates, feature_names, holding_months):
    """Save the precomputed matrices."""
    output_file = OUTPUT_DIR / f'feature_alpha_{holding_months}month.npz'

    np.savez_compressed(
        output_file,
        feature_alpha=feature_alpha,
        feature_hit=feature_hit,
        dates=dates,
        features=feature_names,
        n_satellites_max=N_SATELLITES_MAX
    )

    print(f"\n[SAVED] {output_file}")
    print(f"        Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


# ============================================================
# MAIN
# ============================================================

def parse_horizons(arg_string):
    """
    Parse horizon argument string into list of integers.

    Supports:
        - Single value: "3" -> [3]
        - Comma-separated: "1,2,3,4,5,6" -> [1,2,3,4,5,6]
        - Range: "1-6" -> [1,2,3,4,5,6]
        - Combined: "1,3-5,7" -> [1,3,4,5,7]
    """
    horizons = []
    parts = arg_string.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range notation
            start, end = part.split('-')
            horizons.extend(range(int(start), int(end) + 1))
        else:
            horizons.append(int(part))

    return sorted(set(horizons))  # Remove duplicates and sort


def main(holding_months=DEFAULT_HOLDING_MONTHS):
    """Run the precomputation for a single horizon."""
    print("=" * 60)
    print(f"PRECOMPUTE FEATURE-ALPHA MATRIX - {holding_months} MONTH HORIZON")
    print("=" * 60)

    output_file = OUTPUT_DIR / f'feature_alpha_{holding_months}month.npz'

    if output_file.exists() and not FORCE_RECOMPUTE:
        print(f"\n[EXISTS] {output_file}")
        print("Use FORCE_RECOMPUTE=True to regenerate.")

        # Load and show stats
        data = np.load(output_file, allow_pickle=True)
        print(f"\nExisting matrix shape: {data['feature_alpha'].shape}")
        print(f"Mean alpha: {np.nanmean(data['feature_alpha']):.4f}")
        print(f"Hit rate: {np.nanmean(data['feature_hit']):.2%}")
        return

    # Load data
    alpha_df, rankings, dates, isins, feature_names = load_data(holding_months)

    # Precompute
    feature_alpha, feature_hit = precompute_feature_alpha(
        alpha_df, rankings, dates, isins, feature_names
    )

    # Save
    save_feature_alpha(feature_alpha, feature_hit, dates, feature_names, holding_months)

    print("\n" + "=" * 60)
    print("PRECOMPUTATION COMPLETE")
    print("=" * 60)
    print(f"\nNext step: Run 6_fast_backtest.py for walk-forward backtesting")


def main_multi_horizon(horizons):
    """Run the precomputation for multiple horizons."""
    import time

    print("=" * 60)
    print(f"PRECOMPUTE FEATURE-ALPHA MATRIX - MULTI-HORIZON")
    print("=" * 60)
    print(f"\nHorizons to compute: {horizons}")
    print(f"Total: {len(horizons)} horizons")

    total_start = time.time()
    results = {}

    for i, holding_months in enumerate(horizons, 1):
        print(f"\n{'#' * 60}")
        print(f"# HORIZON {i}/{len(horizons)}: {holding_months} MONTH(S)")
        print(f"{'#' * 60}")

        output_file = OUTPUT_DIR / f'feature_alpha_{holding_months}month.npz'

        if output_file.exists() and not FORCE_RECOMPUTE:
            print(f"\n[EXISTS] {output_file}")
            print("Skipping (use FORCE_RECOMPUTE=True to regenerate)")

            # Load and show stats
            data = np.load(output_file, allow_pickle=True)
            print(f"  Matrix shape: {data['feature_alpha'].shape}")
            print(f"  Mean alpha: {np.nanmean(data['feature_alpha']):.4f}")
            print(f"  Hit rate: {np.nanmean(data['feature_hit']):.2%}")
            continue

        horizon_start = time.time()

        # Load data
        alpha_df, rankings, dates, isins, feature_names = load_data(holding_months)

        # Precompute
        feature_alpha, feature_hit = precompute_feature_alpha(
            alpha_df, rankings, dates, isins, feature_names
        )

        # Save
        save_feature_alpha(feature_alpha, feature_hit, dates, feature_names, holding_months)

        horizon_time = time.time() - horizon_start
        print(f"\n[DONE] {holding_months}-month horizon completed in {horizon_time:.1f}s")

        results[holding_months] = {
            'feature_alpha': feature_alpha,
            'feature_hit': feature_hit
        }

    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("MULTI-HORIZON PRECOMPUTATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    for h in horizons:
        print(f"  feature_alpha_{h}month.npz")
    print("\nNext step: Run 7_multi_horizon_backtest.py for consensus backtesting")
    print(f"           Or run 6_fast_backtest.py for single-horizon backtesting")

    return results


if __name__ == '__main__':
    # Priority: command line arg > HORIZONS config > DEFAULT_HOLDING_MONTHS
    if len(sys.argv) > 1:
        horizons = parse_horizons(sys.argv[1])
    else:
        horizons = HORIZONS if HORIZONS else [DEFAULT_HOLDING_MONTHS]

    if len(horizons) == 1:
        main(horizons[0])
    else:
        main_multi_horizon(horizons)
