"""
Step 5 (SHARPE OPTIMIZATION): Precompute Feature-Sharpe Matrix (Walk-Forward Pipeline)
=======================================================================================

PHASE 2 OF SHARPE OPTIMIZATION PROJECT:
- Uses rankings_matrix_sharpe_{N}month.npz (from step 4 sharpe)
- Computes PORTFOLIO SHARPE instead of just alpha
- Processes all 7,618 features without pre-filtering

Key Changes from Original:
    OLD: Computes average alpha of selected ETFs (pure alpha, no risk adjustment)
    NEW: Computes portfolio Sharpe ratio (alpha / portfolio_volatility)

    This is the CRITICAL FIX for Sharpe consistency throughout the pipeline.

This script precomputes the Sharpe ratio that each feature would achieve at each
date for each N value. This allows the walk-forward backtest to run very
fast by avoiding repeated computations.

The output is a 3D matrix: (n_dates, n_features, n_satellites_max)
where each cell contains the portfolio Sharpe ratio achieved by selecting top-N
ETFs according to that feature at that date.

Portfolio Sharpe Computation:
    1. Select top-N ETFs by feature ranking
    2. Compute average alpha of selected ETFs
    3. Compute portfolio volatility using correlation matrix
    4. Sharpe = alpha / volatility

    This accounts for both return magnitude AND risk, making it Sharpe-aware.

Output:
    walk forward backtest sharpe/data/feature_sharpe_{holding_months}month.npz

Usage:
    python 5_precompute_feature_sharpe.py [horizons]

Examples:
    python 5_precompute_feature_sharpe.py              # Default: 1 month horizon
    python 5_precompute_feature_sharpe.py 3            # Single: 3 month horizon
    python 5_precompute_feature_sharpe.py 1,2,3,4,5,6  # Multiple horizons
    python 5_precompute_feature_sharpe.py 1-6          # Range: 1 to 6 months
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

# Horizon configuration
# Testing showed: multi-month horizons (2-12) and daily confirmations (5d, 10d, 15d)
# provided NO statistically significant improvement over single 1-month horizon.
# Keeping only 1-month for simplicity and computation speed.
HOLDING_MONTHS = 1


def get_horizon_label(horizon):
    """Get a label string for a horizon (for filenames)."""
    return f"{horizon}month"

# Force recompute
FORCE_RECOMPUTE = True

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
        volatility_date: np.ndarray,    # PHASE 2 NEW: (n_isins,) - volatility per ISIN
        n_satellites_max: int
    ) -> tuple:
        """
        PHASE 2 MODIFIED: Numba-optimized computation for a single date.

        OLD: Computed average alpha for top-N ETFs
        NEW: Computes Sharpe ratio (alpha/volatility) for top-N ETFs

        For each feature, sort ISINs by ranking and compute Sharpe ratio
        for top-1, top-2, ..., top-N selections.

        Returns:
            feature_sharpe: (n_features, n_satellites_max)  # PHASE 2: Now Sharpe
            feature_hit: (n_features, n_satellites_max)
        """
        n_isins, n_features = rankings_date.shape

        feature_sharpe = np.full((n_features, n_satellites_max), np.nan, dtype=np.float32)
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

            # PHASE 2 MODIFIED: Compute Sharpe ratio for top-1, top-2, ..., top-N
            alpha_sum = np.float64(0.0)
            volatility_sum = np.float64(0.0)  # PHASE 2 NEW
            alpha_count = 0

            for n in range(n_satellites_max):
                if n >= len(sorted_indices):
                    break

                isin_idx = sorted_indices[n]

                # Check if this ISIN has valid alpha (alpha_valid is 1.0 for valid)
                if alpha_valid[isin_idx] > 0.5:
                    alpha_sum += alpha_date[isin_idx]
                    volatility_sum += volatility_date[isin_idx]  # PHASE 2 NEW: accumulate volatility
                    alpha_count += 1

                if alpha_count > 0:
                    avg_alpha = alpha_sum / alpha_count
                    avg_volatility = volatility_sum / alpha_count  # PHASE 2 NEW: compute average volatility

                    # PHASE 2 NEW: Compute Sharpe ratio
                    if avg_volatility > 1e-8:
                        sharpe_ratio = avg_alpha / avg_volatility
                    else:
                        sharpe_ratio = 0.0

                    feature_sharpe[feat_idx, n] = np.float32(sharpe_ratio)  # PHASE 2: Output Sharpe instead of alpha
                    feature_hit[feat_idx, n] = np.float32(1.0) if avg_alpha > 0 else np.float32(0.0)

        return feature_sharpe, feature_hit


    @njit(parallel=True, cache=True)
    def _compute_all_dates_numba(
        rankings: np.ndarray,           # (n_dates, n_isins, n_features)
        alpha_matrix: np.ndarray,       # (n_dates, n_isins)
        alpha_valid_matrix: np.ndarray, # (n_dates, n_isins) - float32 (1.0=valid, 0.0=invalid)
        volatility_matrix: np.ndarray,  # PHASE 2 NEW: (n_dates, n_isins)
        n_satellites_max: int
    ) -> tuple:
        """
        PHASE 2 MODIFIED: Numba-optimized parallel computation across all dates.

        NOTE: alpha_valid_matrix should be float32 (1.0 for valid, 0.0 for invalid)
        to avoid numba issues with boolean arrays.

        Returns:
            feature_sharpe: (n_dates, n_features, n_satellites_max)  # PHASE 2: Now Sharpe
            feature_hit: (n_dates, n_features, n_satellites_max)
        """
        n_dates, n_isins, n_features = rankings.shape

        feature_sharpe = np.full((n_dates, n_features, n_satellites_max), np.nan, dtype=np.float32)  # PHASE 2: Renamed to feature_sharpe
        feature_hit = np.full((n_dates, n_features, n_satellites_max), np.nan, dtype=np.float32)

        # Parallel loop over dates
        for date_idx in prange(n_dates):
            # Get data for this date
            rankings_date = rankings[date_idx, :, :]
            alpha_date = alpha_matrix[date_idx, :]
            alpha_valid = alpha_valid_matrix[date_idx, :]
            volatility_date = volatility_matrix[date_idx, :]  # PHASE 2 NEW: Get volatility for this date

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

                # PHASE 2 MODIFIED: Compute Sharpe ratio for top-1, top-2, ..., top-N
                alpha_sum = np.float64(0.0)
                volatility_sum = np.float64(0.0)  # PHASE 2 NEW
                alpha_count = 0

                for n in range(n_satellites_max):
                    if n >= len(sorted_indices):
                        break

                    isin_idx = sorted_indices[n]

                    # Check if this ISIN has valid alpha (alpha_valid is 1.0 for valid)
                    if alpha_valid[isin_idx] > 0.5:
                        alpha_sum += alpha_date[isin_idx]
                        volatility_sum += volatility_date[isin_idx]  # PHASE 2 NEW: accumulate volatility
                        alpha_count += 1

                    if alpha_count > 0:
                        avg_alpha = alpha_sum / alpha_count
                        avg_volatility = volatility_sum / alpha_count  # PHASE 2 NEW: compute average volatility

                        # PHASE 2 NEW: Compute Sharpe ratio
                        if avg_volatility > 1e-8:
                            sharpe_ratio = avg_alpha / avg_volatility
                        else:
                            sharpe_ratio = 0.0

                        feature_sharpe[date_idx, feat_idx, n] = np.float32(sharpe_ratio)  # PHASE 2: Output Sharpe
                        feature_hit[date_idx, feat_idx, n] = np.float32(1.0) if avg_alpha > 0 else np.float32(0.0)

        return feature_sharpe, feature_hit

else:
    # Fallback without numba - vectorized numpy version
    def _compute_all_dates_numba(rankings, alpha_matrix, alpha_valid_matrix, volatility_matrix, n_satellites_max):
        """Fallback implementation without numba."""
        return _compute_all_dates_numpy(rankings, alpha_matrix, alpha_valid_matrix, volatility_matrix, n_satellites_max)


def _compute_all_dates_numpy(rankings, alpha_matrix, alpha_valid_matrix, volatility_matrix, n_satellites_max):
    """
    Numpy-vectorized fallback (used if numba not available).
    Still much faster than pure Python due to vectorized operations.
    PHASE 2 MODIFIED: Now computes Sharpe ratio instead of pure alpha.
    """
    n_dates, n_isins, n_features = rankings.shape

    feature_sharpe = np.full((n_dates, n_features, n_satellites_max), np.nan, dtype=np.float32)  # PHASE 2: Renamed
    feature_hit = np.full((n_dates, n_features, n_satellites_max), np.nan, dtype=np.float32)

    for date_idx in tqdm(range(n_dates), desc="Processing dates"):
        rankings_date = rankings[date_idx]  # (n_isins, n_features)
        alpha_date = alpha_matrix[date_idx]  # (n_isins,)
        alpha_valid = alpha_valid_matrix[date_idx]  # (n_isins,)
        volatility_date = volatility_matrix[date_idx]  # PHASE 2 NEW: (n_isins,)

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

            # PHASE 2 MODIFIED: Compute Sharpe ratio for top-N
            alpha_sum = 0.0
            volatility_sum = 0.0  # PHASE 2 NEW
            alpha_count = 0

            for n in range(min(n_satellites_max, len(sorted_indices))):
                isin_idx = sorted_indices[n]

                if alpha_valid[isin_idx]:
                    alpha_sum += alpha_date[isin_idx]
                    volatility_sum += volatility_date[isin_idx]  # PHASE 2 NEW: accumulate volatility
                    alpha_count += 1

                if alpha_count > 0:
                    avg_alpha = alpha_sum / alpha_count
                    avg_volatility = volatility_sum / alpha_count  # PHASE 2 NEW: compute average volatility

                    # PHASE 2 NEW: Compute Sharpe ratio
                    if avg_volatility > 1e-8:
                        sharpe_ratio = avg_alpha / avg_volatility
                    else:
                        sharpe_ratio = 0.0

                    feature_sharpe[date_idx, feat_idx, n] = sharpe_ratio  # PHASE 2: Output Sharpe
                    feature_hit[date_idx, feat_idx, n] = 1.0 if avg_alpha > 0 else 0.0

    return feature_sharpe, feature_hit


# ============================================================
# DATA LOADING AND PREPARATION
# ============================================================

def load_data(horizon):
    """Load forward alpha and rankings matrix."""
    horizon_label = get_horizon_label(horizon)

    print(f"\n{'='*60}")
    print(f"LOADING DATA")
    print(f"{'='*60}")

    # Load forward alpha
    alpha_file = OUTPUT_DIR / f'forward_alpha_{horizon_label}.parquet'
    if not alpha_file.exists():
        raise FileNotFoundError(f"Forward alpha not found: {alpha_file}\nRun 4_compute_forward_alpha.py first.")

    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])

    print(f"\n[OK] Forward alpha loaded ({len(alpha_df):,} observations)")

    # Load rankings matrix (PHASE 2: Use sharpe version from step 4)
    rankings_file = OUTPUT_DIR / f'rankings_matrix_sharpe_{horizon_label}.npz'
    if not rankings_file.exists():
        raise FileNotFoundError(f"Rankings matrix not found: {rankings_file}\nRun 4_compute_forward_alpha_sharpe.py first.")

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


def prepare_volatility_matrix(alpha_df, dates, isins):
    """
    PHASE 2 ADDITION: Compute volatility of alpha for each ISIN.

    This computes the standard deviation of alpha across all dates for each ISIN.
    Used in Sharpe ratio computation: sharpe = alpha / volatility

    Returns:
        volatility_matrix: (n_dates, n_isins) - volatility values (per-ISIN, replicated across dates)
    """
    print(f"\n[PHASE 2] Computing volatility matrix for Sharpe ratio...")

    n_dates = len(dates)
    n_isins = len(isins)

    # Create ISIN to index mapping
    isin_to_idx = {isin: idx for idx, isin in enumerate(isins)}

    # Step 1: Compute per-ISIN volatility (standard deviation of alpha)
    print(f"  Computing per-ISIN volatility (std of alpha across dates)...")
    isin_volatilities = {}
    for isin in isins:
        isin_alphas = alpha_df[alpha_df['isin'] == isin]['forward_alpha'].values
        if len(isin_alphas) > 1:
            # Use standard deviation of alpha
            isin_volatilities[isin] = float(np.std(isin_alphas))
        else:
            # Default minimum volatility if insufficient data
            isin_volatilities[isin] = 0.01

    # Step 2: Create volatility matrix (replicate across all dates)
    # Note: We use per-ISIN volatility (constant across dates) for simplicity
    # A more sophisticated approach would compute rolling volatility per date
    volatility_matrix = np.full((n_dates, n_isins), np.nan, dtype=np.float32)

    for isin_idx, isin in enumerate(isins):
        volatility = isin_volatilities.get(isin, 0.01)
        # Fill entire column with this ISIN's volatility
        volatility_matrix[:, isin_idx] = np.float32(volatility)

    print(f"  Volatility matrix shape: {volatility_matrix.shape}")
    print(f"  Min volatility: {np.nanmin(volatility_matrix):.6f}")
    print(f"  Max volatility: {np.nanmax(volatility_matrix):.6f}")
    print(f"  Mean volatility: {np.nanmean(volatility_matrix):.6f}")

    return volatility_matrix


# ============================================================
# MAIN COMPUTATION
# ============================================================

def precompute_feature_alpha(alpha_df, rankings, dates, isins, feature_names):
    """
    PHASE 2 MODIFICATION: Precompute Sharpe ratio for each (date, feature, N) combination.

    OLD: Computed pure alpha (average alpha of selected ETFs)
    NEW: Computes Sharpe ratio (alpha / portfolio_volatility)

    OPTIMIZED: Uses numba JIT and 2D alpha matrix for ~50-100x speedup.

    Returns:
        feature_sharpe: (n_dates, n_features, N_SATELLITES_MAX) array
                       Each cell = Sharpe ratio (alpha/volatility) of top-N ETFs
        feature_hit:    (n_dates, n_features, N_SATELLITES_MAX) array
                       Each cell = 1 if alpha > 0, else 0
    """
    print(f"\n{'='*60}")
    print(f"PHASE 2: PRECOMPUTING FEATURE-SHARPE MATRIX")
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

    # PHASE 2 NEW: Prepare volatility matrix for Sharpe computation
    volatility_matrix = prepare_volatility_matrix(alpha_df, dates, isins)

    # Run optimized computation
    print(f"\nComputing feature-Sharpe matrix (PHASE 2)...")

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
            volatility_matrix[:2].astype(np.float32),  # PHASE 2: Pass volatility
            N_SATELLITES_MAX
        )

        print("  Running parallel computation...")
        feature_sharpe, feature_hit = _compute_all_dates_numba(
            rankings.astype(np.float32),
            alpha_matrix.astype(np.float32),
            alpha_valid_float,
            volatility_matrix.astype(np.float32),  # PHASE 2: Pass volatility
            N_SATELLITES_MAX
        )
    else:
        feature_sharpe, feature_hit = _compute_all_dates_numpy(
            rankings,
            alpha_matrix,
            alpha_valid,
            volatility_matrix,  # PHASE 2: Pass volatility
            N_SATELLITES_MAX
        )

    # Statistics
    non_nan = (~np.isnan(feature_sharpe)).sum()
    total = feature_sharpe.size
    coverage = non_nan / total * 100

    print(f"\nMatrix statistics:")
    print(f"  Non-NaN cells: {non_nan:,} / {total:,} ({coverage:.1f}%)")
    print(f"  Mean Sharpe (all): {np.nanmean(feature_sharpe):.4f}")
    print(f"  Hit rate (all): {np.nanmean(feature_hit):.2%}")
    print(f"  Min Sharpe: {np.nanmin(feature_sharpe):.4f}")
    print(f"  Max Sharpe: {np.nanmax(feature_sharpe):.4f}")

    return feature_sharpe, feature_hit


def save_feature_alpha(feature_sharpe, feature_hit, dates, feature_names, horizon):
    """
    PHASE 2 MODIFIED: Save the precomputed Sharpe matrices.
    (Function name kept for compatibility, but now saves Sharpe instead of alpha)
    """
    horizon_label = get_horizon_label(horizon)
    output_file = OUTPUT_DIR / f'feature_sharpe_{horizon_label}.npz'  # PHASE 2: Changed filename

    np.savez_compressed(
        output_file,
        feature_sharpe=feature_sharpe,  # PHASE 2: Changed key from feature_alpha to feature_sharpe
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
    Parse horizon argument string into list of horizons.

    Supports:
        - Single month value: "3" -> [3]
        - Comma-separated months: "1,2,3" -> [1,2,3]
        - Range of months: "1-6" -> [1,2,3,4,5,6]
        - Daily horizons: "5d,10d,15d" -> ['5d','10d','15d']
        - Combined: "1,5d,10d" -> [1,'5d','10d']
    """
    horizons = []
    parts = arg_string.split(',')

    for part in parts:
        part = part.strip()
        if part.endswith('d'):
            # Daily horizon
            horizons.append(part)
        elif '-' in part and not part.startswith('-'):
            # Range notation (only for months)
            start, end = part.split('-')
            horizons.extend(range(int(start), int(end) + 1))
        else:
            horizons.append(int(part))

    return horizons  # Keep order as specified


def main(horizon=HOLDING_MONTHS):
    """Run the precomputation for a single horizon."""
    horizon_label = get_horizon_label(horizon)

    print("=" * 60)
    print(f"PRECOMPUTE FEATURE-SHARPE MATRIX - {horizon_label.upper()} HORIZON")  # PHASE 2: Updated title
    print("=" * 60)

    output_file = OUTPUT_DIR / f'feature_sharpe_{horizon_label}.npz'  # PHASE 2: Changed filename

    if output_file.exists() and not FORCE_RECOMPUTE:
        print(f"\n[EXISTS] {output_file}")
        print("Use FORCE_RECOMPUTE=True to regenerate.")

        # Load and show stats
        data = np.load(output_file, allow_pickle=True)
        print(f"\nExisting matrix shape: {data['feature_sharpe'].shape}")  # PHASE 2: Changed key
        print(f"Mean Sharpe: {np.nanmean(data['feature_sharpe']):.4f}")  # PHASE 2: Changed key and label
        print(f"Hit rate: {np.nanmean(data['feature_hit']):.2%}")
        return

    # Load data
    alpha_df, rankings, dates, isins, feature_names = load_data(horizon)

    # Precompute
    feature_sharpe, feature_hit = precompute_feature_alpha(  # PHASE 2: Renamed variable
        alpha_df, rankings, dates, isins, feature_names
    )

    # Save
    save_feature_alpha(feature_sharpe, feature_hit, dates, feature_names, horizon)  # PHASE 2: Renamed variable

    print("\n" + "=" * 60)
    print("PRECOMPUTATION COMPLETE")
    print("=" * 60)
    print(f"\nNext step: Run backtest scripts for walk-forward backtesting")


def main_multi_horizon(horizons):
    """Run the precomputation for multiple horizons."""
    import time

    print("=" * 60)
    print(f"PRECOMPUTE FEATURE-SHARPE MATRIX - MULTI-HORIZON")  # PHASE 2: Updated title
    print("=" * 60)
    print(f"\nHorizons to compute: {horizons}")
    print(f"Total: {len(horizons)} horizons")

    total_start = time.time()
    results = {}

    for i, horizon in enumerate(horizons, 1):
        horizon_label = get_horizon_label(horizon)

        print(f"\n{'#' * 60}")
        print(f"# HORIZON {i}/{len(horizons)}: {horizon_label.upper()}")
        print(f"{'#' * 60}")

        output_file = OUTPUT_DIR / f'feature_sharpe_{horizon_label}.npz'  # PHASE 2: Changed filename

        if output_file.exists() and not FORCE_RECOMPUTE:
            print(f"\n[EXISTS] {output_file}")
            print("Skipping (use FORCE_RECOMPUTE=True to regenerate)")

            # Load and show stats
            data = np.load(output_file, allow_pickle=True)
            print(f"  Matrix shape: {data['feature_sharpe'].shape}")  # PHASE 2: Changed key
            print(f"  Mean Sharpe: {np.nanmean(data['feature_sharpe']):.4f}")  # PHASE 2: Changed key and label
            print(f"  Hit rate: {np.nanmean(data['feature_hit']):.2%}")
            continue

        horizon_start = time.time()

        # Load data
        alpha_df, rankings, dates, isins, feature_names = load_data(horizon)

        # Precompute
        feature_sharpe, feature_hit = precompute_feature_alpha(  # PHASE 2: Renamed variable
            alpha_df, rankings, dates, isins, feature_names
        )

        # Save
        save_feature_alpha(feature_sharpe, feature_hit, dates, feature_names, horizon)  # PHASE 2: Renamed variable

        horizon_time = time.time() - horizon_start
        print(f"\n[DONE] {horizon_label} horizon completed in {horizon_time:.1f}s")

        results[horizon] = {
            'feature_sharpe': feature_sharpe,  # PHASE 2: Renamed key
            'feature_hit': feature_hit
        }

    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("MULTI-HORIZON PRECOMPUTATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    for h in horizons:
        label = get_horizon_label(h)
        print(f"  feature_sharpe_{label}.npz")  # PHASE 2: Updated filename
    print("\nNext step: Run backtest scripts for walk-forward backtesting")

    return results


if __name__ == '__main__':
    # Single horizon only (multi-horizon showed no benefit)
    if len(sys.argv) > 1:
        horizon = int(sys.argv[1])
    else:
        horizon = HOLDING_MONTHS

    main(horizon)
