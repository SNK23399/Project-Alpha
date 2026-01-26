"""
Phase 2: Precompute Feature Information Ratio Matrix
====================================================

INFORMATION RATIO OPTIMIZATION PROJECT - Step 5

For each feature at each date, evaluates: "If we selected top-N ETFs by this
feature's ranking, what Information Ratio would we achieve?"

Simple algorithm:
1. Load forward_ir from Phase 1 (pre-computed per ETF per month)
2. Load feature rankings from Phase 1
3. For each (date, feature, N):
   - Select top-N ETFs by feature ranking
   - Compute: feature_ir = mean(forward_ir[top_N_etfs])
   - Record hit rate: 1 if mean_ir > 0, else 0
4. Save 3D matrix for Phase 3-4 to use

This matrix answers: "Which features reliably predict good IR?"

Output:
    data/feature_ir_1month.npz
        feature_ir: (n_dates, n_features, 10) - mean IR of top-N ETFs
        feature_hit: (n_dates, n_features, 10) - hit rate (1 = positive IR)
        dates, features: metadata

Usage:
    python 5_precompute_feature_ir.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
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

# N values to precompute (1-10 satellites)
N_SATELLITES_MAX = 10

# Holding period (in months) - 1-month horizon is optimal
HOLDING_MONTHS = 1

# Force recompute (set to True to regenerate files)
FORCE_RECOMPUTE = False

# Output directory (relative to this script's location)
OUTPUT_DIR = Path(__file__).parent / 'data'


def get_horizon_label(horizon):
    """Get a label string for a horizon (for filenames)."""
    return f"{horizon}month"


# ============================================================
# NUMBA-OPTIMIZED CORE COMPUTATION
# ============================================================

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _compute_all_dates_numba(
        rankings: np.ndarray,           # (n_dates, n_isins, n_features)
        ir_matrix: np.ndarray,          # (n_dates, n_isins) - forward_ir values
        ir_valid_matrix: np.ndarray,    # (n_dates, n_isins) - float32 (1.0=valid, 0.0=invalid)
        n_satellites_max: int
    ) -> tuple:
        """
        Numba-optimized parallel computation across all dates.

        For each (date, feature), selects top-N ETFs and computes their mean IR.

        Returns:
            feature_ir: (n_dates, n_features, n_satellites_max)
            feature_hit: (n_dates, n_features, n_satellites_max)
        """
        n_dates, n_isins, n_features = rankings.shape

        feature_ir = np.full((n_dates, n_features, n_satellites_max), np.nan, dtype=np.float32)
        feature_hit = np.full((n_dates, n_features, n_satellites_max), np.nan, dtype=np.float32)

        # Parallel loop over dates
        for date_idx in prange(n_dates):
            rankings_date = rankings[date_idx, :, :]
            ir_date = ir_matrix[date_idx, :]
            ir_valid = ir_valid_matrix[date_idx, :]

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

                # Compute mean forward_ir for top-1, top-2, ..., top-N
                ir_sum = np.float64(0.0)
                ir_count = 0

                for n in range(n_satellites_max):
                    if n >= len(sorted_indices):
                        break

                    isin_idx = sorted_indices[n]

                    # Check if this ISIN has valid IR
                    if ir_valid[isin_idx] > 0.5:
                        ir_sum += ir_date[isin_idx]
                        ir_count += 1

                    if ir_count > 0:
                        mean_ir = ir_sum / ir_count
                        feature_ir[date_idx, feat_idx, n] = np.float32(mean_ir)
                        feature_hit[date_idx, feat_idx, n] = np.float32(1.0) if mean_ir > 0 else np.float32(0.0)

        return feature_ir, feature_hit

else:
    # Fallback without numba
    def _compute_all_dates_numba(rankings, ir_matrix, ir_valid_matrix, n_satellites_max):
        """Fallback implementation without numba."""
        return _compute_all_dates_numpy(rankings, ir_matrix, ir_valid_matrix, n_satellites_max)


def _compute_all_dates_numpy(rankings, ir_matrix, ir_valid_matrix, n_satellites_max):
    """
    Numpy-vectorized fallback (used if numba not available).
    Still much faster than pure Python due to vectorized operations.
    """
    n_dates, n_isins, n_features = rankings.shape

    feature_ir = np.full((n_dates, n_features, n_satellites_max), np.nan, dtype=np.float32)
    feature_hit = np.full((n_dates, n_features, n_satellites_max), np.nan, dtype=np.float32)

    for date_idx in tqdm(range(n_dates), desc="Processing dates"):
        rankings_date = rankings[date_idx]  # (n_isins, n_features)
        ir_date = ir_matrix[date_idx]       # (n_isins,)
        ir_valid = ir_valid_matrix[date_idx]  # (n_isins,)

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

            # Compute mean forward_ir for top-N
            ir_sum = 0.0
            ir_count = 0

            for n in range(min(n_satellites_max, len(sorted_indices))):
                isin_idx = sorted_indices[n]

                if ir_valid[isin_idx]:
                    ir_sum += ir_date[isin_idx]
                    ir_count += 1

                if ir_count > 0:
                    mean_ir = ir_sum / ir_count
                    feature_ir[date_idx, feat_idx, n] = mean_ir
                    feature_hit[date_idx, feat_idx, n] = 1.0 if mean_ir > 0 else 0.0

    return feature_ir, feature_hit


# ============================================================
# DATA LOADING AND PREPARATION
# ============================================================

def load_data(horizon):
    """Load forward alpha/IR and rankings matrix from Phase 1."""
    horizon_label = get_horizon_label(horizon)

    print(f"\n{'='*60}")
    print(f"LOADING DATA FROM PHASE 1")
    print(f"{'='*60}")

    # Load forward alpha and IR
    alpha_file = OUTPUT_DIR / f'forward_alpha_{horizon_label}.parquet'
    if not alpha_file.exists():
        raise FileNotFoundError(f"Forward alpha/IR not found: {alpha_file}\nRun Phase 1 (4_compute_forward_ir.py) first.")

    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])

    print(f"\n[OK] Forward alpha/IR loaded ({len(alpha_df):,} observations)")
    print(f"     Columns: {', '.join(alpha_df.columns.tolist())}")

    # Load rankings matrix
    rankings_file = OUTPUT_DIR / f'rankings_matrix_ir_{horizon_label}.npz'
    if not rankings_file.exists():
        raise FileNotFoundError(f"Rankings matrix not found: {rankings_file}\nRun Phase 1 (4_compute_forward_ir.py) first.")

    npz_data = np.load(rankings_file, allow_pickle=True)
    rankings = npz_data['rankings'].copy()
    dates = pd.to_datetime(npz_data['dates'])
    isins = npz_data['isins']
    feature_names = npz_data['features']

    print(f"[OK] Rankings matrix loaded {rankings.shape}")
    print(f"     {len(dates)} dates, {len(isins)} ISINs, {len(feature_names)} features")

    return alpha_df, rankings, dates, isins, feature_names


def prepare_ir_matrix(alpha_df, dates, isins):
    """
    Convert forward_ir column to 2D numpy array for fast O(1) lookups.

    Returns:
        ir_matrix: (n_dates, n_isins) - forward_ir values
        ir_valid: (n_dates, n_isins) - boolean mask of valid entries
    """
    print(f"\nPreparing IR matrix for fast lookups...")

    n_dates = len(dates)
    n_isins = len(isins)

    # Create ISIN to index mapping
    isin_to_idx = {isin: idx for idx, isin in enumerate(isins)}

    # Create date to index mapping
    date_to_idx = {date: idx for idx, date in enumerate(dates)}

    # Initialize matrix
    ir_matrix = np.full((n_dates, n_isins), np.nan, dtype=np.float32)

    # Fill matrix using vectorized operations
    for date, group in tqdm(alpha_df.groupby('date'), desc="Building IR matrix", total=len(dates)):
        if date not in date_to_idx:
            continue

        date_idx = date_to_idx[date]

        # Vectorized: map ISINs to indices and fill
        valid_isins = group['isin'].values
        valid_irs = group['forward_ir'].values

        for isin, ir in zip(valid_isins, valid_irs):
            if isin in isin_to_idx:
                isin_idx = isin_to_idx[isin]
                ir_matrix[date_idx, isin_idx] = ir

    # Create validity mask
    ir_valid = ~np.isnan(ir_matrix)

    print(f"  IR matrix shape: {ir_matrix.shape}")
    print(f"  Valid entries: {ir_valid.sum():,} ({ir_valid.sum() / ir_valid.size * 100:.1f}%)")
    print(f"  Mean IR: {np.nanmean(ir_matrix):.4f}")
    print(f"  IR range: {np.nanmin(ir_matrix):.4f} to {np.nanmax(ir_matrix):.4f}")

    return ir_matrix, ir_valid


# ============================================================
# MAIN COMPUTATION
# ============================================================

def precompute_feature_ir(alpha_df, rankings, dates, isins, feature_names):
    """
    Precompute Information Ratio for each (date, feature, N) combination.

    For each feature at each date:
    - Select top-N ETFs by feature ranking
    - Compute: feature_ir = mean(forward_ir[top_N_etfs])
    - Record hit rate: 1 if mean_ir > 0, else 0

    Returns:
        feature_ir: (n_dates, n_features, N_SATELLITES_MAX) array
                    Each cell = mean IR of top-N ETFs
        feature_hit: (n_dates, n_features, N_SATELLITES_MAX) array
                    Each cell = 1 if mean_ir > 0, else 0
    """
    print(f"\n{'='*60}")
    print(f"PHASE 2: PRECOMPUTING FEATURE-IR MATRIX")
    print(f"{'='*60}")

    n_dates = len(dates)
    n_features = len(feature_names)

    print(f"\nMatrix dimensions:")
    print(f"  Dates: {n_dates}")
    print(f"  Features: {n_features}")
    print(f"  N values: 1-{N_SATELLITES_MAX}")
    print(f"  Total cells: {n_dates * n_features * N_SATELLITES_MAX:,}")
    print(f"\nOptimization: {'Numba JIT (parallel)' if HAS_NUMBA else 'NumPy vectorized'}")

    # Prepare IR matrix for fast lookups
    ir_matrix, ir_valid = prepare_ir_matrix(alpha_df, dates, isins)

    # Run optimized computation
    print(f"\nComputing feature-IR matrix...")

    if HAS_NUMBA:
        # Convert boolean ir_valid to float32 (1.0 for True, 0.0 for False)
        ir_valid_float = ir_valid.astype(np.float32)

        # Warm up JIT compiler
        print("  Warming up JIT compiler...")
        _ = _compute_all_dates_numba(
            rankings[:2].astype(np.float32),
            ir_matrix[:2].astype(np.float32),
            ir_valid_float[:2],
            N_SATELLITES_MAX
        )

        print("  Running parallel computation...")
        feature_ir, feature_hit = _compute_all_dates_numba(
            rankings.astype(np.float32),
            ir_matrix.astype(np.float32),
            ir_valid_float,
            N_SATELLITES_MAX
        )
    else:
        feature_ir, feature_hit = _compute_all_dates_numpy(
            rankings,
            ir_matrix,
            ir_valid,
            N_SATELLITES_MAX
        )

    # Statistics
    non_nan = (~np.isnan(feature_ir)).sum()
    total = feature_ir.size
    coverage = non_nan / total * 100

    print(f"\nMatrix statistics:")
    print(f"  Non-NaN cells: {non_nan:,} / {total:,} ({coverage:.1f}%)")
    print(f"  Mean IR: {np.nanmean(feature_ir):.4f}")
    print(f"  IR range: {np.nanmin(feature_ir):.4f} to {np.nanmax(feature_ir):.4f}")
    print(f"  Mean hit rate: {np.nanmean(feature_hit):.2%}")

    return feature_ir, feature_hit


def save_feature_ir(feature_ir, feature_hit, dates, feature_names, horizon):
    """Save the precomputed IR matrices."""
    horizon_label = get_horizon_label(horizon)
    output_file = OUTPUT_DIR / f'feature_ir_{horizon_label}.npz'

    np.savez_compressed(
        output_file,
        feature_ir=feature_ir,
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

def main(horizon=HOLDING_MONTHS):
    """Run Phase 2: Precompute feature-IR matrix."""
    horizon_label = get_horizon_label(horizon)

    print("=" * 60)
    print(f"PHASE 2: PRECOMPUTE FEATURE-IR MATRIX ({horizon_label.upper()})")
    print("=" * 60)

    output_file = OUTPUT_DIR / f'feature_ir_{horizon_label}.npz'

    if output_file.exists() and not FORCE_RECOMPUTE:
        print(f"\n[EXISTS] {output_file}")
        print("Use FORCE_RECOMPUTE=True to regenerate.")

        # Load and show stats
        data = np.load(output_file, allow_pickle=True)
        print(f"\nExisting matrix shape: {data['feature_ir'].shape}")
        print(f"Mean IR: {np.nanmean(data['feature_ir']):.4f}")
        print(f"Mean hit rate: {np.nanmean(data['feature_hit']):.2%}")
        return

    # Load data from Phase 1
    alpha_df, rankings, dates, isins, feature_names = load_data(horizon)

    # Precompute feature-IR matrix
    feature_ir, feature_hit = precompute_feature_ir(
        alpha_df, rankings, dates, isins, feature_names
    )

    # Save
    save_feature_ir(feature_ir, feature_hit, dates, feature_names, horizon)

    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)
    print(f"\nNext step: Run Phase 3 - 6_precompute_mc_hitrates_ir.py")


if __name__ == '__main__':
    main(HOLDING_MONTHS)
