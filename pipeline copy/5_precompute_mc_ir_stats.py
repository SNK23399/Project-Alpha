"""
Phase 3: Precompute Monte Carlo Information Ratio Statistics (Walk-Forward Pipeline)
====================================================================================

INFORMATION RATIO OPTIMIZATION PROJECT - Step 5

- Uses feature_ir_{N}month.npz (from step 4 - Feature IR computation)
- Computes IR STATISTICS from MC simulations
- Processes ALL filtered features WITHOUT pre-filtering (true Bayesian approach)

For each feature at each test date, runs 1M (configurable) Monte Carlo
simulations to estimate:
- IR mean (average Information Ratio from MC)
- IR standard deviation (uncertainty in IR)

These statistics enable true Bayesian learning where:
- Features start with weak priors
- Evidence accumulates over time
- Even initially poor features can be reconsidered if they improve
- Allows discovery of features that were dormant but become relevant

NOTE: Only IR mean/std are computed as they are the only statistics used in
Step 6 (Bayesian strategy).

MC_SAMPLES_PER_MONTH = 1_000_000 (configurable for tighter/looser CIs)

Output:
    data/mc_ir_mean_{holding_months}month.npz
        mc_ir_mean: MC-estimated IR for each feature/date/N
        mc_ir_std: Uncertainty in IR estimates

Usage:
    python 5_precompute_mc_ir_stats.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import time
from numba import njit, prange, cuda
import math

warnings.filterwarnings('ignore')

# Check for GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"GPU acceleration enabled (CuPy {cp.__version__})")
except ImportError:
    GPU_AVAILABLE = False
    print("WARNING: CuPy not available - GPU acceleration disabled")

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================
# CONFIGURATION
# ============================================================

HOLDING_MONTHS = 1
N_SATELLITES_TO_PRECOMPUTE = [3, 4, 5]
MIN_TRAINING_MONTHS = 36
# Note: No pre-filtering - all 7,618 features evaluated for true Bayesian learning
DECAY_HALF_LIFE_MONTHS = 54

# Monte Carlo settings - ADAPTIVE CONVERGENCE
# Instead of fixed samples, run until priors stabilize
# Since MC is GPU-fast with precomputed data, use tight threshold for true convergence
MC_BATCH_SIZE = 500_000  # Samples per batch
MC_CONVERGENCE_THRESHOLD = 0.005  # % max change in prior means (tight - ensures true convergence)
MC_MIN_SAMPLES = 1_000_000  # Minimum before checking convergence
MC_MAX_SAMPLES = 3_000_000_000  # Safety limit (generous since MC is fast)
MC_FALLBACK_SAMPLES = 3_000_000  # If convergence fails, use this

MC_ENSEMBLE_SIZES = [3, 4, 5, 6, 7, 8, 9, 10]
MC_MIN_SAMPLES_JOB = 100

# GPU settings
GPU_BLOCK_SIZE = 256

# Force recompute (set to True to regenerate files)
FORCE_RECOMPUTE = True
DATA_DIR = Path(__file__).parent / 'data'

# Random seed for deterministic MC simulations
# TEST: Set to None to validate adaptive convergence works WITHOUT seeding
MC_SEED = None  # Adaptive convergence should ensure consistency even with different random samples


# ============================================================
# NUMBA CPU FUNCTIONS
# ============================================================

@njit(cache=True, parallel=True)
def evaluate_all_features_all_dates(feature_ir, n_satellites, test_start_idx, n_dates):
    """
    Evaluate all features for ALL test dates at once.

    Computes weighted average IR from historical feature_ir matrix.
    Also computes standard deviation of IR for uncertainty estimation.

    Uses exponential decay weighting (more recent months weighted higher).
    """
    n_test_dates = n_dates - test_start_idx
    n_features = feature_ir.shape[1]

    all_avg_irs = np.empty((n_test_dates, n_features), dtype=np.float64)
    all_ir_stds = np.empty((n_test_dates, n_features), dtype=np.float64)

    decay_rate = np.log(2) / DECAY_HALF_LIFE_MONTHS

    for test_offset in prange(n_test_dates):
        test_idx = test_start_idx + test_offset

        for feat_idx in range(n_features):
            sum_ir = 0.0
            sum_ir_sq = 0.0
            sum_weight = 0.0
            count = 0

            for i in range(test_idx):
                months_ago = test_idx - i
                weight = np.exp(-decay_rate * months_ago)
                ir = feature_ir[i, feat_idx, n_satellites - 1]
                if not np.isnan(ir):
                    sum_ir += ir * weight
                    sum_ir_sq += (ir ** 2) * weight
                    sum_weight += weight
                    count += 1

            if sum_weight > 0:
                avg_ir = sum_ir / sum_weight
                all_avg_irs[test_offset, feat_idx] = avg_ir

                # Compute IR standard deviation (uncertainty in IR)
                if count > 1:
                    avg_ir_sq = sum_ir_sq / sum_weight
                    variance = avg_ir_sq - (avg_ir ** 2)
                    # Ensure non-negative (avoid floating point errors)
                    if variance > 0:
                        all_ir_stds[test_offset, feat_idx] = np.sqrt(variance)
                    else:
                        all_ir_stds[test_offset, feat_idx] = 0.01
                else:
                    all_ir_stds[test_offset, feat_idx] = 0.01
            else:
                all_avg_irs[test_offset, feat_idx] = -999.0
                all_ir_stds[test_offset, feat_idx] = 0.01

    return all_avg_irs, all_ir_stds


@njit(cache=True)
def aggregate_batch_numba(alphas, ensemble_indices, ensemble_sizes,
                          candidate_offsets, sample_to_job, all_candidates,
                          total_counts, alpha_sums, alpha_sq_sums):
    """
    Numba-accelerated aggregation of MC batch results.
    Accumulates alpha statistics (mean and variance).

    Note: Not using parallel=True because we're updating shared arrays
    and need atomic-like behavior. The GPU kernel is the parallel part.
    """
    n_samples = len(alphas)

    # Process each sample sequentially to avoid race conditions
    for i in range(n_samples):
        if np.isnan(alphas[i]):
            continue

        job_idx = sample_to_job[i]
        alpha_val = alphas[i]
        ens_size = ensemble_sizes[i]

        cand_start = candidate_offsets[job_idx]
        cand_end = candidate_offsets[job_idx + 1]
        n_cands = cand_end - cand_start

        if n_cands < 2:
            continue

        # Get features in this ensemble
        for j in range(ens_size):
            local_cand_idx = ensemble_indices[i, j + 1] % n_cands
            global_cand_idx = cand_start + local_cand_idx
            feat_idx = all_candidates[global_cand_idx]

            total_counts[job_idx, feat_idx] += 1

            # Track alpha statistics (for mean and std)
            alpha_sums[job_idx, feat_idx] += alpha_val
            alpha_sq_sums[job_idx, feat_idx] += alpha_val * alpha_val


# ============================================================
# GPU CUDA KERNEL
# ============================================================

if GPU_AVAILABLE:
    @cuda.jit
    def mc_evaluate_kernel(
        rankings,            # (n_months, n_isins, n_features)
        alpha_matrix,        # (n_months, n_isins)
        alpha_valid,         # (n_months, n_isins)
        candidate_indices,   # (n_candidates,) - ALL candidates flattened
        candidate_offsets,   # (n_jobs+1,) - start offset for each job
        ensemble_indices,    # (total_samples, max_ens_size)
        ensemble_sizes,      # (total_samples,)
        sample_to_job,       # (total_samples,)
        job_train_ends,      # (n_jobs,)
        n_satellites,
        out_alphas
    ):
        sample_idx = cuda.grid(1)
        if sample_idx >= ensemble_sizes.shape[0]:
            return

        job_idx = sample_to_job[sample_idx]
        train_end = job_train_ends[job_idx]
        cand_start = candidate_offsets[job_idx]
        cand_end = candidate_offsets[job_idx + 1]
        n_cands = cand_end - cand_start

        if n_cands < 2:
            out_alphas[sample_idx] = math.nan
            return

        n_isins = rankings.shape[1]
        ens_size = ensemble_sizes[sample_idx]

        # Pick random historical month
        month_idx = ensemble_indices[sample_idx, 0] % train_end
        if month_idx < 0:
            month_idx = 0

        # Track selected ISINs (max 10)
        selected = cuda.local.array(10, dtype=np.int64)
        for i in range(10):
            selected[i] = -1

        best_alpha_sum = 0.0
        best_count = 0

        for k in range(n_satellites):
            best_idx = -1
            best_score = -1e30

            for i in range(n_isins):
                # Check if already selected
                already_selected = False
                for s in range(k):
                    if selected[s] == i:
                        already_selected = True
                        break
                if already_selected:
                    continue

                if not alpha_valid[month_idx, i]:
                    continue

                # Compute score
                score_sum = 0.0
                count = 0
                for j in range(ens_size):
                    local_cand_idx = ensemble_indices[sample_idx, j + 1] % n_cands
                    global_cand_idx = cand_start + local_cand_idx
                    feat_idx = candidate_indices[global_cand_idx]
                    val = rankings[month_idx, i, feat_idx]
                    if not math.isnan(val):
                        score_sum += val
                        count += 1

                if count > 0:
                    score = score_sum / count
                    if score > best_score:
                        best_score = score
                        best_idx = i

            if best_idx < 0:
                out_alphas[sample_idx] = math.nan
                return

            selected[k] = best_idx
            best_alpha_sum += alpha_matrix[month_idx, best_idx]
            best_count += 1

        if best_count == 0:
            out_alphas[sample_idx] = math.nan
        else:
            avg_alpha = best_alpha_sum / best_count
            out_alphas[sample_idx] = avg_alpha


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """
    Load precomputed data from Steps 1, 3, and 4.

    Returns data needed for MC IR prior computation:
    - forward alpha (Step 1) for ground truth IR values
    - ranking matrix (Step 3) for feature evaluation
    - feature IR statistics (Step 4) for historical feature performance
    """
    print("=" * 120)
    print("LOADING DATA FROM STEPS 1, 3, 4")
    print("=" * 120)

    horizon_label = f"{HOLDING_MONTHS}month"

    # Load forward alpha (Step 1)
    alpha_file = DATA_DIR / f'forward_alpha_{horizon_label}.parquet'
    if not alpha_file.exists():
        raise FileNotFoundError(f"Forward alpha not found: {alpha_file}\nRun Step 1 first.")
    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])
    print(f"[OK] Forward alpha loaded")

    # Load ranking matrix from filtered signals (Step 3)
    rankings_file = DATA_DIR / f'rankings_matrix_filtered_{horizon_label}.npz'
    if not rankings_file.exists():
        raise FileNotFoundError(f"Ranking matrix not found: {rankings_file}\nRun Step 3 first.")
    npz_data = np.load(rankings_file, allow_pickle=True)
    rankings = npz_data['rankings'].astype(np.float64)
    dates = pd.to_datetime(npz_data['dates'])
    isins = npz_data['isins']
    feature_names = list(npz_data['features'])
    print(f"[OK] Ranking matrix loaded {rankings.shape}")

    # Load feature IR statistics (Step 4)
    feature_ir_file = DATA_DIR / f'feature_ir_{horizon_label}.npz'
    if not feature_ir_file.exists():
        raise FileNotFoundError(f"Feature IR not found: {feature_ir_file}\nRun Step 4 first.")
    fs_data = np.load(feature_ir_file, allow_pickle=True)
    feature_ir = fs_data['feature_ir'].astype(np.float64)
    print(f"[OK] Feature IR loaded {feature_ir.shape}")

    n_dates, n_isins = len(dates), len(isins)
    isin_to_idx = {isin: idx for idx, isin in enumerate(isins)}
    date_to_idx = {date: idx for idx, date in enumerate(dates)}

    # Build alpha matrix for MC evaluation
    alpha_matrix = np.full((n_dates, n_isins), np.nan, dtype=np.float64)
    for date, group in alpha_df.groupby('date'):
        if date not in date_to_idx:
            continue
        date_idx = date_to_idx[date]
        for isin, alpha in zip(group['isin'].values, group['forward_alpha'].values):
            if isin in isin_to_idx:
                alpha_matrix[date_idx, isin_to_idx[isin]] = alpha

    alpha_valid = ~np.isnan(alpha_matrix)

    print(f"  Rankings: {rankings.shape}")
    print(f"  Dates: {dates[0].date()} to {dates[-1].date()} ({len(dates)} months)")
    print(f"  Filtered signals: {feature_names.__len__()}")

    return {
        'alpha_matrix': alpha_matrix,
        'alpha_valid': alpha_valid,
        'rankings': rankings,
        'dates': dates,
        'isins': isins,
        'feature_names': feature_names,
        'feature_ir': feature_ir
    }


# ============================================================
# PRECOMPUTATION
# ============================================================

def precompute_candidates_all_dates(data, n_satellites, test_start_idx):
    """
    Precompute candidate features for ALL test dates at once.

    NO PRE-FILTERING: All 7,618 features are treated as candidates.
    This enables true Bayesian learning where beliefs can be updated
    based on realized outcomes, even for initially low-performing features.

    Features with negative IR can still be used (inverted).
    """
    feature_ir = data['feature_ir']
    n_dates = len(data['dates'])
    n_features = feature_ir.shape[1]
    n_test_dates = n_dates - test_start_idx

    all_avg_irs, all_ir_stds = evaluate_all_features_all_dates(
        feature_ir, n_satellites, test_start_idx, n_dates
    )

    # ALL features are candidates (no pre-filtering)
    candidate_mask = np.ones((n_test_dates, n_features), dtype=np.bool_)

    # Mark features with negative IR as inverted (could potentially short these)
    inverted_mask = np.zeros((n_test_dates, n_features), dtype=np.bool_)

    for test_offset in range(n_test_dates):
        avg_irs = all_avg_irs[test_offset]
        # Mark as inverted if IR is clearly negative (exclude NaN/sentinel values)
        inverted_mask[test_offset] = (avg_irs < 0) & (avg_irs > -900)

    return candidate_mask, inverted_mask


def run_mc_for_n(data, n_satellites, test_start_idx, candidate_mask, inverted_mask):
    """Run MC for one N value across all test dates."""
    rankings = data['rankings'].copy()
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    n_dates = rankings.shape[0]
    n_features = rankings.shape[2]
    n_test_dates = n_dates - test_start_idx

    # Apply inversions globally
    global_inverted = np.any(inverted_mask, axis=0)
    for feat_idx in np.where(global_inverted)[0]:
        rankings[:, :, feat_idx] = 1.0 - rankings[:, :, feat_idx]

    # Build job structure
    all_candidates = []
    candidate_offsets = [0]
    job_train_ends = []

    for test_offset in range(n_test_dates):
        test_idx = test_start_idx + test_offset
        candidates = np.where(candidate_mask[test_offset])[0]
        all_candidates.extend(candidates.tolist())
        candidate_offsets.append(len(all_candidates))
        job_train_ends.append(test_idx)

    all_candidates = np.array(all_candidates, dtype=np.int64)
    candidate_offsets = np.array(candidate_offsets, dtype=np.int64)
    job_train_ends = np.array(job_train_ends, dtype=np.int64)

    n_jobs = n_test_dates

    # Determine samples per job
    samples_per_job = []
    for job_idx in range(n_jobs):
        n_cands = candidate_offsets[job_idx + 1] - candidate_offsets[job_idx]
        # For adaptive convergence, we'll batch iteratively instead of fixed samples
        samples_per_job.append(n_cands >= 2)  # Just track whether job has work

    if not any(samples_per_job):
        return (np.full((n_test_dates, n_features), np.nan, dtype=np.float32),
                np.full((n_test_dates, n_features), np.nan, dtype=np.float32))

    if not GPU_AVAILABLE:
        raise RuntimeError(
            "GPU (CuPy) required for Monte Carlo simulations.\n"
            "Install with: pip install cupy-cuda11x\n"
            "Or set FORCE_RECOMPUTE=False to skip if output already exists."
        )

    # Accumulate results across batches
    total_counts = np.zeros((n_jobs, n_features), dtype=np.int64)
    alpha_sums = np.zeros((n_jobs, n_features), dtype=np.float64)
    alpha_sq_sums = np.zeros((n_jobs, n_features), dtype=np.float64)

    max_ens_size = max(MC_ENSEMBLE_SIZES) + 1
    ensemble_sizes_choices = np.array(MC_ENSEMBLE_SIZES, dtype=np.int64)

    # Transfer static data to GPU once (GPU is guaranteed available here)
    rankings_gpu = cp.asarray(rankings, dtype=cp.float64)
    alpha_matrix_gpu = cp.asarray(alpha_matrix, dtype=cp.float64)
    alpha_valid_gpu = cp.asarray(alpha_valid, dtype=cp.bool_)
    candidate_indices_gpu = cp.asarray(all_candidates, dtype=cp.int64)
    candidate_offsets_gpu = cp.asarray(candidate_offsets, dtype=cp.int64)
    job_train_ends_gpu = cp.asarray(job_train_ends, dtype=cp.int64)

    # Set random seed for deterministic MC simulations
    if MC_SEED is not None:
        cp.random.seed(MC_SEED)
        np.random.seed(MC_SEED)

    # ADAPTIVE CONVERGENCE LOOP
    total_samples_run = 0
    batch_idx = 0
    prev_prior_means = None
    converged = False
    last_pct_change = 100.0  # Track for progress bar
    first_batch_check = True  # Only print newline once

    while total_samples_run < MC_MAX_SAMPLES and not converged:
        # Calculate samples for this batch based on jobs
        batch_sample_counts = []
        batch_total = 0
        for job_idx in range(n_jobs):
            if samples_per_job[job_idx]:
                batch_sample_counts.append(MC_BATCH_SIZE // n_jobs)  # Distribute evenly
                batch_total += MC_BATCH_SIZE // n_jobs
            else:
                batch_sample_counts.append(0)

        # Build sample-to-job mapping for this batch
        sample_to_job_batch = np.zeros(batch_total, dtype=np.int64)
        idx = 0
        for job_idx, n_samples in enumerate(batch_sample_counts):
            sample_to_job_batch[idx:idx + n_samples] = job_idx
            idx += n_samples

        if batch_total == 0:
            break

        # Generate random data on GPU
        ensemble_sizes_gpu = cp.random.choice(
            cp.array(ensemble_sizes_choices), size=(batch_total,)
        ).astype(cp.int32)

        ensemble_indices_gpu = cp.random.randint(
            0, 2**31 - 1, size=(batch_total, max_ens_size), dtype=cp.int64
        )

        sample_to_job_gpu = cp.asarray(sample_to_job_batch, dtype=cp.int64)
        out_alphas = cp.zeros(batch_total, dtype=cp.float64)

        # Launch kernel
        threads = GPU_BLOCK_SIZE
        blocks = (batch_total + threads - 1) // threads

        mc_evaluate_kernel[blocks, threads](
            rankings_gpu, alpha_matrix_gpu, alpha_valid_gpu,
            candidate_indices_gpu, candidate_offsets_gpu,
            ensemble_indices_gpu, ensemble_sizes_gpu,
            sample_to_job_gpu, job_train_ends_gpu,
            n_satellites, out_alphas
        )
        cuda.synchronize()

        # Transfer back
        alphas_np = cp.asnumpy(out_alphas)
        ensemble_indices_np = cp.asnumpy(ensemble_indices_gpu)
        ensemble_sizes_np = cp.asnumpy(ensemble_sizes_gpu)

        # Aggregate this batch using Numba (fast)
        aggregate_batch_numba(
            alphas_np, ensemble_indices_np, ensemble_sizes_np,
            candidate_offsets, sample_to_job_batch, all_candidates,
            total_counts, alpha_sums, alpha_sq_sums
        )

        total_samples_run += batch_total
        batch_idx += 1

        # CHECK CONVERGENCE after minimum samples
        if total_samples_run >= MC_MIN_SAMPLES:
            # Compute current prior means
            current_prior_means = np.full((n_jobs, n_features), np.nan, dtype=np.float32)
            for job_idx in range(n_jobs):
                for feat_idx in range(n_features):
                    total = total_counts[job_idx, feat_idx]
                    if total > 0:
                        current_prior_means[job_idx, feat_idx] = alpha_sums[job_idx, feat_idx] / total

            # Compare to previous iteration
            if prev_prior_means is not None:
                max_change = np.nanmax(np.abs(current_prior_means - prev_prior_means))
                pct_change = (max_change / (np.nanmax(np.abs(prev_prior_means)) + 1e-10)) * 100
                last_pct_change = pct_change  # Track for progress bar

                newline = "\n" if first_batch_check else ""
                print(f"{newline}    Batch {batch_idx}: {total_samples_run:,} samples | Max change: {pct_change:.4f}%", end='\r', flush=True)
                first_batch_check = False

                if pct_change < MC_CONVERGENCE_THRESHOLD * 100:
                    converged = True
                    break

            prev_prior_means = current_prior_means.copy()

    # Cleanup GPU memory
    del rankings_gpu, alpha_matrix_gpu, alpha_valid_gpu
    del candidate_indices_gpu, candidate_offsets_gpu, job_train_ends_gpu
    cp.get_default_memory_pool().free_all_blocks()

    # Compute final IR statistics (IR mean and std)
    mc_ir_mean = np.full((n_test_dates, n_features), np.nan, dtype=np.float32)
    mc_ir_std = np.full((n_test_dates, n_features), np.nan, dtype=np.float32)

    for job_idx in range(n_jobs):
        for feat_idx in range(n_features):
            total = total_counts[job_idx, feat_idx]
            if total >= MC_MIN_SAMPLES_JOB:
                # IR mean
                mean_ir = alpha_sums[job_idx, feat_idx] / total
                mc_ir_mean[job_idx, feat_idx] = mean_ir
                # IR std (using Welford's formula: var = E[X^2] - E[X]^2)
                if total > 1:
                    variance = (alpha_sq_sums[job_idx, feat_idx] / total) - (mean_ir * mean_ir)
                    # Ensure non-negative due to numerical precision
                    mc_ir_std[job_idx, feat_idx] = np.sqrt(max(0, variance))

    return mc_ir_mean, mc_ir_std, last_pct_change


def precompute_all_mc_ir_stats(data):
    """
    Precompute MC IR statistics for all (date, N) combinations.

    Runs Monte Carlo simulations to estimate IR mean and standard deviation.
    Note: Hitrate computation removed - only IR statistics are computed.
    """
    # Initialize random seeds for determinism
    if MC_SEED is not None:
        np.random.seed(MC_SEED)
        try:
            import cupy as cp
            cp.random.seed(MC_SEED)
        except ImportError:
            pass

    dates = data['dates']
    feature_names = data['feature_names']
    n_dates = len(dates)
    n_features = len(feature_names)

    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0][0]
    n_test_dates = n_dates - test_start_idx

    print(f"\n{'='*120}")
    print("PRECOMPUTING MC INFORMATION RATIO STATISTICS")
    print(f"{'='*120}")
    print(f"Test dates: {n_test_dates}, N values: {len(N_SATELLITES_TO_PRECOMPUTE)}")

    mc_ir_means = {}
    mc_ir_stds = {}
    candidate_masks = {}
    inverted_masks = {}
    stats_summary = []

    pbar = tqdm(N_SATELLITES_TO_PRECOMPUTE, desc="N values", ncols=120)
    for n_satellites in pbar:
        # Step 1: Find candidates
        cand_mask, inv_mask = precompute_candidates_all_dates(data, n_satellites, test_start_idx)

        # Step 2: Run MC (returns IR mean, std, and final convergence change%)
        mc_ir_mean, mc_ir_std, last_pct_change = run_mc_for_n(
            data, n_satellites, test_start_idx, cand_mask, inv_mask
        )

        # Update progress bar with convergence info
        pbar.set_postfix({"N": n_satellites, "max_change%": f"{last_pct_change:.4f}"})

        # Store (expand to full date range)
        full_ir_mean = np.full((n_dates, n_features), np.nan, dtype=np.float32)
        full_ir_std = np.full((n_dates, n_features), np.nan, dtype=np.float32)
        full_cand = np.zeros((n_dates, n_features), dtype=np.bool_)
        full_inv = np.zeros((n_dates, n_features), dtype=np.bool_)

        full_ir_mean[test_start_idx:] = mc_ir_mean
        full_ir_std[test_start_idx:] = mc_ir_std
        full_cand[test_start_idx:] = cand_mask
        full_inv[test_start_idx:] = inv_mask

        mc_ir_means[n_satellites] = full_ir_mean
        mc_ir_stds[n_satellites] = full_ir_std
        candidate_masks[n_satellites] = full_cand
        inverted_masks[n_satellites] = full_inv

        # Collect stats for printing at the end
        valid_ir = mc_ir_mean[~np.isnan(mc_ir_mean)]
        if len(valid_ir) > 0:
            stats_summary.append((
                n_satellites,
                valid_ir.mean(), valid_ir.min(), valid_ir.max()
            ))

    # Print summary after progress bar completes
    print("\nIR Statistics per N:")
    for stats in stats_summary:
        (n_sat, mean_ir, min_ir, max_ir) = stats
        print(f"  N={n_sat}: IR={mean_ir:.4f} [{min_ir:.4f}, {max_ir:.4f}]")

    return mc_ir_means, mc_ir_stds, candidate_masks, inverted_masks, test_start_idx


def save_mc_ir_stats(mc_ir_means, mc_ir_stds, candidate_masks, inverted_masks,
                     test_start_idx, dates, feature_names):
    """
    Save MC IR statistics for Bayesian belief setting.

    Saves Information Ratio mean/std estimates for each feature at each test date.
    Note: Hitrates are no longer computed.
    """
    output_file = DATA_DIR / f'mc_ir_mean_{HOLDING_MONTHS}month.npz'

    mc_ir_mean_arr = np.stack([mc_ir_means[n] for n in N_SATELLITES_TO_PRECOMPUTE], axis=0)
    mc_ir_std_arr = np.stack([mc_ir_stds[n] for n in N_SATELLITES_TO_PRECOMPUTE], axis=0)
    candidate_masks_arr = np.stack([candidate_masks[n] for n in N_SATELLITES_TO_PRECOMPUTE], axis=0)
    inverted_masks_arr = np.stack([inverted_masks[n] for n in N_SATELLITES_TO_PRECOMPUTE], axis=0)

    np.savez_compressed(
        output_file,
        mc_ir_mean=mc_ir_mean_arr,
        mc_ir_std=mc_ir_std_arr,
        candidate_masks=candidate_masks_arr,
        inverted_masks=inverted_masks_arr,
        n_satellites=np.array(N_SATELLITES_TO_PRECOMPUTE, dtype=np.int32),
        dates=dates,
        features=feature_names,
        test_start_idx=test_start_idx,
        config={
            'no_prefiltering': True, 'all_features_evaluated': True,
            'decay_half_life': DECAY_HALF_LIFE_MONTHS,
            'mc_approach': 'adaptive_convergence',
            'mc_batch_size': MC_BATCH_SIZE,
            'mc_convergence_threshold': MC_CONVERGENCE_THRESHOLD,
            'mc_max_samples': MC_MAX_SAMPLES,
            'mc_ensemble_sizes': MC_ENSEMBLE_SIZES,
            'mc_min_samples_before_check': MC_MIN_SAMPLES,
            'min_training_months': MIN_TRAINING_MONTHS
        }
    )

    print(f"\n[SAVED] {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    # Initialize random seeds for determinism
    if MC_SEED is not None:
        np.random.seed(MC_SEED)
        try:
            import cupy as cp
            cp.random.seed(MC_SEED)
        except ImportError:
            pass
        print(f"Random seed set to {MC_SEED} for deterministic MC simulations")
    else:
        print("Random seed not set - MC simulations will be non-deterministic")

    print("=" * 120)
    print("STEP 5: PRECOMPUTE MC INFORMATION RATIO STATISTICS")
    print("=" * 120)

    output_file = DATA_DIR / f'mc_ir_mean_{HOLDING_MONTHS}month.npz'
    if output_file.exists() and not FORCE_RECOMPUTE:
        print(f"\n[EXISTS] {output_file}")
        print("Set FORCE_RECOMPUTE=True to regenerate.")

        # Load and show stats
        data = np.load(output_file, allow_pickle=True)
        print(f"\nExisting output shape: {data['mc_ir_mean'].shape}")
        print(f"N values: {list(data['n_satellites'])}")
        return

    # Load data from Steps 1, 3, 4
    try:
        data = load_data()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1

    # Warmup numba JIT
    print("\nWarming up Numba JIT compiler...")
    _ = evaluate_all_features_all_dates(
        data['feature_ir'][:20], 1, 10, 20
    )

    start_time = time.time()
    (mc_ir_means, mc_ir_stds,
     candidate_masks, inverted_masks, test_start_idx) = precompute_all_mc_ir_stats(data)
    elapsed = time.time() - start_time

    save_mc_ir_stats(mc_ir_means, mc_ir_stds,
                     candidate_masks, inverted_masks, test_start_idx,
                     data['dates'], data['feature_names'])

    print(f"\n{'='*120}")
    print("STEP 5 COMPLETE")
    print(f"{'='*120}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
