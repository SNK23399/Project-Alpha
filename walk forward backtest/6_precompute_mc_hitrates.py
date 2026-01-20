"""
Step 5b: Precompute Monte Carlo Hit Rates (Walk-Forward Pipeline)
=================================================================

OPTIMIZED VERSION - Fast GPU precomputation with Numba-accelerated aggregation.

This script precomputes Monte Carlo hit rates for ALL features at each
(date, N) combination. This allows the walk-forward backtest (script 7)
to run very fast by loading precomputed MC results.

Output:
    data/mc_hitrates_{holding_months}month.npz

Usage:
    python 5b_precompute_mc_hitrates.py
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
    print("CuPy not available - falling back to CPU")

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================
# CONFIGURATION
# ============================================================

HOLDING_MONTHS = 1
N_SATELLITES_TO_PRECOMPUTE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MIN_TRAINING_MONTHS = 36
MIN_ALPHA = 0.00
MIN_HIT_RATE = 0.5
DECAY_HALF_LIFE_MONTHS = 54

# Monte Carlo settings
MC_SAMPLES_PER_MONTH = 100000
MC_ENSEMBLE_SIZES = [2, 3, 4, 5, 6, 7, 8]
MC_MIN_SAMPLES = 100

# GPU settings
GPU_BLOCK_SIZE = 256

FORCE_RECOMPUTE = True
DATA_DIR = Path(__file__).parent / 'data'


# ============================================================
# NUMBA CPU FUNCTIONS
# ============================================================

@njit(cache=True, parallel=True)
def evaluate_all_features_all_dates(feature_alpha, feature_hit, n_satellites, test_start_idx, n_dates):
    """Evaluate all features for ALL test dates at once."""
    n_test_dates = n_dates - test_start_idx
    n_features = feature_alpha.shape[1]

    all_avg_alphas = np.empty((n_test_dates, n_features), dtype=np.float64)
    all_hit_rates = np.empty((n_test_dates, n_features), dtype=np.float64)

    decay_rate = np.log(2) / DECAY_HALF_LIFE_MONTHS

    for test_offset in prange(n_test_dates):
        test_idx = test_start_idx + test_offset

        for feat_idx in range(n_features):
            sum_alpha = 0.0
            sum_hit = 0.0
            sum_weight = 0.0

            for i in range(test_idx):
                months_ago = test_idx - i
                weight = np.exp(-decay_rate * months_ago)
                alpha = feature_alpha[i, feat_idx, n_satellites - 1]
                hit = feature_hit[i, feat_idx, n_satellites - 1]
                if not np.isnan(alpha):
                    sum_alpha += alpha * weight
                    sum_hit += hit * weight
                    sum_weight += weight

            if sum_weight > 0:
                all_avg_alphas[test_offset, feat_idx] = sum_alpha / sum_weight
                all_hit_rates[test_offset, feat_idx] = sum_hit / sum_weight
            else:
                all_avg_alphas[test_offset, feat_idx] = -999.0
                all_hit_rates[test_offset, feat_idx] = -999.0

    return all_avg_alphas, all_hit_rates


@njit(cache=True)
def aggregate_batch_numba(alphas, hits, ensemble_indices, ensemble_sizes,
                          candidate_offsets, sample_to_job, all_candidates,
                          hit_counts, total_counts, alpha_sums, alpha_sq_sums):
    """
    Numba-accelerated aggregation of MC batch results.
    Accumulates into existing arrays for hit rates AND alpha statistics.

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
        is_hit = hits[i] > 0
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
            if is_hit:
                hit_counts[job_idx, feat_idx] += 1

            # Track alpha statistics (for t-test later)
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
        out_alphas,
        out_hits
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
            out_hits[sample_idx] = 0
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
                out_hits[sample_idx] = 0
                return

            selected[k] = best_idx
            best_alpha_sum += alpha_matrix[month_idx, best_idx]
            best_count += 1

        if best_count == 0:
            out_alphas[sample_idx] = math.nan
            out_hits[sample_idx] = 0
        else:
            avg_alpha = best_alpha_sum / best_count
            out_alphas[sample_idx] = avg_alpha
            out_hits[sample_idx] = 1 if avg_alpha > 0 else 0


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load precomputed data."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    horizon_label = f"{HOLDING_MONTHS}month"

    alpha_df = pd.read_parquet(DATA_DIR / f'forward_alpha_{horizon_label}.parquet')
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])

    npz_data = np.load(DATA_DIR / f'rankings_matrix_{horizon_label}.npz', allow_pickle=True)
    rankings = npz_data['rankings'].astype(np.float64)
    dates = pd.to_datetime(npz_data['dates'])
    isins = npz_data['isins']
    feature_names = list(npz_data['features'])

    fa_data = np.load(DATA_DIR / f'feature_alpha_{horizon_label}.npz', allow_pickle=True)
    feature_alpha = fa_data['feature_alpha'].astype(np.float64)
    feature_hit = fa_data['feature_hit'].astype(np.float64)

    n_dates, n_isins = len(dates), len(isins)
    isin_to_idx = {isin: idx for idx, isin in enumerate(isins)}
    date_to_idx = {date: idx for idx, date in enumerate(dates)}

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

    return {
        'alpha_matrix': alpha_matrix,
        'alpha_valid': alpha_valid,
        'rankings': rankings,
        'dates': dates,
        'isins': isins,
        'feature_names': feature_names,
        'feature_alpha': feature_alpha,
        'feature_hit': feature_hit
    }


# ============================================================
# PRECOMPUTATION
# ============================================================

def precompute_candidates_all_dates(data, n_satellites, test_start_idx):
    """Precompute candidate features for ALL test dates at once."""
    feature_alpha = data['feature_alpha']
    feature_hit = data['feature_hit']
    n_dates = len(data['dates'])
    n_features = feature_alpha.shape[1]
    n_test_dates = n_dates - test_start_idx

    all_avg_alphas, all_hit_rates = evaluate_all_features_all_dates(
        feature_alpha, feature_hit, n_satellites, test_start_idx, n_dates
    )

    candidate_mask = np.zeros((n_test_dates, n_features), dtype=np.bool_)
    inverted_mask = np.zeros((n_test_dates, n_features), dtype=np.bool_)

    for test_offset in range(n_test_dates):
        avg_alphas = all_avg_alphas[test_offset]
        hit_rates = all_hit_rates[test_offset]

        positive_mask = (avg_alphas >= MIN_ALPHA) & (hit_rates >= MIN_HIT_RATE) & (avg_alphas > -900)
        negative_mask = (avg_alphas <= -MIN_ALPHA) & ((1 - hit_rates) >= MIN_HIT_RATE) & (avg_alphas > -900)

        candidate_mask[test_offset] = positive_mask | negative_mask
        inverted_mask[test_offset] = negative_mask

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

    # Use the configured samples per month
    samples_per_job = []
    for job_idx in range(n_jobs):
        n_cands = candidate_offsets[job_idx + 1] - candidate_offsets[job_idx]
        samples_per_job.append(MC_SAMPLES_PER_MONTH if n_cands >= 2 else 0)

    total_samples = sum(samples_per_job)
    if total_samples == 0:
        return (np.full((n_test_dates, n_features), np.nan, dtype=np.float32),
                np.zeros((n_test_dates, n_features), dtype=np.int32),
                np.full((n_test_dates, n_features), np.nan, dtype=np.float32),
                np.full((n_test_dates, n_features), np.nan, dtype=np.float32))

    # Process in batches to manage memory
    BATCH_SIZE = 2_000_000  # 2M samples per batch
    n_batches = (total_samples + BATCH_SIZE - 1) // BATCH_SIZE

    # Accumulate results across batches
    hit_counts = np.zeros((n_jobs, n_features), dtype=np.int64)
    total_counts = np.zeros((n_jobs, n_features), dtype=np.int64)
    alpha_sums = np.zeros((n_jobs, n_features), dtype=np.float64)
    alpha_sq_sums = np.zeros((n_jobs, n_features), dtype=np.float64)

    # Build sample-to-job mapping for the full set
    sample_to_job_full = np.zeros(total_samples, dtype=np.int64)
    idx = 0
    for job_idx, n_samples in enumerate(samples_per_job):
        sample_to_job_full[idx:idx + n_samples] = job_idx
        idx += n_samples

    max_ens_size = max(MC_ENSEMBLE_SIZES) + 1
    ensemble_sizes_choices = np.array(MC_ENSEMBLE_SIZES, dtype=np.int64)

    if GPU_AVAILABLE:
        # Transfer static data to GPU once
        rankings_gpu = cp.asarray(rankings, dtype=cp.float64)
        alpha_matrix_gpu = cp.asarray(alpha_matrix, dtype=cp.float64)
        alpha_valid_gpu = cp.asarray(alpha_valid, dtype=cp.bool_)
        candidate_indices_gpu = cp.asarray(all_candidates, dtype=cp.int64)
        candidate_offsets_gpu = cp.asarray(candidate_offsets, dtype=cp.int64)
        job_train_ends_gpu = cp.asarray(job_train_ends, dtype=cp.int64)

    for batch_idx in range(n_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min((batch_idx + 1) * BATCH_SIZE, total_samples)
        batch_samples = batch_end - batch_start

        sample_to_job_batch = sample_to_job_full[batch_start:batch_end]

        if GPU_AVAILABLE:
            # Generate random data on GPU
            ensemble_sizes_gpu = cp.random.choice(
                cp.array(ensemble_sizes_choices), size=(batch_samples,)
            ).astype(cp.int32)

            ensemble_indices_gpu = cp.random.randint(
                0, 2**31 - 1, size=(batch_samples, max_ens_size), dtype=cp.int64
            )

            sample_to_job_gpu = cp.asarray(sample_to_job_batch, dtype=cp.int64)

            out_alphas = cp.zeros(batch_samples, dtype=cp.float64)
            out_hits = cp.zeros(batch_samples, dtype=cp.int32)

            # Launch kernel
            threads = GPU_BLOCK_SIZE
            blocks = (batch_samples + threads - 1) // threads

            mc_evaluate_kernel[blocks, threads](
                rankings_gpu, alpha_matrix_gpu, alpha_valid_gpu,
                candidate_indices_gpu, candidate_offsets_gpu,
                ensemble_indices_gpu, ensemble_sizes_gpu,
                sample_to_job_gpu, job_train_ends_gpu,
                n_satellites, out_alphas, out_hits
            )
            cuda.synchronize()

            # Transfer back
            alphas_np = cp.asnumpy(out_alphas)
            hits_np = cp.asnumpy(out_hits)
            ensemble_indices_np = cp.asnumpy(ensemble_indices_gpu)
            ensemble_sizes_np = cp.asnumpy(ensemble_sizes_gpu)

        else:
            # CPU fallback - skip for now
            continue

        # Aggregate this batch using Numba (fast)
        aggregate_batch_numba(
            alphas_np, hits_np, ensemble_indices_np, ensemble_sizes_np,
            candidate_offsets, sample_to_job_batch, all_candidates,
            hit_counts, total_counts, alpha_sums, alpha_sq_sums
        )

    if GPU_AVAILABLE:
        del rankings_gpu, alpha_matrix_gpu, alpha_valid_gpu
        del candidate_indices_gpu, candidate_offsets_gpu, job_train_ends_gpu
        cp.get_default_memory_pool().free_all_blocks()

    # Compute final hit rates and alpha statistics
    mc_hitrates = np.full((n_test_dates, n_features), np.nan, dtype=np.float32)
    mc_samples = np.zeros((n_test_dates, n_features), dtype=np.int32)
    mc_alpha_mean = np.full((n_test_dates, n_features), np.nan, dtype=np.float32)
    mc_alpha_std = np.full((n_test_dates, n_features), np.nan, dtype=np.float32)

    for job_idx in range(n_jobs):
        for feat_idx in range(n_features):
            total = total_counts[job_idx, feat_idx]
            mc_samples[job_idx, feat_idx] = total
            if total >= MC_MIN_SAMPLES:
                mc_hitrates[job_idx, feat_idx] = hit_counts[job_idx, feat_idx] / total
                # Alpha mean
                mean_alpha = alpha_sums[job_idx, feat_idx] / total
                mc_alpha_mean[job_idx, feat_idx] = mean_alpha
                # Alpha std (using Welford's formula: var = E[X^2] - E[X]^2)
                if total > 1:
                    variance = (alpha_sq_sums[job_idx, feat_idx] / total) - (mean_alpha * mean_alpha)
                    # Ensure non-negative due to numerical precision
                    mc_alpha_std[job_idx, feat_idx] = np.sqrt(max(0, variance))

    return mc_hitrates, mc_samples, mc_alpha_mean, mc_alpha_std


def precompute_all_mc_hitrates(data):
    """Precompute MC hit rates for all (date, N) combinations."""
    dates = data['dates']
    feature_names = data['feature_names']
    n_dates = len(dates)
    n_features = len(feature_names)

    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0][0]
    n_test_dates = n_dates - test_start_idx

    print(f"\n{'='*60}")
    print("PRECOMPUTING MC HIT RATES AND ALPHA STATISTICS")
    print(f"{'='*60}")
    print(f"Test dates: {n_test_dates}, N values: {len(N_SATELLITES_TO_PRECOMPUTE)}")

    mc_hitrates = {}
    mc_samples = {}
    mc_alpha_means = {}
    mc_alpha_stds = {}
    candidate_masks = {}
    inverted_masks = {}
    stats_summary = []

    for n_satellites in tqdm(N_SATELLITES_TO_PRECOMPUTE, desc="N values"):
        # Step 1: Find candidates
        cand_mask, inv_mask = precompute_candidates_all_dates(data, n_satellites, test_start_idx)

        # Step 2: Run MC (now returns alpha stats too)
        mc_hr, mc_samp, mc_alpha_mean, mc_alpha_std = run_mc_for_n(
            data, n_satellites, test_start_idx, cand_mask, inv_mask
        )

        # Store (expand to full date range)
        full_mc = np.full((n_dates, n_features), np.nan, dtype=np.float32)
        full_samp = np.zeros((n_dates, n_features), dtype=np.int32)
        full_alpha_mean = np.full((n_dates, n_features), np.nan, dtype=np.float32)
        full_alpha_std = np.full((n_dates, n_features), np.nan, dtype=np.float32)
        full_cand = np.zeros((n_dates, n_features), dtype=np.bool_)
        full_inv = np.zeros((n_dates, n_features), dtype=np.bool_)

        full_mc[test_start_idx:] = mc_hr
        full_samp[test_start_idx:] = mc_samp
        full_alpha_mean[test_start_idx:] = mc_alpha_mean
        full_alpha_std[test_start_idx:] = mc_alpha_std
        full_cand[test_start_idx:] = cand_mask
        full_inv[test_start_idx:] = inv_mask

        mc_hitrates[n_satellites] = full_mc
        mc_samples[n_satellites] = full_samp
        mc_alpha_means[n_satellites] = full_alpha_mean
        mc_alpha_stds[n_satellites] = full_alpha_std
        candidate_masks[n_satellites] = full_cand
        inverted_masks[n_satellites] = full_inv

        # Collect stats for printing at the end
        valid_hr = mc_hr[~np.isnan(mc_hr)]
        valid_samp = mc_samp[mc_samp > 0]
        valid_alpha = mc_alpha_mean[~np.isnan(mc_alpha_mean)]
        if len(valid_hr) > 0:
            stats_summary.append((
                n_satellites,
                valid_hr.mean(), valid_hr.min(), valid_hr.max(),
                valid_samp.mean(), valid_samp.min(), valid_samp.max(),
                valid_alpha.mean() * 100, valid_alpha.min() * 100, valid_alpha.max() * 100
            ))

    # Print summary after progress bar completes
    print("\nStatistics per N:")
    for stats in stats_summary:
        (n_sat, mean_hr, min_hr, max_hr,
         mean_samp, min_samp, max_samp,
         mean_alpha, min_alpha, max_alpha) = stats
        print(f"  N={n_sat}: HR={mean_hr:.1%} [{min_hr:.1%}, {max_hr:.1%}], "
              f"Alpha={mean_alpha:.2f}% [{min_alpha:.2f}%, {max_alpha:.2f}%], "
              f"samples={mean_samp:.0f}")

    return mc_hitrates, mc_samples, mc_alpha_means, mc_alpha_stds, candidate_masks, inverted_masks, test_start_idx


def save_mc_hitrates(mc_hitrates, mc_samples, mc_alpha_means, mc_alpha_stds,
                     candidate_masks, inverted_masks, test_start_idx, dates, feature_names):
    """Save results including sample counts and alpha statistics for significance testing."""
    output_file = DATA_DIR / f'mc_hitrates_{HOLDING_MONTHS}month.npz'

    mc_hitrates_arr = np.stack([mc_hitrates[n] for n in N_SATELLITES_TO_PRECOMPUTE], axis=0)
    mc_samples_arr = np.stack([mc_samples[n] for n in N_SATELLITES_TO_PRECOMPUTE], axis=0)
    mc_alpha_mean_arr = np.stack([mc_alpha_means[n] for n in N_SATELLITES_TO_PRECOMPUTE], axis=0)
    mc_alpha_std_arr = np.stack([mc_alpha_stds[n] for n in N_SATELLITES_TO_PRECOMPUTE], axis=0)
    candidate_masks_arr = np.stack([candidate_masks[n] for n in N_SATELLITES_TO_PRECOMPUTE], axis=0)
    inverted_masks_arr = np.stack([inverted_masks[n] for n in N_SATELLITES_TO_PRECOMPUTE], axis=0)

    np.savez_compressed(
        output_file,
        mc_hitrates=mc_hitrates_arr,
        mc_samples=mc_samples_arr,
        mc_alpha_mean=mc_alpha_mean_arr,   # NEW: mean alpha per feature
        mc_alpha_std=mc_alpha_std_arr,     # NEW: std alpha per feature (for t-test)
        candidate_masks=candidate_masks_arr,
        inverted_masks=inverted_masks_arr,
        n_satellites=np.array(N_SATELLITES_TO_PRECOMPUTE, dtype=np.int32),
        dates=dates,
        features=feature_names,
        test_start_idx=test_start_idx,
        config={
            'min_alpha': MIN_ALPHA, 'min_hit_rate': MIN_HIT_RATE,
            'decay_half_life': DECAY_HALF_LIFE_MONTHS,
            'mc_samples_per_month': MC_SAMPLES_PER_MONTH,
            'mc_ensemble_sizes': MC_ENSEMBLE_SIZES,
            'mc_min_samples': MC_MIN_SAMPLES,
            'min_training_months': MIN_TRAINING_MONTHS
        }
    )

    print(f"\n[SAVED] {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    print("=" * 60)
    print("PRECOMPUTE MC HIT RATES")
    print("=" * 60)

    output_file = DATA_DIR / f'mc_hitrates_{HOLDING_MONTHS}month.npz'
    if output_file.exists() and not FORCE_RECOMPUTE:
        print(f"\n[EXISTS] {output_file}")
        print("Set FORCE_RECOMPUTE=True to regenerate.")
        return

    data = load_data()

    # Warmup
    print("\nWarming up...")
    _ = evaluate_all_features_all_dates(
        data['feature_alpha'][:20], data['feature_hit'][:20], 1, 10, 20
    )

    start_time = time.time()
    (mc_hitrates, mc_samples, mc_alpha_means, mc_alpha_stds,
     candidate_masks, inverted_masks, test_start_idx) = precompute_all_mc_hitrates(data)
    elapsed = time.time() - start_time

    save_mc_hitrates(mc_hitrates, mc_samples, mc_alpha_means, mc_alpha_stds,
                     candidate_masks, inverted_masks, test_start_idx,
                     data['dates'], data['feature_names'])

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
