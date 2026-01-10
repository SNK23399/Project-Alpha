"""
GPU-Accelerated Signal Framework for ETF Alpha Prediction
==========================================================

Main orchestration script that combines:
- Signal bases (signal_bases.py)
- Smoothing filters (signal_filters.py)
- Indicator transformations (signal_indicators.py)

Uses CuPy for GPU-accelerated numpy operations.
Uses Numba for JIT-compiled feature testing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import time
from numba import njit, prange
from tqdm import tqdm

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print(f"Using device: cuda (CuPy {cp.__version__})")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available, falling back to CPU")
    cp = np

import sys
sys.path.insert(0, str(Path(__file__).parent / 'support'))
from etf_database import ETFDatabase

# Import from our modules
from signal_bases import compute_all_signal_bases
from signal_filters import DEFAULT_FILTER_CONFIGS, apply_filter
from signal_indicators import compute_rolling_stats, compute_indicators_chunked, N_INDICATORS

CORE_ISIN = "IE00B6R52259"


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load satellite prices and core prices from database."""
    db = ETFDatabase()
    prices = db.load_all_prices()
    prices.index = pd.to_datetime(prices.index)

    core_prices = prices[CORE_ISIN]
    satellite_prices = prices.drop(columns=[CORE_ISIN])

    # Filter to ETFs with sufficient data
    min_days = 252 * 5
    valid_cols = satellite_prices.columns[satellite_prices.notna().sum() > min_days]
    satellite_prices = satellite_prices[valid_cols]

    return satellite_prices, core_prices


# =============================================================================
# FEATURE RESULT DATACLASS
# =============================================================================

@dataclass
class FeatureResult:
    name: str
    signal_base: str
    indicator: str
    correlation: float
    quintile_spread: float  # Long-short: top 20% - bottom 20%
    topk_advantage: float   # Long-only: top k - average (like ensemble_test.py)
    hit_rate: float         # % of months above-median feature -> above-median alpha
    hit_rate_topk: float    # % of months top-k beats average
    coverage: float

    @property
    def score(self) -> float:
        # Use topk_advantage as primary metric (matches ensemble_test.py methodology)
        return abs(self.topk_advantage) * 2.0 + abs(self.hit_rate_topk - 0.5) * 1.0


# =============================================================================
# BATCH INDICATOR COMPUTATION WITH FILTERING
# =============================================================================

def compute_and_test_features_streaming(
    signals_3d: np.ndarray,  # Shape: (n_signals, n_time, n_etfs)
    signal_names: List[str],
    alpha_monthly: np.ndarray,  # (n_months, n_etfs)
    monthly_indices: np.ndarray,  # indices into daily data for month-end
    select_k: int = 4
) -> List['FeatureResult']:
    """
    Compute and test features in a streaming fashion to avoid memory issues.

    Instead of pre-allocating all features, we:
    1. Process one filter at a time
    2. Compute indicators for that filter
    3. Test those features immediately
    4. Discard the features and keep only results

    This dramatically reduces memory usage.

    Returns:
        results: list of FeatureResult objects
    """
    n_signals, n_time, n_etfs = signals_3d.shape

    # Use default filter configs from signal_filters module
    filter_configs = DEFAULT_FILTER_CONFIGS

    n_filters = len(filter_configs)
    n_indicators = N_INDICATORS

    print(f"  Signal variants: {n_signals} signals × {n_filters} filters = {n_signals * n_filters}")
    print(f"  Total features: {n_signals * n_filters} × {n_indicators} indicators = {n_signals * n_filters * n_indicators}")
    print(f"  Processing in streaming mode to save memory...")

    # Pre-forward-fill signals ONCE to avoid repeated ffill in filters
    # OPTIMIZED: Use pandas on flattened 2D array for batch ffill
    print("  Pre-filling NaN values...")
    signals_flat = signals_3d.transpose(1, 0, 2).reshape(n_time, -1)  # (n_time, n_signals*n_etfs)
    df_flat = pd.DataFrame(signals_flat)
    signals_filled_flat = df_flat.ffill().values.astype(np.float32)
    signals_filled = signals_filled_flat.reshape(n_time, n_signals, n_etfs).transpose(1, 0, 2)

    # Winsorize signals to 1st/99th percentile (standard in quant finance)
    # Note: signal_bases.py now uses _safe_divide() to prevent extreme values at source
    print("  Winsorizing signals (1st/99th percentile)...")
    for sig_idx in range(n_signals):
        sig_data = signals_filled[sig_idx]
        finite_mask = np.isfinite(sig_data)
        if finite_mask.any():
            p1 = np.percentile(sig_data[finite_mask], 1)
            p99 = np.percentile(sig_data[finite_mask], 99)
            signals_filled[sig_idx] = np.clip(sig_data, p1, p99)

    all_results = []
    indicator_names = None
    n_filters = len(filter_configs)

    # Results file for incremental saving
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    checkpoint_file = results_dir / "feature_discovery_checkpoint.parquet"

    # Check for existing checkpoint to resume from
    completed_filters = set()
    if checkpoint_file.exists():
        try:
            existing_df = pd.read_parquet(checkpoint_file)
            completed_filters = set(existing_df['filter'].unique())
            print(f"  Resuming from checkpoint: {len(completed_filters)} filters already completed")
            print(f"  Completed filters: {sorted(completed_filters)}")
            # Reconstruct all_results from checkpoint
            for _, row in existing_df.iterrows():
                # Reconstruct signal_base (variant_name): "signal" for raw, "signal__filter" for filtered
                if row['filter'] == 'raw':
                    signal_base = row['signal']
                else:
                    signal_base = f"{row['signal']}__{row['filter']}"
                all_results.append(FeatureResult(
                    name=row['feature'],
                    signal_base=signal_base,
                    indicator=row['indicator'],
                    correlation=row['correlation'],
                    quintile_spread=row['spread'],
                    topk_advantage=row['topk_advantage'],
                    hit_rate=row['hit_rate'],
                    hit_rate_topk=row['hit_rate_topk'],
                    coverage=row['coverage']
                ))
            del existing_df
        except Exception as e:
            print(f"  Could not load checkpoint: {e}")

    # Force GPU memory cleanup BEFORE starting filter loop
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()

    # Process each filter with tqdm progress bar
    filter_pbar = tqdm(filter_configs, desc="Filters", unit="filter", position=0)

    for filter_idx, (filter_name, filter_func, filter_kwargs) in enumerate(filter_pbar):
        filter_pbar.set_description(f"Filter: {filter_name:<20}")

        # Skip already completed filters
        if filter_name in completed_filters:
            filter_pbar.set_postfix(status="skipped")
            continue

        try:
            # Step 1: Apply filter
            filtered_signals = apply_filter(signals_filled, filter_name, filter_func, filter_kwargs)

            # Step 2: Compute rolling stats (fast_mode skips skew/kurt/autocorr, fp16 for speed)
            rolling_stats = compute_rolling_stats(
                filtered_signals, n_time, keep_on_gpu=False, verbose=False,
                fast_mode=True, use_fp16=True
            )

            # Build variant names for this filter
            variant_names = []
            for sig_name in signal_names:
                if filter_name == 'raw':
                    variant_names.append(sig_name)
                else:
                    variant_names.append(f'{sig_name}__{filter_name}')

            # Step 3+4: Compute indicators IN CHUNKS and test immediately
            filter_results = []
            chunk_count = 0
            n_chunks_expected = 8  # ~76 indicators / 10 per chunk

            for chunk_features, chunk_ind_names in compute_indicators_chunked(
                filtered_signals, rolling_stats, n_time, n_etfs,
                use_fp16=True, chunk_size=1  # Yield each indicator immediately to save memory
            ):
                chunk_count += 1

                # Store indicator names (same for all filters)
                if indicator_names is None:
                    indicator_names = []
                if chunk_count == 1 and filter_idx == 0:
                    indicator_names = chunk_ind_names.copy()
                elif filter_idx == 0:
                    indicator_names.extend(chunk_ind_names)

                # Test this chunk immediately
                chunk_results = _test_features_for_filter(
                    chunk_features, variant_names, chunk_ind_names,
                    alpha_monthly, monthly_indices, select_k
                )
                filter_results.extend(chunk_results)

                # Free chunk memory immediately
                del chunk_features

            all_results.extend(filter_results)

            # Show results for this filter
            n_good = sum(1 for r in filter_results if r.topk_advantage > 0)
            filter_pbar.set_postfix(positive=f"{n_good}/{len(filter_results)}")

            # Save checkpoint after each filter
            _save_results_checkpoint(all_results, checkpoint_file)
            print(f"  Filter {filter_name} complete, checkpoint saved.", flush=True)

            # Free memory immediately
            del filtered_signals, rolling_stats, filter_results

        except Exception as e:
            print(f"\n  ERROR in filter {filter_name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            # Save what we have so far
            _save_results_checkpoint(all_results, checkpoint_file)
            print(f"  Checkpoint saved with {len(all_results)} results. Continuing...")
            continue

        # Force GPU memory cleanup to prevent OOM on later filters
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    filter_pbar.close()
    print(f"\n  Results saved to: {checkpoint_file.with_suffix('.parquet')}")

    return all_results, indicator_names


def _save_results_checkpoint(results: List, filepath: Path):
    """Save results to Parquet checkpoint file (fast, compressed, queryable)."""
    if not results:
        return

    # Change extension to .parquet
    filepath = filepath.with_suffix('.parquet')

    rows = []
    for r in results:
        # Parse filter from signal_base (format: "signal" for raw, "signal__filter" for filtered)
        # signal_base is set to variant_name which contains the filter info
        if '__' in r.signal_base:
            parts = r.signal_base.split('__')
            signal_name = parts[0]
            filter_name = parts[1]
        else:
            signal_name = r.signal_base
            filter_name = 'raw'

        rows.append({
            'feature': r.name,
            'signal': signal_name,
            'filter': filter_name,
            'indicator': r.indicator,
            'correlation': np.float32(r.correlation),
            'spread': np.float32(r.quintile_spread),
            'topk_advantage': np.float32(r.topk_advantage),
            'hit_rate': np.float32(r.hit_rate),
            'hit_rate_topk': np.float32(r.hit_rate_topk),
            'coverage': np.float32(r.coverage)
        })

    df = pd.DataFrame(rows)

    # Use category dtype for strings (huge compression savings)
    for col in ['feature', 'signal', 'filter', 'indicator']:
        df[col] = df[col].astype('category')

    # Save as parquet with compression
    df.to_parquet(filepath, engine='pyarrow', compression='zstd', index=False)


@njit(parallel=True, cache=True)
def _test_features_numba(
    features_for_pred: np.ndarray,  # (n_features, n_months-1, n_etfs)
    alpha_next: np.ndarray,  # (n_months-1, n_etfs)
    select_k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba JIT-compiled feature testing - runs in parallel across features.
    Returns arrays of metrics for each feature.
    """
    n_features, n_pred_months, n_etfs = features_for_pred.shape

    # Output arrays
    correlations = np.zeros(n_features, dtype=np.float64)
    spreads = np.zeros(n_features, dtype=np.float64)
    topk_advantages = np.zeros(n_features, dtype=np.float64)
    hit_rates = np.zeros(n_features, dtype=np.float64)
    hit_rates_topk = np.zeros(n_features, dtype=np.float64)
    coverage = np.zeros(n_features, dtype=np.float64)

    # Process features in parallel
    for feat_idx in prange(n_features):
        sum_corr = 0.0
        sum_spread = 0.0
        sum_topk = 0.0
        sum_hit = 0.0
        count_topk_positive = 0
        n_valid = 0
        n_corr_valid = 0

        for m in range(n_pred_months):
            # Count valid (non-NaN) entries
            n_valid_etfs = 0
            for e in range(n_etfs):
                if not np.isnan(features_for_pred[feat_idx, m, e]) and not np.isnan(alpha_next[m, e]):
                    n_valid_etfs += 1

            if n_valid_etfs < 20:
                continue

            n_valid += 1

            # Extract valid values into temporary arrays
            feat_vals = np.empty(n_valid_etfs, dtype=np.float64)
            alpha_vals = np.empty(n_valid_etfs, dtype=np.float64)
            idx = 0
            for e in range(n_etfs):
                if not np.isnan(features_for_pred[feat_idx, m, e]) and not np.isnan(alpha_next[m, e]):
                    feat_vals[idx] = features_for_pred[feat_idx, m, e]
                    alpha_vals[idx] = alpha_next[m, e]
                    idx += 1

            # Compute correlation
            feat_mean = np.mean(feat_vals)
            alpha_mean = np.mean(alpha_vals)
            num = 0.0
            denom_feat = 0.0
            denom_alpha = 0.0
            for i in range(n_valid_etfs):
                fc = feat_vals[i] - feat_mean
                ac = alpha_vals[i] - alpha_mean
                num += fc * ac
                denom_feat += fc * fc
                denom_alpha += ac * ac
            denom = np.sqrt(denom_feat * denom_alpha)
            if denom > 1e-10:
                sum_corr += num / denom
                n_corr_valid += 1

            # Sort by feature value (simple bubble sort for small arrays)
            sorted_idx = np.argsort(feat_vals)

            # Quintile spread
            k_quintile = n_valid_etfs // 5
            if k_quintile > 0:
                top_sum = 0.0
                bot_sum = 0.0
                for i in range(k_quintile):
                    top_sum += alpha_vals[sorted_idx[n_valid_etfs - 1 - i]]
                    bot_sum += alpha_vals[sorted_idx[i]]
                sum_spread += (top_sum - bot_sum) / k_quintile

            # Top-k advantage
            k_select = min(select_k, n_valid_etfs // 2)
            if k_select > 0:
                topk_sum = 0.0
                for i in range(k_select):
                    topk_sum += alpha_vals[sorted_idx[n_valid_etfs - 1 - i]]
                topk_avg = topk_sum / k_select
                topk_adv = topk_avg - alpha_mean
                sum_topk += topk_adv
                if topk_adv > 0:
                    count_topk_positive += 1

            # Hit rate (median split)
            med_feat = np.median(feat_vals)
            med_alpha = np.median(alpha_vals)
            hits = 0
            for i in range(n_valid_etfs):
                if (feat_vals[i] > med_feat) == (alpha_vals[i] > med_alpha):
                    hits += 1
            sum_hit += hits / n_valid_etfs

        # Store results
        if n_valid >= 12:
            correlations[feat_idx] = sum_corr / n_corr_valid if n_corr_valid > 0 else 0.0
            spreads[feat_idx] = sum_spread / n_valid
            topk_advantages[feat_idx] = sum_topk / n_valid
            hit_rates[feat_idx] = sum_hit / n_valid
            hit_rates_topk[feat_idx] = count_topk_positive / n_valid
            coverage[feat_idx] = n_valid / n_pred_months
        else:
            coverage[feat_idx] = -1.0  # Mark as invalid

    return correlations, spreads, topk_advantages, hit_rates, hit_rates_topk, coverage


def _test_features_for_filter(
    features_4d: np.ndarray,  # (n_signals, n_indicators, n_time, n_etfs)
    variant_names: List[str],
    indicator_names: List[str],
    alpha_monthly: np.ndarray,
    monthly_indices: np.ndarray,
    select_k: int = 4
) -> List['FeatureResult']:
    """
    Test features for a single filter's worth of signals.
    Uses Numba JIT compilation for massive speedup.
    """
    n_signals, n_indicators = features_4d.shape[:2]
    n_months = len(alpha_monthly)

    # Resample features to monthly (take month-end values)
    features_monthly = features_4d[:, :, monthly_indices, :]

    # Flatten to (n_features, n_months, n_etfs) for batch processing
    n_features = n_signals * n_indicators
    features_flat = features_monthly.reshape(n_features, len(monthly_indices), -1)

    # Prepare data for Numba function
    alpha_next = alpha_monthly[1:].astype(np.float64)
    features_for_pred = features_flat[:, :-1, :].astype(np.float64)

    # Run JIT-compiled feature testing
    correlations, spreads, topk_advantages, hit_rates, hit_rates_topk, coverage = \
        _test_features_numba(features_for_pred, alpha_next, select_k)

    # Build results
    results = []
    for feat_idx in range(n_features):
        if coverage[feat_idx] < 0:
            continue

        sig_idx = feat_idx // n_indicators
        ind_idx = feat_idx % n_indicators
        variant_name = variant_names[sig_idx]

        result = FeatureResult(
            name=f'{variant_name}__{indicator_names[ind_idx]}',
            signal_base=variant_name,
            indicator=indicator_names[ind_idx],
            correlation=correlations[feat_idx],
            quintile_spread=spreads[feat_idx],
            topk_advantage=topk_advantages[feat_idx],
            hit_rate=hit_rates[feat_idx],
            hit_rate_topk=hit_rates_topk[feat_idx],
            coverage=coverage[feat_idx]
        )
        results.append(result)

    return results


# =============================================================================
# MAIN DISCOVERY FUNCTION
# =============================================================================

def run_gpu_discovery(
    etf_prices: pd.DataFrame,
    core_prices: pd.Series,
    top_n: int = 50
) -> pd.DataFrame:
    """
    Run feature discovery with full GPU acceleration.

    Uses streaming approach to avoid memory issues:
    - Process one filter at a time
    - Test features immediately after computation
    - Discard features and keep only results
    """
    total_start = time.time()

    # 1. Compute signal bases (returns 3D array)
    print("\n1. Computing signal bases...")
    start = time.time()
    signals_3d, signal_names = compute_all_signal_bases(etf_prices, core_prices)
    print(f"   Shape: {signals_3d.shape}, Time: {time.time() - start:.1f}s")

    # 2. Compute monthly alpha (needed for testing)
    print("\n2. Computing monthly alpha...")
    monthly_prices = etf_prices.resample('ME').last()
    monthly_core = core_prices.resample('ME').last()
    monthly_ret = monthly_prices.pct_change()
    core_ret = monthly_core.pct_change()
    alpha_monthly = (monthly_ret.sub(core_ret, axis=0) * 100).values

    # Get monthly indices (map monthly dates to daily indices)
    monthly_dates = monthly_prices.index
    daily_dates = etf_prices.index
    monthly_indices = np.array([daily_dates.get_indexer([d], method='ffill')[0] for d in monthly_dates])

    # 3. Compute indicators and test features (streaming mode)
    print("\n3. Computing indicators and testing features (streaming)...")

    # Force GPU memory cleanup before filter/indicator processing
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()

    start = time.time()
    results, _ = compute_and_test_features_streaming(
        signals_3d, signal_names, alpha_monthly, monthly_indices
    )
    print(f"   Tested {len(results)} features, Time: {time.time() - start:.1f}s")

    # 4. Convert to DataFrame and sort
    if not results:
        print("No valid features found!")
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            'feature': r.name,
            'signal_base': r.signal_base,
            'indicator': r.indicator,
            'topk_advantage': r.topk_advantage,
            'quintile_spread': r.quintile_spread,
            'hit_rate_topk': r.hit_rate_topk,
            'hit_rate': r.hit_rate,
            'correlation': r.correlation,
            'coverage': r.coverage,
            'score': r.score
        }
        for r in results
    ])

    df = df.sort_values('score', ascending=False).reset_index(drop=True)

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.1f}s")

    return df.head(top_n)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("GPU-ACCELERATED SIGNAL FRAMEWORK")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    etf_prices, core_prices = load_data()
    print(f"  - ETFs: {len(etf_prices.columns)}")
    print(f"  - Date range: {etf_prices.index.min().date()} to {etf_prices.index.max().date()}")

    # Run GPU-accelerated discovery
    results = run_gpu_discovery(etf_prices, core_prices, top_n=100)

    if results.empty:
        print("No features found!")
        return

    # Print results
    print("\n" + "=" * 70)
    print("TOP 30 FEATURES BY PREDICTIVE POWER (Top-k=4 selection)")
    print("=" * 70)
    print(f"\n{'Rank':<5} {'Feature':<40} {'Top-k Adv':>10} {'Hit Rate':>10} {'Q.Spread':>10}")
    print("-" * 80)

    for i, row in results.head(30).iterrows():
        print(f"{i+1:<5} {row['feature']:<40} {row['topk_advantage']:>+9.3f}% {row['hit_rate_topk']:>9.1%} {row['quintile_spread']:>+9.3f}%")

    # Group by signal base
    print("\n" + "=" * 70)
    print("BEST INDICATOR PER SIGNAL BASE")
    print("=" * 70)

    best_per_signal = results.loc[results.groupby('signal_base')['score'].idxmax()]
    best_per_signal = best_per_signal.sort_values('score', ascending=False)

    print(f"\n{'Signal Base':<25} {'Indicator':<15} {'Top-k Adv':>10} {'Hit Rate':>10}")
    print("-" * 65)

    for _, row in best_per_signal.head(20).iterrows():
        print(f"{row['signal_base']:<25} {row['indicator']:<15} {row['topk_advantage']:>+9.3f}% {row['hit_rate_topk']:>9.1%}")

    # Group by indicator
    print("\n" + "=" * 70)
    print("AVERAGE PERFORMANCE BY INDICATOR")
    print("=" * 70)

    by_indicator = results.groupby('indicator').agg({
        'topk_advantage': 'mean',
        'hit_rate_topk': 'mean',
        'quintile_spread': 'mean',
        'score': 'mean'
    }).sort_values('score', ascending=False)

    print(f"\n{'Indicator':<20} {'Avg Top-k':>12} {'Avg Hit Rate':>12}")
    print("-" * 46)

    for ind, row in by_indicator.iterrows():
        print(f"{ind:<20} {row['topk_advantage']:>+11.3f}% {row['hit_rate_topk']:>11.1%}")

    # Compare RAW vs FILTERED signal performance
    print("\n" + "=" * 70)
    print("RAW vs FILTERED SIGNAL COMPARISON")
    print("=" * 70)

    # Extract filter type from signal_base
    def get_filter_type(signal_base):
        if '__ema_' in signal_base:
            return 'ema'
        elif '__hull_' in signal_base:
            return 'hull'
        elif '__butter_' in signal_base:
            return 'butterworth'
        else:
            return 'raw'

    results['filter_type'] = results['signal_base'].apply(get_filter_type)

    by_filter = results.groupby('filter_type').agg({
        'topk_advantage': 'mean',
        'hit_rate_topk': 'mean',
        'quintile_spread': 'mean',
        'score': 'mean'
    }).sort_values('score', ascending=False)

    print(f"\n{'Filter Type':<15} {'Avg Top-k':>12} {'Avg Hit Rate':>12} {'Avg Score':>12}")
    print("-" * 54)

    for ftype, row in by_filter.iterrows():
        print(f"{ftype:<15} {row['topk_advantage']:>+11.3f}% {row['hit_rate_topk']:>11.1%} {row['score']:>11.3f}")

    # Best feature per filter type
    print("\n" + "=" * 70)
    print("BEST FEATURE PER FILTER TYPE")
    print("=" * 70)

    best_per_filter = results.loc[results.groupby('filter_type')['score'].idxmax()]
    best_per_filter = best_per_filter.sort_values('score', ascending=False)

    print(f"\n{'Filter':<12} {'Feature':<45} {'Top-k Adv':>10}")
    print("-" * 70)

    for _, row in best_per_filter.iterrows():
        print(f"{row['filter_type']:<12} {row['feature'][:44]:<45} {row['topk_advantage']:>+9.3f}%")

    return results


if __name__ == "__main__":
    results = main()
