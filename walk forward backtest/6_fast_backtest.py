"""
Step 6: Fast Walk-Forward Backtest (Walk-Forward Pipeline)
===========================================================

OPTIMIZED VERSION with:
- 2D numpy alpha matrix for O(1) lookups (instead of pandas .loc)
- Vectorized ensemble evaluation
- Parallel processing across N values
- Numba JIT for hot loops (optional)

This script implements TRUE walk-forward backtesting:
- At each test date, feature selection uses ONLY past data
- Features are re-optimized monthly (or at configurable frequency)
- Results reflect what you would have actually achieved

Usage:
    python 6_fast_backtest.py [holding_months] [n_satellites]

Examples:
    python 6_fast_backtest.py           # Default: 1 month, all N values 1-10
    python 6_fast_backtest.py 1 4       # 1 month horizon, N=4 only
    python 6_fast_backtest.py 3         # 3 month horizon, all N values
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
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ============================================================
# CONFIGURATION
# ============================================================

# N values to test (will be filtered by available data)
N_SATELLITES_TO_TEST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Default holding period
DEFAULT_HOLDING_MONTHS = 1

# Training parameters
MIN_TRAINING_MONTHS = 36  # Minimum training history
REOPTIMIZATION_FREQUENCY = 1  # Re-optimize every N months (1 = monthly)

# Feature selection criteria
MIN_ALPHA = 0.001        # Minimum 0.1% alpha to be considered
MIN_HIT_RATE = 0.55      # Minimum 55% hit rate
MAX_ENSEMBLE_SIZE = 20   # Maximum features in ensemble
MIN_IMPROVEMENT = 0.0001 # Minimum improvement to add feature

# Parallel processing
N_CORES = max(1, cpu_count() - 1)
PARALLEL_N_VALUES = True  # Run N values in parallel

# Data and output directories (relative to this script's location)
DATA_DIR = Path(__file__).parent / 'data'
OUTPUT_DIR = DATA_DIR / 'backtest_results'


# ============================================================
# NUMBA-OPTIMIZED FUNCTIONS
# ============================================================

if HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _compute_ensemble_alpha_numba(
        rankings_slice: np.ndarray,     # (n_isins, n_selected_features)
        alpha_row: np.ndarray,          # (n_isins,)
        alpha_valid_row: np.ndarray,    # (n_isins,)
        n_satellites: int
    ) -> float:
        """
        Numba-optimized ensemble alpha computation for a single date.

        Returns average alpha of top-N ETFs selected by ensemble.
        """
        n_isins = rankings_slice.shape[0]

        # Compute mean score across features for each ISIN
        scores = np.empty(n_isins, dtype=np.float32)
        valid_scores = np.zeros(n_isins, dtype=np.bool_)

        for i in range(n_isins):
            count = 0
            total = 0.0
            for j in range(rankings_slice.shape[1]):
                val = rankings_slice[i, j]
                if not np.isnan(val):
                    total += val
                    count += 1
            if count > 0:
                scores[i] = total / count
                valid_scores[i] = True
            else:
                scores[i] = -999.0
                valid_scores[i] = False

        # Count valid scores
        n_valid = 0
        for i in range(n_isins):
            if valid_scores[i]:
                n_valid += 1

        if n_valid < n_satellites:
            return np.nan

        # Find top N indices (simple selection sort for small N)
        top_indices = np.empty(n_satellites, dtype=np.int32)
        used = np.zeros(n_isins, dtype=np.bool_)

        for k in range(n_satellites):
            best_idx = -1
            best_score = -999.0
            for i in range(n_isins):
                if valid_scores[i] and not used[i] and scores[i] > best_score:
                    best_score = scores[i]
                    best_idx = i
            if best_idx >= 0:
                top_indices[k] = best_idx
                used[best_idx] = True

        # Compute average alpha of selected ETFs
        alpha_sum = 0.0
        alpha_count = 0
        for k in range(n_satellites):
            idx = top_indices[k]
            if alpha_valid_row[idx]:
                alpha_sum += alpha_row[idx]
                alpha_count += 1

        if alpha_count == 0:
            return np.nan

        return alpha_sum / alpha_count


# ============================================================
# LOAD DATA
# ============================================================

def load_data(holding_months):
    """Load all precomputed data and prepare numpy arrays."""
    print(f"\n{'='*60}")
    print(f"LOADING PRECOMPUTED DATA")
    print(f"{'='*60}")

    data_dir = DATA_DIR

    # Load forward alpha
    alpha_file = data_dir / f'forward_alpha_{holding_months}month.parquet'
    if not alpha_file.exists():
        raise FileNotFoundError(f"Forward alpha not found: {alpha_file}\nRun 4_compute_forward_alpha.py first.")

    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])

    print(f"\n[OK] Forward alpha: {len(alpha_df):,} observations")

    # Load rankings matrix
    rankings_file = data_dir / f'rankings_matrix_{holding_months}month.npz'
    if not rankings_file.exists():
        raise FileNotFoundError(f"Rankings matrix not found: {rankings_file}")

    npz_data = np.load(rankings_file, allow_pickle=True)
    rankings = npz_data['rankings'].copy()
    dates = pd.to_datetime(npz_data['dates'])
    isins = npz_data['isins']
    feature_names = list(npz_data['features'])

    print(f"[OK] Rankings matrix: {rankings.shape}")

    # Load feature-alpha matrix (optional - for faster evaluation)
    feature_alpha_file = data_dir / f'feature_alpha_{holding_months}month.npz'
    feature_alpha = None
    feature_hit = None

    if feature_alpha_file.exists():
        fa_data = np.load(feature_alpha_file, allow_pickle=True)
        feature_alpha = fa_data['feature_alpha']
        feature_hit = fa_data['feature_hit']
        print(f"[OK] Feature-alpha matrix: {feature_alpha.shape}")
    else:
        print(f"[--] Feature-alpha matrix not found (will compute on-the-fly)")

    # Convert alpha_df to 2D numpy array for O(1) lookups
    print(f"\nPreparing alpha matrix for fast lookups...")

    n_dates = len(dates)
    n_isins = len(isins)

    # Create mappings
    isin_to_idx = {isin: idx for idx, isin in enumerate(isins)}
    date_to_idx = {date: idx for idx, date in enumerate(dates)}

    # Initialize alpha matrix
    alpha_matrix = np.full((n_dates, n_isins), np.nan, dtype=np.float32)

    # Fill matrix efficiently
    for date, group in alpha_df.groupby('date'):
        if date not in date_to_idx:
            continue
        date_idx = date_to_idx[date]

        for isin, alpha in zip(group['isin'].values, group['forward_alpha'].values):
            if isin in isin_to_idx:
                isin_idx = isin_to_idx[isin]
                alpha_matrix[date_idx, isin_idx] = alpha

    alpha_valid = ~np.isnan(alpha_matrix)

    print(f"  Alpha matrix shape: {alpha_matrix.shape}")
    print(f"  Valid entries: {alpha_valid.sum():,} ({alpha_valid.sum() / alpha_valid.size * 100:.1f}%)")

    return {
        'alpha_df': alpha_df.set_index(['date', 'isin']),  # Keep for compatibility
        'alpha_matrix': alpha_matrix,
        'alpha_valid': alpha_valid,
        'rankings': rankings,
        'dates': dates,
        'isins': isins,
        'isin_to_idx': isin_to_idx,
        'date_to_idx': date_to_idx,
        'feature_names': feature_names,
        'feature_alpha': feature_alpha,
        'feature_hit': feature_hit
    }


# ============================================================
# FAST FEATURE EVALUATION (using precomputed matrix)
# ============================================================

def evaluate_feature_fast(feature_alpha, feature_hit, feat_idx, n_satellites, date_mask):
    """
    Evaluate a single feature using precomputed alpha matrix.

    Args:
        feature_alpha: (n_dates, n_features, n_satellites_max) precomputed alphas
        feature_hit: (n_dates, n_features, n_satellites_max) precomputed hits
        feat_idx: Feature index
        n_satellites: N value (1-10)
        date_mask: Boolean mask for training dates

    Returns:
        dict with avg_alpha, hit_rate, n_periods
    """
    # Get alphas for this feature and N
    alphas = feature_alpha[date_mask, feat_idx, n_satellites - 1]
    hits = feature_hit[date_mask, feat_idx, n_satellites - 1]

    # Filter NaN
    valid = ~np.isnan(alphas)
    if valid.sum() == 0:
        return None

    alphas = alphas[valid]
    hits = hits[valid]

    return {
        'avg_alpha': np.mean(alphas),
        'std_alpha': np.std(alphas),
        'hit_rate': np.mean(hits),
        'n_periods': len(alphas)
    }


def evaluate_ensemble_fast(data, feature_indices, n_satellites, date_mask):
    """
    Evaluate an ensemble of features using numpy arrays (OPTIMIZED).

    Uses 2D alpha matrix instead of pandas .loc for O(1) lookups.
    """
    rankings = data['rankings']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']
    isins = data['isins']

    # Get training date indices
    train_indices = np.where(date_mask)[0]

    if len(train_indices) == 0:
        return None

    alphas_list = []

    # Process each training date
    for date_idx in train_indices:
        # Get rankings for selected features: (n_isins, n_features)
        feature_rankings = rankings[date_idx, :, :][:, feature_indices]

        # Compute mean score across features
        scores = np.nanmean(feature_rankings, axis=1)
        valid_mask = ~np.isnan(scores)

        if valid_mask.sum() < n_satellites:
            continue

        # Top N selection using argpartition (O(n) instead of O(n log n))
        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]

        if len(valid_scores) < n_satellites:
            continue

        # Get top N indices
        top_k_in_valid = np.argpartition(valid_scores, -n_satellites)[-n_satellites:]
        top_isin_indices = valid_indices[top_k_in_valid]

        # Lookup alphas from numpy matrix (O(1) per lookup)
        alpha_row = alpha_matrix[date_idx]
        valid_row = alpha_valid[date_idx]

        selected_alphas = []
        for isin_idx in top_isin_indices:
            if valid_row[isin_idx]:
                selected_alphas.append(alpha_row[isin_idx])

        if len(selected_alphas) > 0:
            alphas_list.append(np.mean(selected_alphas))

    if len(alphas_list) == 0:
        return None

    alphas_arr = np.array(alphas_list)
    return {
        'avg_alpha': np.mean(alphas_arr),
        'std_alpha': np.std(alphas_arr),
        'hit_rate': np.mean(alphas_arr > 0),
        'n_periods': len(alphas_arr)
    }


# ============================================================
# GREEDY FEATURE SELECTION
# ============================================================

def fast_greedy_search(data, n_satellites, train_mask):
    """
    Fast greedy ensemble search using precomputed data.

    Args:
        data: Dict with all loaded data
        n_satellites: N value
        train_mask: Boolean mask for training dates

    Returns:
        selected_features: List of selected feature indices
        best_perf: Performance of final ensemble
        rankings_modified: Copy of rankings with any inversions applied
    """
    # IMPORTANT: Make a copy of rankings to avoid modifying original
    rankings = data['rankings'].copy()
    feature_alpha = data['feature_alpha']
    feature_hit = data['feature_hit']
    feature_names = data['feature_names']

    n_features = len(feature_names)

    # Create a modified data dict with the copied rankings
    data_copy = data.copy()
    data_copy['rankings'] = rankings

    # Step 1: Pre-filter features using precomputed matrix
    candidates = []

    for feat_idx in range(n_features):
        if feature_alpha is not None:
            perf = evaluate_feature_fast(feature_alpha, feature_hit, feat_idx, n_satellites, train_mask)
        else:
            perf = evaluate_ensemble_fast(data_copy, [feat_idx], n_satellites, train_mask)

        if perf is None:
            continue

        # Check criteria (accept positive or negative predictors)
        if perf['avg_alpha'] >= MIN_ALPHA and perf['hit_rate'] >= MIN_HIT_RATE:
            candidates.append((feat_idx, 'positive', perf['avg_alpha']))
        elif perf['avg_alpha'] <= -MIN_ALPHA and (1 - perf['hit_rate']) >= MIN_HIT_RATE:
            # Invert negative predictor (only in our copy)
            rankings[:, :, feat_idx] = 1.0 - rankings[:, :, feat_idx]
            candidates.append((feat_idx, 'inverted', -perf['avg_alpha']))

    if len(candidates) == 0:
        return [], None, rankings

    # Sort by alpha for faster convergence
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Step 2: Greedy forward selection
    selected = []
    best_perf = None

    for iteration in range(min(MAX_ENSEMBLE_SIZE, len(candidates))):
        current_indices = [f['idx'] for f in selected]
        remaining = [(idx, ptype, alpha) for idx, ptype, alpha in candidates if idx not in current_indices]

        if len(remaining) == 0:
            break

        best_add = None
        best_add_perf = None
        best_improvement = 0

        for feat_idx, pred_type, _ in remaining:
            test_indices = current_indices + [feat_idx]
            perf = evaluate_ensemble_fast(data_copy, test_indices, n_satellites, train_mask)

            if perf is None:
                continue

            improvement = perf['avg_alpha'] if best_perf is None else perf['avg_alpha'] - best_perf['avg_alpha']

            if improvement > best_improvement:
                best_improvement = improvement
                best_add = feat_idx
                best_add_perf = perf

        if best_add is None or best_improvement < MIN_IMPROVEMENT:
            break

        selected.append({
            'idx': best_add,
            'name': feature_names[best_add],
            'improvement': best_improvement
        })
        best_perf = best_add_perf

    return selected, best_perf, rankings


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================

def walk_forward_backtest(data, n_satellites, show_progress=True):
    """
    Run walk-forward backtest for a specific N value.

    OPTIMIZED: Uses numpy arrays instead of pandas .loc for alpha lookups.

    At each test date:
    1. Use only past data for feature selection
    2. Re-optimize features monthly (or at specified frequency)
    3. Select top N ETFs using current ensemble
    4. Measure actual alpha achieved
    """
    if show_progress:
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD BACKTEST (N={n_satellites})")
        print(f"{'='*60}")

    dates = data['dates']
    isins = data['isins']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    # Find start date (after minimum training period)
    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0]

    if len(test_start_idx) == 0:
        if show_progress:
            print(f"ERROR: Not enough data for {MIN_TRAINING_MONTHS} month training period")
        return pd.DataFrame()

    test_start_idx = test_start_idx[0]

    if show_progress:
        print(f"\nTest period: {dates[test_start_idx].date()} to {dates[-1].date()}")
        print(f"Test months: {len(dates) - test_start_idx}")

    results = []
    current_features = None
    current_rankings = None
    months_since_reopt = 0

    # Walk forward through time
    iterator = range(test_start_idx, len(dates))
    if show_progress:
        iterator = tqdm(iterator, desc=f"N={n_satellites}")

    for test_idx in iterator:
        test_date = dates[test_idx]

        # Re-optimize if needed
        if current_features is None or months_since_reopt >= REOPTIMIZATION_FREQUENCY:
            # Create training mask (all dates before test date)
            train_mask = dates < test_date

            # Run feature selection (returns modified rankings with inversions)
            selected, train_perf, modified_rankings = fast_greedy_search(data, n_satellites, train_mask)

            if len(selected) == 0:
                current_features = None
                current_rankings = None
                continue

            current_features = [f['idx'] for f in selected]
            current_rankings = modified_rankings
            months_since_reopt = 0

        months_since_reopt += 1

        # Skip if no features selected
        if current_features is None or len(current_features) == 0:
            continue

        # Select ETFs using current ensemble (with inversions already applied)
        feature_rankings = current_rankings[test_idx][:, current_features]
        scores = np.nanmean(feature_rankings, axis=1)

        # IMPORTANT: Only consider ETFs that have valid forward alpha at test date
        # This ensures we only select ETFs we can actually measure
        valid_mask = ~np.isnan(scores) & alpha_valid[test_idx]

        if valid_mask.sum() < n_satellites:
            continue

        # Top N selection using argpartition
        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]

        if len(valid_scores) < n_satellites:
            continue

        top_k_in_valid = np.argpartition(valid_scores, -n_satellites)[-n_satellites:]
        selected_isin_indices = valid_indices[top_k_in_valid]

        # Lookup actual alpha from numpy matrix (O(1))
        # All selected ETFs are guaranteed to have valid alpha due to filtering above
        alphas = [alpha_matrix[test_idx, isin_idx] for isin_idx in selected_isin_indices]
        selected_isins = [isins[isin_idx] for isin_idx in selected_isin_indices]

        avg_alpha = np.mean(alphas)
        results.append({
            'date': test_date,
            'n_satellites': n_satellites,
            'n_features': len(current_features),
            'avg_alpha': avg_alpha,
            'selected_isins': ','.join(selected_isins)
        })

    return pd.DataFrame(results)


# ============================================================
# PARALLEL BACKTEST WRAPPER
# ============================================================

def _run_single_backtest(args):
    """Worker function for parallel backtest."""
    data, n_satellites = args
    results_df = walk_forward_backtest(data, n_satellites, show_progress=False)
    return n_satellites, results_df


# ============================================================
# ANALYZE RESULTS
# ============================================================

def analyze_results(results_df, n_satellites, show_output=True):
    """Analyze and print backtest results."""
    if len(results_df) == 0:
        if show_output:
            print(f"\nNo results for N={n_satellites}")
        return None

    if show_output:
        print(f"\n{'='*60}")
        print(f"RESULTS (N={n_satellites})")
        print(f"{'='*60}")

    avg_alpha = results_df['avg_alpha'].mean()
    std_alpha = results_df['avg_alpha'].std()
    hit_rate = (results_df['avg_alpha'] > 0).mean()
    sharpe = avg_alpha / std_alpha if std_alpha > 0 else 0

    # Cumulative return
    results_df = results_df.copy()
    results_df['cumulative'] = (1 + results_df['avg_alpha']).cumprod() - 1
    total_return = results_df['cumulative'].iloc[-1]

    if show_output:
        print(f"\nPerformance:")
        print(f"  Periods: {len(results_df)}")
        print(f"  Monthly Alpha: {avg_alpha*100:.2f}% +/- {std_alpha*100:.2f}%")
        print(f"  Annualized Alpha: {avg_alpha*12*100:.1f}%")
        print(f"  Hit Rate: {hit_rate:.2%}")
        print(f"  Sharpe: {sharpe:.3f}")
        print(f"  Total Return: {total_return*100:.1f}%")

        # Year-by-year
        results_df['year'] = results_df['date'].dt.year
        yearly = results_df.groupby('year').agg({
            'avg_alpha': ['mean', 'count', lambda x: (x > 0).mean()]
        })
        yearly.columns = ['avg', 'n', 'hit']

        print("\nYear-by-Year:")
        for year, row in yearly.iterrows():
            print(f"  {year}: {row['avg']*100:+.2f}% ({int(row['n'])} months, {row['hit']:.0%} hit)")

    return {
        'n_satellites': n_satellites,
        'n_periods': len(results_df),
        'avg_alpha': avg_alpha,
        'std_alpha': std_alpha,
        'annual_alpha': avg_alpha * 12,
        'hit_rate': hit_rate,
        'sharpe': sharpe,
        'total_return': total_return
    }


# ============================================================
# MAIN
# ============================================================

def main(holding_months=DEFAULT_HOLDING_MONTHS, n_satellites=None):
    """Run walk-forward backtest."""
    print("=" * 60)
    print(f"WALK-FORWARD BACKTEST - {holding_months} MONTH HORIZON")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_data(holding_months)

    # Determine N values to test
    if n_satellites is not None:
        n_values = [n_satellites]
    else:
        n_values = N_SATELLITES_TO_TEST

    print(f"\nTesting N values: {n_values}")
    print(f"Reoptimization frequency: Every {REOPTIMIZATION_FREQUENCY} month(s)")
    print(f"Minimum training period: {MIN_TRAINING_MONTHS} months")
    print(f"Parallel processing: {PARALLEL_N_VALUES and len(n_values) > 1}")

    # Run backtest for each N
    all_stats = []
    all_results = {}

    if PARALLEL_N_VALUES and len(n_values) > 1:
        # Parallel execution across N values
        print(f"\nRunning {len(n_values)} backtests in parallel with {N_CORES} cores...")

        args_list = [(data, n) for n in n_values]

        with Pool(N_CORES) as pool:
            results = list(tqdm(
                pool.imap(_run_single_backtest, args_list),
                total=len(args_list),
                desc="Backtests"
            ))

        for n, results_df in results:
            all_results[n] = results_df
    else:
        # Sequential execution
        for n in n_values:
            results_df = walk_forward_backtest(data, n)
            all_results[n] = results_df

    # Analyze and save results
    for n in n_values:
        results_df = all_results[n]

        if len(results_df) == 0:
            continue

        stats = analyze_results(results_df, n)
        if stats is not None:
            all_stats.append(stats)

        # Save individual results
        output_file = OUTPUT_DIR / f'backtest_N{n}_{holding_months}month.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n[SAVED] {output_file}")

    # Summary
    if len(all_stats) > 0:
        summary_df = pd.DataFrame(all_stats)
        summary_file = OUTPUT_DIR / f'summary_{holding_months}month.csv'
        summary_df.to_csv(summary_file, index=False)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        for _, row in summary_df.iterrows():
            print(f"\nN={int(row['n_satellites'])}:")
            print(f"  Monthly Alpha: {row['avg_alpha']*100:.2f}%")
            print(f"  Annual Alpha: {row['annual_alpha']*100:.1f}%")
            print(f"  Hit Rate: {row['hit_rate']:.2%}")
            print(f"  Sharpe: {row['sharpe']:.3f}")

        # Best configuration
        best = summary_df.loc[summary_df['sharpe'].idxmax()]
        print("\n" + "=" * 60)
        print("BEST CONFIGURATION (by Sharpe)")
        print("=" * 60)
        print(f"\nN = {int(best['n_satellites'])}")
        print(f"  Annual Alpha: {best['annual_alpha']*100:.1f}%")
        print(f"  Hit Rate: {best['hit_rate']:.2%}")
        print(f"  Sharpe: {best['sharpe']:.3f}")

        # Portfolio impact estimate (60/40 core/satellite)
        baseline = 0.08  # Assume 8% baseline return
        satellite_ret = baseline + best['annual_alpha']
        portfolio_ret = 0.6 * baseline + 0.4 * satellite_ret
        print(f"\n60/40 Core/Satellite Portfolio:")
        print(f"  Expected Annual Return: {portfolio_ret*100:.1f}%")
        print(f"  vs 100% Core: +{(portfolio_ret - baseline)*100:.1f}%")

        print(f"\n[SAVED] {summary_file}")


if __name__ == '__main__':
    holding_months = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_HOLDING_MONTHS
    n_satellites = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(holding_months, n_satellites)
