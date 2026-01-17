"""
Test Script: Strategy Improvement Combinations
==============================================

Based on initial tests, we found these improvements worked:
- IC-Weighted: +4.88% annual alpha, +9.7pp hit rate
- TimeSeries-Features: Same as IC-Weighted
- Stability-Weighted: +4.62% annual alpha, +8.4pp hit rate
- Dynamic-N: +12.26% alpha but 2x drawdown

This script tests COMBINATIONS of these improvements to see if they stack.

Combinations to test:
1. IC + Stability (weight by IC AND stability)
2. IC + TimeSeries (weight by IC AND signal trend)
3. IC + Stability + TimeSeries (all three)
4. IC + Dynamic-N (best weighting + adaptive concentration)
5. IC + Stability + Dynamic-N (full combo without TimeSeries)
6. All Four Combined

Usage:
    python test_strategy_combinations.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import warnings

warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================
# CONFIGURATION
# ============================================================

HOLDING_MONTHS = 1
N_SATELLITES = 5
N_RANGE = [1, 2, 3, 4, 5]

MIN_TRAINING_MONTHS = 36
MIN_ALPHA = 0.001
MIN_HIT_RATE = 0.55
MAX_ENSEMBLE_SIZE = 20
MIN_IMPROVEMENT = 0.0001

# Parallel processing
N_WORKERS = max(1, cpu_count() - 1)
PARALLEL_STRATEGIES = True  # Run different strategies in parallel

DATA_DIR = Path(__file__).parent / 'data'
OUTPUT_DIR = DATA_DIR / 'combination_tests'


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load all precomputed data."""
    print(f"\n{'='*60}")
    print(f"LOADING DATA")
    print(f"{'='*60}")

    alpha_file = DATA_DIR / f'forward_alpha_{HOLDING_MONTHS}month.parquet'
    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])
    print(f"[OK] Forward alpha: {len(alpha_df):,} observations")

    rankings_file = DATA_DIR / f'rankings_matrix_{HOLDING_MONTHS}month.npz'
    npz_data = np.load(rankings_file, allow_pickle=True)
    rankings = npz_data['rankings'].copy()
    dates = pd.to_datetime(npz_data['dates'])
    isins = npz_data['isins']
    feature_names = list(npz_data['features'])
    print(f"[OK] Rankings matrix: {rankings.shape}")

    feature_alpha_file = DATA_DIR / f'feature_alpha_{HOLDING_MONTHS}month.npz'
    fa_data = np.load(feature_alpha_file, allow_pickle=True)
    feature_alpha = fa_data['feature_alpha']
    feature_hit = fa_data['feature_hit']
    print(f"[OK] Feature-alpha matrix: {feature_alpha.shape}")

    n_dates = len(dates)
    n_isins = len(isins)
    isin_to_idx = {isin: idx for idx, isin in enumerate(isins)}
    date_to_idx = {date: idx for idx, date in enumerate(dates)}

    alpha_matrix = np.full((n_dates, n_isins), np.nan, dtype=np.float32)
    for date, group in alpha_df.groupby('date'):
        if date not in date_to_idx:
            continue
        date_idx = date_to_idx[date]
        for isin, alpha in zip(group['isin'].values, group['forward_alpha'].values):
            if isin in isin_to_idx:
                alpha_matrix[date_idx, isin_to_idx[isin]] = alpha

    alpha_valid = ~np.isnan(alpha_matrix)
    print(f"[OK] Alpha matrix: {alpha_matrix.shape}")

    return {
        'alpha_matrix': alpha_matrix,
        'alpha_valid': alpha_valid,
        'rankings': rankings,
        'dates': dates,
        'isins': isins,
        'feature_names': feature_names,
        'feature_alpha': feature_alpha,
        'feature_hit': feature_hit,
        'isin_to_idx': isin_to_idx,
        'date_to_idx': date_to_idx,
    }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def evaluate_ensemble(data, feature_indices, n_satellites, date_mask):
    """Evaluate an ensemble of features."""
    rankings = data['rankings']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    train_indices = np.where(date_mask)[0]
    if len(train_indices) == 0:
        return None

    alphas_list = []

    for date_idx in train_indices:
        feature_rankings = rankings[date_idx, :, :][:, feature_indices]
        scores = np.nanmean(feature_rankings, axis=1)
        valid_mask = ~np.isnan(scores) & alpha_valid[date_idx]

        if valid_mask.sum() < n_satellites:
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]

        top_k = np.argpartition(valid_scores, -n_satellites)[-n_satellites:]
        top_isin_indices = valid_indices[top_k]

        selected_alphas = [alpha_matrix[date_idx, idx] for idx in top_isin_indices
                          if alpha_valid[date_idx, idx]]

        if len(selected_alphas) > 0:
            alphas_list.append(np.mean(selected_alphas))

    if len(alphas_list) == 0:
        return None

    alphas_arr = np.array(alphas_list)
    return {
        'avg_alpha': np.mean(alphas_arr),
        'std_alpha': np.std(alphas_arr),
        'hit_rate': np.mean(alphas_arr > 0),
        'n_periods': len(alphas_arr),
        'alphas': alphas_arr
    }


def greedy_feature_selection(data, n_satellites, train_mask):
    """Greedy feature selection with IC tracking."""
    rankings = data['rankings'].copy()
    feature_alpha = data['feature_alpha']
    feature_hit = data['feature_hit']
    feature_names = data['feature_names']
    n_features = len(feature_names)

    candidates = []
    candidate_ics = {}

    for feat_idx in range(n_features):
        alphas = feature_alpha[train_mask, feat_idx, n_satellites - 1]
        hits = feature_hit[train_mask, feat_idx, n_satellites - 1]
        valid = ~np.isnan(alphas)

        if valid.sum() == 0:
            continue

        avg_alpha = np.mean(alphas[valid])
        hit_rate = np.mean(hits[valid])
        ic = avg_alpha

        if avg_alpha >= MIN_ALPHA and hit_rate >= MIN_HIT_RATE:
            candidates.append((feat_idx, 'positive', avg_alpha))
            candidate_ics[feat_idx] = ic
        elif avg_alpha <= -MIN_ALPHA and (1 - hit_rate) >= MIN_HIT_RATE:
            rankings[:, :, feat_idx] = 1.0 - rankings[:, :, feat_idx]
            candidates.append((feat_idx, 'inverted', -avg_alpha))
            candidate_ics[feat_idx] = -ic

    if len(candidates) == 0:
        return [], None, rankings, {}

    candidates.sort(key=lambda x: x[2], reverse=True)

    selected = []
    best_perf = None
    data_copy = data.copy()
    data_copy['rankings'] = rankings

    for iteration in range(min(MAX_ENSEMBLE_SIZE, len(candidates))):
        current_indices = [f['idx'] for f in selected]
        remaining = [(idx, ptype, alpha) for idx, ptype, alpha in candidates
                    if idx not in current_indices]

        if len(remaining) == 0:
            break

        best_add = None
        best_add_perf = None
        best_improvement = 0

        for feat_idx, pred_type, _ in remaining:
            test_indices = current_indices + [feat_idx]
            perf = evaluate_ensemble(data_copy, test_indices, n_satellites, train_mask)

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
            'improvement': best_improvement,
            'ic': candidate_ics.get(best_add, 0)
        })
        best_perf = best_add_perf

    return selected, best_perf, rankings, candidate_ics


def compute_feature_stability(data, feature_idx, n_satellites, date_idx, lookback_months=12):
    """Compute feature stability over recent months."""
    feature_alpha = data['feature_alpha']

    start_idx = max(0, date_idx - lookback_months)
    alphas = feature_alpha[start_idx:date_idx, feature_idx, n_satellites - 1]
    valid = ~np.isnan(alphas)

    if valid.sum() < 6:
        return 0.5

    alphas = alphas[valid]
    hit_rate = np.mean(alphas > 0)

    if np.mean(alphas) > 0:
        cv = np.std(alphas) / np.mean(alphas)
        consistency = max(0, 1 - cv)
    else:
        consistency = 0

    recent_hit = np.mean(alphas[-3:] > 0) if len(alphas) >= 3 else 0.5
    stability = 0.4 * hit_rate + 0.3 * consistency + 0.3 * recent_hit
    return np.clip(stability, 0, 1)


def compute_signal_trend(rankings, feature_idx, date_idx, lookback=5):
    """Compute trend/momentum of the signal itself."""
    if date_idx < lookback + 1:
        return 0, 0

    hist_ranks = rankings[date_idx-lookback:date_idx, :, feature_idx]
    mean_ranks = np.nanmean(hist_ranks, axis=1)

    if len(mean_ranks) < 2:
        return 0, 0

    changes = np.diff(mean_ranks)
    trend = np.mean(changes) if len(changes) > 0 else 0
    accel = changes[-1] - changes[0] if len(changes) >= 2 else 0

    return trend, accel


def compute_signal_confidence(data, selected_features, modified_rankings, test_idx, alpha_valid):
    """Compute confidence for dynamic N selection."""
    feature_indices = [f['idx'] for f in selected_features]
    feature_rankings = modified_rankings[test_idx][:, feature_indices]

    scores = np.nanmean(feature_rankings, axis=1)
    valid_mask = ~np.isnan(scores) & alpha_valid[test_idx]
    valid_scores = scores[valid_mask]

    if len(valid_scores) < 5:
        return 0.5

    top_scores = np.sort(valid_scores)[-10:]
    score_std = np.std(valid_scores)
    top_std = np.std(top_scores)
    dispersion_ratio = score_std / (top_std + 0.001)

    if len(feature_indices) > 1:
        top_5_per_feature = []
        for i in range(len(feature_indices)):
            feat_ranks = feature_rankings[:, i]
            valid = ~np.isnan(feat_ranks) & alpha_valid[test_idx]
            if valid.sum() >= 5:
                top_5 = set(np.argsort(feat_ranks[valid])[-5:])
                top_5_per_feature.append(top_5)

        if len(top_5_per_feature) >= 2:
            intersection = top_5_per_feature[0]
            for s in top_5_per_feature[1:]:
                intersection = intersection.intersection(s)
            agreement = len(intersection) / 5
        else:
            agreement = 0.5
    else:
        agreement = 0.5

    confidence = 0.5 * min(dispersion_ratio, 2) / 2 + 0.5 * agreement
    return np.clip(confidence, 0, 1)


def confidence_to_n(confidence):
    """Map confidence to N value."""
    if confidence > 0.7:
        return 1
    elif confidence > 0.55:
        return 2
    elif confidence > 0.4:
        return 3
    elif confidence > 0.25:
        return 4
    else:
        return 5


# ============================================================
# BASELINE
# ============================================================

def run_baseline_backtest(data, n_satellites=N_SATELLITES, show_progress=True):
    """Run baseline walk-forward backtest."""
    dates = data['dates']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0][0]

    results = []

    iterator = range(test_start_idx, len(dates))
    if show_progress and not PARALLEL_STRATEGIES:
        iterator = tqdm(iterator, desc="Baseline")

    for test_idx in iterator:
        test_date = dates[test_idx]
        train_mask = dates < test_date

        selected, train_perf, modified_rankings, _ = greedy_feature_selection(
            data, n_satellites, train_mask
        )

        if len(selected) == 0:
            continue

        feature_indices = [f['idx'] for f in selected]
        feature_rankings = modified_rankings[test_idx][:, feature_indices]
        scores = np.nanmean(feature_rankings, axis=1)
        valid_mask = ~np.isnan(scores) & alpha_valid[test_idx]

        if valid_mask.sum() < n_satellites:
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]
        top_k = np.argpartition(valid_scores, -n_satellites)[-n_satellites:]
        selected_indices = valid_indices[top_k]

        alphas = [alpha_matrix[test_idx, idx] for idx in selected_indices]

        results.append({
            'date': test_date,
            'avg_alpha': np.mean(alphas),
            'n_features': len(selected),
            'n_satellites': n_satellites
        })

    return pd.DataFrame(results)


# ============================================================
# COMBINED STRATEGIES
# ============================================================

def run_combined_backtest(data, use_ic=False, use_stability=False, use_timeseries=False,
                          use_dynamic_n=False, name="Combined", show_progress=True):
    """
    Run backtest with any combination of improvements.

    Args:
        use_ic: Weight features by Information Coefficient
        use_stability: Weight features by stability over time
        use_timeseries: Boost features with positive signal trend
        use_dynamic_n: Adapt N based on signal confidence
        show_progress: Show tqdm progress bar (disable for parallel execution)
    """
    dates = data['dates']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0][0]

    results = []
    n_distribution = {n: 0 for n in N_RANGE} if use_dynamic_n else None

    iterator = range(test_start_idx, len(dates))
    if show_progress and not PARALLEL_STRATEGIES:
        iterator = tqdm(iterator, desc=name)

    for test_idx in iterator:
        test_date = dates[test_idx]
        train_mask = dates < test_date

        # Base N for feature selection (use middle of range if dynamic)
        base_n = 3 if use_dynamic_n else N_SATELLITES

        selected, train_perf, modified_rankings, _ = greedy_feature_selection(
            data, base_n, train_mask
        )

        if len(selected) == 0:
            continue

        # Compute weights for each feature
        for feat in selected:
            weight = 1.0

            # IC component
            if use_ic:
                ic = max(feat['ic'], 0.0001)
                weight *= ic

            # Stability component
            if use_stability:
                stability = compute_feature_stability(
                    data, feat['idx'], base_n, test_idx
                )
                feat['stability'] = stability
                weight *= (0.5 + 0.5 * stability)  # Scale stability to 0.5-1.0

            # TimeSeries component
            if use_timeseries:
                trend, accel = compute_signal_trend(
                    modified_rankings, feat['idx'], test_idx
                )
                feat['trend'] = trend
                # Boost for positive trend
                trend_factor = 1.0 + 0.2 * max(0, trend)  # Up to 20% boost
                weight *= trend_factor

            feat['combined_weight'] = weight

        # Normalize weights
        total_weight = sum(f['combined_weight'] for f in selected)
        for feat in selected:
            feat['normalized_weight'] = feat['combined_weight'] / total_weight

        # Determine N
        if use_dynamic_n:
            confidence = compute_signal_confidence(
                data, selected, modified_rankings, test_idx, alpha_valid
            )
            n_satellites = confidence_to_n(confidence)
            n_distribution[n_satellites] += 1
        else:
            n_satellites = N_SATELLITES

        # Weighted scoring
        feature_indices = [f['idx'] for f in selected]
        feature_rankings = modified_rankings[test_idx][:, feature_indices]
        weights = np.array([f['normalized_weight'] for f in selected])

        scores = np.zeros(feature_rankings.shape[0])
        for i, weight in enumerate(weights):
            valid = ~np.isnan(feature_rankings[:, i])
            scores[valid] += weight * feature_rankings[valid, i]

        valid_mask = ~np.isnan(np.nanmean(feature_rankings, axis=1)) & alpha_valid[test_idx]

        if valid_mask.sum() < n_satellites:
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]
        top_k = np.argpartition(valid_scores, -n_satellites)[-n_satellites:]
        selected_indices = valid_indices[top_k]

        alphas = [alpha_matrix[test_idx, idx] for idx in selected_indices]

        results.append({
            'date': test_date,
            'avg_alpha': np.mean(alphas),
            'n_features': len(selected),
            'n_satellites': n_satellites
        })

    return pd.DataFrame(results)


# ============================================================
# ANALYSIS
# ============================================================

def analyze_results(results_df, name):
    """Compute performance metrics."""
    if len(results_df) == 0:
        return None

    avg_alpha = results_df['avg_alpha'].mean()
    std_alpha = results_df['avg_alpha'].std()
    hit_rate = (results_df['avg_alpha'] > 0).mean()
    sharpe = avg_alpha / std_alpha if std_alpha > 0 else 0

    cumulative = (1 + results_df['avg_alpha']).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    total_return = cumulative.iloc[-1] - 1

    return {
        'name': name,
        'n_periods': len(results_df),
        'avg_alpha': avg_alpha,
        'std_alpha': std_alpha,
        'annual_alpha': avg_alpha * 12,
        'hit_rate': hit_rate,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'total_return': total_return
    }


def print_comparison(all_results):
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("STRATEGY COMPARISON")
    print("=" * 90)

    print(f"\n{'Strategy':<30} {'Ann.Alpha':>10} {'Hit Rate':>10} {'Sharpe':>10} {'MaxDD':>10} {'Total':>10}")
    print("-" * 90)

    baseline = None
    for r in all_results:
        if r is None:
            continue
        if r['name'] == 'Baseline':
            baseline = r

        print(f"{r['name']:<30} {r['annual_alpha']*100:>9.2f}% {r['hit_rate']:>9.1%} "
              f"{r['sharpe']:>10.3f} {r['max_drawdown']*100:>9.1f}% {r['total_return']*100:>9.1f}%")

    if baseline:
        print("\n" + "-" * 90)
        print("IMPROVEMENT VS BASELINE:")
        print("-" * 90)

        for r in all_results:
            if r is None or r['name'] == 'Baseline':
                continue

            alpha_diff = (r['annual_alpha'] - baseline['annual_alpha']) * 100
            hit_diff = (r['hit_rate'] - baseline['hit_rate']) * 100
            sharpe_diff = r['sharpe'] - baseline['sharpe']
            dd_diff = (r['max_drawdown'] - baseline['max_drawdown']) * 100

            print(f"{r['name']:<30} {alpha_diff:>+9.2f}% {hit_diff:>+9.1f}pp "
                  f"{sharpe_diff:>+10.3f} {dd_diff:>+9.1f}%")


# ============================================================
# PARALLEL WORKER FUNCTION
# ============================================================

def _run_strategy_worker(args):
    """
    Worker function for parallel strategy execution.

    Args:
        args: tuple of (data, strategy_config)
        strategy_config: dict with 'name', 'use_ic', 'use_stability', 'use_timeseries', 'use_dynamic_n'

    Returns:
        tuple of (name, results_df)
    """
    data, config = args
    name = config['name']

    if name == 'Baseline':
        df = run_baseline_backtest(data, show_progress=False)
    else:
        df = run_combined_backtest(
            data,
            use_ic=config.get('use_ic', False),
            use_stability=config.get('use_stability', False),
            use_timeseries=config.get('use_timeseries', False),
            use_dynamic_n=config.get('use_dynamic_n', False),
            name=name,
            show_progress=False
        )

    return name, df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 90)
    print("STRATEGY COMBINATION TESTS")
    print("=" * 90)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    # Define all strategy configurations
    # Generate ALL 16 combinations (2^4 = 16)
    # Options: IC weighting, Stability weighting, TimeSeries features, Dynamic N
    strategy_configs = []

    for i in range(16):  # 0 to 15 = all binary combinations
        use_ic = bool(i & 1)
        use_stability = bool(i & 2)
        use_timeseries = bool(i & 4)
        use_dynamic_n = bool(i & 8)

        # Build name
        parts = []
        if use_ic:
            parts.append('IC')
        if use_stability:
            parts.append('Stab')
        if use_timeseries:
            parts.append('TS')
        if use_dynamic_n:
            parts.append('DynN')

        name = '+'.join(parts) if parts else 'Baseline'

        strategy_configs.append({
            'name': name,
            'use_ic': use_ic,
            'use_stability': use_stability,
            'use_timeseries': use_timeseries,
            'use_dynamic_n': use_dynamic_n,
        })

    print(f"\nTesting {len(strategy_configs)} strategy combinations:")

    all_results = []
    all_dfs = {}

    if PARALLEL_STRATEGIES:
        print(f"\nRunning {len(strategy_configs)} strategies in PARALLEL with {N_WORKERS} workers...")
        print("=" * 60)

        # Prepare arguments for parallel execution
        args_list = [(data, config) for config in strategy_configs]

        # Run in parallel
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(_run_strategy_worker, args): args[1]['name']
                      for args in args_list}

            # Collect results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Strategies"):
                name = futures[future]
                try:
                    result_name, df = future.result()
                    all_dfs[result_name] = df
                    print(f"  [DONE] {result_name}")
                except Exception as e:
                    print(f"  [ERROR] {name}: {e}")

        # Analyze results in order
        for config in strategy_configs:
            name = config['name']
            if name in all_dfs:
                all_results.append(analyze_results(all_dfs[name], name))
    else:
        # Sequential execution (original behavior)
        for i, config in enumerate(strategy_configs, 1):
            name = config['name']
            print(f"\n{'='*60}")
            print(f"{i}. {name}")
            print("=" * 60)

            if name == 'Baseline':
                df = run_baseline_backtest(data)
            else:
                df = run_combined_backtest(
                    data,
                    use_ic=config.get('use_ic', False),
                    use_stability=config.get('use_stability', False),
                    use_timeseries=config.get('use_timeseries', False),
                    use_dynamic_n=config.get('use_dynamic_n', False),
                    name=name
                )

            all_dfs[name] = df
            all_results.append(analyze_results(df, name))

    # Print comparison
    print_comparison(all_results)

    # Save results
    summary_df = pd.DataFrame([r for r in all_results if r is not None])
    summary_df.to_csv(OUTPUT_DIR / 'combination_comparison.csv', index=False)

    for name, df in all_dfs.items():
        safe_name = name.lower().replace("+", "_").replace("-", "_")
        df.to_csv(OUTPUT_DIR / f'{safe_name}_results.csv', index=False)

    print(f"\n[SAVED] Results to {OUTPUT_DIR}")

    return all_results, all_dfs


if __name__ == '__main__':
    main()
