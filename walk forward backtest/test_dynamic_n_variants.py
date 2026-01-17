"""
Test Dynamic N Variants with Risk Reduction (PARALLEL VERSION)
===============================================================

The original Dynamic N has high drawdown because it concentrates to N=1.
This script tests safer variants in parallel.

Usage:
    python test_dynamic_n_variants.py
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

MIN_TRAINING_MONTHS = 36
MIN_ALPHA = 0.001
MIN_HIT_RATE = 0.55
MAX_ENSEMBLE_SIZE = 20
MIN_IMPROVEMENT = 0.0001

N_WORKERS = max(1, cpu_count() - 1)

DATA_DIR = Path(__file__).parent / 'data'
OUTPUT_DIR = DATA_DIR / 'dynamic_n_variants'


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load all precomputed data."""
    alpha_file = DATA_DIR / f'forward_alpha_{HOLDING_MONTHS}month.parquet'
    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])

    rankings_file = DATA_DIR / f'rankings_matrix_{HOLDING_MONTHS}month.npz'
    npz_data = np.load(rankings_file, allow_pickle=True)
    rankings = npz_data['rankings'].copy()
    dates = pd.to_datetime(npz_data['dates'])
    isins = npz_data['isins']
    feature_names = list(npz_data['features'])

    feature_alpha_file = DATA_DIR / f'feature_alpha_{HOLDING_MONTHS}month.npz'
    fa_data = np.load(feature_alpha_file, allow_pickle=True)
    feature_alpha = fa_data['feature_alpha']
    feature_hit = fa_data['feature_hit']

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


# ============================================================
# DYNAMIC N MAPPING VARIANTS
# ============================================================

def confidence_to_n_original(confidence):
    """Original: N in [1,2,3,4,5]"""
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


def confidence_to_n_floor3(confidence):
    """Conservative: N in [3,4,5] - never go below 3"""
    if confidence > 0.6:
        return 3
    elif confidence > 0.35:
        return 4
    else:
        return 5


def confidence_to_n_floor2(confidence):
    """Moderate: N in [2,3,4,5] - never go below 2"""
    if confidence > 0.7:
        return 2
    elif confidence > 0.5:
        return 3
    elif confidence > 0.3:
        return 4
    else:
        return 5


def confidence_to_n_smooth(confidence):
    """Smooth: continuous mapping, rounded"""
    n_float = 5 - 3 * confidence
    return max(2, min(5, round(n_float)))


def confidence_to_n_very_conservative(confidence):
    """Very conservative: N in [4,5] - only slight concentration"""
    if confidence > 0.6:
        return 4
    else:
        return 5


# ============================================================
# UNIFIED BACKTEST FUNCTION
# ============================================================

def run_backtest(data, variant_type, confidence_func=None, use_ic=False):
    """
    Unified backtest function for all variants.

    variant_type: 'baseline', 'ic_only', or 'dynamic'
    """
    dates = data['dates']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']

    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0][0]

    results = []
    n_distribution = {n: 0 for n in [1, 2, 3, 4, 5]}

    for test_idx in range(test_start_idx, len(dates)):
        test_date = dates[test_idx]
        train_mask = dates < test_date

        # Use N=3 for feature selection if dynamic, else N=5
        base_n = 3 if variant_type == 'dynamic' else N_SATELLITES

        selected, train_perf, modified_rankings, _ = greedy_feature_selection(
            data, base_n, train_mask
        )

        if len(selected) == 0:
            continue

        # Determine N
        if variant_type == 'dynamic' and confidence_func is not None:
            confidence = compute_signal_confidence(
                data, selected, modified_rankings, test_idx, alpha_valid
            )
            n_satellites = confidence_func(confidence)
            n_distribution[n_satellites] += 1
        else:
            n_satellites = N_SATELLITES
            confidence = None

        # Apply IC weighting if enabled
        if use_ic:
            for feat in selected:
                feat['weight'] = max(feat['ic'], 0.0001)
            total_weight = sum(f['weight'] for f in selected)
            for feat in selected:
                feat['normalized_weight'] = feat['weight'] / total_weight
        else:
            for feat in selected:
                feat['normalized_weight'] = 1.0 / len(selected)

        # Compute scores
        feature_indices = [f['idx'] for f in selected]
        feature_rankings = modified_rankings[test_idx][:, feature_indices]

        if use_ic:
            weights = np.array([f['normalized_weight'] for f in selected])
            scores = np.zeros(feature_rankings.shape[0])
            for i, weight in enumerate(weights):
                valid = ~np.isnan(feature_rankings[:, i])
                scores[valid] += weight * feature_rankings[valid, i]
        else:
            scores = np.nanmean(feature_rankings, axis=1)

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

    return pd.DataFrame(results), n_distribution


# ============================================================
# WORKER FUNCTION FOR PARALLEL EXECUTION
# ============================================================

def _run_variant_worker(args):
    """Worker function for parallel execution."""
    data, name, variant_type, conf_func_name, use_ic = args

    # Map function name back to function
    conf_func_map = {
        'original': confidence_to_n_original,
        'floor2': confidence_to_n_floor2,
        'floor3': confidence_to_n_floor3,
        'smooth': confidence_to_n_smooth,
        'very_conservative': confidence_to_n_very_conservative,
        None: None
    }
    conf_func = conf_func_map.get(conf_func_name)

    df, n_dist = run_backtest(data, variant_type, conf_func, use_ic)
    return name, df, n_dist


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
    avg_n = results_df['n_satellites'].mean() if 'n_satellites' in results_df.columns else N_SATELLITES

    return {
        'name': name,
        'n_periods': len(results_df),
        'avg_alpha': avg_alpha,
        'std_alpha': std_alpha,
        'annual_alpha': avg_alpha * 12,
        'hit_rate': hit_rate,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'total_return': total_return,
        'avg_n': avg_n
    }


def print_comparison(all_results):
    """Print comparison table."""
    print("\n" + "=" * 100)
    print("DYNAMIC N VARIANTS COMPARISON")
    print("=" * 100)

    print(f"\n{'Strategy':<25} {'Ann.Alpha':>10} {'Hit Rate':>10} {'Sharpe':>10} {'MaxDD':>10} {'Avg N':>8}")
    print("-" * 100)

    baseline = None
    for r in all_results:
        if r is None:
            continue
        if r['name'] == 'Baseline':
            baseline = r

        print(f"{r['name']:<25} {r['annual_alpha']*100:>9.2f}% {r['hit_rate']:>9.1%} "
              f"{r['sharpe']:>10.3f} {r['max_drawdown']*100:>9.1f}% {r['avg_n']:>8.1f}")

    if baseline:
        print("\n" + "-" * 100)
        print("IMPROVEMENT VS BASELINE:")
        print("-" * 100)

        for r in all_results:
            if r is None or r['name'] == 'Baseline':
                continue

            alpha_diff = (r['annual_alpha'] - baseline['annual_alpha']) * 100
            hit_diff = (r['hit_rate'] - baseline['hit_rate']) * 100
            sharpe_diff = r['sharpe'] - baseline['sharpe']
            dd_diff = (r['max_drawdown'] - baseline['max_drawdown']) * 100

            print(f"{r['name']:<25} {alpha_diff:>+9.2f}% {hit_diff:>+9.1f}pp "
                  f"{sharpe_diff:>+10.3f} {dd_diff:>+9.1f}%")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("DYNAMIC N VARIANTS - RISK REDUCTION TEST (PARALLEL)")
    print("=" * 100)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading data...")
    data = load_data()
    print(f"[OK] Data loaded")

    # Define variants: (name, variant_type, conf_func_name, use_ic)
    variants = [
        ('Baseline', 'baseline', None, False),
        ('IC Only (N=5)', 'ic_only', None, True),
        ('IC + DynN Original', 'dynamic', 'original', True),
        ('IC + DynN Floor-2', 'dynamic', 'floor2', True),
        ('IC + DynN Floor-3', 'dynamic', 'floor3', True),
        ('IC + DynN VeryConserv', 'dynamic', 'very_conservative', True),
        ('IC + DynN Smooth', 'dynamic', 'smooth', True),
    ]

    print(f"\nRunning {len(variants)} variants in PARALLEL with {N_WORKERS} workers...")

    # Prepare args
    args_list = [(data, name, vtype, cfunc, use_ic) for name, vtype, cfunc, use_ic in variants]

    all_results = []
    all_dfs = {}
    all_n_dists = {}

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_run_variant_worker, args): args[1] for args in args_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Variants"):
            name = futures[future]
            try:
                result_name, df, n_dist = future.result()
                all_dfs[result_name] = df
                all_n_dists[result_name] = n_dist
                print(f"  [DONE] {result_name}")
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")

    # Analyze in order
    for name, _, _, _ in variants:
        if name in all_dfs:
            all_results.append(analyze_results(all_dfs[name], name))

            # Print N distribution for dynamic variants
            if name in all_n_dists and any(all_n_dists[name].values()):
                n_dist = all_n_dists[name]
                total = sum(n_dist.values())
                if total > 0:
                    print(f"\n  N distribution for {name}:")
                    for n in sorted(n_dist.keys()):
                        if n_dist[n] > 0:
                            pct = n_dist[n] / total * 100
                            print(f"    N={n}: {n_dist[n]:3d} ({pct:5.1f}%)")

    # Print comparison
    print_comparison(all_results)

    # Save results
    summary_df = pd.DataFrame([r for r in all_results if r is not None])
    summary_df.to_csv(OUTPUT_DIR / 'dynamic_n_variants.csv', index=False)

    for name, df in all_dfs.items():
        safe_name = name.lower().replace(" ", "_").replace("+", "_").replace("-", "_").replace("(", "").replace(")", "").replace("=", "")
        df.to_csv(OUTPUT_DIR / f'{safe_name}_results.csv', index=False)

    print(f"\n[SAVED] Results to {OUTPUT_DIR}")

    return all_results, all_dfs


if __name__ == '__main__':
    main()
