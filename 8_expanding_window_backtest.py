"""
Step 8: Expanding Window Walk-Forward Backtest (ULTRA-OPTIMIZED)
==================================================================

This script implements realistic backtesting with maximum optimization:
- Sequential but heavily vectorized (avoids multiprocessing overhead)
- Batch operations wherever possible
- Minimal memory copying
- Progress tracking

Expected runtime: 10-15 minutes

Usage:
    python 8_expanding_window_backtest.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

# Test configurations
N_SATELLITES_TO_TEST = [1,2,5,6]
HOLDING_MONTHS = 1

# Training parameters
MIN_TRAINING_MONTHS = 36
REOPTIMIZATION_FREQUENCY = 1  # Monthly

# Feature selection parameters
MIN_ALPHA = 0.001
MIN_HIT_RATE = 0.55
MAX_ENSEMBLE_SIZE = 20
MIN_IMPROVEMENT = 0.0001

# Core benchmark
CORE_ISIN = 'IE00B4L5Y983'

# Output
OUTPUT_DIR = Path('data/expanding_window_backtest')
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load pre-computed rankings and forward alpha."""
    print("\n" + "="*60)
    print("LOADING PRE-COMPUTED DATA")
    print("="*60)

    # Load forward alpha
    alpha_file = Path('data/feature_analysis') / f'forward_alpha_{HOLDING_MONTHS}month.parquet'
    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])
    alpha_df = alpha_df.set_index(['date', 'isin'])

    print(f"\n[✓] Forward alpha loaded ({len(alpha_df):,} observations)")

    # Load rankings matrix
    rankings_file = Path('data/feature_analysis') / f'rankings_matrix_{HOLDING_MONTHS}month.npz'
    npz_data = np.load(rankings_file, allow_pickle=True)
    rankings = npz_data['rankings'].copy()
    dates = pd.to_datetime(npz_data['dates'])
    isins = npz_data['isins']
    feature_names = npz_data['features']

    print(f"[✓] Rankings matrix loaded {rankings.shape}")

    return alpha_df, rankings, dates, isins, feature_names


# ============================================================
# VECTORIZED EVALUATION
# ============================================================

def evaluate_ensemble_vectorized(rankings, dates, isins, feature_indices, alpha_df, n_satellites, date_mask):
    """Ultra-fast vectorized ensemble evaluation."""

    # Average rankings across features (fully vectorized)
    ensemble_scores = np.nanmean(rankings[:, :, feature_indices], axis=2)

    # Get training date indices
    train_idx = np.where(date_mask)[0]
    if len(train_idx) == 0:
        return None

    alphas_list = []

    # Vectorized top-N selection for each date
    for idx in train_idx:
        scores = ensemble_scores[idx]
        valid = ~np.isnan(scores)

        if valid.sum() < n_satellites:
            continue

        # Fast top-N selection
        valid_scores = scores[valid]
        valid_isins = isins[valid]
        top_idx = np.argpartition(valid_scores, -n_satellites)[-n_satellites:]
        selected = valid_isins[top_idx]

        # Batch lookup
        try:
            date = dates[idx]
            alphas = alpha_df.loc[(date, selected), 'forward_alpha'].values
            avg = np.nanmean(alphas)
            if not np.isnan(avg):
                alphas_list.append(avg)
        except:
            continue

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
# FAST GREEDY SEARCH
# ============================================================

def fast_greedy_search(rankings, dates, isins, feature_names, alpha_df, n_satellites, train_mask):
    """Fast greedy ensemble search with minimal overhead."""

    n_features = len(feature_names)

    # Step 1: Pre-filter features (vectorized)
    print(f"  [1/2] Evaluating {n_features} features...")

    candidates = []

    for feat_idx in tqdm(range(n_features), desc="  Features", leave=False):
        perf = evaluate_ensemble_vectorized(
            rankings, dates, isins, [feat_idx],
            alpha_df, n_satellites, train_mask
        )

        if perf is None:
            continue

        # Check criteria
        if perf['avg_alpha'] >= MIN_ALPHA and perf['hit_rate'] >= MIN_HIT_RATE:
            candidates.append(feat_idx)
        elif perf['avg_alpha'] <= -MIN_ALPHA and perf['hit_rate'] >= MIN_HIT_RATE:
            # Invert negative predictor
            rankings[:, :, feat_idx] = 1.0 - rankings[:, :, feat_idx]
            candidates.append(feat_idx)

    print(f"  Found {len(candidates)} candidate features")

    if len(candidates) == 0:
        return [], None

    # Step 2: Greedy forward selection
    print(f"  [2/2] Greedy selection...")

    selected = []
    best_perf = None

    for iteration in range(min(MAX_ENSEMBLE_SIZE, len(candidates))):
        current = [f['idx'] for f in selected]
        remaining = [f for f in candidates if f not in current]

        if len(remaining) == 0:
            break

        best_add = None
        best_add_perf = None
        best_improvement = 0

        # Try each remaining feature
        for feat_idx in remaining:
            test_indices = current + [feat_idx]
            perf = evaluate_ensemble_vectorized(
                rankings, dates, isins, test_indices,
                alpha_df, n_satellites, train_mask
            )

            if perf is None:
                continue

            improvement = perf['avg_alpha'] if best_perf is None else perf['avg_alpha'] - best_perf['avg_alpha']

            if improvement > best_improvement:
                best_improvement = improvement
                best_add = feat_idx
                best_add_perf = perf

        # Add if improves
        if best_add is not None and best_improvement >= MIN_IMPROVEMENT:
            selected.append({
                'idx': best_add,
                'name': feature_names[best_add],
                'improvement': best_improvement
            })
            best_perf = best_add_perf
        else:
            break

    return selected, best_perf


# ============================================================
# EXPANDING WINDOW BACKTEST
# ============================================================

def expanding_window_backtest(alpha_df, rankings, dates, isins, feature_names, n_satellites):
    """Run expanding window backtest."""

    print("\n" + "="*60)
    print(f"EXPANDING WINDOW BACKTEST (N={n_satellites})")
    print("="*60)

    # Find start
    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0][0]

    print(f"\nTest period: {dates[test_start_idx].date()} to {dates[-1].date()}")
    print(f"Test months: {len(dates) - test_start_idx}")

    results = []
    current_features = None
    months_since_reopt = 0

    # Walk forward
    for test_idx in tqdm(range(test_start_idx, len(dates)), desc=f"N={n_satellites}"):
        test_date = dates[test_idx]

        # Re-optimize?
        if current_features is None or months_since_reopt >= REOPTIMIZATION_FREQUENCY:
            print(f"\n  Re-optimizing at {test_date.date()}...")

            train_mask = dates < test_date

            selected, train_perf = fast_greedy_search(
                rankings, dates, isins, feature_names,
                alpha_df, n_satellites, train_mask
            )

            if len(selected) == 0:
                print(f"  WARNING: No features found")
                current_features = None
                continue

            current_features = [f['idx'] for f in selected]
            months_since_reopt = 0

            if train_perf:
                print(f"  Selected {len(current_features)} features")
                print(f"  Train: {train_perf['avg_alpha']*100:.2f}% alpha, {train_perf['hit_rate']:.1%} hit")

        months_since_reopt += 1

        # Select ETFs - use current features to rank
        if current_features is None or len(current_features) == 0:
            continue

        # Get ensemble scores: average rankings across selected features
        # rankings shape: (n_dates, n_isins, n_features)
        # Extract: rankings[test_idx, :, current_features] should be (n_isins, n_current_features)
        feature_rankings = rankings[test_idx][:, current_features]  # (818, len(current_features))
        scores = np.nanmean(feature_rankings, axis=1)  # Average across features -> (818,)
        valid = ~np.isnan(scores)

        if valid.sum() < n_satellites:
            continue

        # Top N - apply mask consistently
        valid_scores = scores[valid]
        valid_isins = isins[valid]
        top_idx = np.argpartition(valid_scores, -n_satellites)[-n_satellites:]
        selected_isins = valid_isins[top_idx]

        # Lookup alpha
        try:
            alphas = alpha_df.loc[(test_date, selected_isins), 'forward_alpha'].values
            avg_alpha = np.nanmean(alphas)

            if not np.isnan(avg_alpha):
                results.append({
                    'date': test_date,
                    'n_satellites': n_satellites,
                    'n_features': len(current_features),
                    'avg_alpha': avg_alpha,
                    'selected_isins': ','.join(selected_isins)
                })
        except:
            continue

    return pd.DataFrame(results)


# ============================================================
# ANALYZE
# ============================================================

def analyze_results(results_df, n_satellites):
    """Analyze backtest results."""

    print("\n" + "="*60)
    print(f"RESULTS (N={n_satellites})")
    print("="*60)

    avg_alpha = results_df['avg_alpha'].mean()
    std_alpha = results_df['avg_alpha'].std()
    hit_rate = (results_df['avg_alpha'] > 0).mean()
    sharpe = avg_alpha / std_alpha if std_alpha > 0 else 0

    # Cumulative
    results_df['cumulative'] = (1 + results_df['avg_alpha']).cumprod() - 1
    total_return = results_df['cumulative'].iloc[-1]

    print(f"\nPerformance:")
    print(f"  Periods: {len(results_df)}")
    print(f"  Monthly Alpha: {avg_alpha*100:.2f}% ± {std_alpha*100:.2f}%")
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

def main():
    print("\n" + "="*60)
    print("EXPANDING WINDOW WALK-FORWARD BACKTEST")
    print("="*60)

    # Load
    alpha_df, rankings, dates, isins, feature_names = load_data()

    # Run for each N
    summary_stats = []

    for n in N_SATELLITES_TO_TEST:
        results_df = expanding_window_backtest(
            alpha_df, rankings, dates, isins, feature_names, n
        )

        if len(results_df) == 0:
            continue

        stats = analyze_results(results_df, n)
        summary_stats.append(stats)

        # Save
        output_file = OUTPUT_DIR / f'backtest_N{n}.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n[Saved] {output_file}")

    # Summary
    if len(summary_stats) > 0:
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(OUTPUT_DIR / 'summary.csv', index=False)

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        for _, row in summary_df.iterrows():
            print(f"\nN={int(row['n_satellites'])}:")
            print(f"  Monthly: {row['avg_alpha']*100:.2f}%")
            print(f"  Annual: {row['annual_alpha']*100:.1f}%")
            print(f"  Hit Rate: {row['hit_rate']:.2%}")
            print(f"  Sharpe: {row['sharpe']:.3f}")

        # Best
        best = summary_df.loc[summary_df['sharpe'].idxmax()]
        print("\n" + "="*60)
        print("RECOMMENDATION")
        print("="*60)
        print(f"\nN={int(best['n_satellites'])}")
        print(f"  Annual Alpha: {best['annual_alpha']*100:.1f}%")
        print(f"  Hit Rate: {best['hit_rate']:.2%}")
        print(f"  Sharpe: {best['sharpe']:.3f}")

        # Portfolio estimate
        baseline = 0.08
        satellite_ret = baseline + best['annual_alpha']
        portfolio_ret = 0.6 * baseline + 0.4 * satellite_ret
        print(f"\n60/40 Portfolio:")
        print(f"  Expected Return: {portfolio_ret*100:.1f}%")


if __name__ == '__main__':
    main()
