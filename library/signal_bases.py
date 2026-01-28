"""
Signal Base Computations - HYBRID STANDALONE Implementation
============================================================

This is a standalone hybrid library that combines:
1. FAST numba/bottleneck implementations for most signals
2. ORIGINAL pandas/empyrical implementations for signals that need them

This file is completely self-contained - no imports from other signal libraries.

Signals using ORIGINAL implementations (pandas rolling + empyrical/quantstats):
- gain_pain: Simple pandas rolling sum
- profit_factor: Simple pandas rolling sum
- market_corr: Correlation with market mean
- dispersion_regime: Cross-sectional quantile spread
- return_dispersion_63d: Cross-sectional std

All other signals use OPTIMIZED numba/bottleneck implementations for speed.

Performance characteristics:
- Simple signals: ~same as original (already fast)
- Rolling signals: 2-5x speedup via bottleneck
- Complex signals: 5-20x speedup via numba (except ORIGINAL_ONLY signals)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Generator, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to import optimization libraries
try:
    import bottleneck as bn
    HAS_BOTTLENECK = True
except ImportError:
    HAS_BOTTLENECK = False
    print("Warning: bottleneck not installed. Install with: pip install bottleneck")

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: numba not installed. Install with: pip install numba")

# Original library dependencies (only for specific signals)
import empyrical
import quantstats as qs
import ta
from multiprocessing import Pool, cpu_count


# Signals that MUST use original implementations
ORIGINAL_ONLY_SIGNALS = {
    'gain_pain',    # Simple pandas rolling, keep original for exact match
    'profit_factor',  # Simple pandas rolling
    'market_corr',    # Correlation with market mean return
    'dispersion_regime',  # Cross-sectional quantile
    'return_dispersion_63d',  # Cross-sectional std
}


# ============================================================================
# MODULE-LEVEL WORKER FUNCTIONS (for multiprocessing Pool.map)
# These must be at module level to be picklable
# ============================================================================

def _payoff_worker(args):
    """Worker function for computing payoff ratio in parallel."""
    col_idx, returns_series, window = args
    def compute_payoff(window_returns):
        if len(window_returns.dropna()) < 5:
            return np.nan
        wins = window_returns[window_returns > 0]
        losses = window_returns[window_returns < 0].abs()
        if len(wins) == 0 or len(losses) == 0:
            return np.nan
        return wins.mean() / losses.mean()
    result = returns_series.rolling(window).apply(compute_payoff, raw=False)
    result = result.clip(0, 100)
    return col_idx, result.values


def _up_capture_worker(args):
    """Worker function for computing up capture ratio in parallel."""
    col_idx, returns_series, window, core_ret = args
    def calc_up_capture(etf_window_ret):
        if len(etf_window_ret.dropna()) < window:
            return np.nan
        core_window_ret = core_ret[etf_window_ret.index]
        up_days = core_window_ret > 0
        if up_days.sum() < 5:
            return np.nan
        etf_up_ret = (1 + etf_window_ret[up_days]).prod() ** (252/up_days.sum()) - 1
        core_up_ret = (1 + core_window_ret[up_days]).prod() ** (252/up_days.sum()) - 1
        if abs(core_up_ret) < 0.001:
            return np.nan
        return etf_up_ret / core_up_ret
    result = returns_series.rolling(window).apply(calc_up_capture, raw=False)
    result = result.clip(-5, 5)
    return col_idx, result.values


def _down_capture_worker(args):
    """Worker function for computing down capture ratio in parallel."""
    col_idx, returns_series, window, core_ret = args
    def calc_down_capture(etf_window_ret):
        if len(etf_window_ret.dropna()) < window:
            return np.nan
        core_window_ret = core_ret[etf_window_ret.index]
        down_days = core_window_ret < 0
        if down_days.sum() < 5:
            return np.nan
        etf_down_ret = (1 + etf_window_ret[down_days]).prod() ** (252/down_days.sum()) - 1
        core_down_ret = (1 + core_window_ret[down_days]).prod() ** (252/down_days.sum()) - 1
        if abs(core_down_ret) < 0.001:
            return np.nan
        return etf_down_ret / core_down_ret
    result = returns_series.rolling(window).apply(calc_down_capture, raw=False)
    result = result.clip(-5, 5)
    return col_idx, result.values


# ============================================================================
# OPTIMIZED ROLLING FUNCTIONS (using bottleneck or numba)
# ============================================================================

def rolling_mean(arr: np.ndarray, window: int, min_count: int = None) -> np.ndarray:
    """Fast rolling mean using bottleneck."""
    if min_count is None:
        min_count = window
    if HAS_BOTTLENECK:
        return bn.move_mean(arr, window=window, min_count=min_count, axis=0)
    return pd.DataFrame(arr).rolling(window, min_periods=min_count).mean().values


def rolling_std(arr: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    """Fast rolling std using bottleneck."""
    if HAS_BOTTLENECK:
        return bn.move_std(arr, window=window, min_count=window, ddof=ddof, axis=0)
    return pd.DataFrame(arr).rolling(window, min_periods=window).std(ddof=ddof).values


def rolling_sum(arr: np.ndarray, window: int, min_count: int = None) -> np.ndarray:
    """Fast rolling sum using bottleneck."""
    if min_count is None:
        min_count = window
    if HAS_BOTTLENECK:
        return bn.move_sum(arr, window=window, min_count=min_count, axis=0)
    return pd.DataFrame(arr).rolling(window, min_periods=min_count).sum().values


def rolling_max(arr: np.ndarray, window: int, min_count: int = 1) -> np.ndarray:
    """Fast rolling max using bottleneck."""
    if HAS_BOTTLENECK:
        return bn.move_max(arr, window=window, min_count=min_count, axis=0)
    return pd.DataFrame(arr).rolling(window, min_periods=min_count).max().values


def rolling_min(arr: np.ndarray, window: int, min_count: int = 1) -> np.ndarray:
    """Fast rolling min using bottleneck."""
    if HAS_BOTTLENECK:
        return bn.move_min(arr, window=window, min_count=min_count, axis=0)
    return pd.DataFrame(arr).rolling(window, min_periods=min_count).min().values


def rolling_var(arr: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    """Fast rolling variance using bottleneck."""
    if HAS_BOTTLENECK:
        return bn.move_var(arr, window=window, min_count=window, ddof=ddof, axis=0)
    return pd.DataFrame(arr).rolling(window, min_periods=window).var(ddof=ddof).values


def cummax_2d(arr: np.ndarray) -> np.ndarray:
    """
    Cumulative max along axis 0, handling NaN like pandas cummax().
    """
    return pd.DataFrame(arr).cummax().values


# ============================================================================
# NUMBA-OPTIMIZED COMPLEX CALCULATIONS
# ============================================================================

if HAS_NUMBA:
    @jit(nopython=True, cache=True)
    def _rolling_cov_numba(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        """Rolling covariance using numba (sample covariance, ddof=1)."""
        n = len(x)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            x_win = x[i - window + 1:i + 1]
            y_win = y[i - window + 1:i + 1]

            valid = ~(np.isnan(x_win) | np.isnan(y_win))
            n_valid = valid.sum()
            if n_valid < window:
                continue

            x_valid = x_win[valid]
            y_valid = y_win[valid]

            x_mean = np.mean(x_valid)
            y_mean = np.mean(y_valid)

            cov = np.sum((x_valid - x_mean) * (y_valid - y_mean)) / (n_valid - 1)
            result[i] = cov

        return result

    @jit(nopython=True, cache=True)
    def _rolling_corr_numba(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        """Rolling correlation using numba."""
        n = len(x)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            x_win = x[i - window + 1:i + 1]
            y_win = y[i - window + 1:i + 1]

            valid = ~(np.isnan(x_win) | np.isnan(y_win))
            if valid.sum() < window:
                continue

            x_valid = x_win[valid]
            y_valid = y_win[valid]

            x_mean = np.mean(x_valid)
            y_mean = np.mean(y_valid)
            x_std = np.std(x_valid)
            y_std = np.std(y_valid)

            if x_std < 1e-10 or y_std < 1e-10:
                continue

            corr = np.mean((x_valid - x_mean) * (y_valid - y_mean)) / (x_std * y_std)
            result[i] = corr

        return result

    @jit(nopython=True, cache=True)
    def _rolling_skew_numba(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling skewness using numba."""
        n = len(arr)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            win = arr[i - window + 1:i + 1]
            valid = ~np.isnan(win)
            if valid.sum() < window:
                continue

            x = win[valid]
            mean = np.mean(x)
            std = np.std(x)

            if std < 1e-10:
                continue

            m3 = np.mean(((x - mean) / std) ** 3)
            result[i] = m3

        return result

    @jit(nopython=True, cache=True)
    def _rolling_kurt_numba(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling kurtosis using numba."""
        n = len(arr)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            win = arr[i - window + 1:i + 1]
            valid = ~np.isnan(win)
            if valid.sum() < window:
                continue

            x = win[valid]
            mean = np.mean(x)
            std = np.std(x)

            if std < 1e-10:
                continue

            m4 = np.mean(((x - mean) / std) ** 4)
            result[i] = m4 - 3  # Excess kurtosis

        return result

    @jit(nopython=True, cache=True)
    def _rolling_autocorr_numba(arr: np.ndarray, window: int, lag: int) -> np.ndarray:
        """Rolling autocorrelation using numba."""
        n = len(arr)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            win = arr[i - window + 1:i + 1]
            valid = ~np.isnan(win)
            if valid.sum() < window:
                continue

            x = win[valid]
            if len(x) <= lag:
                continue

            x1 = x[:-lag]
            x2 = x[lag:]

            m1 = np.mean(x1)
            m2 = np.mean(x2)
            s1 = np.std(x1)
            s2 = np.std(x2)

            if s1 < 1e-10 or s2 < 1e-10:
                continue

            corr = np.mean((x1 - m1) * (x2 - m2)) / (s1 * s2)
            result[i] = corr

        return result

    @jit(nopython=True, cache=True)
    def _rolling_cvar_numba(returns: np.ndarray, window: int, cutoff: float = 0.05) -> np.ndarray:
        """
        Rolling Conditional Value at Risk (CVaR / Expected Shortfall).
        Matches empyrical.conditional_value_at_risk exactly:
        cutoff_index = int((n - 1) * cutoff), then mean of [:cutoff_index + 1]
        """
        n = len(returns)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            win = returns[i - window + 1:i + 1]

            valid_count = 0
            for j in range(len(win)):
                if not np.isnan(win[j]):
                    valid_count += 1

            if valid_count < window:
                continue

            valid_returns = np.empty(valid_count)
            idx = 0
            for j in range(len(win)):
                if not np.isnan(win[j]):
                    valid_returns[idx] = win[j]
                    idx += 1

            valid_returns_sorted = np.sort(valid_returns)

            # Empyrical formula: cutoff_index = int((n-1) * cutoff), use [:cutoff_index+1]
            cutoff_index = int((valid_count - 1) * cutoff)
            num_elements = cutoff_index + 1

            tail_sum = 0.0
            for j in range(num_elements):
                tail_sum += valid_returns_sorted[j]

            result[i] = tail_sum / num_elements

        return result

    @jit(nopython=True, cache=True)
    def _rolling_downside_dev_numba(returns: np.ndarray, window: int, annualization: float = 252.0) -> np.ndarray:
        """
        Rolling Downside Deviation (Target Downside Deviation).
        """
        n = len(returns)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            win = returns[i - window + 1:i + 1]

            valid_count = 0
            for j in range(len(win)):
                if not np.isnan(win[j]):
                    valid_count += 1

            if valid_count < window:
                continue

            sum_sq = 0.0
            for j in range(len(win)):
                if not np.isnan(win[j]):
                    if win[j] < 0:
                        sum_sq += win[j] * win[j]

            downside_dev = np.sqrt(sum_sq / valid_count) * np.sqrt(annualization)
            result[i] = downside_dev

        return result

    @jit(nopython=True, cache=True)
    def _rolling_sortino_numba(returns: np.ndarray, window: int, annualization: float = 252.0) -> np.ndarray:
        """
        Rolling Sortino Ratio.
        """
        n = len(returns)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            win = returns[i - window + 1:i + 1]

            valid_count = 0
            ret_sum = 0.0
            for j in range(len(win)):
                if not np.isnan(win[j]):
                    valid_count += 1
                    ret_sum += win[j]

            if valid_count < window:
                continue

            mean_ret = ret_sum / valid_count
            ann_return = mean_ret * annualization

            sum_sq = 0.0
            for j in range(len(win)):
                if not np.isnan(win[j]):
                    if win[j] < 0:
                        sum_sq += win[j] * win[j]

            downside_dev = np.sqrt(sum_sq / valid_count) * np.sqrt(annualization)

            if downside_dev < 1e-10:
                continue

            sortino = ann_return / downside_dev

            if sortino > 100:
                sortino = 100.0
            elif sortino < -100:
                sortino = -100.0

            result[i] = sortino

        return result

    @jit(nopython=True, cache=True)
    def _rolling_ulcer_numba(returns: np.ndarray, window: int) -> np.ndarray:
        """
        Rolling Ulcer Index.
        """
        n = len(returns)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            win = returns[i - window + 1:i + 1]

            valid_count = 0
            for j in range(len(win)):
                if not np.isnan(win[j]):
                    valid_count += 1

            if valid_count < window:
                continue

            cum_wealth = 1.0
            running_max = 1.0
            dd_sq_sum = 0.0
            dd_count = 0

            for j in range(len(win)):
                if not np.isnan(win[j]):
                    cum_wealth = cum_wealth * (1 + win[j])
                    if cum_wealth > running_max:
                        running_max = cum_wealth
                    dd = (cum_wealth - running_max) / running_max
                    dd_sq_sum += dd * dd
                    dd_count += 1

            if dd_count < 2:
                continue

            ulcer = np.sqrt(dd_sq_sum / (dd_count - 1))
            result[i] = ulcer

        return result

    @jit(nopython=True, cache=True)
    def _rolling_hurst_numba(returns: np.ndarray, window: int) -> np.ndarray:
        """
        Rolling Hurst Exponent using simplified variance ratio method.
        """
        n = len(returns)
        result = np.full(n, np.nan)
        lags = np.array([2, 5, 10, 20])

        for i in range(window - 1, n):
            win = returns[i - window + 1:i + 1]

            valid_count = 0
            for j in range(len(win)):
                if not np.isnan(win[j]):
                    valid_count += 1

            if valid_count < window:
                continue

            valid_returns = np.empty(valid_count)
            idx = 0
            for j in range(len(win)):
                if not np.isnan(win[j]):
                    valid_returns[idx] = win[j]
                    idx += 1

            log_lags = np.empty(len(lags))
            log_tau = np.empty(len(lags))
            valid_lags = 0

            for k in range(len(lags)):
                lag = lags[k]
                if lag >= len(valid_returns):
                    continue

                diff_count = len(valid_returns) - lag
                if diff_count < 5:
                    continue

                diff_sum = 0.0
                diff_sq_sum = 0.0
                for j in range(diff_count):
                    d = valid_returns[j + lag] - valid_returns[j]
                    diff_sum += d
                    diff_sq_sum += d * d

                diff_mean = diff_sum / diff_count
                diff_var = diff_sq_sum / diff_count - diff_mean * diff_mean
                if diff_var <= 0:
                    continue

                diff_std = np.sqrt(diff_var)
                if diff_std <= 0:
                    continue

                log_lags[valid_lags] = np.log(lag)
                log_tau[valid_lags] = np.log(diff_std)
                valid_lags += 1

            if valid_lags < 3:
                continue

            sum_x = 0.0
            sum_y = 0.0
            sum_xy = 0.0
            sum_xx = 0.0

            for k in range(valid_lags):
                sum_x += log_lags[k]
                sum_y += log_tau[k]
                sum_xy += log_lags[k] * log_tau[k]
                sum_xx += log_lags[k] * log_lags[k]

            denom = valid_lags * sum_xx - sum_x * sum_x
            if abs(denom) < 1e-10:
                continue

            slope = (valid_lags * sum_xy - sum_x * sum_y) / denom
            result[i] = slope

        return result

    @jit(nopython=True, cache=True)
    def _rolling_entropy_numba(returns: np.ndarray, window: int, n_bins: int = 10) -> np.ndarray:
        """
        Rolling Shannon Entropy of return distribution.
        Matches numpy histogram behavior: when all values are identical, puts all in middle bin.
        """
        n = len(returns)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            win = returns[i - window + 1:i + 1]

            valid_count = 0
            min_val = np.inf
            max_val = -np.inf

            for j in range(len(win)):
                if not np.isnan(win[j]):
                    valid_count += 1
                    if win[j] < min_val:
                        min_val = win[j]
                    if win[j] > max_val:
                        max_val = win[j]

            if valid_count < window:
                continue

            counts = np.zeros(n_bins)

            if max_val <= min_val:
                # All values identical - numpy puts them all in middle bin
                # This matches numpy.histogram behavior with constant data
                counts[n_bins // 2] = valid_count
                bin_width = 1.0  # Arbitrary for constant data
            else:
                bin_width = (max_val - min_val) / n_bins

                for j in range(len(win)):
                    if not np.isnan(win[j]):
                        bin_idx = int((win[j] - min_val) / bin_width)
                        if bin_idx >= n_bins:
                            bin_idx = n_bins - 1
                        if bin_idx < 0:
                            bin_idx = 0
                        counts[bin_idx] += 1

            density = np.zeros(n_bins)
            for k in range(n_bins):
                density[k] = counts[k] / (valid_count * bin_width)

            entropy = 0.0
            for k in range(n_bins):
                if density[k] > 0:
                    entropy -= density[k] * np.log(density[k])

            result[i] = entropy

        return result

    @jit(nopython=True, cache=True)
    def _rolling_recovery_factor_numba(returns: np.ndarray, window: int) -> np.ndarray:
        """
        Rolling Recovery Factor.
        Matches pandas behavior: cummax starts from first cumulative value, not 1.0.
        """
        n = len(returns)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            win = returns[i - window + 1:i + 1]

            valid_count = 0
            total_ret = 0.0

            for j in range(len(win)):
                if not np.isnan(win[j]):
                    valid_count += 1
                    total_ret += win[j]

            if valid_count < window:
                continue

            # Build cumulative wealth and compute max drawdown
            # Match pandas: running_max starts at first cum_wealth value, not 1.0
            cum_wealth = 1.0
            running_max = -1.0  # Sentinel to detect first iteration
            max_dd = 0.0
            first_value = True

            for j in range(len(win)):
                if not np.isnan(win[j]):
                    cum_wealth = cum_wealth * (1 + win[j])
                    if first_value:
                        running_max = cum_wealth
                        first_value = False
                    elif cum_wealth > running_max:
                        running_max = cum_wealth
                    dd = (cum_wealth - running_max) / running_max
                    if dd < max_dd:
                        max_dd = dd

            if max_dd >= -0.001:
                max_dd = -0.001

            recovery = abs(total_ret) / abs(max_dd)

            if recovery > 100:
                recovery = 100.0
            elif recovery < 0:
                recovery = 0.0

            result[i] = recovery

        return result

    @jit(nopython=True, cache=True)
    def _rolling_payoff_numba(returns: np.ndarray, window: int) -> np.ndarray:
        """
        Rolling Payoff Ratio.
        Matches original pandas behavior: requires ALL window values to be non-NaN
        (since pandas rolling without min_periods skips any window with NaN),
        AND at least 1 win AND 1 loss.
        """
        n = len(returns)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            win = returns[i - window + 1:i + 1]

            # Check for ANY NaN in window (pandas default behavior)
            has_nan = False
            for j in range(len(win)):
                if np.isnan(win[j]):
                    has_nan = True
                    break

            if has_nan:
                continue

            # Count wins and losses
            win_sum = 0.0
            win_count = 0
            loss_sum = 0.0
            loss_count = 0

            for j in range(len(win)):
                if win[j] > 0:
                    win_sum += win[j]
                    win_count += 1
                elif win[j] < 0:
                    loss_sum += abs(win[j])
                    loss_count += 1

            # Need at least 1 win and 1 loss
            if win_count == 0 or loss_count == 0:
                continue

            mean_win = win_sum / win_count
            mean_loss = loss_sum / loss_count

            if mean_loss < 1e-10:
                continue

            payoff = mean_win / mean_loss

            if payoff > 100:
                payoff = 100.0
            elif payoff < 0:
                payoff = 0.0

            result[i] = payoff

        return result

    @jit(nopython=True, cache=True)
    def _rolling_stability_numba(returns: np.ndarray, window: int) -> np.ndarray:
        """
        Rolling Stability of Returns (R-squared).
        """
        n = len(returns)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            win = returns[i - window + 1:i + 1]

            valid_count = 0
            for j in range(len(win)):
                if not np.isnan(win[j]):
                    valid_count += 1

            if valid_count < window:
                continue

            cumsum = np.empty(valid_count)
            running_sum = 0.0

            idx = 0
            for j in range(len(win)):
                if not np.isnan(win[j]):
                    r = win[j]
                    if r <= -0.99:
                        r = -0.99
                    running_sum += np.log(1 + r)
                    cumsum[idx] = running_sum
                    idx += 1

            cumsum_sum = 0.0
            for k in range(valid_count):
                cumsum_sum += cumsum[k]
            cumsum_mean = cumsum_sum / valid_count

            cumsum_var = 0.0
            for k in range(valid_count):
                cumsum_var += (cumsum[k] - cumsum_mean) ** 2

            if cumsum_var < 1e-10:
                continue

            x_mean = (valid_count - 1) / 2.0
            x_var = 0.0
            xy_cov = 0.0

            for k in range(valid_count):
                x_var += (k - x_mean) ** 2
                xy_cov += (k - x_mean) * (cumsum[k] - cumsum_mean)

            if x_var < 1e-10:
                continue

            corr = xy_cov / np.sqrt(x_var * cumsum_var)
            r_squared = corr * corr

            result[i] = r_squared

        return result

    @jit(nopython=True, cache=True)
    def _rolling_calmar_numba(returns: np.ndarray, window: int) -> np.ndarray:
        """
        Rolling Calmar Ratio (annual return / max drawdown).
        Matches empyrical.calmar_ratio behavior.
        """
        n = len(returns)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            win = returns[i - window + 1:i + 1]

            # Check for ANY NaN in window (pandas default behavior)
            has_nan = False
            for j in range(len(win)):
                if np.isnan(win[j]):
                    has_nan = True
                    break

            if has_nan:
                continue

            # Calculate cumulative return (ending value)
            cum_ret = 1.0
            for j in range(len(win)):
                cum_ret *= (1.0 + win[j])

            # Calculate max drawdown
            running_max = 1.0
            max_dd = 0.0

            cum_wealth = 1.0
            for j in range(len(win)):
                cum_wealth *= (1.0 + win[j])
                if cum_wealth > running_max:
                    running_max = cum_wealth
                dd = (cum_wealth - running_max) / running_max
                if dd < max_dd:
                    max_dd = dd

            # max_dd is negative or zero
            if max_dd >= 0:
                continue

            # Calculate annualized return (CAGR)
            num_years = window / 252.0
            annual_ret = cum_ret ** (1.0 / num_years) - 1.0

            # Calmar = annual_return / abs(max_drawdown)
            calmar = annual_ret / abs(max_dd)

            # Clip to [-100, 100]
            if calmar > 100:
                calmar = 100.0
            elif calmar < -100:
                calmar = -100.0

            result[i] = calmar

        return result

    @jit(nopython=True, cache=True)
    def _rolling_up_capture_numba(etf_returns: np.ndarray, core_returns: np.ndarray, window: int) -> np.ndarray:
        """
        Rolling Up Capture Ratio.
        Matches original: requires window non-NaN ETF values AND 5+ up days in core.
        """
        n = len(etf_returns)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            etf_win = etf_returns[i - window + 1:i + 1]
            core_win = core_returns[i - window + 1:i + 1]

            # Count valid ETF returns (matches original dropna check)
            valid_etf_count = 0
            for j in range(len(etf_win)):
                if not np.isnan(etf_win[j]):
                    valid_etf_count += 1

            if valid_etf_count < window:
                continue

            etf_up_prod = 1.0
            core_up_prod = 1.0
            up_count = 0

            for j in range(len(etf_win)):
                if not np.isnan(etf_win[j]) and not np.isnan(core_win[j]):
                    if core_win[j] > 0:
                        etf_up_prod *= (1 + etf_win[j])
                        core_up_prod *= (1 + core_win[j])
                        up_count += 1

            if up_count < 5:
                continue

            etf_up_ret = etf_up_prod ** (252.0 / up_count) - 1
            core_up_ret = core_up_prod ** (252.0 / up_count) - 1

            if abs(core_up_ret) < 0.001:
                continue

            capture = etf_up_ret / core_up_ret

            if capture > 5:
                capture = 5.0
            elif capture < -5:
                capture = -5.0

            result[i] = capture

        return result

    @jit(nopython=True, cache=True)
    def _rolling_down_capture_numba(etf_returns: np.ndarray, core_returns: np.ndarray, window: int) -> np.ndarray:
        """
        Rolling Down Capture Ratio.
        Matches original: requires window non-NaN ETF values AND 5+ down days in core.
        """
        n = len(etf_returns)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            etf_win = etf_returns[i - window + 1:i + 1]
            core_win = core_returns[i - window + 1:i + 1]

            # Count valid ETF returns (matches original dropna check)
            valid_etf_count = 0
            for j in range(len(etf_win)):
                if not np.isnan(etf_win[j]):
                    valid_etf_count += 1

            if valid_etf_count < window:
                continue

            etf_down_prod = 1.0
            core_down_prod = 1.0
            down_count = 0

            for j in range(len(etf_win)):
                if not np.isnan(etf_win[j]) and not np.isnan(core_win[j]):
                    if core_win[j] < 0:
                        etf_down_prod *= (1 + etf_win[j])
                        core_down_prod *= (1 + core_win[j])
                        down_count += 1

            if down_count < 5:
                continue

            etf_down_ret = etf_down_prod ** (252.0 / down_count) - 1
            core_down_ret = core_down_prod ** (252.0 / down_count) - 1

            if abs(core_down_ret) < 0.001:
                continue

            capture = etf_down_ret / core_down_ret

            if capture > 5:
                capture = 5.0
            elif capture < -5:
                capture = -5.0

            result[i] = capture

        return result

else:
    # Fallback implementations without numba (use pandas rolling)
    def _rolling_cov_numba(x, y, window):
        return pd.Series(x).rolling(window).cov(pd.Series(y)).values

    def _rolling_corr_numba(x, y, window):
        return pd.Series(x).rolling(window).corr(pd.Series(y)).values

    def _rolling_skew_numba(arr, window):
        return pd.Series(arr).rolling(window).skew().values

    def _rolling_kurt_numba(arr, window):
        return pd.Series(arr).rolling(window).kurt().values

    def _rolling_autocorr_numba(arr, window, lag):
        def autocorr_fn(x):
            return x.autocorr(lag=lag) if len(x.dropna()) >= window else np.nan
        return pd.Series(arr).rolling(window).apply(autocorr_fn, raw=False).values

    def _rolling_cvar_numba(returns, window, cutoff=0.05):
        result = pd.Series(returns).rolling(window).apply(
            lambda x: empyrical.conditional_value_at_risk(x, cutoff=cutoff) if len(x.dropna()) >= window else np.nan,
            raw=False
        )
        return result.values

    def _rolling_downside_dev_numba(returns, window, annualization=252.0):
        result = pd.Series(returns).rolling(window).apply(
            lambda x: empyrical.downside_risk(x, annualization=annualization) if len(x.dropna()) >= window else np.nan,
            raw=False
        )
        return result.values

    def _rolling_sortino_numba(returns, window, annualization=252.0):
        result = pd.Series(returns).rolling(window).apply(
            lambda x: empyrical.sortino_ratio(x, annualization=annualization) if len(x.dropna()) >= window else np.nan,
            raw=False
        )
        result = result.replace([np.inf, -np.inf], np.nan).clip(-100, 100)
        return result.values

    def _rolling_ulcer_numba(returns, window):
        result = pd.Series(returns).rolling(window).apply(
            lambda x: qs.stats.ulcer_index(x) if len(x.dropna()) >= window else np.nan,
            raw=False
        )
        return result.values

    def _rolling_hurst_numba(returns, window):
        def compute_hurst(window_returns):
            if len(window_returns.dropna()) < window:
                return np.nan
            try:
                lags = [2, 5, 10, 20]
                tau = []
                for lag in lags:
                    std_lag = np.std(window_returns.diff(lag).dropna())
                    if std_lag > 0:
                        tau.append(std_lag)
                    else:
                        return np.nan
                if len(tau) < 3:
                    return np.nan
                poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
                return poly[0]
            except:
                return np.nan
        result = pd.Series(returns).rolling(window).apply(compute_hurst, raw=False)
        return result.values

    def _rolling_entropy_numba(returns, window, n_bins=10):
        def compute_entropy(window_returns):
            if len(window_returns.dropna()) < window:
                return np.nan
            try:
                hist, _ = np.histogram(window_returns.dropna(), bins=n_bins, density=True)
                hist = hist[hist > 0]
                if len(hist) == 0:
                    return np.nan
                return -np.sum(hist * np.log(hist))
            except:
                return np.nan
        result = pd.Series(returns).rolling(window).apply(compute_entropy, raw=False)
        return result.values

    def _rolling_recovery_factor_numba(returns, window):
        def compute_recovery_factor(window_returns):
            if len(window_returns.dropna()) < window:
                return np.nan
            total_ret = window_returns.sum()
            cum_wealth = (1 + window_returns).cumprod()
            running_max = cum_wealth.cummax()
            dd = (cum_wealth - running_max) / running_max
            max_dd = dd.min()
            if max_dd >= -0.001:
                max_dd = -0.001
            return min(100, max(0, abs(total_ret) / abs(max_dd)))
        result = pd.Series(returns).rolling(window).apply(compute_recovery_factor, raw=False)
        return result.values

    def _rolling_payoff_numba(returns, window):
        def compute_payoff(window_returns):
            if len(window_returns.dropna()) < 5:
                return np.nan
            wins = window_returns[window_returns > 0]
            losses = window_returns[window_returns < 0].abs()
            if len(wins) == 0 or len(losses) == 0:
                return np.nan
            return min(100, max(0, wins.mean() / losses.mean()))
        result = pd.Series(returns).rolling(window).apply(compute_payoff, raw=False)
        return result.values

    def _rolling_stability_numba(returns, window):
        def compute_stability(window_returns):
            if len(window_returns.dropna()) < window:
                return np.nan
            log_ret = np.log1p(window_returns.clip(-0.99, None))
            cumsum = log_ret.cumsum()
            x = np.arange(len(cumsum))
            if cumsum.std() < 1e-6:
                return np.nan
            corr = np.corrcoef(x, cumsum)[0, 1]
            return corr ** 2
        result = pd.Series(returns).rolling(window).apply(compute_stability, raw=False)
        return result.values

    def _rolling_up_capture_numba(etf_returns, core_returns, window):
        result = np.full(len(etf_returns), np.nan)
        for i in range(window - 1, len(etf_returns)):
            etf_win = etf_returns[i - window + 1:i + 1]
            core_win = core_returns[i - window + 1:i + 1]
            valid = ~(np.isnan(etf_win) | np.isnan(core_win))
            up_mask = valid & (core_win > 0)
            if up_mask.sum() < 5:
                continue
            etf_up_ret = np.prod(1 + etf_win[up_mask]) ** (252.0 / up_mask.sum()) - 1
            core_up_ret = np.prod(1 + core_win[up_mask]) ** (252.0 / up_mask.sum()) - 1
            if abs(core_up_ret) < 0.001:
                continue
            result[i] = np.clip(etf_up_ret / core_up_ret, -5, 5)
        return result

    def _rolling_down_capture_numba(etf_returns, core_returns, window):
        result = np.full(len(etf_returns), np.nan)
        for i in range(window - 1, len(etf_returns)):
            etf_win = etf_returns[i - window + 1:i + 1]
            core_win = core_returns[i - window + 1:i + 1]
            valid = ~(np.isnan(etf_win) | np.isnan(core_win))
            down_mask = valid & (core_win < 0)
            if down_mask.sum() < 5:
                continue
            etf_down_ret = np.prod(1 + etf_win[down_mask]) ** (252.0 / down_mask.sum()) - 1
            core_down_ret = np.prod(1 + core_win[down_mask]) ** (252.0 / down_mask.sum()) - 1
            if abs(core_down_ret) < 0.001:
                continue
            result[i] = np.clip(etf_down_ret / core_down_ret, -5, 5)
        return result


# ============================================================================
# 2D WRAPPER FUNCTIONS - Apply 1D numba functions across all ETF columns
# ============================================================================

def rolling_cov_2d(arr1: np.ndarray, arr2_1d: np.ndarray, window: int) -> np.ndarray:
    """Rolling covariance of 2D array with 1D array (broadcast)."""
    df1 = pd.DataFrame(arr1)
    s2 = pd.Series(arr2_1d)
    result = np.full(arr1.shape, np.nan)
    for col in df1.columns:
        result[:, col] = df1[col].rolling(window).cov(s2).values
    return result


def rolling_corr_2d(arr1: np.ndarray, arr2_1d: np.ndarray, window: int) -> np.ndarray:
    """Rolling correlation of 2D array with 1D array (broadcast)."""
    df1 = pd.DataFrame(arr1)
    s2 = pd.Series(arr2_1d)
    result = np.full(arr1.shape, np.nan)
    for col in df1.columns:
        result[:, col] = df1[col].rolling(window).corr(s2).values
    return result


def rolling_skew_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling skewness for 2D array using pandas (bias-corrected)."""
    return pd.DataFrame(arr).rolling(window).skew().values


def rolling_kurt_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling kurtosis for 2D array using pandas (bias-corrected)."""
    return pd.DataFrame(arr).rolling(window).kurt().values


def rolling_autocorr_2d(arr: np.ndarray, window: int, lag: int) -> np.ndarray:
    """Rolling autocorrelation for 2D array."""
    n_time, n_cols = arr.shape
    result = np.full((n_time, n_cols), np.nan)

    for col in range(n_cols):
        result[:, col] = _rolling_autocorr_numba(arr[:, col], window, lag)

    return result


# ============================================================================
# 2D WRAPPER FUNCTIONS FOR COMPLEX FINANCIAL METRICS
# ============================================================================

def rolling_cvar_2d(arr: np.ndarray, window: int, cutoff: float = 0.05) -> np.ndarray:
    """Rolling CVaR for 2D array (all ETFs)."""
    n_time, n_cols = arr.shape
    result = np.full((n_time, n_cols), np.nan)
    for col in range(n_cols):
        result[:, col] = _rolling_cvar_numba(arr[:, col], window, cutoff)
    return result


def rolling_downside_dev_2d(arr: np.ndarray, window: int, annualization: float = 252.0) -> np.ndarray:
    """Rolling Downside Deviation for 2D array."""
    n_time, n_cols = arr.shape
    result = np.full((n_time, n_cols), np.nan)
    for col in range(n_cols):
        result[:, col] = _rolling_downside_dev_numba(arr[:, col], window, annualization)
    return result


def rolling_sortino_2d(arr: np.ndarray, window: int, annualization: float = 252.0) -> np.ndarray:
    """Rolling Sortino ratio for 2D array."""
    n_time, n_cols = arr.shape
    result = np.full((n_time, n_cols), np.nan)
    for col in range(n_cols):
        result[:, col] = _rolling_sortino_numba(arr[:, col], window, annualization)
    return result


def rolling_ulcer_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling Ulcer Index for 2D array."""
    n_time, n_cols = arr.shape
    result = np.full((n_time, n_cols), np.nan)
    for col in range(n_cols):
        result[:, col] = _rolling_ulcer_numba(arr[:, col], window)
    return result


def rolling_hurst_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling Hurst exponent for 2D array."""
    n_time, n_cols = arr.shape
    result = np.full((n_time, n_cols), np.nan)
    for col in range(n_cols):
        result[:, col] = _rolling_hurst_numba(arr[:, col], window)
    return result


def rolling_entropy_2d(arr: np.ndarray, window: int, n_bins: int = 10) -> np.ndarray:
    """Rolling entropy for 2D array."""
    n_time, n_cols = arr.shape
    result = np.full((n_time, n_cols), np.nan)
    for col in range(n_cols):
        result[:, col] = _rolling_entropy_numba(arr[:, col], window, n_bins)
    return result


def rolling_recovery_factor_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling Recovery Factor for 2D array."""
    n_time, n_cols = arr.shape
    result = np.full((n_time, n_cols), np.nan)
    for col in range(n_cols):
        result[:, col] = _rolling_recovery_factor_numba(arr[:, col], window)
    return result


def rolling_payoff_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling Payoff ratio for 2D array."""
    n_time, n_cols = arr.shape
    result = np.full((n_time, n_cols), np.nan)
    for col in range(n_cols):
        result[:, col] = _rolling_payoff_numba(arr[:, col], window)
    return result


def rolling_stability_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling Stability of Returns for 2D array."""
    n_time, n_cols = arr.shape
    result = np.full((n_time, n_cols), np.nan)
    for col in range(n_cols):
        result[:, col] = _rolling_stability_numba(arr[:, col], window)
    return result


def rolling_calmar_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling Calmar ratio for 2D array."""
    n_time, n_cols = arr.shape
    result = np.full((n_time, n_cols), np.nan)
    for col in range(n_cols):
        result[:, col] = _rolling_calmar_numba(arr[:, col], window)
    return result


def rolling_up_capture_2d(etf_returns: np.ndarray, core_returns: np.ndarray, window: int) -> np.ndarray:
    """Rolling Up Capture ratio for 2D array."""
    n_time, n_cols = etf_returns.shape
    result = np.full((n_time, n_cols), np.nan)
    for col in range(n_cols):
        result[:, col] = _rolling_up_capture_numba(etf_returns[:, col], core_returns, window)
    return result


def rolling_down_capture_2d(etf_returns: np.ndarray, core_returns: np.ndarray, window: int) -> np.ndarray:
    """Rolling Down Capture ratio for 2D array."""
    n_time, n_cols = etf_returns.shape
    result = np.full((n_time, n_cols), np.nan)
    for col in range(n_cols):
        result[:, col] = _rolling_down_capture_numba(etf_returns[:, col], core_returns, window)
    return result


# ============================================================================
# PARALLEL WORKERS FOR ORIGINAL IMPLEMENTATIONS (calmar, etc.)
# ============================================================================

def _parallel_calmar_worker(args):
    """Worker for Calmar ratio computation using empyrical."""
    col_idx, returns_series, window = args
    result = returns_series.rolling(window).apply(
        lambda x: empyrical.calmar_ratio(x, annualization=252) if len(x.dropna()) >= window else np.nan,
        raw=False
    )
    result = result.clip(-100, 100)
    return col_idx, result.values


def compute_parallel(df, window, worker_func, n_jobs=None, **kwargs):
    """Generic parallel computation across ETF columns."""
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    if kwargs:
        args_list = [(col_idx, df[col], window, *kwargs.values())
                     for col_idx, col in enumerate(df.columns)]
    else:
        args_list = [(col_idx, df[col], window) for col_idx, col in enumerate(df.columns)]

    with Pool(processes=n_jobs) as pool:
        results = pool.map(worker_func, args_list)

    result_array = np.column_stack([r[1] for r in sorted(results, key=lambda x: x[0])])
    return pd.DataFrame(result_array, index=df.index, columns=df.columns)


# ============================================================================
# HYBRID SIGNAL GENERATOR
# ============================================================================

def compute_signal_bases_generator(
    etf_prices: pd.DataFrame,
    core_prices: pd.Series,
    skip_signals: Optional[set] = None
) -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Generator that yields signal bases one at a time as they are computed.
    HYBRID VERSION - uses numba/bottleneck for most signals, original for specific ones.

    Signals using ORIGINAL implementations:
    - calmar: Uses empyrical.calmar_ratio (complex CAGR calculation)
    - gain_pain: Simple pandas rolling sum
    - profit_factor: Simple pandas rolling sum
    - market_corr: Correlation with market mean
    - dispersion_regime: Cross-sectional quantile spread
    - return_dispersion_63d: Cross-sectional std
    """
    if skip_signals is None:
        skip_signals = set()

    n_time = len(etf_prices)
    n_etfs = len(etf_prices.columns)

    # Convert to numpy for fast operations
    prices_arr = etf_prices.values.astype(np.float64)
    core_arr = core_prices.values.astype(np.float64)

    # Track computed signals for internal lookups
    computed_signals = {}

    def should_skip(name):
        return name in skip_signals

    def store_and_yield(data, name):
        if isinstance(data, pd.DataFrame):
            arr = data.values
        elif isinstance(data, np.ndarray):
            arr = data
        else:
            arr = np.asarray(data)
        computed_signals[name] = arr
        if not should_skip(name):
            return (name, arr)
        return None

    def get_signal(name):
        return computed_signals.get(name)

    # =========================================================================
    # RETURNS - Using pandas pct_change() with explicit fill_method=None
    # This prevents forward-filling of NaNs before computing returns,
    # which could introduce subtle look-ahead bias if price data has gaps
    # =========================================================================

    # ETF returns - use pandas pct_change with no forward-fill
    etf_returns = etf_prices.pct_change(fill_method=None)
    etf_returns_arr = etf_returns.values

    # Core returns
    core_returns = core_prices.pct_change(fill_method=None)
    core_returns_arr = core_returns.values

    # Universe returns (mean across ETFs)
    universe_returns = etf_returns.mean(axis=1)
    universe_returns_arr = universe_returns.values

    result = store_and_yield(etf_returns_arr, 'etf_return')
    if result: yield result

    # =========================================================================
    # ALPHA = ETF return - Core return
    # =========================================================================
    alpha_arr = etf_returns_arr - core_returns_arr[:, np.newaxis]
    result = store_and_yield(alpha_arr, 'alpha')
    if result: yield result

    # Alpha vs Universe
    alpha_vs_univ_arr = etf_returns_arr - universe_returns_arr[:, np.newaxis]
    result = store_and_yield(alpha_vs_univ_arr, 'alpha_vs_univ')
    if result: yield result

    # =========================================================================
    # PRICE RATIO = ETF / Core
    # =========================================================================
    ratio_arr = prices_arr / core_arr[:, np.newaxis]
    result = store_and_yield(ratio_arr, 'price_ratio')
    if result: yield result

    # Price Ratio vs Universe
    universe_price_arr = np.nanmean(prices_arr, axis=1)
    ratio_vs_univ_arr = prices_arr / universe_price_arr[:, np.newaxis]
    result = store_and_yield(ratio_vs_univ_arr, 'price_ratio_vs_univ')
    if result: yield result

    # =========================================================================
    # LOG RATIO
    # =========================================================================
    log_ratio_arr = np.log(ratio_arr)
    result = store_and_yield(log_ratio_arr, 'log_ratio')
    if result: yield result

    log_ratio_vs_univ_arr = np.log(ratio_vs_univ_arr)
    result = store_and_yield(log_ratio_vs_univ_arr, 'log_ratio_vs_univ')
    if result: yield result

    # =========================================================================
    # CUMULATIVE RETURNS AND ALPHA - Using fast rolling_sum
    # =========================================================================
    for window in [21, 63, 126, 252]:
        if window < n_time:
            # Cumulative Return
            cum_ret = rolling_sum(etf_returns_arr, window)
            result = store_and_yield(cum_ret, f'cum_return_{window}d')
            if result: yield result

            # Cumulative Alpha vs Core
            cum_alpha_core = rolling_sum(alpha_arr, window)
            result = store_and_yield(cum_alpha_core, f'cum_alpha_core_{window}d')
            if result: yield result

            # Cumulative Alpha vs Universe
            cum_alpha_univ = rolling_sum(alpha_vs_univ_arr, window)
            result = store_and_yield(cum_alpha_univ, f'cum_alpha_univ_{window}d')
            if result: yield result

    # =========================================================================
    # MOMENTUM (using pandas pct_change for exact match with original)
    # =========================================================================
    for period in [21, 63, 126, 252]:
        if period < n_time:
            momentum = etf_prices.pct_change(periods=period).values
            result = store_and_yield(momentum, f'momentum_{period}d')
            if result: yield result

    # =========================================================================
    # RELATIVE STRENGTH (using pandas pct_change for exact match with original)
    # =========================================================================

    # Pre-compute universe price series for RS vs Universe
    universe_price = etf_prices.mean(axis=1)

    for period in [21, 63, 126, 252]:
        if period < n_time:
            # Use pandas pct_change for exact match with original
            etf_mom = etf_prices.pct_change(periods=period)
            core_mom = core_prices.pct_change(periods=period)

            # Avoid division by zero
            core_mom_safe = core_mom.replace(0, np.nan)
            rs = etf_mom.div(core_mom_safe, axis=0)
            rs = rs.replace([np.inf, -np.inf], np.nan)
            result = store_and_yield(rs.values, f'rs_{period}d')
            if result: yield result

            # RS vs Universe
            univ_mom = universe_price.pct_change(periods=period)
            univ_mom_safe = univ_mom.replace(0, np.nan)
            rs_vs_univ = etf_mom.div(univ_mom_safe, axis=0)
            rs_vs_univ = rs_vs_univ.replace([np.inf, -np.inf], np.nan)
            result = store_and_yield(rs_vs_univ.values, f'rs_vs_univ_{period}d')
            if result: yield result

    # =========================================================================
    # SKIP-MONTH MOMENTUM
    # =========================================================================
    for total_period, skip_period in [(63, 21), (63, 42), (63, 63),
                                       (126, 21), (126, 42), (126, 63),
                                       (252, 21), (252, 42), (252, 63)]:
        if total_period < n_time and skip_period < total_period:
            # Shift operations
            shifted_skip = np.empty_like(prices_arr)
            shifted_skip[:skip_period, :] = np.nan
            shifted_skip[skip_period:, :] = prices_arr[:-skip_period, :]

            shifted_total = np.empty_like(prices_arr)
            shifted_total[:total_period, :] = np.nan
            shifted_total[total_period:, :] = prices_arr[:-total_period, :]

            skip_mom = (shifted_skip / shifted_total) - 1
            result = store_and_yield(skip_mom, f'skip_mom_{total_period}d_skip{skip_period}d')
            if result: yield result

    # =========================================================================
    # 52-WEEK HIGH/LOW PROXIMITY
    # =========================================================================
    high_52w = rolling_max(prices_arr, 252, min_count=252)
    low_52w = rolling_min(prices_arr, 252, min_count=252)

    high_proximity = prices_arr / high_52w
    result = store_and_yield(high_proximity, 'high_52w_proximity')
    if result: yield result

    low_proximity = prices_arr / low_52w
    result = store_and_yield(low_proximity, 'low_52w_proximity')
    if result: yield result

    # =========================================================================
    # BETA - Using optimized rolling covariance
    # =========================================================================
    beta_63d_arr = None
    for beta_window in [21, 63, 126, 252]:
        # Beta = Cov(ETF, Core) / Var(Core)
        cov_arr = rolling_cov_2d(etf_returns_arr, core_returns_arr, beta_window)
        var_arr = rolling_var(core_returns_arr[:, np.newaxis], beta_window)
        beta_arr = cov_arr / var_arr

        result = store_and_yield(beta_arr, f'beta_{beta_window}d')
        if result: yield result

        if beta_window == 63:
            beta_63d_arr = beta_arr.copy()
            result = store_and_yield(beta_63d_arr, 'beta')
            if result: yield result

    # =========================================================================
    # IDIOSYNCRATIC RETURN
    # =========================================================================
    for idio_window in [21, 63, 126, 252]:
        beta_w_arr = get_signal(f'beta_{idio_window}d')
        idio_return_w = etf_returns_arr - (beta_w_arr * core_returns_arr[:, np.newaxis])
        result = store_and_yield(idio_return_w, f'idio_return_{idio_window}d')
        if result: yield result

    idio_return = etf_returns_arr - (beta_63d_arr * core_returns_arr[:, np.newaxis])
    result = store_and_yield(idio_return, 'idio_return')
    if result: yield result

    # =========================================================================
    # VOLATILITY - Using fast rolling_std
    # =========================================================================
    for vol_window in [21, 63, 126, 252]:
        vol = rolling_std(etf_returns_arr, vol_window) * np.sqrt(252)
        result = store_and_yield(vol, f'vol_{vol_window}d')
        if result: yield result

    # =========================================================================
    # RELATIVE VOLATILITY
    # =========================================================================
    for rel_vol_window in [21, 63, 126, 252]:
        core_vol_w = rolling_std(core_returns_arr[:, np.newaxis], rel_vol_window) * np.sqrt(252)
        etf_vol_w = rolling_std(etf_returns_arr, rel_vol_window) * np.sqrt(252)
        rel_vol_w = etf_vol_w / core_vol_w
        result = store_and_yield(rel_vol_w, f'rel_vol_{rel_vol_window}d')
        if result: yield result

    core_vol = rolling_std(core_returns_arr[:, np.newaxis], 21) * np.sqrt(252)
    vol_21d = rolling_std(etf_returns_arr, 21) * np.sqrt(252)
    rel_vol = vol_21d / core_vol
    result = store_and_yield(rel_vol, 'rel_vol')
    if result: yield result

    # =========================================================================
    # DRAWDOWN
    # =========================================================================
    running_max = cummax_2d(prices_arr)
    drawdown = (prices_arr - running_max) / running_max
    result = store_and_yield(drawdown, 'drawdown')
    if result: yield result

    # =========================================================================
    # RELATIVE DRAWDOWN
    # =========================================================================
    core_max = pd.Series(core_arr).cummax().values
    core_dd = (core_arr - core_max) / core_max
    rel_dd = drawdown - core_dd[:, np.newaxis]
    result = store_and_yield(rel_dd, 'rel_drawdown')
    if result: yield result

    # =========================================================================
    # DRAWDOWN DURATION
    # =========================================================================
    dd_duration = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns, dtype=float)
    for col in etf_prices.columns:
        prices_col = etf_prices[col]
        at_peak = prices_col >= prices_col.cummax()
        duration = (~at_peak).cumsum() - (~at_peak).cumsum().where(at_peak).ffill().fillna(0)
        dd_duration[col] = duration

    dd_duration_arr = dd_duration.values
    result = store_and_yield(dd_duration_arr, 'drawdown_duration')
    if result: yield result

    # =========================================================================
    # RECOVERY RATE
    # =========================================================================
    for window in [21, 63, 126, 252]:
        dd_change = np.empty_like(drawdown)
        dd_change[:window, :] = np.nan
        dd_change[window:, :] = drawdown[window:, :] - drawdown[:-window, :]
        recovery_rate = dd_change / (dd_duration_arr + 1)
        result = store_and_yield(recovery_rate, f'recovery_rate_{window}d')
        if result: yield result

    # =========================================================================
    # ULCER INDEX - FAST numba implementation
    # =========================================================================
    for window in [21, 63, 126, 252]:
        ulcer = rolling_ulcer_2d(etf_returns_arr, window)
        result = store_and_yield(ulcer, f'ulcer_index_{window}d')
        if result: yield result

    # =========================================================================
    # CVaR - FAST numba implementation
    # =========================================================================
    for window in [21, 63, 126, 252]:
        cvar = rolling_cvar_2d(etf_returns_arr, window, cutoff=0.05)
        result = store_and_yield(cvar, f'cvar_95_{window}d')
        if result: yield result

    # =========================================================================
    # DOWNSIDE DEVIATION - FAST numba implementation
    # =========================================================================
    for window in [21, 63, 126, 252]:
        down_dev = rolling_downside_dev_2d(etf_returns_arr, window)
        result = store_and_yield(down_dev, f'downside_dev_{window}d')
        if result: yield result

    # =========================================================================
    # SHARPE RATIO - Using pandas for exact NaN matching
    # Note: bottleneck std returns tiny values (~1e-12) instead of 0 for constant series,
    # which causes different NaN patterns. Using pandas ensures exact match.
    # =========================================================================
    for window in [21, 63, 126, 252]:
        mean_ret = etf_returns.rolling(window).mean()
        std_ret = etf_returns.rolling(window).std()
        sharpe = (mean_ret / std_ret) * np.sqrt(252)
        result = store_and_yield(sharpe.values, f'sharpe_{window}d')
        if result: yield result

    # =========================================================================
    # INFORMATION RATIO
    # =========================================================================
    for window in [21, 63, 126, 252]:
        mean_alpha = rolling_mean(alpha_arr, window)
        std_alpha = rolling_std(alpha_arr, window)
        ir = (mean_alpha / std_alpha) * np.sqrt(252)
        result = store_and_yield(ir, f'info_ratio_{window}d')
        if result: yield result

    # =========================================================================
    # SORTINO RATIO - FAST numba implementation
    # =========================================================================
    for window in [21, 63, 126, 252]:
        sortino = rolling_sortino_2d(etf_returns_arr, window)
        result = store_and_yield(sortino, f'sortino_{window}d')
        if result: yield result

    # =========================================================================
    # CALMAR RATIO - using numba for speed
    # =========================================================================
    for window in [21, 63, 126, 252]:
        calmar = rolling_calmar_2d(etf_returns_arr, window)
        result = store_and_yield(calmar, f'calmar_{window}d')
        if result: yield result

    # =========================================================================
    # TREYNOR RATIO
    # =========================================================================
    for window in [21, 63, 126, 252]:
        rolling_ret = rolling_mean(etf_returns_arr, window) * 252

        cov_arr = rolling_cov_2d(etf_returns_arr, core_returns_arr, window)
        var_arr = rolling_var(core_returns_arr[:, np.newaxis], window)
        beta_w = cov_arr / var_arr

        treynor = rolling_ret / (np.abs(beta_w) + 0.1)
        treynor = np.clip(treynor, -100, 100)
        result = store_and_yield(treynor, f'treynor_{window}d')
        if result: yield result

    # =========================================================================
    # OMEGA RATIO
    # =========================================================================
    for window in [21, 63, 126, 252]:
        gains = np.clip(etf_returns_arr, 0, None)
        losses = np.clip(etf_returns_arr, None, 0)
        sum_gains = rolling_sum(gains, window)
        sum_losses = np.abs(rolling_sum(losses, window))
        omega = sum_gains / sum_losses
        omega = np.clip(omega, 0, 100)
        result = store_and_yield(omega, f'omega_{window}d')
        if result: yield result

    # =========================================================================
    # GAIN-TO-PAIN RATIO - ORIGINAL implementation (simple pandas rolling)
    # =========================================================================
    for window in [21, 63, 126, 252]:
        sum_returns = etf_returns.rolling(window).sum()
        losses = etf_returns.clip(upper=0).abs()
        sum_losses = losses.rolling(window).sum()
        gpr = sum_returns / sum_losses
        gpr = gpr.clip(-100, 100)
        result = store_and_yield(gpr.values, f'gain_pain_{window}d')
        if result: yield result

    # =========================================================================
    # ULCER PERFORMANCE INDEX
    # =========================================================================
    for window in [21, 63, 126, 252]:
        rolling_ret = rolling_mean(etf_returns_arr, window) * 252
        ulcer_arr = get_signal(f'ulcer_index_{window}d')
        upi = rolling_ret / (ulcer_arr + 0.01)
        upi = np.clip(upi, -100, 100)
        result = store_and_yield(upi, f'ulcer_perf_{window}d')
        if result: yield result

    # =========================================================================
    # RECOVERY FACTOR - FAST numba implementation
    # =========================================================================
    for window in [21, 63, 126, 252]:
        recovery_factor = rolling_recovery_factor_2d(etf_returns_arr, window)
        result = store_and_yield(recovery_factor, f'recovery_factor_{window}d')
        if result: yield result

    # =========================================================================
    # WIN/LOSS ANALYSIS
    # =========================================================================

    # Win Rate
    for window in [21, 63, 126, 252]:
        positive_days = (etf_returns_arr > 0).astype(float)
        win_rate = rolling_mean(positive_days, window)
        result = store_and_yield(win_rate, f'win_rate_{window}d')
        if result: yield result

    # Payoff Ratio - using numba for speed (fixed to match original NaN logic)
    for window in [21, 63, 126, 252]:
        payoff = rolling_payoff_2d(etf_returns_arr, window)
        result = store_and_yield(payoff, f'payoff_{window}d')
        if result: yield result

    # Profit Factor - ORIGINAL implementation (simple pandas rolling)
    for window in [21, 63, 126, 252]:
        gains = etf_returns.clip(lower=0)
        losses = etf_returns.clip(upper=0).abs()
        sum_gains = gains.rolling(window).sum()
        sum_losses = losses.rolling(window).sum()
        profit_factor = sum_gains / sum_losses
        profit_factor = profit_factor.clip(0, 100)
        result = store_and_yield(profit_factor.values, f'profit_factor_{window}d')
        if result: yield result

    # Tail Ratio
    for window in [21, 63, 126, 252]:
        p95 = etf_returns.rolling(window).quantile(0.95).values
        p05 = etf_returns.rolling(window).quantile(0.05).values
        tail_ratio = np.abs(p95) / np.abs(p05)
        tail_ratio = np.clip(tail_ratio, 0, 100)
        result = store_and_yield(tail_ratio, f'tail_ratio_{window}d')
        if result: yield result

    # Stability of Returns (numba)
    for window in [21, 63, 126, 252]:
        stability = rolling_stability_2d(etf_returns_arr, window)
        result = store_and_yield(stability, f'stability_{window}d')
        if result: yield result

    # =========================================================================
    # BETA-ADJUSTED RELATIVE STRENGTH
    # =========================================================================
    rs_252d_arr = get_signal('rs_252d')
    if rs_252d_arr is not None:
        for beta_window in [21, 63, 126, 252]:
            beta_w_arr = get_signal(f'beta_{beta_window}d')
            for damping in [0.3, 0.5, 1.0]:
                beta_adj_rs = rs_252d_arr / (np.abs(beta_w_arr) + damping)
                beta_adj_rs = np.where(np.isinf(beta_adj_rs), np.nan, beta_adj_rs)
                beta_adj_rs = np.clip(beta_adj_rs, -100, 100)
                result = store_and_yield(beta_adj_rs, f'beta_adj_rs_w{beta_window}_d{damping}')
                if result: yield result

    # =========================================================================
    # RATE OF CHANGE
    # =========================================================================
    for period in [10, 20]:
        if period < n_time:
            # Use pandas pct_change for exact match with original
            roc = etf_prices.pct_change(periods=period).values * 100
            result = store_and_yield(roc, f'roc_{period}d')
            if result: yield result

    # =========================================================================
    # MOMENTUM (12-1)
    # =========================================================================
    if 252 < n_time:
        shifted_21 = np.empty_like(prices_arr)
        shifted_21[:21, :] = np.nan
        shifted_21[21:, :] = prices_arr[:-21, :]

        shifted_252 = np.empty_like(prices_arr)
        shifted_252[:252, :] = np.nan
        shifted_252[252:, :] = prices_arr[:-252, :]

        mom_12_1 = (shifted_21 / shifted_252) - 1
        result = store_and_yield(mom_12_1, 'mom_12_1')
        if result: yield result

    # =========================================================================
    # TREND INDICATORS
    # =========================================================================

    # Price vs MA
    for ma_window in [20, 50, 100, 200]:
        ma = rolling_mean(prices_arr, ma_window)
        price_vs_ma = (prices_arr / ma) - 1
        result = store_and_yield(price_vs_ma, f'price_vs_ma{ma_window}')
        if result: yield result

    # MA crossovers
    ma_20 = rolling_mean(prices_arr, 20)
    ma_50 = rolling_mean(prices_arr, 50)
    ma_200 = rolling_mean(prices_arr, 200)

    ma_cross_20_50 = (ma_20 / ma_50) - 1
    result = store_and_yield(ma_cross_20_50, 'ma_cross_20_50')
    if result: yield result

    ma_cross_50_200 = (ma_50 / ma_200) - 1
    result = store_and_yield(ma_cross_50_200, 'ma_cross_50_200')
    if result: yield result

    # =========================================================================
    # MACD - Using ta library (same as original)
    # =========================================================================
    macd_line_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    macd_signal_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    macd_histogram_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)

    for col in etf_prices.columns:
        macd = ta.trend.MACD(etf_prices[col], window_slow=26, window_fast=12, window_sign=9)
        macd_line_vals[col] = macd.macd()
        macd_signal_vals[col] = macd.macd_signal()
        macd_histogram_vals[col] = macd.macd_diff()

    result = store_and_yield(macd_line_vals.values, 'macd_line')
    if result: yield result

    result = store_and_yield(macd_signal_vals.values, 'macd_signal')
    if result: yield result

    result = store_and_yield(macd_histogram_vals.values, 'macd_histogram')
    if result: yield result

    # =========================================================================
    # RSI - Using ta library
    # =========================================================================
    rsi_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    for col in etf_prices.columns:
        rsi_indicator = ta.momentum.RSIIndicator(etf_prices[col], window=14)
        rsi_vals[col] = rsi_indicator.rsi()

    rsi_reversion = 50 - rsi_vals.values
    result = store_and_yield(rsi_reversion, 'rsi_reversion')
    if result: yield result

    # =========================================================================
    # BOLLINGER BANDS - Using ta library
    # =========================================================================
    bb_reversion_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    for col in etf_prices.columns:
        bb = ta.volatility.BollingerBands(etf_prices[col], window=20, window_dev=2)
        pband = bb.bollinger_pband()
        bb_reversion_vals[col] = 1 - pband

    result = store_and_yield(bb_reversion_vals.values, 'bb_reversion')
    if result: yield result

    # =========================================================================
    # CCI - Using ta library
    # =========================================================================
    cci_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    for col in etf_prices.columns:
        cci_indicator = ta.trend.CCIIndicator(
            high=etf_prices[col],
            low=etf_prices[col],
            close=etf_prices[col],
            window=20
        )
        cci_vals[col] = cci_indicator.cci()

    cci_normalized = cci_vals.values / 100
    result = store_and_yield(cci_normalized, 'cci')
    if result: yield result

    # =========================================================================
    # KST - Using ta library
    # =========================================================================
    kst_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    for col in etf_prices.columns:
        kst_indicator = ta.trend.KSTIndicator(etf_prices[col])
        kst_vals[col] = kst_indicator.kst()

    result = store_and_yield(kst_vals.values, 'kst')
    if result: yield result

    # =========================================================================
    # STOCHASTIC OSCILLATOR - Using ta library
    # =========================================================================
    stoch_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    for col in etf_prices.columns:
        stoch_indicator = ta.momentum.StochasticOscillator(
            high=etf_prices[col],
            low=etf_prices[col],
            close=etf_prices[col],
            window=14,
            smooth_window=3
        )
        stoch_vals[col] = stoch_indicator.stoch()

    stoch_normalized = (stoch_vals.values - 50) / 50
    result = store_and_yield(stoch_normalized, 'stochastic')
    if result: yield result

    # =========================================================================
    # HIGHER MOMENTS - Using optimized rolling
    # =========================================================================

    # Skewness
    for window in [21, 63, 126]:
        skew = rolling_skew_2d(etf_returns_arr, window)
        result = store_and_yield(skew, f'skewness_{window}d')
        if result: yield result

    # Kurtosis
    for window in [21, 63, 126]:
        kurt = rolling_kurt_2d(etf_returns_arr, window)
        result = store_and_yield(kurt, f'kurtosis_{window}d')
        if result: yield result

    # =========================================================================
    # CORRELATION WITH CORE
    # =========================================================================
    for window in [21, 63, 126, 252]:
        corr = rolling_corr_2d(etf_returns_arr, core_returns_arr, window)
        result = store_and_yield(corr, f'corr_core_{window}d')
        if result: yield result

    # =========================================================================
    # AUTOCORRELATION - Using optimized version
    # =========================================================================
    for window in [21, 63, 126, 252]:
        for lag in [1, 5]:
            autocorr = rolling_autocorr_2d(etf_returns_arr, window, lag)
            result = store_and_yield(autocorr, f'autocorr_lag{lag}_{window}d')
            if result: yield result

    # =========================================================================
    # HURST EXPONENT - FAST numba implementation
    # =========================================================================
    for window in [63, 126, 252]:
        hurst = rolling_hurst_2d(etf_returns_arr, window)
        result = store_and_yield(hurst, f'hurst_{window}d')
        if result: yield result

    # =========================================================================
    # ENTROPY - FAST numba implementation
    # =========================================================================
    for window in [63, 126]:
        entropy = rolling_entropy_2d(etf_returns_arr, window)
        result = store_and_yield(entropy, f'entropy_{window}d')
        if result: yield result

    # =========================================================================
    # MOMENTUM DYNAMICS
    # =========================================================================

    rs_63d_arr = get_signal('rs_63d')
    if rs_63d_arr is not None:
        mom_accel = np.empty_like(rs_63d_arr)
        mom_accel[:21, :] = np.nan
        mom_accel[21:, :] = rs_63d_arr[21:, :] - rs_63d_arr[:-21, :]
        result = store_and_yield(mom_accel, 'mom_accel')
        if result: yield result

    rs_21d_arr = get_signal('rs_21d')
    rs_252d_arr = get_signal('rs_252d')
    if rs_21d_arr is not None and rs_252d_arr is not None:
        signal_agreement = rs_21d_arr * rs_252d_arr
        result = store_and_yield(signal_agreement, 'signal_agreement')
        if result: yield result

    # =========================================================================
    # Z-SCORE SIGNALS
    # =========================================================================

    # Price z-scores - using pandas for exact NaN matching
    # (bottleneck std gives tiny values instead of 0 for constant series)
    for window in [21, 63, 126, 252]:
        price_mean = etf_prices.rolling(window).mean()
        price_std = etf_prices.rolling(window).std()
        price_zscore = -(etf_prices - price_mean) / price_std
        result = store_and_yield(price_zscore.values, f'price_zscore_{window}d')
        if result: yield result

    # Alpha z-score
    alpha_arr_sig = get_signal('alpha')
    for cumsum_window in [21, 63, 126, 252]:
        alpha_cumsum = rolling_sum(alpha_arr_sig, cumsum_window)
        alpha_long_std = rolling_std(alpha_arr_sig, 252) * np.sqrt(cumsum_window)
        alpha_zscore = -(alpha_cumsum / alpha_long_std)
        result = store_and_yield(alpha_zscore, f'alpha_zscore_{cumsum_window}d')
        if result: yield result

    # Legacy alpha zscore
    alpha_cumsum_63 = rolling_sum(alpha_arr_sig, 63)
    alpha_long_std_63 = rolling_std(alpha_arr_sig, 252) * np.sqrt(63)
    alpha_zscore_legacy = -(alpha_cumsum_63 / alpha_long_std_63)
    result = store_and_yield(alpha_zscore_legacy, 'alpha_zscore')
    if result: yield result

    # RS z-scores
    if rs_252d_arr is not None:
        for window in [21, 63, 126, 252]:
            rs_mean = rolling_mean(rs_252d_arr, window)
            rs_std = rolling_std(rs_252d_arr, window)
            rs_zscore = -(rs_252d_arr - rs_mean) / rs_std
            result = store_and_yield(rs_zscore, f'rs_zscore_{window}d')
            if result: yield result

    # Ratio z-score
    ratio_arr_sig = get_signal('price_ratio')
    ratio_mean = rolling_mean(ratio_arr_sig, 252)
    ratio_std = rolling_std(ratio_arr_sig, 252)
    ratio_zscore = -(ratio_arr_sig - ratio_mean) / ratio_std
    result = store_and_yield(ratio_zscore, 'ratio_zscore')
    if result: yield result

    # =========================================================================
    # DISTANCE TO MA
    # =========================================================================
    for ma_window in [20, 50, 100, 200]:
        ma = rolling_mean(prices_arr, ma_window)
        dist_ma = -((prices_arr - ma) / ma)
        result = store_and_yield(dist_ma, f'dist_ma_{ma_window}d')
        if result: yield result

    # =========================================================================
    # MACD VARIANTS
    # =========================================================================

    macd_line_arr = get_signal('macd_line')
    macd_normalized = (macd_line_arr / prices_arr) * 100
    result = store_and_yield(macd_normalized, 'macd_normalized')
    if result: yield result

    # PPO - Use pandas ewm for correct NaN handling
    ema_12_df = etf_prices.ewm(span=12).mean()
    ema_26_df = etf_prices.ewm(span=26).mean()
    ppo = ((ema_12_df - ema_26_df) / ema_26_df) * 100
    result = store_and_yield(ppo.values, 'ppo')
    if result: yield result

    ppo_signal = ppo.ewm(span=9).mean()
    result = store_and_yield(ppo_signal.values, 'ppo_signal')
    if result: yield result

    ppo_histogram = ppo - ppo_signal
    result = store_and_yield(ppo_histogram.values, 'ppo_histogram')
    if result: yield result

    # =========================================================================
    # MACD/PPO/APO VARIANTS WITH DPO-COMPATIBLE PARAMETERS
    # Testing different EMA pairs aligned with DPO's 20-period timescale
    # =========================================================================

    # MACD (10, 20, 9) - Shorter version, 2:1 ratio like standard but faster
    macd_10_20_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    macd_signal_10_20_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    macd_hist_10_20_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)

    for col in etf_prices.columns:
        macd_10_20 = ta.trend.MACD(etf_prices[col], window_slow=20, window_fast=10, window_sign=9)
        macd_10_20_vals[col] = macd_10_20.macd()
        macd_signal_10_20_vals[col] = macd_10_20.macd_signal()
        macd_hist_10_20_vals[col] = macd_10_20.macd_diff()

    result = store_and_yield(macd_10_20_vals.values, 'macd_line_10_20')
    if result: yield result

    result = store_and_yield(macd_signal_10_20_vals.values, 'macd_signal_10_20')
    if result: yield result

    result = store_and_yield(macd_hist_10_20_vals.values, 'macd_histogram_10_20')
    if result: yield result

    # MACD (20, 40, 9) - Longer version, 2:1 ratio, centered around DPO's 20
    macd_20_40_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    macd_signal_20_40_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    macd_hist_20_40_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)

    for col in etf_prices.columns:
        macd_20_40 = ta.trend.MACD(etf_prices[col], window_slow=40, window_fast=20, window_sign=9)
        macd_20_40_vals[col] = macd_20_40.macd()
        macd_signal_20_40_vals[col] = macd_20_40.macd_signal()
        macd_hist_20_40_vals[col] = macd_20_40.macd_diff()

    result = store_and_yield(macd_20_40_vals.values, 'macd_line_20_40')
    if result: yield result

    result = store_and_yield(macd_signal_20_40_vals.values, 'macd_signal_20_40')
    if result: yield result

    result = store_and_yield(macd_hist_20_40_vals.values, 'macd_histogram_20_40')
    if result: yield result

    # MACD (20, 50, 9) - Longer ratio, 2.5:1, matches standard ratio better
    macd_20_50_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    macd_signal_20_50_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    macd_hist_20_50_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)

    for col in etf_prices.columns:
        macd_20_50 = ta.trend.MACD(etf_prices[col], window_slow=50, window_fast=20, window_sign=9)
        macd_20_50_vals[col] = macd_20_50.macd()
        macd_signal_20_50_vals[col] = macd_20_50.macd_signal()
        macd_hist_20_50_vals[col] = macd_20_50.macd_diff()

    result = store_and_yield(macd_20_50_vals.values, 'macd_line_20_50')
    if result: yield result

    result = store_and_yield(macd_signal_20_50_vals.values, 'macd_signal_20_50')
    if result: yield result

    result = store_and_yield(macd_hist_20_50_vals.values, 'macd_histogram_20_50')
    if result: yield result

    # PPO variants with DPO-compatible parameters
    # PPO (10, 20, 9)
    ema_10_df = etf_prices.ewm(span=10).mean()
    ema_20_df = etf_prices.ewm(span=20).mean()
    ppo_10_20 = ((ema_10_df - ema_20_df) / ema_20_df) * 100
    result = store_and_yield(ppo_10_20.values, 'ppo_10_20')
    if result: yield result

    ppo_10_20_signal = ppo_10_20.ewm(span=9).mean()
    result = store_and_yield(ppo_10_20_signal.values, 'ppo_10_20_signal')
    if result: yield result

    ppo_10_20_histogram = ppo_10_20 - ppo_10_20_signal
    result = store_and_yield(ppo_10_20_histogram.values, 'ppo_10_20_histogram')
    if result: yield result

    # PPO (20, 40, 9)
    ema_40_df = etf_prices.ewm(span=40).mean()
    ppo_20_40 = ((ema_20_df - ema_40_df) / ema_40_df) * 100
    result = store_and_yield(ppo_20_40.values, 'ppo_20_40')
    if result: yield result

    ppo_20_40_signal = ppo_20_40.ewm(span=9).mean()
    result = store_and_yield(ppo_20_40_signal.values, 'ppo_20_40_signal')
    if result: yield result

    ppo_20_40_histogram = ppo_20_40 - ppo_20_40_signal
    result = store_and_yield(ppo_20_40_histogram.values, 'ppo_20_40_histogram')
    if result: yield result

    # PPO (20, 50, 9)
    ema_50_df = etf_prices.ewm(span=50).mean()
    ppo_20_50 = ((ema_20_df - ema_50_df) / ema_50_df) * 100
    result = store_and_yield(ppo_20_50.values, 'ppo_20_50')
    if result: yield result

    ppo_20_50_signal = ppo_20_50.ewm(span=9).mean()
    result = store_and_yield(ppo_20_50_signal.values, 'ppo_20_50_signal')
    if result: yield result

    ppo_20_50_histogram = ppo_20_50 - ppo_20_50_signal
    result = store_and_yield(ppo_20_50_histogram.values, 'ppo_20_50_histogram')
    if result: yield result

    # APO (Absolute Price Oscillator) variants - same as MACD but explicit naming
    # APO (10, 20)
    apo_10_20 = ema_10_df - ema_20_df
    result = store_and_yield(apo_10_20.values, 'apo_10_20')
    if result: yield result

    # APO (20, 40)
    apo_20_40 = ema_20_df - ema_40_df
    result = store_and_yield(apo_20_40.values, 'apo_20_40')
    if result: yield result

    # APO (20, 50)
    apo_20_50 = ema_20_df - ema_50_df
    result = store_and_yield(apo_20_50.values, 'apo_20_50')
    if result: yield result

    # =========================================================================
    # ADVANCED TREND INDICATORS
    # =========================================================================

    # DPO - Original (20-period)
    dpo_period = 20
    shift = dpo_period // 2 + 1
    ma_dpo = rolling_mean(prices_arr, dpo_period)
    dpo = np.empty_like(prices_arr)
    dpo[:-shift, :] = prices_arr[shift:, :] - ma_dpo[:-shift, :]
    dpo[-shift:, :] = np.nan
    result = store_and_yield(dpo, 'dpo')
    if result: yield result

    # =========================================================================
    # DPO VARIANTS - PARAMETER EXPLORATION
    # Testing different lookback periods to find optimal cycle detection
    # =========================================================================

    # DPO (10-period) - Shorter, faster cycle detection
    dpo_10_period = 10
    dpo_10_shift = dpo_10_period // 2 + 1
    ma_dpo_10 = rolling_mean(prices_arr, dpo_10_period)
    dpo_10 = np.empty_like(prices_arr)
    dpo_10[:-dpo_10_shift, :] = prices_arr[dpo_10_shift:, :] - ma_dpo_10[:-dpo_10_shift, :]
    dpo_10[-dpo_10_shift:, :] = np.nan
    # DISABLED: DPO(10d) - Performance: 2.878 IR (too noisy)
    # result = store_and_yield(dpo_10, 'dpo_10d')
    # if result: yield result

    # DPO (15-period)
    dpo_15_period = 15
    dpo_15_shift = dpo_15_period // 2 + 1
    ma_dpo_15 = rolling_mean(prices_arr, dpo_15_period)
    dpo_15 = np.empty_like(prices_arr)
    dpo_15[:-dpo_15_shift, :] = prices_arr[dpo_15_shift:, :] - ma_dpo_15[:-dpo_15_shift, :]
    dpo_15[-dpo_15_shift:, :] = np.nan
    # DISABLED: DPO(15d) - Performance: ~3.0 IR (poor)
    # result = store_and_yield(dpo_15, 'dpo_15d')
    # if result: yield result

    # DPO (25-period)
    dpo_25_period = 25
    dpo_25_shift = dpo_25_period // 2 + 1
    ma_dpo_25 = rolling_mean(prices_arr, dpo_25_period)
    dpo_25 = np.empty_like(prices_arr)
    dpo_25[:-dpo_25_shift, :] = prices_arr[dpo_25_shift:, :] - ma_dpo_25[:-dpo_25_shift, :]
    dpo_25[-dpo_25_shift:, :] = np.nan
    # DISABLED: DPO(25d) - Performance: ~3.7 IR (poor)
    # result = store_and_yield(dpo_25, 'dpo_25d')
    # if result: yield result

    # DPO (30-period)
    dpo_30_period = 30
    dpo_30_shift = dpo_30_period // 2 + 1
    ma_dpo_30 = rolling_mean(prices_arr, dpo_30_period)
    dpo_30 = np.empty_like(prices_arr)
    dpo_30[:-dpo_30_shift, :] = prices_arr[dpo_30_shift:, :] - ma_dpo_30[:-dpo_30_shift, :]
    dpo_30[-dpo_30_shift:, :] = np.nan
    # DISABLED: DPO(30d) - Performance: 3.905 IR (suboptimal vs 50d)
    # result = store_and_yield(dpo_30, 'dpo_30d')
    # if result: yield result

    # DPO (40-period)
    dpo_40_period = 40
    dpo_40_shift = dpo_40_period // 2 + 1
    ma_dpo_40 = rolling_mean(prices_arr, dpo_40_period)
    dpo_40 = np.empty_like(prices_arr)
    dpo_40[:-dpo_40_shift, :] = prices_arr[dpo_40_shift:, :] - ma_dpo_40[:-dpo_40_shift, :]
    dpo_40[-dpo_40_shift:, :] = np.nan
    # DISABLED: DPO(40d) - Performance: 4.474 IR (good but suboptimal vs 50d/60d)
    # result = store_and_yield(dpo_40, 'dpo_40d')
    # if result: yield result

    # DPO (50-period)
    dpo_50_period = 50
    dpo_50_shift = dpo_50_period // 2 + 1
    ma_dpo_50 = rolling_mean(prices_arr, dpo_50_period)
    dpo_50 = np.empty_like(prices_arr)
    dpo_50[:-dpo_50_shift, :] = prices_arr[dpo_50_shift:, :] - ma_dpo_50[:-dpo_50_shift, :]
    dpo_50[-dpo_50_shift:, :] = np.nan
    result = store_and_yield(dpo_50, 'dpo_50d')
    if result: yield result

    # DPO (60-period) - Testing upper bounds for optimal cycle
    dpo_60_period = 60
    dpo_60_shift = dpo_60_period // 2 + 1
    ma_dpo_60 = rolling_mean(prices_arr, dpo_60_period)
    dpo_60 = np.empty_like(prices_arr)
    dpo_60[:-dpo_60_shift, :] = prices_arr[dpo_60_shift:, :] - ma_dpo_60[:-dpo_60_shift, :]
    dpo_60[-dpo_60_shift:, :] = np.nan
    result = store_and_yield(dpo_60, 'dpo_60d')
    if result: yield result

    # DPO (70-period)
    dpo_70_period = 70
    dpo_70_shift = dpo_70_period // 2 + 1
    ma_dpo_70 = rolling_mean(prices_arr, dpo_70_period)
    dpo_70 = np.empty_like(prices_arr)
    dpo_70[:-dpo_70_shift, :] = prices_arr[dpo_70_shift:, :] - ma_dpo_70[:-dpo_70_shift, :]
    dpo_70[-dpo_70_shift:, :] = np.nan
    result = store_and_yield(dpo_70, 'dpo_70d')
    if result: yield result

    # DPO (80-period)
    dpo_80_period = 80
    dpo_80_shift = dpo_80_period // 2 + 1
    ma_dpo_80 = rolling_mean(prices_arr, dpo_80_period)
    dpo_80 = np.empty_like(prices_arr)
    dpo_80[:-dpo_80_shift, :] = prices_arr[dpo_80_shift:, :] - ma_dpo_80[:-dpo_80_shift, :]
    dpo_80[-dpo_80_shift:, :] = np.nan
    # DISABLED: DPO(80d) - Performance: 4.532 IR (degrades ensemble performance)
    # result = store_and_yield(dpo_80, 'dpo_80d')
    # if result: yield result

    # DPO (100-period) - Quarterly cycle test
    dpo_100_period = 100
    dpo_100_shift = dpo_100_period // 2 + 1
    ma_dpo_100 = rolling_mean(prices_arr, dpo_100_period)
    dpo_100 = np.empty_like(prices_arr)
    dpo_100[:-dpo_100_shift, :] = prices_arr[dpo_100_shift:, :] - ma_dpo_100[:-dpo_100_shift, :]
    dpo_100[-dpo_100_shift:, :] = np.nan
    # DISABLED: DPO(100d) - Performance: 4.120 IR (degrades ensemble significantly)
    # result = store_and_yield(dpo_100, 'dpo_100d')
    # if result: yield result

    # TRIX - Use pandas ewm for correct NaN handling
    trix_period = 15
    ema1 = etf_prices.ewm(span=trix_period).mean()
    ema2 = ema1.ewm(span=trix_period).mean()
    ema3 = ema2.ewm(span=trix_period).mean()
    trix = (ema3.pct_change() * 100)
    result = store_and_yield(trix.values, 'trix')
    if result: yield result

    trix_signal = trix.ewm(span=9).mean()
    result = store_and_yield(trix_signal.values, 'trix_signal')
    if result: yield result

    # KST signal
    kst_arr = get_signal('kst')
    kst_signal = rolling_mean(kst_arr, 9)
    result = store_and_yield(kst_signal, 'kst_signal')
    if result: yield result

    # Aroon (using ta library)
    aroon_period = 25
    aroon_osc_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    aroon_up_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    aroon_down_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)

    for col in etf_prices.columns:
        aroon = ta.trend.AroonIndicator(etf_prices[col], etf_prices[col], window=aroon_period)
        aroon_up_vals[col] = aroon.aroon_up()
        aroon_down_vals[col] = aroon.aroon_down()
        aroon_osc_vals[col] = aroon_up_vals[col] - aroon_down_vals[col]

    result = store_and_yield(aroon_osc_vals.values, 'aroon_osc')
    if result: yield result
    result = store_and_yield(aroon_up_vals.values, 'aroon_up')
    if result: yield result
    result = store_and_yield(aroon_down_vals.values, 'aroon_down')
    if result: yield result

    # =========================================================================
    # OSCILLATORS & REVERSION
    # =========================================================================

    stoch_arr = get_signal('stochastic')
    stoch_reversion = 50 - ((stoch_arr * 50) + 50)
    result = store_and_yield(stoch_reversion, 'stoch_reversion')
    if result: yield result

    # Williams %R
    wr_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    for col in etf_prices.columns:
        wr_indicator = ta.momentum.WilliamsRIndicator(
            high=etf_prices[col], low=etf_prices[col], close=etf_prices[col], lbp=14
        )
        wr_vals[col] = wr_indicator.williams_r()

    result = store_and_yield(wr_vals.values, 'williams_r')
    if result: yield result

    williams_r_reversion = wr_vals.values + 50
    result = store_and_yield(williams_r_reversion, 'williams_r_reversion')
    if result: yield result

    # TSI
    tsi_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    for col in etf_prices.columns:
        tsi_indicator = ta.momentum.TSIIndicator(etf_prices[col], window_slow=25, window_fast=13)
        tsi_vals[col] = tsi_indicator.tsi()

    result = store_and_yield(tsi_vals.values, 'tsi')
    if result: yield result

    # Use pandas ewm for correct NaN handling
    tsi_signal = tsi_vals.ewm(span=7).mean()
    result = store_and_yield(tsi_signal.values, 'tsi_signal')
    if result: yield result

    # CCI reversion
    cci_arr = get_signal('cci')
    cci_reversion = -cci_arr
    result = store_and_yield(cci_reversion, 'cci_reversion')
    if result: yield result

    # Ultimate Oscillator
    uo_vals = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    for col in etf_prices.columns:
        uo_indicator = ta.momentum.UltimateOscillator(
            high=etf_prices[col], low=etf_prices[col], close=etf_prices[col]
        )
        uo_vals[col] = uo_indicator.ultimate_oscillator()

    result = store_and_yield(uo_vals.values, 'ultimate_osc')
    if result: yield result

    ultimate_osc_reversion = 50 - uo_vals.values
    result = store_and_yield(ultimate_osc_reversion, 'ultimate_osc_reversion')
    if result: yield result

    # =========================================================================
    # DONCHIAN CHANNELS
    # =========================================================================
    donchian_period = 20
    # Use min_count=donchian_period to match pandas default behavior
    don_high = rolling_max(prices_arr, donchian_period, min_count=donchian_period)
    don_low = rolling_min(prices_arr, donchian_period, min_count=donchian_period)

    donchian_pos = (prices_arr - don_low) / (don_high - don_low)
    donchian_pos = np.clip(donchian_pos, 0, 1)
    result = store_and_yield(donchian_pos, 'donchian_pos')
    if result: yield result

    donchian_reversion = 1 - donchian_pos
    result = store_and_yield(donchian_reversion, 'donchian_reversion')
    if result: yield result

    # =========================================================================
    # DRAWDOWN-BASED REVERSION
    # =========================================================================

    dd_arr = get_signal('drawdown')
    dd_reversion = -dd_arr
    result = store_and_yield(dd_reversion, 'dd_reversion')
    if result: yield result

    # Alpha drawdown reversion
    for window in [21, 63, 126, 252]:
        alpha_cumulative = rolling_sum(alpha_arr_sig, window)
        alpha_max = cummax_2d(alpha_cumulative)
        alpha_dd = alpha_cumulative - alpha_max
        alpha_dd_reversion = -alpha_dd
        result = store_and_yield(alpha_dd_reversion, f'alpha_dd_reversion_{window}d')
        if result: yield result

    # Legacy version
    alpha_cumulative_126 = rolling_sum(alpha_arr_sig, 126)
    alpha_max_126 = cummax_2d(alpha_cumulative_126)
    alpha_dd_126 = alpha_cumulative_126 - alpha_max_126
    alpha_dd_reversion_legacy = -alpha_dd_126
    result = store_and_yield(alpha_dd_reversion_legacy, 'alpha_dd_reversion')
    if result: yield result

    # =========================================================================
    # SINGLE-WINDOW MOMENTS (compatibility)
    # =========================================================================
    skewness_arr = get_signal('skewness_63d')
    result = store_and_yield(skewness_arr, 'skewness')
    if result: yield result

    kurtosis_arr = get_signal('kurtosis_63d')
    result = store_and_yield(kurtosis_arr, 'kurtosis')
    if result: yield result

    corr_core_arr = get_signal('corr_core_63d')
    result = store_and_yield(corr_core_arr, 'corr_core')
    if result: yield result

    # =========================================================================
    # CORRELATION & DISPERSION
    # =========================================================================

    diversification = -corr_core_arr
    result = store_and_yield(diversification, 'diversification')
    if result: yield result

    # Market correlation - ORIGINAL implementation
    market_mean_ret = etf_returns.mean(axis=1)
    market_corr = pd.DataFrame(index=etf_prices.index, columns=etf_prices.columns)
    for col in etf_prices.columns:
        market_corr[col] = etf_returns[col].rolling(63).corr(market_mean_ret)
    result = store_and_yield(market_corr.values, 'market_corr')
    if result: yield result

    # Crowding
    momentum_3m = np.empty_like(prices_arr)
    momentum_3m[:63, :] = np.nan
    momentum_3m[63:, :] = (prices_arr[63:, :] - prices_arr[:-63, :]) / prices_arr[:-63, :]

    positive_mom = (momentum_3m > 0).astype(float)
    pct_positive = np.nanmean(positive_mom, axis=1)
    crowding = np.abs(pct_positive - 0.5) * 2
    crowding_arr = np.tile(crowding[:, np.newaxis], (1, n_etfs))
    result = store_and_yield(crowding_arr, 'crowding')
    if result: yield result

    # Herding direction
    herding_direction = (pct_positive > 0.5).astype(float) * 2 - 1
    herding_arr = np.tile(herding_direction[:, np.newaxis], (1, n_etfs))
    result = store_and_yield(herding_arr, 'herding_direction')
    if result: yield result

    # Return dispersion - ORIGINAL implementation (cross-sectional std)
    return_dispersion = etf_returns.std(axis=1)  # Cross-sectional std at each date
    dispersion_arr = np.tile(return_dispersion.values[:, np.newaxis], (1, n_etfs))
    result = store_and_yield(dispersion_arr, 'return_dispersion_63d')
    if result: yield result

    # =========================================================================
    # CAPTURE RATIOS - Using numba for speed (fixed to match original NaN logic)
    # =========================================================================
    core_returns_arr = core_returns.values

    for window in [21, 63, 126, 252]:
        up_capture = rolling_up_capture_2d(etf_returns_arr, core_returns_arr, window)
        down_capture = rolling_down_capture_2d(etf_returns_arr, core_returns_arr, window)

        result = store_and_yield(up_capture, f'up_capture_{window}d')
        if result: yield result

        result = store_and_yield(down_capture, f'down_capture_{window}d')
        if result: yield result

        capture_spread = up_capture - down_capture
        result = store_and_yield(capture_spread, f'capture_spread_{window}d')
        if result: yield result

    # =========================================================================
    # ADVANCED MOMENTUM DYNAMICS
    # =========================================================================

    rs_126d_arr = get_signal('rs_126d')
    if rs_126d_arr is not None:
        mom_accel_6m = np.empty_like(rs_126d_arr)
        mom_accel_6m[:21, :] = np.nan
        mom_accel_6m[21:, :] = rs_126d_arr[21:, :] - rs_126d_arr[:-21, :]
        result = store_and_yield(mom_accel_6m, 'momentum_accel_6m')
        if result: yield result

        mom_accel_6m_3m = np.empty_like(rs_126d_arr)
        mom_accel_6m_3m[:63, :] = np.nan
        mom_accel_6m_3m[63:, :] = rs_126d_arr[63:, :] - rs_126d_arr[:-63, :]
        result = store_and_yield(mom_accel_6m_3m, 'momentum_accel_6m_3m')
        if result: yield result

    if rs_252d_arr is not None:
        mom_accel_12m = np.empty_like(rs_252d_arr)
        mom_accel_12m[:21, :] = np.nan
        mom_accel_12m[21:, :] = rs_252d_arr[21:, :] - rs_252d_arr[:-21, :]
        result = store_and_yield(mom_accel_12m, 'momentum_accel_12m')
        if result: yield result

        mom_accel_12m_3m = np.empty_like(rs_252d_arr)
        mom_accel_12m_3m[:63, :] = np.nan
        mom_accel_12m_3m[63:, :] = rs_252d_arr[63:, :] - rs_252d_arr[:-63, :]
        result = store_and_yield(mom_accel_12m_3m, 'momentum_accel_12m_3m')
        if result: yield result

    # Disagreement signals
    if rs_21d_arr is not None and rs_252d_arr is not None:
        disagreement_bull = rs_252d_arr - rs_21d_arr
        result = store_and_yield(disagreement_bull, 'disagreement_bull')
        if result: yield result

        disagreement_bear = rs_21d_arr - rs_252d_arr
        result = store_and_yield(disagreement_bear, 'disagreement_bear')
        if result: yield result

    if rs_63d_arr is not None and rs_21d_arr is not None:
        disagreement_1m_3m = rs_63d_arr - rs_21d_arr
        result = store_and_yield(disagreement_1m_3m, 'disagreement_1m_3m')
        if result: yield result

    if rs_126d_arr is not None and rs_63d_arr is not None:
        disagreement_3m_6m = rs_126d_arr - rs_63d_arr
        result = store_and_yield(disagreement_3m_6m, 'disagreement_3m_6m')
        if result: yield result

    if rs_252d_arr is not None and rs_126d_arr is not None:
        disagreement_6m_12m = rs_252d_arr - rs_126d_arr
        result = store_and_yield(disagreement_6m_12m, 'disagreement_6m_12m')
        if result: yield result

    if rs_63d_arr is not None and rs_252d_arr is not None:
        disagreement_3m_12m = rs_252d_arr - rs_63d_arr
        result = store_and_yield(disagreement_3m_12m, 'disagreement_3m_12m')
        if result: yield result

    # Trend-momentum divergence
    pvm_arr = get_signal('price_vs_ma50')
    if pvm_arr is not None and rs_63d_arr is not None:
        pvm_mean = rolling_mean(pvm_arr, 63)
        pvm_std = rolling_std(pvm_arr, 63)
        pvm_z = (pvm_arr - pvm_mean) / pvm_std

        rs_mean = rolling_mean(rs_63d_arr, 63)
        rs_std = rolling_std(rs_63d_arr, 63)
        rs_z = (rs_63d_arr - rs_mean) / rs_std

        trend_momentum_divergence = pvm_z - rs_z
        result = store_and_yield(trend_momentum_divergence, 'trend_momentum_divergence')
        if result: yield result

    # =========================================================================
    # REGIME INDICATORS
    # =========================================================================

    core_vol_63 = rolling_std(core_returns_arr[:, np.newaxis], 63)[:, 0] * np.sqrt(252)
    core_vol_mean = rolling_mean(core_vol_63[:, np.newaxis], 252)[:, 0]
    core_vol_std_arr = rolling_std(core_vol_63[:, np.newaxis], 252)[:, 0]
    vol_regime = (core_vol_63 - core_vol_mean) / core_vol_std_arr
    vol_regime_arr = np.tile(vol_regime[:, np.newaxis], (1, n_etfs))
    result = store_and_yield(vol_regime_arr, 'vol_regime')
    if result: yield result

    core_ma200 = rolling_mean(core_arr[:, np.newaxis], 200)[:, 0]
    trend_regime = (core_arr - core_ma200) / core_ma200
    trend_regime_arr = np.tile(trend_regime[:, np.newaxis], (1, n_etfs))
    result = store_and_yield(trend_regime_arr, 'trend_regime')
    if result: yield result

    core_dd_arr = (core_arr - np.maximum.accumulate(core_arr)) / np.maximum.accumulate(core_arr)
    drawdown_regime_arr = np.tile(core_dd_arr[:, np.newaxis], (1, n_etfs))
    result = store_and_yield(drawdown_regime_arr, 'drawdown_regime')
    if result: yield result

    # Dispersion regime - ORIGINAL implementation (cross-sectional quantile)
    dispersion_regime = etf_returns.quantile(0.8, axis=1) - etf_returns.quantile(0.2, axis=1)
    dispersion_regime_arr = np.tile(dispersion_regime.values[:, np.newaxis], (1, n_etfs))
    result = store_and_yield(dispersion_regime_arr, 'dispersion_regime')
    if result: yield result

    # Trend-boosted momentum
    if rs_126d_arr is not None:
        trend_boosted_momentum = rs_126d_arr * (1 + np.clip(trend_regime_arr, 0, 1))
        result = store_and_yield(trend_boosted_momentum, 'trend_boosted_momentum')
        if result: yield result

    # Vol-boosted reversion
    alpha_dd_rev_arr = get_signal('alpha_dd_reversion')
    if alpha_dd_rev_arr is not None:
        vol_boosted_reversion = alpha_dd_rev_arr * (1 + np.clip(vol_regime_arr, 0, 2))
        result = store_and_yield(vol_boosted_reversion, 'vol_boosted_reversion')
        if result: yield result

    # =========================================================================
    # ADVANCED SEASONALITY
    # =========================================================================

    month = etf_prices.index.month.values

    favorable_months = {11, 12, 1, 2, 3, 4}
    is_favorable = np.array([1.0 if m in favorable_months else 0.0 for m in month])
    favorable_arr = np.tile(is_favorable[:, np.newaxis], (1, n_etfs))
    result = store_and_yield(favorable_arr, 'month_favorable')
    if result: yield result

    month_sin = np.sin(2 * np.pi * month / 12)
    month_sin_arr = np.tile(month_sin[:, np.newaxis], (1, n_etfs))
    result = store_and_yield(month_sin_arr, 'month_sin')
    if result: yield result

    month_cos = np.cos(2 * np.pi * month / 12)
    month_cos_arr = np.tile(month_cos[:, np.newaxis], (1, n_etfs))
    result = store_and_yield(month_cos_arr, 'month_cos')
    if result: yield result

    # =========================================================================
    # AUTOCORRELATION (single-window compatibility)
    # =========================================================================
    autocorr_21d = get_signal('autocorr_lag1_21d')
    autocorr_63d = get_signal('autocorr_lag1_63d')
    if autocorr_21d is not None:
        result = store_and_yield(autocorr_21d, 'autocorr_21d')
        if result: yield result
    if autocorr_63d is not None:
        result = store_and_yield(autocorr_63d, 'autocorr_63d')
        if result: yield result

    # =========================================================================
    # SECTOR ROTATION REVERSION
    # =========================================================================
    for period in [21, 63, 126, 252]:
        rs_arr = get_signal(f'rs_{period}d')
        if rs_arr is not None:
            sector_rotation_rev = -rs_arr
            result = store_and_yield(sector_rotation_rev, f'sector_rotation_rev_{period}d')
            if result: yield result

    if rs_126d_arr is not None:
        sector_rotation_rev_legacy = -rs_126d_arr
        result = store_and_yield(sector_rotation_rev_legacy, 'sector_rotation_rev')
        if result: yield result

    # =========================================================================
    # SEASONALITY (month-of-year effects)
    # =========================================================================
    for m in range(1, 13):
        month_signal = (month == m).astype(float)
        month_arr = np.tile(month_signal[:, np.newaxis], (1, n_etfs))
        result = store_and_yield(month_arr, f'month_{m}')
        if result: yield result


def count_total_signals() -> int:
    """
    Count the total number of signals that will be computed.

    Runs a minimal version of the generator to count yields efficiently.
    Uses dummy data (small arrays) so counting is fast without computation overhead.

    Returns:
        Total number of signals that will be yielded
    """
    # Create minimal dummy price data (just enough to trigger the generator)
    # Use very small arrays: 1000 time periods, 2 ETFs
    dummy_prices = pd.DataFrame(
        data=np.random.randn(1000, 2) + 100,
        index=pd.date_range('2020-01-01', periods=1000),
        columns=['ETF1', 'ETF2']
    )
    dummy_core = pd.Series(
        data=np.random.randn(1000) + 100,
        index=pd.date_range('2020-01-01', periods=1000)
    )

    # Count signals by iterating through generator
    signal_count = 0
    for signal_name, signal_data in compute_signal_bases_generator(dummy_prices, dummy_core):
        signal_count += 1

    return signal_count


def compute_all_signal_bases(
    etf_prices: pd.DataFrame,
    core_prices: pd.Series
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute all signal bases - convenience wrapper.

    This is a convenience wrapper around compute_signal_bases_generator() that
    collects all signals into memory and returns them as a 3D array.

    For incremental saving (to avoid memory issues with large datasets),
    use compute_signal_bases_generator() directly instead.

    Args:
        etf_prices: DataFrame of ETF prices (index=dates, columns=ISINs)
        core_prices: Series of core (ACWI) prices

    Returns:
        signals_3d: (n_signals, n_time, n_etfs) array
        signal_names: list of signal names
    """
    n_time = len(etf_prices)
    n_etfs = len(etf_prices.columns)

    signals_list = []
    signal_names = []

    for signal_name, signal_data in compute_signal_bases_generator(etf_prices, core_prices):
        signal_names.append(signal_name)
        signals_list.append(signal_data)

    print(f"\nCompiled {len(signal_names)} signal bases")

    expected_shape = (n_time, n_etfs)
    mismatched = []
    for i, (sig_name, sig_array) in enumerate(zip(signal_names, signals_list)):
        if sig_array.shape != expected_shape:
            mismatched.append((i, sig_name, sig_array.shape))

    if mismatched:
        print("\nERROR: Shape mismatches detected:")
        for idx, name, shape in mismatched:
            print(f"  Signal {idx}: {name:40s} has shape {shape}, expected {expected_shape}")
        raise ValueError(f"Found {len(mismatched)} signals with incorrect shape. Expected {expected_shape}.")

    signals_3d = np.stack(signals_list, axis=0)

    return signals_3d, signal_names
