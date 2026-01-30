"""
Phase 4: Bayesian Strategy with Information Ratio Optimization
==============================================================

INFORMATION RATIO OPTIMIZATION PROJECT - Step 6

- Uses mc_ir_mean_{N}month.npz (from step 5 - IR MC statistics)
- Optimizes INFORMATION RATIO for Bayesian feature selection
- Processes all filtered features without pre-filtering

Bayesian feature selection learned from MC-simulated Information Ratio statistics.

This is the main Bayesian satellite selection strategy. It uses walk-forward
backtesting with 2 core learned hyperparameters that adapt over time.
LEARNED HYPERPARAMETERS (2 total):
----------------------------------
1. DECAY RATE - Beta prior, controls observation forgetting
   - Higher decay = remember more history
   - Lower decay = adapt faster to recent data

2. PRIOR STRENGTH - Gamma prior, controls MC prior trust
   - Higher = trust MC simulations more
   - Lower = let observed data dominate faster

INFORMATION RATIO OPTIMIZATION:
-------------------------------
  - Consolidates alpha, consistency, and risk into single metric
  - IR = (portfolio_alpha) / (tracking_error)
  - Portfolio IR goal: Maximize consistent outperformance vs MSCI World
  - Uses MC-estimated IR statistics for Bayesian feature selection

Usage:
    python 6_bayesian_strategy_ir.py

Output:
    data/backtest_results/bayesian_backtest_ir_summary.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import time
from numba import njit, prange
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from scipy import stats

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================
# CONFIGURATION
# ============================================================

# Holding period
HOLDING_MONTHS = 1

# N values to test (satellite portfolio size)
# Test range determines how many top-ranked features to select each month
N_SATELLITES_TO_TEST = [3, 4, 5]

# Training parameters
MIN_TRAINING_MONTHS = 36
REOPTIMIZATION_FREQUENCY = 1

# ============================================================
# HYPERPARAMETER LEARNING CONFIGURATION
# ============================================================

# 1. DECAY RATE (Beta distribution, maps to [DECAY_MIN, DECAY_MAX])
#    Data-informed prior centered on converged value: 0.955
DECAY_PRIOR_ALPHA = 30.0
DECAY_PRIOR_BETA = 10.0
DECAY_MIN = 0.85
DECAY_MAX = 0.99

# 2. PRIOR STRENGTH (Gamma distribution, maps to [PRIOR_MIN, PRIOR_MAX])
#    Data-informed prior centered on converged value: 55
#    Note: Uses normalization formula: prior = 10 + (gamma/(gamma+50)) * 190
#    To get prior=55: Need gamma mean = 15.517241
PRIOR_STRENGTH_SHAPE = 15.5
PRIOR_STRENGTH_SCALE = 1.0
PRIOR_STRENGTH_MIN = 10.0
PRIOR_STRENGTH_MAX = 200.0

# Common update parameters
HP_UPDATE_WEIGHT = 1.0
PREDICTION_SUCCESS_THRESHOLD = 1.0

# Feature selection parameters
MIN_ENSEMBLE_SIZE = 3  # Force at least 3 features for stability
MAX_ENSEMBLE_SIZE = 10
SELECTION_METHOD = 'greedy_bayesian'
GREEDY_CANDIDATES = 30
DEFAULT_GREEDY_IMPROVEMENT_THRESHOLD = 0.001

# Enhanced priors
USE_STABILITY_IN_PRIOR = True

# MC prior data quality threshold
# Revalidation showed MIN_MC_VALID_DATES=6 had no performance benefit over =1
# Trade-off: 6 months more training vs 0% improvement = not worth it
# Keeping original threshold for simplicity and more test data
MIN_MC_VALID_DATES = 1

# Data and output directories
DATA_DIR = Path(__file__).parent / 'data'
OUTPUT_DIR = DATA_DIR / 'backtest_results'


# ============================================================
# HYPERPARAMETER BELIEFS
# ============================================================

@dataclass
class DecayBelief:
    """Beta belief over decay rate."""
    alpha: float = DECAY_PRIOR_ALPHA
    beta: float = DECAY_PRIOR_BETA
    decay_min: float = DECAY_MIN
    decay_max: float = DECAY_MAX
    prediction_history: List[Tuple[float, float, float]] = field(default_factory=list)

    def mean(self) -> float:
        beta_mean = self.alpha / (self.alpha + self.beta)
        return self.decay_min + beta_mean * (self.decay_max - self.decay_min)

    def sample(self) -> float:
        beta_sample = np.random.beta(self.alpha, self.beta)
        return self.decay_min + beta_sample * (self.decay_max - self.decay_min)

    def ci(self, confidence: float = 0.90) -> Tuple[float, float]:
        lower_q = (1 - confidence) / 2
        upper_q = 1 - lower_q
        beta_lower = stats.beta.ppf(lower_q, self.alpha, self.beta)
        beta_upper = stats.beta.ppf(upper_q, self.alpha, self.beta)
        return (self.decay_min + beta_lower * (self.decay_max - self.decay_min),
                self.decay_min + beta_upper * (self.decay_max - self.decay_min))

    def update(self, predicted: float, actual: float, std: float):
        self.prediction_history.append((predicted, actual, std))
        if std > 0:
            normalized_error = abs(predicted - actual) / std
        else:
            normalized_error = abs(predicted - actual) / 0.01
        is_success = normalized_error < PREDICTION_SUCCESS_THRESHOLD
        if is_success:
            update_strength = HP_UPDATE_WEIGHT * (1 - normalized_error / PREDICTION_SUCCESS_THRESHOLD)
            self.alpha += max(0, update_strength)
        else:
            excess_error = normalized_error - PREDICTION_SUCCESS_THRESHOLD
            update_strength = HP_UPDATE_WEIGHT * min(1.0, excess_error)
            self.beta += update_strength


@dataclass
class PriorStrengthBelief:
    """Gamma belief over prior strength."""
    shape: float = PRIOR_STRENGTH_SHAPE
    rate: float = 1.0 / PRIOR_STRENGTH_SCALE
    prior_min: float = PRIOR_STRENGTH_MIN
    prior_max: float = PRIOR_STRENGTH_MAX
    prediction_history: List[Tuple[float, float, float]] = field(default_factory=list)

    def mean(self) -> float:
        gamma_mean = self.shape / self.rate
        normalized = gamma_mean / (gamma_mean + 50)
        return self.prior_min + normalized * (self.prior_max - self.prior_min)

    def sample(self) -> float:
        gamma_sample = np.random.gamma(self.shape, 1.0 / self.rate)
        normalized = gamma_sample / (gamma_sample + 50)
        return self.prior_min + normalized * (self.prior_max - self.prior_min)

    def ci(self, confidence: float = 0.90) -> Tuple[float, float]:
        lower_q = (1 - confidence) / 2
        upper_q = 1 - lower_q
        gamma_lower = stats.gamma.ppf(lower_q, self.shape, scale=1.0/self.rate)
        gamma_upper = stats.gamma.ppf(upper_q, self.shape, scale=1.0/self.rate)
        norm_lower = gamma_lower / (gamma_lower + 50)
        norm_upper = gamma_upper / (gamma_upper + 50)
        return (self.prior_min + norm_lower * (self.prior_max - self.prior_min),
                self.prior_min + norm_upper * (self.prior_max - self.prior_min))

    def update(self, prior_predicted: float, actual: float, std: float):
        self.prediction_history.append((prior_predicted, actual, std))
        if std > 0:
            normalized_error = abs(prior_predicted - actual) / std
        else:
            normalized_error = abs(prior_predicted - actual) / 0.01
        is_success = normalized_error < PREDICTION_SUCCESS_THRESHOLD
        if is_success:
            update = HP_UPDATE_WEIGHT * (1 - normalized_error / PREDICTION_SUCCESS_THRESHOLD)
            self.shape += max(0, update * 0.5)
        else:
            excess_error = normalized_error - PREDICTION_SUCCESS_THRESHOLD
            update = HP_UPDATE_WEIGHT * min(1.0, excess_error)
            self.rate += update * 0.1


# ============================================================
# FEATURE BELIEF
# ============================================================

@dataclass
class FeatureBelief:
    """Bayesian belief about a feature's true Information Ratio distribution."""
    mu: float = 0.0  # Mean IR
    sigma: float = 0.05  # IR std (uncertainty)
    n_obs: float = 0.0  # Number of observations (for posterior)
    sum_alpha: float = 0.0  # Cumulative sum of IR values
    sum_sq: float = 0.0  # Cumulative sum of squared IR values

    prior_mu: float = 0.0  # Prior IR mean (from MC)
    prior_sigma: float = 0.05  # Prior IR std (from MC)
    prior_strength: float = 50.0  # Strength of prior belief

    ir_prior_mu: float = 0.0  # IR prior mean (from MC)
    ir_prior_sigma: float = 0.05  # IR prior std (from MC)
    hitrate_prior_mu: float = 0.0  # Hit rate prior (secondary metric)

    def probability_positive(self) -> float:
        if self.sigma <= 0:
            return 1.0 if self.mu > 0 else 0.0
        return 1 - stats.norm.cdf(-self.mu / self.sigma)

    def expected_ir(self) -> float:
        if self.sigma <= 0:
            return 0.0
        return self.mu / self.sigma

    def expected_sharpe(self) -> float:
        """Backward compatibility wrapper for expected_ir()."""
        return self.expected_ir()

    def sample(self) -> float:
        return np.random.normal(self.mu, self.sigma)


class BeliefState:
    """
    Maintains Bayesian beliefs for features and 2 learned hyperparameters.

    Learned hyperparameters:
    1. decay_belief: How quickly to forget old observations
    2. prior_strength_belief: How much to trust MC priors
    """

    def __init__(self, n_features: int, feature_names: List[str]):
        self.n_features = n_features
        self.feature_names = feature_names
        self.beliefs: Dict[int, FeatureBelief] = {}

        for i in range(n_features):
            self.beliefs[i] = FeatureBelief()

        # Hyperparameter beliefs (2 learned parameters)
        self.decay_belief = DecayBelief()
        self.prior_strength_belief = PriorStrengthBelief()

    def get_decay(self) -> float:
        return self.decay_belief.mean()

    def get_prior_strength(self) -> float:
        return self.prior_strength_belief.mean()

    def set_prior_from_mc(self, feat_idx: int, ir_mean: float, ir_std: float,
                          hitrate_mu: float = 0.0):
        """
        Set prior using IR statistics from MC.

        Initializes feature belief from Information Ratio estimates.
        """
        belief = self.beliefs[feat_idx]

        # Store IR statistics
        belief.ir_prior_mu = ir_mean
        belief.ir_prior_sigma = ir_std
        belief.hitrate_prior_mu = hitrate_mu

        prior_strength = self.get_prior_strength()

        # Use IR directly for optimization
        belief.prior_mu = ir_mean
        belief.prior_sigma = ir_std
        belief.prior_strength = prior_strength

        # Initialize from IR statistics
        belief.mu = ir_mean
        belief.sigma = ir_std
        belief.n_obs = prior_strength
        belief.sum_alpha = ir_mean * prior_strength
        belief.sum_sq = (ir_std**2 + ir_mean**2) * prior_strength

    def update(self, feat_idx: int, observed_alpha: float, weight: float = 1.0):
        """Update feature belief and all hyperparameter beliefs."""
        belief = self.beliefs[feat_idx]

        current_decay = self.get_decay()

        posterior_predicted = belief.mu
        prior_predicted = belief.prior_mu
        prediction_std = belief.sigma

        belief.n_obs *= current_decay
        belief.sum_alpha *= current_decay
        belief.sum_sq *= current_decay

        belief.n_obs += weight
        belief.sum_alpha += observed_alpha * weight
        belief.sum_sq += (observed_alpha ** 2) * weight

        total_n = belief.prior_strength + belief.n_obs

        if total_n > 0:
            belief.mu = (belief.prior_strength * belief.prior_mu +
                        belief.sum_alpha) / total_n

            if belief.n_obs > 1:
                obs_mean = belief.sum_alpha / belief.n_obs
                obs_var = (belief.sum_sq / belief.n_obs) - obs_mean**2
                obs_var = max(0.0001, obs_var)

                prior_weight = belief.prior_strength / total_n
                obs_weight = belief.n_obs / total_n
                belief.sigma = np.sqrt(
                    prior_weight * belief.prior_sigma**2 +
                    obs_weight * obs_var
                )
            else:
                belief.sigma = belief.prior_sigma

        # Update hyperparameter beliefs (2 learned hyperparameters)
        self.decay_belief.update(posterior_predicted, observed_alpha, prediction_std)
        self.prior_strength_belief.update(prior_predicted, observed_alpha, prediction_std)

    def get_feature_scores(self, method: str = 'expected_ir') -> np.ndarray:
        scores = np.zeros(self.n_features)
        for i, belief in self.beliefs.items():
            if method == 'expected_ir' or method == 'expected_sharpe':  # Support both names
                scores[i] = belief.expected_ir()
            elif method == 'probability_positive':
                scores[i] = belief.probability_positive()
            elif method == 'mean':
                scores[i] = belief.mu
            else:
                scores[i] = belief.mu
        return scores


# ============================================================
# NUMBA FUNCTIONS
# ============================================================

@njit(cache=True)
def select_top_n_isins_by_scores(rankings_slice, feature_indices, feature_scores,
                                  n_satellites, alpha_valid_mask):
    n_isins = rankings_slice.shape[0]
    n_features = len(feature_indices)

    isin_scores = np.zeros(n_isins)

    for i in range(n_isins):
        if alpha_valid_mask is not None and not alpha_valid_mask[i]:
            isin_scores[i] = -np.inf
            continue

        score_sum = 0.0
        weight_sum = 0.0

        for j in range(n_features):
            feat_idx = feature_indices[j]
            feat_weight = max(0.0, feature_scores[j])

            val = rankings_slice[i, feat_idx]
            if not np.isnan(val) and feat_weight > 0:
                score_sum += val * feat_weight
                weight_sum += feat_weight

        if weight_sum > 0:
            isin_scores[i] = score_sum / weight_sum
        else:
            isin_scores[i] = -np.inf

    n_valid = 0
    for i in range(n_isins):
        if isin_scores[i] > -np.inf:
            n_valid += 1

    if n_valid < n_satellites:
        return np.array([-1], dtype=np.int64)

    selected = np.empty(n_satellites, dtype=np.int64)
    used = np.zeros(n_isins, dtype=np.bool_)

    for k in range(n_satellites):
        best_idx = -1
        best_score = -np.inf
        for i in range(n_isins):
            if not used[i] and isin_scores[i] > best_score:
                best_score = isin_scores[i]
                best_idx = i
        selected[k] = best_idx
        if best_idx >= 0:
            used[best_idx] = True

    return selected


@njit(cache=True, parallel=True)
def compute_feature_alphas_for_date(rankings, feature_indices, alpha_matrix,
                                     alpha_valid, date_idx, n_satellites):
    n_features = len(feature_indices)
    feature_alphas = np.empty(n_features)

    for f_idx in prange(n_features):
        feat_idx = feature_indices[f_idx]
        feat_rankings = rankings[date_idx, :, feat_idx]

        n_isins = len(feat_rankings)
        valid_count = 0
        for i in range(n_isins):
            if not np.isnan(feat_rankings[i]) and alpha_valid[date_idx, i]:
                valid_count += 1

        if valid_count < n_satellites:
            feature_alphas[f_idx] = np.nan
            continue

        selected = np.empty(n_satellites, dtype=np.int64)
        used = np.zeros(n_isins, dtype=np.bool_)

        for k in range(n_satellites):
            best_idx = -1
            best_rank = -np.inf
            for i in range(n_isins):
                if not used[i] and not np.isnan(feat_rankings[i]) and alpha_valid[date_idx, i]:
                    if feat_rankings[i] > best_rank:
                        best_rank = feat_rankings[i]
                        best_idx = i
            if best_idx >= 0:
                selected[k] = best_idx
                used[best_idx] = True
            else:
                selected[k] = -1

        alpha_sum = 0.0
        alpha_count = 0
        for k in range(n_satellites):
            idx = selected[k]
            if idx >= 0 and alpha_valid[date_idx, idx]:
                alpha_sum += alpha_matrix[date_idx, idx]
                alpha_count += 1

        if alpha_count > 0:
            feature_alphas[f_idx] = alpha_sum / alpha_count
        else:
            feature_alphas[f_idx] = np.nan

    return feature_alphas


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load precomputed data from Steps 1, 3, and 5."""
    print("=" * 120)
    print("LOADING DATA FROM STEPS 1, 3, 5")
    print("=" * 120)

    horizon_label = f"{HOLDING_MONTHS}month"

    # Load forward alpha (Step 1)
    alpha_file = DATA_DIR / f'forward_alpha_{horizon_label}.parquet'
    if not alpha_file.exists():
        raise FileNotFoundError(f"Forward alpha file not found: {alpha_file}\nRun Step 1 first.")

    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])
    print(f"  [OK] Forward alpha loaded")

    # Load filtered ranking matrix (Step 3)
    rankings_file = DATA_DIR / f'rankings_matrix_filtered_{horizon_label}.npz'
    if not rankings_file.exists():
        raise FileNotFoundError(f"Ranking matrix not found: {rankings_file}\nRun Step 3 first.")

    npz_data = np.load(rankings_file, allow_pickle=True)
    rankings = npz_data['rankings'].astype(np.float64)
    dates = pd.to_datetime(npz_data['dates'])
    isins = npz_data['isins']
    feature_names = list(npz_data['features'])
    print(f"  [OK] Ranking matrix loaded {rankings.shape}")

    # Load MC IR statistics for priors (Step 5)
    mc_file = DATA_DIR / f'mc_ir_mean_{horizon_label}.npz'
    mc_data = None
    if mc_file.exists():
        mc_data = np.load(mc_file, allow_pickle=True)
        print(f"  [OK] MC IR data loaded for priors (Step 5)")
    else:
        print(f"  [WARN] No MC IR data (Step 5) - using uninformative priors")

    n_dates = len(dates)
    n_isins = len(isins)

    isin_to_idx = {isin: idx for idx, isin in enumerate(isins)}
    date_to_idx = {date: idx for idx, date in enumerate(dates)}

    # Build alpha matrix for backtesting
    alpha_matrix = np.full((n_dates, n_isins), np.nan, dtype=np.float64)

    for date, group in alpha_df.groupby('date'):
        if date not in date_to_idx:
            continue
        date_idx = date_to_idx[date]
        for isin, alpha in zip(group['isin'].values, group['forward_alpha'].values):
            if isin in isin_to_idx:
                isin_idx = isin_to_idx[isin]
                alpha_matrix[date_idx, isin_idx] = alpha

    alpha_valid = ~np.isnan(alpha_matrix)

    print(f"  {horizon_label}: {rankings.shape}")
    print(f"  Dates: {dates[0].date()} to {dates[-1].date()} ({len(dates)} months)")
    print(f"  ISINs: {len(isins)}")
    print(f"  Filtered signals: {len(feature_names)}")

    return {
        'alpha_matrix': alpha_matrix,
        'alpha_valid': alpha_valid,
        'rankings': rankings,
        'dates': dates,
        'isins': isins,
        'isin_to_idx': isin_to_idx,
        'date_to_idx': date_to_idx,
        'feature_names': feature_names,
        'mc_data': mc_data
    }


# ============================================================
# BAYESIAN FEATURE SELECTION
# ============================================================

def initialize_beliefs_from_mc(belief_state: BeliefState, data: dict,
                                n_satellites: int, test_idx: int):
    """Initialize feature beliefs with LEARNED hyperparameters."""
    mc_data = data.get('mc_data')
    if mc_data is None:
        for i in range(belief_state.n_features):
            belief_state.set_prior_from_mc(i, ir_mean=0.0, ir_std=0.03,
                                           hitrate_mu=0.0)
        return

    mc_n_satellites = mc_data['n_satellites']
    n_to_idx = {int(n): i for i, n in enumerate(mc_n_satellites)}

    if n_satellites not in n_to_idx:
        for i in range(belief_state.n_features):
            # Use IR parameters as priors
            belief_state.set_prior_from_mc(i, ir_mean=0.0, ir_std=0.03,
                                           hitrate_mu=0.0)
        return

    mc_idx = n_to_idx[n_satellites]
    # Load IR data from MC
    mc_ir_mean = mc_data['mc_ir_mean'][mc_idx]
    mc_ir_std = mc_data['mc_ir_std'][mc_idx]
    mc_hitrates = mc_data.get('mc_hitrates')  # Optional - may not be present in newer versions
    mc_candidate_mask = mc_data['candidate_masks'][mc_idx]

    # MC data may have fewer features than ranking matrix
    n_mc_features = mc_candidate_mask.shape[1]

    for feat_idx in range(belief_state.n_features):
        # Only try to use MC data if this feature was computed in MC phase
        if feat_idx >= n_mc_features:
            # Feature not in MC data - use default values
            belief_state.set_prior_from_mc(
                feat_idx, ir_mean=0.0, ir_std=0.03, hitrate_mu=0.0
            )
            continue

        valid_dates = []
        for d in range(min(test_idx, mc_ir_mean.shape[0])):
            if mc_candidate_mask[d, feat_idx]:
                ir_mu = mc_ir_mean[d, feat_idx]
                if not np.isnan(ir_mu):
                    valid_dates.append(d)

        # Use any MC data available
        if len(valid_dates) >= MIN_MC_VALID_DATES:
            ir_mus = [mc_ir_mean[d, feat_idx] for d in valid_dates]
            ir_stds = [mc_ir_std[d, feat_idx] for d in valid_dates]
            avg_ir_mu = np.nanmean(ir_mus)
            avg_ir_std = np.nanmean(ir_stds)

            if np.isnan(avg_ir_mu):
                avg_ir_mu = 0.0
            if np.isnan(avg_ir_std) or avg_ir_std <= 0:
                avg_ir_std = 0.03

            # Hitrates are optional - only use if available
            avg_hr = 0.5  # Default value if hitrates not available
            if mc_hitrates is not None:
                hrs = [mc_hitrates[mc_idx, d, feat_idx] for d in valid_dates]
                avg_hr = np.nanmean(hrs)
                if np.isnan(avg_hr):
                    avg_hr = 0.5

            if USE_STABILITY_IN_PRIOR and len(valid_dates) >= 3:
                # Compute stability factor based on IR values
                ir_cv = np.nanstd(ir_mus) / (abs(np.nanmean(ir_mus)) + 0.001)
                stability_factor = 1.0 + min(1.0, ir_cv)
                adjusted_std = avg_ir_std * stability_factor
            else:
                adjusted_std = avg_ir_std

            belief_state.set_prior_from_mc(
                feat_idx,
                ir_mean=avg_ir_mu,
                ir_std=adjusted_std,
                hitrate_mu=avg_hr
            )
        else:
            # Use IR parameters with default values
            belief_state.set_prior_from_mc(
                feat_idx, ir_mean=0.0, ir_std=0.03, hitrate_mu=0.0
            )


def update_beliefs_from_history(belief_state: BeliefState, data: dict,
                                 n_satellites: int, train_end_idx: int):
    """Update beliefs from historical data."""
    rankings = data['rankings']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']
    n_features = belief_state.n_features

    feature_indices = np.arange(n_features, dtype=np.int64)

    for date_idx in range(MIN_TRAINING_MONTHS, train_end_idx):
        feature_alphas = compute_feature_alphas_for_date(
            rankings, feature_indices, alpha_matrix, alpha_valid,
            date_idx, n_satellites
        )

        current_decay = belief_state.get_decay()

        for feat_idx in range(n_features):
            alpha = feature_alphas[feat_idx]
            if not np.isnan(alpha):
                months_ago = train_end_idx - date_idx
                weight = current_decay ** months_ago
                belief_state.update(feat_idx, alpha, weight=weight)


def select_features_greedy_bayesian(belief_state: BeliefState,
                                     candidate_mask: np.ndarray) -> Tuple[List[int], int]:
    """
    Greedy ensemble building with PURE IR selection.

    Uses learned hyperparameters to build ensemble:
    1. Start with highest expected IR feature
    2. Greedily add next feature that improves ensemble utility most
    3. Stop when improvement < threshold or MAX_ENSEMBLE_SIZE reached

    Args:
        belief_state: Bayesian beliefs about features (mu, sigma for each)
        candidate_mask: Boolean mask of valid candidates

    Returns: (selected_features, ensemble_size)
    """
    n_features = belief_state.n_features

    # Use greedy improvement threshold
    greedy_improvement_thresh = DEFAULT_GREEDY_IMPROVEMENT_THRESHOLD

    ir_scores = belief_state.get_feature_scores('expected_ir')
    mean_alpha = belief_state.get_feature_scores('mean')

    # Pure IR selection: include all candidates, let greedy pick best by utility
    valid_candidates = []
    for i in range(n_features):
        if candidate_mask[i]:
            valid_candidates.append(i)

    if len(valid_candidates) == 0:
        return [], 0

    candidate_scores = [(i, ir_scores[i], mean_alpha[i]) for i in valid_candidates]
    candidate_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    top_candidates = [x[0] for x in candidate_scores[:GREEDY_CANDIDATES]]

    if len(top_candidates) == 0:
        return [], 0

    selected = [top_candidates[0]]
    remaining = set(top_candidates[1:])

    while len(selected) < MAX_ENSEMBLE_SIZE and len(remaining) > 0:
        best_candidate = None
        best_improvement = -np.inf

        current_utility = compute_ensemble_expected_utility(selected, belief_state)

        for candidate in remaining:
            test_ensemble = selected + [candidate]
            new_utility = compute_ensemble_expected_utility(test_ensemble, belief_state)
            improvement = new_utility - current_utility

            if improvement > best_improvement:
                best_improvement = improvement
                best_candidate = candidate

        # Continue if: (1) below MIN_ENSEMBLE_SIZE OR (2) improvement > threshold
        # Stop only when: above MIN_ENSEMBLE_SIZE AND improvement <= threshold
        if best_candidate is not None:
            if len(selected) < MIN_ENSEMBLE_SIZE or best_improvement > greedy_improvement_thresh:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        else:
            break

    return selected, len(selected)


def compute_ensemble_expected_utility(feature_indices: List[int],
                                       belief_state: BeliefState) -> float:
    if len(feature_indices) == 0:
        return -np.inf

    mus = [belief_state.beliefs[i].mu for i in feature_indices]
    sigmas = [belief_state.beliefs[i].sigma for i in feature_indices]
    probs = [belief_state.beliefs[i].probability_positive() for i in feature_indices]

    ensemble_mu = np.mean(mus)
    ensemble_prob = np.exp(np.mean(np.log(np.clip(probs, 0.01, 0.99))))

    avg_var = np.mean([s**2 for s in sigmas])
    ensemble_sigma = np.sqrt(avg_var)

    if ensemble_sigma <= 0:
        return -np.inf

    return ensemble_mu * ensemble_prob / ensemble_sigma


def select_features_bayesian(belief_state: BeliefState,
                              method: str = 'expected_ir') -> Tuple[List[int], int]:
    """
    Select features using LEARNED hyperparameters.

    Args:
        belief_state: Bayesian beliefs about features
        method: Selection method ('greedy_bayesian' or other)

    Returns: (selected_features, ensemble_size)
    """
    n_features = belief_state.n_features
    candidate_mask = np.ones(n_features, dtype=bool)

    if method == 'greedy_bayesian':
        return select_features_greedy_bayesian(belief_state, candidate_mask)

    # Pure IR selection (no probability threshold)
    scores = belief_state.get_feature_scores(method)
    scores[~candidate_mask] = -np.inf
    selected_indices = np.argsort(scores)[::-1]

    selected = []
    for idx in selected_indices:
        if not candidate_mask[idx]:
            continue
        selected.append(int(idx))
        if len(selected) >= MAX_ENSEMBLE_SIZE:
            break

    return selected, len(selected)


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================

def walk_forward_backtest(data: dict, n_satellites: int,
                          show_progress: bool = True) -> Tuple[pd.DataFrame, List[dict]]:
    """Run walk-forward backtest with 3 learned hyperparameters."""
    dates = data['dates']
    isins = data['isins']
    rankings = data['rankings']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']
    feature_names = data['feature_names']
    n_features = len(feature_names)

    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0]

    if len(test_start_idx) == 0:
        return pd.DataFrame(), []

    test_start_idx = test_start_idx[0]

    if show_progress:
        print(f"\n  Testing N={n_satellites}")
        print(f"  Test period: {dates[test_start_idx].date()} to {dates[-1].date()}")

    results = []
    hp_diagnostics = []
    belief_state = None
    months_since_reopt = REOPTIMIZATION_FREQUENCY
    selected_features = None

    # Persist ALL hyperparameter beliefs across months (2 learned parameters)
    persistent_decay_belief = None
    persistent_prior_strength_belief = None

    iterator = range(test_start_idx, len(dates))
    if show_progress:
        iterator = tqdm(iterator, desc=f"N={n_satellites}", ncols=120)

    for test_idx in iterator:
        test_date = dates[test_idx]

        if months_since_reopt >= REOPTIMIZATION_FREQUENCY:
            belief_state = BeliefState(n_features, feature_names)

            # Restore persistent hyperparameter beliefs (2 learned parameters)
            if persistent_decay_belief is not None:
                belief_state.decay_belief = persistent_decay_belief
                belief_state.prior_strength_belief = persistent_prior_strength_belief

            initialize_beliefs_from_mc(belief_state, data, n_satellites, test_idx)
            update_beliefs_from_history(belief_state, data, n_satellites, test_idx)

            # Save updated hyperparameter beliefs (2 learned parameters)
            persistent_decay_belief = belief_state.decay_belief
            persistent_prior_strength_belief = belief_state.prior_strength_belief

            selected_features, ensemble_size = select_features_bayesian(
                belief_state, method=SELECTION_METHOD
            )

            months_since_reopt = 0

        months_since_reopt += 1

        if not selected_features or len(selected_features) == 0:
            continue

        feature_scores = np.array([
            belief_state.beliefs[f].expected_ir()
            for f in selected_features
        ])

        feature_arr = np.array(selected_features, dtype=np.int64)
        alpha_valid_mask = alpha_valid[test_idx]

        selected_isins = select_top_n_isins_by_scores(
            rankings[test_idx], feature_arr, feature_scores,
            n_satellites, alpha_valid_mask
        )

        if selected_isins[0] == -1:
            continue

        alphas = []
        selected_isin_names = []

        for isin_idx in selected_isins:
            if alpha_valid[test_idx, isin_idx]:
                alphas.append(alpha_matrix[test_idx, isin_idx])
                selected_isin_names.append(isins[isin_idx])

        if len(alphas) == 0:
            continue

        avg_alpha = np.mean(alphas)

        # Get learned hyperparameters (2 total)
        current_decay = belief_state.get_decay()
        current_prior_strength = belief_state.get_prior_strength()

        # Get feature names for selected features
        selected_feature_names = [feature_names[idx] for idx in selected_features]

        results.append({
            'date': test_date,
            'n_satellites': n_satellites,
            'avg_alpha': avg_alpha,
            'n_selected': len(selected_isin_names),
            'n_features': len(selected_features),
            'selected_features': '|'.join(selected_feature_names),
            'learned_decay': current_decay,
            'learned_prior_strength': current_prior_strength,
            'selected_isins': ','.join(selected_isin_names)
        })

        # Record diagnostics (2 learned hyperparameters)
        hp_diagnostics.append({
            'date': test_date,
            'decay': current_decay,
            'prior_strength': current_prior_strength,
            'monthly_alpha': avg_alpha,
        })

        # Update feature beliefs
        if belief_state is not None:
            for feat_idx in selected_features:
                feat_rankings = rankings[test_idx, :, feat_idx]
                valid_mask = ~np.isnan(feat_rankings) & alpha_valid[test_idx]

                if valid_mask.sum() >= n_satellites:
                    valid_indices = np.where(valid_mask)[0]
                    top_indices = valid_indices[
                        np.argsort(feat_rankings[valid_indices])[-n_satellites:]
                    ]
                    feat_alpha = alpha_matrix[test_idx, top_indices].mean()

                    if not np.isnan(feat_alpha):
                        belief_state.update(feat_idx, feat_alpha, weight=1.0)

    return pd.DataFrame(results), hp_diagnostics


# ============================================================
# ANALYSIS
# ============================================================

def calculate_stability_metrics(results_df: pd.DataFrame) -> dict:
    """
    Calculate stability metrics for monthly returns.

    Returns dict with:
    - mean: Mean monthly return
    - std: Standard deviation of monthly returns
    - cv: Coefficient of Variation (std/mean) - lower is more stable
    - min_return: Minimum monthly return
    - max_return: Maximum monthly return
    - negative_months: Number of negative return months
    - total_months: Total number of months
    """
    if len(results_df) == 0:
        return None

    monthly_returns = results_df['avg_alpha'].values
    mean_return = np.mean(monthly_returns)
    std_return = np.std(monthly_returns)
    cv = std_return / mean_return if mean_return != 0 else 0
    min_return = np.min(monthly_returns)
    max_return = np.max(monthly_returns)
    negative_months = np.sum(monthly_returns < 0)
    total_months = len(monthly_returns)

    return {
        'mean': mean_return,
        'std': std_return,
        'cv': cv,
        'min_return': min_return,
        'max_return': max_return,
        'negative_months': negative_months,
        'total_months': total_months,
    }


def analyze_results(results_df: pd.DataFrame, n_satellites: int,
                    hp_diagnostics: List[dict]) -> Optional[dict]:
    if len(results_df) == 0:
        return None

    avg_alpha = results_df['avg_alpha'].mean()
    std_alpha = results_df['avg_alpha'].std()
    hit_rate = (results_df['avg_alpha'] > 0).mean()
    information_ratio = avg_alpha / std_alpha if std_alpha > 0 else 0

    results_df = results_df.copy()
    results_df['cumulative'] = (1 + results_df['avg_alpha']).cumprod() - 1
    total_return = results_df['cumulative'].iloc[-1]

    # Calculate true CAGR (Compound Annual Growth Rate)
    years = len(results_df) / 12
    cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else 0

    # Hyperparameter stats (2 learned parameters)
    avg_decay = results_df['learned_decay'].mean()
    final_decay = results_df['learned_decay'].iloc[-1]
    avg_prior_strength = results_df['learned_prior_strength'].mean()
    final_prior_strength = results_df['learned_prior_strength'].iloc[-1]

    # Stability metrics
    stability = calculate_stability_metrics(results_df)

    return {
        'n_satellites': n_satellites,
        'n_periods': len(results_df),
        'avg_alpha': avg_alpha,
        'std_alpha': std_alpha,
        'annual_alpha': cagr,
        'hit_rate': hit_rate,
        'information_ratio': information_ratio,
        'total_return': total_return,
        'avg_decay': avg_decay,
        'final_decay': final_decay,
        'avg_prior_strength': avg_prior_strength,
        'final_prior_strength': final_prior_strength,
        'stability_mean': stability['mean'] if stability else 0,
        'stability_std': stability['std'] if stability else 0,
        'stability_cv': stability['cv'] if stability else 0,
        'stability_min': stability['min_return'] if stability else 0,
        'stability_max': stability['max_return'] if stability else 0,
        'stability_neg_months': stability['negative_months'] if stability else 0,
    }


def print_hp_evolution(hp_diagnostics: List[dict], n_satellites: int):
    """Print hyperparameter evolution (2 learned parameters) with yearly cumulative alpha."""
    if len(hp_diagnostics) == 0:
        return

    df = pd.DataFrame(hp_diagnostics)
    df['year'] = df['date'].dt.year

    # Calculate yearly cumulative alpha (compound returns for each year)
    yearly_cumulative = {}
    for year in df['year'].unique():
        year_data = df[df['year'] == year]['monthly_alpha'].values
        # Compound the monthly returns: (1 + r1) * (1 + r2) * ... - 1
        cumulative = np.prod(1 + year_data) - 1
        yearly_cumulative[year] = cumulative

    print(f"\n  Hyperparameter Evolution (N={n_satellites}):")
    print(f"  " + "-" * 80)
    print(f"  {'Date':<12} {'Decay':>8} {'Prior':>8} {'Monthly (%)':>12} {'Yearly (%)':>12}")
    print(f"  " + "-" * 80)

    indices = [0]
    for i in range(12, len(df), 12):
        indices.append(i)
    if len(df) - 1 not in indices:
        indices.append(len(df) - 1)

    for idx in indices:
        row = df.iloc[idx]
        date_str = row['date'].strftime('%Y-%m')
        monthly_alpha_pct = row['monthly_alpha'] * 100
        yearly_alpha_pct = yearly_cumulative[row['year']] * 100
        print(f"  {date_str:<12} {row['decay']:>8.4f} {row['prior_strength']:>8.1f} {monthly_alpha_pct:>12.2f} {yearly_alpha_pct:>12.2f}")

    print(f"  " + "-" * 80)


# ============================================================
# MAIN
# ============================================================

def main():
    """Run Bayesian strategy backtest with 2 learned hyperparameters."""
    print("=" * 120)
    print("BAYESIAN STRATEGY WITH LEARNED HYPERPARAMETERS (2 total)")
    print("=" * 120)
    print(f"\nConfiguration:")
    print(f"  Holding period: {HOLDING_MONTHS} month")
    print(f"  N values to test: {N_SATELLITES_TO_TEST}")
    print(f"\nLearned Hyperparameters (2 total):")
    print(f"  1. Decay Rate: Beta({DECAY_PRIOR_ALPHA}, {DECAY_PRIOR_BETA}) -> [{DECAY_MIN}, {DECAY_MAX}]")
    print(f"  2. Prior Strength: Gamma({PRIOR_STRENGTH_SHAPE}, {PRIOR_STRENGTH_SCALE}) -> [{PRIOR_STRENGTH_MIN}, {PRIOR_STRENGTH_MAX}]")
    print(f"\nFeature Selection:")
    print(f"  Method: {SELECTION_METHOD}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    all_stats = []
    all_results = {}
    all_hp_diagnostics = {}

    start_time = time.time()

    for test_num, n in enumerate(N_SATELLITES_TO_TEST, 1):
        print(f"\n{'='*120}")
        print(f"TEST {test_num}/{len(N_SATELLITES_TO_TEST)}: N={n}")
        print(f"{'='*120}")

        results_df, hp_diagnostics = walk_forward_backtest(data, n, show_progress=True)

        if len(results_df) > 0:
            all_results[n] = results_df
            all_hp_diagnostics[n] = hp_diagnostics

            stats = analyze_results(results_df, n, hp_diagnostics)
            if stats:
                all_stats.append(stats)

                print(f"\n  Results:")
                print(f"    Periods: {stats['n_periods']}")
                print(f"    Monthly Alpha: {stats['avg_alpha']*100:.2f}%")
                print(f"    Annual Alpha: {stats['annual_alpha']*100:.1f}%")
                print(f"    Hit Rate: {stats['hit_rate']:.2%}")
                print(f"    Information Ratio: {stats['information_ratio']:.3f}")
                print(f"\n  Stability Metrics (Return Consistency):")
                print(f"    Mean Monthly Return: {stats['stability_mean']*100:7.3f}%")
                print(f"    Std Dev (Volatility): {stats['stability_std']*100:7.3f}%")
                print(f"    Coefficient of Variation: {stats['stability_cv']:7.4f} (lower = more stable)")
                print(f"    Range (Min - Max): {stats['stability_min']*100:7.3f}% to {stats['stability_max']*100:7.3f}%")
                print(f"    Negative Months: {int(stats['stability_neg_months'])}/{stats['n_periods']}")
                print(f"\n  Learned Hyperparameters (final, 2 total):")
                print(f"    Decay: {stats['final_decay']:.4f}")
                print(f"    Prior Strength: {stats['final_prior_strength']:.1f}")

                print_hp_evolution(hp_diagnostics, n)

                output_file = OUTPUT_DIR / f'bayesian_backtest_N{n}.csv'
                results_df.to_csv(output_file, index=False)
        else:
            print(f"  No results")

    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "=" * 120)
    print("SUMMARY - BAYESIAN STRATEGY WITH 2 LEARNED HYPERPARAMETERS")
    print("=" * 120)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if len(all_stats) > 0:
        summary_df = pd.DataFrame(all_stats)

        print("\n" + "-" * 75)
        print(f"{'N':>3} {'Periods':>8} {'Monthly':>10} {'Annual':>10} {'Hit Rate':>10} "
              f"{'IR':>8} {'Decay':>7} {'Prior':>7}")
        print("-" * 75)

        for _, row in summary_df.sort_values('n_satellites').iterrows():
            print(f"{int(row['n_satellites']):>3} {int(row['n_periods']):>8} "
                  f"{row['avg_alpha']*100:>9.2f}% {row['annual_alpha']*100:>9.1f}% "
                  f"{row['hit_rate']:>9.1%} {row['information_ratio']:>8.3f} "
                  f"{row['final_decay']:>7.3f} {row['final_prior_strength']:>7.1f}")

        print("-" * 75)

        # Stability Analysis
        print("\nSTABILITY ANALYSIS - Monthly Return Distribution")
        print("-" * 75)
        print(f"{'N':>3} {'Mean (%)':>10} {'Std (%)':>10} {'CV':>8} {'Range':>20} {'Neg Mo.':>8}")
        print("-" * 75)

        for _, row in summary_df.sort_values('n_satellites').iterrows():
            n_val = int(row['n_satellites'])
            mean = row['stability_mean'] * 100
            std = row['stability_std'] * 100
            cv = row['stability_cv']
            min_ret = row['stability_min'] * 100
            max_ret = row['stability_max'] * 100
            neg_mo = int(row['stability_neg_months'])
            range_str = f"{min_ret:6.2f}% - {max_ret:6.2f}%"

            print(f"{n_val:>3} {mean:10.3f} {std:10.3f} {cv:8.4f} {range_str:>20} {neg_mo:>8d}")

        print("-" * 75)

        # Best by Information Ratio
        best_idx = summary_df['information_ratio'].idxmax()
        best = summary_df.loc[best_idx]
        print(f"\nBest by Information Ratio: N={int(best['n_satellites'])}")
        print(f"  Information Ratio: {best['information_ratio']:.3f}")
        print(f"  Hit Rate: {best['hit_rate']:.1%}")
        print(f"  Annual Alpha: {best['annual_alpha']*100:.1f}%")
        print(f"  Learned (final): Decay={best['final_decay']:.3f}, Prior={best['final_prior_strength']:.1f}")

        # Best by Stability (lowest CV)
        best_stability_idx = summary_df['stability_cv'].idxmin()
        best_stability = summary_df.loc[best_stability_idx]
        print(f"\nMost Stable (lowest CV): N={int(best_stability['n_satellites'])}")
        print(f"  Coefficient of Variation: {best_stability['stability_cv']:.4f}")
        print(f"  Mean Monthly Return: {best_stability['stability_mean']*100:.3f}%")
        print(f"  Std Dev (Volatility): {best_stability['stability_std']*100:.3f}%")
        print(f"  Annual Return: {best_stability['annual_alpha']*100:.1f}%")

        # Save summary
        summary_file = OUTPUT_DIR / 'bayesian_backtest_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\n[SAVED] {summary_file}")


if __name__ == '__main__':
    main()
