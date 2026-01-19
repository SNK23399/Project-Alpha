"""
Step 8b: Bayesian Decay Learning Strategy
==========================================

Extends script 8 with TRUE Bayesian learning of the decay hyperparameter.

Key insight: Instead of using heuristics for decay rate, we treat decay
as an unknown parameter and learn it from data using Bayesian inference.

The decay rate controls how quickly we "forget" old observations:
- High decay (0.98): Trust historical data more, slow adaptation
- Low decay (0.90): Adapt quickly, less trust in old observations

We learn the optimal decay by observing prediction errors:
- If predictions are accurate, posterior shifts toward higher decay
- If predictions are inaccurate, posterior shifts toward lower decay

This is a hierarchical Bayesian model:
- Level 1: Feature beliefs (Normal prior/posterior on alpha)
- Level 2: Decay belief (Beta prior/posterior on decay rate)

Usage:
    python 8b_bayesian_decay_learning.py

Output:
    data/backtest_results/bayesian_decay_learning_summary.csv
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

# N values to test
N_SATELLITES_TO_TEST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Training parameters
MIN_TRAINING_MONTHS = 36
REOPTIMIZATION_FREQUENCY = 1  # Re-optimize every N months

# Bayesian model parameters for FEATURE beliefs
PRIOR_STRENGTH_BASE = 50

# DECAY LEARNING CONFIGURATION
# ----------------------------
# These configure how we learn the decay rate using Bayesian inference

# Beta prior parameters for decay: Beta(alpha, beta)
# We model decay ~ Beta(alpha, beta), which has:
#   - Mean = alpha / (alpha + beta)
#   - Variance decreases as (alpha + beta) increases
DECAY_PRIOR_ALPHA = 10.0  # Prior pseudo-successes (higher decay)
DECAY_PRIOR_BETA = 2.0    # Prior pseudo-failures (lower decay)
# This prior has mean ~0.83, giving reasonable starting point

# Range for decay rate (Beta dist maps to this range)
DECAY_MIN = 0.85  # Minimum decay (fastest forgetting)
DECAY_MAX = 0.99  # Maximum decay (slowest forgetting)

# Decay update parameters
DECAY_UPDATE_WEIGHT = 1.0  # How much each observation updates decay belief
DECAY_LOOKBACK = 6        # Number of periods to consider for decay updates

# Error threshold for "success" in prediction
# If |predicted - actual| < threshold * std, it's a "success" (supports higher decay)
PREDICTION_SUCCESS_THRESHOLD = 1.0  # Within 1 std = success

# Feature selection parameters
MAX_ENSEMBLE_SIZE = 10
MIN_PROBABILITY_POSITIVE = 0.55

# Selection method
SELECTION_METHOD = 'greedy_bayesian'

# Greedy Bayesian parameters
GREEDY_CANDIDATES = 30
GREEDY_IMPROVEMENT_THRESHOLD = 0.001

# MC pre-filter
USE_MC_PREFILTER = True
MC_CONFIDENCE_LEVEL = 0.90

# Enhanced priors (from script 8)
USE_HITRATE_IN_PRIOR = True
USE_STABILITY_IN_PRIOR = True
HITRATE_PRIOR_WEIGHT = 0.3

# Data and output directories
DATA_DIR = Path(__file__).parent / 'data'
OUTPUT_DIR = DATA_DIR / 'backtest_results'


# ============================================================
# DECAY BELIEF - Beta distribution over decay rate
# ============================================================

@dataclass
class DecayBelief:
    """
    Bayesian belief about the optimal decay rate.

    We model decay using a Beta distribution, which is:
    - Conjugate prior for binomial likelihood
    - Naturally bounded between 0 and 1
    - Flexible shape (can be uniform, peaked, skewed)

    The decay rate controls how quickly we downweight old observations.
    We learn it by treating each prediction as a Bernoulli trial:
    - Success (accurate prediction) -> evidence for HIGHER decay (trust history)
    - Failure (inaccurate prediction) -> evidence for LOWER decay (adapt faster)

    Attributes:
        alpha: Beta parameter (pseudo-count of successes)
        beta: Beta parameter (pseudo-count of failures)
        decay_min: Minimum decay rate in range
        decay_max: Maximum decay rate in range
        prediction_history: List of (predicted, actual) pairs
    """
    alpha: float = DECAY_PRIOR_ALPHA
    beta: float = DECAY_PRIOR_BETA
    decay_min: float = DECAY_MIN
    decay_max: float = DECAY_MAX
    prediction_history: List[Tuple[float, float, float]] = field(default_factory=list)

    def mean_decay(self) -> float:
        """
        Expected decay rate under current belief.

        Beta mean = alpha / (alpha + beta), mapped to [decay_min, decay_max]
        """
        beta_mean = self.alpha / (self.alpha + self.beta)
        return self.decay_min + beta_mean * (self.decay_max - self.decay_min)

    def sample_decay(self) -> float:
        """
        Sample a decay rate from the posterior (for Thompson-style exploration).
        """
        beta_sample = np.random.beta(self.alpha, self.beta)
        return self.decay_min + beta_sample * (self.decay_max - self.decay_min)

    def mode_decay(self) -> float:
        """
        Most likely decay rate (MAP estimate).

        Beta mode = (alpha - 1) / (alpha + beta - 2) for alpha, beta > 1
        """
        if self.alpha > 1 and self.beta > 1:
            beta_mode = (self.alpha - 1) / (self.alpha + self.beta - 2)
        else:
            # Fallback to mean
            beta_mode = self.alpha / (self.alpha + self.beta)
        return self.decay_min + beta_mode * (self.decay_max - self.decay_min)

    def ci_decay(self, confidence: float = 0.90) -> Tuple[float, float]:
        """
        Credible interval for decay rate.
        """
        lower_q = (1 - confidence) / 2
        upper_q = 1 - lower_q

        beta_lower = stats.beta.ppf(lower_q, self.alpha, self.beta)
        beta_upper = stats.beta.ppf(upper_q, self.alpha, self.beta)

        decay_lower = self.decay_min + beta_lower * (self.decay_max - self.decay_min)
        decay_upper = self.decay_min + beta_upper * (self.decay_max - self.decay_min)

        return decay_lower, decay_upper

    def update_from_prediction(self, predicted_alpha: float, actual_alpha: float,
                                prediction_std: float):
        """
        Update decay belief based on a prediction-outcome pair.

        We treat prediction accuracy as a Bernoulli trial:
        - If |predicted - actual| < threshold * std: success (supports higher decay)
        - Otherwise: failure (supports lower decay)

        This updates the Beta posterior:
        - Success: alpha += weight
        - Failure: beta += weight

        The intuition is:
        - Accurate predictions suggest our historical beliefs are correct,
          so we should trust them more (higher decay = slower forgetting)
        - Inaccurate predictions suggest we need to adapt faster,
          so we should forget old data quicker (lower decay = faster forgetting)

        Args:
            predicted_alpha: What we predicted based on beliefs
            actual_alpha: What actually happened
            prediction_std: Standard deviation of prediction (uncertainty)
        """
        # Store for analysis
        self.prediction_history.append((predicted_alpha, actual_alpha, prediction_std))

        # Compute prediction error (normalized by uncertainty)
        if prediction_std > 0:
            normalized_error = abs(predicted_alpha - actual_alpha) / prediction_std
        else:
            normalized_error = abs(predicted_alpha - actual_alpha) / 0.01

        # Determine if this was a "success" (accurate prediction)
        is_success = normalized_error < PREDICTION_SUCCESS_THRESHOLD

        # Update Beta parameters
        # We use a soft update based on how accurate/inaccurate
        if is_success:
            # More accurate = stronger evidence for higher decay
            update_strength = DECAY_UPDATE_WEIGHT * (1 - normalized_error / PREDICTION_SUCCESS_THRESHOLD)
            self.alpha += max(0, update_strength)
        else:
            # More inaccurate = stronger evidence for lower decay
            excess_error = normalized_error - PREDICTION_SUCCESS_THRESHOLD
            update_strength = DECAY_UPDATE_WEIGHT * min(1.0, excess_error)
            self.beta += update_strength

    def get_accuracy_rate(self) -> float:
        """
        Compute accuracy rate from prediction history.
        """
        if len(self.prediction_history) == 0:
            return 0.5

        successes = 0
        for pred, actual, std in self.prediction_history:
            if std > 0:
                error = abs(pred - actual) / std
                if error < PREDICTION_SUCCESS_THRESHOLD:
                    successes += 1

        return successes / len(self.prediction_history)

    def reset_to_prior(self):
        """Reset to prior (for new period)."""
        self.alpha = DECAY_PRIOR_ALPHA
        self.beta = DECAY_PRIOR_BETA
        self.prediction_history = []


# ============================================================
# FEATURE BELIEF (same as script 8, but uses DecayBelief)
# ============================================================

@dataclass
class FeatureBelief:
    """
    Represents our belief about a feature's true alpha distribution.
    """
    mu: float = 0.0
    sigma: float = 0.05
    n_obs: float = 0.0
    sum_alpha: float = 0.0
    sum_sq: float = 0.0

    prior_mu: float = 0.0
    prior_sigma: float = 0.05
    prior_strength: float = 10.0

    def probability_positive(self) -> float:
        """P(true alpha > 0) given current beliefs."""
        if self.sigma <= 0:
            return 1.0 if self.mu > 0 else 0.0
        return 1 - stats.norm.cdf(-self.mu / self.sigma)

    def expected_sharpe(self) -> float:
        """Expected Sharpe ratio = mu / sigma."""
        if self.sigma <= 0:
            return 0.0
        return self.mu / self.sigma

    def sample(self) -> float:
        """Sample from the belief distribution."""
        return np.random.normal(self.mu, self.sigma)

    def ci_lower(self, confidence: float = 0.95) -> float:
        """Lower bound of confidence interval for mean."""
        if self.n_obs < 2:
            return -np.inf
        se = self.sigma / np.sqrt(max(1, self.n_obs))
        t_crit = stats.t.ppf(1 - confidence, df=max(1, self.n_obs - 1))
        return self.mu + t_crit * se


class BeliefState:
    """
    Maintains belief distributions for all features PLUS the decay hyperparameter.

    This is a hierarchical Bayesian model:
    - Feature beliefs: Normal(mu, sigma) for each feature's alpha
    - Decay belief: Beta(alpha, beta) for the decay rate hyperparameter
    """

    def __init__(self, n_features: int, feature_names: List[str]):
        self.n_features = n_features
        self.feature_names = feature_names
        self.beliefs: Dict[int, FeatureBelief] = {}

        # Initialize feature beliefs
        for i in range(n_features):
            self.beliefs[i] = FeatureBelief()

        # Initialize decay belief (the hyperparameter)
        self.decay_belief = DecayBelief()

    def get_current_decay(self) -> float:
        """Get the current estimated decay rate."""
        return self.decay_belief.mean_decay()

    def set_prior_from_mc(self, feat_idx: int, mc_mu: float, mc_sigma: float,
                          strength: float = PRIOR_STRENGTH_BASE):
        """Set prior belief from MC simulation results."""
        belief = self.beliefs[feat_idx]
        belief.prior_mu = mc_mu
        belief.prior_sigma = mc_sigma
        belief.prior_strength = strength

        belief.mu = mc_mu
        belief.sigma = mc_sigma
        belief.n_obs = strength
        belief.sum_alpha = mc_mu * strength
        belief.sum_sq = (mc_sigma**2 + mc_mu**2) * strength

    def update(self, feat_idx: int, observed_alpha: float, weight: float = 1.0):
        """
        Bayesian update after observing a new alpha value.

        Uses the LEARNED decay rate from decay_belief.
        """
        belief = self.beliefs[feat_idx]

        # Get current decay from learned posterior
        current_decay = self.get_current_decay()

        # Record prediction error for decay learning
        predicted = belief.mu
        prediction_std = belief.sigma

        # Decay old observations
        belief.n_obs *= current_decay
        belief.sum_alpha *= current_decay
        belief.sum_sq *= current_decay

        # Add new observation
        belief.n_obs += weight
        belief.sum_alpha += observed_alpha * weight
        belief.sum_sq += (observed_alpha ** 2) * weight

        # Update posterior mean and variance
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

        # Update decay belief based on prediction accuracy
        self.decay_belief.update_from_prediction(predicted, observed_alpha, prediction_std)

    def decay_all(self):
        """Apply decay to all beliefs."""
        current_decay = self.get_current_decay()
        for belief in self.beliefs.values():
            belief.n_obs *= current_decay
            belief.sum_alpha *= current_decay
            belief.sum_sq *= current_decay

    def get_feature_scores(self, method: str = 'expected_sharpe') -> np.ndarray:
        """Get scores for all features based on current beliefs."""
        scores = np.zeros(self.n_features)

        for i, belief in self.beliefs.items():
            if method == 'expected_sharpe':
                scores[i] = belief.expected_sharpe()
            elif method == 'probability_positive':
                scores[i] = belief.probability_positive()
            elif method == 'mean':
                scores[i] = belief.mu
            elif method == 'ci_lower':
                scores[i] = belief.ci_lower()
            else:
                scores[i] = belief.mu

        return scores

    def thompson_sample(self) -> np.ndarray:
        """Sample from all belief distributions."""
        samples = np.zeros(self.n_features)
        for i, belief in self.beliefs.items():
            samples[i] = belief.sample()
        return samples


# ============================================================
# NUMBA FUNCTIONS (same as script 8)
# ============================================================

@njit(cache=True)
def select_top_n_isins_by_scores(rankings_slice, feature_indices, feature_scores,
                                  n_satellites, alpha_valid_mask):
    """Select top-N ISINs based on feature ensemble with score weighting."""
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
    """Compute what alpha each feature would have achieved on a given date."""
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
    """Load precomputed data."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    horizon_label = f"{HOLDING_MONTHS}month"

    # Load forward alpha
    alpha_file = DATA_DIR / f'forward_alpha_{horizon_label}.parquet'
    if not alpha_file.exists():
        raise FileNotFoundError(f"Forward alpha file not found: {alpha_file}")

    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])

    # Load rankings
    rankings_file = DATA_DIR / f'rankings_matrix_{horizon_label}.npz'
    if not rankings_file.exists():
        raise FileNotFoundError(f"Rankings matrix not found: {rankings_file}")

    npz_data = np.load(rankings_file, allow_pickle=True)
    rankings = npz_data['rankings'].astype(np.float64)
    dates = pd.to_datetime(npz_data['dates'])
    isins = npz_data['isins']
    feature_names = list(npz_data['features'])

    # Load MC data for priors
    mc_file = DATA_DIR / f'mc_hitrates_{horizon_label}.npz'
    mc_data = None
    if mc_file.exists():
        mc_data = np.load(mc_file, allow_pickle=True)
        print(f"  [OK] MC data loaded for priors")
    else:
        print(f"  [WARN] No MC data - using uninformative priors")

    # Build alpha matrix
    n_dates = len(dates)
    n_isins = len(isins)

    isin_to_idx = {isin: idx for idx, isin in enumerate(isins)}
    date_to_idx = {date: idx for idx, date in enumerate(dates)}

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

    print(f"  [OK] {horizon_label}: {rankings.shape}")
    print(f"  Dates: {dates[0].date()} to {dates[-1].date()} ({len(dates)} months)")
    print(f"  ISINs: {len(isins)}")
    print(f"  Features: {len(feature_names)}")

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
    """Initialize feature beliefs from MC simulation data."""
    mc_data = data.get('mc_data')
    if mc_data is None:
        for i in range(belief_state.n_features):
            belief_state.set_prior_from_mc(i, mc_mu=0.0, mc_sigma=0.03,
                                           strength=PRIOR_STRENGTH_BASE)
        return

    mc_n_satellites = mc_data['n_satellites']
    n_to_idx = {int(n): i for i, n in enumerate(mc_n_satellites)}

    if n_satellites not in n_to_idx:
        for i in range(belief_state.n_features):
            belief_state.set_prior_from_mc(i, mc_mu=0.0, mc_sigma=0.03,
                                           strength=PRIOR_STRENGTH_BASE)
        return

    mc_idx = n_to_idx[n_satellites]
    mc_alpha_mean = mc_data['mc_alpha_mean'][mc_idx]
    mc_alpha_std = mc_data['mc_alpha_std'][mc_idx]
    mc_hitrates = mc_data['mc_hitrates'][mc_idx]
    mc_candidate_mask = mc_data['candidate_masks'][mc_idx]

    for feat_idx in range(belief_state.n_features):
        valid_dates = []
        for d in range(min(test_idx, mc_alpha_mean.shape[0])):
            if mc_candidate_mask[d, feat_idx]:
                mu = mc_alpha_mean[d, feat_idx]
                if not np.isnan(mu):
                    valid_dates.append(d)

        if len(valid_dates) > 0:
            mus = [mc_alpha_mean[d, feat_idx] for d in valid_dates]
            stds = [mc_alpha_std[d, feat_idx] for d in valid_dates]

            avg_alpha_mu = np.nanmean(mus)
            avg_alpha_std = np.nanmean(stds)

            if np.isnan(avg_alpha_mu):
                avg_alpha_mu = 0.0
            if np.isnan(avg_alpha_std) or avg_alpha_std <= 0:
                avg_alpha_std = 0.03

            if USE_HITRATE_IN_PRIOR:
                hrs = [mc_hitrates[d, feat_idx] for d in valid_dates]
                avg_hr = np.nanmean(hrs)
                if np.isnan(avg_hr):
                    avg_hr = 0.5

                hr_alpha_equiv = (avg_hr - 0.5) * 0.10
                combined_mu = (1 - HITRATE_PRIOR_WEIGHT) * avg_alpha_mu + \
                              HITRATE_PRIOR_WEIGHT * hr_alpha_equiv
            else:
                combined_mu = avg_alpha_mu

            if USE_STABILITY_IN_PRIOR and len(valid_dates) >= 3:
                alpha_cv = np.nanstd(mus) / (abs(np.nanmean(mus)) + 0.001)
                stability_factor = 1.0 + min(1.0, alpha_cv)
                adjusted_std = avg_alpha_std * stability_factor
            else:
                adjusted_std = avg_alpha_std

            belief_state.set_prior_from_mc(feat_idx, mc_mu=combined_mu,
                                           mc_sigma=adjusted_std,
                                           strength=PRIOR_STRENGTH_BASE)
        else:
            belief_state.set_prior_from_mc(feat_idx, mc_mu=0.0, mc_sigma=0.03,
                                           strength=PRIOR_STRENGTH_BASE * 0.5)


def update_beliefs_from_history(belief_state: BeliefState, data: dict,
                                 n_satellites: int, train_end_idx: int):
    """
    Update beliefs based on historical feature performance.

    This is where the decay learning happens:
    - As we process historical data, we update both:
      1. Feature beliefs (what we think each feature will do)
      2. Decay belief (how quickly we should forget old observations)

    The decay is learned from prediction errors: if our beliefs predict
    well, we keep higher decay (trust history); if not, we lower decay
    (adapt faster).
    """
    rankings = data['rankings']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']
    n_features = belief_state.n_features

    feature_indices = np.arange(n_features, dtype=np.int64)

    # Go through historical dates and update beliefs
    for date_idx in range(MIN_TRAINING_MONTHS, train_end_idx):
        # Compute what alpha each feature achieved on this date
        feature_alphas = compute_feature_alphas_for_date(
            rankings, feature_indices, alpha_matrix, alpha_valid,
            date_idx, n_satellites
        )

        # Get CURRENT decay rate (this may change as we update)
        current_decay = belief_state.get_current_decay()

        # Update beliefs
        for feat_idx in range(n_features):
            alpha = feature_alphas[feat_idx]
            if not np.isnan(alpha):
                # Weight by recency using LEARNED decay
                months_ago = train_end_idx - date_idx
                weight = current_decay ** months_ago

                # This update also updates the decay belief based on
                # how well we predicted this outcome
                belief_state.update(feat_idx, alpha, weight=weight)


def select_features_greedy_bayesian(belief_state: BeliefState,
                                     candidate_mask: np.ndarray) -> List[int]:
    """Greedy ensemble building with Bayesian scoring."""
    n_features = belief_state.n_features

    sharpe_scores = belief_state.get_feature_scores('expected_sharpe')
    prob_positive = belief_state.get_feature_scores('probability_positive')
    mean_alpha = belief_state.get_feature_scores('mean')

    valid_candidates = []
    for i in range(n_features):
        if candidate_mask[i] and prob_positive[i] >= MIN_PROBABILITY_POSITIVE:
            valid_candidates.append(i)

    if len(valid_candidates) == 0:
        return []

    candidate_scores = [(i, sharpe_scores[i], mean_alpha[i]) for i in valid_candidates]
    candidate_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    top_candidates = [x[0] for x in candidate_scores[:GREEDY_CANDIDATES]]

    if len(top_candidates) == 0:
        return []

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

        if best_candidate is not None and best_improvement > GREEDY_IMPROVEMENT_THRESHOLD:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break

    return selected


def compute_ensemble_expected_utility(feature_indices: List[int],
                                       belief_state: BeliefState) -> float:
    """Compute expected utility of a feature ensemble."""
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

    utility = ensemble_mu * ensemble_prob / ensemble_sigma
    return utility


def select_features_bayesian(belief_state: BeliefState,
                              method: str = 'expected_sharpe',
                              mc_passing_features: Optional[np.ndarray] = None) -> List[int]:
    """Select features based on current beliefs using specified method."""
    n_features = belief_state.n_features

    if mc_passing_features is not None and USE_MC_PREFILTER:
        candidate_mask = mc_passing_features
    else:
        candidate_mask = np.ones(n_features, dtype=bool)

    if method == 'greedy_bayesian':
        return select_features_greedy_bayesian(belief_state, candidate_mask)

    elif method == 'expected_sharpe':
        scores = belief_state.get_feature_scores('expected_sharpe')
        scores[~candidate_mask] = -np.inf
        selected_indices = np.argsort(scores)[::-1]

    elif method == 'probability_weighted':
        scores = belief_state.get_feature_scores('probability_positive')
        scores[~candidate_mask] = -np.inf
        selected_indices = np.argsort(scores)[::-1]

    else:
        scores = belief_state.get_feature_scores('mean')
        scores[~candidate_mask] = -np.inf
        selected_indices = np.argsort(scores)[::-1]

    selected = []
    for idx in selected_indices:
        if not candidate_mask[idx]:
            continue
        belief = belief_state.beliefs[idx]
        if belief.probability_positive() >= MIN_PROBABILITY_POSITIVE:
            selected.append(int(idx))
            if len(selected) >= MAX_ENSEMBLE_SIZE:
                break

    return selected


def get_mc_passing_features(data: dict, n_satellites: int, test_idx: int) -> Optional[np.ndarray]:
    """Get features that pass MC filter for this date."""
    mc_data = data.get('mc_data')
    if mc_data is None:
        return None

    mc_n_satellites = mc_data['n_satellites']
    n_to_idx = {int(n): i for i, n in enumerate(mc_n_satellites)}

    if n_satellites not in n_to_idx:
        return None

    mc_idx = n_to_idx[n_satellites]
    mc_hitrates = mc_data['mc_hitrates'][mc_idx]
    mc_alpha_mean = mc_data['mc_alpha_mean'][mc_idx]
    mc_alpha_std = mc_data['mc_alpha_std'][mc_idx]
    mc_candidate_mask = mc_data['candidate_masks'][mc_idx]

    mc_config = mc_data.get('config', None)
    if mc_config is not None:
        mc_config = mc_config.item() if hasattr(mc_config, 'item') else mc_config
        mc_n_samples = mc_config.get('n_simulations', 1000)
    else:
        mc_n_samples = 1000

    if test_idx >= mc_hitrates.shape[0]:
        return None

    n_features = mc_hitrates.shape[1]
    passing_mask = np.zeros(n_features, dtype=bool)

    date_candidates = mc_candidate_mask[test_idx]
    date_hr_values = mc_hitrates[test_idx, date_candidates]
    date_alpha_values = mc_alpha_mean[test_idx, date_candidates]

    valid_hr = date_hr_values[~np.isnan(date_hr_values)]
    valid_alpha = date_alpha_values[~np.isnan(date_alpha_values)]

    if len(valid_hr) == 0 or len(valid_alpha) == 0:
        return None

    baseline_hr = np.mean(valid_hr)
    baseline_alpha = np.mean(valid_alpha)

    t_crit = stats.t.ppf(MC_CONFIDENCE_LEVEL, df=max(1, mc_n_samples - 1))

    for feat_idx in range(n_features):
        if not date_candidates[feat_idx]:
            continue

        hr = mc_hitrates[test_idx, feat_idx]
        alpha_mu = mc_alpha_mean[test_idx, feat_idx]
        alpha_std = mc_alpha_std[test_idx, feat_idx]

        if np.isnan(hr) or np.isnan(alpha_mu):
            continue

        hr_se = np.sqrt(hr * (1 - hr) / mc_n_samples) if 0 < hr < 1 else 0
        alpha_se = alpha_std / np.sqrt(mc_n_samples) if alpha_std > 0 else 0

        hr_ci_lower = hr - t_crit * hr_se
        alpha_ci_lower = alpha_mu - t_crit * alpha_se

        hr_significant = hr_ci_lower > baseline_hr
        alpha_significant = alpha_ci_lower > baseline_alpha

        if hr_significant or alpha_significant:
            passing_mask[feat_idx] = True

    return passing_mask


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================

def walk_forward_backtest(data: dict, n_satellites: int,
                          show_progress: bool = True) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Run walk-forward backtest with Bayesian feature selection and decay learning.

    Returns both the results DataFrame and a list of decay learning diagnostics.
    """
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
        print(f"  Selection method: {SELECTION_METHOD}")
        print(f"  Decay prior: Beta({DECAY_PRIOR_ALPHA}, {DECAY_PRIOR_BETA})")
        print(f"  Decay range: [{DECAY_MIN}, {DECAY_MAX}]")

    results = []
    decay_diagnostics = []
    belief_state = None
    months_since_reopt = REOPTIMIZATION_FREQUENCY
    selected_features = None

    iterator = range(test_start_idx, len(dates))
    if show_progress:
        iterator = tqdm(iterator, desc=f"N={n_satellites}")

    for test_idx in iterator:
        test_date = dates[test_idx]

        mc_passing = None
        if USE_MC_PREFILTER:
            mc_passing = get_mc_passing_features(data, n_satellites, test_idx)

        if months_since_reopt >= REOPTIMIZATION_FREQUENCY:
            belief_state = BeliefState(n_features, feature_names)

            initialize_beliefs_from_mc(belief_state, data, n_satellites, test_idx)
            update_beliefs_from_history(belief_state, data, n_satellites, test_idx)

            selected_features = select_features_bayesian(
                belief_state, method=SELECTION_METHOD,
                mc_passing_features=mc_passing
            )

            months_since_reopt = 0

        months_since_reopt += 1

        if not selected_features or len(selected_features) == 0:
            continue

        feature_scores = np.array([
            belief_state.beliefs[f].expected_sharpe()
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

        # Get current decay state for diagnostics
        current_decay = belief_state.get_current_decay()
        decay_ci = belief_state.decay_belief.ci_decay()

        results.append({
            'date': test_date,
            'n_satellites': n_satellites,
            'avg_alpha': avg_alpha,
            'n_selected': len(selected_isin_names),
            'n_features': len(selected_features),
            'learned_decay': current_decay,
            'decay_ci_lower': decay_ci[0],
            'decay_ci_upper': decay_ci[1],
            'selected_isins': ','.join(selected_isin_names)
        })

        # Record decay diagnostics
        decay_diagnostics.append({
            'date': test_date,
            'decay_alpha': belief_state.decay_belief.alpha,
            'decay_beta': belief_state.decay_belief.beta,
            'learned_decay': current_decay,
            'decay_ci_lower': decay_ci[0],
            'decay_ci_upper': decay_ci[1],
            'n_predictions': len(belief_state.decay_belief.prediction_history),
            'accuracy_rate': belief_state.decay_belief.get_accuracy_rate()
        })

        # Update beliefs with observed outcome
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

    return pd.DataFrame(results), decay_diagnostics


# ============================================================
# ANALYSIS
# ============================================================

def analyze_results(results_df: pd.DataFrame, n_satellites: int,
                    decay_diagnostics: List[dict]) -> Optional[dict]:
    """Analyze backtest results including decay learning."""
    if len(results_df) == 0:
        return None

    avg_alpha = results_df['avg_alpha'].mean()
    std_alpha = results_df['avg_alpha'].std()
    hit_rate = (results_df['avg_alpha'] > 0).mean()
    sharpe = avg_alpha / std_alpha if std_alpha > 0 else 0

    results_df = results_df.copy()
    results_df['cumulative'] = (1 + results_df['avg_alpha']).cumprod() - 1
    total_return = results_df['cumulative'].iloc[-1]

    # Analyze decay learning
    avg_learned_decay = results_df['learned_decay'].mean()
    final_decay = results_df['learned_decay'].iloc[-1]

    decay_df = pd.DataFrame(decay_diagnostics)
    if len(decay_df) > 0:
        avg_accuracy = decay_df['accuracy_rate'].mean()
        final_accuracy = decay_df['accuracy_rate'].iloc[-1]
    else:
        avg_accuracy = 0.5
        final_accuracy = 0.5

    return {
        'n_satellites': n_satellites,
        'selection_method': SELECTION_METHOD,
        'decay_prior_alpha': DECAY_PRIOR_ALPHA,
        'decay_prior_beta': DECAY_PRIOR_BETA,
        'decay_range': f"[{DECAY_MIN}, {DECAY_MAX}]",
        'n_periods': len(results_df),
        'avg_alpha': avg_alpha,
        'std_alpha': std_alpha,
        'annual_alpha': avg_alpha * 12,
        'hit_rate': hit_rate,
        'sharpe': sharpe,
        'total_return': total_return,
        # Decay learning stats
        'avg_learned_decay': avg_learned_decay,
        'final_learned_decay': final_decay,
        'avg_prediction_accuracy': avg_accuracy,
        'final_prediction_accuracy': final_accuracy
    }


def print_decay_evolution(decay_diagnostics: List[dict], n_satellites: int):
    """Print how the decay rate evolved during backtest."""
    if len(decay_diagnostics) == 0:
        return

    df = pd.DataFrame(decay_diagnostics)

    print(f"\n  Decay Learning Evolution (N={n_satellites}):")
    print(f"  " + "-" * 70)
    print(f"  {'Date':<12} {'Decay':>8} {'CI':>18} {'Acc Rate':>10} {'Preds':>8}")
    print(f"  " + "-" * 70)

    # Sample every 12 periods (yearly) plus first and last
    indices = [0]
    for i in range(12, len(df), 12):
        indices.append(i)
    if len(df) - 1 not in indices:
        indices.append(len(df) - 1)

    for idx in indices:
        row = df.iloc[idx]
        date_str = row['date'].strftime('%Y-%m')
        decay = row['learned_decay']
        ci_str = f"[{row['decay_ci_lower']:.3f}, {row['decay_ci_upper']:.3f}]"
        acc = row['accuracy_rate']
        n_preds = row['n_predictions']

        print(f"  {date_str:<12} {decay:>8.4f} {ci_str:>18} {acc:>9.1%} {n_preds:>8}")

    print(f"  " + "-" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    """Run Bayesian walk-forward backtest with decay learning."""
    print("=" * 60)
    print("BAYESIAN DECAY LEARNING STRATEGY")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Holding period: {HOLDING_MONTHS} month")
    print(f"  N values to test: {N_SATELLITES_TO_TEST}")
    print(f"\nDecay Learning Parameters:")
    print(f"  Prior: Beta({DECAY_PRIOR_ALPHA}, {DECAY_PRIOR_BETA})")
    prior_mean = DECAY_PRIOR_ALPHA / (DECAY_PRIOR_ALPHA + DECAY_PRIOR_BETA)
    print(f"  Prior mean: {DECAY_MIN + prior_mean * (DECAY_MAX - DECAY_MIN):.4f}")
    print(f"  Decay range: [{DECAY_MIN}, {DECAY_MAX}]")
    print(f"  Success threshold: {PREDICTION_SUCCESS_THRESHOLD} std")
    print(f"  Update weight: {DECAY_UPDATE_WEIGHT}")
    print(f"\nFeature Selection:")
    print(f"  Method: {SELECTION_METHOD}")
    print(f"  Prior strength: {PRIOR_STRENGTH_BASE}")
    print(f"  Use hitrate in prior: {USE_HITRATE_IN_PRIOR}")
    print(f"  Use stability: {USE_STABILITY_IN_PRIOR}")
    print(f"  MC pre-filter: {USE_MC_PREFILTER}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    all_stats = []
    all_results = {}
    all_decay_diagnostics = {}

    start_time = time.time()

    for test_num, n in enumerate(N_SATELLITES_TO_TEST, 1):
        print(f"\n{'='*60}")
        print(f"TEST {test_num}/{len(N_SATELLITES_TO_TEST)}: N={n}")
        print(f"{'='*60}")

        results_df, decay_diagnostics = walk_forward_backtest(data, n, show_progress=True)

        if len(results_df) > 0:
            all_results[n] = results_df
            all_decay_diagnostics[n] = decay_diagnostics

            stats = analyze_results(results_df, n, decay_diagnostics)
            if stats:
                all_stats.append(stats)

                print(f"\n  Results:")
                print(f"    Periods: {stats['n_periods']}")
                print(f"    Monthly Alpha: {stats['avg_alpha']*100:.2f}%")
                print(f"    Annual Alpha: {stats['annual_alpha']*100:.1f}%")
                print(f"    Hit Rate: {stats['hit_rate']:.2%}")
                print(f"    Sharpe: {stats['sharpe']:.3f}")
                print(f"\n  Decay Learning:")
                print(f"    Avg Learned Decay: {stats['avg_learned_decay']:.4f}")
                print(f"    Final Learned Decay: {stats['final_learned_decay']:.4f}")
                print(f"    Avg Prediction Accuracy: {stats['avg_prediction_accuracy']:.1%}")

                # Print decay evolution
                print_decay_evolution(decay_diagnostics, n)

                # Save individual results
                output_file = OUTPUT_DIR / f'bayesian_decay_learning_N{n}.csv'
                results_df.to_csv(output_file, index=False)
        else:
            print(f"  No results")

    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - BAYESIAN DECAY LEARNING")
    print("=" * 60)
    print(f"\nDecay Prior: Beta({DECAY_PRIOR_ALPHA}, {DECAY_PRIOR_BETA})")
    print(f"Decay Range: [{DECAY_MIN}, {DECAY_MAX}]")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if len(all_stats) > 0:
        summary_df = pd.DataFrame(all_stats)

        print("\n" + "-" * 100)
        print(f"{'N':>3} {'Periods':>8} {'Monthly':>10} {'Annual':>10} {'Hit Rate':>10} "
              f"{'Sharpe':>8} {'Avg Decay':>10} {'Final Decay':>12}")
        print("-" * 100)

        for _, row in summary_df.sort_values('n_satellites').iterrows():
            print(f"{int(row['n_satellites']):>3} {int(row['n_periods']):>8} "
                  f"{row['avg_alpha']*100:>9.2f}% {row['annual_alpha']*100:>9.1f}% "
                  f"{row['hit_rate']:>9.1%} {row['sharpe']:>8.3f} "
                  f"{row['avg_learned_decay']:>10.4f} {row['final_learned_decay']:>12.4f}")

        print("-" * 100)

        # Find best
        best_sharpe = summary_df.loc[summary_df['sharpe'].idxmax()]
        best_alpha = summary_df.loc[summary_df['annual_alpha'].idxmax()]
        best_hit = summary_df.loc[summary_df['hit_rate'].idxmax()]

        print(f"\nBest by Sharpe:     N={int(best_sharpe['n_satellites'])}, "
              f"Alpha={best_sharpe['annual_alpha']*100:.1f}%/yr, "
              f"Hit={best_sharpe['hit_rate']:.1%}, "
              f"Sharpe={best_sharpe['sharpe']:.3f}, "
              f"Final Decay={best_sharpe['final_learned_decay']:.4f}")

        print(f"Best by Alpha:      N={int(best_alpha['n_satellites'])}, "
              f"Alpha={best_alpha['annual_alpha']*100:.1f}%/yr, "
              f"Hit={best_alpha['hit_rate']:.1%}, "
              f"Sharpe={best_alpha['sharpe']:.3f}, "
              f"Final Decay={best_alpha['final_learned_decay']:.4f}")

        print(f"Best by Hit Rate:   N={int(best_hit['n_satellites'])}, "
              f"Alpha={best_hit['annual_alpha']*100:.1f}%/yr, "
              f"Hit={best_hit['hit_rate']:.1%}, "
              f"Sharpe={best_hit['sharpe']:.3f}, "
              f"Final Decay={best_hit['final_learned_decay']:.4f}")

        # Save summary
        summary_file = OUTPUT_DIR / 'bayesian_decay_learning_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\n[SAVED] {summary_file}")


if __name__ == '__main__':
    main()
