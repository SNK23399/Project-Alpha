"""
Step 8c: Bayesian Multi-Hyperparameter Learning Strategy
=========================================================

Extends script 8b with Bayesian learning of MULTIPLE hyperparameters:

1. DECAY RATE (from 8b) - Beta prior
   - Controls how quickly we forget old observations
   - Learned from prediction accuracy

2. PRIOR STRENGTH - Gamma prior
   - Controls how much we trust MC priors vs observed data
   - Learned from prior prediction accuracy

3. ALPHA-HITRATE WEIGHT - Beta prior
   - Controls balance between alpha and hit rate in combined prior
   - Learned from which metric better predicts performance

Each hyperparameter has its own belief distribution that updates based
on how well that hyperparameter's current value predicts outcomes.

This is a hierarchical Bayesian model:
- Level 1: Feature beliefs (Normal prior/posterior on alpha)
- Level 2: Multiple hyperparameter beliefs (various distributions)

Usage:
    python 8c_bayesian_multi_hyperparameter.py

Output:
    data/backtest_results/bayesian_multi_hp_summary.csv
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
REOPTIMIZATION_FREQUENCY = 1

# ============================================================
# HYPERPARAMETER LEARNING CONFIGURATION
# ============================================================

# 1. DECAY RATE (Beta distribution, maps to [DECAY_MIN, DECAY_MAX])
DECAY_PRIOR_ALPHA = 10.0
DECAY_PRIOR_BETA = 2.0
DECAY_MIN = 0.85
DECAY_MAX = 0.99

# 2. PRIOR STRENGTH (Gamma distribution, maps to [PRIOR_MIN, PRIOR_MAX])
#    Gamma(shape, scale) has mean = shape * scale
PRIOR_STRENGTH_SHAPE = 5.0   # Shape parameter
PRIOR_STRENGTH_SCALE = 10.0  # Scale parameter (mean = 50)
PRIOR_STRENGTH_MIN = 10.0    # Minimum prior strength
PRIOR_STRENGTH_MAX = 200.0   # Maximum prior strength

# 3. ALPHA-HITRATE WEIGHT (Beta distribution, maps to [0, 1])
#    Weight for alpha in combined prior; hitrate gets (1 - weight)
ALPHA_WEIGHT_PRIOR_ALPHA = 7.0   # Slight preference for alpha
ALPHA_WEIGHT_PRIOR_BETA = 3.0    # Prior mean ~0.7

# Common update parameters
HP_UPDATE_WEIGHT = 1.0        # How much each observation updates beliefs
PREDICTION_SUCCESS_THRESHOLD = 1.0  # Within 1 std = success

# Feature selection parameters
MAX_ENSEMBLE_SIZE = 10
MIN_PROBABILITY_POSITIVE = 0.55
SELECTION_METHOD = 'greedy_bayesian'
GREEDY_CANDIDATES = 30
GREEDY_IMPROVEMENT_THRESHOLD = 0.001

# MC pre-filter
USE_MC_PREFILTER = True
MC_CONFIDENCE_LEVEL = 0.90

# Enhanced priors
USE_HITRATE_IN_PRIOR = True
USE_STABILITY_IN_PRIOR = True

# Data and output directories
DATA_DIR = Path(__file__).parent / 'data'
OUTPUT_DIR = DATA_DIR / 'backtest_results'


# ============================================================
# HYPERPARAMETER BELIEFS
# ============================================================

@dataclass
class DecayBelief:
    """
    Beta belief over decay rate.
    Maps Beta(alpha, beta) to [decay_min, decay_max] range.
    """
    alpha: float = DECAY_PRIOR_ALPHA
    beta: float = DECAY_PRIOR_BETA
    decay_min: float = DECAY_MIN
    decay_max: float = DECAY_MAX
    prediction_history: List[Tuple[float, float, float]] = field(default_factory=list)

    def mean(self) -> float:
        """Expected decay rate."""
        beta_mean = self.alpha / (self.alpha + self.beta)
        return self.decay_min + beta_mean * (self.decay_max - self.decay_min)

    def sample(self) -> float:
        """Sample from posterior."""
        beta_sample = np.random.beta(self.alpha, self.beta)
        return self.decay_min + beta_sample * (self.decay_max - self.decay_min)

    def ci(self, confidence: float = 0.90) -> Tuple[float, float]:
        """Credible interval."""
        lower_q = (1 - confidence) / 2
        upper_q = 1 - lower_q
        beta_lower = stats.beta.ppf(lower_q, self.alpha, self.beta)
        beta_upper = stats.beta.ppf(upper_q, self.alpha, self.beta)
        return (self.decay_min + beta_lower * (self.decay_max - self.decay_min),
                self.decay_min + beta_upper * (self.decay_max - self.decay_min))

    def update(self, predicted: float, actual: float, std: float):
        """Update based on prediction accuracy."""
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
    """
    Gamma belief over prior strength.

    Prior strength controls how much we trust MC priors vs observed data.
    Higher strength = trust priors more (slower adaptation to new data).
    Lower strength = trust observations more (faster adaptation).

    We learn this by tracking how well priors predict outcomes:
    - If priors are accurate → increase strength (trust them more)
    - If priors are inaccurate → decrease strength (rely on observations)

    Uses Gamma distribution because:
    - Naturally positive (strength must be > 0)
    - Flexible shape (can be peaked or spread)
    - Conjugate for rate parameters
    """
    shape: float = PRIOR_STRENGTH_SHAPE
    rate: float = 1.0 / PRIOR_STRENGTH_SCALE  # Gamma uses rate = 1/scale
    prior_min: float = PRIOR_STRENGTH_MIN
    prior_max: float = PRIOR_STRENGTH_MAX
    prediction_history: List[Tuple[float, float, float]] = field(default_factory=list)

    def mean(self) -> float:
        """Expected prior strength."""
        gamma_mean = self.shape / self.rate  # = shape * scale
        # Map to [prior_min, prior_max] using sigmoid-like transform
        # Gamma mean can be anywhere from 0 to inf, we squash to range
        normalized = gamma_mean / (gamma_mean + 50)  # 50 is midpoint
        return self.prior_min + normalized * (self.prior_max - self.prior_min)

    def sample(self) -> float:
        """Sample from posterior."""
        gamma_sample = np.random.gamma(self.shape, 1.0 / self.rate)
        normalized = gamma_sample / (gamma_sample + 50)
        return self.prior_min + normalized * (self.prior_max - self.prior_min)

    def ci(self, confidence: float = 0.90) -> Tuple[float, float]:
        """Credible interval."""
        lower_q = (1 - confidence) / 2
        upper_q = 1 - lower_q
        gamma_lower = stats.gamma.ppf(lower_q, self.shape, scale=1.0/self.rate)
        gamma_upper = stats.gamma.ppf(upper_q, self.shape, scale=1.0/self.rate)

        norm_lower = gamma_lower / (gamma_lower + 50)
        norm_upper = gamma_upper / (gamma_upper + 50)

        return (self.prior_min + norm_lower * (self.prior_max - self.prior_min),
                self.prior_min + norm_upper * (self.prior_max - self.prior_min))

    def update(self, prior_predicted: float, actual: float, std: float):
        """
        Update based on how well the PRIOR predicted the outcome.

        This is different from decay: we specifically track whether
        the MC prior (before any observation updates) was accurate.

        If prior was accurate → increase shape (higher mean strength)
        If prior was inaccurate → increase rate (lower mean strength)
        """
        self.prediction_history.append((prior_predicted, actual, std))

        if std > 0:
            normalized_error = abs(prior_predicted - actual) / std
        else:
            normalized_error = abs(prior_predicted - actual) / 0.01

        is_success = normalized_error < PREDICTION_SUCCESS_THRESHOLD

        if is_success:
            # Prior was accurate - increase strength (increase shape)
            update = HP_UPDATE_WEIGHT * (1 - normalized_error / PREDICTION_SUCCESS_THRESHOLD)
            self.shape += max(0, update * 0.5)  # Scale down to prevent explosion
        else:
            # Prior was inaccurate - decrease strength (increase rate)
            excess_error = normalized_error - PREDICTION_SUCCESS_THRESHOLD
            update = HP_UPDATE_WEIGHT * min(1.0, excess_error)
            self.rate += update * 0.1  # Small rate increases


@dataclass
class AlphaWeightBelief:
    """
    Beta belief over alpha-hitrate weight.

    This weight controls the balance between alpha and hit rate
    when combining them into the prior belief.

    weight = 1.0: Only use alpha
    weight = 0.0: Only use hit rate
    weight = 0.7: 70% alpha, 30% hit rate (default)

    We learn this by tracking which metric better predicts outcomes:
    - If alpha-heavy priors are more accurate → increase weight
    - If hitrate-heavy priors are more accurate → decrease weight
    """
    alpha: float = ALPHA_WEIGHT_PRIOR_ALPHA
    beta: float = ALPHA_WEIGHT_PRIOR_BETA

    # Track separate predictions to determine which is better
    alpha_predictions: List[Tuple[float, float]] = field(default_factory=list)
    hitrate_predictions: List[Tuple[float, float]] = field(default_factory=list)

    def mean(self) -> float:
        """Expected alpha weight (0 to 1)."""
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        """Sample from posterior."""
        return np.random.beta(self.alpha, self.beta)

    def ci(self, confidence: float = 0.90) -> Tuple[float, float]:
        """Credible interval."""
        lower_q = (1 - confidence) / 2
        upper_q = 1 - lower_q
        return (stats.beta.ppf(lower_q, self.alpha, self.beta),
                stats.beta.ppf(upper_q, self.alpha, self.beta))

    def update(self, alpha_pred: float, hr_pred: float, actual: float):
        """
        Update based on which prediction was more accurate.

        We compare alpha-based and hitrate-based predictions:
        - If alpha prediction closer → evidence for higher weight
        - If hitrate prediction closer → evidence for lower weight
        """
        self.alpha_predictions.append((alpha_pred, actual))
        self.hitrate_predictions.append((hr_pred, actual))

        alpha_error = abs(alpha_pred - actual)
        hr_error = abs(hr_pred - actual)

        # Normalize to get relative accuracy
        total_error = alpha_error + hr_error
        if total_error > 0:
            alpha_accuracy = 1 - (alpha_error / total_error)  # 0 to 1
            hr_accuracy = 1 - (hr_error / total_error)        # 0 to 1

            # Update Beta parameters
            # More accurate metric gets boost
            self.alpha += HP_UPDATE_WEIGHT * alpha_accuracy * 0.5
            self.beta += HP_UPDATE_WEIGHT * hr_accuracy * 0.5


# ============================================================
# FEATURE BELIEF (updated to use learned hyperparameters)
# ============================================================

@dataclass
class FeatureBelief:
    """Belief about a feature's true alpha distribution."""
    mu: float = 0.0
    sigma: float = 0.05
    n_obs: float = 0.0
    sum_alpha: float = 0.0
    sum_sq: float = 0.0

    prior_mu: float = 0.0
    prior_sigma: float = 0.05
    prior_strength: float = 50.0

    # Store component priors for alpha weight learning
    alpha_prior_mu: float = 0.0    # Prior from alpha only
    hitrate_prior_mu: float = 0.0  # Prior from hitrate only

    def probability_positive(self) -> float:
        if self.sigma <= 0:
            return 1.0 if self.mu > 0 else 0.0
        return 1 - stats.norm.cdf(-self.mu / self.sigma)

    def expected_sharpe(self) -> float:
        if self.sigma <= 0:
            return 0.0
        return self.mu / self.sigma

    def sample(self) -> float:
        return np.random.normal(self.mu, self.sigma)


class BeliefState:
    """
    Maintains beliefs for features AND multiple hyperparameters.

    Hyperparameters learned:
    1. decay_belief: How quickly to forget old observations
    2. prior_strength_belief: How much to trust MC priors
    3. alpha_weight_belief: Balance between alpha and hitrate
    """

    def __init__(self, n_features: int, feature_names: List[str]):
        self.n_features = n_features
        self.feature_names = feature_names
        self.beliefs: Dict[int, FeatureBelief] = {}

        for i in range(n_features):
            self.beliefs[i] = FeatureBelief()

        # Initialize hyperparameter beliefs
        self.decay_belief = DecayBelief()
        self.prior_strength_belief = PriorStrengthBelief()
        self.alpha_weight_belief = AlphaWeightBelief()

    def get_decay(self) -> float:
        """Get current decay rate."""
        return self.decay_belief.mean()

    def get_prior_strength(self) -> float:
        """Get current prior strength."""
        return self.prior_strength_belief.mean()

    def get_alpha_weight(self) -> float:
        """Get current alpha weight."""
        return self.alpha_weight_belief.mean()

    def set_prior_from_mc(self, feat_idx: int, alpha_mu: float, hitrate_mu: float,
                          mc_sigma: float):
        """
        Set prior using LEARNED alpha weight.

        Combines alpha and hitrate priors using the learned weight.
        """
        belief = self.beliefs[feat_idx]

        # Store component priors for later learning
        belief.alpha_prior_mu = alpha_mu
        belief.hitrate_prior_mu = hitrate_mu

        # Get current learned hyperparameters
        alpha_weight = self.get_alpha_weight()
        prior_strength = self.get_prior_strength()

        # Combine using learned weight
        combined_mu = alpha_weight * alpha_mu + (1 - alpha_weight) * hitrate_mu

        belief.prior_mu = combined_mu
        belief.prior_sigma = mc_sigma
        belief.prior_strength = prior_strength

        # Initialize posterior to prior
        belief.mu = combined_mu
        belief.sigma = mc_sigma
        belief.n_obs = prior_strength
        belief.sum_alpha = combined_mu * prior_strength
        belief.sum_sq = (mc_sigma**2 + combined_mu**2) * prior_strength

    def update(self, feat_idx: int, observed_alpha: float, weight: float = 1.0):
        """
        Update feature belief and all hyperparameter beliefs.
        """
        belief = self.beliefs[feat_idx]

        # Get current hyperparameters
        current_decay = self.get_decay()

        # Record predictions for hyperparameter learning
        posterior_predicted = belief.mu
        prior_predicted = belief.prior_mu
        prediction_std = belief.sigma

        # Decay old observations using LEARNED decay
        belief.n_obs *= current_decay
        belief.sum_alpha *= current_decay
        belief.sum_sq *= current_decay

        # Add new observation
        belief.n_obs += weight
        belief.sum_alpha += observed_alpha * weight
        belief.sum_sq += (observed_alpha ** 2) * weight

        # Update posterior
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

        # Update hyperparameter beliefs
        # 1. Decay: based on posterior prediction accuracy
        self.decay_belief.update(posterior_predicted, observed_alpha, prediction_std)

        # 2. Prior strength: based on prior prediction accuracy
        self.prior_strength_belief.update(prior_predicted, observed_alpha, prediction_std)

        # 3. Alpha weight: based on which component predicted better
        self.alpha_weight_belief.update(
            belief.alpha_prior_mu,
            belief.hitrate_prior_mu,
            observed_alpha
        )

    def get_feature_scores(self, method: str = 'expected_sharpe') -> np.ndarray:
        scores = np.zeros(self.n_features)
        for i, belief in self.beliefs.items():
            if method == 'expected_sharpe':
                scores[i] = belief.expected_sharpe()
            elif method == 'probability_positive':
                scores[i] = belief.probability_positive()
            elif method == 'mean':
                scores[i] = belief.mu
            else:
                scores[i] = belief.mu
        return scores


# ============================================================
# NUMBA FUNCTIONS (same as 8b)
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
    """Load precomputed data."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    horizon_label = f"{HOLDING_MONTHS}month"

    alpha_file = DATA_DIR / f'forward_alpha_{horizon_label}.parquet'
    if not alpha_file.exists():
        raise FileNotFoundError(f"Forward alpha file not found: {alpha_file}")

    alpha_df = pd.read_parquet(alpha_file)
    alpha_df['date'] = pd.to_datetime(alpha_df['date'])

    rankings_file = DATA_DIR / f'rankings_matrix_{horizon_label}.npz'
    if not rankings_file.exists():
        raise FileNotFoundError(f"Rankings matrix not found: {rankings_file}")

    npz_data = np.load(rankings_file, allow_pickle=True)
    rankings = npz_data['rankings'].astype(np.float64)
    dates = pd.to_datetime(npz_data['dates'])
    isins = npz_data['isins']
    feature_names = list(npz_data['features'])

    mc_file = DATA_DIR / f'mc_hitrates_{horizon_label}.npz'
    mc_data = None
    if mc_file.exists():
        mc_data = np.load(mc_file, allow_pickle=True)
        print(f"  [OK] MC data loaded for priors")
    else:
        print(f"  [WARN] No MC data - using uninformative priors")

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
    """Initialize feature beliefs with LEARNED hyperparameters."""
    mc_data = data.get('mc_data')
    if mc_data is None:
        for i in range(belief_state.n_features):
            belief_state.set_prior_from_mc(i, alpha_mu=0.0, hitrate_mu=0.0,
                                           mc_sigma=0.03)
        return

    mc_n_satellites = mc_data['n_satellites']
    n_to_idx = {int(n): i for i, n in enumerate(mc_n_satellites)}

    if n_satellites not in n_to_idx:
        for i in range(belief_state.n_features):
            belief_state.set_prior_from_mc(i, alpha_mu=0.0, hitrate_mu=0.0,
                                           mc_sigma=0.03)
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
            # Get alpha prior
            mus = [mc_alpha_mean[d, feat_idx] for d in valid_dates]
            stds = [mc_alpha_std[d, feat_idx] for d in valid_dates]
            avg_alpha_mu = np.nanmean(mus)
            avg_alpha_std = np.nanmean(stds)

            if np.isnan(avg_alpha_mu):
                avg_alpha_mu = 0.0
            if np.isnan(avg_alpha_std) or avg_alpha_std <= 0:
                avg_alpha_std = 0.03

            # Get hitrate prior (convert to alpha equivalent)
            hrs = [mc_hitrates[d, feat_idx] for d in valid_dates]
            avg_hr = np.nanmean(hrs)
            if np.isnan(avg_hr):
                avg_hr = 0.5
            hr_alpha_equiv = (avg_hr - 0.5) * 0.10

            # Apply stability penalty
            if USE_STABILITY_IN_PRIOR and len(valid_dates) >= 3:
                alpha_cv = np.nanstd(mus) / (abs(np.nanmean(mus)) + 0.001)
                stability_factor = 1.0 + min(1.0, alpha_cv)
                adjusted_std = avg_alpha_std * stability_factor
            else:
                adjusted_std = avg_alpha_std

            # Set prior using learned hyperparameters
            belief_state.set_prior_from_mc(
                feat_idx,
                alpha_mu=avg_alpha_mu,
                hitrate_mu=hr_alpha_equiv,
                mc_sigma=adjusted_std
            )
        else:
            belief_state.set_prior_from_mc(
                feat_idx, alpha_mu=0.0, hitrate_mu=0.0, mc_sigma=0.03
            )


def update_beliefs_from_history(belief_state: BeliefState, data: dict,
                                 n_satellites: int, train_end_idx: int):
    """Update beliefs from historical data, learning all hyperparameters."""
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
                              method: str = 'expected_sharpe',
                              mc_passing_features: Optional[np.ndarray] = None) -> List[int]:
    n_features = belief_state.n_features

    if mc_passing_features is not None and USE_MC_PREFILTER:
        candidate_mask = mc_passing_features
    else:
        candidate_mask = np.ones(n_features, dtype=bool)

    if method == 'greedy_bayesian':
        return select_features_greedy_bayesian(belief_state, candidate_mask)

    scores = belief_state.get_feature_scores(method)
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
    """Get features that pass MC filter."""
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

        if hr_ci_lower > baseline_hr or alpha_ci_lower > baseline_alpha:
            passing_mask[feat_idx] = True

    return passing_mask


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================

def walk_forward_backtest(data: dict, n_satellites: int,
                          show_progress: bool = True) -> Tuple[pd.DataFrame, List[dict]]:
    """Run backtest with multi-hyperparameter learning."""
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
        print(f"  Learning: Decay, Prior Strength, Alpha Weight")
        print(f"  Initial priors: Decay~{DECAY_MIN + DECAY_PRIOR_ALPHA/(DECAY_PRIOR_ALPHA+DECAY_PRIOR_BETA)*(DECAY_MAX-DECAY_MIN):.3f}, "
              f"PriorStr~{PRIOR_STRENGTH_SHAPE*PRIOR_STRENGTH_SCALE:.0f}, "
              f"AlphaW~{ALPHA_WEIGHT_PRIOR_ALPHA/(ALPHA_WEIGHT_PRIOR_ALPHA+ALPHA_WEIGHT_PRIOR_BETA):.2f}")

    results = []
    hp_diagnostics = []
    belief_state = None
    months_since_reopt = REOPTIMIZATION_FREQUENCY
    selected_features = None

    # IMPORTANT: Persist hyperparameter beliefs across months
    # These will be carried forward and updated incrementally
    persistent_decay_belief = None
    persistent_prior_strength_belief = None
    persistent_alpha_weight_belief = None

    iterator = range(test_start_idx, len(dates))
    if show_progress:
        iterator = tqdm(iterator, desc=f"N={n_satellites}")

    for test_idx in iterator:
        test_date = dates[test_idx]

        mc_passing = None
        if USE_MC_PREFILTER:
            mc_passing = get_mc_passing_features(data, n_satellites, test_idx)

        if months_since_reopt >= REOPTIMIZATION_FREQUENCY:
            # Create new belief state for features
            belief_state = BeliefState(n_features, feature_names)

            # RESTORE persistent hyperparameter beliefs (if we have them)
            if persistent_decay_belief is not None:
                belief_state.decay_belief = persistent_decay_belief
                belief_state.prior_strength_belief = persistent_prior_strength_belief
                belief_state.alpha_weight_belief = persistent_alpha_weight_belief

            initialize_beliefs_from_mc(belief_state, data, n_satellites, test_idx)
            update_beliefs_from_history(belief_state, data, n_satellites, test_idx)

            # SAVE updated hyperparameter beliefs for next iteration
            persistent_decay_belief = belief_state.decay_belief
            persistent_prior_strength_belief = belief_state.prior_strength_belief
            persistent_alpha_weight_belief = belief_state.alpha_weight_belief

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

        # Get current hyperparameters
        current_decay = belief_state.get_decay()
        current_prior_strength = belief_state.get_prior_strength()
        current_alpha_weight = belief_state.get_alpha_weight()

        results.append({
            'date': test_date,
            'n_satellites': n_satellites,
            'avg_alpha': avg_alpha,
            'n_selected': len(selected_isin_names),
            'n_features': len(selected_features),
            'learned_decay': current_decay,
            'learned_prior_strength': current_prior_strength,
            'learned_alpha_weight': current_alpha_weight,
            'selected_isins': ','.join(selected_isin_names)
        })

        # Record diagnostics
        hp_diagnostics.append({
            'date': test_date,
            # Decay
            'decay': current_decay,
            'decay_ci': belief_state.decay_belief.ci(),
            # Prior strength
            'prior_strength': current_prior_strength,
            'prior_strength_ci': belief_state.prior_strength_belief.ci(),
            # Alpha weight
            'alpha_weight': current_alpha_weight,
            'alpha_weight_ci': belief_state.alpha_weight_belief.ci(),
        })

        # Update beliefs
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

def analyze_results(results_df: pd.DataFrame, n_satellites: int,
                    hp_diagnostics: List[dict]) -> Optional[dict]:
    if len(results_df) == 0:
        return None

    avg_alpha = results_df['avg_alpha'].mean()
    std_alpha = results_df['avg_alpha'].std()
    hit_rate = (results_df['avg_alpha'] > 0).mean()
    sharpe = avg_alpha / std_alpha if std_alpha > 0 else 0

    results_df = results_df.copy()
    results_df['cumulative'] = (1 + results_df['avg_alpha']).cumprod() - 1
    total_return = results_df['cumulative'].iloc[-1]

    # Hyperparameter stats
    avg_decay = results_df['learned_decay'].mean()
    final_decay = results_df['learned_decay'].iloc[-1]
    avg_prior_strength = results_df['learned_prior_strength'].mean()
    final_prior_strength = results_df['learned_prior_strength'].iloc[-1]
    avg_alpha_weight = results_df['learned_alpha_weight'].mean()
    final_alpha_weight = results_df['learned_alpha_weight'].iloc[-1]

    return {
        'n_satellites': n_satellites,
        'n_periods': len(results_df),
        'avg_alpha': avg_alpha,
        'std_alpha': std_alpha,
        'annual_alpha': avg_alpha * 12,
        'hit_rate': hit_rate,
        'sharpe': sharpe,
        'total_return': total_return,
        # Learned hyperparameters
        'avg_decay': avg_decay,
        'final_decay': final_decay,
        'avg_prior_strength': avg_prior_strength,
        'final_prior_strength': final_prior_strength,
        'avg_alpha_weight': avg_alpha_weight,
        'final_alpha_weight': final_alpha_weight,
    }


def print_hp_evolution(hp_diagnostics: List[dict], n_satellites: int):
    """Print hyperparameter evolution."""
    if len(hp_diagnostics) == 0:
        return

    df = pd.DataFrame(hp_diagnostics)

    print(f"\n  Hyperparameter Evolution (N={n_satellites}):")
    print(f"  " + "-" * 90)
    print(f"  {'Date':<12} {'Decay':>8} {'Prior Str':>12} {'Alpha Wt':>10}")
    print(f"  " + "-" * 90)

    indices = [0]
    for i in range(12, len(df), 12):
        indices.append(i)
    if len(df) - 1 not in indices:
        indices.append(len(df) - 1)

    for idx in indices:
        row = df.iloc[idx]
        date_str = row['date'].strftime('%Y-%m')
        print(f"  {date_str:<12} {row['decay']:>8.4f} {row['prior_strength']:>12.1f} {row['alpha_weight']:>10.3f}")

    print(f"  " + "-" * 90)


# ============================================================
# MAIN
# ============================================================

def main():
    """Run multi-hyperparameter Bayesian backtest."""
    print("=" * 60)
    print("BAYESIAN MULTI-HYPERPARAMETER LEARNING")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Holding period: {HOLDING_MONTHS} month")
    print(f"  N values to test: {N_SATELLITES_TO_TEST}")
    print(f"\nLearned Hyperparameters:")
    print(f"  1. Decay Rate: Beta({DECAY_PRIOR_ALPHA}, {DECAY_PRIOR_BETA}) -> [{DECAY_MIN}, {DECAY_MAX}]")
    print(f"  2. Prior Strength: Gamma({PRIOR_STRENGTH_SHAPE}, {PRIOR_STRENGTH_SCALE}) -> [{PRIOR_STRENGTH_MIN}, {PRIOR_STRENGTH_MAX}]")
    print(f"  3. Alpha Weight: Beta({ALPHA_WEIGHT_PRIOR_ALPHA}, {ALPHA_WEIGHT_PRIOR_BETA}) -> [0, 1]")
    print(f"\nFeature Selection:")
    print(f"  Method: {SELECTION_METHOD}")
    print(f"  MC pre-filter: {USE_MC_PREFILTER}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    all_stats = []
    all_results = {}
    all_hp_diagnostics = {}

    start_time = time.time()

    for test_num, n in enumerate(N_SATELLITES_TO_TEST, 1):
        print(f"\n{'='*60}")
        print(f"TEST {test_num}/{len(N_SATELLITES_TO_TEST)}: N={n}")
        print(f"{'='*60}")

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
                print(f"    Sharpe: {stats['sharpe']:.3f}")
                print(f"\n  Learned Hyperparameters (final):")
                print(f"    Decay: {stats['final_decay']:.4f}")
                print(f"    Prior Strength: {stats['final_prior_strength']:.1f}")
                print(f"    Alpha Weight: {stats['final_alpha_weight']:.3f}")

                print_hp_evolution(hp_diagnostics, n)

                output_file = OUTPUT_DIR / f'bayesian_multi_hp_N{n}.csv'
                results_df.to_csv(output_file, index=False)
        else:
            print(f"  No results")

    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - MULTI-HYPERPARAMETER LEARNING")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if len(all_stats) > 0:
        summary_df = pd.DataFrame(all_stats)

        print("\n" + "-" * 120)
        print(f"{'N':>3} {'Periods':>8} {'Monthly':>10} {'Annual':>10} {'Hit Rate':>10} "
              f"{'Sharpe':>8} {'Decay':>8} {'Prior':>8} {'Alpha W':>8}")
        print("-" * 120)

        for _, row in summary_df.sort_values('n_satellites').iterrows():
            print(f"{int(row['n_satellites']):>3} {int(row['n_periods']):>8} "
                  f"{row['avg_alpha']*100:>9.2f}% {row['annual_alpha']*100:>9.1f}% "
                  f"{row['hit_rate']:>9.1%} {row['sharpe']:>8.3f} "
                  f"{row['final_decay']:>8.4f} {row['final_prior_strength']:>8.1f} "
                  f"{row['final_alpha_weight']:>8.3f}")

        print("-" * 120)

        best_sharpe = summary_df.loc[summary_df['sharpe'].idxmax()]
        best_alpha = summary_df.loc[summary_df['annual_alpha'].idxmax()]
        best_hit = summary_df.loc[summary_df['hit_rate'].idxmax()]

        print(f"\nBest by Sharpe:     N={int(best_sharpe['n_satellites'])}, "
              f"Alpha={best_sharpe['annual_alpha']*100:.1f}%/yr, "
              f"Hit={best_sharpe['hit_rate']:.1%}, "
              f"Sharpe={best_sharpe['sharpe']:.3f}")

        print(f"Best by Alpha:      N={int(best_alpha['n_satellites'])}, "
              f"Alpha={best_alpha['annual_alpha']*100:.1f}%/yr, "
              f"Hit={best_alpha['hit_rate']:.1%}, "
              f"Sharpe={best_alpha['sharpe']:.3f}")

        print(f"Best by Hit Rate:   N={int(best_hit['n_satellites'])}, "
              f"Alpha={best_hit['annual_alpha']*100:.1f}%/yr, "
              f"Hit={best_hit['hit_rate']:.1%}, "
              f"Sharpe={best_hit['sharpe']:.3f}")

        summary_file = OUTPUT_DIR / 'bayesian_multi_hp_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\n[SAVED] {summary_file}")


if __name__ == '__main__':
    main()
