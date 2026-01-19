"""
Step 8: Bayesian Probabilistic Feature Selection Strategy
=========================================================

A fully probabilistic approach to feature selection that:
1. Maintains belief distributions over feature performance (not point estimates)
2. Updates beliefs using Bayesian inference as new data arrives
3. Selects features by optimizing expected utility given uncertainty
4. Uses Thompson Sampling for exploration-exploitation balance

Key differences from script 6/7:
- No hard cutoffs (MIN_ALPHA, MIN_HIT_RATE) - everything is probabilistic
- Features have belief distributions that update over time
- Selection optimizes for expected Sharpe (mean/std) not just mean
- Naturally handles uncertainty - confident features get more weight

Prior beliefs come from MC simulation (what random picking achieves).
Posterior beliefs update as we observe actual feature performance.

Usage:
    python 8_bayesian_strategy.py

Output:
    data/backtest_results/bayesian_backtest_summary.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import time
from numba import njit, prange
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================
# CONFIGURATION
# ============================================================

# Holding period
HOLDING_MONTHS = 1

# N values to test (ignored if USE_DYNAMIC_N is True)
N_SATELLITES_TO_TEST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Dynamic N: let the system decide how many satellites based on opportunities
USE_DYNAMIC_N = False  # If True, ignores N_SATELLITES_TO_TEST and picks optimal N each month
DYNAMIC_N_MIN = 1      # Minimum satellites when using dynamic N
DYNAMIC_N_MAX = 10     # Maximum satellites when using dynamic N
DYNAMIC_N_UTILITY_THRESHOLD = 0.5  # Min utility improvement to add another satellite

# Training parameters
MIN_TRAINING_MONTHS = 36
REOPTIMIZATION_FREQUENCY = 1  # Re-optimize every N months

# Bayesian model parameters
PRIOR_STRENGTH_BASE = 50  # Base prior strength (can be adjusted dynamically)
USE_DYNAMIC_PRIOR_STRENGTH = True  # Adjust prior strength based on MC sample quality

# Belief decay parameters
BELIEF_DECAY_BASE = 0.95  # Base decay (0.95 = ~1.5 year half-life)
USE_DYNAMIC_DECAY = True  # Adjust decay based on prediction accuracy
DECAY_MIN = 0.90  # Minimum decay (faster forgetting when predictions are bad)
DECAY_MAX = 0.98  # Maximum decay (slower forgetting when predictions are good)

# Enhanced priors: what MC metrics to use
USE_HITRATE_IN_PRIOR = True  # Include hit rate in prior (not just alpha)
USE_STABILITY_IN_PRIOR = True  # Penalize features with inconsistent MC performance
HITRATE_PRIOR_WEIGHT = 0.3  # Weight for hit rate in combined prior (alpha gets 1-this)

# Feature selection parameters
MAX_ENSEMBLE_SIZE = 10  # Maximum features in ensemble
MIN_PROBABILITY_POSITIVE = 0.55  # Minimum P(alpha > 0) to consider feature

# Selection method: 'expected_sharpe', 'thompson', 'probability_weighted', 'greedy_bayesian'
SELECTION_METHOD = 'greedy_bayesian'  # New method combining greedy + Bayesian

# Greedy Bayesian parameters
GREEDY_CANDIDATES = 30  # Top features to consider for greedy selection
GREEDY_IMPROVEMENT_THRESHOLD = 0.001  # Minimum expected improvement to add feature

# MC pre-filter (use MC data to pre-screen features)
USE_MC_PREFILTER = True  # Filter features using MC confidence intervals first
MC_CONFIDENCE_LEVEL = 0.90  # Lower than before - be more permissive

# Thompson sampling parameters (if used)
THOMPSON_SAMPLES = 100  # Number of samples for Thompson sampling

# Backward compatibility aliases (used when dynamic features are disabled)
PRIOR_STRENGTH = PRIOR_STRENGTH_BASE
BELIEF_DECAY = BELIEF_DECAY_BASE

# Data and output directories
DATA_DIR = Path(__file__).parent / 'data'
OUTPUT_DIR = DATA_DIR / 'backtest_results'


# ============================================================
# BELIEF STATE - Tracks distribution over feature performance
# ============================================================

@dataclass
class FeatureBelief:
    """
    Represents our belief about a feature's true alpha distribution.

    We model alpha as Normal(mu, sigma^2) and track:
    - mu: Current belief about mean alpha
    - sigma: Current belief about std of alpha
    - n_obs: Effective number of observations (for updating)
    - sum_alpha: Running sum of observed alphas (weighted)
    - sum_sq_alpha: Running sum of squared alphas (for variance)
    """
    mu: float = 0.0           # Believed mean alpha
    sigma: float = 0.05       # Believed std of alpha (5% default)
    n_obs: float = 0.0        # Effective observation count
    sum_alpha: float = 0.0    # Weighted sum of alphas
    sum_sq: float = 0.0       # Weighted sum of squared alphas

    # Prior from MC simulation
    prior_mu: float = 0.0
    prior_sigma: float = 0.05
    prior_strength: float = 10.0

    def probability_positive(self) -> float:
        """P(true alpha > 0) given current beliefs."""
        if self.sigma <= 0:
            return 1.0 if self.mu > 0 else 0.0
        # Use normal CDF: P(X > 0) = 1 - Phi(-mu/sigma)
        from scipy import stats
        return 1 - stats.norm.cdf(-self.mu / self.sigma)

    def expected_sharpe(self) -> float:
        """Expected Sharpe ratio = mu / sigma."""
        if self.sigma <= 0:
            return 0.0
        return self.mu / self.sigma

    def sample(self) -> float:
        """Sample from the belief distribution (for Thompson sampling)."""
        return np.random.normal(self.mu, self.sigma)

    def ci_lower(self, confidence: float = 0.95) -> float:
        """Lower bound of confidence interval for mean."""
        from scipy import stats
        if self.n_obs < 2:
            return -np.inf
        se = self.sigma / np.sqrt(max(1, self.n_obs))
        t_crit = stats.t.ppf(1 - confidence, df=max(1, self.n_obs - 1))
        return self.mu + t_crit * se


class BeliefState:
    """
    Maintains belief distributions for all features.

    This is the "brain" of the Bayesian system - it tracks what we
    believe about each feature and updates as we observe outcomes.
    """

    def __init__(self, n_features: int, feature_names: List[str]):
        self.n_features = n_features
        self.feature_names = feature_names
        self.beliefs: Dict[int, FeatureBelief] = {}

        # Initialize with uninformative priors
        for i in range(n_features):
            self.beliefs[i] = FeatureBelief()

    def set_prior_from_mc(self, feat_idx: int, mc_mu: float, mc_sigma: float,
                          strength: float = PRIOR_STRENGTH):
        """
        Set prior belief from MC simulation results.

        MC tells us: "random ensembles containing this feature achieve
        mean alpha mc_mu with std mc_sigma". This becomes our prior.
        """
        belief = self.beliefs[feat_idx]
        belief.prior_mu = mc_mu
        belief.prior_sigma = mc_sigma
        belief.prior_strength = strength

        # Initialize posterior to prior
        belief.mu = mc_mu
        belief.sigma = mc_sigma
        belief.n_obs = strength
        belief.sum_alpha = mc_mu * strength
        belief.sum_sq = (mc_sigma**2 + mc_mu**2) * strength

    def update(self, feat_idx: int, observed_alpha: float, weight: float = 1.0):
        """
        Bayesian update after observing a new alpha value.

        Uses conjugate normal-normal update:
        - Prior: alpha ~ N(mu_prior, sigma_prior^2)
        - Likelihood: observation ~ N(alpha, sigma_obs^2)
        - Posterior: alpha ~ N(mu_post, sigma_post^2)
        """
        belief = self.beliefs[feat_idx]

        # Decay old observations (gives more weight to recent data)
        belief.n_obs *= BELIEF_DECAY
        belief.sum_alpha *= BELIEF_DECAY
        belief.sum_sq *= BELIEF_DECAY

        # Add new observation
        belief.n_obs += weight
        belief.sum_alpha += observed_alpha * weight
        belief.sum_sq += (observed_alpha ** 2) * weight

        # Update posterior mean and variance
        # Combine prior and observations
        total_n = belief.prior_strength + belief.n_obs

        if total_n > 0:
            # Posterior mean: weighted average of prior and observed
            belief.mu = (belief.prior_strength * belief.prior_mu +
                        belief.sum_alpha) / total_n

            # Posterior variance: from observed data
            if belief.n_obs > 1:
                obs_mean = belief.sum_alpha / belief.n_obs
                obs_var = (belief.sum_sq / belief.n_obs) - obs_mean**2
                obs_var = max(0.0001, obs_var)  # Ensure positive

                # Combine prior and observed variance (simplified)
                prior_weight = belief.prior_strength / total_n
                obs_weight = belief.n_obs / total_n
                belief.sigma = np.sqrt(
                    prior_weight * belief.prior_sigma**2 +
                    obs_weight * obs_var
                )
            else:
                belief.sigma = belief.prior_sigma

    def decay_all(self):
        """Apply decay to all beliefs (called each period)."""
        for belief in self.beliefs.values():
            belief.n_obs *= BELIEF_DECAY
            belief.sum_alpha *= BELIEF_DECAY
            belief.sum_sq *= BELIEF_DECAY

    def get_feature_scores(self, method: str = 'expected_sharpe') -> np.ndarray:
        """
        Get scores for all features based on current beliefs.

        Methods:
        - 'expected_sharpe': mu / sigma (risk-adjusted expected return)
        - 'probability_positive': P(alpha > 0)
        - 'mean': just the mean (like current greedy)
        - 'ci_lower': lower confidence bound
        """
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
        """Sample from all belief distributions (for Thompson sampling)."""
        samples = np.zeros(self.n_features)
        for i, belief in self.beliefs.items():
            samples[i] = belief.sample()
        return samples


# ============================================================
# NUMBA FUNCTIONS FOR FAST COMPUTATION
# ============================================================

@njit(cache=True)
def select_top_n_isins_by_scores(rankings_slice, feature_indices, feature_scores,
                                  n_satellites, alpha_valid_mask):
    """
    Select top-N ISINs based on feature ensemble with score weighting.

    Unlike simple averaging, this weights features by their belief scores.
    """
    n_isins = rankings_slice.shape[0]
    n_features = len(feature_indices)

    # Compute weighted ensemble scores for each ISIN
    isin_scores = np.zeros(n_isins)

    for i in range(n_isins):
        if alpha_valid_mask is not None and not alpha_valid_mask[i]:
            isin_scores[i] = -np.inf
            continue

        score_sum = 0.0
        weight_sum = 0.0

        for j in range(n_features):
            feat_idx = feature_indices[j]
            feat_weight = max(0.0, feature_scores[j])  # Use belief score as weight

            val = rankings_slice[i, feat_idx]
            if not np.isnan(val) and feat_weight > 0:
                score_sum += val * feat_weight
                weight_sum += feat_weight

        if weight_sum > 0:
            isin_scores[i] = score_sum / weight_sum
        else:
            isin_scores[i] = -np.inf

    # Select top N
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
    """
    Compute what alpha each feature would have achieved on a given date.

    For each feature, select top-N ISINs using that feature alone,
    and compute the average alpha of those ISINs.
    """
    n_features = len(feature_indices)
    feature_alphas = np.empty(n_features)

    for f_idx in prange(n_features):
        feat_idx = feature_indices[f_idx]

        # Get rankings for this feature
        feat_rankings = rankings[date_idx, :, feat_idx]

        # Find valid ISINs
        n_isins = len(feat_rankings)
        valid_count = 0
        for i in range(n_isins):
            if not np.isnan(feat_rankings[i]) and alpha_valid[date_idx, i]:
                valid_count += 1

        if valid_count < n_satellites:
            feature_alphas[f_idx] = np.nan
            continue

        # Select top N by this feature's ranking
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

        # Compute average alpha of selected ISINs
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
    """
    Initialize feature beliefs from MC simulation data.

    ENHANCED: Now uses both alpha AND hit rate from MC, plus stability metrics.

    MC data tells us what random ensembles achieve - this is our prior
    for what each feature can do.
    """
    mc_data = data.get('mc_data')
    if mc_data is None:
        # No MC data - use uninformative priors
        for i in range(belief_state.n_features):
            belief_state.set_prior_from_mc(i, mc_mu=0.0, mc_sigma=0.03,
                                           strength=PRIOR_STRENGTH_BASE)
        return

    # Get MC data for this N
    mc_n_satellites = mc_data['n_satellites']
    n_to_idx = {int(n): i for i, n in enumerate(mc_n_satellites)}

    if n_satellites not in n_to_idx:
        # No MC data for this N - use uninformative priors
        for i in range(belief_state.n_features):
            belief_state.set_prior_from_mc(i, mc_mu=0.0, mc_sigma=0.03,
                                           strength=PRIOR_STRENGTH_BASE)
        return

    mc_idx = n_to_idx[n_satellites]
    mc_alpha_mean = mc_data['mc_alpha_mean'][mc_idx]  # (n_dates, n_features)
    mc_alpha_std = mc_data['mc_alpha_std'][mc_idx]
    mc_hitrates = mc_data['mc_hitrates'][mc_idx]  # NEW: hit rates
    mc_candidate_mask = mc_data['candidate_masks'][mc_idx]

    # Use MC data up to (but not including) test_idx as prior
    # This ensures no look-ahead bias
    for feat_idx in range(belief_state.n_features):
        # Get historical MC stats for this feature
        valid_dates = []
        for d in range(min(test_idx, mc_alpha_mean.shape[0])):
            if mc_candidate_mask[d, feat_idx]:
                mu = mc_alpha_mean[d, feat_idx]
                if not np.isnan(mu):
                    valid_dates.append(d)

        if len(valid_dates) > 0:
            # Collect alpha stats
            mus = [mc_alpha_mean[d, feat_idx] for d in valid_dates]
            stds = [mc_alpha_std[d, feat_idx] for d in valid_dates]

            avg_alpha_mu = np.nanmean(mus)
            avg_alpha_std = np.nanmean(stds)

            if np.isnan(avg_alpha_mu):
                avg_alpha_mu = 0.0
            if np.isnan(avg_alpha_std) or avg_alpha_std <= 0:
                avg_alpha_std = 0.03

            # NEW: Collect hit rate stats
            if USE_HITRATE_IN_PRIOR:
                hrs = [mc_hitrates[d, feat_idx] for d in valid_dates]
                avg_hr = np.nanmean(hrs)
                if np.isnan(avg_hr):
                    avg_hr = 0.5

                # Convert hit rate to "alpha equivalent"
                # Hit rate of 0.5 = 0 alpha equivalent, 1.0 = high positive
                hr_alpha_equiv = (avg_hr - 0.5) * 0.10  # Scale: 60% HR -> 1% equiv

                # Combine alpha and hit rate into final prior mu
                combined_mu = (1 - HITRATE_PRIOR_WEIGHT) * avg_alpha_mu + \
                              HITRATE_PRIOR_WEIGHT * hr_alpha_equiv
            else:
                combined_mu = avg_alpha_mu

            # NEW: Stability penalty - features with inconsistent MC performance
            # get higher uncertainty (wider prior)
            if USE_STABILITY_IN_PRIOR and len(valid_dates) >= 3:
                # Coefficient of variation across dates
                alpha_cv = np.nanstd(mus) / (abs(np.nanmean(mus)) + 0.001)
                # Stability factor: 1.0 for stable, up to 2.0 for unstable
                stability_factor = 1.0 + min(1.0, alpha_cv)
                adjusted_std = avg_alpha_std * stability_factor
            else:
                adjusted_std = avg_alpha_std

            # NEW: Dynamic prior strength based on data quality
            if USE_DYNAMIC_PRIOR_STRENGTH:
                # More historical dates = stronger prior
                # More consistent data = stronger prior
                data_quality = min(1.0, len(valid_dates) / 24)  # Max at 2 years
                if USE_STABILITY_IN_PRIOR and len(valid_dates) >= 3:
                    consistency = 1.0 / (1.0 + alpha_cv)  # Higher for stable features
                else:
                    consistency = 0.5
                # Scale prior strength: 25 to 100 based on quality
                dynamic_strength = PRIOR_STRENGTH_BASE * (0.5 + data_quality * consistency)
            else:
                dynamic_strength = PRIOR_STRENGTH_BASE

            belief_state.set_prior_from_mc(feat_idx, mc_mu=combined_mu,
                                           mc_sigma=adjusted_std,
                                           strength=dynamic_strength)
        else:
            # No MC data for this feature
            belief_state.set_prior_from_mc(feat_idx, mc_mu=0.0, mc_sigma=0.03,
                                           strength=PRIOR_STRENGTH_BASE * 0.5)


def compute_dynamic_decay(belief_state: BeliefState, data: dict,
                          n_satellites: int, recent_window: int = 6) -> float:
    """
    Compute dynamic decay rate based on recent prediction accuracy.

    If predictions have been accurate recently, use slower decay (trust history more).
    If predictions have been inaccurate, use faster decay (adapt quickly).

    Returns: decay rate between DECAY_MIN and DECAY_MAX
    """
    if not USE_DYNAMIC_DECAY:
        return BELIEF_DECAY_BASE

    # This gets called during belief updates, so we look at how well
    # the prior beliefs predicted actual outcomes
    # For simplicity, we use a heuristic based on belief confidence

    # Average confidence (probability positive) across features
    confidences = []
    for belief in belief_state.beliefs.values():
        prob = belief.probability_positive()
        # How far from 0.5 (uncertain) is this belief?
        confidence = abs(prob - 0.5) * 2  # 0 to 1 scale
        confidences.append(confidence)

    avg_confidence = np.mean(confidences) if confidences else 0.5

    # Higher confidence -> slower decay (trust beliefs more)
    # Lower confidence -> faster decay (adapt more)
    decay = DECAY_MIN + (DECAY_MAX - DECAY_MIN) * avg_confidence

    return decay


def update_beliefs_from_history(belief_state: BeliefState, data: dict,
                                 n_satellites: int, train_end_idx: int):
    """
    Update beliefs based on historical feature performance.

    ENHANCED: Uses dynamic decay that adjusts based on prediction accuracy.

    For each historical date, compute what alpha each feature achieved
    and use that to update our beliefs about the feature.
    """
    rankings = data['rankings']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']
    n_features = belief_state.n_features

    feature_indices = np.arange(n_features, dtype=np.int64)

    # Compute dynamic decay rate
    current_decay = compute_dynamic_decay(belief_state, data, n_satellites)

    # Track prediction accuracy for adaptive decay
    prediction_errors = []

    # Go through historical dates and update beliefs
    for date_idx in range(MIN_TRAINING_MONTHS, train_end_idx):
        # Compute what alpha each feature achieved on this date
        feature_alphas = compute_feature_alphas_for_date(
            rankings, feature_indices, alpha_matrix, alpha_valid,
            date_idx, n_satellites
        )

        # Update beliefs
        for feat_idx in range(n_features):
            alpha = feature_alphas[feat_idx]
            if not np.isnan(alpha):
                # Track prediction error (for adaptive decay)
                if USE_DYNAMIC_DECAY:
                    prior_mu = belief_state.beliefs[feat_idx].mu
                    error = abs(alpha - prior_mu)
                    prediction_errors.append(error)

                # Weight by recency (more recent = higher weight)
                months_ago = train_end_idx - date_idx
                weight = current_decay ** months_ago
                belief_state.update(feat_idx, alpha, weight=weight)

    # Adjust decay for next iteration based on prediction accuracy
    if USE_DYNAMIC_DECAY and len(prediction_errors) > 10:
        avg_error = np.mean(prediction_errors)
        # Normalize error (typical alpha is ~2-3%)
        normalized_error = min(1.0, avg_error / 0.05)
        # High error -> lower decay (faster forgetting)
        # Low error -> higher decay (slower forgetting)
        adjusted_decay = DECAY_MAX - (DECAY_MAX - DECAY_MIN) * normalized_error
        # Store for reporting (not used in this pass, but could be logged)
        belief_state.last_decay = adjusted_decay


def select_features_bayesian(belief_state: BeliefState,
                              method: str = 'expected_sharpe',
                              mc_passing_features: Optional[np.ndarray] = None,
                              rankings: Optional[np.ndarray] = None,
                              date_idx: Optional[int] = None) -> List[int]:
    """
    Select features based on current beliefs using specified method.

    Methods:
    - 'expected_sharpe': Pick features with highest E[alpha]/std[alpha]
    - 'thompson': Sample from beliefs, pick highest samples
    - 'probability_weighted': Weight by P(alpha > 0), pick highest
    - 'greedy_bayesian': Greedy ensemble building using belief-weighted scores
    """
    n_features = belief_state.n_features

    # Apply MC pre-filter if available
    if mc_passing_features is not None and USE_MC_PREFILTER:
        candidate_mask = mc_passing_features
    else:
        candidate_mask = np.ones(n_features, dtype=bool)

    if method == 'thompson':
        # Thompson Sampling: sample multiple times, pick features that win most
        win_counts = np.zeros(n_features)

        for _ in range(THOMPSON_SAMPLES):
            samples = belief_state.thompson_sample()
            # Apply mask
            samples[~candidate_mask] = -np.inf
            # Find features with positive samples
            positive_mask = samples > 0
            if positive_mask.any():
                winner = np.argmax(samples)
                win_counts[winner] += 1

        # Select features that won often
        selected_indices = np.argsort(win_counts)[::-1]

    elif method == 'greedy_bayesian':
        # Greedy ensemble building using belief-weighted evaluation
        return select_features_greedy_bayesian(
            belief_state, candidate_mask, rankings, date_idx
        )

    elif method == 'expected_sharpe':
        # Select by expected Sharpe ratio
        scores = belief_state.get_feature_scores('expected_sharpe')
        scores[~candidate_mask] = -np.inf
        selected_indices = np.argsort(scores)[::-1]

    elif method == 'probability_weighted':
        # Select by P(alpha > 0)
        scores = belief_state.get_feature_scores('probability_positive')
        scores[~candidate_mask] = -np.inf
        selected_indices = np.argsort(scores)[::-1]

    else:
        # Default: by mean
        scores = belief_state.get_feature_scores('mean')
        scores[~candidate_mask] = -np.inf
        selected_indices = np.argsort(scores)[::-1]

    # Filter to features with P(alpha > 0) > threshold
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


def select_features_greedy_bayesian(belief_state: BeliefState,
                                     candidate_mask: np.ndarray,
                                     rankings: Optional[np.ndarray] = None,
                                     date_idx: Optional[int] = None) -> List[int]:
    """
    Greedy ensemble building with Bayesian scoring.

    Start with the best expected Sharpe feature, then greedily add features
    that improve the ensemble's expected utility (combining hit rate and alpha).

    This combines the stability of greedy selection with the uncertainty
    awareness of Bayesian beliefs.
    """
    n_features = belief_state.n_features

    # Get scores for all features
    sharpe_scores = belief_state.get_feature_scores('expected_sharpe')
    prob_positive = belief_state.get_feature_scores('probability_positive')
    mean_alpha = belief_state.get_feature_scores('mean')

    # Filter candidates by P(alpha > 0) threshold and MC mask
    valid_candidates = []
    for i in range(n_features):
        if candidate_mask[i] and prob_positive[i] >= MIN_PROBABILITY_POSITIVE:
            valid_candidates.append(i)

    if len(valid_candidates) == 0:
        return []

    # Sort candidates by expected Sharpe (primary) and alpha (secondary)
    # Take top GREEDY_CANDIDATES for consideration
    candidate_scores = [(i, sharpe_scores[i], mean_alpha[i]) for i in valid_candidates]
    candidate_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    top_candidates = [x[0] for x in candidate_scores[:GREEDY_CANDIDATES]]

    if len(top_candidates) == 0:
        return []

    # Start with best feature
    selected = [top_candidates[0]]
    remaining = set(top_candidates[1:])

    # Greedy addition
    while len(selected) < MAX_ENSEMBLE_SIZE and len(remaining) > 0:
        best_candidate = None
        best_improvement = -np.inf

        # Current ensemble expected utility
        current_utility = compute_ensemble_expected_utility(
            selected, belief_state
        )

        for candidate in remaining:
            # Compute utility with candidate added
            test_ensemble = selected + [candidate]
            new_utility = compute_ensemble_expected_utility(
                test_ensemble, belief_state
            )

            improvement = new_utility - current_utility

            if improvement > best_improvement:
                best_improvement = improvement
                best_candidate = candidate

        # Only add if improvement exceeds threshold
        if best_candidate is not None and best_improvement > GREEDY_IMPROVEMENT_THRESHOLD:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break

    return selected


def compute_ensemble_expected_utility(feature_indices: List[int],
                                       belief_state: BeliefState) -> float:
    """
    Compute expected utility of a feature ensemble.

    Utility = E[alpha] * P(alpha > 0) / std[alpha]

    This balances expected return, hit rate, and stability.
    """
    if len(feature_indices) == 0:
        return -np.inf

    # Get beliefs for selected features
    mus = [belief_state.beliefs[i].mu for i in feature_indices]
    sigmas = [belief_state.beliefs[i].sigma for i in feature_indices]
    probs = [belief_state.beliefs[i].probability_positive() for i in feature_indices]

    # Ensemble mean (average of feature means)
    ensemble_mu = np.mean(mus)

    # Ensemble probability positive (geometric mean - captures joint probability)
    ensemble_prob = np.exp(np.mean(np.log(np.clip(probs, 0.01, 0.99))))

    # Ensemble std (assuming some correlation, so not just average)
    # Use sqrt of average variance as approximation
    avg_var = np.mean([s**2 for s in sigmas])
    ensemble_sigma = np.sqrt(avg_var)

    if ensemble_sigma <= 0:
        return -np.inf

    # Utility: expected alpha * hit probability / volatility
    # This rewards: high alpha, high hit rate, low volatility
    utility = ensemble_mu * ensemble_prob / ensemble_sigma

    return utility


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================

def get_mc_passing_features(data: dict, n_satellites: int, test_idx: int) -> Optional[np.ndarray]:
    """
    Get features that pass MC filter for this date.

    Uses the same logic as script 7: features must have CI lower bounds
    above the baseline (mean random performance).
    """
    mc_data = data.get('mc_data')
    if mc_data is None:
        return None

    mc_n_satellites = mc_data['n_satellites']
    n_to_idx = {int(n): i for i, n in enumerate(mc_n_satellites)}

    if n_satellites not in n_to_idx:
        return None

    mc_idx = n_to_idx[n_satellites]
    mc_hitrates = mc_data['mc_hitrates'][mc_idx]  # (n_dates, n_features)
    mc_alpha_mean = mc_data['mc_alpha_mean'][mc_idx]
    mc_alpha_std = mc_data['mc_alpha_std'][mc_idx]
    mc_candidate_mask = mc_data['candidate_masks'][mc_idx]

    # Get config to find n_samples
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

    from scipy import stats

    # Compute baseline from this date's candidates (dynamic baseline)
    date_candidates = mc_candidate_mask[test_idx]
    date_hr_values = mc_hitrates[test_idx, date_candidates]
    date_alpha_values = mc_alpha_mean[test_idx, date_candidates]

    valid_hr = date_hr_values[~np.isnan(date_hr_values)]
    valid_alpha = date_alpha_values[~np.isnan(date_alpha_values)]

    if len(valid_hr) == 0 or len(valid_alpha) == 0:
        return None

    baseline_hr = np.mean(valid_hr)
    baseline_alpha = np.mean(valid_alpha)

    # t-critical value for confidence interval
    t_crit = stats.t.ppf(MC_CONFIDENCE_LEVEL, df=max(1, mc_n_samples - 1))

    for feat_idx in range(n_features):
        if not date_candidates[feat_idx]:
            continue

        hr = mc_hitrates[test_idx, feat_idx]
        alpha_mu = mc_alpha_mean[test_idx, feat_idx]
        alpha_std = mc_alpha_std[test_idx, feat_idx]

        if np.isnan(hr) or np.isnan(alpha_mu):
            continue

        # Confidence intervals
        # For hit rate (proportion), use binomial standard error: sqrt(p*(1-p)/n)
        hr_se = np.sqrt(hr * (1 - hr) / mc_n_samples) if 0 < hr < 1 else 0
        alpha_se = alpha_std / np.sqrt(mc_n_samples) if alpha_std > 0 else 0

        hr_ci_lower = hr - t_crit * hr_se
        alpha_ci_lower = alpha_mu - t_crit * alpha_se

        # OR logic: pass if either HR or Alpha CI is above baseline
        hr_significant = hr_ci_lower > baseline_hr
        alpha_significant = alpha_ci_lower > baseline_alpha

        if hr_significant or alpha_significant:
            passing_mask[feat_idx] = True

    return passing_mask


def walk_forward_backtest(data: dict, n_satellites: int,
                          show_progress: bool = True) -> pd.DataFrame:
    """
    Run walk-forward backtest with Bayesian feature selection.
    """
    dates = data['dates']
    isins = data['isins']
    rankings = data['rankings']
    alpha_matrix = data['alpha_matrix']
    alpha_valid = data['alpha_valid']
    feature_names = data['feature_names']
    n_features = len(feature_names)

    # Find test start
    min_train_date = dates[0] + pd.DateOffset(months=MIN_TRAINING_MONTHS)
    test_start_idx = np.where(dates >= min_train_date)[0]

    if len(test_start_idx) == 0:
        return pd.DataFrame()

    test_start_idx = test_start_idx[0]

    if show_progress:
        print(f"\n  Testing N={n_satellites}")
        print(f"  Test period: {dates[test_start_idx].date()} to {dates[-1].date()}")
        print(f"  Selection method: {SELECTION_METHOD}")
        print(f"  MC pre-filter: {USE_MC_PREFILTER}")

    results = []
    belief_state = None
    months_since_reopt = REOPTIMIZATION_FREQUENCY
    selected_features = None

    iterator = range(test_start_idx, len(dates))
    if show_progress:
        iterator = tqdm(iterator, desc=f"N={n_satellites}")

    for test_idx in iterator:
        test_date = dates[test_idx]

        # Get MC passing features for this date (if using pre-filter)
        mc_passing = None
        if USE_MC_PREFILTER:
            mc_passing = get_mc_passing_features(data, n_satellites, test_idx)

        # Re-initialize and update beliefs periodically
        if months_since_reopt >= REOPTIMIZATION_FREQUENCY:
            # Create fresh belief state
            belief_state = BeliefState(n_features, feature_names)

            # Set priors from MC data
            initialize_beliefs_from_mc(belief_state, data, n_satellites, test_idx)

            # Update beliefs from historical performance
            update_beliefs_from_history(belief_state, data, n_satellites, test_idx)

            # Select features based on current beliefs (with MC pre-filter)
            selected_features = select_features_bayesian(
                belief_state, method=SELECTION_METHOD,
                mc_passing_features=mc_passing,
                rankings=rankings,
                date_idx=test_idx
            )

            months_since_reopt = 0

        months_since_reopt += 1

        # Skip if no valid features
        if not selected_features or len(selected_features) == 0:
            continue

        # Get feature scores for weighting
        feature_scores = np.array([
            belief_state.beliefs[f].expected_sharpe()
            for f in selected_features
        ])

        # Select top-N ETFs using belief-weighted ensemble
        feature_arr = np.array(selected_features, dtype=np.int64)
        alpha_valid_mask = alpha_valid[test_idx]

        selected_isins = select_top_n_isins_by_scores(
            rankings[test_idx], feature_arr, feature_scores,
            n_satellites, alpha_valid_mask
        )

        if selected_isins[0] == -1:
            continue

        # Compute average alpha
        alphas = []
        selected_isin_names = []

        for isin_idx in selected_isins:
            if alpha_valid[test_idx, isin_idx]:
                alphas.append(alpha_matrix[test_idx, isin_idx])
                selected_isin_names.append(isins[isin_idx])

        if len(alphas) == 0:
            continue

        avg_alpha = np.mean(alphas)

        # Record result
        results.append({
            'date': test_date,
            'n_satellites': n_satellites,
            'avg_alpha': avg_alpha,
            'n_selected': len(selected_isin_names),
            'n_features': len(selected_features),
            'selected_isins': ','.join(selected_isin_names)
        })

        # Update beliefs with observed outcome
        # (This gives us online learning - beliefs update as we go)
        if belief_state is not None:
            for feat_idx in selected_features:
                # Compute what this feature achieved
                feat_rankings = rankings[test_idx, :, feat_idx]
                valid_mask = ~np.isnan(feat_rankings) & alpha_valid[test_idx]

                if valid_mask.sum() >= n_satellites:
                    # Get top-N by this feature
                    valid_indices = np.where(valid_mask)[0]
                    top_indices = valid_indices[
                        np.argsort(feat_rankings[valid_indices])[-n_satellites:]
                    ]
                    feat_alpha = alpha_matrix[test_idx, top_indices].mean()

                    if not np.isnan(feat_alpha):
                        belief_state.update(feat_idx, feat_alpha, weight=1.0)

    return pd.DataFrame(results)


# ============================================================
# ANALYSIS
# ============================================================

def analyze_results(results_df: pd.DataFrame, n_satellites: int) -> Optional[dict]:
    """Analyze backtest results."""
    if len(results_df) == 0:
        return None

    avg_alpha = results_df['avg_alpha'].mean()
    std_alpha = results_df['avg_alpha'].std()
    hit_rate = (results_df['avg_alpha'] > 0).mean()
    sharpe = avg_alpha / std_alpha if std_alpha > 0 else 0

    results_df = results_df.copy()
    results_df['cumulative'] = (1 + results_df['avg_alpha']).cumprod() - 1
    total_return = results_df['cumulative'].iloc[-1]

    return {
        'n_satellites': n_satellites,
        'selection_method': SELECTION_METHOD,
        'prior_strength': PRIOR_STRENGTH_BASE,
        'dynamic_prior': USE_DYNAMIC_PRIOR_STRENGTH,
        'belief_decay': BELIEF_DECAY_BASE,
        'dynamic_decay': USE_DYNAMIC_DECAY,
        'use_hitrate_prior': USE_HITRATE_IN_PRIOR,
        'use_stability': USE_STABILITY_IN_PRIOR,
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
    """Run Bayesian walk-forward backtest."""
    print("=" * 60)
    print("BAYESIAN PROBABILISTIC FEATURE SELECTION (ENHANCED)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Holding period: {HOLDING_MONTHS} month")
    print(f"  N values to test: {N_SATELLITES_TO_TEST}")
    print(f"\nBayesian Parameters:")
    print(f"  Selection method: {SELECTION_METHOD}")
    print(f"  Prior strength (base): {PRIOR_STRENGTH_BASE}")
    print(f"  Dynamic prior strength: {USE_DYNAMIC_PRIOR_STRENGTH}")
    print(f"  Belief decay (base): {BELIEF_DECAY_BASE}")
    print(f"  Dynamic decay: {USE_DYNAMIC_DECAY}" + (f" (range: {DECAY_MIN}-{DECAY_MAX})" if USE_DYNAMIC_DECAY else ""))
    print(f"  Min P(alpha > 0): {MIN_PROBABILITY_POSITIVE:.0%}")
    print(f"  Max ensemble size: {MAX_ENSEMBLE_SIZE}")
    print(f"\nEnhanced Priors:")
    print(f"  Use hit rate in prior: {USE_HITRATE_IN_PRIOR}" + (f" (weight: {HITRATE_PRIOR_WEIGHT:.0%})" if USE_HITRATE_IN_PRIOR else ""))
    print(f"  Use stability penalty: {USE_STABILITY_IN_PRIOR}")
    if SELECTION_METHOD == 'greedy_bayesian':
        print(f"\nGreedy Bayesian Parameters:")
        print(f"  Greedy candidates: {GREEDY_CANDIDATES}")
        print(f"  Improvement threshold: {GREEDY_IMPROVEMENT_THRESHOLD}")
    print(f"\nMC Pre-filter:")
    print(f"  Use MC pre-filter: {USE_MC_PREFILTER}")
    if USE_MC_PREFILTER:
        print(f"  MC confidence level: {MC_CONFIDENCE_LEVEL:.0%}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_data()

    all_stats = []
    all_results = {}

    start_time = time.time()

    # Test each N value
    for test_num, n in enumerate(N_SATELLITES_TO_TEST, 1):
        print(f"\n{'='*60}")
        print(f"TEST {test_num}/{len(N_SATELLITES_TO_TEST)}: N={n}")
        print(f"{'='*60}")

        results_df = walk_forward_backtest(data, n, show_progress=True)

        if len(results_df) > 0:
            all_results[n] = results_df

            stats = analyze_results(results_df, n)
            if stats:
                all_stats.append(stats)

                print(f"\n  Results:")
                print(f"    Periods: {stats['n_periods']}")
                print(f"    Monthly Alpha: {stats['avg_alpha']*100:.2f}%")
                print(f"    Annual Alpha: {stats['annual_alpha']*100:.1f}%")
                print(f"    Hit Rate: {stats['hit_rate']:.2%}")
                print(f"    Sharpe: {stats['sharpe']:.3f}")

                # Save individual results
                output_file = OUTPUT_DIR / f'bayesian_backtest_N{n}.csv'
                results_df.to_csv(output_file, index=False)
        else:
            print(f"  No results")

    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - BAYESIAN STRATEGY")
    print("=" * 60)
    print(f"\nMethod: {SELECTION_METHOD}")
    print(f"Prior strength: {PRIOR_STRENGTH}")
    print(f"Belief decay: {BELIEF_DECAY}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if len(all_stats) > 0:
        summary_df = pd.DataFrame(all_stats)

        print("\n" + "-" * 80)
        print(f"{'N':>3} {'Periods':>8} {'Monthly':>10} {'Annual':>10} {'Hit Rate':>10} {'Sharpe':>8}")
        print("-" * 80)

        for _, row in summary_df.sort_values('n_satellites').iterrows():
            print(f"{int(row['n_satellites']):>3} {int(row['n_periods']):>8} "
                  f"{row['avg_alpha']*100:>9.2f}% {row['annual_alpha']*100:>9.1f}% "
                  f"{row['hit_rate']:>9.1%} {row['sharpe']:>8.3f}")

        print("-" * 80)

        # Find best
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

        # Save summary
        summary_file = OUTPUT_DIR / 'bayesian_backtest_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\n[SAVED] {summary_file}")


if __name__ == '__main__':
    main()
