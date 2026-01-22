# Three Potential Pipeline Improvements

## 1. ALTERNATIVE RANKING METHODS

### Current Method: Percentile Ranking
**Code** (Step 4, line 428):
```python
group['rank'] = group['value'].rank(pct=True)  # Returns [0, 1] percentile
```

**What it does**:
- For each date, for each feature:
  - Rank all 100 ETFs by their feature value
  - Convert rank to percentile: 0 = worst, 1 = best
  - ETF at position 1 → percentile 0.01
  - ETF at position 50 → percentile 0.50
  - ETF at position 100 → percentile 1.00

**Example**:
```
Date: 2024-01-31
Feature: Momentum (12-month return)

ETF Rankings:
  ETF A: 25% return → rank 100/100 → percentile 1.00 (best)
  ETF B: 18% return → rank 75/100  → percentile 0.75
  ETF C: 5% return  → rank 50/100  → percentile 0.50
  ETF D: -3% return → rank 1/100   → percentile 0.01 (worst)
```

**Why percentile might be suboptimal**:
1. **Ignores spread**: Doesn't distinguish between:
   - Tightly clustered values (25%, 24%, 23% → all look good)
   - Spread-out values (25%, 5%, -10% → highly variable)

2. **Treats extremes equally**: Gives same weight to:
   - Mild outlier: +1% vs median
   - Extreme outlier: +50% vs median

3. **Non-linear**: Big differences at extremes become small percentile differences
   - If median is 10%, difference between 10% and 11% is small (percentile 0.48→0.52 = 4% change)
   - But difference between 40% and 41% is also small (percentile 0.98→0.99 = 1% change)

---

### Alternative 1: Z-Score Ranking

**Concept**: How many standard deviations is each ETF from the mean?

```python
def zscore_ranking(values):
    """
    Rank by z-score: (value - mean) / std
    Then map to [0, 1] via sigmoid: 1 / (1 + exp(-z))
    """
    mean = np.nanmean(values)
    std = np.nanstd(values)

    if std == 0:
        return percentile_ranking(values)  # Fallback

    z_scores = (values - mean) / std

    # Map z-scores to [0, 1] using sigmoid
    # This gives more weight to extreme performers
    rankings = 1.0 / (1.0 + np.exp(-z_scores))

    return rankings
```

**Example**:
```
Date: 2024-01-31
Feature: Momentum

Mean return: 8%
Std return: 5%

ETF A: 25% return → z = (25-8)/5 = 3.4 → sigmoid(3.4) = 0.97 ✓✓✓
ETF B: 18% return → z = (18-8)/5 = 2.0 → sigmoid(2.0) = 0.88 ✓✓
ETF C: 5% return  → z = (5-8)/5  = -0.6 → sigmoid(-0.6) = 0.35
ETF D: -3% return → z = (-3-8)/5 = -2.2 → sigmoid(-2.2) = 0.10 ✗
```

**Advantages**:
- ✅ Accounts for variance (tightly clustered vs spread)
- ✅ Extreme performers get disproportionate weight (good for momentum)
- ✅ Naturally handles outliers (sigmoid saturates)
- ✅ More robust to scale (doesn't matter if values are 0-100 or 0-1)

**Disadvantages**:
- ❌ Assumes normal distribution (not true for financial returns)
- ❌ Less stable when std is near zero
- ❌ May over-reward extreme values (noise vs signal)

**Expected impact**: +0.5-1% alpha (if momentum is important, this captures it better)

---

### Alternative 2: Robust Ranking (Percentile of Winsorized Values)

**Concept**: Use percentile, but remove extreme outliers first

```python
def robust_ranking(values, percentile_bounds=[5, 95]):
    """
    Rank using percentiles, but clip extreme outliers first.

    This prevents a single noise outlier from distorting rankings.
    """
    lower_bound = np.nanpercentile(values, percentile_bounds[0])
    upper_bound = np.nanpercentile(values, percentile_bounds[1])

    # Clip values to [5th, 95th] percentile
    clipped = np.clip(values, lower_bound, upper_bound)

    # Standard percentile ranking on clipped values
    rankings = pd.Series(clipped).rank(pct=True).values

    return rankings
```

**Example**:
```
Date: 2024-01-31
Feature: Some noisy indicator

Raw values: [0.1, 0.2, 0.3, 0.4, 0.5, 1000.0]  ← Last one is outlier!

Standard percentile rank:
  0.1 → 0.17, 0.2 → 0.33, 0.3 → 0.50, 0.4 → 0.67, 0.5 → 0.83, 1000 → 1.00

Robust ranking (clip to [5%, 95%]):
  Bounds: [5th% = 0.15, 95th% = 0.525]
  After clipping: [0.15, 0.2, 0.3, 0.4, 0.5, 0.525]
  Percentile rank: 0.20, 0.40, 0.60, 0.80, 1.00, 1.00

Result: Outlier doesn't distort rankings of normal values
```

**Advantages**:
- ✅ Robust to noise and outliers
- ✅ Preserves percentile intuition
- ✅ Simple, easy to understand
- ✅ Good for features with occasional extreme spikes

**Disadvantages**:
- ❌ Loses information from true outliers
- ❌ May not distinguish between top performers well
- ❌ Requires tuning winsorization bounds

**Expected impact**: +0.3-0.7% alpha (good if data has outlier noise)

---

### Alternative 3: Signal-to-Noise Ranking

**Concept**: Rank by how much signal each feature has relative to noise

```python
def signal_to_noise_ranking(values, lookback_window=20):
    """
    Rank by the ratio of signal strength to noise level.

    Signal = how much the value deviates from recent mean
    Noise = volatility of recent values
    """
    # This is more complex - compute rolling stats
    rolling_mean = pd.Series(values).rolling(lookback_window).mean()
    rolling_std = pd.Series(values).rolling(lookback_window).std()

    # Signal-to-noise ratio
    signal = values - rolling_mean
    snr = signal / (rolling_std + 1e-8)

    # Rank by SNR
    rankings = pd.Series(snr).rank(pct=True).values

    return rankings
```

**Advantages**:
- ✅ Distinguishes signal from noise explicitly
- ✅ Adapts to changing volatility
- ✅ More sophisticated than simple percentile

**Disadvantages**:
- ❌ More complex, harder to debug
- ❌ Requires tuning lookback window
- ❌ May be unstable with changing regimes
- ❌ Slower to compute

**Expected impact**: +0.2-0.5% alpha (conditional on feature design)

---

## RECOMMENDATION FOR RANKING

**Start with Z-Score Ranking** because:
1. Momentum features benefit from capturing extremes
2. Natural interpretation (standard deviations from mean)
3. Simple to implement
4. Fast to test (30 min)
5. Highest expected impact (+0.5-1%)

**Quick test procedure**:
```
1. Edit Step 4 (4_compute_forward_alpha.py, line 428)
2. Replace: group['rank'] = group['value'].rank(pct=True)
3. With:    group['rank'] = zscore_ranking(group['value'].values)
4. Rerun step 4 (backward compatible output)
5. Run step 5-7 with new rankings
6. Compare alpha results
```

---

---

## 2. FEATURE CORRELATION MODELING

### The Problem: Ensemble Assumes Independence

**Current ensemble utility calculation** (bayesian_strategy.py, line 1029):
```python
def compute_ensemble_expected_utility(feature_indices, belief_state):
    mus = [belief_state.beliefs[i].mu for i in feature_indices]
    sigmas = [belief_state.beliefs[i].sigma for i in feature_indices]
    probs = [belief_state.beliefs[i].probability_positive() for i in feature_indices]

    ensemble_mu = np.mean(mus)                      # Average alpha
    ensemble_prob = np.exp(np.mean(np.log(probs))) # Geometric mean

    avg_var = np.mean([s**2 for s in sigmas])
    ensemble_sigma = np.sqrt(avg_var)               # ← ASSUMES INDEPENDENCE

    return ensemble_mu * ensemble_prob / ensemble_sigma
```

**The issue with independence assumption**:

When you have 3 correlated features (all momentum-based):
```
Feature 1 (12m momentum):  σ = 0.05, ρ_12 = 0.90, ρ_13 = 0.85
Feature 2 (6m momentum):   σ = 0.05, ρ_23 = 0.92
Feature 3 (3m momentum):   σ = 0.05

Current calculation (assumes independence):
ensemble_sigma = sqrt((0.05² + 0.05² + 0.05²) / 3)
              = sqrt(0.0075)
              = 0.087

Correct calculation (accounts for correlation):
Cov matrix = [0.0025   0.00225  0.00212]
             [0.00225  0.0025   0.0023 ]
             [0.00212  0.0023   0.0025 ]

ensemble_sigma = sqrt(w^T * Cov * w)  where w = [1/3, 1/3, 1/3]
              = sqrt(0.00717)
              = 0.085

Difference: 0.087 vs 0.085 = 2% error (SMALL in this case)

But with 5 highly correlated momentum features:
Current (independent):    0.071
Correct (correlated):     0.063
Difference: 13% ERROR! ✗✗✗
```

**Why this hurts performance**:
1. **Over-optimistic risk estimates**: Thinks ensemble is more diversified than it is
2. **Selects high-correlation ensembles**: Thinks adding correlated features helps more than it does
3. **Overconfident positions**: Results in too much allocation to similar strategies

### The Solution: Covariance-Adjusted Utility

```python
def compute_ensemble_expected_utility_with_correlation(feature_indices, belief_state, feature_cov=None):
    """
    Improved utility calculation that accounts for feature correlation.
    """
    mus = np.array([belief_state.beliefs[i].mu for i in feature_indices])
    sigmas = np.array([belief_state.beliefs[i].sigma for i in feature_indices])
    probs = np.array([belief_state.beliefs[i].probability_positive() for i in feature_indices])

    ensemble_mu = np.mean(mus)
    ensemble_prob = np.exp(np.mean(np.log(np.clip(probs, 0.01, 0.99))))

    # VERSION 1: Simple - assume pairwise correlations
    if feature_cov is None:
        # Estimate from historical feature values
        # (requires computing correlation matrix of past feature alphas)
        avg_var = np.mean(sigmas**2)
        ensemble_sigma = np.sqrt(avg_var)
    else:
        # VERSION 2: Complex - use actual covariance
        weights = np.ones(len(feature_indices)) / len(feature_indices)

        # Extract relevant submatrix from covariance
        cov_sub = feature_cov[np.ix_(feature_indices, feature_indices)]

        # Portfolio variance = w^T * Cov * w
        portfolio_var = weights @ cov_sub @ weights
        ensemble_sigma = np.sqrt(max(portfolio_var, 1e-6))

    if ensemble_sigma <= 0:
        return -np.inf

    return ensemble_mu * ensemble_prob / ensemble_sigma
```

### How to Compute Feature Covariance Matrix

**Step 5a: Add covariance computation** (after existing feature-alpha computation):

```python
def compute_feature_correlation_matrix(feature_alpha_matrix, n_dates_lookback=36):
    """
    Compute correlation/covariance of feature predictions.

    Args:
        feature_alpha_matrix: (n_dates, n_features)
        n_dates_lookback: Use last N months to estimate correlation

    Returns:
        cov_matrix: (n_features, n_features) covariance matrix
    """
    # Use last n_dates_lookback observations
    recent_alphas = feature_alpha_matrix[-n_dates_lookback:, :]

    # Compute correlation (handle NaNs)
    # Use Ledoit-Wolf shrinkage for stability
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf()
    cov, _ = lw.fit(recent_alphas).covariance_, lw.shrinkage_

    return cov
```

### Why This Helps

**Example: Adding 4th momentum feature**

Without correlation modeling:
```
Feature 1 (12m momentum): alpha=3%, sigma=5%
Feature 2 (6m momentum):  alpha=2%, sigma=4%
Feature 3 (3m momentum):  alpha=2%, sigma=4%
Feature 4 (1m momentum):  alpha=1%, sigma=3%

Greedy decision: "Adding feature 4 reduces ensemble variance from 3.2% to 2.1%"
Decision: ✓ Add it

Realized: Features are all 0.95 correlated
True variance reduction: 3.2% → 3.0% (only 0.2% improvement)
Result: Wasted slot in ensemble! ✗
```

With correlation modeling:
```
Same features, but we know correlation matrix

Greedy decision: "Adding feature 4 reduces ensemble variance from 3.2% to 3.0%"
Improvement = 0.2% (realistic)
Threshold check: Is 0.2% improvement > greedy_threshold (0.008)?
Decision: ✗ Don't add it (improvement too small)

Result: Reserve ensemble slot for uncorrelated feature → Better diversification ✓
```

### Expected Impact

**Expected alpha gain**: +1-3%

**Why**:
- Current greedy is building over-correlated ensembles
- Better diversification = more stable returns
- Hit rates should improve (more uncorrelated signals)
- Sharpe ratio should improve (less hidden risk)

---

---

## 3. QUADRATIC PROGRAMMING ENSEMBLE SELECTION

### The Problem with Greedy Selection

**Current greedy approach**:
```
1. Start with best single feature
2. Try adding each remaining feature
3. Add the one with biggest improvement
4. Repeat until improvement < threshold
5. Return ensemble
```

**Why greedy is suboptimal**:

```
Features ranked by expected Sharpe:
Feature A: Sharpe = 2.0, vol = 5%    ← Highest expected return/risk
Feature B: Sharpe = 1.5, vol = 3%
Feature C: Sharpe = 1.4, vol = 2%
Feature D: Sharpe = 0.8, vol = 10%
Feature E: Sharpe = 0.3, vol = 1%    ← Lowest Sharpe

Greedy algorithm:
Step 1: Select A (highest Sharpe)
Step 2: Try adding B, C, D, E
        Best = B (improves ensemble Sharpe most)
Step 3: Try adding C, D, E to [A,B]
        Best = C
Step 4: Try adding D, E to [A,B,C]
        Adding D: Huge increase in volatility, small alpha gain
        Adding E: Tiny alpha, small vol → improvement marginal
Result: [A, B, C] - seems good

But what about [A, B, E]?
A + B + E have:
  - High Sharpe from A & B (1.75 and 1.4)
  - Near-zero volatility from E (1%)
  - Less correlation between B and E vs B and C
Result: [A, B, E] could have better Sharpe!

Greedy would never find this because it adds features one-by-one.
```

### The Solution: Quadratic Programming

**Concept**: Directly optimize the Sharpe ratio of the ensemble

```python
def select_features_qp_sharpe_optimization(feature_indices_candidates,
                                           belief_state,
                                           feature_cov,
                                           target_ensemble_size=5,
                                           max_ensemble_size=10):
    """
    Use quadratic programming to find ensemble that maximizes Sharpe ratio.

    Sharpe = (ensemble_mu * prob) / ensemble_sigma

    This is equivalent to maximizing:
    w^T * mu - 0.5 * lambda * w^T * Cov * w

    subject to: sum(w) = 1, w >= 0 (long-only), w <= w_max
    """
    from scipy.optimize import minimize
    import numpy as np

    n_candidates = len(feature_indices_candidates)

    # Get candidate statistics
    mus = np.array([belief_state.beliefs[i].mu
                    for i in feature_indices_candidates])
    probs = np.array([belief_state.beliefs[i].probability_positive()
                      for i in feature_indices_candidates])

    # Effective alpha = alpha * prob
    effective_alphas = mus * probs

    # Extract candidate submatrix from covariance
    cov_candidates = feature_cov[np.ix_(feature_indices_candidates,
                                        feature_indices_candidates)]

    def negative_sharpe(w):
        """
        Objective: minimize negative Sharpe (= maximize Sharpe)

        Sharpe = w^T * mu / sqrt(w^T * Cov * w)
        """
        portfolio_return = w @ effective_alphas
        portfolio_var = w @ cov_candidates @ w
        portfolio_vol = np.sqrt(portfolio_var + 1e-8)

        sharpe = portfolio_return / portfolio_vol
        return -sharpe  # Minimize negative = maximize

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Sum to 1
    ]

    # Bounds: each weight in [0, 1] but with sparsity penalty
    bounds = [(0, 1/max_ensemble_size) for _ in range(n_candidates)]

    # Try multiple random initializations
    best_w = None
    best_sharpe = -np.inf

    for _ in range(10):
        # Random initialization
        w0 = np.random.dirichlet(np.ones(n_candidates))

        # Optimize
        result = minimize(negative_sharpe, w0,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)

        if result.success:
            sharpe = -result.fun
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_w = result.x

    if best_w is None:
        # Fallback to greedy if optimization fails
        return select_features_greedy_bayesian(belief_state,
                                               np.ones(len(belief_state.beliefs), dtype=bool))

    # Extract selected features (weights > 1% threshold)
    selected = [feature_indices_candidates[i]
                for i, w in enumerate(best_w)
                if w > 0.01]  # Only include if weight > 1%

    # Limit to max ensemble size
    selected = selected[:max_ensemble_size]

    return selected, len(selected)
```

### Why This Works Better

**Example: 5 candidate features**

```
Feature rankings (by individual Sharpe):
A: mu=0.03, sigma=0.02, prob=0.60 → Sharpe=0.90
B: mu=0.02, sigma=0.01, prob=0.65 → Sharpe=1.30  ← Best individual
C: mu=0.015, sigma=0.01, prob=0.55 → Sharpe=0.82
D: mu=0.01, sigma=0.005, prob=0.50 → Sharpe=1.00
E: mu=0.005, sigma=0.015, prob=0.70 → Sharpe=0.23  ← Worst individual

Correlation structure:
     A     B     C     D     E
A   1.00  0.15  0.10  0.05  -0.3   ← E is uncorrelated!
B   0.15  1.00  0.80  0.20  -0.1
C   0.10  0.80  1.00  0.15  -0.2
D   0.05  0.20  0.15  1.00  -0.25
E  -0.3  -0.1  -0.2  -0.25  1.00

Greedy approach:
1. Start with B (best individual Sharpe = 1.30)
2. Add C (high correlation with B but still improves)
3. Add A (improves)
4. Add D (marginal improvement)
5. Stop (E looks bad individually)
Result: [B, C, A, D]
Ensemble Sharpe: ~1.1

QP Optimization:
"Let me find the BEST combination of 5 features for Sharpe"

Optimal: [B, A, E]
Why? E is negatively correlated with others!
- B & A provide high alpha (1.3 + 0.9 = 2.2)
- E provides negative correlation → diversification benefit
- 3-feature ensemble: Sharpe = 1.35 > greedy's 1.1

Result: Better risk-adjusted returns ✓
```

### Implementation Effort

**Difficulty**: MEDIUM-HIGH
- Requires scipy.optimize (already installed)
- Need feature covariance matrix (from section 2)
- Need to handle numerical stability (portfolio variance can go negative with bad correlation matrix)

**Computational cost**:
- QP optimization: 10-100ms per ensemble
- Greedy: 10-20ms per ensemble
- Impact: Step 7 backtest takes ~10-15% longer (negligible)

**Expected impact**: +1-3% alpha gain

**Why**:
- Better diversified ensembles (uses negatively correlated features)
- Directly optimizes Sharpe (our goal)
- Can find non-obvious feature combinations

### Implementation Path

```
1. Add feature correlation computation (Section 2)
2. Modify select_features_bayesian to call QP optimizer instead of greedy
3. Keep greedy as fallback for robustness
4. Run step 7 with QP ensemble selection
5. Compare to greedy results
```

---

## SUMMARY & RECOMMENDATIONS

### Quick Impact Assessment

| Improvement | Effort | Expected Gain | Time to Test |
|------------|--------|--------------|-------------|
| **Z-Score Ranking** | 30 min | +0.5-1.0% | 1 hour |
| **Correlation Modeling** | 2-3 hours | +1-3% | 4-5 hours |
| **QP Optimization** | 4-6 hours | +1-3% | 6-8 hours |

### Recommended Priority

**PHASE 1 (This week)**:
1. ✅ Run 5M MC validation (already planned)
2. Test Z-Score ranking (quickest win)
3. Pick best ranking method

**PHASE 2 (Next week)**:
4. Implement feature correlation modeling
5. Rerun step 7 with correlation-adjusted utility
6. Compare results

**PHASE 3 (Optional)**:
7. Implement QP ensemble selection
8. A/B test vs correlation-adjusted greedy

### Expected Total Gain

- **Current**: 34.3% alpha (N=5, all-features with 1.5M MC)
- **After Z-Score**: ~35.0% (+0.7%)
- **After Correlation**: ~36.5% (+1.5%)
- **After QP (optional)**: ~37.5% (+1.0%)

**Target**: Reach or exceed original 36.5% alpha while using all 7,618 features
