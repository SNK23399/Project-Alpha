# Proper Out-of-Sample Backtesting Plan

## Current Situation vs What We Need

### What Script 4 & 5 Currently Do:
1. Calculate forward alpha (future returns - core returns)
2. Calculate features (momentum, volatility, etc.)
3. Find which features CORRELATE with future alpha
4. Report: "If we selected top N ETFs by these features, average alpha would be 3.06%"

**Problem**: This is NOT a real backtest! You're measuring correlation, not actual trading performance.

### What Proper Backtesting Should Do:
1. Start at date T
2. Calculate features using ONLY data up to date T
3. Rank ETFs by features (like your ensemble does)
4. Select top N ETFs
5. Hold for 1 month
6. Measure ACTUAL returns during that month
7. Compare to core benchmark
8. Move to next month and repeat

This is **walk-forward backtesting** - you never use future data.

## Implementation Plan

### Option A: Use Script 6 (Out-of-Sample Validation) - EXISTS!

You already have a script for this! Let me check if it does proper backtesting:

**File**: `7_out_of_sample_validation.py`

This should:
- Split data into train/test periods
- Select ETFs using only training data
- Measure performance on test period
- Report realistic alpha

### Option B: Create New Backtesting Script

If script 7 doesn't do full backtesting, create:

**File**: `backtest_consensus_strategy.py`

**Workflow**:
```
For each month in 2015-2023:
  1. Load ensemble rankings (already computed in script 4)
  2. Apply consensus method (from script 5)
  3. Get top N ISINs
  4. Look up ACTUAL 1-month forward returns
  5. Calculate:
     - Portfolio return (average of N satellites)
     - Core return
     - Alpha = Portfolio - Core
  6. Track cumulative performance

Report:
  - Cumulative alpha
  - Hit rate (% positive alpha months)
  - Sharpe ratio
  - Drawdowns
  - Year-by-year performance
```

## Key Differences from Current Analysis

### Current (Script 4 & 5):
- "These features correlate with alpha"
- "Average alpha of top-ranked ETFs was 3.06%"
- ❌ NOT testing actual trading

### Proper Backtest:
- "I selected these ETFs based on features"
- "They actually returned X% that month"
- ✅ Simulates real trading

## Expected Results

### Realistic Scenarios:

**Optimistic (Features work well)**:
- Predicted alpha: 3.06% monthly
- Actual alpha: 2.0-2.5% monthly (66-82% of predicted)
- Annual alpha: 24-30%
- Hit rate: Still ~85-90%

**Realistic (Features work moderately)**:
- Predicted alpha: 3.06% monthly
- Actual alpha: 1.0-1.5% monthly (33-50% of predicted)
- Annual alpha: 12-18%
- Hit rate: ~75-80%

**Pessimistic (Features barely work)**:
- Predicted alpha: 3.06% monthly
- Actual alpha: 0.3-0.5% monthly (10-16% of predicted)
- Annual alpha: 3.6-6%
- Hit rate: ~60-65%

### Why the Decay?

1. **Feature lag**: Features based on past data, markets change
2. **Overfitting**: Features optimized on training data
3. **Implementation gap**: Perfect ranking vs real-world execution
4. **Transaction costs**: Not modeled in current analysis
5. **Timing**: Monthly rebalancing has entry/exit timing risk

## What Good Looks Like

A successful strategy would have:
- **Actual alpha > 1% monthly (12%+ annual)**
- **Hit rate > 70%**
- **Sharpe > 0.8**
- **Max drawdown < 20%**
- **Consistent across different time periods**

Even 1% monthly alpha (12% annual) is EXCELLENT for a systematic strategy.

## Implementation Steps

### Step 1: Check Existing Scripts
```bash
# Check if script 7 does proper backtesting
python 7_out_of_sample_validation.py
```

### Step 2: Create Proper Backtest Script
- Load ensemble rankings (from script 4)
- Apply consensus selection (from script 5)
- Measure ACTUAL forward returns
- Compare to core benchmark
- Report realistic metrics

### Step 3: Validate Results
- Split into train/test periods (e.g., 2015-2019 train, 2020-2023 test)
- Check if alpha holds in test period
- Look for overfitting signs

### Step 4: Sensitivity Analysis
- Test different N values (1, 3, 4, 6)
- Test different consensus methods
- Test different rebalancing frequencies
- Check robustness to parameter changes

## Critical Questions to Answer

1. **Does the alpha persist out-of-sample?**
   - If train: 3% monthly, test: 0.5% monthly → Overfitting!
   - If train: 3% monthly, test: 2% monthly → Good!

2. **Is it consistent across time?**
   - Works in 2015-2017 but fails 2020-2023 → Regime change
   - Works in most years → Robust

3. **What's the worst drawdown?**
   - Current analysis doesn't show this
   - Real trading will have losing streaks

4. **How sensitive to parameters?**
   - Change N from 3 to 4: Does alpha disappear? → Unstable
   - Similar alpha across N=2-5 → Robust

## Example Backtest Output

```
WALK-FORWARD BACKTEST RESULTS (2015-2023)
Strategy: N=3 Satellites, Primary_Only Consensus
========================================

Performance:
  Total Return: +287% (vs Core: +156%)
  Annualized Return: 18.2% (vs Core: 11.5%)
  Annualized Alpha: 6.7%
  Monthly Alpha: 0.56% ± 3.2%

Consistency:
  Hit Rate: 78.7% (85/108 months positive)
  Sharpe Ratio: 1.24
  Sortino Ratio: 1.89

Risk:
  Max Drawdown: -18.3%
  Max Drawdown Duration: 7 months

By Year:
  2015: +12.3% alpha
  2016: +8.1% alpha
  2017: +15.2% alpha
  2018: -2.1% alpha  ⚠️ Loss year
  2019: +9.8% alpha
  2020: +6.4% alpha
  2021: +11.2% alpha
  2022: -1.3% alpha  ⚠️ Loss year
  2023: +4.8% alpha
```

This would be REALISTIC and actionable!

## Next Steps

1. **Check if script 7 already does this**
2. **If not, I'll create a proper backtest script**
3. **Run it for N=3 with primary_only**
4. **Compare predicted vs actual alpha**
5. **Make go/no-go decision based on real results**

Should I check script 7 to see what it does, or should we create a new dedicated backtesting script from scratch?
