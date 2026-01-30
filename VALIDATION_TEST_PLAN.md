# Adaptive Convergence Validation Test Plan

## Objective
Verify that adaptive convergence at 0.1% threshold ensures **true convergence independent of random seeding**.

## Test Configuration

**pipeline_copy settings:**
```python
MC_SEED = None  # NO SEEDING - random samples differ each run
MC_CONVERGENCE_THRESHOLD = 0.001  # 0.1% threshold
MC_MIN_SAMPLES = 1_000_000
MC_MAX_SAMPLES = 100_000_000
MC_BATCH_SIZE = 500_000
```

## Hypothesis

**If adaptive convergence works correctly:**
- Run 1 (random seed A): Different random samples → same priors → same satellites selected
- Run 2 (random seed B): Different random samples → same priors → same satellites selected
- Run 3 (random seed C): Different random samples → same priors → same satellites selected

**Result:** All three runs select identical satellites (or extremely close)

## Test Execution

### Test 1: Full Pipeline Run (No Seeding)
```bash
cd pipeline_copy
python ../main.py --steps 1,2,3,4,5,6
# Wait for completion, note the satellite selections
```

### Test 2: Repeat Run (Different Random Samples)
```bash
# Run again - CuPy will use different random seed
python ../main.py --steps 1,2,3,4,5,6
# Compare satellite selections to Test 1
```

### Test 3: Third Run (Further Validation)
```bash
# Third run with yet another random seed
python ../main.py --steps 1,2,3,4,5,6
# Compare to previous runs
```

## Success Criteria

### Metric 1: Prior Stability (Convergence)
**Goal**: MC priors converge at 0.1% threshold
- ✓ Step 5 output shows: "CONVERGED! (change <0.1%)"
- ✓ Samples used: 40-50M as expected
- ✓ Convergence happens for all N values (1-5)

### Metric 2: Satellite Selection Consistency
**Goal**: All runs select identical or very similar satellites

**Check for each month:**
- Count matching satellites across runs
- Expected: 100% match (5/5 satellites same)
- Acceptable: 80%+ match (4/5 satellites same)
- Failure: <80% match

**Summary statistics:**
```
Satellite Agreement:
  Run 1 vs Run 2: X% match (Y/5 satellites)
  Run 1 vs Run 3: X% match (Y/5 satellites)
  Run 2 vs Run 3: X% match (Y/5 satellites)
  Average agreement: X%
```

### Metric 3: Performance (Should be similar)
**Goal**: All runs produce similar backtest results

**Metrics to compare:**
- Mean monthly alpha (should be identical or <0.01% difference)
- Hit rate (should be identical)
- Sharpe ratio (should be <1% difference)
- Number of test periods (should be identical)

## Comparison: Original Pipeline (for reference)

Original pipeline runs with MC_SEED = 42 (seeded):
- Deterministic by design (seeding forces same results)
- Masked the underlying convergence issue
- 3M fixed samples (insufficient for true convergence)

**Expected difference from adaptive:**
- Original: 3M samples, priors change 10% between runs (hidden by seeding)
- Adaptive: 40-50M samples, priors truly converge at <0.1%

## Interpretation

### If Test Succeeds (Satellites match 100% or >95%)
- ✓ Adaptive convergence truly works
- ✓ Different random samples → same priors (as intended)
- ✓ Safe to use without seeding
- ✓ Solution addresses root cause, not masking it

### If Test Fails (Satellites match <80%)
- ✗ 0.1% threshold may be insufficient
- ✗ May need to investigate signal quality or increase threshold
- ✗ Consider alternatives: 0.05% threshold, or signal parameter improvements

## Expected Timeline

| Task | Time |
|------|------|
| Run 1 (full pipeline) | ~2-3 hours |
| Run 2 (full pipeline) | ~2-3 hours |
| Run 3 (full pipeline) | ~2-3 hours |
| Analysis & comparison | ~1 hour |
| **Total** | **~7-10 hours** |

## Output Files to Compare

After each run, check:
- `pipeline_copy/data/mc_ir_mean_1month.npz` - Prior means (compare across runs)
- `pipeline_copy/data/backtest_results/bayesian_backtest_*.csv` - Satellite selections

### Comparison Script (to be created)
```python
# Load results from all 3 runs
results_run1 = pd.read_csv("backtest_results_run1.csv")
results_run2 = pd.read_csv("backtest_results_run2.csv")
results_run3 = pd.read_csv("backtest_results_run3.csv")

# Compare satellite selections
agreement_1_2 = (results_run1['selected_isins'] == results_run2['selected_isins']).mean()
agreement_1_3 = (results_run1['selected_isins'] == results_run3['selected_isins']).mean()
agreement_2_3 = (results_run2['selected_isins'] == results_run3['selected_isins']).mean()

# Compare performance metrics
alpha_diff_1_2 = abs(results_run1['monthly_alpha'].mean() - results_run2['monthly_alpha'].mean())
alpha_diff_1_3 = abs(results_run1['monthly_alpha'].mean() - results_run3['monthly_alpha'].mean())
```

## Decision Logic

**After results:**
- If >95% agreement on satellites AND alphas match (±0.01%):
  - ✓ **ADOPT** pipeline_copy with adaptive convergence
  - Use 0.1% threshold in production
  - Remove MC_SEED entirely

- If 80-95% agreement:
  - ? Consider tightening threshold to 0.05%
  - ? Or investigate signal quality improvements

- If <80% agreement:
  - ✗ Revert to seeded 3M approach
  - Investigate why convergence insufficient

---

## Status
- **Configuration**: Ready (MC_SEED = None, 0.1% threshold)
- **Data**: Available (Steps 1-4 complete in pipeline_copy)
- **Next**: Execute full pipeline runs 1-3
