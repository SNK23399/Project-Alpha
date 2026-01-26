# Sharpe Optimization Project: Pure Sharpe Throughout the Pipeline

Welcome to the **Sharpe Optimization** project folder. This folder contains a refactored version of the ETF portfolio optimization pipeline that optimizes for **Sharpe ratio throughout**, rather than mixing alpha and hybrid metrics.

---

## 📋 Project Overview

### The Problem We're Solving

The current all-features pipeline has **inconsistent optimization metrics**:
- **Steps 4-5**: Optimize pure alpha (ignore volatility)
- **Step 8**: Optimize hybrid metric (alpha + hitrate blend)

This inconsistency means we're not truly optimizing for Sharpe ratio, which should be the primary goal.

### The Solution

Refactor the pipeline to use **pure Sharpe optimization** throughout:
- **Step 4**: Rank by volatility-adjusted signal (z-score)
- **Step 5**: Compute portfolio Sharpe (not just alpha)
- **Step 6**: Output Sharpe statistics from MC
- **Step 8**: Optimize pure Sharpe ratio

---

## 📁 Documentation Files

Start here to understand the project:

### 1. **SHARPE_OPTIMIZATION_PROJECT.md** ⭐ START HERE
   - **What**: Complete project plan and implementation guide
   - **Contains**:
     - Current state vs target state
     - 4 phases of implementation
     - Timeline and milestones
     - Testing strategy
     - Risk mitigation
   - **Use this when**: You need the full context or detailed implementation guide

### 2. **PROGRESS_TRACKER.md**
   - **What**: Status tracking and decision log
   - **Contains**:
     - Checklist for each phase
     - Data files status
     - Decisions to be made
     - Known issues and solutions
   - **Use this when**: You want to see what's done and what's next

### 3. **QUICK_REFERENCE_CHANGES.md**
   - **What**: Quick summary of exactly what changes in each phase
   - **Contains**:
     - Before/after code for each phase
     - Data flow comparison
     - Hyperparameter changes
     - Testing checklist
   - **Use this when**: You're implementing and need to see specific code changes

---

## 🚀 Quick Start

### If you're new to this project:
1. Read: **SHARPE_OPTIMIZATION_PROJECT.md** (overview + plan)
2. Check: **PROGRESS_TRACKER.md** (what's done)
3. Reference: **QUICK_REFERENCE_CHANGES.md** (while implementing)

### If you're continuing work:
1. Check: **PROGRESS_TRACKER.md** (what's current status)
2. Reference: **QUICK_REFERENCE_CHANGES.md** (for the phase you're on)
3. Implement: The corresponding script (`4_*.py`, `5_*.py`, `6_*.py`, `8_*.py`)

---

## 📊 Project Phases

### Phase 1: Step 4 Refactoring (Z-Score Ranking)
**File**: `4_compute_forward_alpha_sharpe.py`
**Status**: Ready to implement
**Key change**: Add z-score normalization before percentile ranking
- Input: `rankings_matrix_all_1month.npz` (from all-features folder)
- Output: `rankings_matrix_sharpe_1month.npz` (to sharpe folder)

### Phase 2: Step 5 Refactoring (Portfolio Sharpe)
**File**: `5_precompute_feature_sharpe.py`
**Status**: Ready to implement
**Key change**: Compute portfolio Sharpe instead of just alpha
- Input: `rankings_matrix_sharpe_1month.npz` (from Phase 1)
- Output: `feature_sharpe_all_1month.npz` (to sharpe folder)

### Phase 3: Step 6 Refactoring (MC Sharpe Statistics)
**File**: `6_precompute_mc_hitrates_sharpe.py`
**Status**: Ready to implement
**Key change**: Output Sharpe statistics from Monte Carlo
- Input: `feature_sharpe_all_1month.npz` (from Phase 2)
- Output: `mc_sharpe_mean.npz` (to sharpe folder)

### Phase 4: Step 8 Refactoring (Bayesian Sharpe Optimization)
**File**: `bayesian_strategy_proper_allfeatures.py` (or `8_bayesian_strategy_sharpe.py` when created)
**Status**: Ready to implement
**Key change**: Optimize pure Sharpe instead of hybrid metric
- Input: `mc_sharpe_mean.npz` (from Phase 3)
- Output: `bayesian_backtest_summary_sharpe.csv` (comparison to old approach)

---

## 🔗 Related Documentation

For deeper understanding of the problem, see the all-features folder:

- **PIPELINE_OPTIMIZATION_METRICS_ANALYSIS.md** - Detailed analysis of current inconsistencies
- **OPTIMIZATION_INCONSISTENCY_FOUND.md** - How we discovered the hybrid metric
- **CORRELATION_MATRIX_OPPORTUNITY.md** - Why we need correlation matrices for portfolio Sharpe

---

## ✅ Success Criteria

### Minimum Success
- [ ] All 4 phases implemented without errors
- [ ] Backtest completes and produces results
- [ ] Results are interpretable (alpha, hitrate, Sharpe)

### Target Success
- [ ] Alpha maintains (34.3% or better)
- [ ] Hit rate maintains (91.5% or better)
- [ ] Sharpe improves (1.185 or better)
- [ ] Pipeline consistency is clear

### Stretch Success
- [ ] Alpha improves by 0.5%+ (34.8%+)
- [ ] Hit rate improves by 1%+ (92.5%+)
- [ ] Sharpe improves by 0.05+ (1.235+)

---

## 🗂️ Folder Structure

```
walk forward backtest sharpe/
├── README.md                          (this file)
├── SHARPE_OPTIMIZATION_PROJECT.md     (detailed plan)
├── PROGRESS_TRACKER.md                (status tracking)
├── QUICK_REFERENCE_CHANGES.md         (implementation guide)
│
├── Scripts (to implement):
├── 4_compute_forward_alpha_sharpe.py      [Phase 1]
├── 5_precompute_feature_sharpe.py         [Phase 2]
├── 6_precompute_mc_hitrates_sharpe.py     [Phase 3]
├── bayesian_strategy_proper_allfeatures.py [Phase 4]
│
└── data/                              (output directory)
    └── (will contain output files from each phase)
```

---

## 🔄 Data Flow

```
Input (from ../walk forward backtest all features/data/):
├── rankings_matrix_all_1month.npz
├── feature_alpha_all_1month.npz
└── mc_hitrates_all_1month.npz

Phase 1: Step 4 (Z-Score Ranking)
├── Input: rankings_matrix_all_1month.npz
└── Output: rankings_matrix_sharpe_1month.npz

Phase 2: Step 5 (Portfolio Sharpe)
├── Input: rankings_matrix_sharpe_1month.npz
└── Output: feature_sharpe_all_1month.npz

Phase 3: Step 6 (MC Sharpe Statistics)
├── Input: feature_sharpe_all_1month.npz
└── Output: mc_sharpe_mean.npz, mc_alpha_mean.npz, mc_alpha_std.npz

Phase 4: Step 8 (Bayesian Optimization)
├── Input: mc_sharpe_mean.npz
└── Output: bayesian_backtest_summary_sharpe.csv

Comparison: Old vs New
└── Compare to: ../walk forward backtest all features/data/bayesian_backtest_summary.csv
```

---

## 🎯 Key Decisions

### Decision 1: Full Pipeline or Phases 1-2 First?
**Recommendation**: Start with Phases 1-2, evaluate results, then decide on 3-4
- Lower risk to implement just rankings and alpha first
- Can validate changes before full optimization refactor

### Decision 2: Correlation Matrix Handling?
**Recommendation**: Compute on-the-fly per month with Ledoit-Wolf shrinkage
- More efficient than pre-computing all correlations
- Handles estimation error with shrinkage

### Decision 3: Keep Hitrate as Secondary?
**Recommendation**: Start with pure Sharpe, add hitrate if results degrade
- Simplest first, enhance if needed
- Can always revert to hybrid if better

---

## 📈 Expected Outcomes

### Best Case (Pure Sharpe is Better)
```
Current (Hybrid):
  Alpha: 34.3%
  Hit rate: 91.5%
  Sharpe: 1.185

New (Pure Sharpe):
  Alpha: 34.8%+
  Hit rate: 92.0%+
  Sharpe: 1.25+
```

### Likely Case (Similar Performance)
```
New approach gives similar results, but:
  ✓ Pipeline is now consistent
  ✓ Code is cleaner (no hybrid metric)
  ✓ Future improvements easier
```

### Worst Case (Hybrid is Actually Better)
```
New approach gives worse results, but:
  ✓ We understand why hybrid works
  ✓ Valuable learning about Sharpe vs magnitude
  ✓ Easy to revert to old approach
```

---

## 🛠️ Implementation Workflow

### For each phase:

1. **Plan**: Read SHARPE_OPTIMIZATION_PROJECT.md for that phase
2. **Reference**: Check QUICK_REFERENCE_CHANGES.md for exact code changes
3. **Implement**: Modify the corresponding script
4. **Test**: Run tests specified in project plan
5. **Validate**: Check output files are created correctly
6. **Document**: Update PROGRESS_TRACKER.md

---

## 🔧 Troubleshooting

### "Correlation matrix is invalid"
→ Use Ledoit-Wolf shrinkage estimator
→ Check eigenvalues are positive
→ Fall back to uncorrelated assumption if needed

### "Sharpe values are negative"
→ Verify alpha is positive (it should be from filtered features)
→ Check volatility computation isn't zero-dividing
→ Validate MC output

### "Results got worse"
→ This is expected if current hybrid metric is better
→ Analyze why: which features changed?
→ Consider keeping both approaches available

---

## 📚 References & Related Files

### In this folder:
- **SHARPE_OPTIMIZATION_PROJECT.md** - Detailed plan
- **PROGRESS_TRACKER.md** - Status and decisions
- **QUICK_REFERENCE_CHANGES.md** - Implementation guide

### In all-features folder (for context):
- **PIPELINE_OPTIMIZATION_METRICS_ANALYSIS.md** - Problem analysis
- **OPTIMIZATION_INCONSISTENCY_FOUND.md** - Discovery of inconsistency
- **CORRELATION_MATRIX_OPPORTUNITY.md** - Correlation usage

### Original documentation:
- **BAYESIAN_APPROACH_EXPLANATION.md** - How Bayesian strategy works
- **HYBRID_METRIC_SYNTHESIS.md** - How current hybrid metric emerged

---

## ✉️ Status & Updates

**Project Created**: 2026-01-22
**Current Phase**: Planning & Documentation Complete
**Next Step**: Phase 1 Implementation

**To track progress**: See **PROGRESS_TRACKER.md**

---

## 🎓 Learning Outcomes

By completing this project, we'll learn:
1. ✓ Whether pure Sharpe optimization is better than hybrid
2. ✓ How correlation structure affects ensemble selection
3. ✓ How to compute portfolio Sharpe efficiently
4. ✓ When balanced objectives (hybrid) outperform pure optimization

---

Good luck! Start with **SHARPE_OPTIMIZATION_PROJECT.md** and **PROGRESS_TRACKER.md** for orientation.
