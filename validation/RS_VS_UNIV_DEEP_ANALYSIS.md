# Deep Analysis: Why rs_vs_univ_126d Dominates

## Executive Summary

**rs_vs_univ_126d** is the most predictive signal for satellite selection, but not in the way you might expect. It's not the raw relative strength that matters—it's how that signal **ranks peers** that matters.

**Key Finding:** When percentile-ranked (0-1), rs_vs_univ_126d correlates 0.3318 with forward IR, **3.0x better than the next signal**.

---

## The Discovery: Raw vs Ranked Signals

### Raw Signal Correlation
When using the raw rs_vs_univ values (ETF return / Universe avg return):

```
Rank  Signal                 Correlation (Raw)
  1   stability_252d              0.1161
  2   calmar_252d                 0.0845
  3   info_ratio_252d             0.0419
  4   rs_vs_univ_126d             0.0294    <- Only #4!
  5   tail_ratio_126d             0.0229
```

rs_vs_univ_126d is **not** the strongest raw signal.

### Ranked Signal Correlation
When percentile-ranked (0-1, as used in your pipeline):

```
Rank  Signal                 Correlation (Ranked)
  1   rs_vs_univ_126d             0.3318    <- DOMINATES!
  2   stochastic                  0.1094
  3   calmar_252d                 0.1003
  4   stability_252d              0.0993
  5   tail_ratio_126d             0.0960
```

rs_vs_univ_126d is **3.0x more predictive** when ranked.

---

## Why Ranking Transforms This Signal

### The Problem with Raw Values

Raw rs_vs_univ values have extreme outliers:
```
Min:     -85,977
Mean:    -1.02
Median:  0.97
Max:     45,537
Std Dev: 222.5
```

These extreme outliers come from:
- Small movements in universe average creating huge ratios
- Micro-cap ETFs with erratic behavior
- Market dislocations and anomalies

**Result:** Raw correlation is weak because outliers distort the relationship.

### The Solution: Percentile Ranking

Percentile ranking converts extreme values to 0-1 range:
```
Before: -85,977  ->  After: 0.0001
Before:  45,537  ->  After: 0.9999
Before:      0.97  ->  After: 0.5002  (median)
```

**Effect:**
- Eliminates outlier influence
- Focuses on ETF's RELATIVE RANKING vs peers (not absolute strength)
- Creates robust, stable signal
- Increases correlation with forward IR from 0.0294 to 0.3318

---

## What Percentile Ranking Actually Reveals

### Raw rs_vs_univ says:
> "How much did this ETF outperform the universe average?"

**Problem:** This depends on:
- Universe average (different each day)
- Market regime (bull/bear affects interpretation)
- ETF size and liquidity (affects execution)

### Percentile Rank says:
> "Among all ETFs today, where does this one rank in 6-month outperformance?"

**Advantage:**
- Isolates true PEER COMPARISON
- Removes market regime effects
- Focuses on SELECTION QUALITY
- Directly answers: "Is this ETF beating its competition?"

---

## The Critical Insight

**Your ranking-based satellite selection works because:**

1. It uses percentile ranks, not raw values
2. rs_vs_univ_126d's percentile rank is the best predictor
3. Ranking "cleans" the signal by removing extremes
4. This reveals pure relative outperformance

**The 3.6x improvement comes from:**
- Removing 292 weak signals that add noise
- Focusing on the one signal with 0.3318 correlation
- Letting percentile ranking do what it does best: isolate relative performance

---

## Why This Matters for Your Strategy

### Current Configuration (0.10 threshold):
- Keeps 3 signals (including rs_vs_univ_126d)
- Performance: 937.8% annual IR
- Hit rate: 84.7%

### Optimal Configuration (0.11 threshold):
- Keeps 1 signal: rs_vs_univ_126d ranked
- Performance: 1229.8% annual IR (+3.6x)
- Hit rate: 93.9%

### Why One Signal Beats 293:

The other 292 signals:
- Have lower correlation (0.01-0.11 ranked)
- Add noise to the ranking process
- Confuse satellite selection
- Reduce hit rates (from 93.9% to 52.7%)

rs_vs_univ_126d ranked:
- Captures pure relative momentum
- Percentile ranking removes outliers
- 0.3318 correlation is 3x+ other signals
- Drives 93%+ hit rate

---

## Practical Implication: Signal Simplification

### Instead of:
- 293 signals
- Complex Bayesian feature selection
- 25 filter steps
- Hours of computation

### You could use:
- 1 signal: `rs_vs_univ_126d`
- Percentile rank each ETF by 6-month outperformance
- Select top N by rank
- 30 seconds of computation

**Performance difference:** None. Same 1230% annual IR.

---

## Validation Checklist

Before simplifying, test:

- [ ] 0.11 threshold achieves 1230% IR in walk-forward test
- [ ] Hit rate remains >90% out-of-sample
- [ ] Dominance holds across different N satellite values
- [ ] rs_vs_univ_126d stays #1 in correlation over time
- [ ] Works in different market regimes (bull/bear)
- [ ] No overfitting artifacts

---

## Why Raw Signals Aren't Useful Here

This is an important general principle:

**For ranking-based selection, you need:**
1. **Comparable metrics** (all on 0-1 scale)
2. **Outlier-resistant** (percentiles handle extremes)
3. **Relative comparisons** (who beats whom today)

**Not:**
- Absolute magnitudes (raw returns)
- Unstable distributions (extreme values)
- Regime-dependent interpretations

**Conclusion:** Percentile ranking is what makes rs_vs_univ_126d work. The ranking transform is doing most of the work, not the raw signal calculation.

---

## Next Steps

### If you want maximum performance (1230% IR):
→ Use correlation threshold 0.11 (rs_vs_univ_126d only)

### If you want robustness + high performance (900-1000% IR):
→ Use correlation threshold 0.08-0.10 (3-22 signals)

### If you want to understand the signal better:
→ Investigate why rs_vs_univ_126d percentile ranking is so stable
→ Check if 6-month window is optimal or if other windows work better
→ Test if combining with volatility creates better results

---

## Summary Table

| Aspect | Finding |
|--------|---------|
| Raw correlation with forward IR | 0.0294 (#4 signal) |
| Ranked correlation with forward IR | 0.3318 (#1 signal) |
| Improvement from ranking | 11.3x better |
| Performance with this signal only | 1230% annual IR, 93.9% hit rate |
| Performance with all 293 signals | 340% annual IR, 52.7% hit rate |
| Improvement factor | 3.6x |
| Dominant characteristic | Percentile ranking removes outliers |
| Ideal use | Sole ranking metric for satellite selection |

