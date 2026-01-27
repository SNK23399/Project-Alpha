# Walk-Forward Validation Framework

This directory contains tools to validate the Core-Satellite portfolio strategy for **future bias** through walk-forward analysis.

## Overview

The framework simulates deploying the pipeline on historical dates, using only data available at that time. This detects any look-ahead bias that might have been introduced during feature engineering or signal computation.

### Key Concept

Instead of backtesting on all data at once (which could leak future information), we:
1. **Truncate the database** to each historical month
2. **Run the full pipeline** as if deployed at that time
3. **Store satellite selections** separately for each month
4. **Compare** historical predictions to current results
5. **Detect** systematic bias if predictions are too optimistic

## Workflow

### Step 1: Run Walk-Forward Validation

```bash
cd validation
./validate_walk_forward.sh --start-date 2014-01-31 --end-date 2025-12-31 --step 1
```

This script:
- Creates database backups for each month
- Truncates the database to only include data available at that time
- Runs the full pipeline (Stages 1-6) with truncated data
- Saves predictions to `historical_predictions/YYYY-MM-DD/`
- Restores the original database after each run

**Arguments:**
- `--start-date YYYY-MM-DD`: First month to validate
- `--end-date YYYY-MM-DD`: Last month to validate
- `--step N`: Process every N months (default: 1)
- `--only-date YYYY-MM-DD`: Process single month only

**Examples:**
```bash
# Full 2009-2025 validation
./validate_walk_forward.sh --start-date 2009-09-30 --end-date 2025-12-31 --step 1

# Just 2020
./validate_walk_forward.sh --start-date 2020-01-31 --end-date 2020-12-31

# Every 3 months (faster)
./validate_walk_forward.sh --start-date 2014-01-31 --end-date 2025-12-31 --step 3

# Single date test
./validate_walk_forward.sh --start-date 2020-01-31 --end-date 2020-12-31 --only-date 2020-01-31
```

### Step 2: Compare Results

After validation completes, compare historical predictions to current results:

```bash
python compare_predictions.py ./historical_predictions
```

This generates a report showing:
- **Satellite match accuracy**: Which % of predicted satellites appear in current results
- **Alpha differences**: How much better/worse predictions were vs current
- **Look-ahead bias detection**: Are historical results systematically too optimistic?
- **Conclusion**: Is the validation methodology sound?

## File Descriptions

### `truncate_database.py`

Creates a time-bounded copy of the ETF database.

**Usage:**
```bash
python truncate_database.py <source_db> <target_date> <output_db>
```

**Parameters:**
- `source_db`: Path to full ETF database
- `target_date`: YYYY-MM-DD date (data up to this date is kept)
- `output_db`: Where to save truncated database

**Example:**
```bash
python truncate_database.py ../maintenance/data/etf_database.db 2020-01-31 db_truncated_2020-01-31.db
```

**Output:**
- Copies full database
- Deletes prices after target_date
- Deletes metadata snapshots after target_date
- Shows statistics on records removed

### `validate_walk_forward.sh`

Orchestrates end-to-end walk-forward validation.

**Workflow:**
```
For each month from start_date to end_date:
  1. Truncate database to target_date
  2. Swap truncated DB into pipeline
  3. Run full pipeline (Stages 1-6)
  4. Save results to historical_predictions/
  5. Restore original database
  6. Report progress
```

**Output Structure:**
```
historical_predictions/
├── 2014-01-31/
│   ├── bayesian_backtest_N3.csv
│   ├── bayesian_backtest_N4.csv
│   ├── bayesian_backtest_N5.csv
│   ├── bayesian_backtest_N6.csv
│   ├── bayesian_backtest_N7.csv
│   └── metadata.txt
├── 2014-02-28/
├── 2014-03-31/
└── ... (one per month)
```

### `compare_predictions.py`

Analyzes walk-forward results for bias.

**Usage:**
```bash
python compare_predictions.py /path/to/historical_predictions
```

**Output Sections:**

1. **Overall Statistics**
   - Number of validation dates
   - Number of overlapping dates with current results
   - Satellite match percentage (mean ± std)
   - Average alpha difference (mean ± std)

2. **Detailed Comparison Table**
   - For each month and N value
   - Predicted vs current satellite selections
   - Match count and percentage
   - Alpha difference

3. **Look-Ahead Bias Analysis**
   - Mean alpha difference per N value
   - Statistical significance test
   - Detection of systematic positive bias
   - Recommendation on data integrity

4. **Conclusion**
   - ✓ No bias detected → Validation sound
   - ⚠️ Bias detected → Investigate feature computation

## Interpreting Results

### Satellite Match Percentage

The % of predicted satellites that appear in current results.

- **100%**: Perfect match (expected for simple selections)
- **80-90%**: Good match (normal variation)
- **<50%**: Significant drift (investigate)

### Alpha Difference

`Current Avg Alpha - Historical Avg Alpha`

- **Close to 0 (±0.001)**: No systematic bias ✓
- **Consistently positive**: Current results are better
  - Could indicate look-ahead bias (historical predictions too pessimistic)
  - Or natural improvement with more data ✓
- **Consistently negative**: Current results worse
  - Suggests historical predictions were over-optimistic
  - ⚠️ Possible look-ahead bias

### Look-Ahead Bias Interpretation

**No bias (✓):**
- Historical predictions align with current results
- Historical simulations used only appropriate data
- Backtesting results reliable

**Bias detected (⚠️):**
- Historical predictions systematically differ from current
- Possible causes:
  1. Features computed on data that shouldn't be available
  2. Signal bases use forward-looking information
  3. Database didn't properly exclude future data
  4. Price data had look-ahead issues

## Implementation Details

### Database Truncation

The truncation process:
1. Copies full database to temporary location
2. Deletes all price records after target_date
3. Deletes metadata history records after target_date
4. Optimizes database (VACUUM)
5. Reports statistics

This ensures the pipeline only "sees" data available at that time.

### Pipeline Execution

The validation script runs the full pipeline with truncated data:
- **Stage 1**: Computes forward IR on truncated data
- **Stage 2**: Computes signal bases with truncated prices
- **Stage 3**: Applies filters
- **Stage 4**: Precomputes feature-IR matrix
- **Stage 5**: Precomputes MC IR stats (Bayesian priors)
- **Stage 6**: Runs Bayesian satellite selection

Each stage uses only data up to the target date.

### Database Restoration

After each month:
1. Pipeline results are saved
2. Truncated database is deleted
3. Original database is restored
4. Next month proceeds with clean state

This ensures no cross-contamination between months.

## Performance Considerations

### Time Estimates

- **Single month**: ~30-60 seconds (depending on pipeline)
- **Full year (2020)**: ~8-15 minutes
- **Full history (2014-2025, ~130 months)**: ~2-4 hours
- **With --step 3 (quarterly)**: ~1 hour

### Disk Space

- Each truncated database: ~100-200 MB
- Historical predictions storage: ~50 MB per month
- Total for full validation: ~10 GB (backups are cleaned up)

### Recommendations

- Start with `--step 3` (quarterly) for initial testing
- Use `--only-date` to test specific months
- Run full validation `--step 1` only after confirming no issues

## Validation Checklist

- [ ] Run `./validate_walk_forward.sh` with test date
- [ ] Verify `historical_predictions/` directory is created
- [ ] Check that predictions CSV files are generated
- [ ] Run `python compare_predictions.py ./historical_predictions`
- [ ] Review bias analysis report
- [ ] Verify "no systematic bias detected" ✓
- [ ] Compare allocation outputs between historical and current

## Troubleshooting

### Database File Not Found
```
ERROR: Database not found: /path/to/database.db
```
Make sure the database exists at `../maintenance/data/etf_database.db`

### Pipeline Failed for Date
```
ERROR: Pipeline failed for 2020-01-31
```
Check `/tmp/pipeline_2020_01_31.log` for details. Pipeline may have failed due to:
- Missing data for that month
- Insufficient historical data for signal computation
- Pipeline code issues (should be fixed in main branch)

### No Overlapping Dates
```
WARNING: No overlapping dates between historical and current results
```
This happens if:
- Historical validation didn't complete
- Historical predictions are for different date range
- Current results were regenerated with different date range

### Memory Issues

If running out of memory during walk-forward:
- Use `--step 3` or `--step 6` (less frequent)
- Reduce pipeline parallelism (edit main.py)
- Run on machine with more RAM

## Next Steps After Validation

If validation confirms no look-ahead bias:

1. **Deploy strategy**: Use current backtest results with confidence
2. **Monthly updates**: Run pipeline with new month's data
3. **Allocation generation**: Run Stage 9 with latest satellite selections
4. **Periodic re-validation**: Re-run walk-forward every 6-12 months

If bias is detected:

1. **Identify source**: Check which stage is leaking future data
2. **Review code**: Look for forward-looking features or signals
3. **Fix issues**: Adjust feature engineering or data access
4. **Re-validate**: Run walk-forward again to confirm fix
5. **Update strategy**: Use corrected backtest results

## References

- Walk-forward validation: https://en.wikipedia.org/wiki/Walk_forward_optimization
- Look-ahead bias: https://en.wikipedia.org/wiki/Look-ahead_bias
- Backtesting pitfalls: https://en.wikipedia.org/wiki/Backtesting#Pitfalls
