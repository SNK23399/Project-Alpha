# Pipeline Analysis & Validation Notes

## STEP 1: Compute Forward IR (`1_compute_forward_ir.py`)

### Purpose
Computes the **target variable** for the entire system: Forward Information Ratio (IR).
This is NOT a signal generator - it calculates actual realized returns vs the benchmark.

### What It Does (High Level)
For each ETF and each month:
1. Calculate the **forward return**: ETF return over the month
2. Calculate **alpha**: ETF return minus MSCI World return (core benchmark)
3. Calculate **realized volatility**: standard deviation of daily alphas DURING that month
4. Calculate **forward IR**: alpha / realized volatility

### Key Inputs
- **ETF Price Database**: SQLite database at `maintenance/data/etf_database.db`
  - Contains daily closing prices for 869+ ETFs since 2009
  - Includes fund size (AUM) metadata for each ETF

- **Core Benchmark ISIN**: `IE00B4L5Y983` (iShares Core MSCI World)
  - All alphas are computed relative to this benchmark
  - The "Core" part of Core-Satellite

### Computational Flow

#### 1. Universe Filtering
- Load all ETFs from database
- **Apply AUM Filter**: Keep only ETFs with fund size >= 75M EUR
  - Reduces survivor bias (small ETFs are more likely to be delisted)
  - Excludes ~X% of ETFs
  - **CRITICAL**: Core ISIN must pass this filter or script fails

#### 2. Load Price Data
- Parallel load of price data for all filtered ETFs (32 threads)
- Convert to aligned DataFrame: (days × ETFs)
- Handle index conversion to datetime

#### 3. Generate Monthly Periods
- Start: 2015-01-01 (or actual data start, whichever is later)
- End: Current date extended by HOLDING_MONTHS
- Maps theoretical month-ends to actual trading dates (handles weekends/holidays)

#### 4. Vectorized Return Calculation
```
For each of N periods:
  - Get start price for each ETF
  - Get end price (1 month forward) for each ETF
  - returns[period, etf] = (end_price / start_price) - 1
  - core_returns[period] = returns[period, CORE_ISIN]
  - alpha[period, etf] = returns[period, etf] - core_returns[period]
```

#### 5. Realized Volatility Calculation (Per-Month Per-ETF)
**IMPORTANT**: This is NOT annualized vol - it's the actual vol realized DURING that month
```
For each period:
  - Get all daily prices from start_date to end_date
  - Compute daily returns: returns[day] = (price[day] / price[day-1]) - 1
  - Compute daily alphas: alphas[day] = returns[day] - benchmark_return[day]
  - volatility = std(daily_alphas)
  - Minimum volatility floor: 1e-8 (to avoid division by zero)
```

#### 6. Information Ratio Calculation
```
forward_ir[period, etf] = forward_alpha[period, etf] / realized_volatility[period, etf]
```

#### 7. Validity Checks
Each (date, isin) observation is kept only if:
- `forward_alpha` is not NaN
- ETF is NOT the core ETF itself
- ETF has sufficient history: first_valid_date <= start_date - (252 days * 1.5)
  - Ensures 18+ months of data available before each prediction

### Output
**File**: `data/forward_alpha_1month.parquet`

**Columns**:
| Column | Type | Meaning |
|--------|------|---------|
| `date` | datetime | Month-end (or next trading day if month-end is weekend) |
| `isin` | str | ETF identifier |
| `forward_return` | float | ETF return for the month |
| `core_return` | float | Core (MSCI World) return for the month |
| `forward_alpha` | float | ETF return - Core return |
| `forward_ir` | float | forward_alpha / realized_volatility |

**Stats from summary**:
- Multiple observations per date (one per valid ETF)
- Covers from ~2015 to current date
- Example: ~500K observations across ~140 months × ~350 unique ETFs

### Configuration Parameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `CORE_ISIN` | `IE00B4L5Y983` | MSCI World benchmark |
| `HOLDING_MONTHS` | 1 | Forward-looking period (1 month) |
| `MIN_HISTORY_DAYS` | 252 | Minimum 1 year of data before a prediction |
| `MIN_AUM_MILLIONS` | 75 | Exclude ETFs with small AUM (survivor bias) |
| `FORCE_RECOMPUTE` | True | Always regenerate (not cached) |

### Critical Implementation Details

1. **Month-End vs Trading Dates**
   - If month-end falls on weekend/holiday, uses next trading day
   - This is important because all signals will reference these same dates

2. **Realized Volatility Definition**
   - NOT simple monthly price std dev
   - It's std dev of **daily alphas** (ETF return - benchmark return) during the month
   - This captures alpha volatility, not absolute volatility
   - **Critical for Step 7 validation**: Must understand this exact calculation

3. **History Requirement**
   - `required_start = start_date - (252 * 1.5) days` ≈ 18 months prior
   - If ETF didn't exist 18 months before this prediction, it's excluded
   - Ensures we only pick ETFs with sufficient track record

4. **NaN Handling**
   - Any NaN values in returns → excluded from results
   - Division by zero protection: volatility floored at 1e-8

### Data Dependencies
- Input: ETF price database (established, static)
- Output: `forward_alpha_1month.parquet` (required input to Steps 2-6)

### Caching Behavior
- If `forward_alpha_1month.parquet` exists AND `FORCE_RECOMPUTE=False` → loads from cache
- Current setting: `FORCE_RECOMPUTE=True` → always recomputes

### For Step 7 Validation
**What matters**:
- This script computes the ground truth target variable (forward IR)
- Step 7 will need to use this same forward IR calculation when predicting for a past month
- If Step 7 uses a different method to compute forward IR, predictions will differ from backtest
- **Key validation point**: Ensure Step 7 can "look back" to any historical month and compute forward IR the same way

---

## STEP 2: Compute Signal Bases (`2_compute_signal_bases.py`)

### Purpose
Generates **raw DPO signal variations** that will be used as features to predict forward IR.
Essentially creates different "momentum oscillators" with varying smoothing and time windows.

### What It Does (High Level)
1. Computes 4 DPO (Detrended Price Oscillator) variants for each ETF
   - 2 periods (21d, 63d) × 2 shifts (responsive, conservative) = 4 signals
2. Each variant: Savgol-smoothed with polyorder 4
3. Saves each signal as a parquet file (one file per signal variant)
4. Builds a **ranking matrix**: Z-score normalized percentile rankings of each ETF by each signal on each date

### Key Inputs
- **ETF Price DataFrame**: From database (same as Step 1)
- **Core Benchmark Prices**: MSCI World (IE00B4L5Y983) for price reference
- **Forward IR Data**: From Step 1 (`forward_alpha_1month.parquet`) - optional, used only for reference/logging

### Signal Generation
Uses `library/dpo_enhanced_variants.py` which computes:

**DPO (Detrended Price Oscillator) Formula**:
```
dpo[t] = price[t + shift] - SMA[t]
```
Where:
- `price[t + shift]`: Price shifted forward by `shift` periods
- `SMA[t]`: Simple moving average of price
- Removes trend to isolate oscillations

**Variants Generated**:
Multiple combinations of:
- **Window periods**: 21 days, 63 days (bi-scale momentum - 1 month, 3 months)
- **Shift divisors**: 1.0 (responsive), 2.0 (conservative)
  - `shift = window / shift_divisor`
- **Smoothing**: Savitzky-Golay (polyorder 4 - quartic/heavy smoothing)
  - GPU-accelerated causal implementation prevents look-ahead bias

**Example signal names**:
- `dpo_21d__savgol__polyorder_4__shift_1_0`
- `dpo_63d__savgol__polyorder_4__shift_2_0`
- (Total: 4 variants: 2 periods × 2 shift divisors)

### Computational Flow

#### 1. Load Price Data
- Extra lookback: Start from `start_date - 400 days` for rolling window warmup
- Load all ETFs and core ETF prices
- Ensures rolling calculations have sufficient history

#### 2. Clear Previous Signals
- Backs up existing signal files to `backup/2_compute_signal_bases/YYYY_MM_DD/`
- Clears all old signals for fresh computation

#### 3. Compute Signals Incrementally
**Key feature**: Compute-Save-Rank loop (compute one, save one, rank one)
```
For each of 4 DPO variants:
  1. Compute 2D array: (all_dates × all_etfs)
  2. Save to parquet immediately (streaming efficiency)
  3. While in memory, compute rankings and add to ranking matrix
  4. Move to next signal
```

#### 4. Ranking Matrix Creation
For each (date, signal, etf):
```
1. Get signal values for all ETFs on that date
2. Z-score normalize: z = (value - mean) / std
3. Percentile rank: 0-100 percentile ranking
4. Store in rankings[date_idx, etf_idx, signal_idx]
```

Ranking matrix = (n_dates × n_etfs × 4 signals) showing how each ETF ranks by each of the 4 signals

### Outputs

**1. Signal Parquet Files**
- Location: `data/signals/`
- One file per signal variant
- Format: (dates × etfs) - full historical signal values
- Example: `dpo_21d__savgol__polyorder_4__shift_1_0.parquet`

**2. Ranking Matrix**
- File: `data/rankings_matrix_signal_bases_1month.npz` (NumPy compressed)
- Contains:
  - `rankings`: Array shape (n_dates, n_unique_etfs, n_filtered_signals)
  - `dates`: Array of dates
  - `isins`: Array of ETF ISINs
  - `features`: Array of signal names that passed filter
  - `n_filtered`: Count of signals that passed correlation filter

### Configuration
| Parameter | Value | Purpose |
|-----------|-------|---------|
| DPO periods | 21d, 63d | Bi-scale momentum windows |
| Shift divisors | 1.0, 2.0 | Responsive and conservative lag-shifts |
| Savgol polyorder | 4 | Quartic smoothing (heavy) |
| Default date range | 2009-09-25 to today | Core ETF inception to current |
| Lookback padding | 400 days | Extra history for rolling window warmup |

### Critical Implementation Details

1. **No Look-Ahead Bias**
   - DPO uses `price[t + shift]` (forward shift) but only for signal computation
   - Signal values are lagged so actual prediction happens *after* signal is computed
   - Still: this is something to verify carefully for Step 7 validation

2. **Ranking Normalization**
   - Not the raw signal values, but Z-score normalized percentile ranks
   - Each date/signal pair: ETFs ranked 0-100 based on their signal strength
   - Important: this is what feeds into Step 4, not raw signal values

3. **Incremental Save Pattern**
   - Signals saved immediately after computation (memory efficient)
   - Filtering happens inline while signal is in memory
   - This is crucial for scaling to many signals

4. **Missing Data Handling**
   - Signals can have NaN values (e.g., new ETFs early in history)
   - NaN values are excluded from ranking computation
   - Preserved in parquet files for later filtering

### Data Dependencies
- Input: `forward_alpha_1month.parquet` (Step 1 output)
- Input: ETF prices (database)
- Output: Signal parquet files + `rankings_matrix_signal_bases_1month.npz`

### For Step 7 Validation
**What matters**:
- Ranking matrix is the KEY intermediate output used by later steps
- The ranking matrix contains exactly 4 signals (always - no filtering)
- For validation: Step 7 must use the SAME ranking matrix file that Step 2 created
  - 4 signals guaranteed: 2 periods (21d, 63d) × 2 shifts (1.0, 2.0)
  - Savgol polyorder 4 smoothing applied
  - Z-score normalization to 0-100 percentile ranks
- If Step 7 uses a different ranking matrix or computes it differently, predictions will diverge
- **Critical**: Verify Step 7 loads `data/rankings_matrix_signal_bases_1month.npz` and doesn't recompute

---

## STEP 3: Apply Filters (`3_apply_filters.py`)

### Purpose
Applies additional Savgol smoothing filters to the 4 base DPO signals, creating a larger feature set.
Tests which filtered variants correlate with forward IR.

### What It Does (High Level)
1. Loads the 4 base DPO signals from Step 2
2. For each base signal, applies Savgol filter with 30 different window sizes (6d to 35d, polyorder 4)
3. Creates 120 filtered signal variants (4 bases × 30 windows)
4. Each filtered signal saved as parquet file
5. Filters signals by correlation with forward IR (threshold 0.1)
6. Builds ranking matrix from filtered signals that passed threshold

### Key Inputs
- **Base Signals**: 4 DPO signals from Step 2 (pre-computed parquet files)
- **Forward IR Data**: From Step 1 - used for correlation filtering

### Filtering Configuration
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Savgol windows | 6d to 35d | Explore various smoothing strengths |
| Polyorder | 4 | Quartic smoothing (consistent with base signals) |
| Correlation threshold | 0.1 | Minimum |correlation| to keep signal |

**Total combinations**: 4 base signals × 30 Savgol variants = **120 filtered signals**

### Computational Flow

#### 1. Load Base Signals
- Reads 4 DPO signals from `data/signals/` as parquet files
- Stores in memory as numpy arrays (float32 for efficiency)

#### 2. Backup Existing Filtered Signals
- Backs up to `backup/3_apply_filters/YYYY_MM_DD/`
- Clears old filtered signals for fresh computation

#### 3. Apply Filters in Batches
```
For each of 4 base signals:
  For each of 30 Savgol window sizes (6d to 35d):
    1. Apply causal Savgol(window, polyorder=4) to base signal
    2. Queue filtered result for parallel saving
    3. Move to next window
```

#### 4. Parallel Saving + Correlation Filtering
**8 worker threads** process the queue:
- Save filtered signal to parquet
- While signal in memory, compute correlation with forward IR
- If |correlation| >= 0.1: add to ranking matrix
- If |correlation| < 0.1: discard signal

#### 5. Ranking Matrix Creation
Same as Step 2:
- Z-score normalize signal values per date
- Percentile rank: 0-100
- Store in rankings array

### Outputs

**1. Filtered Signal Parquet Files**
- Location: `data/signals/filtered_signals/`
- Naming: `{base_signal_name}__savgol_{window}d_polyorder_{polyorder}.parquet`
- Example: `dpo_21d__savgol__polyorder_4__shift_1_0__savgol_6d_polyorder_4.parquet`
- **Note**: Only files that passed correlation filter (0.1 threshold) are kept

**2. Ranking Matrix**
- File: `data/rankings_matrix_filtered_1month.npz`
- Shape: (n_dates, n_unique_etfs, n_signals_that_passed_filter)
- Contains same fields as Step 2 ranking matrix

### Critical Implementation Details

1. **Causal Savgol**
   - GPU-accelerated if available
   - Prevents look-ahead bias for backtesting

2. **Correlation Threshold = 0.1**
   - Unlike Step 2 (threshold 0.0, all signals kept)
   - Here, only signals with meaningful correlation are kept
   - Expected to filter out ~50-80% of the 120 signals
   - Results in smaller ranking matrix

3. **Parallel I/O**
   - 8 worker threads handle saving while main thread applies filters
   - Batch size: 30 combinations per iteration
   - Reduces I/O bottleneck

4. **Memory Efficiency**
   - Loads all 4 base signals into memory (small)
   - Computes filtered signals on-the-fly
   - Saves to disk immediately (streaming)

### Data Dependencies
- Input: 4 DPO parquet files from Step 2
- Input: `forward_alpha_1month.parquet` (Step 1 output)
- Output: Filtered signal parquet files + `rankings_matrix_filtered_1month.npz`

### For Step 7 Validation
**Critical decision point**: Does Step 7 use Step 2 or Step 3 ranking matrices?
- If Step 7 uses Step 2 ranking matrix: 4 signals, no filtering
- If Step 7 uses Step 3 ranking matrix: ~30-60 signals (after 0.1 correlation filter)
- **Key validation**: Verify which ranking matrix file Step 7 actually loads
  - `rankings_matrix_signal_bases_1month.npz` (Step 2 output)
  - `rankings_matrix_filtered_1month.npz` (Step 3 output)
- If wrong ranking matrix is used, predictions will be completely different
- **Critical**: Check Step 7 code for which file it references

---

## STEP 4: Precompute Feature IR (`4_precompute_feature_ir.py`)

### Purpose
Evaluates predictive power of each filtered signal by computing: "If we selected top-N ETFs by this signal, what Information Ratio would we achieve?"

This bridges signals to actual performance and enables walk-forward backtesting.

### What It Does (High Level)
1. Loads forward_ir data from Step 1 (actual realized returns per ETF per month)
2. Loads ranking matrix from Step 3 (signal rankings for each ETF)
3. For each (date, signal):
   - Select top-1, top-2, ..., top-10 ETFs by that signal's ranking
   - Compute mean forward_ir for those top-N ETFs
   - This shows: "what IR would we have achieved?"
4. Saves 3D matrix: (n_dates, n_signals, 10)

### Key Inputs
- **Forward IR Data**: From Step 1 (`forward_alpha_1month.parquet`)
  - Actual realized Information Ratios per ETF per month
- **Ranking Matrix**: From Step 3 (`rankings_matrix_filtered_1month.npz`)
  - Pre-computed percentile rankings for each signal

### Computation Algorithm
```
For each date:
  For each filtered signal:
    For N = 1 to 10:
      1. Get top-N ETFs by this signal's ranking
      2. Get their forward_ir values from Step 1 data
      3. Compute mean(forward_ir[top_N])
      4. Store in feature_ir[date, signal, N-1]
```

**Result**: A 3D matrix showing signal predictive power across different portfolio sizes

### Outputs

**File**: `data/feature_ir_1month.npz`

**Contains**:
- `feature_ir`: Shape (n_dates, n_signals, 10) - Mean IR for top-N ETFs
- `dates`: Array of dates
- `features`: Array of signal names (filtered signals from Step 3)
- `n_satellites_max`: 10
- `n_signals`: Count of signals

**Interpretation**:
- `feature_ir[date_idx, signal_idx, 0]` = mean IR if we picked top-1 ETF by this signal
- `feature_ir[date_idx, signal_idx, 1]` = mean IR if we picked top-2 ETFs by this signal
- ... up to top-10

### Critical Implementation Details

1. **Pre-computed Rankings**
   - Uses rankings from Step 3, not recomputed
   - Rankings are percentile ranks (0-100) from Z-score normalization
   - Top-N selection done by ranking scores

2. **Forward IR Lookup**
   - Converts `forward_ir` DataFrame to 2D numpy array for O(1) lookups
   - Maps (date, isin) → IR value
   - Handles missing data (NaN values)

3. **Numba JIT Optimization**
   - Parallel computation across dates (prange)
   - Warm-up run before full computation
   - Falls back to NumPy if Numba not available
   - Significant speedup on multi-core systems

4. **Validity Handling**
   - Skips (date, signal) pairs with insufficient valid rankings
   - Requires at least 10 valid rankings to compute top-10
   - Skips ETFs with missing forward_ir values

### Data Flow
```
Step 1 output (forward_ir)  ─┐
                             ├─→ Step 4 ─→ feature_ir matrix
Step 3 output (rankings)    ─┘
```

### For Step 7 Validation
**What matters**:
- Feature IR tells us which signals are predictive
- Step 7 uses this to decide: "which signals should we trust?"
- If Step 7 computes feature_ir differently, satellite selection will differ
- **Key validation**: Verify Step 7 uses the pre-computed `feature_ir_1month.npz` file
  - Should NOT recompute signal-IR relationships
  - Should load pre-computed values for decisions
- If Step 7 uses different signal rankings or IR calculations, validation fails
- **Critical**: Check if Step 7 references `data/feature_ir_1month.npz`

---

## STEP 5: Empirical IR Statistics (`5_empirical_ir_stats.py`)

### Purpose
Calculates **empirical statistics** on signal predictive power using expanding historical windows.
Replaces Monte Carlo sampling with deterministic analysis of historical performance.

This bridges the gap between signal quality (Steps 1-4) and decision-making (Step 7) by quantifying
the reliability of each signal based on its historical track record.

### What It Does (High Level)
For each test date (after 12 months of training data):
1. For each signal, use expanding window (all historical data before test date)
2. Calculate three statistics on feature_ir values:
   - **Mean IR**: Average Information Ratio of the signal
   - **Std IR**: Uncertainty/volatility of the signal's performance
   - **Hit Rate**: Percentage of months where signal was positive
3. Store these empirical priors for later decision-making

### Key Inputs
- **Feature IR Data**: From Step 4 (`feature_ir_1month.npz`)
  - Pre-computed IR values for each (date, signal, N) combination
  - Shape: (n_dates, n_signals, 10) where 10 = top-1 through top-10 satellites

### Configuration
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Holding period | 1 month | Match Step 1 forward IR calculation |
| Training minimum | 12 months | Require sufficient history before first test |
| Ensemble sizes | [3, 4, 5] | Analyze satellites at these portfolio sizes |
| Expansion window | All prior data | Use every date before test date for training |

### Computational Algorithm

```
For each test date T (starting after month 12):
  For each signal S:
    1. Get historical feature_ir values: feature_ir[0:T, S, N-1]
       (where N is the satellite count: 3, 4, or 5)
    2. Remove NaN values
    3. Calculate:
       - mean_ir = mean(historical_irs)
       - std_ir = std(historical_irs)
       - hit_rate = % of months where historical_ir > 0
    4. Store in results[N][test_date, signal]

Key: Expanding window (all prior data) for each test date
```

### Outputs

**File**: `data/empirical_ir_stats_1month.npz`

**Contains**:
- `ir_mean_N3`: Shape (n_test_dates, n_signals) - Mean IR for N=3
- `ir_mean_N4`: Shape (n_test_dates, n_signals) - Mean IR for N=4
- `ir_mean_N5`: Shape (n_test_dates, n_signals) - Mean IR for N=5
- `ir_std_N3/4/5`: Standard deviation of IR for each N
- `hit_rate_N3/4/5`: Hit rate (% positive months) for each N
- `test_dates`: Array of test dates (starting from month 13)
- `features`: Array of signal names
- `n_satellites_analyzed`: Array of ensemble sizes computed [3, 4, 5]

**Interpretation**:
- `ir_mean_N3[date, signal]` = Average IR if we used this signal to pick 3 satellites on this date
- `ir_std_N3[date, signal]` = How uncertain/volatile this signal's performance is
- `hit_rate_N3[date, signal]` = What % of past months was this signal positive?

### Critical Implementation Details

1. **Expanding Window (No Look-Ahead)**
   - For test date T, uses ALL data from start to T-1
   - Never uses data on or after the test date
   - Implements true walk-forward backtesting
   - Expanding windows get larger as you move forward in time

2. **Training Period Requirement**
   - First 12 months are pure training (no test dates)
   - Test dates start from month 13 onwards
   - Ensures reasonable sample size for statistics

3. **Deterministic (No Randomness)**
   - Replaces Monte Carlo sampling with empirical calculation
   - Same input always produces same output
   - 10x faster than MC sampling
   - Transparent (can verify exact statistics)

4. **Per-Signal, Per-Date Statistics**
   - Each signal has different empirical priors at each test date
   - Signals that have performed well historically have higher mean_ir
   - Signals that are volatile have higher std_ir
   - Signals with positive track record have higher hit_rate

5. **Multiple Ensemble Sizes**
   - Calculates for N=3, 4, 5 simultaneously
   - Allows Step 7 to choose different portfolio sizes
   - Trade-off: smaller N = concentration risk, larger N = diversification

### Data Flow
```
Step 4 output (feature_ir)  ──→  Step 5  ──→  empirical_ir_stats
```

### For Step 7 Validation
**What matters**:
- Empirical statistics quantify signal reliability based on history
- Step 7 should use these priors to weight signal selection
- Different signals will have different mean/std/hit_rate at each test date
- Walk-forward testing means each test uses only past data (no look-ahead)
- **Key validation**: Verify Step 7 uses `empirical_ir_stats_1month.npz`
  - Should NOT recompute statistics
  - Should load pre-computed empirical priors
- If Step 7 uses different statistical calculations, validation fails
- **Critical**: Check if Step 7 references empirical statistics for decisions

---

## STEP 6: Deterministic Strategy with Empirical Priors (`6_deterministic_strategy_ir.py`)

### Purpose
Runs a **walk-forward backtest** of the satellite selection strategy using fixed empirical priors.
This is the deterministic replacement for Bayesian Monte Carlo simulation - it uses empirical IR statistics
to make feature selection decisions throughout the backtest period.

### What It Does (High Level)
For each test date (after 12 months of training):
1. Initialize feature beliefs with empirical IR statistics from Step 5
2. Use greedy Bayesian selection to pick best signal ensemble (52 signals)
3. Use selected signals to rank and pick top-N satellites (N=3, 4, or 5)
4. Calculate actual realized alpha for that month
5. Record portfolio performance and signal selections

### Key Inputs
- **Forward Alpha Data**: From Step 1 (`forward_alpha_1month.parquet`)
  - Actual realized returns for each ETF
- **Ranking Matrix**: From Step 3 (`rankings_matrix_filtered_1month.npz`)
  - Signal rankings for each ETF
- **Empirical IR Statistics**: From Step 5 (`empirical_ir_stats_1month.npz`)
  - Pre-computed signal reliability metrics (mean, std, hit rate)

### Configuration
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Holding period | 1 month | Match forward IR calculation |
| N satellites | 3, 4, 5 | Different portfolio sizes to test |
| Training period | 12 months | Warm-up before first test date |
| Ensemble size | 52 signals | Fixed feature selection size |
| Selection method | Greedy Bayesian | IR-optimized signal ensemble |
| Reoptimization | Every 1 month | Refresh signal ensemble each month |

### Computational Algorithm

```
For each test date T (after 12 months):
  For each N in [3, 4, 5]:
    1. Initialize Bayesian beliefs with empirical priors
       - Load mean_ir, std_ir, hit_rate from Step 5
       - NO learning/updates (priors are fixed)
    2. Select feature ensemble using greedy Bayesian:
       - Start with highest expected IR signal
       - Greedily add signals that improve ensemble utility
       - Continue until 52 signals or improvement drops below threshold
    3. Use selected signals to score ETFs:
       - For each ETF, weighted average of rankings (weights = signal expected_ir)
       - Select top N ETFs by score
    4. Calculate portfolio alpha:
       - Get realized alpha for selected ETFs from Step 1 data
       - Average alpha = mean(forward_alpha[selected_etfs])
    5. Record: (date, N, selected_signals, selected_etfs, alpha)
```

### Key Concepts

**Bayesian Feature Selection**:
- Each signal has a belief distribution (mu, sigma)
- Greedy selection: Start with best signal, add next signal if it improves ensemble
- Ensemble utility = E[IR] * P(positive) / std(IR)
- Maximizes expected risk-adjusted Information Ratio

**Empirical Priors**:
- Priors from Step 5 are FIXED throughout backtest
- NO belief updates (deterministic)
- Same priors used for every test date
- This enables true validation: same historical patterns every time

**Walk-Forward Testing**:
- Test dates start from month 13 onwards (after 12 month training)
- For each test date, only use data up to that date
- No future information leakage

### Outputs

**Backtest Results Files**:
- Location: `data/backtest_results/`
- Files: `bayesian_backtest_N3.csv`, `bayesian_backtest_N4.csv`, `bayesian_backtest_N5.csv`
- Each CSV contains columns:
  - `date`: Test date
  - `n_satellites`: Portfolio size (3, 4, or 5)
  - `avg_alpha`: Realized portfolio alpha that month
  - `n_selected`: Number of ETFs selected
  - `n_features`: Number of signals selected
  - `selected_features`: Signal names used
  - `selected_isins`: ETF ISINs selected

**Summary Statistics**:
- File: `data/backtest_results/bayesian_backtest_summary.csv`
- Contains performance metrics for each N:
  - `avg_alpha`: Mean monthly alpha
  - `std_alpha`: Volatility of alpha
  - `information_ratio`: Sharpe-like ratio (mean / std)
  - `hit_rate`: % of positive months
  - `annual_alpha`: Compounded annual return
  - Stability metrics (min, max, negative months)

### Critical Implementation Details

1. **Deterministic Priors**
   - Empirical priors loaded from Step 5
   - Used directly without modification
   - Same priors for all test dates (stationary belief)
   - NO belief updates during backtest

2. **Greedy Bayesian Selection**
   - Greedy ensemble building (not exhaustive search)
   - Minimizes computational time (exponential search space)
   - Adds 52 signals total (MIN_ENSEMBLE_SIZE = MAX_ENSEMBLE_SIZE = 52)
   - Fixed ensemble size for consistency

3. **Signal Scoring**
   - Signals scored by expected_ir = mu / sigma
   - Higher expected IR = higher weight in portfolio
   - Each ETF gets weighted sum of signal rankings

4. **Top-N Selection**
   - Deterministic selection: highest ranked ETFs chosen
   - Uses numba JIT compilation for speed
   - Handles missing data (NaN values skipped)

5. **Performance Metrics**
   - All metrics calculated deterministically from data
   - Information Ratio = mean_alpha / std_alpha
   - Hit rate = % months with positive alpha
   - Stability metrics quantify return consistency

### Data Flow
```
Step 1 output (forward_alpha)  ─┐
                               ├─→ Step 6 ──→ backtest_results
Step 3 output (rankings)       ─┤
Step 5 output (empirical_ir)   ─┘
```

### For Step 7 Validation
**What matters**:
- Step 6 is the historical walk-forward backtest
- Shows what strategy would have done in the past
- Step 7 should make current (live) predictions
- Predictions should match backtest when "rewound" to historical dates
- **Key validation**: Verify Step 7 uses same:
  - Empirical priors from Step 5
  - Feature selection method (greedy Bayesian)
  - Signal ensemble size (52)
  - Top-N selection logic
- If Step 7 uses different selection method or priors, predictions will diverge

---

## STEP 7: Generate Monthly Portfolio Allocation (`7_generate_monthly_allocation.py`)

### Purpose
Converts the latest backtest results into a practical portfolio allocation with:
- Dollar amounts for each position
- Share quantities based on current market prices
- Core-Satellite split (60% core, 40% satellites)
- Cash optimization to minimize uninvested cash

### What It Does (High Level)
1. **Load Latest Backtest Results**: Reads the most recent CSV from Step 6
2. **Extract Selected Satellites**: Gets ISINs for N satellites from the latest row
3. **Get Current Prices**: Looks up live/current ETF prices from database
4. **Calculate Allocations**:
   - Core ETF: 60% of budget
   - Each satellite: (40% of budget) / N satellites
5. **Optimize Quantities**:
   - Rounds down to minimize overinvestment
   - Opportunistically rounds up to minimize uninvested cash
6. **Output**: Detailed allocation table with positions, quantities, and deviations

### Key Inputs
- **Backtest Results**: `backtest_results/bayesian_backtest_N{n}.csv` (from Step 6)
  - CSV with columns: `date`, `selected_isins`, `avg_alpha`, etc.
  - Reads the **latest row only** (most recent date)

- **ETF Database**: `maintenance/data/etf_database.db`
  - Loads current prices for selected ISINs
  - Loads ETF names for display

- **Core ETF ISIN**: `IE00B4L5Y983` (iShares Core MSCI World)
  - Always 60% of portfolio

- **Budget**: User input (EUR amount)

### Data Flow
```
Backtest Results (Step 6)
         │
         ├─→ Get latest row (latest date)
         ├─→ Extract selected_isins
         │
         ├─→ Lookup current prices from DB
         │
         └─→ Calculate allocations
                    │
                    └─→ Console output + optional CSV
```

### Configuration Parameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `CORE_ISIN` | `IE00B4L5Y983` | MSCI World benchmark (always 60%) |
| `CORE_ALLOCATION_PCT` | 0.60 | Core ETF allocation |
| `SATELLITE_ALLOCATION_PCT` | 0.40 | Satellites total allocation |
| `N_SATELLITES` | 3 (default) | Can be 3, 4, or 5 based on backtest availability |
| `DB_PATH` | `maintenance/data/etf_database.db` | ETF price/metadata database |
| `BACKTEST_DIR` | `pipeline/data/backtest_results` | Where backtest CSVs are stored |

### Implementation Details

#### 1. Loading Latest Results
```python
backtest_file = BACKTEST_DIR / f'bayesian_backtest_N{n_satellites}.csv'
df = pd.read_csv(backtest_file)
latest_row = df[df['date'] == df['date'].max()].iloc[0]  # Latest row only!
isins = [x.strip() for x in latest_row['selected_isins'].split(',')]
```

#### 2. Price Lookup
```python
for isin in isins:
    series = db.load_prices(isin)
    latest_price = series.iloc[-1]  # Most recent price in DB
```

#### 3. Allocation Calculation
```python
core_target = budget * 0.60
satellite_per_target = budget * 0.40 / n_satellites

allocations = {
    CORE_ISIN: core_target,
    sat_isin_1: satellite_per_target,
    sat_isin_2: satellite_per_target,
    ...
}
```

#### 4. Quantity Optimization
```
Step 1: Round down all quantities
  qty = floor(target_amount / price)

Step 2: Identify candidates for rounding up
  - Only if uninvested_cash >= price
  - Sort by allocation deviation (biggest shortfalls first)

Step 3: Round up highest-deviation positions
  - While uninvested_cash >= price
  - Reduce uninvested cash
```

### Output
**Format**: Console table with detailed allocation breakdown

**Columns**:
- `Type`: CORE or SATELLITE
- `ISIN`: ETF identifier
- `Name`: ETF name (first 30 chars)
- `Target %`: Desired allocation percentage
- `Target EUR`: Desired EUR amount
- `Price`: Current ETF price
- `Quantity`: Number of shares to buy
- `Actual EUR`: Actual invested amount (qty × price)
- `Deviation`: Difference from target (EUR and %)

**Summary Statistics**:
- Total target vs actual investment
- Uninvested cash amount and percentage
- Displayed for N=3, N=4, N=5 scenarios

### Critical Architectural Issue ⚠️

**PROBLEM**: Step 7 does **NOT** replicate the prediction logic from Step 6.

Step 7 currently:
- ✓ Reads pre-computed satellite selections from Step 6
- ✓ Looks up current prices
- ✓ Calculates dollar amounts
- ✗ Does NOT load Steps 1-5 data
- ✗ Does NOT recalculate feature selection
- ✗ Does NOT recalculate satellite selections
- ✗ Only uses the **latest** backtest results

**Why this matters for validation**:
- User wants to verify: "If I run Step 7 on a historical date, do I get the same satellites as Step 6's backtest for that date?"
- Current Step 7 can ONLY access the latest backtest results
- Cannot "rewind" to historical dates because it doesn't replicate the selection logic
- Step 7 is essentially a **portfolio construction tool**, not a **prediction validation tool**

**What Step 7 should do to enable validation**:
1. Accept a historical date as input parameter
2. Load Steps 1-5 outputs as of that date
3. Replicate Step 6's feature selection logic:
   - Initialize BeliefState with empirical priors
   - Run greedy Bayesian feature selection
   - Select top-N ETFs using same scoring
4. Return selected satellites for that historical date
5. Then calculate allocation using historical prices from that date

**Current limitation**: Step 7 as written can only generate allocations for the most recent date, and only uses pre-computed selections from Step 6. It cannot validate historical predictions.

### Validation Strategy

**To properly validate that Step 7 matches Step 6 backtest**:

Option A: **Extend Step 7** (Recommended)
- Modify Step 7 to accept `--date` parameter
- Load historical data for that date
- Replicate Step 6 selection logic
- Return predicted satellites
- Compare with Step 6 backtest results for that date

Option B: **Create Step 7.5** (Alternative)
- New script: `7_validate_predictions.py`
- Takes historical date as input
- Replicates Step 6 logic at that date
- Returns comparison: Step 6 satellites vs predicted satellites
- Shows if they match (✓) or differ (✗)

Option C: **Use Step 6 directly** (Current workaround)
- Step 6 already has all the logic
- Backtest results CSV shows what was predicted each month
- Manual approach: pick a historical date from the CSV, verify manually

**Recommendation**: Option A or B would enable automated validation that Step 7 produces correct predictions.
- **Critical**: Step 7 should produce same selections as Step 6 for historical dates

---

## VALIDATION: Historical Prediction Validation (`validate_predictions_historical.py`)

### Purpose
Comprehensive walk-forward validation that ensures Step 7 produces correct predictions by:
1. Selecting historical test dates from the backtest
2. For each date, truncating the database to only data available **up to that date**
3. Running the full pipeline (Steps 1-6) on truncated data
4. Comparing predictions with original Step 6 backtest results

This test eliminates all forward-looking bias and verifies the strategy is reproducible.

### What It Does

For each historical test date in the backtest:
```
1. Load original backtest results for that date
   - Contains predicted satellite ISINs for N=3, 4, 5

2. Create truncated database copy
   - Only includes price data up to and including the test date
   - Eliminates any possibility of forward-looking bias

3. Run pipeline Steps 1-6 with truncated data
   - Step 1: Compute forward alpha (using only available data)
   - Step 2: Compute signal bases (DPO variants)
   - Step 3: Apply filters (correlation filtering)
   - Step 4: Precompute feature IR
   - Step 5: Calculate empirical IR statistics
   - Step 6: Run walk-forward backtest

4. Extract predictions from rerun
   - Get predicted satellite ISINs for that date
   - Extract from backtest results CSV

5. Compare predictions
   - Original prediction (from full backtest)
   - Rerun prediction (from truncated pipeline)
   - Check if ISINs match exactly
```

### Configuration Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `N_SATELLITES_TO_TEST` | [3, 4, 5] | Satellite counts to validate |
| `MIN_TEST_DATES` | 3 | Minimum historical dates required |
| `test_date_strategy` | 'recent_5' | Last 5 dates; can use 'all', 'sample_N', 'recent_N' |

### Usage

```bash
# Validate last 5 historical dates
python validate_predictions_historical.py

# Validate all backtest dates
python validate_predictions_historical.py --strategy all

# Validate random sample of 10 dates
python validate_predictions_historical.py --strategy sample_10

# Validate last 20 dates
python validate_predictions_historical.py --strategy recent_20
```

### What This Test Validates

✓ **No Forward-Looking Bias**: Database truncation ensures pipeline only uses data available at prediction time

✓ **Deterministic Pipeline**: Same historical data produces same predictions every time

✓ **Reproducibility**: Strategy can be reproduced by running pipeline at any historical point

✓ **Consistency**: Step 6 backtest results match what Step 7 would predict

✓ **Data Integrity**: Truncated databases don't corrupt pipeline calculations

### Expected Results

**Success Criteria**:
- For each historical test date
- For each N in [3, 4, 5]
- Predicted satellite ISINs match exactly between:
  - Original backtest (Step 6 output)
  - Rerun with truncated data (Steps 1-6 on limited data)

**Output Example**:
```
Date: 2023-06-30
✓ N=3: ISINs match exactly
✓ N=4: ISINs match exactly
✓ N=5: ISINs match exactly

Overall: [SUCCESS] ALL VALIDATION TESTS PASSED!
```

### Output Location

**Validation Results**: `validation_historical/`
- Temporary working directories for each test date
- Database copies with truncated data
- Full pipeline outputs for comparison

**Summary**: Printed to console with detailed match/mismatch reporting

### Implementation Notes

1. **Database Truncation**
   - Creates temporary copy of ETF database
   - Deletes all price records after test date
   - Preserves database schema and metadata
   - Automatically cleaned up after validation

2. **Pipeline Isolation**
   - Each test date gets separate working directory
   - Pipeline data directory cleared between runs
   - No cross-contamination between test runs

3. **Error Handling**
   - Graceful failure if database truncation fails
   - Continues validation with warnings
   - Reports which steps failed
   - Captures stdout/stderr for debugging

4. **Performance**
   - Recent 5 dates: ~5-15 minutes
   - All dates: Can take 1+ hour depending on backtest length
   - Recommend starting with `recent_5` strategy

### Interpretation Guide

**Match (✓)**: ISINs from both predictions are identical
- Means Step 7 would produce identical results to backtest
- No systematic differences detected
- Strategy is reproducible

**Mismatch (✗)**: ISINs differ between predictions
- Indicates Step 7 logic doesn't match Step 6 exactly
- May reveal issues in feature selection or data handling
- Requires investigation

**Error**: Validation script encountered exception
- May indicate data corruption
- May indicate missing files or database issues
- Check error message for details

### Next Steps After Validation

If all tests pass:
- ✓ Step 7 is correctly implementing the backtest logic
- ✓ Predictions are reproducible at any historical point
- ✓ No forward-looking bias detected
- ✓ Ready to run Step 7 for live predictions

If tests fail:
- Review mismatch details
- Check which N (3, 4, or 5) has issues
- Compare original vs rerun feature selections
- Verify database truncation worked correctly
- Review Step 6 vs Step 7 logic differences
