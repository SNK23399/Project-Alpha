# ETF Portfolio Management - Core-Satellite Strategy

A Python-based pipeline implementing a **Core-Satellite investment strategy** with Bayesian signal-based satellite selection for ETF alpha prediction.

## Overview

This system generates **consistent positive alpha** through tactical satellite allocation over a global market baseline (MSCI World) using Bayesian learning from historical performance data.

**Key Philosophy**: Select 3-7 satellite ETFs monthly based on detrended price oscillators (DPO) with triple exponential moving average (TEMA) filtering and Savitzky-Golay smoothing.

## Features

- **869+ ETFs** tracked with daily price data since 2009
- **287 DPO signal bases** with multiple shift divisors (detrended price oscillators using TEMA)
- **9 Savitzky-Golay filter variants** per signal (~2,500 total filtered signals)
- **Walk-forward backtesting** with expanding training windows
- **Bayesian satellite selection** with learned hyperparameters
- **GPU acceleration** (CuPy/CUDA) for MC simulations and filtering
- **Numba JIT compilation** for parallel optimization

## Project Structure

```
Core Satellite/
├── pipeline/                          # Main 7-step pipeline
│   ├── 1_compute_forward_ir.py         # Step 1: Compute target variable (forward IR)
│   ├── 2_compute_signal_bases.py       # Step 2: Generate 287 DPO signal bases
│   ├── 3_apply_filters.py              # Step 3: Apply Savitzky-Golay filters
│   ├── 4_precompute_feature_ir.py      # Step 4: Compute feature-IR matrix
│   ├── 5_precompute_mc_ir_stats.py     # Step 5: Monte Carlo IR statistics
│   ├── 6_bayesian_strategy_ir.py       # Step 6: Bayesian satellite selection
│   ├── 7_generate_monthly_allocation.py # Step 7: Interactive portfolio allocation
│   ├── main.py                         # Orchestrator (run any step combination)
│   └── data/                           # Pipeline outputs
│       ├── forward_alpha_1month.parquet
│       ├── signals/                    # Signal bases (parquet files)
│       ├── feature_ir_1month.npz
│       ├── mc_ir_mean_1month.npz
│       ├── backtest_results/           # Satellite selections
│       └── allocation/                 # Monthly portfolio allocations
│
├── library/                            # Signal computation
│   ├── signal_filters.py               # GPU-accelerated smoothing filters (causal_ema, causal_tema, causal_savgol, etc.)
│   ├── dpo_enhanced_variants.py        # DPO signal generator (287 variants)
│   ├── signal_bases.py                 # DEPRECATED: Legacy signal library (228 signals, not used)
│   ├── signal_indicators.py            # DEPRECATED: Legacy feature engineering (not used)
│   └── __init__.py
│
├── support/                            # Data access & utilities
│   ├── etf_database.py                 # ETF price database (SQLite)
│   ├── signal_database.py              # Signal storage (Parquet)
│   ├── backtester.py                   # DEPRECATED: Superseded by Step 6
│   ├── degiro_client.py                # DORMANT: Future trading API integration
│   ├── price_fetcher.py                # DORMANT: Maintenance scripts only
│   └── etf_fetcher.py                  # DORMANT: Maintenance scripts only
│
├── maintenance/                        # Data collection & updates
│   ├── 1_collect_etf_data.py          # Fetch ETF prices & metadata
│   ├── 2_compare_databases.py         # Data quality validation
│   └── 3_analyze_data_quality.py      # Data statistics & analysis
│
└── README.md                           # This file
```

## Quick Start

### Prerequisites

```bash
pip install pandas numpy scipy matplotlib tqdm numba
# Optional (for GPU acceleration):
pip install cupy-cuda11x  # Replace 11x with your CUDA version
```

### Run the Pipeline

**Default (all steps 1-6):**
```bash
python main.py
```

**Run specific steps:**
```bash
python main.py --steps 2,3,4,5,6       # Skip Step 1 (uses pre-computed forward_ir)
python main.py --only-step 6           # Only satellite selection
python main.py --steps 1,2,3,4,5,6,7   # Full pipeline including allocation
```

**Available Arguments:**
```
--steps STEPS           Comma-separated list (e.g., "1,2,3,4,5,6")
--only-step STEP_NUM    Run single step (e.g., "6")
```

## Pipeline Overview

### Step 1: Compute Forward IR (Target Variable)
**File**: `pipeline/1_compute_forward_ir.py`
- Loads ETF prices from database
- Computes 1-month forward returns and Information Ratio for all ETFs
- IR = forward_alpha / realized_volatility (consolidates alpha, consistency, risk)
- **Output**: `data/forward_alpha_1month.parquet`

### Step 2: Compute Signal Bases (DPO with TEMA)
**File**: `pipeline/2_compute_signal_bases.py`
- Generates 287 DPO (Detrended Price Oscillator) signal bases
- Uses Triple Exponential Moving Average (TEMA) for detrending
- DPO formula: `price[t + shift] - TEMA[t]` where shift = period / divisor
- Divisors: 1.1 to 1.7 (7 variants per DPO period)
- Periods: 30d to 50d (21 periods)
- **Output**: `data/signals/*.parquet` + ranking matrix

### Step 3: Apply Filters (Savitzky-Golay Smoothing)
**File**: `pipeline/3_apply_filters.py`
- Applies 9 Savitzky-Golay filter variants to each signal base
- Creates ~2,500 filtered signals (287 bases × 9 filters, minus correlations)
- Savitzky-Golay preserves signal features better than simple moving averages
- GPU-accelerated if CuPy available
- **Output**: `data/signals/filtered_signals/*.parquet` + ranking matrix

### Step 4: Precompute Feature-IR Matrix
**File**: `pipeline/4_precompute_feature_ir.py`
- For each signal, evaluates the IR of top-N selected ETFs
- If top-3 ETFs ranked by signal A have mean IR=0.85, signal A gets IR score 0.85
- This is the "signal strength" measure used in Step 6
- **Output**: `data/feature_ir_1month.npz`

### Step 5: Precompute MC Information Ratio Statistics
**File**: `pipeline/5_precompute_mc_ir_stats.py`
- Runs Monte Carlo simulations (1M samples per month per feature)
- Computes IR distribution statistics for Bayesian priors
- Uses GPU acceleration (CuPy) for speed
- Estimates how reliably each signal predicts good IR
- **Output**: `data/mc_ir_mean_1month.npz`

### Step 6: Bayesian Satellite Selection
**File**: `pipeline/6_bayesian_strategy_ir.py`
- Walk-forward backtesting with expanding training window
- For each month T: train on months 0..T-1, test on month T
- Uses Bayesian learning with decay rate and prior strength hyperparameters
- Selects 3-7 satellites based on expected IR
- Equally weights selected satellites
- **Output**: `data/backtest_results/bayesian_backtest_N*.csv`

### Step 7: Generate Monthly Allocation (Interactive)
**File**: `pipeline/7_generate_monthly_allocation.py`
- Interactive script: takes user budget input
- Generates buy orders for 60/40 core-satellite portfolio
- Core (60%): MSCI World index (IE00B4L5Y983)
- Satellites (40%): Top-N from Stage 6, equally weighted
- Minimizes uninvested cash, uses integer quantities
- **Output**: `data/allocation/allocation_YYYYMMDD_HHMMSS.csv`

## Data Dependencies

```
Step 1: Forward IR
    ↓
Step 2: Signal Bases → Ranking Matrix
    ↓
Step 3: Apply Filters → Ranking Matrix
    ↓
Step 4: Feature-IR Matrix (uses Steps 1 & 3)
    ↓
Step 5: MC IR Stats (uses Steps 1, 3, 4)
    ↓
Step 6: Bayesian Selection (uses Steps 1, 3, 5)
    ↓
Step 7: Portfolio Allocation (uses Step 6)
```

**To skip Step 1** (if forward_ir already computed):
```bash
python main.py --steps 2,3,4,5,6
```

## Configuration

### Core ETF
```python
CORE_ISIN = 'IE00B4L5Y983'  # iShares Core MSCI World
```

### Portfolio Allocation
```python
CORE_WEIGHT = 0.60         # Core allocation
SATELLITE_WEIGHT = 0.40    # Satellite allocation
N_SATELLITES = [3,4,5,6,7] # Test different portfolio sizes
```

### Signal Computation
```python
# Step 2: DPO signal bases
DPO_PERIODS = range(30, 51)              # 30-50 days
TEMA_SHIFT_DIVISORS = [1.1, 1.2, ..., 1.7]  # 7 shift variants

# Step 3: Savitzky-Goyal filters
FILTER_VARIANTS = 9  # 9 different Savgol window configurations
```

### Monte Carlo Settings (Step 5)
```python
MC_SAMPLES = 1_000_000   # Per month per feature
MIN_TRAINING_MONTHS = 36  # Minimum backtest history
```

## Key Concepts

### DPO (Detrended Price Oscillator)
- Removes trend from price to isolate oscillations
- Formula: `price[t + shift] - MA[t]`
- Forward-looking shift creates predictive signal
- TEMA chosen after testing 12 MA types (94-100% selection rate)

### Information Ratio (IR)
- Measures risk-adjusted alpha generation
- `IR = mean(alpha) / std(alpha)`
- Consolidates: alpha magnitude, consistency, and risk control
- Used as target variable for entire pipeline

### Bayesian Learning
- Learns which signals reliably predict good future IR
- Uses decay rate: older months weighted less
- Uses prior strength: confidence in model before seeing data
- Walk-forward: expands training window each month

### Walk-Forward Backtesting
- **No look-ahead bias**: Train on months with known outcomes only
- Month T-1 result is known before predicting month T
- Expanding window: each month adds 1 new training observation
- Typical: 36+ months training, then 50+ months of out-of-sample testing

## Performance

### Signal Computation Time
- Step 2 (287 DPO bases): ~5-10 minutes
- Step 3 (9 Savgol variants): ~10-20 minutes
- Step 4 (Feature-IR): ~2-5 minutes
- Step 5 (MC stats, GPU): ~10-30 minutes (CPU: hours)

### Optimization
- **Numba JIT**: Parallel loops in Steps 4, 5, 6
- **GPU Acceleration**: CuPy in Step 3 (filter application), Step 5 (MC)
- **Vectorization**: NumPy/Pandas across all steps
- **Storage**: Parquet format (10-50x faster than SQLite for analytics)

## Maintenance

### Monthly Update
```bash
cd maintenance
python 1_collect_etf_data.py     # Fetch new prices
python 2_compare_databases.py    # Validate & swap

cd ../pipeline
python main.py --steps 2,3,4,5,6  # Recompute signals & selection
```

### Data Quality
```bash
cd maintenance
python 3_analyze_data_quality.py  # Check for gaps, outliers
```

## Unused/Deprecated Components

These modules exist but are **not used** in the current pipeline:

| File | Status | Reason |
|------|--------|--------|
| `signal_bases.py` | DEPRECATED | Replaced by DPO-focused approach |
| `signal_indicators.py` | DEPRECATED | Cross-sectional features not used |
| `backtester.py` | SUPERSEDED | Functionality in Step 6 |
| `price_fetcher.py` | DORMANT | Maintenance scripts only |
| `etf_fetcher.py` | DORMANT | Maintenance scripts only |
| `degiro_client.py` | NOT INTEGRATED | Future execution feature |

Can be safely removed if not needed for maintenance workflows.

## Troubleshooting

### Step 1 Unicode Error
**Issue**: `UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'`
**Solution**: Skip Step 1, use pre-computed data
```bash
python main.py --steps 2,3,4,5,6
```

### Step 5 GPU Error
**Issue**: `CuPy not installed` or CUDA mismatch
**Solution**: Falls back to CPU (slower), or install matching CuPy version
```bash
pip install cupy-cuda11x  # Replace 11x with your CUDA version
```

### Missing Forward IR File
**Issue**: `FileNotFoundError: forward_alpha_1month.parquet`
**Solution**: Run Step 1 first
```bash
python main.py --only-step 1
```

## License & Disclaimer

**Educational and research purposes only.** Not financial advice.

- Consult a qualified financial advisor before investing
- Past performance ≠ future results
- Authors not responsible for financial losses

## Next Steps

- Monitor backtest results in `data/backtest_results/`
- Generate monthly allocations via Step 7
- Analyze parameter stability across months
- Consider portfolio-level constraints (max 10% per ETF, etc.)

---

**Last Updated**: January 2026
**Version**: 1.0 - Core-Satellite Bayesian Selection
