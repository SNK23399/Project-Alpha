# ETF Portfolio Management - Core-Satellite Strategy

A Python-based portfolio management system implementing a **Core-Satellite investment strategy** with machine learning signal generation for ETF alpha prediction.

## Overview

This project aims to generate **consistent positive alpha** over a global market baseline (MSCI World) through tactical satellite allocation using a comprehensive signal-based approach.

**Key Philosophy**: Focus on **alpha consistency** (percentage of time with positive rolling alpha), not just maximum total returns.

## Features

- **869 ETFs** tracked with daily price data from 2009
- **293 signal bases** computed for each ETF (momentum, volatility, risk metrics, etc.)
- **25 smoothing filters** applied to each signal (7,325 filtered signals)
- **Signal combination** into features for ML-based prediction
- **Backtesting engine** for strategy validation
- **Numba JIT acceleration** for filter computations

## Installation

### Prerequisites

```bash
pip install pandas numpy scipy matplotlib tqdm justetf-scraping degiro-connector scikit-learn empyrical quantstats ta numba
```

## Project Structure

```
Core Satellite/
├── Data Pipeline (pipeline/)
│   ├── 1_compute_forward_ir.py           # Compute target variable (forward alpha)
│   ├── 2_compute_signal_bases.py         # Compute 293 raw signals
│   ├── 3_apply_filters.py                # Apply 25 smoothing filters
│   ├── 4_precompute_feature_ir.py        # Compute feature-IR matrix
│   ├── 5_precompute_mc_ir_stats.py       # Compute Bayesian priors
│   ├── 6_bayesian_strategy_ir.py         # Bayesian satellite selection (equal-weighted)
│   ├── 7_rule_parameter_discovery.py     # Test portfolio rule parameters (100k MC)
│   ├── 8_adaptive_allocation_predictor.py # Predict core/satellite weights
│   ├── models/                           # Trained allocation prediction models
│   └── main.py                           # Orchestrate full pipeline
│
├── Signal Libraries (library/)
│   ├── signal_bases.py               # Signal computation (parallelized, 293 signals)
│   ├── signal_filters.py             # Filter definitions (Numba-accelerated, 25 filters)
│   ├── signal_indicators.py          # Cross-sectional feature transformations
│   └── __init__.py                   # Package exports
│
├── Support Modules (support/)
│   ├── backtester.py                 # Backtesting engine
│   ├── degiro_client.py              # DEGIRO API client
│   ├── etf_database.py               # ETF data storage (SQLite)
│   ├── etf_fetcher.py                # ETF catalog fetching
│   ├── price_fetcher.py              # Price data fetching
│   └── signal_database.py            # Signal data storage (Parquet)
│
├── Maintenance (maintenance/)
│   ├── 1_collect_etf_data.py         # Fetch ETF data
│   ├── 2_compare_databases.py        # Validate and replace database
│   └── 3_analyze_data_quality.py     # Data quality analysis
│
└── Documentation
    ├── README.md                     # This file
    └── CLAUDE.md                     # Detailed project documentation
```

## Quick Start

### 1. Collect ETF Data

```bash
cd maintenance
python 1_collect_etf_data.py
```

This will prompt for your DEGIRO credentials (never stored) and fetch:
- ETF metadata from DEGIRO
- Historical prices from JustETF
- Data filtered from core ETF inception (2009-09-25)

### 2. Compute Signal Bases

```bash
python 1_compute_signal_bases.py full
```

Computes 293 signal bases including:
- Returns & Alpha (vs core and universe)
- Momentum (multiple timeframes)
- Volatility & Risk metrics
- Beta & Idiosyncratic returns
- Technical indicators (RSI, MACD, Bollinger)
- Drawdown metrics
- And more...

### 3. Apply Filters

```bash
python 2_apply_filters.py
```

Applies 25 smoothing/transformation filters to each signal:
- Raw passthrough
- EMA smoothing (5d, 10d, 21d spans)
- SMA smoothing (5d, 10d, 21d windows)
- Hull MA (fast adaptive smoothing)
- KAMA (Kaufman Adaptive MA)
- Butterworth lowpass filter
- Z-score normalization (multiple windows)
- Percentile ranking
- Rate of change
- Regime switching detection

### 4. Satellite Selection

```bash
cd pipeline
python 6_bayesian_strategy_ir.py
```

Runs monthly Bayesian satellite selection with equal-weighted allocation:
- Selects 3-7 ETFs per month based on predicted Information Ratio
- Uses Bayesian feature selection with learned hyperparameters (decay, prior strength)
- Satellites equally weighted (equal-weighting outperforms IR-score weighting)
- Walk-forward validation with expanding training window
- Learns feature beliefs from historical alpha data

### 5. Portfolio Rule Parameter Discovery

```bash
python 7_rule_parameter_discovery.py
```

Tests 100,000 Monte Carlo portfolio rule combinations:
- Evaluates 5 rule parameters (core_weight, rebalance_threshold, etc.)
- Identifies which parameters significantly affect Information Ratio
- Discovers that only core_weight matters (correlation: -0.39)
- Output: parameter sensitivity analysis and optimal distributions

### 6. Predict Allocation

```bash
python 8_adaptive_allocation_predictor.py
```

Trains predictive models for optimal portfolio allocation:
- **Phase 1 (Training)**: Learns from 464 months of historical Stage 6 selections
- **Phase 2 (Prediction)**: Predicts optimal allocation for any new satellite selection:
  - Optimal core/satellite split (R² = 0.973)
  - Individual satellite weights within satellite allocation (R² = 0.620)
- Models saved to: `pipeline/models/`
- Usage: `predictor.predict(selected_isins, alpha_df, target_date)`

## Core Concepts

### Core-Satellite Strategy

- **Core**: iShares Core MSCI World (IE00B4L5Y983) - Global diversification
- **Satellites**: Regional/thematic ETFs for tactical allocation
- **Satellite Allocation**: Equal weighting across selected satellites
  - Tested IR-score weighting and score² weighting, both showed negligible improvement
  - Equal weighting is simpler and equally effective for selected universe

### Signal-Based Alpha Generation

The system generates alpha predictions through:

1. **Raw Signals** (293): Technical and fundamental indicators
2. **Filtered Signals** (7,325): Smoothed/transformed variants
3. **Features**: Combined signals for ML models
4. **Predictions**: Expected alpha for each ETF
5. **Allocation**: Overweight predicted outperformers

### Alpha Consistency

**Primary metric**: % of time with positive rolling alpha

- Better: +0.5% alpha 80% of the time
- Worse: +3% alpha 50% of the time

### Satellite Allocation Strategy

**Key Finding**: Equal weighting of selected satellites is optimal.

Tested three allocation strategies for selected satellites:
1. **Equal weight** (Stage 6): Baseline approach
2. **IR-score weight**: Weight by selection score (negligible improvement: +0.03%)
3. **Score² weight**: Squared weighting for concentration (negligible improvement: +0.04%)

**Result**: All approaches perform virtually identically, suggesting:
- The satellite selection (which ETFs to pick) is the critical factor
- How those satellites are weighted has minimal impact once selected
- **Decision**: Use equal weighting for simplicity and interpretability

## Signal Categories

| Category | Signals | Description |
|----------|---------|-------------|
| Returns | 8 | Daily returns, alpha vs core/universe |
| Momentum | 28 | Multi-timeframe momentum, skip-month |
| Relative Strength | 16 | RS vs core and universe |
| Volatility | 20 | Rolling volatility, realized vol |
| Risk Metrics | 24 | Sharpe, Sortino, Calmar ratios |
| Beta/Alpha | 16 | Rolling beta, idiosyncratic returns |
| Technical | 40+ | RSI, MACD, Bollinger, etc. |
| Drawdown | 20 | Max DD, recovery metrics |
| Capture Ratios | 8 | Up/Down market capture |
| Autocorrelation | 16 | Return persistence |
| Seasonality | 15 | Month-of-year effects |

## Monthly Update Workflow

```bash
# 1. Fetch new price data (creates etf_database_new.db)
cd maintenance
python 1_collect_etf_data.py

# 2. Compare databases and replace if valid (prompts y/n, creates dated backup)
python 2_compare_databases.py

# 3. Analyze data quality
python 3_analyze_data_quality.py

# 4. Recompute signals and run satellite selection
cd ../pipeline
python 2_compute_signal_bases.py incremental
python 3_apply_filters.py
python 4_precompute_feature_ir.py
python 5_precompute_mc_ir_stats.py
python 6_bayesian_strategy_ir.py
```

## Configuration

### Core ETF
```python
CORE_ISIN = 'IE00B4L5Y983'  # iShares Core MSCI World
CORE_INCEPTION_DATE = '2009-09-25'
```

### Default Parameters
```python
INITIAL_INVESTMENT = 50000   # EUR
MONTHLY_CONTRIBUTION = 1500  # EUR
LOOKBACK_DAYS = 252          # 12 months
REBALANCE_FREQUENCY = 3      # Quarterly
```

## Data Sources

- **DEGIRO**: ETF catalog and metadata
- **JustETF**: Historical NAV prices (more complete than DEGIRO)
- **iShares**: Regional composition weights

## Performance

Signal computation is parallelized for performance:
- Uses all available CPU cores via multiprocessing
- Numba JIT compilation for filter operations
- Processes ~293 signals × 869 ETFs × 5,951 days
- Signal base computation: ~10-15 minutes
- Filter application: ~80 minutes (7,325 filtered signals)

## Documentation

For comprehensive documentation including:
- Detailed signal descriptions
- Filter explanations
- Backtesting API
- Strategy development guide

See **[CLAUDE.md](CLAUDE.md)**

## License & Disclaimer

This project is for **educational and research purposes only**. Not financial advice.

- Always consult a qualified financial advisor before making investment decisions
- Past performance does not guarantee future results
- The authors are not responsible for any financial losses

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Acknowledgments

- DEGIRO for API access
- JustETF for price data
- Academic references in CLAUDE.md
