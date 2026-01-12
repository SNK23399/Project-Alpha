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
├── Data Pipeline
│   ├── 1_compute_signal_bases.py     # Compute 293 raw signals
│   ├── 2_apply_filters.py            # Apply 25 smoothing filters
│   └── 3_compute_features.py         # Combine into ML features
│
├── Signal Libraries
│   ├── library_signal_bases.py       # Signal computation (parallelized)
│   ├── library_signal_filters.py     # Filter definitions (Numba-accelerated)
│   └── signal_indicators.py          # Indicator transformations
│
├── Testing & Validation
│   ├── test_signal_correctness.py    # Signal correctness tests
│   └── validate_filtered_signals.py  # Filtered signal validation
│
├── Analysis
│   └── ensemble_discovery.py         # Ensemble model discovery
│
├── Maintenance (Database Updates)
│   ├── maintenance/1_collect_etf_data.py      # Fetch ETF data
│   ├── maintenance/2_compare_databases.py     # Validate and replace database
│   └── maintenance/3_analyze_data_quality.py  # Data quality analysis
│
├── Support Modules
│   ├── support/backtester.py         # Backtesting engine
│   ├── support/degiro_client.py      # DEGIRO API client
│   ├── support/etf_database.py       # ETF data storage (SQLite)
│   ├── support/etf_fetcher.py        # ETF catalog fetching
│   ├── support/price_fetcher.py      # Price data fetching
│   └── support/signal_database.py    # Signal data storage (Parquet)
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

### 4. Compute Features

```bash
python 3_compute_features.py
```

Combines filtered signals into ML-ready features for prediction.

## Core Concepts

### Core-Satellite Strategy

- **Core**: iShares Core MSCI World (IE00B4L5Y983) - Global diversification
- **Satellites**: Regional/thematic ETFs for tactical allocation

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

# 4. Recompute signals (incremental)
cd ..
python 1_compute_signal_bases.py incremental
python 2_apply_filters.py
python 3_compute_features.py
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
