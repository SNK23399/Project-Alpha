# ETF Portfolio Management - Core-Satellite Strategy

## Project Overview

This is a sophisticated Python-based portfolio management system that implements and backtests various **Core-Satellite investment strategies** using ETFs. The goal is to achieve **consistent positive alpha** over a global market baseline (ACWI) through tactical satellite allocation.

**Primary Focus**: Finding strategies with **consistent alpha** (high percentage of time with positive rolling alpha), NOT just maximum total returns.

## Technology Stack

- **Python 3.x** with Jupyter notebooks for analysis
- **Data Sources**:
  - DEGIRO API (via degiro-connector) for ETF catalog and historical prices
  - JustETF web scraping for enriched metadata and longer price history
- **Key Libraries**: pandas, numpy, scipy, matplotlib, mplcursors, scikit-learn
- **Authentication**: DEGIRO credentials via .env file with in-app approval flow

## Project Structure

```
Core Satellite/
├── Python Modules
│   ├── degiro_client.py       # DEGIRO API singleton with auto-authentication
│   ├── etf_fetcher.py          # ETF universe filtering (DEGIRO + JustETF)
│   ├── price_fetcher.py        # OHLC historical price data from DEGIRO
│   ├── backtester.py           # Core backtesting engine (CRITICAL MODULE)
│   └── notebook_style.py       # Dark mode Jupyter styling + mplcursors
│
├── Analysis Notebooks (Sequential workflow)
│   ├── 1_etf_universe_selection.ipynb          # Filter 7500+ ETFs → 511 candidates
│   ├── 2_satellite_optimization.ipynb          # Select optimal satellites per region
│   ├── 3_backtesting.ipynb                     # Core vs Core+Satellite comparison
│   ├── 4_rebalancing_strategy.ipynb            # Test Core/Satellite allocation splits
│   ├── 5_mean_reversion_strategy.ipynb         # Mean reversion for alpha generation
│   ├── 6_consistent_alpha_strategy.ipynb       # Find consistent (not just high) alpha
│   ├── 7_portfolio_optimization_methods.ipynb  # 6 academic optimization methods
│   └── 8_rolling_markowitz_optimization.ipynb  # Smart satellite prediction strategy
│
└── Data Storage
    ├── data/acwi_regional_weights.csv          # ACWI regional composition (2010-2025)
    ├── data/etf_universe.csv                   # Filtered ETF catalog
    ├── data/etfs/{ISIN}_prices.parquet         # Individual ETF price histories
    └── data/etfs/{ISIN}_meta.json              # ETF metadata (optional)
```

## Core Concepts

### Core-Satellite Strategy

**Philosophy**: Combine passive diversification (Core) with active alpha generation (Satellites)

- **Core (25-100%)**: iShares MSCI ACWI (IE00B4L5Y983)
  - Global market-cap weighted index
  - ~3000 stocks across developed + emerging markets
  - Low cost (0.20% TER), high liquidity

- **Satellites (0-75%)**: Regional/thematic ETFs
  - North America (S&P 500, NASDAQ, etc.)
  - Europe (EMU, UK, etc.)
  - Emerging Markets
  - Japan
  - Pacific ex-Japan
  - Sector-specific or factor-based ETFs

**Dynamic Allocation**: Satellite weights can mirror ACWI regional composition or use optimization methods

### Alpha Consistency Philosophy

**Traditional Approach** (rejected): Maximize cumulative return
- Problem: May have long periods of underperformance
- Example: +10% total alpha but only 40% of time positive

**This Project's Approach** (adopted): Maximize consistency of alpha
- Goal: Positive rolling alpha as much as possible
- **Metric**: % of time with positive 6-month rolling alpha
- Example: +2% total alpha but 85% of time positive → BETTER

## Key Modules

### 1. backtester.py - The Core Engine

**Purpose**: Flexible backtesting framework for portfolio strategies

**Key Classes**:

```python
class Backtester:
    """Main backtesting engine"""
    def __init__(self, prices, core_isin, satellite_isins,
                 core_weight, regional_weights)
    def run(self, start_date, end_date, rebalance_trigger,
            weight_calculator, initial_investment, monthly_contribution)

class PortfolioMetrics:
    """Performance metric calculations"""
    @staticmethod
    def calculate(portfolio: pd.Series) -> Dict[str, float]
    # Returns: total_return, ann_return, ann_vol, sharpe, sortino, calmar, max_drawdown

@dataclass
class BacktestResult:
    portfolio: pd.Series              # Daily portfolio values
    rebalance_dates: List[Timestamp]  # Rebalancing history
    weights_history: pd.DataFrame     # Weight evolution
    holdings_history: pd.DataFrame    # Units held over time
```

**Features**:
- Walk-forward backtesting (no look-ahead bias)
- Custom rebalancing triggers (monthly, quarterly, custom functions)
- Custom weight calculators (strategy-specific logic)
- Monthly contribution support
- Dynamic regional weight allocation based on ACWI composition

**Usage Example**:
```python
from backtester import Backtester, PortfolioMetrics

bt = Backtester(
    prices=price_df,
    core_isin='IE00B4L5Y983',
    satellite_isins={
        'North America': 'IE00B3YCGJ38',
        'Europe': 'IE00B53QG562',
        'Emerging Markets': 'IE00B4L5YC18',
        'Japan': 'IE00B4L5YX21',
        'Pacific ex-Japan': 'IE00B52MJY50'
    },
    core_weight=0.25,  # 25% ACWI
    regional_weights=df_regional_weights
)

result = bt.run(
    start_date='2020-01-01',
    end_date='2024-12-31',
    initial_investment=50000,
    monthly_contribution=1500
)

metrics = PortfolioMetrics.calculate(result.portfolio)
PortfolioMetrics.print_summary(metrics)
```

### 2. etf_fetcher.py - Universe Selection

**Purpose**: Filter DEGIRO's 7500+ ETF catalog to find suitable candidates

**Key Classes**:

```python
@dataclass
class ETFFilter:
    isins: List[str]              # Specific ISINs (overrides other filters)
    isin_prefix: str              # Domicile filter (e.g., "IE00" = Ireland)
    distribution: str             # "Accumulating" or "Distributing"
    max_ter: float                # Maximum TER in %
    min_fund_size: float          # Minimum AUM in millions EUR
    min_years: float              # Minimum years of price history
    currency: str                 # Currency (e.g., "EUR")
    exchange: str                 # Exchange code (e.g., "XET" = Xetra)
    deduplicate: bool             # One listing per ISIN (True by default)

class ETFFetcher:
    def fetch(self, filter: ETFFilter) -> pd.DataFrame
    # Returns: ISIN, Name, vwdId, exchange, currency, TER, fund_size, years_of_data
```

**Data Enrichment**: Combines DEGIRO catalog with JustETF metadata for better filtering

**Usage Example**:
```python
from etf_fetcher import ETFFetcher, ETFFilter

filter = ETFFilter(
    isin_prefix="IE00",           # Irish domicile (tax efficient)
    distribution="Accumulating",   # Reinvest dividends
    max_ter=0.30,                  # TER ≤ 0.30%
    min_fund_size=100,             # AUM ≥ 100M EUR
    min_years=5,                   # ≥ 5 years history
    currency="EUR"                 # EUR listings
)

fetcher = ETFFetcher()
df_universe = fetcher.fetch(filter)
# Result: ~511 ETFs matching criteria
```

### 3. price_fetcher.py - Historical Data

**Purpose**: Fetch OHLC price data from DEGIRO

**Key Features**:
- Automatic caching to parquet files (`data/prices/`)
- Batch fetching with progress bars
- Combined close price DataFrame generation

**Usage Example**:
```python
from price_fetcher import PriceFetcher

fetcher = PriceFetcher()
df = fetcher.fetch_ohlc("590959784")  # Single ETF
prices_dict = fetcher.fetch_all(vwd_ids)  # Multiple ETFs
closes = fetcher.get_combined_closes(vwd_ids)  # DataFrame with all closes
```

### 4. degiro_client.py - API Connection

**Purpose**: Singleton DEGIRO API client with automatic authentication

**Key Features**:
- In-app approval flow (mobile app confirmation)
- Credential management via .env file
- Connection error handling
- Singleton pattern (one connection per session)

**Configuration** (.env file in parent directory):
```bash
DEGIRO_USERNAME=your_username
DEGIRO_PASSWORD=your_password
DEGIRO_TOTP_SECRET=your_totp_secret  # Optional
```

## Analysis Notebooks Overview

### Chapter 1: ETF Universe Selection

**Goal**: Filter 7500+ ETFs → 511 qualified candidates

**Filters**:
- Irish domicile (IE00) - tax efficiency
- Accumulating distribution policy
- EUR currency
- TER ≤ 0.30%
- Fund size ≥ 100M EUR
- Price history ≥ 5 years

**Output**: [data/etf_universe.csv](data/etf_universe.csv)

### Chapter 2: Satellite Optimization

**Goal**: Select one optimal satellite ETF per region

**Methodology**:
- Analyze correlation between candidates within each region
- Evaluate tracking error, liquidity, TER
- Select best representative per region

**Selected Satellites**:
| Region | ETF | ISIN | TER | AUM |
|--------|-----|------|-----|-----|
| North America | Invesco S&P 500 | IE00B3YCGJ38 | 0.05% | €30.9B |
| Europe | iShares MSCI EMU | IE00B53QG562 | 0.12% | €5.7B |
| Emerging Markets | iShares MSCI EM | IE00B4L5YC18 | 0.18% | €5.2B |
| Japan | iShares MSCI Japan IMI | IE00B4L5YX21 | 0.12% | €5.9B |
| Pacific ex-Japan | iShares MSCI Pacific ex-Japan | IE00B52MJY50 | 0.20% | €2.9B |

### Chapter 3: Backtesting

**Goal**: Compare Core-only vs Core+Satellite strategies

**Tests**:
- 100% ACWI baseline
- Core+Satellite with dynamic regional allocation
- Drawdown analysis during market crashes
- Risk-adjusted return comparison

### Chapter 4: Rebalancing Strategy

**Goal**: Test different Core/Satellite allocation splits

**Allocations Tested**:
- 100/0 (100% Core - baseline)
- 75/25 (75% Core, 25% Satellites)
- 50/50 (balanced)
- 25/75 (satellite-heavy)
- 0/100 (100% Satellites)

**Finding**: Need balance between diversification and alpha generation

### Chapter 5: Mean Reversion Strategy

**Goal**: Test mean reversion for alpha generation

**Hypothesis**: Overweight underperforming regions (expecting reversion to mean)

**Configurations**: 27 combinations tested
- Lookback periods: 3, 6, 12 months
- Tilt strengths: Conservative, Moderate, Aggressive
- Rebalancing: Monthly, Quarterly, Semi-annual

**Result**: Mean reversion CAN generate alpha but requires careful parameter tuning

### Chapter 6: Consistent Alpha Strategy

**Goal**: Find strategies with positive alpha MOST of the time (not just highest cumulative alpha)

**Key Paradigm Shift**:
- Traditional: Maximize total return
- This chapter: Maximize % time with positive rolling alpha

**Evaluation Metric**: % of 6-month rolling windows with positive alpha vs baseline

**Strategies Tested**:
- Pure mean reversion
- Risk parity
- Composite blends

**Finding**: Composite strategies (multi-factor) are most consistent

### Chapter 7: Portfolio Optimization Methods ⭐ RECOMMENDED

**Goal**: Systematically compare 6 academic portfolio optimization methods

**Methods**:
1. **1/N Naïve** - Equal weight (20% each satellite)
2. **Mean-Variance** (Markowitz) - Minimize variance
3. **Minimum CVaR** - Minimize tail risk (95% confidence)
4. **Risk Parity Std** - Equal volatility contribution
5. **Risk Parity CVaR** - Equal tail risk contribution
6. **RP-CVaR Robust** - Worst-case risk parity (20% uncertainty)

**Configuration**:
- 25% Core (fixed), optimize 75% Satellite allocation
- Quarterly rebalancing
- 12-month lookback (252 trading days)
- CVaR at 95% confidence level

**Primary Metric**: % time with positive 6-month rolling alpha
**Secondary Metrics**: Sharpe ratio, max drawdown, turnover

**Academic References**:
- Markowitz (1952) - Portfolio Selection
- DeMiguel et al. (2009) - 1/N vs Optimal Diversification
- Andersson et al. (2000) - CVaR Optimization
- Maillard et al. (2010) - Risk Parity Properties
- Boudt et al. (2013) - CVaR Budgets
- Ben-Tal & Nemirovski (1998) - Robust Optimization

### Chapter 8: Rolling Markowitz Optimization

**Goal**: Smart satellite selection - predict which ETFs will outperform NEXT month

**Strategy**:
- Core: 60-100% ACWI (dynamic based on satellites found)
- Satellites: Up to 4 equity ETFs at 10% each (max 40% total)
- **Key Innovation**: Only select satellites predicted to beat core's expected return

**Selection Criteria**:
- Must be equity ETF (exclude bonds, commodities, leveraged)
- Must have positive 12-month return
- Must have Sharpe ratio > Core's Sharpe ratio
- Must have expected return > Core's expected return

**Multi-Factor Prediction System**:

**Momentum Factor** (40% weight):
- 12-1 month momentum (60% sub-weight)
- 6-1 month momentum (30% sub-weight)
- 3-1 month momentum (10% sub-weight)

**Quality Factor** (30% weight):
- Sharpe ratio (60% sub-weight)
- Sortino ratio (40% sub-weight)

**Trend Factor** (30% weight):
- Trend strength - % days above 200-day MA (40% sub-weight)
- Drawdown recovery (30% sub-weight)
- Calmar ratio (30% sub-weight)

**Prediction Methods Tested**:
1. Momentum-Based Prediction (momentum signals only)
2. Quality-Based Prediction (risk-adjusted returns)
3. Trend-Based Prediction (trend strength)
4. Combined Prediction (weighted multi-factor)

**Results** (13.9 years backtest):
| Method | Ann Return | Sharpe | Alpha vs ACWI | Avg Satellites |
|--------|-----------|--------|---------------|----------------|
| Quality-Based | 13.27% | 1.104 | **+1.44%** | 3.9 |
| Combined | 11.59% | 0.961 | -0.24% | 3.9 |
| Trend-Based | 11.58% | 1.006 | -0.26% | 2.4 |
| Momentum-Based | 10.59% | 0.854 | -1.24% | 3.9 |

**Winner**: Quality-Based Prediction (focus on Sharpe/Sortino ratios)

**Key Features**:
- Walk-forward backtest (no look-ahead bias)
- Daily portfolio tracking (not just rebalance dates)
- Dynamic allocation based on opportunities found
- Fallback to 100% Core if no satellites beat it

## Data Files

### ACWI Regional Weights (data/acwi_regional_weights.csv)

Monthly snapshots of iShares MSCI ACWI regional composition (2010-2025)

**Structure**:
```csv
date,North America,Europe,Emerging Markets,Japan,Pacific ex-Japan
2024-12-31,67.2,14.1,9.8,5.4,2.3
```

**Usage**: Dynamic satellite allocation that mirrors global market-cap distribution

### ETF Universe (data/etf_universe.csv)

Filtered ETF catalog after Chapter 1 criteria

**Columns**: ISIN, Name, vwdId, exchange, currency, TER, fund_size, years_of_data

**Count**: 511 ETFs (from original 7500+)

### Price Data (data/etfs/{ISIN}_prices.parquet)

Individual ETF price histories in parquet format

**Columns**: date (index), price (or open, high, low, close)

**Format**: Pandas DataFrame with DatetimeIndex

**Source**: JustETF (NAV-based daily prices, more complete than DEGIRO)

## Performance Metrics Explained

### Basic Metrics

**Total Return**: `(final_value / initial_value) - 1`

**Annualized Return**: `(1 + total_return) ^ (1 / years) - 1`

**Annualized Volatility**: `std(daily_returns) * sqrt(252)`

### Risk-Adjusted Metrics

**Sharpe Ratio**: `ann_return / ann_volatility`
- Measures return per unit of total risk
- Higher is better (>1.0 is good, >2.0 is excellent)
- Assumes risk-free rate = 0

**Sortino Ratio**: `ann_return / downside_deviation`
- Like Sharpe but only penalizes downside volatility
- Better reflects actual investor pain
- Higher values indicate better downside protection

**Calmar Ratio**: `ann_return / abs(max_drawdown)`
- Return per unit of worst loss
- Useful for understanding recovery potential

### Drawdown Metrics

**Drawdown**: `(portfolio_value - running_max) / running_max`
- Current loss from peak
- Measures portfolio "pain" at any point

**Maximum Drawdown**: `min(all_drawdowns)`
- Worst peak-to-trough decline
- Key metric for risk tolerance assessment

### Alpha Metrics

**Alpha vs Baseline**: `portfolio_ann_return - baseline_ann_return`
- Outperformance vs ACWI benchmark
- Positive = beating the market

**Rolling Alpha Consistency**: `% of rolling windows with positive alpha`
- This project's PRIMARY metric
- Measures how often strategy beats benchmark
- More valuable than cumulative alpha for investor experience

## Optimization Methods Deep Dive

### 1. Equal Weight (1/N Naïve)

**Formula**: `w_i = 1/N` for all assets

**Pros**:
- No estimation error
- Low turnover
- Robust to parameter uncertainty

**Cons**:
- Ignores risk and correlation structure
- May concentrate risk in volatile assets

**Use Case**: Benchmark, works well when estimation error is high

### 2. Markowitz Mean-Variance

**Objective**: Minimize variance subject to constraints

**Optimization**:
```python
minimize: w^T Σ w                    # Portfolio variance
subject to: sum(w) = 1               # Fully invested
            w >= 0                   # Long only
```

**Pros**:
- Theoretically optimal for risk minimization
- Well-studied framework

**Cons**:
- Sensitive to covariance estimation errors
- Tends to concentrate in low-volatility assets
- High turnover

**Enhancement**: Ledoit-Wolf shrinkage estimator for covariance matrix

### 3. Minimum CVaR (Conditional Value-at-Risk)

**Objective**: Minimize tail risk (expected loss in worst 5% scenarios)

**CVaR Definition**: `CVaR_α = E[loss | loss > VaR_α]`
- VaR_α = Value at Risk at confidence level α (e.g., 95%)
- CVaR = average loss beyond VaR threshold

**Optimization**:
```python
minimize: CVaR_0.95(portfolio_returns)
subject to: sum(w) = 1
            w >= 0
```

**Pros**:
- Better tail risk protection than variance
- Coherent risk measure (subadditive)

**Cons**:
- Requires good return distribution estimates
- Can be unstable with limited data

### 4. Risk Parity (Volatility)

**Objective**: Equalize risk contribution across assets

**Risk Contribution**: `RC_i = w_i * (Σw)_i`
- Each asset contributes equally to total portfolio risk

**Target**: `RC_i = (1/N) * σ_portfolio` for all i

**Optimization**:
```python
minimize: sum((RC_i - target)^2)
subject to: sum(w) = 1
            w >= 0
```

**Pros**:
- Better diversification than market-cap or equal weight
- Reduces concentration in volatile assets

**Cons**:
- Can overweight low-volatility, low-return assets
- Doesn't account for expected returns

### 5. Risk Parity (CVaR)

**Objective**: Equalize CVaR contribution (tail risk) across assets

**CVaR Contribution**: Like risk parity but using CVaR instead of volatility

**Pros**:
- Better tail risk diversification
- More relevant for drawdown-averse investors

**Cons**:
- More computationally intensive
- Requires robust CVaR estimation

### 6. Robust Risk Parity CVaR

**Objective**: Risk parity with worst-case parameter uncertainty

**Uncertainty Set**: Allow ±20% uncertainty in estimated parameters

**Optimization**:
```python
minimize: max_over_uncertainty_set(CVaR_portfolio)
subject to: equal CVaR contribution
            sum(w) = 1
            w >= 0
```

**Pros**:
- Robust to estimation errors
- Conservative allocation
- Better out-of-sample performance

**Cons**:
- May be overly conservative
- Computationally expensive

## Walk-Forward Backtesting

**Critical Concept**: Prevent look-ahead bias in strategy evaluation

### Standard Backtest (WRONG)
```python
# Calculate metrics using ALL data
metrics = calculate_metrics(full_price_history)

# Use metrics to select portfolio
weights = optimize(metrics)

# Test on same data
result = backtest(weights, full_price_history)
```
**Problem**: Uses future information to make past decisions = overfitting

### Walk-Forward Backtest (CORRECT)
```python
for rebalance_date in rebalance_schedule:
    # Only use data UP TO rebalance date
    historical_data = prices[prices.index <= rebalance_date]

    # Calculate metrics using ONLY historical data
    metrics = calculate_metrics(historical_data.tail(lookback_days))

    # Optimize using historical metrics
    weights = optimize(metrics)

    # Hold these weights until NEXT rebalance
    # (performance measured on unseen future data)
```

**Benefits**:
- Realistic performance estimates
- Prevents data snooping
- Tests strategy's true predictive power

**Implementation in backtester.py**:
- Rebalancing dates determined in advance
- Weight calculation receives `date` parameter
- Only prices up to `date` accessible
- Performance measured on holding period after optimization

## Common Workflows

### 1. Quick Strategy Backtest

```python
from backtester import Backtester, PortfolioMetrics
import pandas as pd

# Load prices
prices = pd.read_parquet('combined_prices.parquet')

# Define portfolio
bt = Backtester(
    prices=prices,
    core_isin='IE00B4L5Y983',
    satellite_isins={'North America': 'IE00B3YCGJ38'},
    core_weight=0.75  # 75% ACWI, 25% S&P 500
)

# Run backtest
result = bt.run(
    start_date='2020-01-01',
    end_date='2024-12-31',
    initial_investment=50000,
    monthly_contribution=1500
)

# Analyze
metrics = PortfolioMetrics.calculate(result.portfolio)
PortfolioMetrics.print_summary(metrics)

# Plot
import matplotlib.pyplot as plt
result.portfolio.plot(title='Portfolio Growth')
plt.show()
```

### 2. Custom Rebalancing Strategy

```python
# Define custom rebalancing trigger
def quarterly_on_first_monday(date, prices, units_held, daily_prices,
                              last_rebalance, **kwargs):
    if last_rebalance is None:
        return True

    # Check if quarter changed
    months_diff = (date.year - last_rebalance.year) * 12 + \
                  (date.month - last_rebalance.month)

    if months_diff >= 3:
        # First Monday of new quarter
        return date.weekday() == 0

    return False

# Use custom trigger
result = bt.run(rebalance_trigger=quarterly_on_first_monday)
```

### 3. Custom Weight Calculator (Mean Reversion)

```python
def mean_reversion_weights(date, prices):
    """Overweight underperforming regions"""
    # Get historical prices up to date
    hist = prices[prices.index <= date].tail(252)  # 12 months

    # Calculate returns
    returns = (hist.iloc[-1] / hist.iloc[0]) - 1

    # Invert returns (underperformers get higher weight)
    inverted = 1 / (1 + returns)
    inverted = inverted / inverted.sum()  # Normalize

    # Build weights dict
    weights = {
        'IE00B4L5Y983': 0.25,  # Core fixed
        'IE00B3YCGJ38': inverted['IE00B3YCGJ38'] * 0.75,  # Satellites
        # ... other satellites
    }

    return weights

# Use custom weights
result = bt.run(weight_calculator=mean_reversion_weights)
```

### 4. Comparing Multiple Strategies

```python
strategies = {
    '100% Core': {'core_weight': 1.0},
    '75/25': {'core_weight': 0.75},
    '50/50': {'core_weight': 0.50},
    '25/75': {'core_weight': 0.25}
}

results = {}
for name, config in strategies.items():
    bt = Backtester(prices=prices, core_isin=core,
                    satellite_isins=satellites, **config)
    result = bt.run(start_date='2020-01-01', end_date='2024-12-31')
    metrics = PortfolioMetrics.calculate(result.portfolio)
    results[name] = metrics

# Compare
import pandas as pd
df_comparison = pd.DataFrame(results).T
print(df_comparison.sort_values('sharpe', ascending=False))
```

## Investment Configuration

### Default Parameters (from README)

```python
# Portfolio allocation
CORE_WEIGHT = 0.25           # 25% in ACWI
SATELLITE_WEIGHT = 0.75      # 75% in satellites

# Investment amounts
INITIAL_INVESTMENT = 50000   # EUR 50,000
MONTHLY_CONTRIBUTION = 1500  # EUR 1,500/month

# Optimization parameters
LOOKBACK_DAYS = 252          # 12 months (252 trading days)
REBALANCE_FREQUENCY = 3      # Quarterly (every 3 months)
CVAR_ALPHA = 0.05           # 95% confidence level
UNCERTAINTY = 0.20          # 20% parameter uncertainty (robust methods)
```

### Filtering Criteria (Chapter 1)

```python
ETFFilter(
    isin_prefix="IE00",           # Irish domicile
    distribution="Accumulating",  # Reinvest dividends
    currency="EUR",               # EUR currency
    max_ter=0.30,                 # TER ≤ 0.30%
    min_fund_size=100,            # AUM ≥ 100M EUR
    min_years=5                   # ≥ 5 years history
)
```

## Tips & Best Practices

### Data Management

1. **Cache Aggressively**: ETF and price data changes infrequently
   - Save fetched data to parquet files
   - Only re-fetch when needed (monthly/quarterly)

2. **Handle Missing Data**: Not all ETFs have complete history
   - Use `dropna(how='all')` for dates
   - Use `ffill()` for small gaps
   - Filter ETFs with insufficient data before backtesting

3. **Align Dates**: DEGIRO and JustETF may have different calendars
   - Use JustETF for longer history (NAV-based)
   - Resample to business days only
   - Forward-fill to align timeseries

### Backtesting

1. **Always Walk-Forward**: Never use future data
   - Calculate metrics only on historical data
   - Optimize on past, test on future
   - Use `date` parameter in custom functions

2. **Account for Transaction Costs**: This project doesn't model costs
   - Consider adding spread/commission in production
   - Penalize high-turnover strategies
   - Batch small trades to reduce costs

3. **Rebalancing Frequency**: Balance performance vs costs
   - Monthly = more responsive but higher turnover
   - Quarterly = good balance (used in Chapter 7)
   - Annual = low cost but may miss opportunities

4. **Lookback Period**: Longer ≠ Better
   - 12 months (252 days) = good default
   - Too short = noisy estimates
   - Too long = stale data, misses regime changes

### Strategy Development

1. **Start Simple**: Equal weight often beats complex optimization
   - Test 1/N naïve as baseline
   - Add complexity only if it improves consistency
   - Watch out for overfitting

2. **Focus on Consistency**: Not just total return
   - Plot rolling alpha over time
   - Calculate % positive periods
   - Prefer steady gains over volatile outperformance

3. **Test Robustness**: Good strategies work in multiple regimes
   - Backtest multiple time periods
   - Test on out-of-sample data
   - Vary parameters to check sensitivity

4. **Combine Signals**: Multi-factor often more stable
   - Momentum + Quality + Trend
   - Weight factors based on evidence
   - Rebalance factor weights periodically

## Troubleshooting

### DEGIRO Connection Issues

**Problem**: "Connection failed" or "No approval received"

**Solutions**:
1. Check .env file exists in parent directory
2. Verify credentials are correct
3. Approve quickly on mobile app (<2 minutes)
4. Reset singleton if needed: `DegiroClient.reset()`

### Missing Price Data

**Problem**: Some ETFs have no price history

**Solutions**:
1. Check if ETF is listed on DEGIRO (may be delisted)
2. Try JustETF scraping instead: `justetf_scraping.load_chart(isin)`
3. Filter universe to require minimum data availability
4. Use `min_years` parameter in ETFFilter

### Optimization Failures

**Problem**: Optimizer doesn't converge or returns extreme weights

**Solutions**:
1. Use Ledoit-Wolf shrinkage for covariance estimation
2. Add regularization (L2 penalty on weights)
3. Impose stricter constraints (max weight per asset)
4. Increase lookback period for more stable estimates
5. Fall back to 1/N if optimization fails

### Performance Issues

**Problem**: Backtests take too long

**Solutions**:
1. Reduce universe size (filter before backtesting)
2. Use quarterly instead of monthly rebalancing
3. Cache intermediate calculations
4. Parallelize multiple strategy tests
5. Use numpy/pandas operations (avoid Python loops)

## Future Enhancements

### Potential Improvements

1. **Transaction Costs**: Model spreads, commissions, market impact
2. **Tax Optimization**: Consider dividend taxes, capital gains
3. **Currency Hedging**: Add FX risk management for non-EUR ETFs
4. **Factor Exposure**: Analyze style tilts (value, growth, momentum, quality)
5. **Out-of-Sample Testing**: Reserve recent data for validation
6. **Monte Carlo**: Simulate future scenarios for robustness testing
7. **Machine Learning**: Use ML for satellite selection predictions
8. **Real-Time Monitoring**: Alert when strategy drifts from targets

### Code Architecture Ideas

1. **Strategy Interface**: Abstract base class for all strategies
2. **Pipeline Framework**: Modular data → optimize → backtest → evaluate
3. **Configuration Management**: YAML/JSON for all parameters
4. **Logging**: Track all decisions for post-mortem analysis
5. **Unit Tests**: Ensure backtester correctness (especially walk-forward)
6. **Web Dashboard**: Streamlit/Dash for interactive exploration

## Quick Reference

### Key ISINs

- **Core**: IE00B4L5Y983 (iShares MSCI ACWI)
- **North America**: IE00B3YCGJ38 (Invesco S&P 500)
- **Europe**: IE00B53QG562 (iShares MSCI EMU)
- **Emerging Markets**: IE00B4L5YC18 (iShares MSCI EM)
- **Japan**: IE00B4L5YX21 (iShares MSCI Japan IMI)
- **Pacific ex-Japan**: IE00B52MJY50 (iShares MSCI Pacific ex-Japan)

### Important Functions

```python
# Fetching
from etf_fetcher import fetch_etfs, ETFFilter
from price_fetcher import PriceFetcher
from degiro_client import get_api

# Backtesting
from backtester import Backtester, PortfolioMetrics, BacktestResult

# Styling
from notebook_style import apply_dark_style, add_cursor
```

### Directory Paths

- ETF metadata: `data/etfs/{ISIN}_meta.json`
- Price data: `data/etfs/{ISIN}_prices.parquet`
- Regional weights: `data/acwi_regional_weights.csv`
- Universe: `data/etf_universe.csv`

## Conclusion

This project demonstrates a systematic approach to **Core-Satellite portfolio construction** with emphasis on **alpha consistency** over maximum returns. The backtesting framework is robust (walk-forward, no look-ahead), and the methodology is grounded in academic research while remaining practical for retail investors.

**Key Takeaway**: Simple strategies (1/N, risk parity) often outperform complex optimization, especially when accounting for estimation error and transaction costs. The best strategy is one that you can understand, stick with through volatility, and that generates alpha MOST of the time, not just in aggregate.

**Recommended Starting Point**:
1. Read this file (you're here!)
2. Run Chapter 7 to compare optimization methods
3. Pick the method with highest alpha consistency
4. Implement with quarterly rebalancing
5. Monitor rolling alpha and adjust annually

Happy investing!
