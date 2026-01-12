"""
Improved Backtest: Quarterly Rebalancing with Momentum + Quality Selection

Key Improvements:
1. QUARTERLY rebalancing (every 63 days) - features strongest at this horizon
2. COMBINED selection: Momentum (12-month return) + Quality (Sharpe ratio)
3. DYNAMIC allocation: 30-50% satellites based on opportunities found
4. CORRECT transaction costs: EUR 1 per trade

Strategy:
- Start with 60% core / 40% satellites as baseline
- Each quarter: rank ETFs by combined momentum + quality score
- Select top 4 ETFs with positive returns and Sharpe > Core's Sharpe
- If fewer than 4 ETFs qualify, reduce satellite allocation proportionally
- Minimum satellite allocation: 30% (if at least 3 ETFs qualify)
- Maximum satellite allocation: 50% (if 5+ ETFs qualify with excellent scores)

Expected: Positive alpha by:
- Reducing transaction costs (112 → ~40 rebalances)
- Better ETF selection (quality + momentum)
- Dynamic allocation (avoid weak periods)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from support.etf_database import ETFDatabase
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
CORE_ISIN = 'IE00B4L5Y983'  # iShares MSCI World
MONTHLY_CONTRIBUTION = 1500  # EUR
INITIAL_INVESTMENT = 50000  # EUR
TRANSACTION_COST_PER_TRADE = 1.0  # EUR 1 per trade

# Strategy parameters
LOOKBACK_DAYS = 252  # 12 months for momentum/Sharpe
REBALANCE_DAYS = 63  # Quarterly (matches strongest feature horizon)
MIN_SATELLITES = 3
TARGET_SATELLITES = 4
MAX_SATELLITES = 5

# Allocation ranges
ALLOCATION_CONFIG = {
    'min_core': 0.50,      # Never less than 50% core (defensive)
    'max_core': 0.70,      # Never more than 70% core (if many good satellites)
    'baseline_satellite': 0.40  # Target 40% satellites if 4 ETFs qualify
}

START_DATE = '2015-01-01'
END_DATE = '2024-12-31'

print("=" * 80)
print("IMPROVED BACKTEST: Quarterly Rebalancing + Momentum+Quality Selection")
print("=" * 80)
print(f"\nPeriod: {START_DATE} to {END_DATE}")
print(f"Initial Investment: EUR {INITIAL_INVESTMENT:,.0f}")
print(f"Monthly Contribution: EUR {MONTHLY_CONTRIBUTION:,.0f}")
print(f"Transaction Cost: EUR {TRANSACTION_COST_PER_TRADE:.2f} per trade")
print(f"\nStrategy:")
print(f"- Rebalance every {REBALANCE_DAYS} days (quarterly)")
print(f"- Select {MIN_SATELLITES}-{MAX_SATELLITES} ETFs by momentum + quality")
print(f"- Dynamic allocation: {ALLOCATION_CONFIG['max_core']*100:.0f}-{ALLOCATION_CONFIG['min_core']*100:.0f}% core")
print(f"- Satellite allocation: {(1-ALLOCATION_CONFIG['max_core'])*100:.0f}-{(1-ALLOCATION_CONFIG['min_core'])*100:.0f}%")

# --- LOAD PRICE DATA ---
print("\n" + "-" * 80)
print("Loading price data...")
print("-" * 80)

# Load ETF database
etf_db = ETFDatabase("data/etf_database.db")

# Load ETF universe
universe_df = etf_db.load_universe()
all_isins = universe_df['isin'].tolist()
print(f"\nETF universe: {len(all_isins)} ETFs")

# Load prices for ALL ETFs
print("\nLoading ETF prices (this may take a minute)...")

all_prices = {}

# Core
core_prices = etf_db.load_prices(CORE_ISIN)
if core_prices is not None:
    all_prices[CORE_ISIN] = core_prices
else:
    print(f"\nERROR: Core ETF {CORE_ISIN} not found in database!")
    exit(1)

# All universe ETFs
for isin in tqdm(all_isins, desc="Loading prices"):
    if isin == CORE_ISIN:
        continue

    prices = etf_db.load_prices(isin)
    if prices is not None and len(prices) > 0:
        all_prices[isin] = prices

prices_df = pd.DataFrame(all_prices)

# Filter to backtest period
prices_df = prices_df[(prices_df.index >= START_DATE) & (prices_df.index <= END_DATE)]

print(f"\nLoaded {len(prices_df.columns)} ETFs")
print(f"Date range: {prices_df.index.min().date()} to {prices_df.index.max().date()}")
print(f"Total days: {len(prices_df)}")
print(f"Core ETF: {CORE_ISIN}")

# --- SATELLITE SELECTION FUNCTION ---
def calculate_sharpe_ratio(returns_series):
    """Calculate annualized Sharpe ratio (assuming 0% risk-free rate)"""
    if len(returns_series) < 21:
        return 0.0
    mean_return = returns_series.mean() * 252  # Annualized
    std_return = returns_series.std() * np.sqrt(252)  # Annualized
    if std_return == 0:
        return 0.0
    return mean_return / std_return


def select_top_etfs_quality(prices_df, date, lookback_days=252,
                            min_select=3, target_select=4, max_select=5,
                            exclude_isin=CORE_ISIN):
    """
    Select top ETFs using COMBINED momentum + quality scoring.

    Scoring:
    - 60% weight: 12-month momentum (trailing return)
    - 40% weight: Sharpe ratio (risk-adjusted return quality)

    Filters:
    - Must have positive 12-month return
    - Must have Sharpe ratio > Core's Sharpe ratio (better quality)
    - Must have sufficient data

    Dynamic allocation:
    - If 5+ ETFs qualify with excellent scores: 50% satellites (5 ETFs x 10%)
    - If 4 ETFs qualify: 40% satellites (baseline, 4 ETFs x 10%)
    - If 3 ETFs qualify: 30% satellites (3 ETFs x 10%)
    - If <3 ETFs qualify: 0% satellites (100% core)

    Returns:
        selected: List of ISINs
        satellite_allocation: Total % to allocate to satellites
    """
    end_date = date
    start_date = date - pd.Timedelta(days=lookback_days * 1.5)  # Extra buffer

    hist = prices_df[(prices_df.index >= start_date) & (prices_df.index <= end_date)]

    if len(hist) < lookback_days * 0.9:  # Need at least 90% of lookback
        return [], 0.0

    # Take last lookback_days
    hist = hist.tail(lookback_days)

    if len(hist) < 21:  # Need minimum data
        return [], 0.0

    # Calculate core Sharpe (threshold for quality)
    core_returns = hist[CORE_ISIN].pct_change().dropna()
    core_sharpe = calculate_sharpe_ratio(core_returns)

    # Score each ETF
    scores = {}
    momentum_vals = {}
    sharpe_vals = {}

    for isin in prices_df.columns:
        if isin == exclude_isin:
            continue

        if isin not in hist.columns or hist[isin].isna().all():
            continue

        # Calculate momentum (12-month return)
        start_price = hist[isin].iloc[0]
        end_price = hist[isin].iloc[-1]

        if pd.isna(start_price) or pd.isna(end_price) or start_price <= 0:
            continue

        momentum = (end_price / start_price) - 1

        # Filter: must be positive
        if momentum <= 0:
            continue

        # Calculate Sharpe ratio
        returns = hist[isin].pct_change().dropna()
        if len(returns) < 21:
            continue

        sharpe = calculate_sharpe_ratio(returns)

        # Filter: must beat core's Sharpe
        if sharpe <= core_sharpe:
            continue

        # Combined score: 60% momentum, 40% Sharpe
        # Normalize both to 0-1 range for combining
        momentum_vals[isin] = momentum
        sharpe_vals[isin] = sharpe

    # Normalize scores to 0-1 range
    if not momentum_vals:
        return [], 0.0

    min_momentum = min(momentum_vals.values())
    max_momentum = max(momentum_vals.values())
    momentum_range = max_momentum - min_momentum

    min_sharpe = min(sharpe_vals.values())
    max_sharpe = max(sharpe_vals.values())
    sharpe_range = max_sharpe - min_sharpe

    for isin in momentum_vals.keys():
        # Normalize momentum
        if momentum_range > 0:
            norm_momentum = (momentum_vals[isin] - min_momentum) / momentum_range
        else:
            norm_momentum = 1.0

        # Normalize Sharpe
        if sharpe_range > 0:
            norm_sharpe = (sharpe_vals[isin] - min_sharpe) / sharpe_range
        else:
            norm_sharpe = 1.0

        # Combined score
        scores[isin] = 0.6 * norm_momentum + 0.4 * norm_sharpe

    # Sort by combined score
    sorted_etfs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Dynamic allocation based on opportunities
    n_qualified = len(sorted_etfs)

    if n_qualified >= max_select:
        # Many good opportunities: allocate 50% to satellites
        selected = [isin for isin, score in sorted_etfs[:max_select]]
        satellite_allocation = 0.50
    elif n_qualified >= target_select:
        # Target met: baseline 40% allocation
        selected = [isin for isin, score in sorted_etfs[:target_select]]
        satellite_allocation = 0.40
    elif n_qualified >= min_select:
        # Minimum met: conservative 30% allocation
        selected = [isin for isin, score in sorted_etfs[:min_select]]
        satellite_allocation = 0.30
    else:
        # Not enough good ETFs: go 100% core
        selected = []
        satellite_allocation = 0.0

    return selected, satellite_allocation


# --- BACKTEST LOOP ---
print("\n" + "-" * 80)
print("Running backtest...")
print("-" * 80)

# Initialize
portfolio_values = []
dates = []
holdings = {CORE_ISIN: 0}  # ISIN -> number of shares
cash = INITIAL_INVESTMENT
last_rebalance_date = None
rebalance_dates = []
weights_history = []
total_transaction_costs = 0
rebalance_count = 0

for i, date in enumerate(prices_df.index):
    # Monthly contribution (add on first trading day of each month)
    if i > 0:
        prev_date = prices_df.index[i-1]
        if date.month != prev_date.month:
            cash += MONTHLY_CONTRIBUTION

    # Check if we should rebalance
    should_rebalance = False

    if last_rebalance_date is None:
        should_rebalance = True  # First day
    else:
        days_since_rebalance = (date - last_rebalance_date).days
        if days_since_rebalance >= REBALANCE_DAYS:
            should_rebalance = True

    # Rebalance
    if should_rebalance:
        rebalance_count += 1

        # Select satellites
        selected_isins, satellite_allocation = select_top_etfs_quality(
            prices_df, date,
            lookback_days=LOOKBACK_DAYS,
            min_select=MIN_SATELLITES,
            target_select=TARGET_SATELLITES,
            max_select=MAX_SATELLITES,
            exclude_isin=CORE_ISIN
        )

        core_allocation = 1 - satellite_allocation

        # Calculate current portfolio value
        portfolio_value = cash
        for isin, shares in holdings.items():
            if shares > 0 and isin in prices_df.columns:
                price = prices_df.loc[date, isin]
                if pd.notna(price):
                    portfolio_value += shares * price

        # Build target weights
        target_weights = {CORE_ISIN: core_allocation}

        if selected_isins:
            weight_per_satellite = satellite_allocation / len(selected_isins)
            for isin in selected_isins:
                target_weights[isin] = weight_per_satellite

        # Calculate target holdings (number of shares)
        target_holdings = {}
        for isin, weight in target_weights.items():
            if isin in prices_df.columns:
                price = prices_df.loc[date, isin]
                if pd.notna(price) and price > 0:
                    target_value = portfolio_value * weight
                    target_holdings[isin] = target_value / price

        # Calculate trades needed
        n_trades = 0
        old_holdings_set = set(holdings.keys())
        new_holdings_set = set(target_holdings.keys())

        # Positions being closed
        for isin in old_holdings_set - new_holdings_set:
            if holdings.get(isin, 0) > 0:
                n_trades += 1

        # Positions being opened or changed
        for isin in new_holdings_set:
            old_shares = holdings.get(isin, 0)
            new_shares = target_holdings[isin]

            # Count as trade if difference > 1%
            if abs(new_shares - old_shares) / max(new_shares, old_shares, 1) > 0.01:
                n_trades += 1

        # Apply transaction costs
        transaction_cost = n_trades * TRANSACTION_COST_PER_TRADE
        total_transaction_costs += transaction_cost

        # Sell all current holdings
        for isin, shares in holdings.items():
            if shares > 0 and isin in prices_df.columns:
                price = prices_df.loc[date, isin]
                if pd.notna(price):
                    cash += shares * price

        # Deduct transaction costs
        cash -= transaction_cost

        # Buy new holdings
        holdings = {}
        for isin, shares in target_holdings.items():
            price = prices_df.loc[date, isin]
            if pd.notna(price) and price > 0:
                cost = shares * price
                if cost <= cash:
                    holdings[isin] = shares
                    cash -= cost

        # Record rebalance
        last_rebalance_date = date
        rebalance_dates.append(date)
        weights_history.append({
            'date': date,
            'core_weight': core_allocation,
            'satellite_weight': satellite_allocation,
            'n_satellites': len(selected_isins),
            'transaction_cost': transaction_cost
        })

    # Calculate portfolio value
    portfolio_value = cash
    for isin, shares in holdings.items():
        if shares > 0 and isin in prices_df.columns:
            price = prices_df.loc[date, isin]
            if pd.notna(price):
                portfolio_value += shares * price

    portfolio_values.append(portfolio_value)
    dates.append(date)

# --- CREATE RESULTS DATAFRAME ---
print(f"\nCompleted {rebalance_count} rebalances")

result_df = pd.DataFrame({
    'date': dates,
    'portfolio_value': portfolio_values
})
result_df.set_index('date', inplace=True)

# Calculate portfolio returns
result_df['portfolio_return'] = result_df['portfolio_value'].pct_change()

# --- CALCULATE METRICS ---
print("\n" + "=" * 80)
print("IMPROVED STRATEGY RESULTS")
print("=" * 80)

years = (result_df.index[-1] - result_df.index[0]).days / 365.25
total_contributions = INITIAL_INVESTMENT + (MONTHLY_CONTRIBUTION * len(result_df) / 21)  # Approx monthly
final_value = result_df['portfolio_value'].iloc[-1]
total_return = (final_value / INITIAL_INVESTMENT) - 1
annual_return = (1 + total_return) ** (1 / years) - 1

returns = result_df['portfolio_return'].dropna()
annual_vol = returns.std() * np.sqrt(252)
sharpe = annual_return / annual_vol if annual_vol > 0 else 0

# Drawdown
cummax = result_df['portfolio_value'].cummax()
drawdown = (result_df['portfolio_value'] - cummax) / cummax
max_drawdown = drawdown.min()

print(f"\nFinal Portfolio Value: EUR {final_value:,.2f}")
print(f"Total Return: {total_return*100:.2f}%")
print(f"Annual Return: {annual_return*100:.2f}%")
print(f"Annual Volatility: {annual_vol*100:.2f}%")
print(f"Sharpe Ratio: {sharpe:.3f}")
print(f"Max Drawdown: {max_drawdown*100:.2f}%")
print(f"\nTotal Transaction Costs: EUR {total_transaction_costs:,.2f}")
print(f"Number of Rebalances: {rebalance_count}")
print(f"Avg Cost per Rebalance: EUR {total_transaction_costs/rebalance_count:.2f}")

# --- COMPARE TO BASELINE ---
print("\n" + "=" * 80)
print("COMPARISON: Improved vs Original 60/40 Monthly Strategy")
print("=" * 80)

# Calculate 100% Core baseline
core_prices = prices_df[CORE_ISIN]
core_portfolio = [INITIAL_INVESTMENT]
core_cash = 0

for i in range(1, len(core_prices)):
    prev_date = core_prices.index[i-1]
    curr_date = core_prices.index[i]

    # Monthly contribution
    if curr_date.month != prev_date.month:
        core_cash += MONTHLY_CONTRIBUTION
        # Buy core
        shares = core_cash / core_prices.iloc[i]
        core_portfolio[-1] += core_cash
        core_cash = 0

    # Calculate value
    value = core_portfolio[0] * (core_prices.iloc[i] / core_prices.iloc[0])
    core_portfolio.append(value)

core_df = pd.DataFrame({
    'date': core_prices.index,
    'portfolio_value': core_portfolio
})
core_df.set_index('date', inplace=True)

core_final = core_df['portfolio_value'].iloc[-1]
core_total_return = (core_final / INITIAL_INVESTMENT) - 1
core_annual_return = (1 + core_total_return) ** (1 / years) - 1

# Alpha calculation
alpha = annual_return - core_annual_return

print(f"\n100% Core Strategy:")
print(f"  Final Value: EUR {core_final:,.2f}")
print(f"  Annual Return: {core_annual_return*100:.2f}%")

print(f"\nOriginal 60/40 Monthly (from previous backtest):")
print(f"  Final Value: EUR 299,186")
print(f"  Annual Return: 19.59%")
print(f"  Alpha: -0.64%")
print(f"  TX Costs: EUR 763")

print(f"\nImproved Quarterly Strategy:")
print(f"  Final Value: EUR {final_value:,.2f}")
print(f"  Annual Return: {annual_return*100:.2f}%")
print(f"  Alpha vs Core: {alpha*100:+.2f}%")
print(f"  TX Costs: EUR {total_transaction_costs:,.2f}")

if alpha > 0:
    print(f"\n*** SUCCESS: Positive alpha of {alpha*100:+.2f}% achieved! ***")
else:
    print(f"\n*** Still underperforming by {-alpha*100:.2f}% ***")

# Savings analysis
monthly_costs = 763  # From previous backtest
cost_savings = monthly_costs - total_transaction_costs

print(f"\nTransaction Cost Comparison:")
print(f"  Monthly rebalancing: EUR {monthly_costs:,.2f} (112 rebalances)")
print(f"  Quarterly rebalancing: EUR {total_transaction_costs:,.2f} ({rebalance_count} rebalances)")
print(f"  Savings: EUR {cost_savings:,.2f}")

# --- SAVE RESULTS ---
print("\n" + "-" * 80)
print("Saving results...")
print("-" * 80)

output_dir = Path('data/backtest_results')
output_dir.mkdir(exist_ok=True)

result_df.to_parquet(output_dir / 'improved_strategy_results.parquet')
core_df.to_parquet(output_dir / 'core_baseline_results.parquet')

weights_df = pd.DataFrame(weights_history)
weights_df.to_parquet(output_dir / 'improved_strategy_weights.parquet')

print(f"\nResults saved to {output_dir}")
print("\nRun the dashboard script to visualize results!")
print("\n" + "=" * 80)
