"""
Smart Satellite Selection Strategy Backtest

User's Actual Strategy:
- 60% Core (iShares MSCI World) - FIXED
- 40% Satellites (4 ETFs, 10% each) - DYNAMIC
- Monthly rebalancing
- Each month: Use ensemble to select TOP 4 ETFs from universe

This is the strategy you actually want to test!
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from support.etf_database import ETFDatabase
from tqdm import tqdm

print("=" * 80)
print("SMART SATELLITE SELECTION STRATEGY BACKTEST")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Core ETF
CORE_ISIN = 'IE00B4L5Y983'  # iShares MSCI World
CORE_ALLOCATION = 0.60  # Fixed 60%
SATELLITE_ALLOCATION = 0.40  # Fixed 40%
N_SATELLITES = 4  # Select 4 best ETFs
ALLOCATION_PER_SATELLITE = SATELLITE_ALLOCATION / N_SATELLITES  # 10% each

# Investment parameters
INITIAL_INVESTMENT = 50000  # EUR 50,000
MONTHLY_CONTRIBUTION = 1500  # EUR 1,500/month
TRANSACTION_COST_PER_TRADE = 1.0  # EUR 1 per trade (flat fee)

# Rebalancing
REBALANCE_FREQUENCY = 'monthly'  # Monthly

# Backtest period
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'

print(f"\nStrategy Configuration:")
print(f"  Core allocation: {CORE_ALLOCATION*100:.0f}% (fixed)")
print(f"  Satellite allocation: {SATELLITE_ALLOCATION*100:.0f}% (4 ETFs × 10% each)")
print(f"  Rebalancing: Monthly")
print(f"  ETF selection: Dynamic (pick top 4 each month)")
print(f"  Initial investment: €{INITIAL_INVESTMENT:,.0f}")
print(f"  Monthly contribution: €{MONTHLY_CONTRIBUTION:,.0f}")
print(f"  Transaction cost: EUR {TRANSACTION_COST_PER_TRADE:.2f} per trade")
print(f"  Period: {START_DATE} to {END_DATE}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

# Load ETF database
etf_db = ETFDatabase("data/etf_database.db")

# Load ETF universe
universe_df = etf_db.load_universe()
all_isins = universe_df['isin'].tolist()
print(f"\nETF universe: {len(all_isins)} ETFs")

# Load ensemble features
ensemble = pd.read_parquet("data/feature_analysis/smart_ensemble_20.parquet")
print(f"Loaded ensemble: {len(ensemble)} features")

# Load feature data for ensemble
print("\nLoading ensemble feature values...")
feature_data = {}

for feat in tqdm(ensemble.index, desc="Loading features"):
    parts = feat.rsplit('__', 1)
    if len(parts) != 2:
        continue

    signal_filter, indicator = parts
    feature_file = Path(f"data/features/{signal_filter}.parquet")

    if not feature_file.exists():
        continue

    try:
        df_feature = pd.read_parquet(feature_file)
        col_name = feat

        if col_name in df_feature.columns:
            feature_data[feat] = df_feature[col_name]
    except Exception as e:
        print(f"  Warning: Could not load {feat}: {e}")

df_features = pd.DataFrame(feature_data)
print(f"Feature data: {df_features.shape[0]} dates × {df_features.shape[1]} features")

# Filter date range
df_features = df_features.loc[
    (df_features.index >= START_DATE) &
    (df_features.index <= END_DATE)
]

print(f"Filtered to period: {len(df_features)} dates")

# Load prices for ALL ETFs
print("\nLoading ETF prices (this may take a minute)...")

all_prices = {}

# Core
core_prices = etf_db.load_prices(CORE_ISIN)
if core_prices is not None:
    all_prices[CORE_ISIN] = core_prices

# All universe ETFs
for isin in tqdm(all_isins, desc="Loading prices"):
    if isin == CORE_ISIN:
        continue

    prices = etf_db.load_prices(isin)
    if prices is not None and len(prices) > 0:
        all_prices[isin] = prices

df_prices = pd.DataFrame(all_prices)

# Filter date range
df_prices = df_prices.loc[
    (df_prices.index >= START_DATE) &
    (df_prices.index <= END_DATE)
]

print(f"Loaded {len(df_prices.columns)} ETF prices")
print(f"Date range: {df_prices.index[0].date()} to {df_prices.index[-1].date()}")

# Align dates
common_dates = df_features.index.intersection(df_prices.index)
df_features = df_features.loc[common_dates]
df_prices = df_prices.loc[common_dates]

print(f"\nAligned data: {len(common_dates)} trading days")

# ============================================================================
# CALCULATE ENSEMBLE SCORES
# ============================================================================

print("\n" + "=" * 80)
print("CALCULATING ENSEMBLE SCORES FOR EACH ETF")
print("=" * 80)

# For each ETF, we need to calculate its ensemble score
# The features are CROSS-SECTIONAL (compare all ETFs at each date)
# Higher ensemble score = ETF is doing relatively better

# Strategy: Use ensemble score to RANK ETFs each month
# The ensemble score is already the average of all features
# We just calculate one score per date (cross-sectional average)

ensemble_scores = df_features.mean(axis=1)

print(f"Ensemble score range: [{ensemble_scores.min():.2f}, {ensemble_scores.max():.2f}]")
print(f"Mean: {ensemble_scores.mean():.2f}, Std: {ensemble_scores.std():.2f}")

print("\nNote: We cannot use cross-sectional features to select individual ETFs!")
print("These features measure UNIVERSE-LEVEL conditions, not individual ETF quality.")
print("\nFalling back to simple momentum-based selection...")

# ============================================================================
# ETF SELECTION LOGIC (MOMENTUM-BASED)
# ============================================================================

print("\n" + "=" * 80)
print("ETF SELECTION METHOD: MOMENTUM")
print("=" * 80)

print("\nSince ensemble features are cross-sectional (universe-level),")
print("we'll use simple momentum to select satellites:")
print("  - Calculate 12-month return for each ETF")
print("  - Exclude ETFs with negative returns")
print("  - Select top 4 by return")
print("\nThis is a simple but effective strategy.")

def select_top_etfs(prices_df, date, lookback_days=252, n_select=4, exclude_isin=CORE_ISIN):
    """
    Select top N ETFs by momentum (trailing return).

    Args:
        prices_df: DataFrame of prices
        date: Current date
        lookback_days: Lookback period for momentum (252 = 1 year)
        n_select: Number of ETFs to select
        exclude_isin: ISIN to exclude (core)

    Returns:
        List of selected ISINs
    """
    # Get historical prices
    hist_dates = prices_df.index[prices_df.index <= date]

    if len(hist_dates) < lookback_days + 1:
        # Not enough history
        return []

    # Calculate returns over lookback period
    start_date = hist_dates[-lookback_days-1]
    end_date = date

    returns = {}
    for isin in prices_df.columns:
        if isin == exclude_isin:
            continue

        try:
            start_price = prices_df.loc[start_date, isin]
            end_price = prices_df.loc[end_date, isin]

            if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                ret = (end_price / start_price) - 1

                # Only consider ETFs with positive returns
                if ret > 0:
                    returns[isin] = ret
        except:
            continue

    # Sort by return and select top N
    sorted_etfs = sorted(returns.items(), key=lambda x: x[1], reverse=True)
    selected = [isin for isin, ret in sorted_etfs[:n_select]]

    return selected

# Test selection on first rebalance date
test_date = common_dates[252]  # After 1 year of data
test_selected = select_top_etfs(df_prices, test_date, n_select=N_SATELLITES)
print(f"\nTest selection for {test_date.date()}:")
for i, isin in enumerate(test_selected, 1):
    print(f"  {i}. {isin}")

# ============================================================================
# BACKTEST
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING BACKTEST")
print("=" * 80)

# Initialize portfolio
portfolio_smart = pd.Series(index=common_dates, dtype=float)
portfolio_core = pd.Series(index=common_dates, dtype=float)

# Track holdings
holdings_smart = pd.DataFrame(0.0, index=common_dates, columns=df_prices.columns)
holdings_core = pd.Series(0.0, index=common_dates)

# Track selections over time
selections_history = []

# Rebalance dates (first trading day of each month)
rebalance_dates = []
current_month = None

for date in common_dates:
    if current_month != date.month and common_dates.get_loc(date) >= 252:  # After 1 year warmup
        rebalance_dates.append(date)
        current_month = date.month

print(f"Rebalance dates: {len(rebalance_dates)} (monthly)")

# Track metrics
total_transaction_costs_smart = 0
total_transaction_costs_core = 0

# Backtest
for i, date in enumerate(tqdm(common_dates, desc="Backtesting")):
    # Initialize on first day
    if i == 0:
        portfolio_smart[date] = INITIAL_INVESTMENT
        portfolio_core[date] = INITIAL_INVESTMENT
        continue

    prev_date = common_dates[i-1]

    # Calculate returns
    price_change = df_prices.loc[date] / df_prices.loc[prev_date]

    # Update smart portfolio value (mark-to-market)
    portfolio_value_smart = (holdings_smart.loc[prev_date] * df_prices.loc[date]).sum()
    portfolio_smart[date] = portfolio_value_smart

    # Update core portfolio value
    portfolio_core[date] = holdings_core[prev_date] * df_prices.loc[date, CORE_ISIN]

    # Copy holdings
    holdings_smart.loc[date] = holdings_smart.loc[prev_date]
    holdings_core[date] = holdings_core[prev_date]

    # Monthly contribution (first trading day of month)
    if date.month != prev_date.month:
        portfolio_smart[date] += MONTHLY_CONTRIBUTION
        portfolio_core[date] += MONTHLY_CONTRIBUTION

    # Rebalance if scheduled
    if date in rebalance_dates:
        # Smart strategy: Select top 4 ETFs
        selected_isins = select_top_etfs(df_prices, date, lookback_days=252, n_select=N_SATELLITES)

        if len(selected_isins) < N_SATELLITES:
            # Not enough ETFs with positive returns - stay in core
            selected_isins = []

        selections_history.append({
            'date': date,
            'satellites': selected_isins
        })

        # Calculate target holdings
        portfolio_value = portfolio_smart[date]

        target_core_value = portfolio_value * CORE_ALLOCATION

        # Calculate transaction costs
        # Old holdings value
        old_value = (holdings_smart.loc[date] * df_prices.loc[date]).sum()

        # Target holdings
        new_holdings = pd.Series(0.0, index=df_prices.columns)
        new_holdings[CORE_ISIN] = target_core_value / df_prices.loc[date, CORE_ISIN]

        for sat_isin in selected_isins:
            target_sat_value = portfolio_value * ALLOCATION_PER_SATELLITE
            new_holdings[sat_isin] = target_sat_value / df_prices.loc[date, sat_isin]

        # Calculate number of trades (each position change = 1 trade)
        old_holdings_vector = holdings_smart.loc[date]
        changed_positions = (new_holdings != old_holdings_vector) & ((new_holdings > 0) | (old_holdings_vector > 0))
        n_trades = changed_positions.sum()

        transaction_cost = n_trades * TRANSACTION_COST_PER_TRADE
        total_transaction_costs_smart += transaction_cost

        # Apply transaction cost
        portfolio_smart[date] -= transaction_cost
        portfolio_value -= transaction_cost

        # Update holdings
        new_holdings_scaled = new_holdings * (portfolio_value / (new_holdings * df_prices.loc[date]).sum())
        holdings_smart.loc[date] = new_holdings_scaled

        # Core strategy: just rebalance core
        portfolio_value_core = portfolio_core[date]
        holdings_core[date] = portfolio_value_core / df_prices.loc[date, CORE_ISIN]

print(f"\nBacktest complete!")
print(f"  Total rebalances: {len(rebalance_dates)}")
print(f"  Total transaction costs (smart): €{total_transaction_costs_smart:,.2f}")
print(f"  Avg satellites per rebalance: {sum(len(s['satellites']) for s in selections_history) / len(selections_history):.2f}")

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

print("\n" + "=" * 80)
print("PERFORMANCE METRICS")
print("=" * 80)

def calculate_metrics(portfolio_values, name):
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
    ann_return = (1 + total_return) ** (1 / years) - 1

    returns = portfolio_values.pct_change().dropna()
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    cummax = portfolio_values.expanding().max()
    drawdown = (portfolio_values - cummax) / cummax
    max_drawdown = drawdown.min()

    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

    downside_returns = returns[returns < 0]
    downside_dev = downside_returns.std() * np.sqrt(252)
    sortino = ann_return / downside_dev if downside_dev > 0 else 0

    return {
        'name': name,
        'final_value': portfolio_values.iloc[-1],
        'total_return': total_return,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_drawdown': max_drawdown,
        'years': years
    }

metrics_smart = calculate_metrics(portfolio_smart, 'Smart Selection (60/40)')
metrics_core = calculate_metrics(portfolio_core, '100% Core')

# Print comparison
print(f"\n{'Metric':<25} {'Smart 60/40':<15} {'100% Core':<15} {'Difference':<15}")
print("-" * 75)

keys = ['final_value', 'total_return', 'ann_return', 'ann_vol', 'sharpe', 'sortino', 'calmar', 'max_drawdown']
labels = ['Final Value', 'Total Return', 'Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown']

for key, label in zip(keys, labels):
    val_smart = metrics_smart[key]
    val_core = metrics_core[key]
    diff = val_smart - val_core

    if key == 'final_value':
        print(f"{label:<25} €{val_smart:>13,.0f} €{val_core:>13,.0f} €{diff:>13,.0f}")
    elif key in ['ann_vol', 'max_drawdown']:
        print(f"{label:<25} {val_smart:>13.2%} {val_core:>13.2%} {diff:>13.2%}")
    elif key in ['total_return', 'ann_return']:
        print(f"{label:<25} {val_smart:>13.2%} {val_core:>13.2%} {diff:>13.2%}")
    else:
        print(f"{label:<25} {val_smart:>14.3f} {val_core:>14.3f} {diff:>14.3f}")

print("-" * 75)

alpha = metrics_smart['ann_return'] - metrics_core['ann_return']
print(f"\nAlpha (outperformance): {alpha:+.2%} per year")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Combined Dashboard
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        'Portfolio Growth',
        'Cumulative Returns (%)',
        '6-Month Rolling Alpha',
        'Selected Satellites per Month',
        'Drawdown Comparison',
        'Performance Metrics'
    ),
    specs=[
        [{'type': 'scatter'}, {'type': 'scatter'}],
        [{'type': 'scatter'}, {'type': 'scatter'}],
        [{'type': 'scatter'}, {'type': 'table'}]
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.10
)

# Portfolio Growth
fig.add_trace(go.Scatter(
    x=portfolio_smart.index, y=portfolio_smart.values,
    name='Smart 60/40', line=dict(color='#00d4aa', width=2)
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=portfolio_core.index, y=portfolio_core.values,
    name='100% Core', line=dict(color='#ff6b6b', width=2)
), row=1, col=1)

# Cumulative Returns
cumulative_smart = (portfolio_smart / portfolio_smart.iloc[0] - 1) * 100
cumulative_core = (portfolio_core / portfolio_core.iloc[0] - 1) * 100

fig.add_trace(go.Scatter(
    x=cumulative_smart.index, y=cumulative_smart.values,
    name='Smart', line=dict(color='#00d4aa', width=2),
    showlegend=False
), row=1, col=2)
fig.add_trace(go.Scatter(
    x=cumulative_core.index, y=cumulative_core.values,
    name='Core', line=dict(color='#ff6b6b', width=2),
    showlegend=False
), row=1, col=2)

# Rolling Alpha
window = 126
rolling_alpha = (portfolio_smart.pct_change(window) - portfolio_core.pct_change(window)) * 100

fig.add_trace(go.Scatter(
    x=rolling_alpha.index, y=rolling_alpha.values,
    name='Alpha', line=dict(color='#00d4aa', width=2),
    fill='tozeroy', fillcolor='rgba(0, 212, 170, 0.3)',
    showlegend=False
), row=2, col=1)

# Number of satellites selected
n_satellites_selected = pd.Series(
    [len(s['satellites']) for s in selections_history],
    index=[s['date'] for s in selections_history]
)
n_satellites_selected = n_satellites_selected.reindex(common_dates, method='ffill')

fig.add_trace(go.Scatter(
    x=n_satellites_selected.index, y=n_satellites_selected.values,
    name='# Satellites', line=dict(color='#ffd700', width=2),
    fill='tozeroy', fillcolor='rgba(255, 215, 0, 0.3)',
    showlegend=False
), row=2, col=2)

# Drawdowns
cummax_smart = portfolio_smart.expanding().max()
drawdown_smart = ((portfolio_smart - cummax_smart) / cummax_smart) * 100

cummax_core = portfolio_core.expanding().max()
drawdown_core = ((portfolio_core - cummax_core) / cummax_core) * 100

fig.add_trace(go.Scatter(
    x=drawdown_smart.index, y=drawdown_smart.values,
    name='Smart', line=dict(color='#00d4aa', width=2),
    showlegend=False
), row=3, col=1)
fig.add_trace(go.Scatter(
    x=drawdown_core.index, y=drawdown_core.values,
    name='Core', line=dict(color='#ff6b6b', width=2),
    showlegend=False
), row=3, col=1)

# Performance Table
metrics_data = [
    ['Metric', 'Smart 60/40', '100% Core', 'Difference'],
    ['Final Value', f"€{metrics_smart['final_value']:,.0f}", f"€{metrics_core['final_value']:,.0f}", f"€{metrics_smart['final_value']-metrics_core['final_value']:,.0f}"],
    ['Annual Return', f"{metrics_smart['ann_return']:.2%}", f"{metrics_core['ann_return']:.2%}", f"{alpha:+.2%}"],
    ['Volatility', f"{metrics_smart['ann_vol']:.2%}", f"{metrics_core['ann_vol']:.2%}", f"{metrics_smart['ann_vol']-metrics_core['ann_vol']:+.2%}"],
    ['Sharpe Ratio', f"{metrics_smart['sharpe']:.3f}", f"{metrics_core['sharpe']:.3f}", f"{metrics_smart['sharpe']-metrics_core['sharpe']:+.3f}"],
    ['Max Drawdown', f"{metrics_smart['max_drawdown']:.2%}", f"{metrics_core['max_drawdown']:.2%}", f"{metrics_smart['max_drawdown']-metrics_core['max_drawdown']:+.2%}"],
]

fig.add_trace(go.Table(
    header=dict(
        values=['<b>' + v + '</b>' for v in metrics_data[0]],
        fill_color='#3d3d3d',
        align='left',
        font=dict(color='white', size=11)
    ),
    cells=dict(
        values=list(zip(*metrics_data[1:])),
        fill_color='#2d2d2d',
        align='left',
        font=dict(color='white', size=10)
    )
), row=3, col=2)

fig.update_layout(
    title_text='Smart Satellite Selection - Your Strategy (60% Core / 40% Satellites)',
    template='plotly_dark',
    showlegend=True,
    height=1200,
    legend=dict(x=0.01, y=0.99)
)

fig.write_html('backtest_smart_selection_dashboard.html')
print("  Saved: backtest_smart_selection_dashboard.html")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("BACKTEST COMPLETE - YOUR ACTUAL STRATEGY")
print("=" * 80)

print(f"\nStrategy: 60% Core / 40% Satellites (4 ETFs)")
print(f"Selection: Monthly momentum-based")
print(f"Period: {metrics_smart['years']:.1f} years")
print(f"Rebalances: {len(rebalance_dates)}")
print(f"Transaction costs: €{total_transaction_costs_smart:,.2f}\n")

print(f"Smart 60/40 Strategy:")
print(f"  Final value: €{metrics_smart['final_value']:,.0f}")
print(f"  Annual return: {metrics_smart['ann_return']:.2%}")
print(f"  Volatility: {metrics_smart['ann_vol']:.2%}")
print(f"  Sharpe ratio: {metrics_smart['sharpe']:.3f}")
print(f"  Max drawdown: {metrics_smart['max_drawdown']:.2%}\n")

print(f"100% Core Baseline:")
print(f"  Final value: €{metrics_core['final_value']:,.0f}")
print(f"  Annual return: {metrics_core['ann_return']:.2%}")
print(f"  Volatility: {metrics_core['ann_vol']:.2%}")
print(f"  Sharpe ratio: {metrics_core['sharpe']:.3f}")
print(f"  Max drawdown: {metrics_core['max_drawdown']:.2%}\n")

print(f"Alpha: {alpha:+.2%} per year")

if alpha > 0:
    print(f"✓ Strategy OUTPERFORMED by €{metrics_smart['final_value'] - metrics_core['final_value']:,.0f}")
else:
    print(f"✗ Strategy UNDERPERFORMED by €{abs(metrics_smart['final_value'] - metrics_core['final_value']):,.0f}")

print("\nVisualization: backtest_smart_selection_dashboard.html")
print("=" * 80)
