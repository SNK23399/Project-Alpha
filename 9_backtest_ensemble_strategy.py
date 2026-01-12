"""
Backtest Ensemble Strategy

This notebook backtests the 20-feature ensemble for ETF portfolio allocation.

Strategy:
1. Calculate ensemble score (average of 20 features)
2. Map score to satellite allocation (0-75%)
3. Rebalance quarterly (63 days)
4. Compare to 100% core baseline

Output: Interactive Plotly visualizations + performance metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from support.etf_database import ETFDatabase

print("=" * 80)
print("ENSEMBLE STRATEGY BACKTEST")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Core ETF (benchmark)
CORE_ISIN = 'IE00B4L5Y983'  # iShares MSCI World

# Selected satellite ETFs (from Chapter 2)
SATELLITE_ISINS = {
    'North America': 'IE00B3YCGJ38',    # Invesco S&P 500
    'Europe': 'IE00B53QG562',            # iShares MSCI EMU
    'Emerging Markets': 'IE00B4L5YC18',  # iShares MSCI EM
    'Japan': 'IE00B4L5YX21',             # iShares MSCI Japan
    'Pacific ex-Japan': 'IE00B52MJY50',  # iShares MSCI Pacific ex-Japan
}

# Investment parameters
INITIAL_INVESTMENT = 50000  # EUR 50,000
MONTHLY_CONTRIBUTION = 1500  # EUR 1,500/month
TRANSACTION_COST = 0.005  # 0.5% per trade

# Strategy parameters
REBALANCE_DAYS = 63  # Quarterly
MAX_SATELLITE_ALLOCATION = 0.75  # Maximum 75% in satellites

# Backtest period
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'

print(f"\nConfiguration:")
print(f"  Core ETF: {CORE_ISIN}")
print(f"  Satellites: {len(SATELLITE_ISINS)}")
print(f"  Initial investment: €{INITIAL_INVESTMENT:,.0f}")
print(f"  Monthly contribution: €{MONTHLY_CONTRIBUTION:,.0f}")
print(f"  Rebalancing: Every {REBALANCE_DAYS} days")
print(f"  Transaction cost: {TRANSACTION_COST*100:.2f}% per trade")
print(f"  Period: {START_DATE} to {END_DATE}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

# Load ETF database
etf_db = ETFDatabase("data/etf_database.db")

# Load ensemble features
ensemble = pd.read_parquet("data/feature_analysis/smart_ensemble_20.parquet")
print(f"\nLoaded ensemble: {len(ensemble)} features")

# Load feature data for each ensemble feature
print("\nLoading feature values...")
feature_data = {}
feature_files_found = 0

for feat in ensemble.index:
    # Parse feature name: signal_filter__indicator
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
            feature_files_found += 1
    except Exception as e:
        print(f"  Warning: Could not load {feat}: {e}")

print(f"Successfully loaded {feature_files_found} / {len(ensemble)} features")

# Convert to DataFrame
df_features = pd.DataFrame(feature_data)

# Filter date range
df_features = df_features.loc[
    (df_features.index >= START_DATE) &
    (df_features.index <= END_DATE)
]

print(f"Feature data: {df_features.shape[0]} dates × {df_features.shape[1]} features")
print(f"Date range: {df_features.index[0].date()} to {df_features.index[-1].date()}")

# Load prices for core and satellites
print("\nLoading ETF prices...")

# Core
core_prices = etf_db.load_prices(CORE_ISIN)
core_prices = core_prices.loc[
    (core_prices.index >= START_DATE) &
    (core_prices.index <= END_DATE)
]
print(f"  Core ({CORE_ISIN}): {len(core_prices)} prices")

# Satellites
satellite_prices = {}
for name, isin in SATELLITE_ISINS.items():
    prices = etf_db.load_prices(isin)
    if prices is not None:
        prices = prices.loc[
            (prices.index >= START_DATE) &
            (prices.index <= END_DATE)
        ]
        satellite_prices[name] = prices
        print(f"  {name} ({isin}): {len(prices)} prices")

# Combine all prices
all_prices = pd.DataFrame({
    'Core': core_prices,
    **satellite_prices
})

# Align dates (trading days only)
common_dates = df_features.index.intersection(all_prices.index)
df_features = df_features.loc[common_dates]
all_prices = all_prices.loc[common_dates]

print(f"\nAligned data: {len(common_dates)} trading days")

# ============================================================================
# CALCULATE ENSEMBLE SCORES
# ============================================================================

print("\n" + "=" * 80)
print("CALCULATING ENSEMBLE SCORES")
print("=" * 80)

# Equal-weighted ensemble (average of all features)
ensemble_scores = df_features.mean(axis=1)

# Normalize to z-scores
ensemble_mean = ensemble_scores.mean()
ensemble_std = ensemble_scores.std()
ensemble_zscores = (ensemble_scores - ensemble_mean) / ensemble_std

print(f"\nEnsemble statistics:")
print(f"  Mean: {ensemble_mean:.6f}")
print(f"  Std: {ensemble_std:.6f}")
print(f"  Min z-score: {ensemble_zscores.min():.2f}")
print(f"  Max z-score: {ensemble_zscores.max():.2f}")

# Map z-scores to satellite allocation
def zscore_to_allocation(z):
    """
    Convert ensemble z-score to satellite allocation.

    Higher z-score = universe doing relatively better = can use satellites
    Lower z-score = universe struggling = stay defensive
    """
    if z > 2:
        return 0.75  # Very bullish
    elif z > 1:
        return 0.60  # Moderately bullish
    elif z > 0:
        return 0.40  # Slightly bullish
    elif z > -1:
        return 0.25  # Neutral
    else:
        return 0.00  # Defensive

satellite_allocations = ensemble_zscores.apply(zscore_to_allocation)

print(f"\nSatellite allocation distribution:")
for alloc in [0.00, 0.25, 0.40, 0.60, 0.75]:
    count = (satellite_allocations == alloc).sum()
    pct = count / len(satellite_allocations) * 100
    print(f"  {alloc*100:5.0f}%: {count:4d} days ({pct:5.1f}%)")

# ============================================================================
# BACKTEST STRATEGY
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING BACKTEST")
print("=" * 80)

# Initialize portfolios
portfolio_ensemble = pd.Series(index=common_dates, dtype=float)
portfolio_core = pd.Series(index=common_dates, dtype=float)

# Track holdings (number of units)
holdings_ensemble = pd.DataFrame(
    0.0,
    index=common_dates,
    columns=['Core'] + list(satellite_prices.keys())
)

holdings_core = pd.Series(0.0, index=common_dates)

# Track allocations over time
allocations_history = pd.DataFrame(index=common_dates, columns=['satellite_pct'])

# Rebalance dates
rebalance_dates = [common_dates[0]]  # Start with first date
current_date = common_dates[0]

while current_date < common_dates[-1]:
    # Next rebalance is REBALANCE_DAYS later
    next_date = current_date + timedelta(days=REBALANCE_DAYS)

    # Find nearest trading day
    future_dates = common_dates[common_dates > current_date]
    if len(future_dates) > 0:
        nearest = future_dates[future_dates >= next_date]
        if len(nearest) > 0:
            rebalance_dates.append(nearest[0])
            current_date = nearest[0]
        else:
            break
    else:
        break

print(f"Rebalance dates: {len(rebalance_dates)}")

# Track metrics
n_trades = 0
total_transaction_costs = 0

# Backtest
for i, date in enumerate(common_dates):
    # Get previous value
    if i == 0:
        # Initial investment
        portfolio_ensemble[date] = INITIAL_INVESTMENT
        portfolio_core[date] = INITIAL_INVESTMENT
    else:
        prev_date = common_dates[i-1]

        # Calculate returns
        returns = all_prices.loc[date] / all_prices.loc[prev_date] - 1

        # Update ensemble portfolio value
        portfolio_value = (holdings_ensemble.loc[prev_date] * all_prices.loc[date]).sum()
        portfolio_ensemble[date] = portfolio_value

        # Update core portfolio value
        portfolio_core[date] = holdings_core[prev_date] * all_prices.loc[date, 'Core']

        # Copy holdings from previous day
        holdings_ensemble.loc[date] = holdings_ensemble.loc[prev_date]
        holdings_core[date] = holdings_core[prev_date]

    # Monthly contribution (first trading day of month)
    if i > 0:
        prev_date = common_dates[i-1]
        if date.month != prev_date.month:
            # Add contribution
            portfolio_ensemble[date] += MONTHLY_CONTRIBUTION
            portfolio_core[date] += MONTHLY_CONTRIBUTION

    # Rebalance if needed
    if date in rebalance_dates:
        # Ensemble strategy
        target_satellite = satellite_allocations[date]
        target_core = 1 - target_satellite

        allocations_history.loc[date, 'satellite_pct'] = target_satellite

        # Calculate target values
        portfolio_value = portfolio_ensemble[date]

        # Target allocation per satellite (equal weight)
        n_satellites = len(satellite_prices)
        target_per_satellite = (portfolio_value * target_satellite) / n_satellites
        target_core_value = portfolio_value * target_core

        # Calculate transaction cost
        old_holdings_value = (holdings_ensemble.loc[date] * all_prices.loc[date]).sum()
        trades_needed = abs(target_core_value - holdings_ensemble.loc[date, 'Core'] * all_prices.loc[date, 'Core'])

        for sat_name in satellite_prices.keys():
            trades_needed += abs(target_per_satellite - holdings_ensemble.loc[date, sat_name] * all_prices.loc[date, sat_name])

        transaction_cost = trades_needed * TRANSACTION_COST
        total_transaction_costs += transaction_cost
        portfolio_ensemble[date] -= transaction_cost
        portfolio_value -= transaction_cost

        # Update holdings
        holdings_ensemble.loc[date, 'Core'] = target_core_value / all_prices.loc[date, 'Core']
        for sat_name in satellite_prices.keys():
            holdings_ensemble.loc[date, sat_name] = target_per_satellite / all_prices.loc[date, sat_name]

        n_trades += 1

        # Core-only strategy (rebalance to add monthly contribution)
        portfolio_value_core = portfolio_core[date]
        holdings_core[date] = portfolio_value_core / all_prices.loc[date, 'Core']

print(f"\nBacktest complete!")
print(f"  Total rebalances: {n_trades}")
print(f"  Total transaction costs: €{total_transaction_costs:,.2f}")
print(f"  Avg cost per rebalance: €{total_transaction_costs/n_trades:,.2f}")

# Forward-fill allocations
allocations_history['satellite_pct'] = allocations_history['satellite_pct'].ffill()

# ============================================================================
# CALCULATE PERFORMANCE METRICS
# ============================================================================

print("\n" + "=" * 80)
print("PERFORMANCE METRICS")
print("=" * 80)

def calculate_metrics(portfolio_values, name):
    """Calculate performance metrics for a portfolio."""
    # Total return
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1

    # Time period (in years)
    years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25

    # Annualized return
    ann_return = (1 + total_return) ** (1 / years) - 1

    # Daily returns
    returns = portfolio_values.pct_change().dropna()

    # Volatility
    ann_vol = returns.std() * np.sqrt(252)

    # Sharpe ratio (assuming risk-free rate = 0)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Maximum drawdown
    cummax = portfolio_values.expanding().max()
    drawdown = (portfolio_values - cummax) / cummax
    max_drawdown = drawdown.min()

    # Calmar ratio
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Sortino ratio (downside deviation)
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

# Calculate metrics for both strategies
metrics_ensemble = calculate_metrics(portfolio_ensemble, 'Ensemble Strategy')
metrics_core = calculate_metrics(portfolio_core, '100% Core')

# Print comparison
print(f"\n{'Metric':<25} {'Ensemble':<15} {'100% Core':<15} {'Difference':<15}")
print("-" * 75)

keys = ['final_value', 'total_return', 'ann_return', 'ann_vol', 'sharpe', 'sortino', 'calmar', 'max_drawdown']
labels = ['Final Value', 'Total Return', 'Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown']

for key, label in zip(keys, labels):
    val_ens = metrics_ensemble[key]
    val_core = metrics_core[key]
    diff = val_ens - val_core

    if key == 'final_value':
        print(f"{label:<25} €{val_ens:>13,.0f} €{val_core:>13,.0f} €{diff:>13,.0f}")
    elif key in ['ann_vol', 'max_drawdown']:
        print(f"{label:<25} {val_ens:>13.2%} {val_core:>13.2%} {diff:>13.2%}")
    elif key in ['total_return', 'ann_return']:
        print(f"{label:<25} {val_ens:>13.2%} {val_core:>13.2%} {diff:>13.2%}")
    else:
        print(f"{label:<25} {val_ens:>14.3f} {val_core:>14.3f} {diff:>14.3f}")

print("-" * 75)

# Alpha (outperformance)
alpha = metrics_ensemble['ann_return'] - metrics_core['ann_return']
print(f"\nAlpha (outperformance): {alpha:+.2%} per year")

# ============================================================================
# VISUALIZATIONS (Dark Mode Plotly)
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Dark mode template
template = {
    'layout': {
        'paper_bgcolor': '#1e1e1e',
        'plot_bgcolor': '#2d2d2d',
        'font': {'color': '#e0e0e0'},
        'xaxis': {'gridcolor': '#3d3d3d', 'zerolinecolor': '#3d3d3d'},
        'yaxis': {'gridcolor': '#3d3d3d', 'zerolinecolor': '#3d3d3d'},
    }
}

# Figure 1: Portfolio Growth
print("  Creating portfolio growth chart...")

fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=portfolio_ensemble.index,
    y=portfolio_ensemble.values,
    name='Ensemble Strategy',
    line=dict(color='#00d4aa', width=2),
    hovertemplate='%{y:€,.0f}<extra></extra>'
))

fig1.add_trace(go.Scatter(
    x=portfolio_core.index,
    y=portfolio_core.values,
    name='100% Core',
    line=dict(color='#ff6b6b', width=2),
    hovertemplate='%{y:€,.0f}<extra></extra>'
))

fig1.update_layout(
    title='Portfolio Growth Over Time',
    xaxis_title='Date',
    yaxis_title='Portfolio Value (€)',
    template='plotly_dark',
    hovermode='x unified',
    legend=dict(x=0.01, y=0.99)
)

fig1.write_html('backtest_results_portfolio_growth.html')
print("    Saved: backtest_results_portfolio_growth.html")

# Figure 2: Cumulative Returns
print("  Creating cumulative returns chart...")

cumulative_ensemble = (portfolio_ensemble / portfolio_ensemble.iloc[0] - 1) * 100
cumulative_core = (portfolio_core / portfolio_core.iloc[0] - 1) * 100

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=cumulative_ensemble.index,
    y=cumulative_ensemble.values,
    name='Ensemble Strategy',
    line=dict(color='#00d4aa', width=2),
    hovertemplate='%{y:+.1f}%<extra></extra>'
))

fig2.add_trace(go.Scatter(
    x=cumulative_core.index,
    y=cumulative_core.values,
    name='100% Core',
    line=dict(color='#ff6b6b', width=2),
    hovertemplate='%{y:+.1f}%<extra></extra>'
))

fig2.update_layout(
    title='Cumulative Returns',
    xaxis_title='Date',
    yaxis_title='Cumulative Return (%)',
    template='plotly_dark',
    hovermode='x unified',
    legend=dict(x=0.01, y=0.99)
)

fig2.write_html('backtest_results_cumulative_returns.html')
print("    Saved: backtest_results_cumulative_returns.html")

# Figure 3: Rolling Alpha (6-month)
print("  Creating rolling alpha chart...")

# Calculate 6-month rolling returns
window = 126  # ~6 months
rolling_ret_ensemble = portfolio_ensemble.pct_change(window)
rolling_ret_core = portfolio_core.pct_change(window)
rolling_alpha = (rolling_ret_ensemble - rolling_ret_core) * 100

fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=rolling_alpha.index,
    y=rolling_alpha.values,
    name='6-Month Rolling Alpha',
    line=dict(color='#00d4aa', width=2),
    fill='tozeroy',
    fillcolor='rgba(0, 212, 170, 0.3)',
    hovertemplate='%{y:+.2f}%<extra></extra>'
))

fig3.add_hline(
    y=0,
    line_dash='dash',
    line_color='gray',
    annotation_text='Zero Alpha',
    annotation_position='right'
)

fig3.update_layout(
    title='6-Month Rolling Alpha vs 100% Core',
    xaxis_title='Date',
    yaxis_title='Rolling Alpha (%)',
    template='plotly_dark',
    hovermode='x unified'
)

fig3.write_html('backtest_results_rolling_alpha.html')
print("    Saved: backtest_results_rolling_alpha.html")

# Figure 4: Satellite Allocation Over Time
print("  Creating allocation chart...")

fig4 = go.Figure()

fig4.add_trace(go.Scatter(
    x=allocations_history.index,
    y=allocations_history['satellite_pct'].values * 100,
    name='Satellite Allocation',
    line=dict(color='#ffd700', width=2),
    fill='tozeroy',
    fillcolor='rgba(255, 215, 0, 0.3)',
    hovertemplate='%{y:.0f}%<extra></extra>'
))

# Add regime annotations
for alloc_pct in [0, 25, 40, 60, 75]:
    fig4.add_hline(
        y=alloc_pct,
        line_dash='dot',
        line_color='gray',
        opacity=0.5
    )

fig4.update_layout(
    title='Satellite Allocation Over Time',
    xaxis_title='Date',
    yaxis_title='Satellite Allocation (%)',
    template='plotly_dark',
    hovermode='x unified'
)

fig4.write_html('backtest_results_allocation.html')
print("    Saved: backtest_results_allocation.html")

# Figure 5: Drawdown Comparison
print("  Creating drawdown comparison...")

# Calculate drawdowns
cummax_ensemble = portfolio_ensemble.expanding().max()
drawdown_ensemble = ((portfolio_ensemble - cummax_ensemble) / cummax_ensemble) * 100

cummax_core = portfolio_core.expanding().max()
drawdown_core = ((portfolio_core - cummax_core) / cummax_core) * 100

fig5 = go.Figure()

fig5.add_trace(go.Scatter(
    x=drawdown_ensemble.index,
    y=drawdown_ensemble.values,
    name='Ensemble Strategy',
    line=dict(color='#00d4aa', width=2),
    fill='tozeroy',
    fillcolor='rgba(0, 212, 170, 0.3)',
    hovertemplate='%{y:.2f}%<extra></extra>'
))

fig5.add_trace(go.Scatter(
    x=drawdown_core.index,
    y=drawdown_core.values,
    name='100% Core',
    line=dict(color='#ff6b6b', width=2),
    fill='tozeroy',
    fillcolor='rgba(255, 107, 107, 0.3)',
    hovertemplate='%{y:.2f}%<extra></extra>'
))

fig5.update_layout(
    title='Drawdown Over Time',
    xaxis_title='Date',
    yaxis_title='Drawdown (%)',
    template='plotly_dark',
    hovermode='x unified',
    legend=dict(x=0.01, y=0.01)
)

fig5.write_html('backtest_results_drawdown.html')
print("    Saved: backtest_results_drawdown.html")

# Figure 6: Combined Dashboard
print("  Creating combined dashboard...")

fig6 = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        'Portfolio Growth',
        'Cumulative Returns (%)',
        '6-Month Rolling Alpha',
        'Satellite Allocation',
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
fig6.add_trace(go.Scatter(
    x=portfolio_ensemble.index, y=portfolio_ensemble.values,
    name='Ensemble', line=dict(color='#00d4aa', width=1.5)
), row=1, col=1)
fig6.add_trace(go.Scatter(
    x=portfolio_core.index, y=portfolio_core.values,
    name='Core', line=dict(color='#ff6b6b', width=1.5)
), row=1, col=1)

# Cumulative Returns
fig6.add_trace(go.Scatter(
    x=cumulative_ensemble.index, y=cumulative_ensemble.values,
    name='Ensemble', line=dict(color='#00d4aa', width=1.5),
    showlegend=False
), row=1, col=2)
fig6.add_trace(go.Scatter(
    x=cumulative_core.index, y=cumulative_core.values,
    name='Core', line=dict(color='#ff6b6b', width=1.5),
    showlegend=False
), row=1, col=2)

# Rolling Alpha
fig6.add_trace(go.Scatter(
    x=rolling_alpha.index, y=rolling_alpha.values,
    name='Alpha', line=dict(color='#00d4aa', width=1.5),
    fill='tozeroy', fillcolor='rgba(0, 212, 170, 0.2)',
    showlegend=False
), row=2, col=1)

# Satellite Allocation
fig6.add_trace(go.Scatter(
    x=allocations_history.index, y=allocations_history['satellite_pct'].values * 100,
    name='Allocation', line=dict(color='#ffd700', width=1.5),
    fill='tozeroy', fillcolor='rgba(255, 215, 0, 0.2)',
    showlegend=False
), row=2, col=2)

# Drawdown
fig6.add_trace(go.Scatter(
    x=drawdown_ensemble.index, y=drawdown_ensemble.values,
    name='Ensemble', line=dict(color='#00d4aa', width=1.5),
    showlegend=False
), row=3, col=1)
fig6.add_trace(go.Scatter(
    x=drawdown_core.index, y=drawdown_core.values,
    name='Core', line=dict(color='#ff6b6b', width=1.5),
    showlegend=False
), row=3, col=1)

# Performance Table
metrics_data = [
    ['Metric', 'Ensemble', 'Core', 'Difference'],
    ['Final Value', f"€{metrics_ensemble['final_value']:,.0f}", f"€{metrics_core['final_value']:,.0f}", f"€{metrics_ensemble['final_value']-metrics_core['final_value']:,.0f}"],
    ['Annual Return', f"{metrics_ensemble['ann_return']:.2%}", f"{metrics_core['ann_return']:.2%}", f"{alpha:+.2%}"],
    ['Volatility', f"{metrics_ensemble['ann_vol']:.2%}", f"{metrics_core['ann_vol']:.2%}", f"{metrics_ensemble['ann_vol']-metrics_core['ann_vol']:+.2%}"],
    ['Sharpe Ratio', f"{metrics_ensemble['sharpe']:.3f}", f"{metrics_core['sharpe']:.3f}", f"{metrics_ensemble['sharpe']-metrics_core['sharpe']:+.3f}"],
    ['Max Drawdown', f"{metrics_ensemble['max_drawdown']:.2%}", f"{metrics_core['max_drawdown']:.2%}", f"{metrics_ensemble['max_drawdown']-metrics_core['max_drawdown']:+.2%}"],
]

fig6.add_trace(go.Table(
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

fig6.update_layout(
    title_text='Ensemble Strategy - Complete Dashboard',
    template='plotly_dark',
    showlegend=True,
    height=1200,
    legend=dict(x=0.01, y=0.99)
)

fig6.write_html('backtest_results_dashboard.html')
print("    Saved: backtest_results_dashboard.html")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("BACKTEST COMPLETE")
print("=" * 80)

print(f"\nStrategy Performance ({START_DATE} to {END_DATE}):")
print(f"  Period: {metrics_ensemble['years']:.1f} years")
print(f"  Rebalances: {n_trades}")
print(f"  Transaction costs: €{total_transaction_costs:,.2f}\n")

print(f"Ensemble Strategy:")
print(f"  Final value: €{metrics_ensemble['final_value']:,.0f}")
print(f"  Annual return: {metrics_ensemble['ann_return']:.2%}")
print(f"  Volatility: {metrics_ensemble['ann_vol']:.2%}")
print(f"  Sharpe ratio: {metrics_ensemble['sharpe']:.3f}")
print(f"  Max drawdown: {metrics_ensemble['max_drawdown']:.2%}\n")

print(f"100% Core Baseline:")
print(f"  Final value: €{metrics_core['final_value']:,.0f}")
print(f"  Annual return: {metrics_core['ann_return']:.2%}")
print(f"  Volatility: {metrics_core['ann_vol']:.2%}")
print(f"  Sharpe ratio: {metrics_core['sharpe']:.3f}")
print(f"  Max drawdown: {metrics_core['max_drawdown']:.2%}\n")

print(f"Alpha: {alpha:+.2%} per year")
print(f"Outperformance: €{metrics_ensemble['final_value'] - metrics_core['final_value']:,.0f}\n")

print("Visualizations saved:")
print("  - backtest_results_portfolio_growth.html")
print("  - backtest_results_cumulative_returns.html")
print("  - backtest_results_rolling_alpha.html")
print("  - backtest_results_allocation.html")
print("  - backtest_results_drawdown.html")
print("  - backtest_results_dashboard.html (ALL IN ONE)\n")

print("Open the HTML files in your browser to view interactive charts!")
print("=" * 80)
