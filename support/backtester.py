"""
Backtester Module - Portfolio Backtesting Engine

Provides a flexible backtesting framework for Core-Satellite portfolios with:
- Dynamic regional weight allocation (based on ACWI or custom weights)
- Configurable rebalancing strategies and triggers
- Comprehensive performance metrics
- Support for contributions and monthly deposits

Usage:
    from backtester import Backtester, PortfolioMetrics

    bt = Backtester(
        prices=price_dataframe,
        core_isin='IE00B6R52259',
        satellite_isins={'North America': 'IE00B3YCGJ38', ...},
        core_weight=0.60,
        regional_weights=df_weights  # Optional
    )

    result = bt.run(start_date='2020-01-01', end_date='2024-12-31')
    metrics = PortfolioMetrics.calculate(result['portfolio'])
"""

from typing import Dict, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class BacktestResult:
    """Container for backtest results."""
    portfolio: pd.Series  # Daily portfolio values
    rebalance_dates: List[pd.Timestamp]  # When rebalances occurred
    weights_history: pd.DataFrame  # Weight allocation over time
    holdings_history: pd.DataFrame  # Units held over time

    def __repr__(self):
        return (f"BacktestResult(dates={len(self.portfolio)}, "
                f"rebalances={len(self.rebalance_dates)})")


class PortfolioMetrics:
    """Calculate portfolio performance metrics."""

    @staticmethod
    def calculate(portfolio: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Args:
            portfolio: Series of daily portfolio values

        Returns:
            Dictionary with performance metrics
        """
        if len(portfolio) < 2:
            return {
                'total_return': 0.0,
                'ann_return': 0.0,
                'ann_vol': 0.0,
                'sharpe': 0.0,
                'sortino': 0.0,
                'calmar': 0.0,
                'max_drawdown': 0.0,
            }

        returns = portfolio.pct_change().dropna()

        # Basic return metrics
        total_return = (portfolio.iloc[-1] / portfolio.iloc[0]) - 1
        n_days = len(returns)
        ann_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        ann_vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

        # Sharpe ratio (assumes risk-free rate = 0)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Sortino ratio - only penalizes downside volatility
        negative_rets = returns[returns < 0]
        downside_vol = negative_rets.std() * np.sqrt(252) if len(negative_rets) > 0 else 0
        sortino = ann_return / downside_vol if downside_vol > 0 else 0

        # Max drawdown
        rolling_max = portfolio.cummax()
        drawdown = (portfolio - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # Calmar ratio (ann return / abs(max drawdown))
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        return {
            'total_return': total_return,
            'ann_return': ann_return,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'max_drawdown': max_dd,
        }

    @staticmethod
    def print_summary(metrics: Dict[str, float]):
        """Print formatted metrics summary."""
        print("Portfolio Performance Metrics")
        print("=" * 50)
        print(f"  Total Return:       {metrics['total_return']*100:>7.2f}%")
        print(f"  Annualized Return:  {metrics['ann_return']*100:>7.2f}%")
        print(f"  Annualized Vol:     {metrics['ann_vol']*100:>7.2f}%")
        print(f"  Sharpe Ratio:       {metrics['sharpe']:>7.3f}")
        print(f"  Sortino Ratio:      {metrics['sortino']:>7.3f}")
        print(f"  Calmar Ratio:       {metrics['calmar']:>7.3f}")
        print(f"  Max Drawdown:       {metrics['max_drawdown']*100:>7.2f}%")
        print("=" * 50)


class Backtester:
    """
    Core-Satellite portfolio backtesting engine.

    Supports:
    - Dynamic regional weight allocation
    - Custom rebalancing triggers
    - Multiple weight calculation methods
    - Monthly contributions
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        core_isin: str,
        satellite_isins: Dict[str, str],
        core_weight: float = 0.60,
        regional_weights: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize backtester.

        Args:
            prices: DataFrame with columns = ISINs, index = dates
            core_isin: ISIN of core holding
            satellite_isins: Dict mapping region -> ISIN
            core_weight: Allocation to core (e.g., 0.60 = 60%)
            regional_weights: Optional DataFrame with regional weight history
                             (index=dates, columns=region names)
        """
        self.prices = prices
        self.core_isin = core_isin
        self.satellite_isins = satellite_isins
        self.core_weight = core_weight
        self.satellite_weight = 1 - core_weight
        self.regional_weights = regional_weights
        self.regions = list(satellite_isins.keys())

        # Validate ISINs
        all_isins = [core_isin] + list(satellite_isins.values())
        missing = [isin for isin in all_isins if isin not in prices.columns]
        if missing:
            raise ValueError(f"Missing price data for ISINs: {missing}")

    def get_weight_date_for(self, dt: pd.Timestamp) -> pd.Timestamp:
        """Get the most recent weight snapshot date <= dt."""
        if self.regional_weights is None or self.regional_weights.empty:
            return dt

        available_dates = self.regional_weights.index[self.regional_weights.index <= dt]
        if len(available_dates) == 0:
            return self.regional_weights.index[0]
        return available_dates[-1]

    def get_baseline_weights(self, dt: pd.Timestamp) -> Dict[str, float]:
        """
        Get portfolio weights for a given date based on regional allocation.

        If regional_weights is provided, uses ACWI-based dynamic allocation.
        Otherwise uses equal-weight satellites.
        """
        if self.regional_weights is None or self.regional_weights.empty:
            # Equal-weight satellites
            if len(self.satellite_isins) == 0:
                # No satellites defined - 100% core
                return {self.core_isin: 1.0}

            equal_sat_weight = self.satellite_weight / len(self.satellite_isins)
            weights = {self.core_isin: self.core_weight}
            for isin in self.satellite_isins.values():
                weights[isin] = equal_sat_weight
            return weights

        # Get regional weights for this date
        weight_date = self.get_weight_date_for(dt)
        region_weights = self.regional_weights.loc[weight_date]

        # Normalize to sum to 100 (only our regions)
        total = sum(region_weights.get(r, 0) for r in self.regions if r in region_weights.index)
        if total == 0:
            total = 1

        # Build portfolio weights
        weights = {self.core_isin: self.core_weight}

        for region, isin in self.satellite_isins.items():
            if region in region_weights.index:
                region_pct = region_weights[region] / total
                satellite_allocation = self.satellite_weight * region_pct

                if isin in weights:
                    weights[isin] += satellite_allocation
                else:
                    weights[isin] = satellite_allocation
            else:
                weights[isin] = self.satellite_weight / len(self.satellite_isins)

        return weights

    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        rebalance_trigger: Optional[Callable] = None,
        weight_calculator: Optional[Callable] = None,
        initial_investment: float = 10000,
        monthly_contribution: float = 0,
        smart_cashflow: bool = True,
        verbose: bool = False,
        **trigger_kwargs,
    ) -> BacktestResult:
        """
        Run backtest with specified parameters.

        Args:
            start_date: Start date (YYYY-MM-DD) or None for earliest
            end_date: End date (YYYY-MM-DD) or None for latest
            rebalance_trigger: Function(date, prices, units_held, daily_prices,
                              last_rebalance_date, **kwargs) -> bool
                              If None, uses monthly rebalancing
            weight_calculator: Function(date, prices) -> Dict[str, float]
                              If None, uses get_baseline_weights
            initial_investment: Starting capital
            monthly_contribution: Monthly deposit amount
            verbose: Print rebalancing details
            **trigger_kwargs: Additional keyword arguments passed to rebalance_trigger

        Returns:
            BacktestResult with portfolio values and rebalancing history
        """
        # Filter to date range for backtest iteration
        # Keep full_prices for weight calculator (needs historical data for momentum)
        full_prices = self.prices.copy()
        prices = full_prices.copy()
        if start_date:
            prices = prices[prices.index >= pd.to_datetime(start_date)]
        if end_date:
            prices = prices[prices.index <= pd.to_datetime(end_date)]

        if len(prices) == 0:
            raise ValueError("No data in specified date range")

        # Default weight calculator (wrap method to match expected signature)
        if weight_calculator is None:
            def default_calculator(date, prices):
                return self.get_baseline_weights(date)
            weight_calculator = default_calculator

        # Default rebalance trigger (monthly)
        if rebalance_trigger is None:
            def monthly_trigger(date, prices, units_held, daily_prices, last_rebalance, **kwargs):
                if last_rebalance is None:
                    return True
                months_diff = (date.year - last_rebalance.year) * 12 + (date.month - last_rebalance.month)
                return months_diff >= 1
            rebalance_trigger = monthly_trigger

        # Initialize tracking
        all_isins = [self.core_isin] + list(self.satellite_isins.values())
        units_held = {}  # Start with empty dict - will be populated dynamically
        portfolio_values = []
        rebalance_dates = []
        weights_history = []
        holdings_history = []

        last_rebalance_date = None
        last_contribution_month = None
        rebalance_count = 0

        # Main backtest loop
        for date, daily_prices in prices.iterrows():
            # Calculate current portfolio value (only from ISINs we actually hold)
            pv_before = sum(
                units_held.get(isin, 0) * daily_prices[isin]
                for isin in units_held.keys()
                if isin in daily_prices and pd.notna(daily_prices[isin])
            )

            # Handle contributions
            contribution = 0.0
            make_contribution = False

            if last_contribution_month is None:
                # First day: initial investment
                contribution = initial_investment
                make_contribution = True
            elif monthly_contribution > 0:
                # Check if new month
                if (date.month != last_contribution_month.month or
                    date.year != last_contribution_month.year):
                    contribution = monthly_contribution
                    make_contribution = True

            # Check rebalancing trigger
            should_rebalance = rebalance_trigger(
                date, prices, units_held, daily_prices, last_rebalance_date,
                start_date=prices.index[0],
                **trigger_kwargs
            )

            if should_rebalance:
                # Rebalance with smart cash flow management
                # Pass full_prices so weight calculator has historical data for momentum
                weights = weight_calculator(date, full_prices)
                total_value = pv_before + contribution

                if smart_cashflow and contribution > 0:
                    # Smart cash flow: minimize transactions
                    # Step 1: Remove positions that are no longer needed (not in weights or weight=0)
                    cash_from_selling = 0
                    isins_to_remove = []

                    for isin in list(units_held.keys()):
                        if isin not in weights or weights[isin] == 0:
                            # Sell this position completely
                            if isin in daily_prices and pd.notna(daily_prices[isin]) and daily_prices[isin] > 0:
                                cash_from_selling += units_held[isin] * daily_prices[isin]
                            isins_to_remove.append(isin)

                    for isin in isins_to_remove:
                        del units_held[isin]

                    # Step 2: Trim over-allocated positions to target
                    cash_from_trimming = 0
                    for isin in list(units_held.keys()):
                        if isin in weights and weights[isin] > 0:
                            if isin in daily_prices and pd.notna(daily_prices[isin]) and daily_prices[isin] > 0:
                                target_value = total_value * weights[isin]
                                current_value = units_held[isin] * daily_prices[isin]

                                if current_value > target_value:
                                    # Trim to target
                                    trim_amount = current_value - target_value
                                    cash_from_trimming += trim_amount
                                    units_held[isin] = target_value / daily_prices[isin]

                    # Step 3: Use cash (selling + trimming + contribution) to fill positions
                    available_cash = contribution + cash_from_selling + cash_from_trimming

                    # Buy/add to positions that are under-allocated
                    for isin, target_weight in weights.items():
                        if target_weight > 0 and isin in daily_prices and pd.notna(daily_prices[isin]) and daily_prices[isin] > 0:
                            target_value = total_value * target_weight
                            current_value = units_held.get(isin, 0) * daily_prices[isin]

                            if current_value < target_value:
                                # Need to buy more
                                needed = target_value - current_value
                                to_buy = min(needed, available_cash)
                                units_held[isin] = units_held.get(isin, 0) + (to_buy / daily_prices[isin])
                                available_cash -= to_buy

                                if available_cash <= 0.01:  # Small tolerance
                                    break

                    # If still have cash, allocate proportionally to all positions
                    if available_cash > 0.01:
                        for isin, target_weight in weights.items():
                            if target_weight > 0 and isin in daily_prices and pd.notna(daily_prices[isin]) and daily_prices[isin] > 0:
                                add_value = available_cash * target_weight
                                units_held[isin] = units_held.get(isin, 0) + (add_value / daily_prices[isin])
                else:
                    # Standard rebalancing: sell everything and rebuy
                    units_held = {}
                    for isin, weight in weights.items():
                        if weight > 0 and isin in daily_prices and pd.notna(daily_prices[isin]) and daily_prices[isin] > 0:
                            units_held[isin] = (total_value * weight) / daily_prices[isin]

                last_rebalance_date = date
                rebalance_count += 1
                rebalance_dates.append(date)

                if make_contribution:
                    last_contribution_month = date

                if verbose:
                    print(f"Rebalance #{rebalance_count} on {date.date()}")
                    for region, isin in [('Core', self.core_isin)] + list(self.satellite_isins.items()):
                        print(f"  {region}: {weights.get(isin, 0)*100:.1f}%")

            elif make_contribution:
                # Invest contribution with baseline weights (no rebalance)
                weights = self.get_baseline_weights(date)

                for isin, weight in weights.items():
                    if weight > 0 and isin in daily_prices and pd.notna(daily_prices[isin]) and daily_prices[isin] > 0:
                        if isin not in units_held:
                            units_held[isin] = 0.0
                        units_held[isin] += (contribution * weight) / daily_prices[isin]

                last_contribution_month = date

            # Calculate final portfolio value
            pv = sum(
                units_held.get(isin, 0) * daily_prices[isin]
                for isin in units_held.keys()
                if isin in daily_prices and pd.notna(daily_prices[isin])
            )

            portfolio_values.append({'date': date, 'value': pv})

            # Track weights and holdings
            current_weights = {}
            current_holdings = {}
            for isin in units_held.keys():
                if isin in daily_prices and pd.notna(daily_prices[isin]):
                    holding_value = units_held.get(isin, 0) * daily_prices[isin]
                    current_weights[isin] = holding_value / pv if pv > 0 else 0
                    current_holdings[isin] = units_held.get(isin, 0)

            weights_history.append({'date': date, **current_weights})
            holdings_history.append({'date': date, **current_holdings})

        if verbose:
            print(f"\nBacktest complete: {rebalance_count} rebalances")

        # Build result
        portfolio_series = pd.DataFrame(portfolio_values).set_index('date')['value']
        weights_df = pd.DataFrame(weights_history).set_index('date')
        holdings_df = pd.DataFrame(holdings_history).set_index('date')

        return BacktestResult(
            portfolio=portfolio_series,
            rebalance_dates=rebalance_dates,
            weights_history=weights_df,
            holdings_history=holdings_df,
        )


# Convenience function for simple backtests
def backtest_portfolio(
    prices: pd.DataFrame,
    portfolio: Dict[str, float],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    rebalance_frequency: str = 'monthly',
    initial_investment: float = 10000,
    monthly_contribution: float = 0,
) -> BacktestResult:
    """
    Simple backtest with static portfolio weights.

    Args:
        prices: DataFrame with ISIN columns
        portfolio: Dict mapping ISIN -> weight (must sum to 1.0)
        start_date: Start date or None
        end_date: End date or None
        rebalance_frequency: 'monthly', 'quarterly', 'annual', or 'never'
        initial_investment: Starting capital
        monthly_contribution: Monthly deposit

    Returns:
        BacktestResult
    """
    # Validate weights
    total_weight = sum(portfolio.values())
    if not 0.99 <= total_weight <= 1.01:
        raise ValueError(f"Portfolio weights must sum to 1.0 (got {total_weight})")

    # Create dummy core/satellite structure
    isins = list(portfolio.keys())
    core_isin = isins[0]
    satellite_isins = {f"Satellite_{i}": isin for i, isin in enumerate(isins[1:])} if len(isins) > 1 else {}

    # Create backtester
    bt = Backtester(
        prices=prices,
        core_isin=core_isin,
        satellite_isins=satellite_isins,
        core_weight=portfolio[core_isin],
        regional_weights=None,
    )

    # Custom weight calculator (returns static weights)
    def static_weights(date, prices):
        return portfolio

    # Set rebalance trigger
    rebalance_months = {'monthly': 1, 'quarterly': 3, 'annual': 12, 'never': 9999}
    months = rebalance_months.get(rebalance_frequency, 1)

    def trigger(date, prices, units_held, daily_prices, last_rebalance, **kwargs):
        if last_rebalance is None:
            return True
        months_diff = (date.year - last_rebalance.year) * 12 + (date.month - last_rebalance.month)
        return months_diff >= months

    return bt.run(
        start_date=start_date,
        end_date=end_date,
        rebalance_trigger=trigger,
        weight_calculator=static_weights,
        initial_investment=initial_investment,
        monthly_contribution=monthly_contribution,
    )


if __name__ == "__main__":
    # Example usage
    print("\nBacktester Module")
    print("=" * 60)
    print("\nExample usage:")
    print("""
    from backtester import Backtester, PortfolioMetrics

    # Create backtester
    bt = Backtester(
        prices=price_df,
        core_isin='IE00B6R52259',
        satellite_isins={
            'North America': 'IE00B3YCGJ38',
            'Europe': 'IE00B53QG562',
        },
        core_weight=0.60,
        regional_weights=df_weights
    )

    # Run backtest
    result = bt.run(
        start_date='2020-01-01',
        end_date='2024-12-31',
        initial_investment=10000,
        monthly_contribution=500
    )

    # Calculate metrics
    metrics = PortfolioMetrics.calculate(result.portfolio)
    PortfolioMetrics.print_summary(metrics)

    # Access results
    print(f"Final value: {result.portfolio.iloc[-1]:.2f}")
    print(f"Rebalances: {len(result.rebalance_dates)}")
    """)
