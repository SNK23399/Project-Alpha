"""
Monthly Portfolio Allocation Generator - Stage 9
================================================

Generates actionable buy orders for a 60/40 core-satellite portfolio.

Input:
  - User provides investment budget
  - User chooses number of satellites (3-7)
  - Script reads latest satellite selections from Stage 6

Output:
  - CSV file with buy orders (ISIN, quantity, price, total cost)
  - Console report showing allocation vs target
  - Minimizes uninvested cash while matching 60/40 split

Algorithm:
  1. Load latest satellite selection from Stage 6 backtest results
  2. Query month-end prices from ETF database
  3. Calculate target amounts: Core 60%, Each Satellite (40%/N)
  4. Greedily allocate integer quantities to minimize allocation error
  5. Output actionable buy orders

Usage:
    python 9_generate_monthly_allocation.py

Output:
    data/allocation/allocation_YYYYMMDD_HHMMSS.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from support.etf_database import ETFDatabase

# ============================================================
# CONFIGURATION
# ============================================================

CORE_ISIN = 'IE00B4L5Y983'  # iShares Core MSCI World
CORE_TARGET_PCT = 0.60
SATELLITE_TARGET_PCT = 0.40

DATA_DIR = Path(__file__).parent / 'data'
BACKTEST_RESULTS_DIR = DATA_DIR / 'backtest_results'
OUTPUT_DIR = DATA_DIR / 'allocation'


# ============================================================
# USER INPUT
# ============================================================

def get_user_inputs():
    """Get budget and N value from user."""
    print("\n" + "="*100)
    print("MONTHLY PORTFOLIO ALLOCATION GENERATOR")
    print("="*100)
    print("\nThis script generates fresh allocation for new investment.")
    print("Note: Rebalancing of existing holdings is not supported in this version.")

    print("\nPlease provide the following information:")
    budget = float(input("  Available budget (EUR): "))

    print("\n  Number of satellites (3-7, default 5):")
    print("  Backtest Information Ratios:")
    print("    N=3: IR=1.135, Annual Alpha=44.9%")
    print("    N=4: IR=1.193, Annual Alpha=42.9%")
    print("    N=5: IR=1.228, Annual Alpha=42.9% (BEST IR)")
    print("    N=6: IR=1.194, Annual Alpha=41.6%")
    print("    N=7: IR=1.169, Annual Alpha=40.4%")

    n_input = input("  Enter N: ").strip()
    n_satellites = int(n_input) if n_input else 5

    if n_satellites < 3 or n_satellites > 7:
        print(f"  Warning: N={n_satellites} is outside tested range (3-7), using N=5")
        n_satellites = 5

    return budget, n_satellites


# ============================================================
# LOAD SATELLITE SELECTION
# ============================================================

def load_latest_selection(n_satellites):
    """
    Load the most recent satellite selection from Stage 6.

    Returns:
        (selection_date, list of ISINs)
    """
    backtest_file = BACKTEST_RESULTS_DIR / f'bayesian_backtest_N{n_satellites}.csv'

    if not backtest_file.exists():
        raise FileNotFoundError(
            f"Backtest results not found: {backtest_file}\n"
            f"Run Stage 6 (bayesian_strategy_ir.py) first."
        )

    # Load CSV and get last row
    df = pd.read_csv(backtest_file)

    if len(df) == 0:
        raise ValueError(f"Empty backtest results: {backtest_file}")

    # Get the most recent selection (last row)
    latest_row = df.iloc[-1]

    # Parse date and ISINs
    selection_date = pd.to_datetime(latest_row['date'])
    isins_str = latest_row['selected_isins']

    # Parse comma-separated ISINs
    satellite_isins = [isin.strip() for isin in isins_str.split(',')]

    if len(satellite_isins) != n_satellites:
        print(f"  Warning: Expected {n_satellites} satellites, but got {len(satellite_isins)}")

    return selection_date, satellite_isins


# ============================================================
# QUERY PRICES
# ============================================================

def get_latest_prices(isins, core_isin, selection_date):
    """
    Query month-end prices from ETF database.

    Uses the last trading day of the month corresponding to selection_date.
    This ensures price date matches the satellite selection date from Stage 6.

    Args:
        isins: List of satellite ISINs
        core_isin: ISIN of core ETF
        selection_date: Date of satellite selection (pandas Timestamp)

    Returns:
        (price_date, prices_series, names_dict)
    """
    # Include core + satellites
    all_isins = [core_isin] + isins

    # Load prices from database
    db = ETFDatabase(str(project_root / 'maintenance' / 'data' / 'etf_database.db'),
                     readonly=True)

    # Load recent price data (last 3 months to ensure we have the target month)
    start_date = (selection_date - pd.DateOffset(months=3)).strftime('%Y-%m-%d')
    end_date = selection_date.strftime('%Y-%m-%d')

    print(f"\n  Loading prices from database...")
    print(f"    Date range: {start_date} to {end_date}")

    prices_df = db.load_all_prices(all_isins, start_date=start_date, end_date=end_date)

    if len(prices_df) == 0:
        raise ValueError(f"No price data found for date range {start_date} to {end_date}")

    # Get prices for the selection month (last available day in that month)
    month_prices = prices_df[prices_df.index.to_period('M') == selection_date.to_period('M')]

    if len(month_prices) == 0:
        # Fallback: use last available prices
        print(f"  Warning: No prices found for {selection_date.strftime('%Y-%m')}, "
              f"using last available ({prices_df.index[-1].strftime('%Y-%m-%d')})")
        latest_prices = prices_df.iloc[-1]
        latest_date = prices_df.index[-1]
    else:
        # Use last trading day of the month
        latest_prices = month_prices.iloc[-1]
        latest_date = month_prices.index[-1]

    print(f"    Using prices from: {latest_date.strftime('%Y-%m-%d')}")

    # Load ETF names for display
    etf_names = {}
    for isin in all_isins:
        try:
            etf_info = db.get_etf(isin)
            etf_names[isin] = etf_info['name'] if etf_info else isin
        except Exception:
            etf_names[isin] = isin

    return latest_date, latest_prices, etf_names


# ============================================================
# CALCULATE ALLOCATION
# ============================================================

def calculate_allocation(budget, core_isin, satellite_isins, prices):
    """
    Calculate optimal integer quantities to match 60/40 split.

    Target allocation:
    - Core: 60% of budget
    - Each satellite: 40% / N satellites

    Algorithm:
    1. Calculate target amounts for each ETF
    2. Start with floor(target / price) for each ETF
    3. Use remaining budget to buy additional shares greedily:
       - Pick ETF that minimizes allocation error when buying 1 more share
       - Repeat until budget exhausted

    Args:
        budget: Total investment budget in EUR
        core_isin: ISIN of core ETF
        satellite_isins: List of satellite ISINs
        prices: pd.Series with prices indexed by ISIN

    Returns:
        (quantities_dict, uninvested_cash)
    """
    n_satellites = len(satellite_isins)
    core_target_pct = CORE_TARGET_PCT
    satellite_target_pct = SATELLITE_TARGET_PCT / n_satellites

    # Calculate target amounts for each ETF
    targets = {
        core_isin: budget * core_target_pct
    }
    for sat in satellite_isins:
        targets[sat] = budget * satellite_target_pct

    # Check if prices are available
    all_isins = [core_isin] + satellite_isins
    for isin in all_isins:
        if pd.isna(prices[isin]):
            raise ValueError(f"Missing price data for {isin}")

    # Initial allocation (floor of target / price)
    quantities = {}
    for isin in targets:
        quantities[isin] = int(targets[isin] / prices[isin])

    # Calculate remaining budget
    spent = sum(quantities[isin] * prices[isin] for isin in quantities)
    remaining = budget - spent

    # Greedy improvement: buy additional shares to minimize allocation error
    max_iterations = 1000  # Safety limit
    iteration = 0

    while remaining > 0 and iteration < max_iterations:
        iteration += 1
        best_isin = None
        best_error = float('inf')

        for isin in quantities:
            # Check if we can afford one more share
            if prices[isin] <= remaining:
                # Calculate squared allocation error with one more share
                test_quantities = quantities.copy()
                test_quantities[isin] += 1

                # Calculate total allocated value for error
                total_allocated = sum(test_quantities[i] * prices[i] for i in test_quantities)

                # Calculate squared allocation error
                error = 0
                for test_isin in test_quantities:
                    actual_amt = test_quantities[test_isin] * prices[test_isin]
                    actual_pct = actual_amt / total_allocated
                    target_pct = core_target_pct if test_isin == core_isin else satellite_target_pct
                    error += (actual_pct - target_pct) ** 2

                if error < best_error:
                    best_error = error
                    best_isin = isin

        # If we found an affordable ETF with improvement, buy one more share
        if best_isin is not None:
            quantities[best_isin] += 1
            remaining -= prices[best_isin]
        else:
            break  # No more affordable shares

    return quantities, remaining


# ============================================================
# OUTPUT FORMATTING
# ============================================================

def print_allocation_report(price_date, budget, core_isin, satellite_isins,
                            quantities, prices, names, uninvested):
    """Print clear, actionable allocation report."""

    print("\n" + "="*100)
    print(f"MONTHLY PORTFOLIO ALLOCATION - {price_date.strftime('%Y-%m-%d')}")
    print("="*100)

    print(f"\nBudget: {budget:,.2f} EUR")
    print(f"Target Split: 60% Core / 40% Satellites ({len(satellite_isins)} satellites)")
    print(f"Target per satellite: {40/len(satellite_isins):.2f}%")

    print("\n" + "-"*100)
    print(f"{'ISIN':<15} {'Name':<40} {'Qty':>5} {'Price':>10} {'Total':>12} {'%':>7}")
    print("-"*100)

    total_invested = 0

    # Core ETF
    core_qty = quantities[core_isin]
    core_cost = core_qty * prices[core_isin]
    core_pct = core_cost / budget * 100
    total_invested += core_cost
    print(f"{core_isin:<15} {names[core_isin][:40]:<40} {core_qty:>5} "
          f"{prices[core_isin]:>10.2f} {core_cost:>12.2f} {core_pct:>6.1f}%")

    print()  # Blank line before satellites

    # Satellites
    for sat_isin in satellite_isins:
        qty = quantities[sat_isin]
        cost = qty * prices[sat_isin]
        pct = cost / budget * 100
        total_invested += cost
        print(f"{sat_isin:<15} {names[sat_isin][:40]:<40} {qty:>5} "
              f"{prices[sat_isin]:>10.2f} {cost:>12.2f} {pct:>6.1f}%")

    print("-"*100)
    print(f"{'TOTAL INVESTED':<56} {'':<5} {'':<10} {total_invested:>12.2f} "
          f"{total_invested/budget*100:>6.1f}%")
    print(f"{'UNINVESTED CASH':<56} {'':<5} {'':<10} {uninvested:>12.2f} "
          f"{uninvested/budget*100:>6.1f}%")
    print("="*100)

    # Summary statistics
    if total_invested > 0:
        core_actual = core_cost / total_invested * 100
        satellite_actual = (total_invested - core_cost) / total_invested * 100
        print(f"\nActual Split: {core_actual:.1f}% Core / {satellite_actual:.1f}% Satellites")
        print(f"Target Split: 60.0% Core / 40.0% Satellites")
        print(f"Deviation: {abs(core_actual - 60):.2f}% (Core), {abs(satellite_actual - 40):.2f}% (Satellites)")


# ============================================================
# MAIN
# ============================================================

def main():
    """Main entry point."""
    try:
        # Get user inputs
        budget, n_satellites = get_user_inputs()

        print(f"\n{'='*100}")
        print(f"Loading satellite selection for N={n_satellites}...")
        print(f"{'='*100}")

        # Load latest selection from Stage 6
        selection_date, satellite_isins = load_latest_selection(n_satellites)

        print(f"\n  Selection date: {selection_date.strftime('%Y-%m-%d')}")
        print(f"  Satellites ({n_satellites}): {', '.join(satellite_isins)}")

        # Get month-end prices matching the selection date
        price_date, prices, names = get_latest_prices(satellite_isins, CORE_ISIN, selection_date)

        # Check for missing prices
        all_isins = [CORE_ISIN] + satellite_isins
        missing = [isin for isin in all_isins if pd.isna(prices[isin])]
        if missing:
            raise ValueError(f"Missing price data for: {', '.join(missing)}")

        print(f"\n{'='*100}")
        print(f"Calculating optimal allocation...")
        print(f"{'='*100}")

        # Calculate optimal allocation
        quantities, uninvested = calculate_allocation(
            budget, CORE_ISIN, satellite_isins, prices
        )

        # Print report
        print_allocation_report(
            price_date, budget, CORE_ISIN, satellite_isins,
            quantities, prices, names, uninvested
        )

        # Save to CSV
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = OUTPUT_DIR / f'allocation_{timestamp}.csv'

        # Create output DataFrame with metadata
        allocation_data = []
        for isin in [CORE_ISIN] + satellite_isins:
            allocation_data.append({
                'isin': isin,
                'name': names[isin],
                'quantity': quantities[isin],
                'price': float(prices[isin]),
                'total_cost': float(quantities[isin] * prices[isin]),
                'percentage': float(quantities[isin] * prices[isin] / budget * 100),
                'selection_date': selection_date.strftime('%Y-%m-%d'),
                'price_date': price_date.strftime('%Y-%m-%d'),
                'budget': budget,
                'n_satellites': n_satellites
            })

        df = pd.DataFrame(allocation_data)
        df.to_csv(output_file, index=False)

        print(f"\n[SAVED] Allocation saved to: {output_file}")

        return 0

    except Exception as e:
        print(f"\n[ERROR] {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
