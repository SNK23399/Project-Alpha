"""
STEP 7: Generate Monthly Portfolio Allocation
==============================================

Final step in pipeline - Converts backtest results to actual portfolio allocation.

Takes the latest backtest results (satellite selections) and:
1. Reads user's total portfolio budget
2. Allocates 60% to core ETF, 40% to satellites
3. Divides 40% equally among N selected satellites (each gets 40/N%)
4. Uses current ETF prices to calculate quantities
5. Optimizes quantities to minimize uninvested cash while respecting target allocations
6. Outputs detailed allocation table

Output:
    - Console table with allocation details
    - Optional CSV export of allocation

Usage:
    python 7_generate_monthly_allocation.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from support.etf_database import ETFDatabase


# ============================================================
# CONFIGURATION
# ============================================================

# Core ETF (benchmark) - iShares Core MSCI World UCITS ETF USD (Acc)
CORE_ISIN = 'IE00B4L5Y983'

# Allocation split
CORE_ALLOCATION_PCT = 0.60  # 60% to core
SATELLITE_ALLOCATION_PCT = 0.40  # 40% to satellites

# Database path
DB_PATH = Path(__file__).parent.parent / 'maintenance' / 'data' / 'etf_database.db'

# Backtest results path
BACKTEST_DIR = Path(__file__).parent / 'data' / 'backtest_results'


def get_latest_selected_satellites():
    """
    Read latest backtest results to get selected satellite ISINs.

    Returns:
        List of selected satellite ISINs
    """
    # Read N3 backtest (best performer based on summary)
    backtest_file = BACKTEST_DIR / 'bayesian_backtest_N3.csv'

    if not backtest_file.exists():
        raise FileNotFoundError(f"Backtest file not found: {backtest_file}")

    df = pd.read_csv(backtest_file)

    # Get latest row
    latest_row = df[df['date'] == df['date'].max()].iloc[0]

    # Parse ISINs
    isins = [x.strip() for x in latest_row['selected_isins'].split(',')]

    return isins, latest_row['date']


def get_current_prices(isins):
    """
    Get current (latest available) prices for given ISINs.

    Args:
        isins: List of ISIN codes

    Returns:
        Dictionary mapping ISIN -> price
    """
    db = ETFDatabase(db_path=str(DB_PATH), readonly=True)

    prices = {}
    missing = []

    for isin in isins:
        try:
            series = db.load_prices(isin)
            if len(series) > 0:
                latest_price = series.iloc[-1]
                prices[isin] = latest_price
            else:
                missing.append(isin)
        except Exception as e:
            print(f"  [ERROR] Failed to load price for {isin}: {e}")
            missing.append(isin)

    if missing:
        raise ValueError(f"Could not load prices for: {missing}")

    return prices


def get_etf_names(isins):
    """
    Get ETF names from database.

    Args:
        isins: List of ISIN codes

    Returns:
        Dictionary mapping ISIN -> name
    """
    db = ETFDatabase(db_path=str(DB_PATH), readonly=True)

    names = {}
    for isin in isins:
        try:
            etf = db.get_etf(isin)
            if etf:
                names[isin] = etf.get('name', 'Unknown')
            else:
                names[isin] = 'Unknown'
        except:
            names[isin] = 'Unknown'

    return names


def optimize_quantities(budget, allocations, prices):
    """
    Optimize quantities to minimize uninvested cash while respecting target allocations.

    Args:
        budget: Total portfolio budget
        allocations: Dict mapping ISIN -> target allocation amount
        prices: Dict mapping ISIN -> current price

    Returns:
        Tuple of (quantities dict, total invested amount)
    """
    quantities = {}
    total_invested = 0

    for isin, target_amount in allocations.items():
        price = prices[isin]

        # Calculate ideal quantity
        ideal_qty = target_amount / price

        # Round down to integer
        qty = int(np.floor(ideal_qty))

        # Record quantity
        quantities[isin] = qty
        total_invested += qty * price

    uninvested_cash = budget - total_invested

    # Try to reduce uninvested cash by rounding up one quantity at a time
    candidates = []
    for isin, qty in quantities.items():
        price = prices[isin]
        current_invested = qty * price
        target_amount = allocations[isin]
        shortfall = target_amount - current_invested

        # Can we afford to round up?
        if uninvested_cash >= price:
            candidates.append((isin, price, shortfall))

    # Sort by allocation deviation (biggest first)
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Round up quantities where possible
    for isin, price, _ in candidates:
        if uninvested_cash >= price:
            quantities[isin] += 1
            total_invested += price
            uninvested_cash -= price

    return quantities, total_invested


def print_allocation(budget, core_isin, satellite_isins, prices, names, quantities, invested):
    """
    Print allocation table.

    Args:
        budget: Total budget
        core_isin: Core ETF ISIN
        satellite_isins: List of satellite ISINs
        prices: Dict ISIN -> price
        names: Dict ISIN -> name
        quantities: Dict ISIN -> quantity
        invested: Total amount invested
    """
    print("\n" + "=" * 120)
    print("PORTFOLIO ALLOCATION RESULTS")
    print("=" * 120)

    print(f"\nTotal Portfolio Budget: EUR {budget:,.2f}")

    # Collect all ISINs in order (core first, then satellites)
    all_isins = [core_isin] + satellite_isins

    # Build table data
    rows = []
    total_target = 0
    total_actual = 0

    for isin in all_isins:
        name = names.get(isin, 'Unknown')
        price = prices[isin]
        qty = quantities[isin]
        actual_invested = qty * price

        if isin == core_isin:
            target_pct = CORE_ALLOCATION_PCT * 100
            allocation_type = "CORE"
        else:
            target_pct = (SATELLITE_ALLOCATION_PCT / len(satellite_isins)) * 100
            allocation_type = "SATELLITE"

        target_amount = budget * (target_pct / 100)
        deviation = actual_invested - target_amount
        deviation_pct = (deviation / target_amount * 100) if target_amount > 0 else 0

        rows.append({
            'Type': allocation_type,
            'ISIN': isin,
            'Name': name[:30],
            'Target %': f"{target_pct:.1f}%",
            'Target EUR': f"{target_amount:,.0f}",
            'Price': f"EUR {price:.2f}",
            'Quantity': qty,
            'Actual EUR': f"{actual_invested:,.0f}",
            'Deviation': f"{deviation:+,.0f} ({deviation_pct:+.1f}%)"
        })

        total_target += target_amount
        total_actual += actual_invested

    # Print table
    print("\n" + "-" * 120)
    print(f"{'Type':<12} {'ISIN':<15} {'Name':<32} {'Target %':<12} {'Target EUR':<14} {'Price':<14} {'Quantity':<10} {'Actual EUR':<14} {'Deviation':<20}")
    print("-" * 120)

    for row in rows:
        print(f"{row['Type']:<12} {row['ISIN']:<15} {row['Name']:<32} {row['Target %']:<12} {row['Target EUR']:<14} {row['Price']:<14} {row['Quantity']:<10} {row['Actual EUR']:<14} {row['Deviation']:<20}")

    print("-" * 120)
    print(f"{'TOTAL':<12} {'':<15} {'':<32} {'100.0%':<12} {f'{total_target:,.0f}':<14} {'':<14} {'':<10} {f'{total_actual:,.0f}':<14}")
    print(f"\nUninvested Cash: EUR {budget - invested:,.2f} ({(budget - invested) / budget * 100:.2f}%)")
    print("\n" + "=" * 120)


def main():
    """Main entry point."""
    print("\n" + "=" * 120)
    print("STEP 7: GENERATE MONTHLY PORTFOLIO ALLOCATION")
    print("=" * 120)

    try:
        # Get latest satellite selections
        print("\n  Reading latest backtest results...")
        satellite_isins, backtest_date = get_latest_selected_satellites()
        n_satellites = len(satellite_isins)
        print(f"  [OK] Found {n_satellites} selected satellites (backtest date: {backtest_date})")
        print(f"       ISINs: {', '.join(satellite_isins)}")

        # Get prices
        print("\n  Loading current ETF prices...")
        all_isins = [CORE_ISIN] + satellite_isins
        prices = get_current_prices(all_isins)
        print(f"  [OK] Loaded prices for {len(prices)} ETFs")

        # Get names
        print("\n  Loading ETF names...")
        names = get_etf_names(all_isins)

        # Ask for budget
        print("\n" + "-" * 120)
        while True:
            try:
                budget_input = input("  Enter total portfolio budget (EUR): ").strip()
                budget = float(budget_input)
                if budget <= 0:
                    print("  Error: Budget must be positive")
                    continue
                break
            except ValueError:
                print("  Error: Please enter a valid number")
        print("-" * 120)

        # Calculate allocations
        print("\n  Calculating allocation...")

        core_target = budget * CORE_ALLOCATION_PCT
        satellite_per_target = budget * SATELLITE_ALLOCATION_PCT / n_satellites

        allocations = {
            CORE_ISIN: core_target
        }
        for isin in satellite_isins:
            allocations[isin] = satellite_per_target

        # Optimize quantities
        quantities, invested = optimize_quantities(budget, allocations, prices)
        print(f"  [OK] Calculated quantities for {len(quantities)} ETFs")

        # Print results
        print_allocation(budget, CORE_ISIN, satellite_isins, prices, names, quantities, invested)

        print("\n  [OK] Allocation generation completed")

        return {'status': 'completed', 'budget': budget, 'n_satellites': n_satellites}

    except Exception as e:
        print(f"\n  [ERROR] in step 7: {str(e)}")
        raise


if __name__ == "__main__":
    main()
