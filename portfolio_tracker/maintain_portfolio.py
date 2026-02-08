#!/usr/bin/env python3
"""
DEGIRO Portfolio Maintenance Script

Handles all portfolio data operations:
- Fetches transaction history from DEGIRO
- Looks up ISINs, fund names, and vwd_ids
- Maintains portfolio_analysis.db with normalized schema
- Collects historical price data from DEGIRO (via ChartFetcher)
- Calculates daily holdings
- Computes portfolio values

Run once to sync everything:
    python maintain_portfolio.py
"""

import sys
import time
from pathlib import Path
from datetime import date, datetime
import sqlite3
import pandas as pd
import socket

# Fix DNS resolution by using Google's public DNS (8.8.8.8)
# The local DNS resolver may not have charting.vwdservices.com
original_getaddrinfo = socket.getaddrinfo

def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """Patched getaddrinfo that uses Google's DNS for vwdservices.com"""
    if 'vwdservices.com' in host:
        try:
            # Try to resolve using Google's DNS
            import socket as sock
            sock_obj = sock.socket(sock.AF_INET, sock.SOCK_DGRAM)
            sock_obj.connect(('8.8.8.8', 53))
            # Use dns.resolver if available
            try:
                import dns.resolver
                resolver = dns.resolver.Resolver()
                resolver.nameservers = ['8.8.8.8']
                result = resolver.resolve(host, 'A')
                ip = str(result[0])
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, port))]
            except:
                # Fallback: use known IPs for charting.vwdservices.com
                if host == 'charting.vwdservices.com':
                    return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('128.127.9.36', port))]
        except:
            pass
    return original_getaddrinfo(host, port, family, type, proto, flags)

socket.getaddrinfo = patched_getaddrinfo

# Add paths for imports (portfolio_tracker is inside Core Satellite)
script_dir = Path(__file__).parent  # portfolio_tracker folder
project_root = script_dir.parent     # Core Satellite folder
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'support'))

from degiro_client import get_client
from price_fetcher import PriceFetcher


# ============================================================================
# DEGIRO API Setup
# ============================================================================

def setup_degiro():
    """Connect to DEGIRO API."""
    print("=" * 80)
    print("CONNECTING TO DEGIRO")
    print("=" * 80)
    print()

    client = get_client()
    api = client.api

    print("✓ Connected to DEGIRO")
    print()
    return client, api


# ============================================================================
# Data Collection
# ============================================================================

def fetch_transactions(api):
    """Fetch complete transaction history from DEGIRO."""
    from degiro_connector.trading.models.transaction import HistoryRequest

    print("Fetching transactions from 2015 to today...")

    result = api.get_transactions_history(
        transaction_request=HistoryRequest(
            from_date=date(2015, 1, 1),
            to_date=date.today(),
        ),
        raw=False,
    )

    transactions_list = []
    if hasattr(result, 'data') and result.data:
        for tx in result.data:
            if hasattr(tx, 'model_dump'):
                tx_dict = tx.model_dump()
            elif hasattr(tx, 'dict'):
                tx_dict = tx.dict()
            elif isinstance(tx, dict):
                tx_dict = tx
            else:
                tx_dict = {}

            transactions_list.append(tx_dict)

    tx_df = pd.DataFrame(transactions_list)
    if len(tx_df) > 0:
        tx_df['date'] = pd.to_datetime(tx_df['date'])
        tx_df = tx_df.sort_values('date').reset_index(drop=True)

    print(f"✓ Found {len(tx_df)} transactions")
    if len(tx_df) > 0:
        print(f"  Date range: {tx_df['date'].min().date()} to {tx_df['date'].max().date()}")
        print(f"  Products: {tx_df['product_id'].nunique()} unique")
        print(f"  Transaction types: {tx_df['type'].unique() if 'type' in tx_df.columns else 'N/A'}\n")

    return tx_df


def fetch_product_details(api, tx_df):
    """Fetch ISIN, name, and vwd_id for each product."""
    print("Fetching product details...")

    # Extract product IDs
    product_ids = []
    for pid in tx_df['product_id'].unique():
        if pd.notna(pid):
            try:
                product_ids.append(int(float(pid)) if isinstance(pid, str) else int(pid))
            except (ValueError, TypeError):
                pass

    product_ids = sorted(list(set(product_ids)))
    print(f"Looking up {len(product_ids)} products\n")

    product_details = {}

    if product_ids:
        try:
            # Get ISIN, name, and vwdId from get_products_info
            response = api.get_products_info(product_list=product_ids, raw=False)

            if hasattr(response, 'data') and response.data:
                for product_id, product_item in response.data.items():
                    details = {}

                    if hasattr(product_item, 'isin'):
                        details['isin'] = product_item.isin

                    if hasattr(product_item, 'name'):
                        details['name'] = product_item.name

                    if hasattr(product_item, 'product_type'):
                        details['type'] = product_item.product_type

                    # Extract vwd_id (in vwdkey format for Tradegate: e.g., IE00B4L5Y983.TRADE,E)
                    if hasattr(product_item, 'vwd_id') and product_item.vwd_id:
                        details['vwd_id'] = str(product_item.vwd_id)

                    product_details[int(product_id)] = details

                found_count = sum(1 for d in product_details.values() if 'vwd_id' in d)
                print(f"✓ Got ISIN, names, and vwdIds for {len(product_details)} products")
                print(f"  ({found_count} with vwdId)\n")

        except Exception as e:
            print(f"✗ Error fetching products: {e}\n")

    return product_details


# ============================================================================
# Database Management
# ============================================================================

def setup_database(db_path):
    """Create normalized database schema."""
    print("=" * 80)
    print("SETUP PORTFOLIO DATABASE")
    print("=" * 80 + "\n")

    # Ensure parent directory exists
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create etfs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS etfs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            isin TEXT UNIQUE NOT NULL,
            name TEXT,
            type TEXT,
            currency TEXT DEFAULT 'EUR',
            vwd_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create trades table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            etf_id INTEGER NOT NULL,
            trade_date TEXT NOT NULL,
            buy_sell TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            fee REAL DEFAULT 0,
            degiro_tx_id TEXT UNIQUE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (etf_id) REFERENCES etfs(id)
        )
    """)

    # Create prices table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            etf_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            close_price REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(etf_id, date),
            FOREIGN KEY (etf_id) REFERENCES etfs(id)
        )
    """)

    # Create holdings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            etf_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            quantity REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(etf_id, date),
            FOREIGN KEY (etf_id) REFERENCES etfs(id)
        )
    """)

    # Create cash_movements table (deposits and withdrawals)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cash_movements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            type TEXT NOT NULL,
            amount REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create cash_balance table (calculated uninvested cash per date)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cash_balance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL UNIQUE,
            uninvested_cash REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add vwd_id column if it doesn't exist (migration for existing databases)
    cursor.execute("PRAGMA table_info(etfs)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'vwd_id' not in columns:
        print("  Migrating: Adding vwd_id column...")
        cursor.execute("ALTER TABLE etfs ADD COLUMN vwd_id TEXT")

    conn.commit()
    print("✓ Database schema created\n")
    conn.close()


def fetch_cash_movements(api):
    """Fetch cash movements (deposits/withdrawals) from DEGIRO."""
    from degiro_connector.trading.models.account import OverviewRequest

    print("Fetching cash movements (deposits/withdrawals) from DEGIRO...\n")

    try:
        # Fetch account overview with full history
        overview = api.get_account_overview(
            OverviewRequest(
                from_date=date(2015, 1, 1),
                to_date=date.today(),
            )
        )

        # Extract deposits and withdrawals (not trade-related fees)
        movements_list = []
        if overview and hasattr(overview, 'cash_movements') and overview.cash_movements:
            print(f"Found {len(overview.cash_movements)} cash movements from DEGIRO API:\n")
            for movement in overview.cash_movements:
                try:
                    # Get movement data
                    if hasattr(movement, 'model_dump'):
                        movement_dict = movement.model_dump()
                    elif hasattr(movement, 'dict'):
                        movement_dict = movement.dict()
                    elif isinstance(movement, dict):
                        movement_dict = movement
                    else:
                        continue

                    # Only extract DEPOSITS and WITHDRAWALS (not trades, not fees tied to trades)
                    description = movement_dict.get('description', '').upper()
                    change = movement_dict.get('change')
                    movement_date = movement_dict.get('date')

                    # Convert date to string for logging
                    if isinstance(movement_date, datetime):
                        date_str_log = movement_date.strftime('%Y-%m-%d')
                    else:
                        date_str_log = str(movement_date)[:10]

                    # Skip if no change amount
                    if pd.isna(change) or change is None or change == 0:
                        print(f"  [SKIPPED - No change amount] {date_str_log}: {description}")
                        continue

                    # Skip Deposit confirmations - we only need Reservations (when $ becomes available)
                    # Deposit confirmations are just follow-ups after Reservation, same money
                    if 'DEPOSIT' in description and 'RESERVATION' not in description:
                        print(f"  [SKIPPED - Deposit without Reservation] {date_str_log}: {description} (€{change})")
                        continue

                    # Identify deposits/withdrawals by description
                    # Use Reservations (when money becomes available), not Deposit confirmations
                    if 'RESERVATION' in description or 'INLEG' in description or 'IDEM OVERBOEKING' in description:
                        movement_type = 'DEPOSIT'
                        amount = abs(change)
                        print(f"  [EXTRACTED - DEPOSIT] {date_str_log}: {description} (€{amount:.2f})")
                    elif 'WITHDRAWAL' in description or 'OPNAME' in description:
                        movement_type = 'WITHDRAWAL'
                        amount = abs(change)
                        print(f"  [EXTRACTED - WITHDRAWAL] {date_str_log}: {description} (€{amount:.2f})")
                    else:
                        # Skip other movements (cash sweeps, fees, trades, interest, etc.)
                        print(f"  [SKIPPED - Other] {date_str_log}: {description} (€{change})")
                        continue

                    # Use the already-converted date_str_log for storage
                    date_str = date_str_log

                    movements_list.append({
                        'date': date_str,
                        'type': movement_type,
                        'amount': amount
                    })
                except Exception:
                    continue

        if movements_list:
            movements_df = pd.DataFrame(movements_list)

            # Deduplicate consecutive deposits of the same amount (within 5 days)
            # DEGIRO sometimes returns both Reservation and follow-up confirmations
            # Keep the earliest one (the true deposit date)
            if len(movements_df) > 1:
                movements_df['date_obj'] = pd.to_datetime(movements_df['date'])
                movements_df = movements_df.sort_values('date_obj').reset_index(drop=True)

                rows_to_drop = []
                for i in range(len(movements_df) - 1):
                    current = movements_df.iloc[i]
                    next_row = movements_df.iloc[i + 1]

                    # If same type, same amount, and within 5 days -> keep earliest, drop later ones
                    if (current['type'] == next_row['type'] == 'DEPOSIT' and
                        current['amount'] == next_row['amount'] and
                        (pd.to_datetime(next_row['date']) - pd.to_datetime(current['date'])).days <= 5):
                        rows_to_drop.append(i + 1)
                        print(f"  [DEDUPLICATED] {next_row['date']}: €{next_row['amount']:.2f} (duplicate of {current['date']})")

                movements_df = movements_df.drop(rows_to_drop).reset_index(drop=True)
                movements_df = movements_df[['date', 'type', 'amount']]

            print(f"\n✓ Found {len(movements_df)} deposit/withdrawal movements (after deduplication):")
            for movement_type in ['DEPOSIT', 'WITHDRAWAL']:
                count = len(movements_df[movements_df['type'] == movement_type])
                if count > 0:
                    total = movements_df[movements_df['type'] == movement_type]['amount'].sum()
                    print(f"  {movement_type}: {count} movements, Total: €{total:.2f}")
            print()
            return movements_df
        else:
            print("⚠ No deposits or withdrawals found\n")
            return pd.DataFrame()

    except Exception as e:
        print(f"⚠ Error fetching cash movements: {e}\n")
        return pd.DataFrame()


def sync_transactions(db_path, tx_df, product_details, cash_df=None):
    """Sync transactions and cash movements from DEGIRO into database."""
    print("=" * 80)
    print("SYNC TRANSACTIONS TO DATABASE")
    print("=" * 80 + "\n")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Insert unique ETFs
    unique_etfs = tx_df[['product_id']].drop_duplicates()
    inserted_etfs = 0

    for _, row in unique_etfs.iterrows():
        pid = int(row['product_id']) if pd.notna(row['product_id']) else None
        if pid and pid in product_details:
            info = product_details[pid]
            if 'isin' in info and pd.notna(info['isin']):
                try:
                    cursor.execute(
                        "INSERT OR IGNORE INTO etfs (isin, name, type, currency, vwd_id) VALUES (?, ?, ?, ?, ?)",
                        (info['isin'], info.get('name'), info.get('type'), 'EUR', info.get('vwd_id'))
                    )
                    inserted_etfs += 1
                except sqlite3.IntegrityError:
                    pass

    conn.commit()
    print(f"✓ Inserted {inserted_etfs} unique ETFs\n")

    # Map ETF IDs
    etf_map = pd.read_sql_query("SELECT id, isin FROM etfs", conn).set_index('isin')['id'].to_dict()

    # Insert trades
    inserted = 0
    dups = 0

    for idx, tx in tx_df.iterrows():
        pid = int(tx['product_id']) if pd.notna(tx['product_id']) else None
        if pid and pid in product_details and 'isin' in product_details[pid]:
            isin = product_details[pid]['isin']
            if isin in etf_map:
                try:
                    cursor.execute("""
                        INSERT INTO trades (etf_id, trade_date, buy_sell, quantity, price, fee, degiro_tx_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        etf_map[isin],
                        pd.to_datetime(tx['date']).strftime('%Y-%m-%d'),
                        tx.get('buysell'),
                        tx.get('quantity'),
                        tx.get('price'),
                        tx.get('fee_in_base_currency') if pd.notna(tx.get('fee_in_base_currency')) else 0,
                        str(tx.get('id', ''))
                    ))
                    inserted += 1
                except sqlite3.IntegrityError:
                    dups += 1

    conn.commit()
    print(f"Inserted: {inserted}, Duplicates: {dups}\n")

    cursor.execute("SELECT COUNT(*) FROM trades")
    total = cursor.fetchone()[0]
    print(f"✓ Total trades in database: {total}\n")

    # Insert cash movements (deposits/withdrawals) from DEGIRO account overview (if available)
    if cash_df is not None and len(cash_df) > 0:
        # Clear old cash movements to avoid duplicates from previous runs
        cursor.execute("DELETE FROM cash_movements")
        conn.commit()

        inserted_movements = 0
        for idx, row in cash_df.iterrows():
            try:
                movement_date = row.get('date')
                movement_type = row.get('type')
                amount = row.get('amount')

                if pd.notna(movement_date) and pd.notna(movement_type) and pd.notna(amount):
                    cursor.execute("""
                        INSERT INTO cash_movements (date, type, amount)
                        VALUES (?, ?, ?)
                    """, (
                        movement_date,
                        movement_type,
                        float(amount)
                    ))
                    inserted_movements += 1
            except (sqlite3.IntegrityError, TypeError, ValueError):
                pass

        conn.commit()
        if inserted_movements > 0:
            print(f"✓ Inserted {inserted_movements} cash movements (deposits/withdrawals)\n")

    conn.close()


def forward_fill_prices(db_path):
    """Fill missing dates (weekends/holidays) with last available price."""
    print("=" * 80)
    print("FORWARD-FILL MISSING PRICE DATA (Weekends/Holidays)")
    print("=" * 80 + "\n")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get all ETFs with prices
    etfs = pd.read_sql_query(
        "SELECT DISTINCT etf_id FROM prices ORDER BY etf_id",
        conn
    )['etf_id'].tolist()

    total_filled = 0

    for etf_id in etfs:
        # Get price date range
        date_range = pd.read_sql_query(
            "SELECT MIN(date) as first_date, MAX(date) as last_date FROM prices WHERE etf_id = ?",
            conn,
            params=(etf_id,)
        ).iloc[0]

        first_date = datetime.strptime(date_range['first_date'], '%Y-%m-%d')
        last_price_date = datetime.strptime(date_range['last_date'], '%Y-%m-%d')

        # Extend to today so weekends/holidays are included
        end_date = max(last_price_date, datetime.now())

        # Get all existing prices
        existing_prices = pd.read_sql_query(
            "SELECT date, close_price FROM prices WHERE etf_id = ? ORDER BY date",
            conn,
            params=(etf_id,)
        )
        existing_prices['date'] = pd.to_datetime(existing_prices['date'])

        # Generate all dates between first and today
        all_dates = pd.date_range(start=first_date, end=end_date, freq='D')

        # Forward fill to create continuous date range
        price_series = existing_prices.set_index('date')['close_price']
        price_series = price_series.reindex(all_dates)
        price_series = price_series.ffill()  # Forward fill NaN with last known price

        # Insert missing prices
        inserted = 0
        for date, price in price_series.items():
            if pd.notna(price):
                date_str = date.strftime('%Y-%m-%d')
                try:
                    cursor.execute(
                        "INSERT OR IGNORE INTO prices (etf_id, date, close_price) VALUES (?, ?, ?)",
                        (etf_id, date_str, float(price))
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    pass

        if inserted > 0:
            conn.commit()
            total_filled += inserted

    conn.close()
    print(f"✓ Forward-filled {total_filled} prices for weekends/holidays\n")


def collect_price_data(db_path):
    """Download historical price data from DEGIRO for all held ETFs."""
    print("=" * 80)
    print("COLLECT HISTORICAL PRICE DATA (DEGIRO)")
    print("=" * 80 + "\n")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get ETFs with trades and vwd_id
    etfs_df = pd.read_sql_query("""
        SELECT DISTINCT e.id, e.isin, e.name, e.vwd_id
        FROM etfs e
        WHERE EXISTS (SELECT 1 FROM trades WHERE etf_id = e.id)
        AND e.vwd_id IS NOT NULL
        ORDER BY e.isin
    """, conn)

    print(f"Found {len(etfs_df)} ETF(s) with trades and vwd_id:")
    for _, etf in etfs_df.iterrows():
        print(f"  {etf['isin']}: {etf['name']} (vwd_id: {etf['vwd_id']})")
    print()

    # Initialize PriceFetcher
    fetcher = PriceFetcher(verbose=False)

    success_count = 0
    fail_count = 0

    # Fetch prices for each ETF
    for _, etf in etfs_df.iterrows():
        etf_id = etf['id']
        isin = etf['isin']
        vwd_id = str(etf['vwd_id'])

        # Check if prices already exist
        cursor.execute("SELECT COUNT(*) FROM prices WHERE etf_id = ?", (etf_id,))
        price_count = cursor.fetchone()[0]

        if price_count > 0:
            print(f"✓ {isin}: Already has {price_count} price records")
            continue

        print(f"Fetching {isin} (vwd_id: {vwd_id})...", end=" ", flush=True)

        try:
            # Fetch OHLC data from DEGIRO
            df = fetcher.fetch_ohlc(vwd_id)

            if df is not None and len(df) > 0:
                # Insert prices (using close price)
                inserted = 0
                for price_date, row in df.iterrows():
                    if pd.notna(row['close']):
                        try:
                            cursor.execute(
                                "INSERT INTO prices (etf_id, date, close_price) VALUES (?, ?, ?)",
                                (etf_id, price_date.strftime('%Y-%m-%d'), float(row['close']))
                            )
                            inserted += 1
                        except sqlite3.IntegrityError:
                            pass

                conn.commit()
                print(f"✓ {inserted} records")
                success_count += 1

            else:
                print("✗ No data")
                fail_count += 1

        except Exception as e:
            print(f"✗ Error: {e}")
            fail_count += 1

        # Rate limiting to avoid overloading DEGIRO
        time.sleep(0.5)

    conn.close()
    print(f"\n✓ Price collection complete: {success_count} succeeded, {fail_count} failed\n")


def calculate_holdings(db_path):
    """Calculate daily quantity held for each ETF."""
    print("=" * 80)
    print("CALCULATE DAILY HOLDINGS")
    print("=" * 80 + "\n")

    conn = sqlite3.connect(str(db_path))

    # Get ETFs
    etfs = pd.read_sql_query(
        "SELECT id, isin FROM etfs WHERE id IN (SELECT DISTINCT etf_id FROM trades)",
        conn
    )

    # Clear holdings
    cursor = conn.cursor()
    cursor.execute("DELETE FROM holdings")
    conn.commit()

    # Calculate holdings for each ETF
    for _, etf in etfs.iterrows():
        etf_id = etf['id']
        isin = etf['isin']

        # Get trades
        trades = pd.read_sql_query(
            "SELECT trade_date, buy_sell, quantity FROM trades WHERE etf_id = ? ORDER BY trade_date",
            conn,
            params=(etf_id,)
        )

        if len(trades) == 0:
            continue

        # Date range
        first_date = datetime.strptime(trades['trade_date'].min(), '%Y-%m-%d')
        last_date = max(datetime.strptime(trades['trade_date'].max(), '%Y-%m-%d'), datetime.now())

        # Generate all dates
        date_range = pd.date_range(start=first_date, end=last_date, freq='D')

        # Calculate quantity per date
        holdings_list = []

        for current_date in date_range:
            trades_up_to = trades[trades['trade_date'] <= current_date.strftime('%Y-%m-%d')]

            qty = 0
            for _, trade in trades_up_to.iterrows():
                if trade['buy_sell'] == 'B':
                    qty += trade['quantity']
                else:
                    qty -= trade['quantity']

            if qty != 0:
                holdings_list.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'quantity': qty
                })

        # Insert holdings
        cursor = conn.cursor()
        for holding in holdings_list:
            try:
                cursor.execute(
                    "INSERT INTO holdings (etf_id, date, quantity) VALUES (?, ?, ?)",
                    (etf_id, holding['date'], holding['quantity'])
                )
            except sqlite3.IntegrityError:
                pass

        conn.commit()
        print(f"✓ {isin}: {len(holdings_list)} holding periods")

    conn.close()
    print("\n✓ Daily holdings calculated\n")


def calculate_cash_balance(db_path):
    """Calculate uninvested cash for each date from first principles.

    Formula: Uninvested = Deposits - Withdrawals - Buys + Sells - Fees

    This is deterministic and doesn't rely on DEGIRO's balance API.
    """
    print("=" * 80)
    print("CALCULATE UNINVESTED CASH FROM TRADES")
    print("=" * 80 + "\n")

    conn = sqlite3.connect(str(db_path))

    # Get all holding dates
    holding_dates = pd.read_sql_query(
        "SELECT DISTINCT date FROM holdings ORDER BY date",
        conn
    )

    if len(holding_dates) == 0:
        print("⚠ No holdings data found\n")
        conn.close()
        return

    # Get all deposits and withdrawals
    cash_movements = pd.read_sql_query(
        "SELECT date, type, amount FROM cash_movements ORDER BY date",
        conn
    )

    # Get all trades (buys, sells, fees)
    trades = pd.read_sql_query(
        "SELECT trade_date as date, buy_sell, quantity, price, fee FROM trades ORDER BY trade_date",
        conn
    )

    # Calculate uninvested cash for each date
    # Formula: Deposits - Withdrawals - Buys + Sells - Fees
    uninvested_list = []

    for _, row in holding_dates.iterrows():
        date_str = row['date']
        current_date = pd.to_datetime(date_str)

        # Sum deposits up to this date
        deposits = 0
        if len(cash_movements) > 0:
            deposits_df = cash_movements[
                (cash_movements['type'] == 'DEPOSIT') &
                (pd.to_datetime(cash_movements['date']) <= current_date)
            ]
            deposits = deposits_df['amount'].sum()

        # Sum withdrawals up to this date
        withdrawals = 0
        if len(cash_movements) > 0:
            withdrawals_df = cash_movements[
                (cash_movements['type'] == 'WITHDRAWAL') &
                (pd.to_datetime(cash_movements['date']) <= current_date)
            ]
            withdrawals = withdrawals_df['amount'].sum()

        # Sum buys (negative), sells (positive), fees (negative) up to this date
        buys = 0
        sells = 0
        fees = 0

        if len(trades) > 0:
            trades_up_to = trades[pd.to_datetime(trades['date']) <= current_date]

            # Buys reduce cash
            buys_df = trades_up_to[trades_up_to['buy_sell'] == 'B']
            buys = (buys_df['quantity'] * buys_df['price']).sum()

            # Sells increase cash
            sells_df = trades_up_to[trades_up_to['buy_sell'] == 'S']
            sells = (sells_df['quantity'] * sells_df['price']).sum()

            # Fees reduce cash
            fees = abs(trades_up_to['fee'].sum())

        # Formula: Uninvested = Deposits - Withdrawals - Buys + Sells - Fees
        uninvested_cash = deposits - withdrawals - buys + sells - fees

        uninvested_list.append({
            'date': date_str,
            'uninvested_cash': uninvested_cash
        })

    # Insert into database
    conn.execute("DELETE FROM cash_balance")
    for item in uninvested_list:
        conn.execute(
            "INSERT INTO cash_balance (date, uninvested_cash) VALUES (?, ?)",
            (item['date'], float(item['uninvested_cash']))
        )

    conn.commit()

    # Show summary
    if uninvested_list:
        first_item = uninvested_list[0]
        last_item = uninvested_list[-1]
        print(f"✓ Calculated uninvested cash for {len(uninvested_list)} dates")
        print(f"  First date ({first_item['date']}): €{first_item['uninvested_cash']:.2f}")
        print(f"  Last date ({last_item['date']}): €{last_item['uninvested_cash']:.2f}\n")

    conn.close()


# ============================================================================
# Main
# ============================================================================

def main():
    """Run complete portfolio maintenance."""
    print("\n" + "="*80)
    print("DEGIRO PORTFOLIO MAINTENANCE")
    print("="*80 + "\n")

    # Paths (database stored in portfolio_tracker/data)
    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "portfolio_analysis.db"

    print(f"Data folder: {data_dir}\n")

    # Step 1: Connect and fetch data
    client, api = setup_degiro()
    tx_df = fetch_transactions(api)
    product_details = fetch_product_details(api, tx_df)
    cash_df = fetch_cash_movements(api)

    # Step 2: Setup database and sync transactions
    setup_database(db_path)
    sync_transactions(db_path, tx_df, product_details, cash_df)

    # Step 3: Collect prices from DEGIRO
    collect_price_data(db_path)

    # Step 3b: Forward-fill missing dates (weekends/holidays)
    forward_fill_prices(db_path)

    # Step 4: Calculate holdings
    calculate_holdings(db_path)

    # Step 5: Calculate cash balance from transaction history
    calculate_cash_balance(db_path)

    print("="*80)
    print("PORTFOLIO MAINTENANCE COMPLETE")
    print("="*80)
    print(f"\nDatabase ready: {db_path}")
    print(f"\nRun the visualization notebook to view your portfolio")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
