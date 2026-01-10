"""
ETF Data Collection Script

This script builds and maintains the ETF database:
1. Connects to DEGIRO
2. Applies base filter criteria (Irish domicile, accumulating, EUR)
3. Fetches price data for each ETF from JustETF
4. Filters data to only include dates >= core ETF inception (2009-09-25)
5. Saves everything to SQLite database

Output:
- data/etf_database.db - SQLite database containing:
  - etfs table: ETF metadata (name, TER, fund size, etc.)
  - prices table: Historical price data (from 2009-09-25 onwards)

Usage:
  python 0_collect_etf_data.py              # Full collection
  python 0_collect_etf_data.py --update     # Update existing database (monthly workflow)
"""

import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import justetf_scraping

# Add support directory to path
sys.path.insert(0, str(Path(__file__).parent / "support"))

from degiro_client import get_client
from etf_database import ETFDatabase
from etf_fetcher import ETFFetcher, ETFFilter

warnings.filterwarnings('ignore')

# Core ETF configuration
CORE_ISIN = 'IE00B4L5Y983'  # iShares Core MSCI World
CORE_INCEPTION_DATE = pd.Timestamp('2009-09-25')


def connect_degiro():
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


def initialize_database(db_path="data/etf_database_new.db", update_mode=False):
    """Initialize or connect to database."""
    print("=" * 80)
    print("INITIALIZING DATABASE")
    print("=" * 80)
    print()
    print(f"Database path: {db_path}")
    print()

    db = ETFDatabase(db_path)

    if update_mode:
        # Save metadata snapshot before updating
        print("Saving metadata snapshot (preserves historical TER, fund_size)...")
        snapshot_count = db.save_metadata_snapshot()
        print(f"✓ Saved metadata snapshot for {snapshot_count} ETFs")
        print()

    stats = db.get_stats()
    if stats['total_etfs'] > 0:
        print(f"✓ Connected to existing database:")
        print(f"  ETFs: {stats['total_etfs']}")
        print(f"  Price records: {stats['total_price_records']:,}")
        print(f"  Database size: {stats['db_size_mb']:.1f} MB")
    else:
        print("✓ Created new empty database")

    print()
    return db


def fetch_etf_universe():
    """Fetch ETF universe from DEGIRO with filters."""
    print("=" * 80)
    print("FETCHING ETF UNIVERSE")
    print("=" * 80)
    print()

    # Initialize fetcher
    fetcher = ETFFetcher(verbose=True)

    # Define filter criteria
    satellite_filter = ETFFilter(
        isin_prefix="IE00",           # Irish domiciled (tax efficient)
        distribution="Accumulating",  # Reinvest dividends
        currency="EUR",               # Currency in Euros
    )

    # Fetch ETFs matching criteria
    print("\nApplying filters...")
    df_universe = fetcher.fetch(satellite_filter)
    print(f"✓ Found {len(df_universe)} ETFs matching base criteria")

    # Filter out leveraged ETFs
    leverage_keywords = ['leveraged', 'leverage', '2x', '3x', 'ultra', 'double', 'triple',
                         'levered', 'geared', 'x2', 'x3', '200%', '300%']

    initial_count = len(df_universe)
    df_universe = df_universe[
        ~df_universe['Name'].str.lower().str.contains('|'.join(leverage_keywords), na=False)
    ]

    filtered_count = initial_count - len(df_universe)
    print(f"✓ Filtered out {filtered_count} leveraged ETFs")
    print(f"✓ Final universe: {len(df_universe)} ETFs")
    print()

    return df_universe


def add_etfs_to_database(db, df_universe):
    """Add ETF metadata to database."""
    print("=" * 80)
    print("ADDING ETFs TO DATABASE")
    print("=" * 80)
    print()

    def to_python(val):
        """Convert pandas NA/NaN to Python None."""
        if pd.isna(val):
            return None
        return val

    added_count = 0
    updated_count = 0

    for _, row in tqdm(df_universe.iterrows(), total=len(df_universe), desc="Adding ETFs"):
        is_new = db.add_etf(
            isin=row['ISIN'],
            name=row['Name'],
            vwd_id=str(row.get('vwdId', '')),
            exchange=to_python(row.get('exchange')),
            currency=to_python(row.get('currency')),
            ter=to_python(row.get('TER')),
            fund_size=to_python(row.get('fund_size')),
            distribution='Accumulating',
            months_of_data=to_python(row.get('months_of_data'))
        )
        if is_new:
            added_count += 1
        else:
            updated_count += 1

    print(f"\n✓ Added {added_count} new ETFs")
    if updated_count > 0:
        print(f"✓ Updated {updated_count} existing ETFs")
    print()


def fetch_and_store_prices(db, df_universe):
    """Fetch price data from JustETF and store in database."""
    print("=" * 80)
    print("FETCHING PRICE DATA")
    print("=" * 80)
    print()

    isins = df_universe['ISIN'].tolist()
    success_count = 0
    fail_count = 0
    validation_warnings = []
    filtered_count = 0
    total_records_filtered = 0

    print(f"Fetching price data for {len(isins)} ETFs...")
    print(f"Core ETF: {CORE_ISIN} (iShares Core MSCI World)")
    print(f"Filtering to dates >= {CORE_INCEPTION_DATE.date()}")
    print("(This may take several minutes)\n")

    for isin in tqdm(isins, desc="Fetching prices"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prices = justetf_scraping.load_chart(isin, currency='EUR')

            if prices is not None and len(prices) > 0:
                # Convert to Series if DataFrame
                if isinstance(prices, pd.DataFrame):
                    prices = prices.iloc[:, 0]

                # Filter to only keep dates >= core inception
                original_len = len(prices)
                prices = prices[prices.index >= CORE_INCEPTION_DATE]
                records_filtered = original_len - len(prices)

                if records_filtered > 0:
                    filtered_count += 1
                    total_records_filtered += records_filtered

                if len(prices) == 0:
                    # ETF has no data after core inception date
                    fail_count += 1
                    continue

                # Validate new prices against existing (if any)
                validation = db.validate_new_prices(isin, prices)

                if not validation['valid']:
                    # Price mismatch in overlap period - warn but continue
                    validation_warnings.append({
                        'isin': isin,
                        'mismatches': len(validation['mismatches']),
                        'overlap_days': validation['overlap_days'],
                        'sample': validation['mismatches'][:3]  # First 3 mismatches
                    })

                # Store in database (replace=True for full refresh)
                records_added = db.update_prices(isin, prices, replace=True)
                if records_added > 0:
                    success_count += 1
                else:
                    fail_count += 1
            else:
                fail_count += 1
        except Exception as e:
            fail_count += 1

        # Rate limiting for JustETF
        time.sleep(0.3)

    print(f"\n✓ Successfully fetched prices for {success_count}/{len(isins)} ETFs")
    if fail_count > 0:
        print(f"✗ Failed to fetch prices for {fail_count} ETFs")

    print(f"\n✓ Filtered {total_records_filtered:,} price records before {CORE_INCEPTION_DATE.date()}")
    if filtered_count > 0:
        print(f"  ({filtered_count} ETFs had data before core inception)")

    # Show validation warnings (if any)
    if validation_warnings:
        print(f"\n{'='*70}")
        print(f"⚠ WARNING: {len(validation_warnings)} ETFs had price mismatches in overlap period")
        print(f"{'='*70}")
        print("(Historical prices may have been corrected by the data source)\n")

        for warn in validation_warnings[:5]:  # Show first 5
            print(f"  {warn['isin']}: {warn['mismatches']} mismatches in {warn['overlap_days']} overlap days")

        if len(validation_warnings) > 5:
            print(f"\n  ... and {len(validation_warnings) - 5} more ETFs with warnings")

        print(f"\n{'='*70}")

    print()

    # Retry failed ETFs (if any)
    if fail_count > 0:
        retry_failed_etfs(db, isins, success_count)


def retry_failed_etfs(db, all_isins, initial_success):
    """Retry fetching prices for ETFs with no data."""
    print("=" * 80)
    print("RETRYING FAILED ETFs")
    print("=" * 80)
    print()

    # Find ETFs with no price data
    no_data_isins = []
    for isin in all_isins:
        prices = db.load_prices(isin)
        if len(prices) == 0:
            no_data_isins.append(isin)

    if not no_data_isins:
        print("✓ No ETFs need retry - all have price data!")
        print()
        return

    print(f"Retrying {len(no_data_isins)} ETFs with no data...")
    print("(Using longer delay to avoid rate limiting)\n")

    retry_success = 0
    retry_fail = 0

    for isin in tqdm(no_data_isins, desc="Retrying"):
        try:
            time.sleep(1.0)  # Longer delay for retry

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prices = justetf_scraping.load_chart(isin, currency='EUR')

            if prices is not None and len(prices) > 0:
                if isinstance(prices, pd.DataFrame):
                    prices = prices.iloc[:, 0]

                # Filter to core inception date
                prices = prices[prices.index >= CORE_INCEPTION_DATE]

                if len(prices) > 0:
                    records_added = db.update_prices(isin, prices, replace=True)
                    if records_added > 0:
                        retry_success += 1
                    else:
                        retry_fail += 1
                else:
                    retry_fail += 1
            else:
                retry_fail += 1
        except Exception as e:
            retry_fail += 1

    print(f"\n✓ Retry results: {retry_success} succeeded, {retry_fail} still failed")

    total_success = initial_success + retry_success
    total_fail = retry_fail

    stats = db.get_stats()
    print(f"✓ ETFs with prices: {stats['etfs_with_prices']}/{stats['total_etfs']}")
    print()


def check_data_quality(db):
    """Check data quality and report issues."""
    print("=" * 80)
    print("DATA QUALITY CHECK")
    print("=" * 80)
    print()

    all_isins = db.list_isins()
    quality_issues = []
    good_etfs = 0

    for isin in tqdm(all_isins, desc="Checking quality"):
        prices = db.load_prices(isin)
        etf_info = db.get_etf(isin)
        name = etf_info['name'][:40] if etf_info else isin

        if len(prices) == 0:
            quality_issues.append((isin, name, "NO DATA", 0))
            continue

        # Calculate metrics
        date_range = (prices.index.max() - prices.index.min()).days
        expected_trading_days = date_range * 5 / 7  # Rough estimate
        actual_days = len(prices)
        coverage = actual_days / expected_trading_days if expected_trading_days > 0 else 0

        # Check for gaps
        date_diff = prices.index.to_series().diff().dt.days
        max_gap = date_diff.max() if len(date_diff) > 0 else 0

        # Check for stale data
        days_since_update = (pd.Timestamp.now() - prices.index.max()).days

        # Flag issues
        issues = []
        if coverage < 0.85:
            issues.append(f"Low coverage ({coverage:.0%})")
        if max_gap > 10:
            issues.append(f"Gap of {max_gap} days")
        if days_since_update > 7:
            issues.append(f"Stale ({days_since_update}d old)")
        if len(prices) < 252:  # Less than 1 year
            issues.append(f"Short history ({len(prices)} days)")

        if issues:
            quality_issues.append((isin, name, ", ".join(issues), len(prices)))
        else:
            good_etfs += 1

    # Summary
    print(f"\n{'Status':<15} {'Count':>8}")
    print("-" * 25)
    print(f"{'Good ETFs':<15} {good_etfs:>8}")
    print(f"{'With Issues':<15} {len(quality_issues):>8}")
    print(f"{'Total':<15} {len(all_isins):>8}")

    # Show issues if any (first 20)
    if quality_issues:
        print(f"\n⚠ ETFs with data quality issues (showing first 20):")
        print(f"{'ISIN':<15} {'Name':<40} {'Issue':<30}")
        print("-" * 85)
        for isin, name, issue, days in sorted(quality_issues, key=lambda x: x[2])[:20]:
            print(f"{isin:<15} {name:<40} {issue:<30}")

        if len(quality_issues) > 20:
            print(f"\n... and {len(quality_issues) - 20} more")

        # Categorize
        no_data = [q for q in quality_issues if "NO DATA" in q[2]]
        stale = [q for q in quality_issues if "Stale" in q[2]]
        short = [q for q in quality_issues if "Short history" in q[2]]

        print(f"\nIssue breakdown:")
        if no_data:
            print(f"  - No data: {len(no_data)} ETFs")
        if stale:
            print(f"  - Stale data: {len(stale)} ETFs")
        if short:
            print(f"  - Short history: {len(short)} ETFs")

    print()


def print_summary(db):
    """Print final database summary."""
    stats = db.get_stats()

    print("=" * 80)
    print("DATABASE COLLECTION COMPLETE")
    print("=" * 80)
    print()
    print(f"Database: {db.db_path}")
    print(f"Total ETFs: {stats['total_etfs']}")
    print(f"ETFs with prices: {stats['etfs_with_prices']}")
    print(f"Total price records: {stats['total_price_records']:,}")
    if stats['price_date_range']:
        print(f"Price date range: {stats['price_date_range'][0]} to {stats['price_date_range'][1]}")
    print(f"Database size: {stats['db_size_mb']:.2f} MB")
    print()
    print("Filter criteria:")
    print(f"  - Core ETF: {CORE_ISIN} (iShares Core MSCI World)")
    print(f"  - Data from: {CORE_INCEPTION_DATE.date()} onwards")
    print(f"  - Irish domicile (IE00)")
    print(f"  - Accumulating distribution")
    print(f"  - EUR currency")
    print(f"  - No leveraged ETFs")
    print()
    print("✓ Data collection complete!")
    print()
    print("Next steps:")
    print("  1. Run 0_analyze_data_quality.py to verify data")
    print("  2. Run 1_compute_signal_bases.py to compute signals")
    print("=" * 80)


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description='Collect ETF data from DEGIRO/JustETF')
    parser.add_argument('--update', action='store_true',
                       help='Update mode: save metadata snapshot before refresh')
    args = parser.parse_args()

    start_time = time.time()

    try:
        # 1. Connect to DEGIRO
        client, api = connect_degiro()

        # 2. Initialize database
        db = initialize_database(update_mode=args.update)

        # 3. Fetch ETF universe
        df_universe = fetch_etf_universe()

        # 4. Add ETFs to database
        add_etfs_to_database(db, df_universe)

        # 5. Fetch and store price data
        fetch_and_store_prices(db, df_universe)

        # 6. Check data quality
        check_data_quality(db)

        # 7. Print summary
        elapsed = time.time() - start_time
        print(f"Total time: {elapsed/60:.1f} minutes")
        print()
        print_summary(db)

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
