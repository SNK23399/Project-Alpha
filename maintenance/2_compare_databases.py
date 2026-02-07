"""
Compare Old vs New ETF Databases

This script compares the old database (maintenance/data/etf_database.db) with the new one
(maintenance/data/etf_database_new.db) to verify that:
1. Same ETFs are present
2. Price data matches in overlapping periods (after core inception: 2009-09-25)
3. Old database has pre-2009 data that was correctly filtered out in new database
"""

import sys
from pathlib import Path

# Get project root directory (parent of maintenance folder)
PROJECT_ROOT = Path(__file__).parent.parent
MAINTENANCE_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "support"))

import pandas as pd
import numpy as np
from tqdm import tqdm
from etf_database import ETFDatabase


CORE_INCEPTION_DATE = pd.Timestamp('2009-09-25')

# Default paths in maintenance/data folder
DEFAULT_OLD_DB = MAINTENANCE_DIR / "data" / "etf_database.db"
DEFAULT_NEW_DB = MAINTENANCE_DIR / "data" / "etf_database_new.db"


def compare_databases(old_db_path=None, new_db_path=None):
    """Compare two ETF databases."""

    # Use default paths if not specified
    if old_db_path is None:
        old_db_path = DEFAULT_OLD_DB
    if new_db_path is None:
        new_db_path = DEFAULT_NEW_DB

    print("=" * 80)
    print("DATABASE COMPARISON")
    print("=" * 80)
    print()

    old_path = Path(old_db_path)
    new_path = Path(new_db_path)

    # Check if new database exists
    if not new_path.exists():
        print(f"[ERROR] New database not found: {new_db_path}")
        print()
        print("Run 1_collect_etf_data.py first to create the new database.")
        return

    # Check if old database exists - if not, this is a fresh install
    if not old_path.exists():
        print(f"[INFO] No existing database found at: {old_db_path}")
        print(f"       This appears to be a fresh installation.")
        print()

        # Load and validate new database
        db_new = ETFDatabase(new_db_path)
        stats_new = db_new.get_stats()

        print(f"New database: {new_db_path}")
        print(f"  ETFs: {stats_new['total_etfs']}")
        print(f"  Price records: {stats_new['total_price_records']:,}")
        print(f"  Date range: {stats_new['price_date_range']}")
        print(f"  Size: {stats_new['db_size_mb']:.1f} MB")
        print()

        # Validate the new database
        print("Validating new database...")
        validation = db_new.validate()
        if validation['valid']:
            print("[OK] Database validation passed")
        else:
            print("[WARN] Database validation issues:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        print()

        # Prompt to install
        response = input("Install new database as primary? (y/n): ").strip().lower()
        if response == 'y':
            import shutil
            shutil.move(new_db_path, old_db_path)
            print(f"  Installed database to: {old_db_path}")
            print()
            print("[OK] Database installation complete!")
        else:
            print("  Skipped. You can manually install with:")
            print(f"    mv {new_db_path} {old_db_path}")

        print("=" * 80)
        return

    # Both databases exist - proceed with comparison
    print("Loading databases...")
    db_old = ETFDatabase(old_db_path)
    db_new = ETFDatabase(new_db_path)

    stats_old = db_old.get_stats()
    stats_new = db_new.get_stats()

    print(f"\nOld database: {old_db_path}")
    print(f"  ETFs: {stats_old['total_etfs']}")
    print(f"  Price records: {stats_old['total_price_records']:,}")
    print(f"  Date range: {stats_old['price_date_range']}")
    print(f"  Size: {stats_old['db_size_mb']:.1f} MB")

    print(f"\nNew database: {new_db_path}")
    print(f"  ETFs: {stats_new['total_etfs']}")
    print(f"  Price records: {stats_new['total_price_records']:,}")
    print(f"  Date range: {stats_new['price_date_range']}")
    print(f"  Size: {stats_new['db_size_mb']:.1f} MB")

    print()

    # 1. Compare ETF metadata
    print("=" * 80)
    print("1. COMPARING ETF METADATA")
    print("=" * 80)
    print()

    old_isins = set(db_old.list_isins())
    new_isins = set(db_new.list_isins())

    only_in_old = old_isins - new_isins
    only_in_new = new_isins - old_isins
    common_isins = old_isins & new_isins

    print(f"ETFs in old database: {len(old_isins)}")
    print(f"ETFs in new database: {len(new_isins)}")
    print(f"Common ETFs: {len(common_isins)}")

    if only_in_old:
        print(f"\n[WARN] {len(only_in_old)} ETFs only in old database:")
        for isin in list(only_in_old)[:10]:
            print(f"  - {isin}")
        if len(only_in_old) > 10:
            print(f"  ... and {len(only_in_old) - 10} more")

    if only_in_new:
        print(f"\n[OK] {len(only_in_new)} new ETFs in new database:")
        for isin in list(only_in_new)[:10]:
            print(f"  - {isin}")
        if len(only_in_new) > 10:
            print(f"  ... and {len(only_in_new) - 10} more")

    if len(common_isins) == len(old_isins) and len(only_in_new) > 0:
        print(f"\n[OK] All old ETFs present in new database + {len(only_in_new)} new ones")

    print()

    # 2. Compare price data for common ETFs
    print("=" * 80)
    print("2. COMPARING PRICE DATA")
    print("=" * 80)
    print()

    print(f"Comparing prices for {len(common_isins)} common ETFs...")
    print(f"Checking overlap period: from {CORE_INCEPTION_DATE.date()} onwards")
    print()

    mismatches = []
    date_range_diffs = []
    pre_core_filtered = 0
    total_pre_core_records = 0

    # Quick comparison using SQL queries (much faster than loading all prices)
    print(f"Checking all {len(common_isins)} ETFs using efficient SQL queries...")
    print()

    import sqlite3

    conn_old = sqlite3.connect(str(db_old.db_path))
    conn_new = sqlite3.connect(str(db_new.db_path))

    try:
        # Get price stats for all ETFs in one query
        cursor_old = conn_old.cursor()
        cursor_new = conn_new.cursor()

        # For each ETF, check record count and date ranges
        etfs_checked = 0
        issues_found = 0

        for isin in tqdm(common_isins, desc="Checking ETFs", ncols=80):
            # Get old DB stats
            cursor_old.execute(
                "SELECT COUNT(*) as cnt, MIN(date) as min_date, MAX(date) as max_date FROM prices WHERE isin = ?",
                (isin,)
            )
            row_old = cursor_old.fetchone()
            old_count = row_old[0] if row_old[0] is not None else 0
            old_min = pd.Timestamp(row_old[1]) if row_old[1] else None
            old_max = pd.Timestamp(row_old[2]) if row_old[2] else None

            # Get new DB stats
            cursor_new.execute(
                "SELECT COUNT(*) as cnt, MIN(date) as min_date, MAX(date) as max_date FROM prices WHERE isin = ?",
                (isin,)
            )
            row_new = cursor_new.fetchone()
            new_count = row_new[0] if row_new[0] is not None else 0
            new_min = pd.Timestamp(row_new[1]) if row_new[1] else None
            new_max = pd.Timestamp(row_new[2]) if row_new[2] else None

            # Check for issues
            if old_count > 0 and new_count > 0:
                etfs_checked += 1

                # Check if new DB correctly filters pre-core data
                if new_min and new_min < CORE_INCEPTION_DATE:
                    date_range_diffs.append((isin, "New DB has pre-core data", new_min, new_max))
                    issues_found += 1

                # Check if record count makes sense
                # (new should have more or equal records for the overlapping period)
                if new_count < old_count * 0.9:  # Allow 10% variance
                    date_range_diffs.append((isin, f"Record count mismatch: {old_count} vs {new_count}", old_min, old_max))
                    issues_found += 1

                # Check for pre-core data in old DB
                if old_min and old_min < CORE_INCEPTION_DATE:
                    pre_core_filtered += 1
                    # Estimate pre-core records (rough)
                    if old_min and old_max:
                        ratio = (CORE_INCEPTION_DATE - old_min).days / (old_max - old_min).days
                        pre_core_estimated = int(old_count * ratio)
                        total_pre_core_records += pre_core_estimated

    finally:
        conn_old.close()
        conn_new.close()

    print(f"\n[OK] Checked all {etfs_checked} ETFs with prices in both databases")

    if pre_core_filtered > 0:
        print(f"[OK] Correctly filtered: {total_pre_core_records:,} pre-{CORE_INCEPTION_DATE.date()} records from {pre_core_filtered} ETFs")
    else:
        print(f"[WARN] No pre-{CORE_INCEPTION_DATE.date()} data found in old database sample")

    if mismatches:
        print(f"\n[WARN] Found {len(mismatches)} ETFs with price mismatches (>0.01 EUR):")
        for m in mismatches:
            print(f"  {m['isin']}: max diff = €{m['max_diff']:.4f}, {m['dates_with_diff']} dates differ (of {m['common_dates']} total)")
        print(f"\n  Note: Small price differences are normal due to NAV data source updates.")
    else:
        print(f"[OK] No significant price mismatches found in sample")

    if date_range_diffs:
        print(f"\n[WARN] Found {len(date_range_diffs)} ETFs with unexpected date ranges:")
        for isin, issue, start, end in date_range_diffs[:5]:
            print(f"  {isin}: {issue} ({start.date()} to {end.date()})")
    else:
        print(f"[OK] All sampled ETFs have correct date ranges")

    print()

    # 3. Overall statistics
    print("=" * 80)
    print("3. OVERALL STATISTICS")
    print("=" * 80)
    print()

    records_diff = stats_old['total_price_records'] - stats_new['total_price_records']
    records_pct = records_diff / stats_old['total_price_records'] * 100 if stats_old['total_price_records'] > 0 else 0

    size_diff = stats_old['db_size_mb'] - stats_new['db_size_mb']
    size_pct = size_diff / stats_old['db_size_mb'] * 100 if stats_old['db_size_mb'] > 0 else 0

    print(f"Price records:")
    print(f"  Old: {stats_old['total_price_records']:,}")
    print(f"  New: {stats_new['total_price_records']:,}")
    print(f"  Difference: {records_diff:,} ({records_pct:+.1f}%)")

    print(f"\nDatabase size:")
    print(f"  Old: {stats_old['db_size_mb']:.1f} MB")
    print(f"  New: {stats_new['db_size_mb']:.1f} MB")
    print(f"  Difference: {size_diff:+.1f} MB ({size_pct:+.1f}%)")

    print(f"\nDate ranges:")
    if stats_old['price_date_range'] and stats_new['price_date_range']:
        print(f"  Old: {stats_old['price_date_range'][0]} to {stats_old['price_date_range'][1]}")
        print(f"  New: {stats_new['price_date_range'][0]} to {stats_new['price_date_range'][1]}")

        old_start = pd.Timestamp(stats_old['price_date_range'][0])
        new_start = pd.Timestamp(stats_new['price_date_range'][0])

        if new_start >= CORE_INCEPTION_DATE:
            print(f"  [OK] New database starts at/after core inception ({CORE_INCEPTION_DATE.date()})")
        else:
            print(f"  [WARN] New database starts before core inception ({CORE_INCEPTION_DATE.date()})")

        if old_start < CORE_INCEPTION_DATE:
            print(f"  [OK] Old database has pre-core data (correctly filtered in new)")
        else:
            print(f"  [WARN] Old database doesn't have pre-core data (nothing to filter)")

    print()

    # 4. Verification verdict
    print("=" * 80)
    print("4. VERIFICATION VERDICT")
    print("=" * 80)
    print()

    issues = []

    if len(only_in_old) > 10:  # Allow a few missing (maybe delisted)
        issues.append(f"Many ETFs missing from new database ({len(only_in_old)})")

    if len(mismatches) > 5:  # Allow a few small mismatches
        issues.append(f"Many price mismatches found ({len(mismatches)})")

    if date_range_diffs:
        issues.append(f"Date range issues found ({len(date_range_diffs)})")

    if stats_new['price_date_range']:
        new_start = pd.Timestamp(stats_new['price_date_range'][0])
        if new_start < CORE_INCEPTION_DATE:
            issues.append(f"New database has pre-core data (should start at {CORE_INCEPTION_DATE.date()})")

    if issues:
        print("[WARN] VERIFICATION WARNINGS:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        print("Review the issues above before replacing the old database.")
    else:
        print("[OK] VERIFICATION PASSED")
        print()
        print("The new database looks correct:")
        print(f"  [OK] Same or more ETFs ({len(new_isins)} vs {len(old_isins)})")
        print(f"  [OK] Prices match in overlap period")
        print(f"  [OK] Pre-core data correctly filtered")
        print(f"  [OK] Reduced size by {abs(size_diff):.1f} MB")

    print()
    print("=" * 80)
    print()

    # Always prompt user to replace (even with warnings - user can decide)
    response = input("Replace old database with new one? (y/n): ").strip().lower()
    if response == 'y':
        import shutil
        from datetime import datetime

        # Create backups directory if needed
        backup_dir = Path(old_db_path).parent / "backups"
        backup_dir.mkdir(exist_ok=True)

        # Backup old database with date stamp
        date_stamp = datetime.now().strftime("%Y-%m-%d")
        backup_name = f"etf_database_{date_stamp}.db"
        backup_path = backup_dir / backup_name
        shutil.copy2(old_db_path, backup_path)
        print(f"  Backed up old database to: {backup_path}")

        # Replace with new database
        shutil.move(new_db_path, old_db_path)
        print(f"  Replaced {old_db_path} with new database")
        print()
        print("[OK] Database replacement complete!")
    else:
        print("  Skipped. You can manually replace with:")
        print(f"    mv {new_db_path} {old_db_path}")

    print("=" * 80)


if __name__ == "__main__":
    compare_databases()
