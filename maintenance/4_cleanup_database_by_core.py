"""
Clean Database by Core ETF Inception Date

This script removes all price data before the core ETF's inception date,
since alpha calculations require the core ETF to be available.

Core ETF: IE00B4L5Y983 (iShares Core MSCI World)
Inception: 2009-09-25
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "support"))

import pandas as pd
import sqlite3
from etf_database import ETFDatabase


def cleanup_database(core_isin='IE00B4L5Y983', dry_run=True):
    """
    Remove all price data before core ETF's inception date.

    Args:
        core_isin: ISIN of core ETF
        dry_run: If True, only show what would be deleted (don't actually delete)
    """
    print("=" * 80)
    print("DATABASE CLEANUP BY CORE ETF INCEPTION DATE")
    print("=" * 80)
    print()

    db = ETFDatabase("data/etf_database.db")

    # Load core ETF to get inception date
    print(f"Loading core ETF ({core_isin})...")
    all_prices = db.load_all_prices()

    if core_isin not in all_prices.columns:
        print(f"ERROR: Core ETF {core_isin} not found in database!")
        return

    core_prices = all_prices[core_isin].dropna()
    core_inception = core_prices.index[0]

    print(f"  Core ETF inception date: {core_inception.date()}")
    print(f"  Core ETF data points: {len(core_prices)}")
    print()

    # Check how much data would be removed
    print("Analyzing database...")

    dates_before = all_prices.index < core_inception
    n_dates_before = dates_before.sum()
    n_dates_after = (~dates_before).sum()

    print(f"  Total dates in database: {len(all_prices)}")
    print(f"  Dates before core inception: {n_dates_before} ({n_dates_before/len(all_prices)*100:.1f}%)")
    print(f"  Dates after core inception: {n_dates_after} ({n_dates_after/len(all_prices)*100:.1f}%)")
    print()

    # Count records that would be deleted
    conn = sqlite3.connect("data/etf_database.db")
    cursor = conn.cursor()

    # Get total record count
    cursor.execute("SELECT COUNT(*) FROM prices")
    total_records = cursor.fetchone()[0]

    # Count records before core inception
    cursor.execute("""
        SELECT COUNT(*) FROM prices
        WHERE date < ?
    """, (core_inception.strftime('%Y-%m-%d'),))
    records_before = cursor.fetchone()[0]

    records_after = total_records - records_before

    print(f"Database statistics:")
    print(f"  Total records: {total_records:,}")
    print(f"  Records before {core_inception.date()}: {records_before:,} ({records_before/total_records*100:.1f}%)")
    print(f"  Records after {core_inception.date()}: {records_after:,} ({records_after/total_records*100:.1f}%)")
    print()

    # Check database size
    cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
    db_size_bytes = cursor.fetchone()[0]
    db_size_mb = db_size_bytes / 1024**2
    estimated_size_after = db_size_mb * (records_after / total_records)
    size_saved = db_size_mb - estimated_size_after

    print(f"Storage:")
    print(f"  Current database size: {db_size_mb:.1f} MB")
    print(f"  Estimated size after cleanup: {estimated_size_after:.1f} MB")
    print(f"  Space saved: {size_saved:.1f} MB ({size_saved/db_size_mb*100:.1f}%)")
    print()

    if dry_run:
        print("=" * 80)
        print("[DRY RUN] No changes made to database")
        print("=" * 80)
        print()
        print("To actually perform the cleanup, run:")
        print("  python cleanup_database_by_core.py --execute")
        print()
        print("WARNING: This operation cannot be undone!")
        print("         Make a backup of data/etf_database.db before proceeding")
    else:
        print("=" * 80)
        print("PERFORMING CLEANUP")
        print("=" * 80)
        print()

        # Delete records before core inception
        print(f"Deleting {records_before:,} records before {core_inception.date()}...")
        cursor.execute("""
            DELETE FROM prices
            WHERE date < ?
        """, (core_inception.strftime('%Y-%m-%d'),))

        deleted = cursor.rowcount
        conn.commit()

        print(f"  Deleted: {deleted:,} records")
        print()

        # Vacuum to reclaim space
        print("Vacuuming database to reclaim space...")
        cursor.execute("VACUUM")
        conn.commit()
        print("  Done")
        print()

        # Check new size
        cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        new_size_bytes = cursor.fetchone()[0]
        new_size_mb = new_size_bytes / 1024**2
        actual_saved = db_size_mb - new_size_mb

        print("=" * 80)
        print("CLEANUP COMPLETE")
        print("=" * 80)
        print()
        print(f"Results:")
        print(f"  Records deleted: {deleted:,}")
        print(f"  Old database size: {db_size_mb:.1f} MB")
        print(f"  New database size: {new_size_mb:.1f} MB")
        print(f"  Space saved: {actual_saved:.1f} MB ({actual_saved/db_size_mb*100:.1f}%)")
        print()
        print("✓ Database cleaned successfully!")

    conn.close()


if __name__ == '__main__':
    import sys

    # Check for --execute flag
    execute = '--execute' in sys.argv

    if execute:
        response = input("This will permanently delete data before 2009-09-25. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            sys.exit(0)

    cleanup_database(dry_run=not execute)
