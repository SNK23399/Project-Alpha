#!/usr/bin/env python3
"""
Generate All Monthly Truncated Databases for Walk-Forward Validation
=====================================================================

Creates ~95 truncated database versions, one for each month from inception to current.
Each database contains only data available up to that month's end date.

Based on ACTUAL month-end trading dates (not estimated).

Usage:
    python generate_monthly_databases.py <source_db> [output_dir]

Example:
    python generate_monthly_databases.py ../maintenance/data/etf_database.db ./databases

Output:
    databases/
    ├── 2009-09-30/
    │   └── etf_database.db  (data up to 2009-09-30)
    ├── 2009-10-31/
    │   └── etf_database.db  (data up to 2009-10-31)
    └── ... (one per month)

    summary.csv:
    date,records_kept,records_deleted,first_date,last_date,db_size_mb
"""

import sys
import sqlite3
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from multiprocessing import Pool, cpu_count

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_OUTPUT_DIR = Path(__file__).parent / 'databases'
NUM_WORKERS = max(1, cpu_count() - 1)  # Use all cores except one

# ============================================================
# FUNCTIONS
# ============================================================

def get_month_end_dates(source_db):
    """
    Extract all unique month-end dates from the database.

    Returns list of dates sorted chronologically.
    """
    conn = sqlite3.connect(source_db)

    # Get all unique dates with data
    query = """
    SELECT DISTINCT date FROM prices
    ORDER BY date ASC
    """

    dates = pd.read_sql_query(query, conn, parse_dates=['date'])
    conn.close()

    if len(dates) == 0:
        raise ValueError("No price data found in database")

    dates = dates['date'].tolist()

    print(f"  Found {len(dates)} trading dates from {dates[0].date()} to {dates[-1].date()}")

    # Group by month and get last date of each month
    month_ends = {}
    for date in dates:
        month_key = date.strftime('%Y-%m')
        month_ends[month_key] = date

    month_end_dates = sorted(month_ends.values())

    print(f"  Identified {len(month_end_dates)} unique month-end dates")

    return month_end_dates

def process_single_date(args):
    """
    Worker function for multiprocessing - create one truncated database.

    Args:
        args: tuple of (source_db, target_date, output_dir, index, total)

    Returns:
        Tuple of (success, date_str, stats or error_msg)
    """
    source_db, target_date, output_dir, index, total = args
    date_str = target_date.strftime('%Y-%m-%d')
    month_dir = Path(output_dir) / date_str
    output_db = month_dir / 'etf_database.db'

    try:
        # Truncate database
        stats = truncate_database_for_date(source_db, target_date, str(output_db))
        return (True, index, total, date_str, stats, None)
    except Exception as e:
        return (False, index, total, date_str, None, str(e))

def truncate_database_for_date(source_db, target_date, output_db):
    """
    Create a new database with only data up to target_date.
    Uses ATTACH DATABASE for efficient cross-database operations.

    Returns: stats dictionary
    """
    source_path = Path(source_db)
    output_path = Path(output_db)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_date_str = target_date.strftime('%Y-%m-%d')

    # Get source database stats first
    source_conn = sqlite3.connect(source_db)
    source_cursor = source_conn.cursor()

    source_cursor.execute("SELECT COUNT(*) FROM prices")
    source_total = source_cursor.fetchone()[0]

    source_conn.close()

    # Create new database and attach source
    output_conn = sqlite3.connect(output_db)
    output_cursor = output_conn.cursor()

    try:
        # Attach source database
        output_cursor.execute(f"ATTACH DATABASE '{source_db}' AS source")

        # Copy schema from source
        # Get all table definitions (skip internal sqlite stats tables)
        output_cursor.execute(
            "SELECT sql FROM source.sqlite_master WHERE type='table' AND sql NOT NULL "
            "AND name NOT LIKE 'sqlite_stat%'"
        )
        tables = output_cursor.fetchall()

        for table_def in tables:
            try:
                output_cursor.execute(table_def[0])
            except sqlite3.OperationalError:
                pass

        # Copy indexes (skip sqlite internal indexes)
        output_cursor.execute(
            "SELECT sql FROM source.sqlite_master WHERE type='index' AND sql NOT NULL "
            "AND name NOT LIKE 'sqlite_autoindex%' AND name NOT LIKE 'sqlite_stat%'"
        )
        indexes = output_cursor.fetchall()

        for index_def in indexes:
            try:
                output_cursor.execute(index_def[0])
            except sqlite3.OperationalError:
                pass

        # Copy data up to target date using ATTACH
        # 1. Copy ETF metadata
        output_cursor.execute(
            """INSERT INTO main.etf_metadata_history
               SELECT * FROM source.etf_metadata_history
               WHERE snapshot_date <= ?""",
            (target_date_str,)
        )

        # 2. Copy ETFs table (all records, skip duplicates from parallel workers)
        output_cursor.execute(
            "INSERT OR IGNORE INTO main.etfs SELECT * FROM source.etfs"
        )

        # 3. Copy prices up to target date
        output_cursor.execute(
            """INSERT INTO main.prices
               SELECT * FROM source.prices
               WHERE date <= ?""",
            (target_date_str,)
        )

        output_conn.commit()

        # Get stats from output database
        output_cursor.execute("SELECT COUNT(*) FROM main.prices")
        records_kept = output_cursor.fetchone()[0]

        output_cursor.execute("SELECT MIN(date), MAX(date) FROM main.prices")
        date_range = output_cursor.fetchone()
        first_date, last_date = date_range if date_range[0] else (None, None)

        records_deleted = source_total - records_kept

        # Get file size
        db_size_mb = os.path.getsize(output_db) / (1024 * 1024) if os.path.exists(output_db) else 0

        stats = {
            'target_date': target_date_str,
            'records_kept': records_kept,
            'records_deleted': records_deleted,
            'first_date': first_date,
            'last_date': last_date,
            'total_records': source_total,
            'db_size_mb': db_size_mb
        }

        return stats

    finally:
        output_conn.close()

def format_size(bytes_val):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python generate_monthly_databases.py <source_db> [output_dir]")
        print("\nExample:")
        print("  python generate_monthly_databases.py ../maintenance/data/etf_database.db ./databases")
        sys.exit(1)

    source_db = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR

    try:
        print("\n" + "="*100)
        print("MONTHLY DATABASE TRUNCATION - WALK-FORWARD VALIDATION")
        print("="*100)

        # Validate source database
        source_path = Path(source_db)
        if not source_path.exists():
            raise FileNotFoundError(f"Source database not found: {source_db}")

        print(f"\nSource Database: {source_path.absolute()}")
        print(f"Output Directory: {output_dir.absolute()}")
        print(f"Worker Processes: {NUM_WORKERS} (parallel processing enabled)")

        # Extract month-end dates
        print(f"\nExtracting trading dates...")
        month_end_dates = get_month_end_dates(source_db)

        print(f"\nMonth-end dates:")
        print(f"  First: {month_end_dates[0].strftime('%Y-%m-%d')}")
        print(f"  Last:  {month_end_dates[-1].strftime('%Y-%m-%d')}")
        print(f"  Count: {len(month_end_dates)}")

        # Create truncated databases in parallel
        print(f"\nGenerating {len(month_end_dates)} truncated databases in parallel...")
        print(f"Using {NUM_WORKERS} worker processes")
        print("-" * 100)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare arguments for worker function
        worker_args = [
            (str(source_path), target_date, output_dir, idx, len(month_end_dates))
            for idx, target_date in enumerate(month_end_dates, 1)
        ]

        summary_data = []
        failed_dates = []

        # Process in parallel
        completed = 0
        with Pool(processes=NUM_WORKERS) as pool:
            for success, idx, total, date_str, stats, error in pool.imap_unordered(
                process_single_date, worker_args, chunksize=1
            ):
                completed += 1
                if success:
                    progress_pct = (completed / total) * 100
                    print(f"[{completed:3d}/{total}] {date_str} "
                          f"({progress_pct:5.1f}%) - "
                          f"{stats['records_kept']:,} records "
                          f"({stats['db_size_mb']:6.1f} MB)")

                    # Add to summary
                    summary_data.append({
                        'date': date_str,
                        'records_kept': stats['records_kept'],
                        'records_deleted': stats['records_deleted'],
                        'first_date': stats['first_date'],
                        'last_date': stats['last_date'],
                        'db_size_mb': round(stats['db_size_mb'], 2),
                        'total_records': stats['total_records']
                    })
                else:
                    progress_pct = (completed / total) * 100
                    print(f"[{completed:3d}/{total}] {date_str} ({progress_pct:5.1f}%) - ERROR: {error}")
                    failed_dates.append((date_str, error))

        # Save summary (sorted by date)
        summary_file = output_dir / 'summary.csv'
        summary_df = pd.DataFrame(summary_data)
        summary_df['date'] = pd.to_datetime(summary_df['date'])
        summary_df = summary_df.sort_values('date')
        summary_df['date'] = summary_df['date'].dt.strftime('%Y-%m-%d')
        summary_df.to_csv(summary_file, index=False)

        # Print summary
        print("-" * 100)
        print("\n" + "="*100)
        print("GENERATION COMPLETE")
        print("="*100)

        print(f"\nDatabases created: {len(summary_data)}")
        print(f"Failed: {len(failed_dates)}")

        if len(summary_data) > 0:
            print(f"\nDatabase Summary:")
            print(f"  First month: {summary_data[0]['first_date']} to {summary_data[0]['last_date']}")
            print(f"  Last month:  {summary_data[-1]['first_date']} to {summary_data[-1]['last_date']}")
            print(f"  Records (first): {summary_data[0]['records_kept']:,}")
            print(f"  Records (last):  {summary_data[-1]['records_kept']:,}")
            print(f"  Size (first):    {summary_data[0]['db_size_mb']:.1f} MB")
            print(f"  Size (last):     {summary_data[-1]['db_size_mb']:.1f} MB")
            print(f"  Total disk used: {sum(d['db_size_mb'] for d in summary_data):.1f} MB")

        print(f"\nOutput:")
        print(f"  Databases:  {output_dir}")
        print(f"  Summary:    {summary_file}")

        if failed_dates:
            print(f"\nFailed dates:")
            for date_str, error in failed_dates:
                print(f"  {date_str}: {error}")

        # Print directory structure example
        print(f"\nDirectory structure:")
        print(f"  databases/")
        print(f"  ├── 2009-09-30/")
        print(f"  │   └── etf_database.db")
        print(f"  ├── 2009-10-31/")
        print(f"  │   └── etf_database.db")
        print(f"  └── ... ({len(summary_data)} total)")

        print("\n" + "="*100 + "\n")

        return 0 if len(failed_dates) == 0 else 1

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
