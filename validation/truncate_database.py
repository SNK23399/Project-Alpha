"""
Truncate ETF database to a specific date for walk-forward validation.

Creates a copy of the database with only data up to the target date.
This simulates having only historical data available at a specific point in time.

Usage:
    python truncate_database.py source.db target_date output.db

Example:
    python truncate_database.py maintenance/data/etf_database.db 2020-01-31 validation/db_2020_01_31.db
"""

import sqlite3
import sys
import shutil
from pathlib import Path
from datetime import datetime


def truncate_database(source_db, target_date, output_db):
    """
    Create a copy of database with only data up to target_date.

    Args:
        source_db: Path to original database
        target_date: Cutoff date (YYYY-MM-DD format)
        output_db: Path to save truncated database

    Returns:
        Dictionary with truncation statistics
    """

    # Validate date format
    try:
        datetime.strptime(target_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date format: {target_date}. Use YYYY-MM-DD")

    # Copy database
    source_path = Path(source_db)
    output_path = Path(output_db)

    if not source_path.exists():
        raise FileNotFoundError(f"Source database not found: {source_db}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_db, output_db)

    print(f"Truncating database to {target_date}...")
    print(f"  Source: {source_db}")
    print(f"  Output: {output_db}")

    # Connect and truncate
    conn = sqlite3.connect(output_db)
    cursor = conn.cursor()

    try:
        # Get stats before deletion
        cursor.execute("SELECT COUNT(*) FROM prices")
        initial_count = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(date), MAX(date) FROM prices")
        min_date, max_date = cursor.fetchone()

        # Delete prices after target date
        cursor.execute(
            "DELETE FROM prices WHERE date > ?",
            (target_date,)
        )
        deleted_count = cursor.rowcount

        # Delete metadata history entries after target date
        cursor.execute(
            "DELETE FROM etf_metadata_history WHERE snapshot_date > ?",
            (target_date,)
        )
        deleted_metadata = cursor.rowcount

        conn.commit()

        # Get stats after deletion
        cursor.execute("SELECT COUNT(*) FROM prices")
        final_count = cursor.fetchone()[0]

        cursor.execute("SELECT MAX(date) FROM prices")
        final_max_date = cursor.fetchone()[0]

        stats = {
            'target_date': target_date,
            'initial_price_records': initial_count,
            'final_price_records': final_count,
            'deleted_price_records': deleted_count,
            'deleted_metadata_records': deleted_metadata,
            'original_date_range': f"{min_date} to {max_date}",
            'truncated_date_range': f"{min_date} to {final_max_date}",
        }

        return stats

    finally:
        conn.close()


def main():
    if len(sys.argv) != 4:
        print("Usage: python truncate_database.py source.db target_date output.db")
        print("Example: python truncate_database.py etf_database.db 2020-01-31 db_2020_01_31.db")
        sys.exit(1)

    source_db = sys.argv[1]
    target_date = sys.argv[2]
    output_db = sys.argv[3]

    try:
        stats = truncate_database(source_db, target_date, output_db)

        print(f"\n✓ Database truncation successful!")
        print(f"\nStatistics:")
        print(f"  Target date: {stats['target_date']}")
        print(f"  Original date range: {stats['original_date_range']}")
        print(f"  Truncated date range: {stats['truncated_date_range']}")
        print(f"  Price records: {stats['initial_price_records']} → {stats['final_price_records']}")
        print(f"  Deleted: {stats['deleted_price_records']} price records, {stats['deleted_metadata_records']} metadata records")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
