"""
ETF Database - SQLite-based storage for ETF universe and price data.

Provides persistent storage with support for:
- Incremental price updates (append new data without rewriting)
- ETF universe management (add/remove ETFs)
- Cross-ETF queries
- History tracking

Usage:
    from etf_database import ETFDatabase

    db = ETFDatabase()

    # Add ETFs to universe
    db.add_etf('IE00B4L5Y983', name='iShares MSCI ACWI', TER=0.20, fund_size=1000)

    # Update prices
    db.update_prices('IE00B4L5Y983', price_series)

    # Load data
    prices = db.load_prices('IE00B4L5Y983')
    universe = db.load_universe()
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Union

import pandas as pd
import numpy as np


class ETFDatabase:
    """SQLite database for ETF universe and price data."""

    def __init__(self, db_path: str = "data/etf_database.db", readonly: bool = True):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
            readonly: If True (default), open in read-only mode (prevents accidental writes).
                      Set to False explicitly to enable write operations.
        """
        self.db_path = Path(db_path)
        self.readonly = readonly

        if not readonly:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with optimized settings."""
        if self.readonly:
            # Open in read-only mode using URI
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        else:
            conn = sqlite3.connect(self.db_path)
            # Optimize for bulk inserts (only in write mode)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        # Enable foreign key enforcement (industry standard)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _check_writable(self):
        """Raise error if database is in read-only mode."""
        if self.readonly:
            raise PermissionError(
                "Database is open in read-only mode. "
                "Use ETFDatabase(db_path, readonly=False) to enable writes."
            )

    def _init_schema(self):
        """Initialize database schema if not exists."""
        conn = self._get_connection()
        try:
            # Enable auto-vacuum for automatic space reclamation
            conn.execute("PRAGMA auto_vacuum=INCREMENTAL")

            # ETF metadata table (current values)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS etfs (
                    isin TEXT PRIMARY KEY,
                    name TEXT,
                    vwd_id TEXT,
                    exchange TEXT,
                    currency TEXT,
                    ter REAL,
                    fund_size REAL,
                    distribution TEXT,
                    added_date TEXT,
                    last_price_update TEXT,
                    months_of_data INTEGER,
                    extra_data TEXT
                )
            """)

            # ETF metadata history table (track changes over time)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS etf_metadata_history (
                    isin TEXT NOT NULL,
                    snapshot_date TEXT NOT NULL,
                    ter REAL,
                    fund_size REAL,
                    months_of_data INTEGER,
                    PRIMARY KEY (isin, snapshot_date),
                    FOREIGN KEY (isin) REFERENCES etfs(isin) ON DELETE CASCADE
                )
            """)

            # Price data table with NOT NULL constraints for data integrity
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    isin TEXT NOT NULL,
                    date TEXT NOT NULL,
                    price REAL NOT NULL,
                    PRIMARY KEY (isin, date),
                    FOREIGN KEY (isin) REFERENCES etfs(isin) ON DELETE CASCADE
                )
            """)

            # Composite index for the most common query pattern: prices by ISIN and date range
            # This is more efficient than separate indexes for range queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prices_isin_date ON prices(isin, date)
            """)

            # Index for date-only queries (e.g., "all prices on a specific date")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)
            """)

            # Index for metadata history lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metadata_history_isin ON etf_metadata_history(isin)
            """)

            conn.commit()
        finally:
            conn.close()

    # ==================== ETF Management ====================

    def add_etf(
        self,
        isin: str,
        name: str = None,
        vwd_id: str = None,
        exchange: str = None,
        currency: str = None,
        ter: float = None,
        fund_size: float = None,
        distribution: str = None,
        months_of_data: int = None,
        **extra
    ) -> bool:
        """
        Add or update an ETF in the universe.

        Args:
            isin: ISIN identifier (required)
            name: ETF name
            vwd_id: DEGIRO vwdId for price fetching
            exchange: Exchange code (e.g., 'XET')
            currency: Trading currency
            ter: Total Expense Ratio (%)
            fund_size: Fund size in millions EUR
            distribution: 'Accumulating' or 'Distributing'
            months_of_data: Months of price history available
            **extra: Additional metadata stored as JSON

        Returns:
            True if added, False if updated existing
        """
        self._check_writable()
        conn = self._get_connection()
        try:
            # Check if exists
            existing = conn.execute(
                "SELECT isin FROM etfs WHERE isin = ?", (isin,)
            ).fetchone()

            extra_json = json.dumps(extra) if extra else None
            now = datetime.now().isoformat()

            if existing:
                # Update existing
                conn.execute("""
                    UPDATE etfs SET
                        name = COALESCE(?, name),
                        vwd_id = COALESCE(?, vwd_id),
                        exchange = COALESCE(?, exchange),
                        currency = COALESCE(?, currency),
                        ter = COALESCE(?, ter),
                        fund_size = COALESCE(?, fund_size),
                        distribution = COALESCE(?, distribution),
                        months_of_data = COALESCE(?, months_of_data),
                        extra_data = COALESCE(?, extra_data)
                    WHERE isin = ?
                """, (name, vwd_id, exchange, currency, ter, fund_size,
                      distribution, months_of_data, extra_json, isin))
                conn.commit()
                return False
            else:
                # Insert new
                conn.execute("""
                    INSERT INTO etfs (isin, name, vwd_id, exchange, currency, ter,
                                     fund_size, distribution, added_date, months_of_data, extra_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (isin, name, vwd_id, exchange, currency, ter, fund_size,
                      distribution, now, months_of_data, extra_json))
                conn.commit()
                return True
        finally:
            conn.close()

    def add_etfs_bulk(self, etf_data: List[Dict[str, Any]]) -> tuple:
        """
        Add multiple ETFs efficiently.

        Args:
            etf_data: List of dicts with ETF metadata

        Returns:
            Tuple of (added_count, updated_count)
        """
        added = 0
        updated = 0

        for etf in etf_data:
            isin = etf.pop('isin', None) or etf.pop('ISIN', None)
            if isin:
                if self.add_etf(isin, **etf):
                    added += 1
                else:
                    updated += 1

        return added, updated

    def remove_etf(self, isin: str) -> bool:
        """
        Remove an ETF and its price data.

        Args:
            isin: ISIN to remove

        Returns:
            True if removed, False if not found
        """
        self._check_writable()
        conn = self._get_connection()
        try:
            # Delete prices first (foreign key)
            conn.execute("DELETE FROM prices WHERE isin = ?", (isin,))

            # Delete ETF
            cursor = conn.execute("DELETE FROM etfs WHERE isin = ?", (isin,))
            conn.commit()

            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_etf(self, isin: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a single ETF."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM etfs WHERE isin = ?", (isin,)
            ).fetchone()

            if row:
                result = dict(row)
                if result.get('extra_data'):
                    result['extra'] = json.loads(result['extra_data'])
                    del result['extra_data']
                return result
            return None
        finally:
            conn.close()

    def load_universe(self) -> pd.DataFrame:
        """
        Load all ETF metadata as DataFrame.

        Returns:
            DataFrame with columns: isin, name, ter, fund_size, etc.
        """
        conn = self._get_connection()
        try:
            df = pd.read_sql_query("SELECT * FROM etfs ORDER BY name", conn)
            return df
        finally:
            conn.close()

    def list_isins(self) -> List[str]:
        """Get list of all ISINs in the universe."""
        conn = self._get_connection()
        try:
            rows = conn.execute("SELECT isin FROM etfs ORDER BY isin").fetchall()
            return [row['isin'] for row in rows]
        finally:
            conn.close()

    # ==================== Price Management ====================

    def update_prices(
        self,
        isin: str,
        prices: Union[pd.Series, pd.DataFrame],
        replace: bool = False
    ) -> int:
        """
        Update price data for an ETF.

        Args:
            isin: ISIN identifier
            prices: Price data (Series with DatetimeIndex or DataFrame with 'price' column)
            replace: If True, replace all existing prices. If False, only add new dates.

        Returns:
            Number of price records added/updated
        """
        self._check_writable()
        if prices is None or len(prices) == 0:
            return 0

        # Normalize to Series
        if isinstance(prices, pd.DataFrame):
            if 'price' in prices.columns:
                prices = prices['price']
            else:
                prices = prices.iloc[:, 0]

        conn = self._get_connection()
        try:
            if replace:
                # Delete existing prices
                conn.execute("DELETE FROM prices WHERE isin = ?", (isin,))

            # Prepare data for insert
            records = []
            for dt, price in prices.items():
                if pd.notna(price):
                    date_str = pd.Timestamp(dt).strftime('%Y-%m-%d')
                    records.append((isin, date_str, float(price)))

            # Insert with conflict handling
            conn.executemany("""
                INSERT OR REPLACE INTO prices (isin, date, price)
                VALUES (?, ?, ?)
            """, records)

            # Update last_price_update timestamp
            now = datetime.now().isoformat()
            conn.execute(
                "UPDATE etfs SET last_price_update = ? WHERE isin = ?",
                (now, isin)
            )

            conn.commit()
            return len(records)
        finally:
            conn.close()

    def load_prices(
        self,
        isin: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.Series:
        """
        Load price data for an ETF.

        Args:
            isin: ISIN identifier
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            Series with DatetimeIndex and price values
        """
        conn = self._get_connection()
        try:
            query = "SELECT date, price FROM prices WHERE isin = ?"
            params = [isin]

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY date"

            df = pd.read_sql_query(query, conn, params=params)

            if len(df) == 0:
                return pd.Series(dtype=float, name=isin)

            df['date'] = pd.to_datetime(df['date'])
            series = df.set_index('date')['price']
            series.name = isin
            return series
        finally:
            conn.close()

    def load_all_prices(
        self,
        isins: List[str] = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Load prices for multiple ETFs as a DataFrame.

        Args:
            isins: List of ISINs (None = all)
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with dates as index and ISINs as columns
        """
        if isins is None:
            isins = self.list_isins()

        prices_dict = {}
        for isin in isins:
            series = self.load_prices(isin, start_date, end_date)
            if len(series) > 0:
                prices_dict[isin] = series

        if not prices_dict:
            return pd.DataFrame()

        df = pd.DataFrame(prices_dict)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def get_latest_price_date(self, isin: str) -> Optional[str]:
        """Get the most recent price date for an ETF."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT MAX(date) as max_date FROM prices WHERE isin = ?",
                (isin,)
            ).fetchone()
            return row['max_date'] if row else None
        finally:
            conn.close()

    def get_price_date_range(self, isin: str) -> Optional[tuple]:
        """Get the date range of prices for an ETF."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT MIN(date) as min_date, MAX(date) as max_date FROM prices WHERE isin = ?",
                (isin,)
            ).fetchone()
            if row and row['min_date']:
                return (row['min_date'], row['max_date'])
            return None
        finally:
            conn.close()

    # ==================== Metadata History ====================

    def save_metadata_snapshot(self, snapshot_date: str = None):
        """
        Save current metadata (TER, fund_size, months_of_data) to history table.
        Call this before updating metadata to preserve historical values.

        Args:
            snapshot_date: Date for snapshot (default: today)
        """
        self._check_writable()
        if snapshot_date is None:
            snapshot_date = datetime.now().strftime('%Y-%m-%d')

        conn = self._get_connection()
        try:
            # Get current metadata for all ETFs
            rows = conn.execute("""
                SELECT isin, ter, fund_size, months_of_data FROM etfs
            """).fetchall()

            # Insert into history (ignore if already exists for this date)
            for row in rows:
                conn.execute("""
                    INSERT OR IGNORE INTO etf_metadata_history
                    (isin, snapshot_date, ter, fund_size, months_of_data)
                    VALUES (?, ?, ?, ?, ?)
                """, (row['isin'], snapshot_date, row['ter'], row['fund_size'], row['months_of_data']))

            conn.commit()
            return len(rows)
        finally:
            conn.close()

    def load_metadata_history(self, isin: str = None) -> pd.DataFrame:
        """
        Load metadata history for one or all ETFs.

        Args:
            isin: Specific ISIN or None for all

        Returns:
            DataFrame with columns: isin, snapshot_date, ter, fund_size, months_of_data
        """
        conn = self._get_connection()
        try:
            if isin:
                query = "SELECT * FROM etf_metadata_history WHERE isin = ? ORDER BY snapshot_date"
                df = pd.read_sql_query(query, conn, params=[isin])
            else:
                query = "SELECT * FROM etf_metadata_history ORDER BY isin, snapshot_date"
                df = pd.read_sql_query(query, conn)

            if len(df) > 0:
                df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
            return df
        finally:
            conn.close()

    def get_fund_size_history(self, isin: str) -> pd.Series:
        """Get fund size over time for an ETF."""
        df = self.load_metadata_history(isin)
        if len(df) == 0:
            return pd.Series(dtype=float)
        return df.set_index('snapshot_date')['fund_size']

    # ==================== Price Validation ====================

    def validate_new_prices(
        self,
        isin: str,
        new_prices: pd.Series,
        tolerance: float = 0.01
    ) -> Dict[str, Any]:
        """
        Validate new prices against existing data.
        Checks for overlap and consistency.

        Args:
            isin: ISIN to validate
            new_prices: New price series to validate
            tolerance: Max allowed price difference ratio (default 1%)

        Returns:
            Dict with validation results:
            - valid: bool
            - overlap_days: int
            - mismatches: list of (date, old_price, new_price)
            - new_start: first date in new data
            - new_end: last date in new data
            - old_end: last date in existing data
        """
        existing = self.load_prices(isin)

        result = {
            'valid': True,
            'overlap_days': 0,
            'mismatches': [],
            'new_start': new_prices.index.min() if len(new_prices) > 0 else None,
            'new_end': new_prices.index.max() if len(new_prices) > 0 else None,
            'old_start': existing.index.min() if len(existing) > 0 else None,
            'old_end': existing.index.max() if len(existing) > 0 else None,
        }

        if len(existing) == 0 or len(new_prices) == 0:
            return result

        # Find overlapping dates
        overlap_dates = existing.index.intersection(new_prices.index)
        result['overlap_days'] = len(overlap_dates)

        # Check price consistency in overlap
        for date in overlap_dates:
            old_price = existing[date]
            new_price = new_prices[date]
            if abs(old_price - new_price) / old_price > tolerance:
                result['mismatches'].append((date, old_price, new_price))

        if result['mismatches']:
            result['valid'] = False

        return result

    # ==================== Utility Methods ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = self._get_connection()
        try:
            etf_count = conn.execute("SELECT COUNT(*) FROM etfs").fetchone()[0]
            price_count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]

            # ETFs with prices
            etfs_with_prices = conn.execute("""
                SELECT COUNT(DISTINCT isin) FROM prices
            """).fetchone()[0]

            # Date range
            date_range = conn.execute("""
                SELECT MIN(date), MAX(date) FROM prices
            """).fetchone()

            return {
                'total_etfs': etf_count,
                'etfs_with_prices': etfs_with_prices,
                'total_price_records': price_count,
                'price_date_range': (date_range[0], date_range[1]) if date_range[0] else None,
                'db_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
            }
        finally:
            conn.close()

    def vacuum(self):
        """Optimize database file size (full rebuild)."""
        self._check_writable()
        conn = self._get_connection()
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()

    def optimize(self):
        """
        Run maintenance optimizations on the database.

        Performs:
        - WAL checkpoint (merge WAL file into main database)
        - Incremental vacuum (reclaim free pages)
        - Analyze (update query planner statistics)

        Call this periodically (e.g., after monthly data updates).
        This will also clean up .db-wal and .db-shm files.
        """
        self._check_writable()
        conn = self._get_connection()
        try:
            # Checkpoint WAL - merge changes into main database file
            # TRUNCATE mode: checkpoint and truncate WAL file to zero bytes
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

            # Incremental vacuum - reclaim free pages without full rebuild
            conn.execute("PRAGMA incremental_vacuum")

            # Update statistics for query optimizer
            conn.execute("ANALYZE")

            conn.commit()
        finally:
            conn.close()

    def validate(self) -> Dict[str, Any]:
        """
        Validate database integrity and data consistency.

        Returns:
            Dict with validation results:
            - integrity_ok: bool - SQLite integrity check passed
            - orphan_prices: int - prices without matching ETF record
            - etfs_without_prices: int - ETFs with no price data
            - duplicate_prices: int - duplicate (isin, date) entries
            - null_prices: int - NULL values in price column
            - issues: list of issue descriptions
        """
        conn = self._get_connection()
        try:
            issues = []

            # SQLite integrity check
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            integrity_ok = integrity == "ok"
            if not integrity_ok:
                issues.append(f"Integrity check failed: {integrity}")

            # Check for orphan prices (prices without ETF record)
            orphan_prices = conn.execute("""
                SELECT COUNT(*) FROM prices p
                WHERE NOT EXISTS (SELECT 1 FROM etfs e WHERE e.isin = p.isin)
            """).fetchone()[0]
            if orphan_prices > 0:
                issues.append(f"{orphan_prices} orphan price records (no matching ETF)")

            # Check for ETFs without prices
            etfs_without_prices = conn.execute("""
                SELECT COUNT(*) FROM etfs e
                WHERE NOT EXISTS (SELECT 1 FROM prices p WHERE p.isin = e.isin)
            """).fetchone()[0]
            if etfs_without_prices > 0:
                issues.append(f"{etfs_without_prices} ETFs have no price data")

            # Check for NULL prices (shouldn't happen with NOT NULL constraint)
            null_prices = conn.execute("""
                SELECT COUNT(*) FROM prices WHERE price IS NULL
            """).fetchone()[0]
            if null_prices > 0:
                issues.append(f"{null_prices} NULL price values found")

            # Check foreign key violations
            fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            if fk_violations:
                issues.append(f"{len(fk_violations)} foreign key violations")

            return {
                'integrity_ok': integrity_ok,
                'orphan_prices': orphan_prices,
                'etfs_without_prices': etfs_without_prices,
                'null_prices': null_prices,
                'foreign_key_violations': len(fk_violations) if fk_violations else 0,
                'issues': issues,
                'valid': len(issues) == 0
            }
        finally:
            conn.close()

    def __repr__(self):
        stats = self.get_stats()
        return f"ETFDatabase({stats['total_etfs']} ETFs, {stats['total_price_records']} prices)"


# Convenience function for quick access
_default_db = None

def get_database(db_path: str = "data/etf_database.db") -> ETFDatabase:
    """Get or create the default database instance."""
    global _default_db
    if _default_db is None or str(_default_db.db_path) != db_path:
        _default_db = ETFDatabase(db_path)
    return _default_db


if __name__ == "__main__":
    # Test the database
    print("ETF Database Test\n")

    db = ETFDatabase("data/etf_database_test.db", readonly=False)

    # Add test ETF
    db.add_etf(
        isin="IE00TEST1234",
        name="Test ETF",
        ter=0.20,
        fund_size=1000
    )

    # Add test prices
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    prices = pd.Series(100 + np.random.randn(len(dates)).cumsum(), index=dates)
    db.update_prices("IE00TEST1234", prices)

    # Load back
    loaded = db.load_prices("IE00TEST1234")
    print(f"Loaded {len(loaded)} prices")

    # Stats
    print(f"\nDatabase stats: {db.get_stats()}")

    # Cleanup test
    Path("data/etf_database_test.db").unlink()
    print("\nTest complete!")
