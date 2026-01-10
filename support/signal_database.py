"""
Signal Database - SQLite-based storage for incremental signal computation.

Extends ETFDatabase to store:
- Signal bases (293 raw signals)
- Filtered signals (signal bases × 27 filters = 7,325+ variants)
- Filter states (for incremental updates)

Key features:
- Incremental updates: Only recalculate windows with new data
- State preservation: Filters resume from last state (EMA, Kalman, etc.)
- Efficient storage: Same SQLite approach as ETF price data
- Validation: Verify incremental matches full calculation

Usage:
    from signal_database import SignalDatabase

    db = SignalDatabase()

    # Compute and store signal bases
    signals_3d, signal_names = compute_all_signal_bases(etf_prices, core_prices)
    db.update_signal_bases(signals_3d, signal_names, dates, isins)

    # Apply filters and store
    filtered = apply_filter(signal_data, filter_name='ema_21d', state=last_state)
    db.update_filtered_signals(signal_name, filter_name, filtered, dates, isins, new_state)

    # Load signals for inference
    signals = db.load_signal_bases(['ret_1d', 'vol_21d'], start_date='2024-01-01')
    filtered = db.load_filtered_signals('ret_1d', 'ema_21d', start_date='2024-01-01')
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Tuple

import pandas as pd
import numpy as np


class SignalDatabase:
    """SQLite database for signal bases, filtered signals, and filter states."""

    def __init__(self, db_path: str = "data/etf_database.db"):
        """
        Initialize database connection.

        Uses same database as ETFDatabase for consistency.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with optimized settings."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Optimize for bulk inserts
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        # Increase cache size for better performance
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        return conn

    def _init_schema(self):
        """Initialize database schema for signals."""
        conn = self._get_connection()
        try:
            # Signal bases table (raw computed signals)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_bases (
                    signal_name TEXT,
                    isin TEXT,
                    date TEXT,
                    value REAL,
                    PRIMARY KEY (signal_name, isin, date)
                )
            """)

            # Filtered signals table (signal bases after applying filters)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS filtered_signals (
                    signal_name TEXT,
                    filter_name TEXT,
                    isin TEXT,
                    date TEXT,
                    value REAL,
                    PRIMARY KEY (signal_name, filter_name, isin, date)
                )
            """)

            # Filter states table (for incremental updates)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS filter_states (
                    signal_name TEXT,
                    filter_name TEXT,
                    isin TEXT,
                    last_update TEXT,
                    state_json TEXT,
                    PRIMARY KEY (signal_name, filter_name, isin)
                )
            """)

            # Computation metadata (track when signals were last computed)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS computation_log (
                    computation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    computation_type TEXT,
                    computation_date TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    n_etfs INTEGER,
                    n_signals INTEGER,
                    computation_time_seconds REAL,
                    notes TEXT
                )
            """)

            # Indexes for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signal_bases_name_date
                ON signal_bases(signal_name, date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signal_bases_isin_date
                ON signal_bases(isin, date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_filtered_signals_name_filter_date
                ON filtered_signals(signal_name, filter_name, date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_filtered_signals_isin_date
                ON filtered_signals(isin, date)
            """)

            conn.commit()
        finally:
            conn.close()

    # ==================== Signal Bases ====================

    def update_signal_bases(
        self,
        signals_3d: np.ndarray,
        signal_names: List[str],
        dates: pd.DatetimeIndex,
        isins: List[str],
        replace: bool = False
    ) -> int:
        """
        Update signal base data.

        Args:
            signals_3d: (n_signals, n_time, n_etfs) array
            signal_names: List of signal names
            dates: DatetimeIndex for time axis
            isins: List of ISINs for ETF axis
            replace: If True, replace existing data. If False, only add new.

        Returns:
            Number of records inserted/updated
        """
        n_signals, n_time, n_etfs = signals_3d.shape

        if len(signal_names) != n_signals:
            raise ValueError(f"Signal names ({len(signal_names)}) doesn't match array ({n_signals})")
        if len(dates) != n_time:
            raise ValueError(f"Dates ({len(dates)}) doesn't match array ({n_time})")
        if len(isins) != n_etfs:
            raise ValueError(f"ISINs ({len(isins)}) doesn't match array ({n_etfs})")

        conn = self._get_connection()
        try:
            # Process ONE SIGNAL AT A TIME to avoid memory issues
            dates_arr = dates.strftime('%Y-%m-%d').values
            total_records = 0

            if replace:
                # Delete existing records for these signals
                signal_placeholders = ','.join(['?'] * len(signal_names))
                conn.execute(
                    f"DELETE FROM signal_bases WHERE signal_name IN ({signal_placeholders})",
                    signal_names
                )
                conn.commit()

            for sig_idx, signal_name in enumerate(signal_names):
                # Get this signal's 2D array (n_time, n_etfs)
                signal_2d = signals_3d[sig_idx]

                # Build records for this signal only
                records = []
                for time_idx, date_str in enumerate(dates_arr):
                    for etf_idx, isin in enumerate(isins):
                        value = signal_2d[time_idx, etf_idx]
                        if not np.isnan(value):
                            records.append((signal_name, isin, date_str, float(value)))

                        # Batch insert every 50000 records
                        if len(records) >= 50000:
                            conn.executemany("""
                                INSERT OR REPLACE INTO signal_bases (signal_name, isin, date, value)
                                VALUES (?, ?, ?, ?)
                            """, records)
                            total_records += len(records)
                            records = []

                # Insert remaining records for this signal
                if records:
                    conn.executemany("""
                        INSERT OR REPLACE INTO signal_bases (signal_name, isin, date, value)
                        VALUES (?, ?, ?, ?)
                    """, records)
                    total_records += len(records)

                # Commit after each signal and print progress
                conn.commit()
                if (sig_idx + 1) % 10 == 0:
                    print(f"    [Saved {sig_idx + 1}/{n_signals} signals...]")

            print(f"    [Saved all {n_signals} signals to database]")
            return total_records
        finally:
            conn.close()

    def save_single_signal(
        self,
        signal_name: str,
        signal_df: 'pd.DataFrame',
        replace: bool = True
    ) -> int:
        """
        Save a single signal to the database (memory-efficient).

        Args:
            signal_name: Name of the signal
            signal_df: DataFrame with dates as index, ISINs as columns
            replace: If True, delete existing records first

        Returns:
            Number of records saved
        """
        conn = self._get_connection()
        try:
            if replace:
                conn.execute("DELETE FROM signal_bases WHERE signal_name = ?", (signal_name,))

            # Build records one date at a time (memory efficient)
            records = []
            dates_arr = signal_df.index.strftime('%Y-%m-%d').values
            isins = signal_df.columns.tolist()

            for time_idx, date_str in enumerate(dates_arr):
                row = signal_df.iloc[time_idx]
                for etf_idx, isin in enumerate(isins):
                    value = row.iloc[etf_idx]
                    if not np.isnan(value):
                        records.append((signal_name, isin, date_str, float(value)))

                # Batch insert every 10000 records to avoid memory buildup
                if len(records) >= 10000:
                    conn.executemany("""
                        INSERT OR REPLACE INTO signal_bases (signal_name, isin, date, value)
                        VALUES (?, ?, ?, ?)
                    """, records)
                    records = []

            # Insert remaining records
            if records:
                conn.executemany("""
                    INSERT OR REPLACE INTO signal_bases (signal_name, isin, date, value)
                    VALUES (?, ?, ?, ?)
                """, records)

            conn.commit()
            return len(dates_arr) * len(isins)  # Approximate count
        finally:
            conn.close()

    def load_signal_bases(
        self,
        signal_names: Union[str, List[str]],
        isins: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        as_3d_array: bool = False
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, List[str], pd.DatetimeIndex, List[str]]]:
        """
        Load signal base data.

        Args:
            signal_names: Single signal or list of signals
            isins: List of ISINs (None = all)
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            as_3d_array: If True, return (signals_3d, signal_names, dates, isins)
                        If False, return DataFrame with MultiIndex

        Returns:
            DataFrame or 3D array depending on as_3d_array flag
        """
        if isinstance(signal_names, str):
            signal_names = [signal_names]

        conn = self._get_connection()
        try:
            # Build query
            query = "SELECT signal_name, isin, date, value FROM signal_bases WHERE signal_name IN ({})".format(
                ','.join(['?'] * len(signal_names))
            )
            params = list(signal_names)

            if isins:
                query += " AND isin IN ({})".format(','.join(['?'] * len(isins)))
                params.extend(isins)
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY signal_name, date, isin"

            df = pd.read_sql_query(query, conn, params=params)

            if len(df) == 0:
                if as_3d_array:
                    return np.array([]), [], pd.DatetimeIndex([]), []
                return pd.DataFrame()

            df['date'] = pd.to_datetime(df['date'])

            if as_3d_array:
                # Convert to 3D array
                unique_signals = df['signal_name'].unique()
                unique_dates = df['date'].unique()
                unique_isins = df['isin'].unique()

                n_signals = len(unique_signals)
                n_time = len(unique_dates)
                n_etfs = len(unique_isins)

                signals_3d = np.full((n_signals, n_time, n_etfs), np.nan, dtype=np.float32)

                # Create lookup dictionaries
                signal_idx = {sig: i for i, sig in enumerate(unique_signals)}
                date_idx = {date: i for i, date in enumerate(unique_dates)}
                isin_idx = {isin: i for i, isin in enumerate(unique_isins)}

                # Fill array
                for _, row in df.iterrows():
                    sig_i = signal_idx[row['signal_name']]
                    time_i = date_idx[row['date']]
                    etf_i = isin_idx[row['isin']]
                    signals_3d[sig_i, time_i, etf_i] = row['value']

                return signals_3d, list(unique_signals), pd.DatetimeIndex(unique_dates), list(unique_isins)
            else:
                # Return as DataFrame with MultiIndex
                df = df.set_index(['signal_name', 'isin', 'date'])
                return df['value'].unstack('isin')
        finally:
            conn.close()

    def get_signal_base_date_range(
        self,
        signal_name: str,
        isin: str = None
    ) -> Optional[Tuple[str, str]]:
        """Get date range for a signal base."""
        conn = self._get_connection()
        try:
            if isin:
                query = """
                    SELECT MIN(date) as min_date, MAX(date) as max_date
                    FROM signal_bases
                    WHERE signal_name = ? AND isin = ?
                """
                params = [signal_name, isin]
            else:
                query = """
                    SELECT MIN(date) as min_date, MAX(date) as max_date
                    FROM signal_bases
                    WHERE signal_name = ?
                """
                params = [signal_name]

            row = conn.execute(query, params).fetchone()
            if row and row['min_date']:
                return (row['min_date'], row['max_date'])
            return None
        finally:
            conn.close()

    # ==================== Filtered Signals ====================

    def update_filtered_signals(
        self,
        signal_name: str,
        filter_name: str,
        filtered_data: np.ndarray,
        dates: pd.DatetimeIndex,
        isins: List[str],
        filter_state: Dict[str, Any] = None,
        replace: bool = False
    ) -> int:
        """
        Update filtered signal data and optionally save filter state.

        Args:
            signal_name: Base signal name
            filter_name: Filter name (e.g., 'ema_21d')
            filtered_data: (n_time, n_etfs) array or (1, n_time, n_etfs)
            dates: DatetimeIndex
            isins: List of ISINs
            filter_state: Optional dict with filter state to save
            replace: If True, replace existing data

        Returns:
            Number of records inserted
        """
        # Handle 3D array with single signal
        if filtered_data.ndim == 3:
            if filtered_data.shape[0] == 1:
                filtered_data = filtered_data[0]
            else:
                raise ValueError("filtered_data must be 2D or 3D with first dim=1")

        n_time, n_etfs = filtered_data.shape

        if len(dates) != n_time:
            raise ValueError(f"Dates ({len(dates)}) doesn't match array ({n_time})")
        if len(isins) != n_etfs:
            raise ValueError(f"ISINs ({len(isins)}) doesn't match array ({n_etfs})")

        conn = self._get_connection()
        try:
            records = []

            for time_idx, date in enumerate(dates):
                date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
                for etf_idx, isin in enumerate(isins):
                    value = float(filtered_data[time_idx, etf_idx])
                    if not np.isnan(value):
                        records.append((signal_name, filter_name, isin, date_str, value))

            if replace:
                # Delete existing records
                conn.execute("""
                    DELETE FROM filtered_signals
                    WHERE signal_name = ? AND filter_name = ?
                """, (signal_name, filter_name))

            # Batch insert
            conn.executemany("""
                INSERT OR REPLACE INTO filtered_signals
                (signal_name, filter_name, isin, date, value)
                VALUES (?, ?, ?, ?, ?)
            """, records)

            # Update filter states if provided
            if filter_state:
                now = datetime.now().isoformat()
                state_json = json.dumps(filter_state)

                for isin in isins:
                    # Get isin-specific state if available
                    if isinstance(filter_state, dict) and isin in filter_state:
                        isin_state = filter_state[isin]
                        state_json = json.dumps(isin_state)

                    conn.execute("""
                        INSERT OR REPLACE INTO filter_states
                        (signal_name, filter_name, isin, last_update, state_json)
                        VALUES (?, ?, ?, ?, ?)
                    """, (signal_name, filter_name, isin, now, state_json))

            conn.commit()
            return len(records)
        finally:
            conn.close()

    def load_filtered_signals(
        self,
        signal_name: str,
        filter_name: str,
        isins: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        as_2d_array: bool = False
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, pd.DatetimeIndex, List[str]]]:
        """
        Load filtered signal data.

        Args:
            signal_name: Base signal name
            filter_name: Filter name
            isins: List of ISINs (None = all)
            start_date: Optional start date
            end_date: Optional end date
            as_2d_array: If True, return (data_2d, dates, isins)
                        If False, return DataFrame

        Returns:
            DataFrame or 2D array depending on as_2d_array flag
        """
        conn = self._get_connection()
        try:
            query = """
                SELECT isin, date, value FROM filtered_signals
                WHERE signal_name = ? AND filter_name = ?
            """
            params = [signal_name, filter_name]

            if isins:
                query += " AND isin IN ({})".format(','.join(['?'] * len(isins)))
                params.extend(isins)
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY date, isin"

            df = pd.read_sql_query(query, conn, params=params)

            if len(df) == 0:
                if as_2d_array:
                    return np.array([]), pd.DatetimeIndex([]), []
                return pd.DataFrame()

            df['date'] = pd.to_datetime(df['date'])

            if as_2d_array:
                # Convert to 2D array
                pivot = df.pivot(index='date', columns='isin', values='value')
                return pivot.values, pivot.index, list(pivot.columns)
            else:
                # Return as DataFrame
                return df.pivot(index='date', columns='isin', values='value')
        finally:
            conn.close()

    def load_filter_state(
        self,
        signal_name: str,
        filter_name: str,
        isin: str
    ) -> Optional[Dict[str, Any]]:
        """Load filter state for incremental updates."""
        conn = self._get_connection()
        try:
            row = conn.execute("""
                SELECT state_json, last_update FROM filter_states
                WHERE signal_name = ? AND filter_name = ? AND isin = ?
            """, (signal_name, filter_name, isin)).fetchone()

            if row and row['state_json']:
                return {
                    'state': json.loads(row['state_json']),
                    'last_update': row['last_update']
                }
            return None
        finally:
            conn.close()

    def load_all_filter_states(
        self,
        signal_name: str,
        filter_name: str,
        isins: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Load filter states for all ISINs."""
        conn = self._get_connection()
        try:
            query = """
                SELECT isin, state_json, last_update FROM filter_states
                WHERE signal_name = ? AND filter_name = ?
            """
            params = [signal_name, filter_name]

            if isins:
                query += " AND isin IN ({})".format(','.join(['?'] * len(isins)))
                params.extend(isins)

            rows = conn.execute(query, params).fetchall()

            states = {}
            for row in rows:
                states[row['isin']] = {
                    'state': json.loads(row['state_json']),
                    'last_update': row['last_update']
                }
            return states
        finally:
            conn.close()

    # ==================== Computation Logging ====================

    def log_computation(
        self,
        computation_type: str,
        start_date: str,
        end_date: str,
        n_etfs: int,
        n_signals: int,
        computation_time: float,
        notes: str = None
    ) -> int:
        """Log a computation for tracking."""
        conn = self._get_connection()
        try:
            now = datetime.now().isoformat()
            cursor = conn.execute("""
                INSERT INTO computation_log
                (computation_type, computation_date, start_date, end_date,
                 n_etfs, n_signals, computation_time_seconds, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (computation_type, now, start_date, end_date,
                  n_etfs, n_signals, computation_time, notes))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_computation_log(self, limit: int = 20) -> pd.DataFrame:
        """Get recent computation log entries."""
        conn = self._get_connection()
        try:
            query = """
                SELECT * FROM computation_log
                ORDER BY computation_date DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=[limit])
            if len(df) > 0:
                df['computation_date'] = pd.to_datetime(df['computation_date'])
            return df
        finally:
            conn.close()

    # ==================== Statistics ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = self._get_connection()
        try:
            signal_base_count = conn.execute(
                "SELECT COUNT(*) FROM signal_bases"
            ).fetchone()[0]

            filtered_signal_count = conn.execute(
                "SELECT COUNT(*) FROM filtered_signals"
            ).fetchone()[0]

            filter_state_count = conn.execute(
                "SELECT COUNT(*) FROM filter_states"
            ).fetchone()[0]

            unique_signals = conn.execute(
                "SELECT COUNT(DISTINCT signal_name) FROM signal_bases"
            ).fetchone()[0]

            unique_filters = conn.execute(
                "SELECT COUNT(DISTINCT filter_name) FROM filtered_signals"
            ).fetchone()[0]

            signal_date_range = conn.execute("""
                SELECT MIN(date), MAX(date) FROM signal_bases
            """).fetchone()

            return {
                'signal_base_records': signal_base_count,
                'filtered_signal_records': filtered_signal_count,
                'filter_states': filter_state_count,
                'unique_signal_bases': unique_signals,
                'unique_filters': unique_filters,
                'signal_date_range': (signal_date_range[0], signal_date_range[1])
                                    if signal_date_range[0] else None,
                'db_size_mb': self.db_path.stat().st_size / (1024 * 1024)
                             if self.db_path.exists() else 0
            }
        finally:
            conn.close()

    def vacuum(self):
        """Optimize database file size."""
        conn = self._get_connection()
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()

    def __repr__(self):
        stats = self.get_stats()
        return (f"SignalDatabase({stats['unique_signal_bases']} signals, "
                f"{stats['unique_filters']} filters, "
                f"{stats['signal_base_records']:,} records)")


# Convenience function for quick access
_default_signal_db = None

def get_signal_database(db_path: str = "data/etf_database.db") -> SignalDatabase:
    """Get or create the default signal database instance."""
    global _default_signal_db
    if _default_signal_db is None or str(_default_signal_db.db_path) != db_path:
        _default_signal_db = SignalDatabase(db_path)
    return _default_signal_db


if __name__ == "__main__":
    # Test the database
    print("Signal Database Test\n")

    db = SignalDatabase("data/signal_database_test.db")

    # Test signal bases
    print("Testing signal bases...")
    dates = pd.date_range("2024-01-01", "2024-01-31", freq="D")
    isins = ["IE00TEST1", "IE00TEST2"]
    signal_names = ["ret_1d", "vol_21d"]

    # Create test data: (2 signals, 31 days, 2 ETFs)
    signals_3d = np.random.randn(2, 31, 2).astype(np.float32)

    n_records = db.update_signal_bases(signals_3d, signal_names, dates, isins)
    print(f"  Inserted {n_records} records")

    # Load back
    loaded = db.load_signal_bases(signal_names, isins, as_3d_array=True)
    loaded_3d, loaded_names, loaded_dates, loaded_isins = loaded
    print(f"  Loaded shape: {loaded_3d.shape}")

    # Test filtered signals
    print("\nTesting filtered signals...")
    filtered_data = signals_3d[0]  # Take first signal (31, 2)

    # Create test state
    filter_state = {
        "IE00TEST1": {"last_value": 0.001234, "alpha": 0.0909},
        "IE00TEST2": {"last_value": 0.002345, "alpha": 0.0909}
    }

    n_records = db.update_filtered_signals(
        "ret_1d", "ema_21d", filtered_data, dates, isins, filter_state
    )
    print(f"  Inserted {n_records} records")

    # Load filter state
    state = db.load_filter_state("ret_1d", "ema_21d", "IE00TEST1")
    print(f"  Loaded state: {state}")

    # Stats
    print(f"\nDatabase stats:")
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Cleanup test
    Path("data/signal_database_test.db").unlink()
    print("\nTest complete!")
