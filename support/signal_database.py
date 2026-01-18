"""
Signal Database - Parquet-based storage for incremental signal computation.

Uses parquet files for fast columnar storage:
- One parquet file per signal (293 files for signal bases)
- Each file: rows=dates, columns=ISINs, values=signal values
- ~10-50x faster writes than SQLite for analytical workloads
- Native pandas/numpy integration

Key features:
- Incremental updates: Only recalculate windows with new data
- State preservation: Filter states stored as JSON
- Efficient storage: Columnar compression with snappy
- Fast reads: Direct memory-mapping, no SQL parsing

Storage structure:
    data/signals/
    ├── signal_bases/           # 293 parquet files
    │   ├── ret_1d.parquet
    │   ├── vol_21d.parquet
    │   └── ...
    ├── filtered_signals/       # signal_name/filter_name.parquet
    │   ├── ret_1d/
    │   │   ├── ema_21d.parquet
    │   │   └── ...
    │   └── ...
    ├── filter_states.json      # All filter states
    └── metadata.json           # Computation log, stats

Usage:
    from signal_database import SignalDatabase

    db = SignalDatabase()

    # Save a single signal (fast!)
    db.save_signal_base('ret_1d', signal_df)

    # Load signals for inference
    signals = db.load_signal_bases(['ret_1d', 'vol_21d'], start_date='2024-01-01')
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Tuple

import pandas as pd
import numpy as np


class SignalDatabase:
    """Parquet-based database for signal bases and filtered signals."""

    def __init__(self, data_dir: str = "data/signals"):
        """
        Initialize signal database.

        Args:
            data_dir: Directory for parquet files
        """
        self.data_dir = Path(data_dir)
        self.signal_bases_dir = self.data_dir / "signal_bases"
        self.filtered_dir = self.data_dir / "filtered_signals"
        self.states_file = self.data_dir / "filter_states.json"
        self.metadata_file = self.data_dir / "metadata.json"

        # Create directories
        self.signal_bases_dir.mkdir(parents=True, exist_ok=True)
        self.filtered_dir.mkdir(parents=True, exist_ok=True)

    # ==================== Signal Bases ====================

    def save_signal_base(
        self,
        signal_name: str,
        signal_df: pd.DataFrame,
    ) -> int:
        """
        Save a single signal base to parquet file.

        Args:
            signal_name: Name of the signal
            signal_df: DataFrame with dates as index, ISINs as columns

        Returns:
            Number of records (non-NaN values)
        """
        # Ensure index is DatetimeIndex
        if not isinstance(signal_df.index, pd.DatetimeIndex):
            signal_df.index = pd.to_datetime(signal_df.index)

        # Save to parquet with snappy compression (fast + good compression)
        filepath = self.signal_bases_dir / f"{signal_name}.parquet"
        signal_df.to_parquet(filepath, compression='snappy')

        # Return count of non-NaN values
        return int((~signal_df.isna()).sum().sum())

    def save_signal_base_from_array(
        self,
        signal_name: str,
        signal_2d: np.ndarray,
        dates: pd.DatetimeIndex,
        isins: List[str]
    ) -> int:
        """
        Save a single signal from numpy array.

        Args:
            signal_name: Name of the signal
            signal_2d: 2D array (n_time, n_etfs)
            dates: DatetimeIndex for rows
            isins: List of ISIN column names

        Returns:
            Number of records (non-NaN values)
        """
        df = pd.DataFrame(signal_2d, index=dates, columns=isins)
        return self.save_signal_base(signal_name, df)

    def load_signal_base(
        self,
        signal_name: str,
        isins: List[str] = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Load a single signal base.

        Args:
            signal_name: Name of the signal
            isins: List of ISINs to load (None = all)
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            DataFrame with dates as index, ISINs as columns
        """
        filepath = self.signal_bases_dir / f"{signal_name}.parquet"
        if not filepath.exists():
            return pd.DataFrame()

        # Load with optional column filter
        columns = isins if isins else None
        df = pd.read_parquet(filepath, columns=columns)

        # Apply date filter
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        return df

    def load_signal_bases(
        self,
        signal_names: Union[str, List[str]],
        isins: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        as_3d_array: bool = False
    ) -> Union[Dict[str, pd.DataFrame], Tuple[np.ndarray, List[str], pd.DatetimeIndex, List[str]]]:
        """
        Load multiple signal bases.

        Args:
            signal_names: Single signal or list of signals
            isins: List of ISINs (None = all)
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            as_3d_array: If True, return (signals_3d, signal_names, dates, isins)
                        If False, return dict of DataFrames

        Returns:
            Dict of DataFrames or 3D array depending on as_3d_array flag
        """
        if isinstance(signal_names, str):
            signal_names = [signal_names]

        # Load each signal
        signals = {}
        for name in signal_names:
            df = self.load_signal_base(name, isins, start_date, end_date)
            if len(df) > 0:
                signals[name] = df

        if not signals:
            if as_3d_array:
                return np.array([]), [], pd.DatetimeIndex([]), []
            return {}

        if as_3d_array:
            # Stack into 3D array
            # First, align all signals to same dates/isins
            all_dates = sorted(set().union(*[set(df.index) for df in signals.values()]))
            all_isins = sorted(set().union(*[set(df.columns) for df in signals.values()]))

            n_signals = len(signals)
            n_time = len(all_dates)
            n_etfs = len(all_isins)

            signals_3d = np.full((n_signals, n_time, n_etfs), np.nan, dtype=np.float32)

            for sig_idx, (name, df) in enumerate(signals.items()):
                # Reindex to common dates/isins
                df_aligned = df.reindex(index=all_dates, columns=all_isins)
                signals_3d[sig_idx] = df_aligned.values

            return signals_3d, list(signals.keys()), pd.DatetimeIndex(all_dates), all_isins

        return signals

    def get_available_signals(self) -> List[str]:
        """Get list of available signal base names (sorted for determinism)."""
        return sorted([f.stem for f in self.signal_bases_dir.glob("*.parquet")])

    def get_completed_signals(self) -> set:
        """Get set of signal names already saved (for resume capability)."""
        return set(self.get_available_signals())

    def signal_exists(self, signal_name: str) -> bool:
        """Check if a signal base exists."""
        return (self.signal_bases_dir / f"{signal_name}.parquet").exists()

    def get_signal_date_range(self, signal_name: str) -> Optional[Tuple[str, str]]:
        """Get date range for a signal base."""
        df = self.load_signal_base(signal_name)
        if len(df) == 0:
            return None
        return (str(df.index.min().date()), str(df.index.max().date()))

    def delete_signal_base(self, signal_name: str) -> bool:
        """Delete a signal base file."""
        filepath = self.signal_bases_dir / f"{signal_name}.parquet"
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def clear_all_signal_bases(self):
        """Delete all signal base files."""
        for f in self.signal_bases_dir.glob("*.parquet"):
            f.unlink()

    # ==================== Filtered Signals ====================

    def save_filtered_signal(
        self,
        signal_name: str,
        filter_name: str,
        filtered_df: pd.DataFrame
    ) -> int:
        """
        Save a filtered signal to parquet.

        Args:
            signal_name: Base signal name
            filter_name: Filter name (e.g., 'ema_21d')
            filtered_df: DataFrame with dates as index, ISINs as columns

        Returns:
            Number of records (non-NaN values)
        """
        # Create subdirectory for this signal
        signal_dir = self.filtered_dir / signal_name
        signal_dir.mkdir(exist_ok=True)

        # Save parquet
        filepath = signal_dir / f"{filter_name}.parquet"
        filtered_df.to_parquet(filepath, compression='snappy')

        return int((~filtered_df.isna()).sum().sum())

    def load_filtered_signal(
        self,
        signal_name: str,
        filter_name: str,
        isins: List[str] = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Load a filtered signal.

        Args:
            signal_name: Base signal name
            filter_name: Filter name
            isins: List of ISINs to load (None = all)
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with dates as index, ISINs as columns
        """
        filepath = self.filtered_dir / signal_name / f"{filter_name}.parquet"
        if not filepath.exists():
            return pd.DataFrame()

        columns = isins if isins else None
        df = pd.read_parquet(filepath, columns=columns)

        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        return df

    def get_available_filters(self, signal_name: str) -> List[str]:
        """Get list of available filters for a signal."""
        signal_dir = self.filtered_dir / signal_name
        if not signal_dir.exists():
            return []
        return [f.stem for f in signal_dir.glob("*.parquet")]

    def save_filtered_signal_from_array(
        self,
        filtered_signal_name: str,
        filtered_data: np.ndarray,
        dates: pd.DatetimeIndex,
        isins: List[str]
    ) -> int:
        """
        Save a filtered signal from numpy array.

        The filtered_signal_name format is: {base_signal}__{filter_name}
        Example: "momentum_252d__ema_21d"

        Args:
            filtered_signal_name: Full filtered signal name (signal__filter)
            filtered_data: 2D array (n_time, n_etfs)
            dates: DatetimeIndex for rows
            isins: List of ISIN column names

        Returns:
            Number of records (non-NaN values)
        """
        # Save as flat parquet file (not nested by signal name)
        filepath = self.filtered_dir / f"{filtered_signal_name}.parquet"

        # Create DataFrame
        df = pd.DataFrame(filtered_data, index=dates, columns=isins)

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Save to parquet
        df.to_parquet(filepath, compression='snappy')

        return int((~df.isna()).sum().sum())

    def load_filtered_signal_by_name(
        self,
        filtered_signal_name: str,
        isins: List[str] = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Load a filtered signal by its full name.

        Args:
            filtered_signal_name: Full name like "momentum_252d__ema_21d"
            isins: List of ISINs to load (None = all)
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with dates as index, ISINs as columns
        """
        filepath = self.filtered_dir / f"{filtered_signal_name}.parquet"
        if not filepath.exists():
            return pd.DataFrame()

        columns = isins if isins else None
        df = pd.read_parquet(filepath, columns=columns)

        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        return df

    def get_completed_filtered_signals(self) -> set:
        """
        Get set of completed filtered signal names (for resume capability).

        Returns:
            Set of filtered signal names like {'momentum_252d__ema_21d', ...}
        """
        return {f.stem for f in self.filtered_dir.glob("*.parquet")}

    def filtered_signal_exists(self, filtered_signal_name: str) -> bool:
        """Check if a filtered signal exists."""
        return (self.filtered_dir / f"{filtered_signal_name}.parquet").exists()

    def delete_filtered_signal(self, filtered_signal_name: str) -> bool:
        """Delete a filtered signal file."""
        filepath = self.filtered_dir / f"{filtered_signal_name}.parquet"
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def clear_all_filtered_signals(self):
        """Delete all filtered signal files."""
        for f in self.filtered_dir.glob("*.parquet"):
            f.unlink()

    # ==================== Filter States ====================

    def save_filter_state(
        self,
        signal_name: str,
        filter_name: str,
        states: Dict[str, Any]
    ):
        """
        Save filter states for incremental updates.

        Args:
            signal_name: Base signal name
            filter_name: Filter name
            states: Dict mapping ISIN -> state dict
        """
        # Load existing states
        all_states = self._load_all_states()

        # Update this signal/filter's states
        key = f"{signal_name}:{filter_name}"
        all_states[key] = {
            'states': states,
            'last_update': datetime.now().isoformat()
        }

        # Save back
        self._save_all_states(all_states)

    def load_filter_state(
        self,
        signal_name: str,
        filter_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load filter states for incremental updates.

        Returns:
            Dict with 'states' (ISIN -> state) and 'last_update', or None
        """
        all_states = self._load_all_states()
        key = f"{signal_name}:{filter_name}"
        return all_states.get(key)

    def _load_all_states(self) -> Dict:
        """Load all filter states from JSON."""
        if self.states_file.exists():
            with open(self.states_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_all_states(self, states: Dict):
        """Save all filter states to JSON."""
        with open(self.states_file, 'w') as f:
            json.dump(states, f)

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
    ):
        """Log a computation for tracking."""
        metadata = self._load_metadata()

        if 'computation_log' not in metadata:
            metadata['computation_log'] = []

        metadata['computation_log'].append({
            'computation_type': computation_type,
            'computation_date': datetime.now().isoformat(),
            'start_date': start_date,
            'end_date': end_date,
            'n_etfs': n_etfs,
            'n_signals': n_signals,
            'computation_time_seconds': computation_time,
            'notes': notes
        })

        # Keep only last 100 entries
        metadata['computation_log'] = metadata['computation_log'][-100:]

        self._save_metadata(metadata)

    def get_computation_log(self, limit: int = 20) -> pd.DataFrame:
        """Get recent computation log entries."""
        metadata = self._load_metadata()
        log = metadata.get('computation_log', [])
        df = pd.DataFrame(log[-limit:])
        if len(df) > 0 and 'computation_date' in df.columns:
            df['computation_date'] = pd.to_datetime(df['computation_date'])
        return df

    def _load_metadata(self) -> Dict:
        """Load metadata from JSON."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self, metadata: Dict):
        """Save metadata to JSON."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    # ==================== Statistics ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        # Count signal bases
        signal_files = list(self.signal_bases_dir.glob("*.parquet"))
        n_signals = len(signal_files)

        # Count filtered signals
        n_filters = 0
        n_filtered_files = 0
        for signal_dir in self.filtered_dir.iterdir():
            if signal_dir.is_dir():
                filters = list(signal_dir.glob("*.parquet"))
                n_filtered_files += len(filters)
                if filters:
                    n_filters = max(n_filters, len(filters))

        # Get date range from first signal
        date_range = None
        if signal_files:
            df = pd.read_parquet(signal_files[0])
            if len(df) > 0:
                date_range = (str(df.index.min().date()), str(df.index.max().date()))

        # Calculate total size
        total_size = sum(f.stat().st_size for f in self.data_dir.rglob("*") if f.is_file())

        return {
            'unique_signal_bases': n_signals,
            'unique_filters': n_filters,
            'filtered_signal_files': n_filtered_files,
            'signal_date_range': date_range,
            'total_size_mb': total_size / (1024 * 1024)
        }

    def __repr__(self):
        stats = self.get_stats()
        return (f"SignalDatabase({stats['unique_signal_bases']} signals, "
                f"{stats['unique_filters']} filters, "
                f"{stats['total_size_mb']:.1f} MB)")


# Convenience function for quick access
_default_signal_db = None


def get_signal_database(data_dir: str = "data/signals") -> SignalDatabase:
    """Get or create the default signal database instance."""
    global _default_signal_db
    if _default_signal_db is None or str(_default_signal_db.data_dir) != data_dir:
        _default_signal_db = SignalDatabase(data_dir)
    return _default_signal_db


if __name__ == "__main__":
    # Test the database
    print("Signal Database Test (Parquet-based)\n")

    import time

    db = SignalDatabase("data/signals_test")

    # Test signal bases
    print("Testing signal bases...")
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    isins = [f"IE00TEST{i:04d}" for i in range(100)]  # 100 ETFs
    signal_names = ["ret_1d", "vol_21d", "momentum_63d"]

    # Create test data
    for signal_name in signal_names:
        df = pd.DataFrame(
            np.random.randn(len(dates), len(isins)),
            index=dates,
            columns=isins
        )

        start = time.time()
        n_records = db.save_signal_base(signal_name, df)
        elapsed = time.time() - start
        print(f"  {signal_name}: {n_records:,} records in {elapsed:.3f}s")

    # Load back
    print("\nLoading signals...")
    start = time.time()
    signals = db.load_signal_bases(signal_names, as_3d_array=True)
    elapsed = time.time() - start
    signals_3d, loaded_names, loaded_dates, loaded_isins = signals
    print(f"  Loaded shape: {signals_3d.shape} in {elapsed:.3f}s")

    # Test filtered signals
    print("\nTesting filtered signals...")
    filtered_df = pd.DataFrame(
        np.random.randn(len(dates), len(isins)),
        index=dates,
        columns=isins
    )
    n_records = db.save_filtered_signal("ret_1d", "ema_21d", filtered_df)
    print(f"  Saved {n_records:,} filtered records")

    # Stats
    print(f"\nDatabase stats:")
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Cleanup test
    import shutil
    shutil.rmtree("data/signals_test")
    print("\nTest complete!")
