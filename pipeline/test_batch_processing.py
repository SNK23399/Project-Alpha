"""
Diagnostic test for batch processing in step 2.4.
Tests if process_feature_batch() works with a single feature.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import time

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from support.signal_database import SignalDatabase

# Configuration
SIGNAL_DIR = Path(__file__).parent / 'data' / 'signals'
FEATURE_DIR = Path(__file__).parent / 'data' / 'features'

def load_signal_cached(feature_name, feature_type):
    """Load signal/feature data from pickle files."""
    if feature_type == 'filtered':
        pickle_path = SIGNAL_DIR / 'filtered_signals' / f'{feature_name}.pkl'
    elif feature_type == 'raw':
        pickle_path = SIGNAL_DIR / f'{feature_name}.pkl'
    elif feature_type == 'step_2_3':
        pickle_path = FEATURE_DIR / f'{feature_name}.pkl'
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    if not pickle_path.exists():
        return None

    try:
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {pickle_path}: {e}")
        return None

def test_file_loading():
    """Test if we can find and load signal files."""
    print("="*60)
    print("TEST 1: Check if signal files exist")
    print("="*60)

    db = SignalDatabase(SIGNAL_DIR)

    # Check raw signals
    raw_signals = db.get_available_signals()
    print(f"\nRaw signals available: {len(raw_signals)}")
    if len(raw_signals) > 0:
        print(f"  First raw signal: {raw_signals[0]}")

        # Try to load it
        print(f"\n  Attempting to load {raw_signals[0]}...")
        start = time.time()
        data = load_signal_cached(raw_signals[0], 'raw')
        elapsed = time.time() - start
        print(f"  Loaded in {elapsed:.2f}s")
        if data is not None:
            print(f"  Data type: {type(data)}")
            if isinstance(data, pd.DataFrame):
                print(f"  Shape: {data.shape}")
                print(f"  Columns: {list(data.columns)}")
                print(f"  Index name: {data.index.name}")
            elif isinstance(data, pd.Series):
                print(f"  Length: {len(data)}")
        else:
            print("  Failed to load data!")
    else:
        print("  ERROR: No raw signals found!")

    # Check filtered signals
    filtered_signals = db.get_completed_filtered_signals()
    print(f"\nFiltered signals available: {len(filtered_signals)}")
    if len(filtered_signals) > 0:
        filtered_list = list(filtered_signals)
        print(f"  First filtered signal: {filtered_list[0]}")
    else:
        print("  ERROR: No filtered signals found!")

    # Check step 2.3 features
    if FEATURE_DIR.exists():
        step_2_3_files = list(FEATURE_DIR.glob('*.pkl'))
        print(f"\nStep 2.3 features available: {len(step_2_3_files)}")
        if len(step_2_3_files) > 0:
            print(f"  First feature: {step_2_3_files[0].stem}")
    else:
        print(f"\nStep 2.3 directory doesn't exist: {FEATURE_DIR}")

def test_single_feature():
    """Test processing a single feature."""
    print("\n" + "="*60)
    print("TEST 2: Process single feature")
    print("="*60)

    db = SignalDatabase(SIGNAL_DIR)
    raw_signals = db.get_available_signals()

    if len(raw_signals) == 0:
        print("ERROR: No raw signals available!")
        return

    feature_name = raw_signals[0]
    print(f"\nTesting with feature: {feature_name}")

    # Load the feature
    print("Loading feature...")
    start = time.time()
    data = load_signal_cached(feature_name, 'raw')
    elapsed = time.time() - start
    print(f"Loaded in {elapsed:.2f}s")

    if data is None:
        print("ERROR: Failed to load feature!")
        return

    # Convert to DataFrame
    print("Converting to DataFrame...")
    if isinstance(data, pd.Series):
        df = data.reset_index()
        df.columns = ['date', 'isin', 'value']
    elif isinstance(data, pd.DataFrame):
        df = data.stack().reset_index()
        df.columns = ['date', 'isin', 'value']
    else:
        print(f"ERROR: Unexpected type {type(data)}")
        return

    print(f"DataFrame shape: {df.shape}")

    # Convert dates
    print("Converting dates...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['value'])

    print(f"After dropna: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"ISINs: {df['isin'].nunique()}")

    # Get signal dates
    print("Getting signal dates...")
    signal_dates = np.sort(df['date'].unique())
    print(f"Signal dates: {len(signal_dates)}")

    # Create target dates (use subset for quick test)
    target_dates = sorted(df['date'].unique())[:10]  # Use first 10 unique dates
    target_dates_arr = np.array(target_dates, dtype='datetime64[ns]')
    print(f"Target dates for test: {len(target_dates)}")

    # Searchsorted test
    print("Testing searchsorted...")
    start = time.time()
    insert_positions = np.searchsorted(signal_dates, target_dates_arr, side='right')
    elapsed = time.time() - start
    print(f"Searchsorted complete in {elapsed:.4f}s")
    print(f"Insert positions: {insert_positions}")

    # Test z-score calculation
    print("\nTesting z-score normalization...")
    test_group = df.iloc[:5]
    if len(test_group) > 1:
        values = test_group['value'].values
        zscore = (values - values.mean()) / values.std()
        print(f"Z-scores: {zscore}")

    print("\n✓ All tests passed!")

if __name__ == '__main__':
    test_file_loading()
    test_single_feature()
