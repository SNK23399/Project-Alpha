"""
Direct test of pickle file loading - no abstraction layers.
"""

import pickle
from pathlib import Path

SIGNAL_DIR = Path(__file__).parent / 'data' / 'signals'

# Test 1: Check if raw signal directory exists
print("TEST 1: Directory structure")
print(f"Signal dir: {SIGNAL_DIR}")
print(f"  Exists: {SIGNAL_DIR.exists()}")

raw_pkl_files = list(SIGNAL_DIR.glob('*.pkl'))
print(f"  Raw signal .pkl files: {len(raw_pkl_files)}")
if len(raw_pkl_files) > 0:
    print(f"    First 5:")
    for f in raw_pkl_files[:5]:
        print(f"      {f.name} ({f.stat().st_size / 1024:.1f} KB)")

filtered_pkl_dir = SIGNAL_DIR / 'filtered_signals'
print(f"\nFiltered signals dir: {filtered_pkl_dir}")
print(f"  Exists: {filtered_pkl_dir.exists()}")

if filtered_pkl_dir.exists():
    filtered_pkl_files = list(filtered_pkl_dir.glob('*.pkl'))
    print(f"  Filtered signal .pkl files: {len(filtered_pkl_files)}")
    if len(filtered_pkl_files) > 0:
        print(f"    First 5:")
        for f in filtered_pkl_files[:5]:
            print(f"      {f.name} ({f.stat().st_size / 1024:.1f} KB)")

# Test 2: Try to load a raw signal directly
print("\n" + "="*60)
print("TEST 2: Load raw signal 'alpha'")
print("="*60)

alpha_file = SIGNAL_DIR / 'alpha.pkl'
print(f"\nFile path: {alpha_file}")
print(f"  Exists: {alpha_file.exists()}")

if alpha_file.exists():
    print(f"  Size: {alpha_file.stat().st_size / 1024:.1f} KB")

    try:
        print("\n  Attempting to load...")
        with open(alpha_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  ✓ SUCCESS")
        print(f"  Type: {type(data)}")
        if hasattr(data, 'shape'):
            print(f"  Shape: {data.shape}")
        if hasattr(data, 'columns'):
            print(f"  Columns: {list(data.columns)}")
        if hasattr(data, 'index'):
            print(f"  Index type: {type(data.index)}")
    except Exception as e:
        print(f"  ✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  File does not exist!")

# Test 3: List and load all raw signals
print("\n" + "="*60)
print("TEST 3: Load all raw signals")
print("="*60)

if len(raw_pkl_files) > 0:
    for i, pkl_file in enumerate(raw_pkl_files[:5]):  # Test first 5
        print(f"\n[{i+1}] {pkl_file.name} ({pkl_file.stat().st_size / 1024:.1f} KB)")
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            print(f"    ✓ Loaded ({type(data).__name__})")
        except Exception as e:
            print(f"    ✗ FAILED: {type(e).__name__}: {str(e)[:100]}")
