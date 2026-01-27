#!/usr/bin/env python3
"""
Quick test: Run pipeline against a single truncated database
to verify everything works before running full validation
"""

import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MAIN_DB_PATH = PROJECT_ROOT / 'maintenance' / 'data' / 'etf_database.db'
PIPELINE_SCRIPT = PROJECT_ROOT / 'main.py'
BACKTEST_OUTPUT_DIR = PROJECT_ROOT / 'pipeline' / 'data' / 'backtest_results'
DATABASES_DIR = SCRIPT_DIR / 'databases'

# Test with first validation date
TEST_DATE = '2018-02-28'
TEST_DB = DATABASES_DIR / TEST_DATE / 'etf_database.db'

print(f"\n{'='*100}")
print(f"SINGLE MONTH TEST - {TEST_DATE}")
print(f"{'='*100}\n")

# 1. Check database exists
print(f"Checking test database exists...")
if not TEST_DB.exists():
    print(f"  ERROR: {TEST_DB} not found")
    sys.exit(1)
print(f"  [OK] Found: {TEST_DB}")

# 2. Backup main database
print(f"\nBacking up main database...")
backup_path = MAIN_DB_PATH.parent / f'{MAIN_DB_PATH.name}.backup'
if MAIN_DB_PATH.exists():
    shutil.copy2(MAIN_DB_PATH, backup_path)
    print(f"  [OK] Backup: {backup_path}")

# 3. Swap in test database
print(f"\nSwapping in test database...")
if MAIN_DB_PATH.exists():
    MAIN_DB_PATH.unlink()
shutil.copy2(TEST_DB, MAIN_DB_PATH)
print(f"  [OK] Swapped: {MAIN_DB_PATH}")

# 4. Run pipeline
print(f"\nRunning pipeline (stages 1-6)...")
print(f"  Command: python {PIPELINE_SCRIPT} --steps 1,2,3,4,5,6")
print(f"  Working dir: {PROJECT_ROOT / 'pipeline'}")
print(f"-" * 100)

cmd = ['python', str(PIPELINE_SCRIPT), '--steps', '1,2,3,4,5,6']

try:
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=False,  # Show output directly
        text=True,
        timeout=600
    )

    print(f"-" * 100)

    if result.returncode == 0:
        print(f"\n[OK] Pipeline execution successful!")

        # Check output files
        print(f"\nChecking output files...")
        for csv_file in BACKTEST_OUTPUT_DIR.glob('bayesian_backtest_N*.csv'):
            if csv_file.is_file():
                size = csv_file.stat().st_size
                print(f"  [OK] {csv_file.name} ({size:,} bytes)")
    else:
        print(f"\n[ERROR] Pipeline execution failed with code {result.returncode}")

except subprocess.TimeoutExpired:
    print(f"\n[ERROR] Pipeline execution timed out after 10 minutes")
except Exception as e:
    print(f"\n[ERROR] Error running pipeline: {e}")

# 5. Restore main database
print(f"\nRestoring original database...")
if MAIN_DB_PATH.exists():
    MAIN_DB_PATH.unlink()
shutil.copy2(backup_path, MAIN_DB_PATH)
print(f"  [OK] Restored: {MAIN_DB_PATH}")

print(f"\n{'='*100}")
print(f"TEST COMPLETE")
print(f"{'='*100}\n")
