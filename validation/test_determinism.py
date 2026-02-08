#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Determinism Test for Pipeline Steps 1-6

Runs each step 5 times and compares outputs to verify deterministic behavior.
"""

import sys
import subprocess
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Paths
project_root = Path(__file__).parent
pipeline_dir = project_root / "pipeline"
data_dir = pipeline_dir / "data"
backup_dir = project_root / "determinism_test"

# Files to compare
COMPARISON_FILES = {
    1: ["forward_alpha_1month.parquet"],
    2: ["rankings_matrix_signal_bases_1month.npz"],
    3: ["rankings_matrix_filtered_1month.npz"],
    4: ["feature_ir_1month.npz"],
    5: ["empirical_ir_stats_1month.npz"],
    6: ["backtest_results/bayesian_backtest_N3.csv", "backtest_results/bayesian_backtest_N4.csv", "backtest_results/bayesian_backtest_N5.csv"],
}


def run_step(step_num: int):
    """Run a single pipeline step."""
    script = pipeline_dir / f"{step_num}_compute_forward_ir.py" if step_num == 1 else \
             pipeline_dir / f"{step_num}_compute_signal_bases.py" if step_num == 2 else \
             pipeline_dir / f"{step_num}_apply_filters.py" if step_num == 3 else \
             pipeline_dir / f"{step_num}_precompute_feature_ir.py" if step_num == 4 else \
             pipeline_dir / f"{step_num}_empirical_ir_stats.py" if step_num == 5 else \
             pipeline_dir / f"{step_num}_deterministic_strategy_ir.py"

    print(f"\n{'='*120}")
    print(f"Running Step {step_num}: {script.name}")
    print(f"{'='*120}")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(pipeline_dir),
        capture_output=False
    )

    if result.returncode != 0:
        print(f"ERROR: Step {step_num} failed with return code {result.returncode}")
        return False

    print(f"[OK] Step {step_num} completed successfully")
    return True


def backup_outputs(run_num: int, steps: list):
    """Backup output files to timestamped directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = backup_dir / f"run_{run_num}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for step in steps:
        for filename in COMPARISON_FILES[step]:
            src = data_dir / filename
            if src.exists():
                dst = run_dir / f"step{step}_{filename}".replace("/", "_")
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.suffix == ".npz":
                    shutil.copy2(src, dst)
                elif src.suffix == ".parquet":
                    shutil.copy2(src, dst)
                elif src.suffix == ".csv":
                    shutil.copy2(src, dst)
                print(f"  Backed up Step {step}: {filename}")

    return run_dir


def compare_parquet_files(file1: Path, file2: Path) -> tuple:
    """Compare two parquet files. Returns (are_equal, differences)."""
    try:
        df1 = pd.read_parquet(file1)
        df2 = pd.read_parquet(file2)

        # Sort both by all columns to handle ordering differences
        df1_sorted = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
        df2_sorted = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)

        # Compare
        if df1_sorted.equals(df2_sorted):
            return True, "Files are identical"
        else:
            # Find differences
            diffs = []
            if len(df1_sorted) != len(df2_sorted):
                diffs.append(f"Row count differs: {len(df1_sorted)} vs {len(df2_sorted)}")

            if list(df1_sorted.columns) != list(df2_sorted.columns):
                diffs.append(f"Columns differ")

            # Check value differences
            for col in df1_sorted.columns:
                if col in df2_sorted.columns:
                    if not df1_sorted[col].equals(df2_sorted[col]):
                        diffs.append(f"Column '{col}' differs")

            return False, "; ".join(diffs)
    except Exception as e:
        return False, f"Error comparing: {str(e)}"


def compare_npz_files(file1: Path, file2: Path) -> tuple:
    """Compare two NPZ files. Returns (are_equal, differences)."""
    try:
        data1 = np.load(file1, allow_pickle=True)
        data2 = np.load(file2, allow_pickle=True)

        # Check keys
        keys1 = set(data1.files)
        keys2 = set(data2.files)

        if keys1 != keys2:
            return False, f"Keys differ"

        # Compare each array
        diffs = []
        for key in keys1:
            arr1 = data1[key]
            arr2 = data2[key]

            if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
                if arr1.shape != arr2.shape:
                    diffs.append(f"Array '{key}' shape differs: {arr1.shape} vs {arr2.shape}")
                elif arr1.dtype != arr2.dtype:
                    diffs.append(f"Array '{key}' dtype differs: {arr1.dtype} vs {arr2.dtype}")
                elif np.issubdtype(arr1.dtype, np.number):
                    # Numeric arrays: use allclose with tolerance
                    if not np.allclose(arr1, arr2, rtol=1e-9, atol=1e-12, equal_nan=True):
                        max_diff = np.nanmax(np.abs(arr1 - arr2))
                        diffs.append(f"Array '{key}' values differ (max diff: {max_diff:.2e})")
                else:
                    # String/datetime/object arrays: use direct equality (no equal_nan for non-numeric)
                    try:
                        match = (arr1 == arr2).all() if arr1.size > 0 else True
                    except:
                        match = np.array_equal(arr1, arr2)
                    if not match:
                        diffs.append(f"Array '{key}' values differ (non-numeric type)")
            else:
                # Non-array objects (strings, scalars, etc.)
                try:
                    match = (arr1 == arr2).all() if hasattr(arr1, '__len__') else arr1 == arr2
                except:
                    match = np.array_equal(arr1, arr2)
                if not match:
                    diffs.append(f"Array '{key}' differs")

        if diffs:
            return False, "; ".join(diffs)
        else:
            return True, "Files are identical"
    except Exception as e:
        return False, f"Error comparing: {str(e)}"


def compare_csv_files(file1: Path, file2: Path) -> tuple:
    """Compare two CSV files. Returns (are_equal, differences)."""
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Sort both by all columns to handle ordering differences
        df1_sorted = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
        df2_sorted = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)

        # Compare
        if df1_sorted.equals(df2_sorted):
            return True, "Files are identical"
        else:
            # Find differences
            diffs = []
            if len(df1_sorted) != len(df2_sorted):
                diffs.append(f"Row count differs: {len(df1_sorted)} vs {len(df2_sorted)}")

            if list(df1_sorted.columns) != list(df2_sorted.columns):
                diffs.append(f"Columns differ")

            # Check value differences
            for col in df1_sorted.columns:
                if col in df2_sorted.columns:
                    if not df1_sorted[col].equals(df2_sorted[col]):
                        diffs.append(f"Column '{col}' differs")

            return False, "; ".join(diffs)
    except Exception as e:
        return False, f"Error comparing: {str(e)}"


def compare_runs(run1_dir: Path, run2_dir: Path, steps: list) -> dict:
    """Compare two runs. Returns results dict."""
    results = {}

    for step in steps:
        results[step] = {}
        for filename in COMPARISON_FILES[step]:
            # Replace slashes with underscores for subdirectories
            file1 = run1_dir / f"step{step}_{filename}".replace("/", "_")
            file2 = run2_dir / f"step{step}_{filename}".replace("/", "_")

            if not file1.exists() or not file2.exists():
                results[step][filename] = ("MISSING", f"File not found")
                continue

            if filename.endswith(".parquet"):
                are_equal, diff = compare_parquet_files(file1, file2)
            elif filename.endswith(".npz"):
                are_equal, diff = compare_npz_files(file1, file2)
            elif filename.endswith(".csv"):
                are_equal, diff = compare_csv_files(file1, file2)
            else:
                are_equal, diff = False, "Unknown file type"

            results[step][filename] = ("PASS" if are_equal else "FAIL", diff)

    return results


def main():
    """Main test runner."""
    print(f"\n{'='*120}")
    print("PIPELINE DETERMINISM TEST (Steps 1-6) - 5 RUNS")
    print(f"{'='*120}")
    print(f"\nTest directory: {backup_dir}")

    steps_to_test = [1, 2, 3, 4, 5, 6]
    run_dirs = []

    # Run 5 times
    for run_num in range(1, 6):
        print(f"\n{'='*120}")
        print(f"RUN {run_num} - Execution {run_num}/5")
        print(f"{'='*120}")

        for step in steps_to_test:
            if not run_step(step):
                print(f"ERROR: Aborting test - Step {step} failed")
                return 1

        run_dir = backup_outputs(run_num, steps_to_test)
        run_dirs.append(run_dir)
        print(f"\n[OK] Run {run_num} outputs backed up to: {run_dir}")

    # Compare all runs against Run 1
    print(f"\n{'='*120}")
    print("COMPARING ALL RUNS AGAINST RUN 1")
    print(f"{'='*120}\n")

    all_passed = True
    for run_num in range(2, 6):
        print(f"\nComparing Run 1 vs Run {run_num}:")
        print(f"  Run 1: {run_dirs[0]}")
        print(f"  Run {run_num}: {run_dirs[run_num-1]}\n")

        results = compare_runs(run_dirs[0], run_dirs[run_num-1], steps_to_test)

        # Print results for this comparison
        for step in sorted(results.keys()):
            for filename, (status, detail) in results[step].items():
                icon = "[PASS]" if status == "PASS" else "[FAIL]"
                print(f"  {icon} Step {step} {filename}: {status}")
                if status != "PASS":
                    print(f"      Details: {detail}")
                    all_passed = False

    print(f"\n{'='*120}")
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED - Pipeline is deterministic across all 5 runs!")
    else:
        print("[FAILURE] SOME TESTS FAILED - Pipeline has non-deterministic behavior")
    print(f"{'='*120}")
    print(f"\nTest outputs saved to: {backup_dir}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
