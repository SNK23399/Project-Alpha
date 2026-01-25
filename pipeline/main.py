"""
Monthly Satellite Selection Pipeline - Main Orchestrator
========================================================

Runs on the first day of each month to predict which N ETFs to buy for the next month.

Workflow:
  2.1 Compute all signal bases (with automatic backup)
  2.2 Apply all filters (with automatic backup)
  2.3 Compute cross-sectional features from filtered signals
  2.4 Compute forward alpha and information ratio (walk-forward data)

Usage:
  python main.py                           # All steps, next month
  python main.py --month YYYY-MM           # All steps, specific month
  python main.py --steps 2.1,2.2,2.3,2.4   # Run specific steps (comma-separated)
  python main.py --only-step 2.4           # Run only one step

Examples:
  python main.py --month 2024-02           # Run all steps for Feb 2024
  python main.py --steps 2.4               # Run only step 2.4 (next month)
  python main.py --month 2024-02 --steps 2.3,2.4  # Run steps 2.3 & 2.4 for Feb 2024
  python main.py --only-step 2.2           # Run only step 2.2 (next month)
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import importlib.util


class MonthlySatelliteSelectionPipeline:
    """
    Orchestrates monthly satellite selection pipeline.

    Steps 2.1-2.2: Compute signal bases and apply filters.
    """

    def __init__(self, target_month: str = None):
        """
        Initialize pipeline.

        Args:
            target_month: Target month (YYYY-MM). If None, use next month.
        """
        self.pipeline_dir = Path(__file__).parent
        self.core_satellite_dir = self.pipeline_dir.parent

        # Determine target month
        if target_month is None:
            # Use next month
            today = datetime.now()
            if today.day > 1:
                # We're past day 1, so target next month
                target = (today + timedelta(days=32)).replace(day=1)
            else:
                # Today is the 1st, target this month
                target = today.replace(day=1)
            self.target_month = target.strftime('%Y-%m')
            self.target_date = target.strftime('%Y-%m-%d')
        else:
            # Parse provided month
            try:
                target = pd.Timestamp(target_month).to_period('M').to_timestamp('M')
                self.target_month = target.strftime('%Y-%m')
                self.target_date = target.strftime('%Y-%m-%d')
            except:
                raise ValueError(f"Invalid month format: {target_month}. Use YYYY-MM")

        # Results storage
        self.results = {
            'month': self.target_month,
            'target_date': self.target_date,
            'steps': {}  # Store results from each step
        }

    def print_header(self, step_name: str, step_num: str):
        """Print formatted step header."""
        print(f"\n{'='*80}")
        print(f"STEP {step_num}: {step_name}")
        print(f"{'='*80}")

    def print_progress(self, message: str):
        """Print progress message."""
        print(f"  [OK] {message}")

    # ========================================================================
    # STEP 2.1: Compute Signal Bases
    # ========================================================================

    def step_2_1_compute_signal_bases(self) -> dict:
        """
        Step 2.1: Compute all signal bases

        Calls 1_compute_signal_bases.py in FULL mode to ensure:
        - Any price corrections in the database are captured
        - Rolling window calculations are fresh
        - All data is consistent after ETF changes

        Returns:
            Dictionary with computation stats
        """
        self.print_header("Compute All Signal Bases", "2.1")

        try:
            # Load 1_compute_signal_bases module
            script_path = self.pipeline_dir / '1_compute_signal_bases.py'
            spec = importlib.util.spec_from_file_location("compute_signal_bases_module", script_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules['compute_signal_bases_module'] = module  # Register for multiprocessing
            spec.loader.exec_module(module)

            # Run in FULL mode - recomputes all signal bases from scratch
            # This ensures price corrections and rolling window updates are captured
            print(f"\n  Running in FULL mode (recompute all data from scratch)")
            print(f"  Reason: Ensures price corrections and rolling calculations are fresh")
            print(f"  Target month: {self.target_month}")
            print(f"  Backup: Enabled - will save to backup/1_compute_signal_bases/YYYY_MM_DD/")

            stats = module.compute_and_save_signal_bases(backup=True)

            # Handle case where computation failed
            if stats is None:
                raise RuntimeError("Signal bases computation failed")

            # Store results
            self.results['steps']['2.1_compute_signal_bases'] = stats

            # Check results
            records_saved = stats.get('records_saved', 0)
            self.print_progress(f"Signal bases computed: {stats.get('n_signals', 0)} signals")
            self.print_progress(f"Date range: {stats.get('start_date')} to {stats.get('end_date')}")
            self.print_progress(f"Records saved: {records_saved:,}")
            self.print_progress(f"Time: {stats.get('total_time', 0):.1f}s")

            return stats

        except Exception as e:
            print(f"\n  [ERROR] in step 2.1: {str(e)}")
            raise

    def step_2_2_apply_filters(self) -> dict:
        """
        Step 2.2: Apply all filters to signal bases

        Applies smoothing filters (EMA, Hull MA, etc.) to all computed signal bases,
        creating filtered signals for downstream feature computation.
        """
        self.print_header("Apply All Filters", "2.2")

        try:
            # Load 2_apply_filters module from pipeline directory
            script_path = self.pipeline_dir / '2_apply_filters.py'

            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")

            spec = importlib.util.spec_from_file_location("apply_filters_module", script_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules['apply_filters_module'] = module  # Register for multiprocessing
            spec.loader.exec_module(module)

            print(f"\n  Applying filters to all signal bases...")
            print(f"  Backup: Enabled - will save to backup/2_apply_filters/YYYY_MM_DD/")

            # Call the main function
            # Set sys.argv to simulate command line (script 2 always backs up by default)
            original_argv = sys.argv
            try:
                sys.argv = [str(script_path)]
                result = module.main()
            finally:
                sys.argv = original_argv

            if result != 0:
                raise RuntimeError(f"Filter application failed with code {result}")

            self.print_progress("All filters applied successfully")

            # For now, return empty dict since main() doesn't return stats
            # The printed output above contains the details
            return {'status': 'completed'}

        except Exception as e:
            print(f"\n  [ERROR] in step 2.2: {str(e)}")
            raise

    def step_2_3_compute_features(self) -> dict:
        """
        Step 2.3: Compute cross-sectional features from filtered signals

        Computes 25 cross-sectional indicators for each filtered signal,
        generating feature data for the Bayesian satellite selection model.
        """
        self.print_header("Compute Cross-Sectional Features", "2.3")

        try:
            # Load 3_compute_features module from pipeline directory
            script_path = self.pipeline_dir / '3_compute_features.py'

            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")

            spec = importlib.util.spec_from_file_location("compute_features_module", script_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules['compute_features_module'] = module  # Register for multiprocessing
            spec.loader.exec_module(module)

            print(f"\n  Computing cross-sectional features from all filtered signals...")
            print(f"  This generates 25 indicators for each of the filtered signals")

            # Call compute_all_features() with no arguments
            module.compute_all_features()

            self.print_progress("All features computed successfully")

            # Return status (feature computation doesn't return detailed stats)
            return {'status': 'completed'}

        except Exception as e:
            print(f"\n  [ERROR] in step 2.3: {str(e)}")
            raise

    def step_2_4_compute_forward_ir(self) -> dict:
        """
        Step 2.4: Compute forward alpha and information ratio (walk-forward data)

        Computes forward returns, forward alpha, forward IR, and rankings matrix
        for the Bayesian satellite selection model. Uses 1-month holding period.
        """
        self.print_header("Compute Forward Alpha & Information Ratio", "2.4")

        try:
            # Run 4_compute_forward_ir.py as subprocess (avoids multiprocessing pickling issues)
            script_path = self.pipeline_dir / '4_compute_forward_ir.py'

            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")

            print(f"\n  Computing walk-forward data for IR-based optimization...")
            print(f"  - Forward alpha: ETF return - Benchmark return")
            print(f"  - Forward IR: forward_alpha / realized volatility")
            print(f"  - Rankings matrix: z-score normalized features")

            # Run as subprocess (separate Python process with own environment)
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.pipeline_dir),
                capture_output=False,
                check=False
            )

            if result.returncode != 0:
                raise RuntimeError(f"Forward IR computation failed with code {result.returncode}")

            # Load results from saved files
            alpha_file = self.pipeline_dir / 'data' / 'forward_alpha_1month.parquet'
            if not alpha_file.exists():
                raise FileNotFoundError(f"Forward alpha file not found: {alpha_file}")

            alpha_df = pd.read_parquet(alpha_file)

            self.print_progress("Forward IR computation completed successfully")

            # Return computation statistics
            return {
                'status': 'completed',
                'n_observations': len(alpha_df),
                'n_dates': alpha_df['date'].nunique(),
                'n_isins': alpha_df['isin'].nunique(),
                'mean_forward_ir': float(alpha_df['forward_ir'].mean()),
            }

        except Exception as e:
            print(f"\n  [ERROR] in step 2.4: {str(e)}")
            raise

    def run(self, steps: list = None) -> dict:
        """
        Execute the pipeline (selected steps or all steps).

        Args:
            steps: List of step names to execute (e.g., ['2.1', '2.2', '2.3', '2.4']).
                   If None, execute all steps: ['2.1', '2.2', '2.3', '2.4']

        Returns:
            Dictionary with all results
        """
        # Default to all steps if none specified
        if steps is None:
            steps = ['2.1', '2.2', '2.3', '2.4']

        # Map step names to methods
        step_methods = {
            '2.1': self.step_2_1_compute_signal_bases,
            '2.2': self.step_2_2_apply_filters,
            '2.3': self.step_2_3_compute_features,
            '2.4': self.step_2_4_compute_forward_ir,
        }

        # Validate requested steps
        invalid_steps = [s for s in steps if s not in step_methods]
        if invalid_steps:
            raise ValueError(f"Invalid steps: {invalid_steps}. Valid steps: {list(step_methods.keys())}")

        print("\n" + "="*80)
        print(f"MONTHLY SATELLITE SELECTION PIPELINE")
        print(f"Target Month: {self.target_month} ({self.target_date})")
        print(f"Steps to execute: {', '.join(steps)}")
        print("="*80)

        try:
            # Execute selected steps
            for step_name in steps:
                step_methods[step_name]()

            print("\n" + "="*80)
            print(f"PIPELINE COMPLETE - STEPS {', '.join(steps)} DONE")
            print("="*80)

            return self.results

        except Exception as e:
            print(f"\n{'='*80}")
            print(f"PIPELINE FAILED")
            print(f"{'='*80}")
            print(f"Error: {str(e)}")
            raise


def main():
    """Entry point for command line execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Monthly Satellite Selection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # All steps, next month
  python main.py --month 2024-02           # All steps, specific month
  python main.py --steps 2.3               # Only step 2.3, next month
  python main.py --month 2024-02 --steps 2.1,2.3  # Steps 2.1 & 2.3, Feb 2024
  python main.py --only-step 2.2           # Only step 2.2, next month
        """
    )

    parser.add_argument(
        '--month',
        type=str,
        default=None,
        help='Target month (YYYY-MM). Default: next month'
    )

    parser.add_argument(
        '--steps',
        type=str,
        default=None,
        help='Comma-separated list of steps to run (e.g., "2.1,2.3"). Default: all steps (2.1,2.2,2.3)'
    )

    parser.add_argument(
        '--only-step',
        type=str,
        default=None,
        help='Run only this step (e.g., "2.3"). Shorthand for --steps'
    )

    args = parser.parse_args()

    # Handle step selection
    steps_to_run = None

    if args.only_step:
        steps_to_run = [args.only_step]
    elif args.steps:
        steps_to_run = [s.strip() for s in args.steps.split(',')]

    # Create and run pipeline
    pipeline = MonthlySatelliteSelectionPipeline(
        target_month=args.month
    )

    results = pipeline.run(steps=steps_to_run)

    # Save results to JSON
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"selection_{pipeline.target_month}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
