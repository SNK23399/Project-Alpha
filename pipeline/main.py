"""
Walk-Forward Satellite Selection Pipeline - Main Orchestrator
=============================================================

Runs complete pipeline to predict which N ETFs to select based on walk-forward backtesting.

Workflow:
  1 Compute forward alpha and information ratio (target variable)
  2 Compute all signal bases (with automatic backup)
  3 Apply all filters (with automatic backup)
  4 Precompute feature-IR matrix (signal predictions)
  5 Precompute MC Information Ratio statistics (Bayesian priors)
  6 Run Bayesian satellite selection (final backtest)

Note: Step 3 (features) is skipped due to disk space constraints

Usage:
  python main.py                           # All steps, full pipeline
  python main.py --month YYYY-MM           # All steps, specific period
  python main.py --steps 1,2,3,4,5,6       # Run specific steps (comma-separated)
  python main.py --only-step 4             # Run only one step

Examples:
  python main.py --month 2024-02           # Run all steps
  python main.py --steps 2,3               # Run only steps 2 & 3
  python main.py --month 2024-02 --steps 4,5,6  # Run only downstream steps for Feb 2024
  python main.py --only-step 1             # Run only step 1
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import importlib.util


class WalkForwardSatelliteSelectionPipeline:
    """
    Orchestrates complete walk-forward satellite selection pipeline.

    Steps 1-3: Compute forward alpha, signal bases, and filters.
    Steps 4-6: Precompute feature-IR, MC statistics, and select satellites.
    """

    def __init__(self):
        """Initialize pipeline."""
        self.pipeline_dir = Path(__file__).parent
        self.core_satellite_dir = self.pipeline_dir.parent

        # Results storage
        self.results = {
            'pipeline': 'walk-forward-satellite-selection',
            'timestamp': datetime.now().isoformat(),
            'steps': {}  # Store results from each step
        }

    def print_header(self, step_name: str, step_num: str):
        """Print formatted step header."""
        print(f"\n{'='*120}")
        print(f"STEP {step_num}: {step_name}")
        print(f"{'='*120}")

    def print_progress(self, message: str):
        """Print progress message."""
        print(f"  [OK] {message}")

    # ========================================================================
    # STEP 1: Compute Forward Alpha & Information Ratio
    # ========================================================================

    def step_1_compute_forward_ir(self) -> dict:
        """
        Step 1: Compute forward alpha and information ratio (target variable)

        Computes forward returns, forward alpha, and forward IR for all ETFs.
        This is the target variable for all downstream steps.
        Uses 1-month holding period.
        """
        self.print_header("Compute Forward Alpha & Information Ratio", "1")

        try:
            script_path = self.pipeline_dir / '1_compute_forward_ir.py'

            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")

            print(f"\n  Computing walk-forward target variable...")
            print(f"  - Forward alpha: ETF return - Benchmark return")
            print(f"  - Forward IR: forward_alpha / realized volatility")
            print(f"  - Holding period: 1 month")

            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.pipeline_dir),
                capture_output=False,
                check=False
            )

            if result.returncode != 0:
                raise RuntimeError(f"Forward IR computation failed with code {result.returncode}")

            alpha_file = self.pipeline_dir / 'data' / 'forward_alpha_1month.parquet'
            if not alpha_file.exists():
                raise FileNotFoundError(f"Forward alpha file not found: {alpha_file}")

            alpha_df = pd.read_parquet(alpha_file)

            self.print_progress("Forward IR computation completed")

            return {
                'status': 'completed',
                'n_observations': len(alpha_df),
                'n_dates': alpha_df['date'].nunique(),
                'n_isins': alpha_df['isin'].nunique(),
                'mean_forward_ir': float(alpha_df['forward_ir'].mean()),
            }

        except Exception as e:
            print(f"\n  [ERROR] in step 1: {str(e)}")
            raise

    # ========================================================================
    # STEP 1: Compute Signal Bases
    # ========================================================================

    def step_2_compute_signal_bases(self) -> dict:
        """
        Step 2: Compute all signal bases

        Calls 2_compute_signal_bases.py to compute all 293 signal bases.
        Includes inline correlation filtering and ranking matrix computation.

        Returns:
            Dictionary with computation stats
        """
        self.print_header("Compute All Signal Bases", "2")

        try:
            script_path = self.pipeline_dir / '2_compute_signal_bases.py'
            spec = importlib.util.spec_from_file_location("compute_signal_bases_module", script_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules['compute_signal_bases_module'] = module
            spec.loader.exec_module(module)

            print(f"\n  Computing 293 signal bases from all ETF prices...")
            print(f"  - Momentum, volatility, technical indicators, etc.")
            print(f"  - Includes inline correlation filtering (|r| > 0.1)")
            print(f"  - Backup: Enabled - will save to backup/1_compute_signal_bases/YYYY_MM_DD/")

            stats = module.compute_and_save_signal_bases()

            if stats is None:
                raise RuntimeError("Signal bases computation failed")

            self.results['steps']['1_compute_signal_bases'] = stats

            self.print_progress(f"Signal bases computed: {stats.get('n_signals', 0)} signals")
            self.print_progress(f"Date range: {stats.get('start_date')} to {stats.get('end_date')}")
            self.print_progress(f"Records saved: {stats.get('records_saved', 0):,}")
            self.print_progress(f"Time: {stats.get('total_time', 0):.1f}s")

            return stats

        except Exception as e:
            print(f"\n  [ERROR] in step 1: {str(e)}")
            raise

    def step_3_apply_filters(self) -> dict:
        """
        Step 3: Apply all filters to signal bases

        Applies 25 smoothing filters (EMA, Hull MA, etc.) to all signal bases,
        creating 7,325 filtered signals.
        Includes inline correlation filtering and ranking matrix computation.
        """
        self.print_header("Apply All Filters", "3")

        try:
            script_path = self.pipeline_dir / '3_apply_filters.py'

            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")

            spec = importlib.util.spec_from_file_location("apply_filters_module", script_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules['apply_filters_module'] = module
            spec.loader.exec_module(module)

            print(f"\n  Applying 25 filters to {self.results['steps'].get('1_compute_signal_bases', {}).get('n_signals', 293)} signal bases...")
            print(f"  - Exponential Moving Average (EMA), Hull Moving Average, etc.")
            print(f"  - Generates ~7,325 filtered signals")
            print(f"  - Includes inline correlation filtering (|r| > 0.1)")
            print(f"  - Backup: Enabled - will save to backup/2_apply_filters/YYYY_MM_DD/")

            original_argv = sys.argv
            try:
                sys.argv = [str(script_path)]
                result = module.main()
            finally:
                sys.argv = original_argv

            if result != 0:
                raise RuntimeError(f"Filter application failed with code {result}")

            self.print_progress("All filters applied successfully")

            return {'status': 'completed'}

        except Exception as e:
            print(f"\n  [ERROR] in step 2: {str(e)}")
            raise

    # ========================================================================
    # STEP 5: Precompute Feature-IR Matrix
    # ========================================================================

    def step_4_precompute_feature_ir(self) -> dict:
        """
        Step 4: Precompute Feature Information Ratio Matrix

        For each filtered signal at each date, evaluates what IR would be achieved
        if we selected top-N ETFs by that signal's ranking.
        Uses pre-computed ranking matrices from Step 3.
        """
        self.print_header("Precompute Feature-IR Matrix", "4")

        try:
            script_path = self.pipeline_dir / '4_precompute_feature_ir.py'

            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")

            print(f"\n  Precomputing feature-IR statistics...")
            print(f"  - For each filtered signal: mean IR of top-N ETFs")
            print(f"  - Uses rankings from Step 3 and forward IR from Step 1")
            print(f"  - Answers: Which signals reliably predict good IR?")

            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.pipeline_dir),
                capture_output=False,
                check=False
            )

            if result.returncode != 0:
                raise RuntimeError(f"Feature-IR computation failed with code {result.returncode}")

            self.print_progress("Feature-IR matrix computed successfully")

            return {'status': 'completed'}

        except Exception as e:
            print(f"\n  [ERROR] in step 5: {str(e)}")
            raise

    # ========================================================================
    # STEP 5: Precompute MC Information Ratio Statistics
    # ========================================================================

    def step_5_precompute_mc_ir_stats(self) -> dict:
        """
        Step 5: Precompute MC Information Ratio Statistics

        Runs Monte Carlo simulations to compute IR distribution statistics
        for each (feature, N_satellites) combination. These become Bayesian priors.
        """
        self.print_header("Precompute MC Information Ratio Statistics", "5")

        try:
            script_path = self.pipeline_dir / '5_precompute_mc_ir_stats.py'

            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")

            print(f"\n  Computing Monte Carlo IR statistics...")
            print(f"  - Simulates 1M samples per month per feature")
            print(f"  - Generates Bayesian priors for satellite selection")
            print(f"  - Uses rankings from Step 3 and forward IR from Step 1")

            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.pipeline_dir),
                capture_output=False,
                check=False
            )

            if result.returncode != 0:
                raise RuntimeError(f"MC IR computation failed with code {result.returncode}")

            self.print_progress("MC IR statistics computed successfully")

            return {'status': 'completed'}

        except Exception as e:
            print(f"\n  [ERROR] in step 6: {str(e)}")
            raise

    # ========================================================================
    # STEP 7: Bayesian Satellite Selection
    # ========================================================================

    def step_6_bayesian_strategy(self) -> dict:
        """
        Step 6: Bayesian Satellite Selection with Walk-Forward Backtesting

        Uses pre-computed feature-IR and MC statistics to select satellites
        using Bayesian learning with 2 hyperparameters (decay rate, prior strength).
        Runs complete walk-forward backtest.
        """
        self.print_header("Bayesian Satellite Selection & Backtest", "6")

        try:
            script_path = self.pipeline_dir / '6_bayesian_strategy_ir.py'

            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")

            print(f"\n  Running Bayesian satellite selection...")
            print(f"  - Uses learned hyperparameters (decay, prior strength)")
            print(f"  - Performs walk-forward backtest")
            print(f"  - Outputs satellite selections and performance metrics")

            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.pipeline_dir),
                capture_output=False,
                check=False
            )

            if result.returncode != 0:
                raise RuntimeError(f"Bayesian selection failed with code {result.returncode}")

            self.print_progress("Bayesian satellite selection completed")

            return {'status': 'completed'}

        except Exception as e:
            print(f"\n  [ERROR] in step 7: {str(e)}")
            raise

    def run(self, steps: list = None) -> dict:
        """
        Execute the pipeline (selected steps or all steps).

        Args:
            steps: List of step names to execute (e.g., ['0', '1', '2', '5', '6', '7']).
                   If None, execute all steps in order: ['0', '1', '2', '5', '6', '7']

        Returns:
            Dictionary with all results
        """
        # Default to all steps if none specified
        if steps is None:
            steps = ['1', '2', '3', '4', '5', '6']

        # Map step names to methods
        step_methods = {
            '1': self.step_1_compute_forward_ir,
            '2': self.step_2_compute_signal_bases,
            '3': self.step_3_apply_filters,
            '4': self.step_4_precompute_feature_ir,
            '5': self.step_5_precompute_mc_ir_stats,
            '6': self.step_6_bayesian_strategy,
        }

        # Validate requested steps
        invalid_steps = [s for s in steps if s not in step_methods]
        if invalid_steps:
            raise ValueError(f"Invalid steps: {invalid_steps}. Valid steps: {list(step_methods.keys())}")

        print("\n" + "="*120)
        print(f"WALK-FORWARD SATELLITE SELECTION PIPELINE")
        print(f"Steps to execute: {', '.join(steps)}")
        print("="*120)

        try:
            # Execute selected steps
            for step_name in steps:
                step_methods[step_name]()

            print("\n" + "="*120)
            print(f"PIPELINE COMPLETE - STEPS {', '.join(steps)} DONE")
            print("="*120)

            return self.results

        except Exception as e:
            print(f"\n{'='*120}")
            print(f"PIPELINE FAILED")
            print(f"{'='*120}")
            print(f"Error: {str(e)}")
            raise


def main():
    """Entry point for command line execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Walk-Forward Satellite Selection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # All steps (1,2,3,4,5,6)
  python main.py --steps 2,3               # Only steps 2 & 3
  python main.py --steps 4,5,6             # Only downstream steps (feature-IR through backtest)
  python main.py --only-step 1             # Only step 1 (forward IR)
  python main.py --only-step 5             # Only step 5 (MC IR statistics)
        """
    )

    parser.add_argument(
        '--steps',
        type=str,
        default=None,
        help='Comma-separated list of steps to run (e.g., "1,2,3,4,5,6"). Default: all steps'
    )

    parser.add_argument(
        '--only-step',
        type=str,
        default=None,
        help='Run only this step (e.g., "4"). Shorthand for --steps'
    )

    args = parser.parse_args()

    # Handle step selection
    steps_to_run = None

    if args.only_step:
        steps_to_run = [args.only_step]
    elif args.steps:
        steps_to_run = [s.strip() for s in args.steps.split(',')]

    # Create and run pipeline
    pipeline = WalkForwardSatelliteSelectionPipeline()

    results = pipeline.run(steps=steps_to_run)

    # Save results to JSON
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"pipeline_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
