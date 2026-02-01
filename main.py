"""
Core-Satellite Portfolio Pipeline - Main Orchestrator
=====================================================

Runs complete pipeline for signal generation and strategy evaluation.

Workflow:
  1. Compute forward alpha and information ratio (target variable)
  2. Compute DPO base signal variants
  3. Apply Savgol post-processing filter to signals
  4. Precompute feature-IR matrix (signal predictions)
  5. Compute empirical IR statistics
  6. Evaluate deterministic strategies with IR metrics
  7. Generate monthly portfolio allocation

Usage:
  python main.py                           # All steps (1,2,3,4,5,6,7)
  python main.py --steps 1,2,3,4,5,6       # Run specific steps (comma-separated)
  python main.py --only-step 7             # Run only one step

Examples:
  python main.py                           # Run full pipeline
  python main.py --steps 4,5,6             # Skip to feature-IR computation
  python main.py --only-step 7             # Only run allocation generation
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import importlib.util


class WalkForwardSatelliteSelectionPipeline:
    """
    Orchestrates complete pipeline for signal generation and strategy evaluation.

    Steps 1-3: Compute target variable (forward IR), DPO signals, and post-processing filter.
    Steps 4-6: Precompute feature-IR matrix, compute IR statistics, and evaluate strategies.
    Step 7: Generate monthly portfolio allocation based on latest backtest results.
    """

    def __init__(self, pipeline_dir=None):
        """Initialize pipeline."""
        self.core_satellite_dir = Path(__file__).parent
        if pipeline_dir is None:
            self.pipeline_dir = self.core_satellite_dir / 'pipeline'
        else:
            # If relative path, make it relative to core_satellite_dir
            pipeline_path = Path(pipeline_dir)
            if not pipeline_path.is_absolute():
                self.pipeline_dir = self.core_satellite_dir / pipeline_path
            else:
                self.pipeline_dir = pipeline_path

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
        Step 2: Compute DPO base signal variants

        Calls 2_compute_signal_bases.py to compute Savgol-based DPO variants.
        Includes inline correlation filtering and ranking matrix computation.

        Returns:
            Dictionary with computation stats
        """
        self.print_header("Compute DPO Base Signal Variants", "2")

        try:
            script_path = self.pipeline_dir / '2_compute_signal_bases.py'
            spec = importlib.util.spec_from_file_location("compute_signal_bases_module", script_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules['compute_signal_bases_module'] = module
            spec.loader.exec_module(module)

            print(f"\n  Computing DPO base signal variants...")

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
        Step 3: Apply Savgol post-processing filter to DPO base signals

        Applies post-processing Savgol filter to DPO base signals,
        creating a signal library for satellite selection.
        Includes inline correlation filtering and ranking matrix computation.
        """
        self.print_header("Apply Savgol Post-Processing Filter", "3")

        try:
            script_path = self.pipeline_dir / '3_apply_filters.py'

            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")

            spec = importlib.util.spec_from_file_location("apply_filters_module", script_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules['apply_filters_module'] = module
            spec.loader.exec_module(module)

            print(f"\n  Applying Savgol post-processing filter...")

            original_argv = sys.argv
            try:
                sys.argv = [str(script_path)]
                result = module.main()
            finally:
                sys.argv = original_argv

            if result != 0:
                raise RuntimeError(f"Filter application failed with code {result}")

            self.print_progress("Post-processing filter applied successfully")

            return {'status': 'completed'}

        except Exception as e:
            print(f"\n  [ERROR] in step 2: {str(e)}")
            raise

    # ========================================================================
    # STEP 4: Precompute Feature-IR Matrix
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

            print(f"\n  Precomputing feature-IR matrix...")

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
        Step 5: Compute Empirical Information Ratio Statistics

        Computes empirical IR statistics for signal evaluation.
        """
        self.print_header("Compute Empirical IR Statistics", "5")

        try:
            script_path = self.pipeline_dir / '5_empirical_ir_stats.py'

            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")

            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.pipeline_dir),
                capture_output=False,
                check=False
            )

            if result.returncode != 0:
                raise RuntimeError(f"Empirical IR computation failed with code {result.returncode}")

            self.print_progress("Empirical IR statistics computed successfully")

            return {'status': 'completed'}

        except Exception as e:
            print(f"\n  [ERROR] in step 5: {str(e)}")
            raise

    # ========================================================================
    # STEP 6: Deterministic Strategy IR Evaluation
    # ========================================================================

    def step_6_bayesian_strategy(self) -> dict:
        """
        Step 6: Deterministic Strategy with Information Ratio Evaluation

        Evaluates deterministic strategies using information ratio metrics.
        Runs walk-forward backtest with strategy evaluation.
        """
        self.print_header("Deterministic Strategy IR Evaluation", "6")

        try:
            script_path = self.pipeline_dir / '6_deterministic_strategy_ir.py'

            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")

            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.pipeline_dir),
                capture_output=False,
                check=False
            )

            if result.returncode != 0:
                raise RuntimeError(f"Strategy evaluation failed with code {result.returncode}")

            self.print_progress("Strategy evaluation completed")

            return {'status': 'completed'}

        except Exception as e:
            print(f"\n  [ERROR] in step 6: {str(e)}")
            raise

    # ========================================================================
    # STEP 7: Generate Monthly Portfolio Allocation
    # ========================================================================

    def step_7_generate_allocation(self) -> dict:
        """
        Step 7: Generate Monthly Portfolio Allocation

        Converts latest backtest results to actual portfolio allocation.
        Reads user input for total budget and generates allocation recommendations
        with 60/40 core-satellite split.
        """
        self.print_header("Generate Monthly Portfolio Allocation", "7")

        try:
            script_path = self.pipeline_dir / '7_generate_monthly_allocation.py'

            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")

            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.pipeline_dir),
                capture_output=False,
                check=False
            )

            if result.returncode != 0:
                raise RuntimeError(f"Allocation generation failed with code {result.returncode}")

            self.print_progress("Monthly allocation generated successfully")

            return {'status': 'completed'}

        except Exception as e:
            print(f"\n  [ERROR] in step 7: {str(e)}")
            raise


    def run(self, steps: list = None) -> dict:
        """
        Execute the pipeline (selected steps or all steps).

        Args:
            steps: List of step names to execute (e.g., ['1', '2', '3', '4', '5', '6', '7']).
                   If None, execute all steps in order: ['1', '2', '3', '4', '5', '6', '7']

        Returns:
            Dictionary with all results
        """
        # Default to steps 1-7 if none specified
        if steps is None:
            steps = ['1', '2', '3', '4', '5', '6', '7']

        # Map step names to methods
        step_methods = {
            '1': self.step_1_compute_forward_ir,
            '2': self.step_2_compute_signal_bases,
            '3': self.step_3_apply_filters,
            '4': self.step_4_precompute_feature_ir,
            '5': self.step_5_precompute_mc_ir_stats,
            '6': self.step_6_bayesian_strategy,
            '7': self.step_7_generate_allocation,
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
        description='Core-Satellite Portfolio Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # All steps (1,2,3,4,5,6,7)
  python main.py --steps 2,3               # Only steps 2 & 3
  python main.py --steps 4,5,6             # Feature-IR through strategy evaluation
  python main.py --only-step 7             # Only step 7 (generate allocation)
        """
    )

    parser.add_argument(
        '--steps',
        type=str,
        default=None,
        help='Comma-separated list of steps to run (e.g., "1,2,3,4,5,6,7"). Default: all steps'
    )

    parser.add_argument(
        '--only-step',
        type=str,
        default=None,
        help='Run only this step (e.g., "7"). Shorthand for --steps'
    )

    parser.add_argument(
        '--pipeline-dir',
        type=str,
        default=None,
        help='Path to pipeline directory (default: ./pipeline, can use ./pipeline_copy for testing)'
    )

    args = parser.parse_args()

    # Handle step selection
    steps_to_run = None

    if args.only_step:
        steps_to_run = [args.only_step]
    elif args.steps:
        steps_to_run = [s.strip() for s in args.steps.split(',')]

    # Create and run pipeline
    pipeline = WalkForwardSatelliteSelectionPipeline(pipeline_dir=args.pipeline_dir)

    results = pipeline.run(steps=steps_to_run)

    return results


if __name__ == "__main__":
    main()
