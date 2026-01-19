#!/bin/bash
# Full Pipeline Runner for Walk-Forward Backtest
# Run this script from the project root or walk forward backtest directory

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================================"
echo "WALK-FORWARD BACKTEST - FULL PIPELINE"
echo "================================================================================"
echo ""
echo "This script runs all 6 steps sequentially:"
echo "  1. Compute signal bases (293 signals)"
echo "  2. Apply filters (25 filters -> 7,325 filtered signals)"
echo "  3. Compute features (cross-sectional indicators)"
echo "  4. Compute forward alpha and rankings matrix"
echo "  5. Precompute feature-alpha matrix"
echo "  6. Run backtest with decay weighting"
echo ""
echo "================================================================================"

# Step 1
echo ""
echo "STEP 1/6: Computing signal bases..."
echo "--------------------------------------------------------------------------------"
python "$SCRIPT_DIR/1_compute_signal_bases.py"

# Step 2
echo ""
echo "STEP 2/6: Applying filters..."
echo "--------------------------------------------------------------------------------"
python "$SCRIPT_DIR/2_apply_filters.py"

# Step 3
echo ""
echo "STEP 3/6: Computing features..."
echo "--------------------------------------------------------------------------------"
python "$SCRIPT_DIR/3_compute_features.py"

# Step 4
echo ""
echo "STEP 4/6: Computing forward alpha and rankings..."
echo "--------------------------------------------------------------------------------"
python "$SCRIPT_DIR/4_compute_forward_alpha.py"

# Step 5
echo ""
echo "STEP 5/6: Precomputing feature-alpha matrix..."
echo "--------------------------------------------------------------------------------"
python "$SCRIPT_DIR/5_precompute_feature_alpha.py"

# Step 5
echo ""
echo "STEP 6/6: Precomputing MC hitrates..."
echo "--------------------------------------------------------------------------------"
python "$SCRIPT_DIR/5b_precompute_mc_hitrates.py"

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to: $SCRIPT_DIR/data/backtest_results/"
