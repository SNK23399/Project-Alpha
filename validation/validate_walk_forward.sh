#!/bin/bash
# Walk-Forward Validation: Test pipeline for future bias by month-by-month simulation
#
# This script simulates deploying the pipeline on historical dates, using only
# data available at that time. Results are stored separately for comparison.
#
# Usage:
#   ./validate_walk_forward.sh --start-date 2014-01-31 --end-date 2025-12-31 [--step N]
#
# Examples:
#   ./validate_walk_forward.sh --start-date 2020-01-31 --end-date 2020-12-31  # Year 2020
#   ./validate_walk_forward.sh --start-date 2014-01-31 --end-date 2025-12-31 --step 3  # Every 3 months
#   ./validate_walk_forward.sh --start-date 2014-01-31 --end-date 2025-12-31 --step 1 --only-date 2020-01-31  # Single month

set -e

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE_DIR="$PROJECT_ROOT"
MAINTENANCE_DIR="$PROJECT_ROOT/maintenance"
VALIDATION_DIR="$SCRIPT_DIR"
BACKUP_DIR="$VALIDATION_DIR/database_backups"
PREDICTIONS_DIR="$VALIDATION_DIR/historical_predictions"
TRUNCATE_SCRIPT="$VALIDATION_DIR/truncate_database.py"

DB_PATH="$MAINTENANCE_DIR/data/etf_database.db"
DB_BACKUP="$DB_PATH.full"

# ============================================================
# PARSE ARGUMENTS
# ============================================================

START_DATE=""
END_DATE=""
STEP=1
ONLY_DATE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --start-date)
            START_DATE="$2"
            shift 2
            ;;
        --end-date)
            END_DATE="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --only-date)
            ONLY_DATE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$START_DATE" ] || [ -z "$END_DATE" ]; then
    echo "Usage: $0 --start-date YYYY-MM-DD --end-date YYYY-MM-DD [--step N] [--only-date YYYY-MM-DD]"
    exit 1
fi

# ============================================================
# FUNCTIONS
# ============================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log_section() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "$1"
    echo "═══════════════════════════════════════════════════════════════════════════════"
}

validate_date_format() {
    if ! [[ "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        log "ERROR: Invalid date format: $1 (use YYYY-MM-DD)"
        return 1
    fi
    return 0
}

# ============================================================
# MAIN EXECUTION
# ============================================================

log_section "WALK-FORWARD VALIDATION - Testing for Future Bias"

log "Configuration:"
log "  Start date: $START_DATE"
log "  End date: $END_DATE"
log "  Step: $STEP months"
if [ -n "$ONLY_DATE" ]; then
    log "  Single date mode: $ONLY_DATE"
fi
log "  Project root: $PROJECT_ROOT"
log "  Predictions saved to: $PREDICTIONS_DIR"

validate_date_format "$START_DATE" || exit 1
validate_date_format "$END_DATE" || exit 1

mkdir -p "$BACKUP_DIR" "$PREDICTIONS_DIR"

# Ensure full database exists
if [ ! -f "$DB_PATH" ]; then
    log "ERROR: Database not found: $DB_PATH"
    exit 1
fi

log "Database location: $DB_PATH"

# ============================================================
# PROCESS DATES
# ============================================================

if [ -n "$ONLY_DATE" ]; then
    # Single date mode
    DATES=("$ONLY_DATE")
else
    # Generate date range
    DATES=()
    CURRENT_DATE="$START_DATE"

    while [[ "$CURRENT_DATE" < "$END_DATE" ]] || [[ "$CURRENT_DATE" == "$END_DATE" ]]; do
        DATES+=("$CURRENT_DATE")

        # Add STEP months to CURRENT_DATE
        CURRENT_DATE=$(date -d "$CURRENT_DATE +${STEP} month" '+%Y-%m-%d' 2>/dev/null || \
                       date -jf "%Y-%m-%d" -v +"${STEP}m" "+%Y-%m-%d" "$CURRENT_DATE" 2>/dev/null || \
                       python3 -c "from datetime import datetime, timedelta; from dateutil.relativedelta import relativedelta; d = datetime.strptime('$CURRENT_DATE', '%Y-%m-%d') + relativedelta(months=${STEP}); print(d.strftime('%Y-%m-%d'))")
    done
fi

log "Will process ${#DATES[@]} dates"

# ============================================================
# RUN PIPELINE FOR EACH DATE
# ============================================================

PROCESSED=0
FAILED=0

for TARGET_DATE in "${DATES[@]}"; do
    log_section "Processing: $TARGET_DATE"

    DATE_FORMATTED="${TARGET_DATE//-/_}"
    PREDICTION_DIR="$PREDICTIONS_DIR/$TARGET_DATE"
    TRUNCATED_DB="$BACKUP_DIR/etf_database_$DATE_FORMATTED.db"

    # 1. Create truncated database
    log "Truncating database to $TARGET_DATE..."
    if ! python3 "$TRUNCATE_SCRIPT" "$DB_PATH" "$TARGET_DATE" "$TRUNCATED_DB"; then
        log "ERROR: Failed to truncate database"
        ((FAILED++))
        continue
    fi

    # 2. Backup full database and swap in truncated one
    log "Swapping databases..."
    cp "$DB_PATH" "$DB_BACKUP"
    cp "$TRUNCATED_DB" "$DB_PATH"

    # 3. Run pipeline
    log "Running pipeline with truncated data..."
    if cd "$PIPELINE_DIR" && python main.py --steps 1,2,3,4,5,6 > "/tmp/pipeline_${DATE_FORMATTED}.log" 2>&1; then
        log "✓ Pipeline completed successfully"

        # 4. Save results
        log "Saving results to $PREDICTION_DIR..."
        mkdir -p "$PREDICTION_DIR"
        cp "data/backtest_results/bayesian_backtest_N"*.csv "$PREDICTION_DIR/" 2>/dev/null || true

        # Create summary
        cat > "$PREDICTION_DIR/metadata.txt" << EOF
Validation Date: $TARGET_DATE
Pipeline Run: $(date)
Status: Success

Satellite selections (latest month):
EOF

        for N in 3 4 5 6 7; do
            if [ -f "$PREDICTION_DIR/bayesian_backtest_N${N}.csv" ]; then
                LAST_ROW=$(tail -1 "$PREDICTION_DIR/bayesian_backtest_N${N}.csv")
                echo "N=$N: $LAST_ROW" >> "$PREDICTION_DIR/metadata.txt"
            fi
        done

        ((PROCESSED++))
    else
        log "ERROR: Pipeline failed for $TARGET_DATE"
        cat "/tmp/pipeline_${DATE_FORMATTED}.log" >> "$PREDICTION_DIR/error.log" 2>/dev/null || true
        ((FAILED++))
    fi

    # 5. Restore full database
    log "Restoring full database..."
    rm "$DB_PATH"
    mv "$DB_BACKUP" "$DB_PATH"

    cd "$SCRIPT_DIR"

done

# ============================================================
# SUMMARY
# ============================================================

log_section "Validation Complete"
log "Processed: $PROCESSED dates"
log "Failed: $FAILED dates"
log "Results directory: $PREDICTIONS_DIR"
log ""
log "Next step: Compare historical predictions with current results using:"
log "  python3 compare_predictions.py $PREDICTIONS_DIR"
