#!/bin/bash
# Master script to run all 13 LSTM optimization experiments
#
# Usage:
#   ./scripts/run_all_phases.sh [--device cuda|cpu] [--mlflow-uri URI] [--skip-phase PHASE]
#
# Environment Variables:
#   MLFLOW_TRACKING_URI - MLflow tracking server URI
#   AWS_ACCESS_KEY_ID - AWS credentials for S3 artifact storage
#   AWS_SECRET_ACCESS_KEY - AWS credentials
#   SLACK_WEBHOOK_URL - Slack webhook for notifications

set -e  # Exit on error
set -u  # Error on undefined variable
set -o pipefail  # Catch errors in pipes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default configuration
DEVICE="cuda"
MLFLOW_URI="${MLFLOW_TRACKING_URI:-file:///tmp/mlruns}"
MLFLOW_EXPERIMENT="lstm-optimization-2025"
SEED=1337
SKIP_PHASES=""
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_TIME=$(date +%s)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --mlflow-uri)
            MLFLOW_URI="$2"
            shift 2
            ;;
        --skip-phase)
            SKIP_PHASES="$SKIP_PHASES $2"
            shift 2
            ;;
        --help)
            cat <<EOF
LSTM Optimization Pipeline Runner

Usage: $0 [OPTIONS]

Options:
    --device DEVICE         Device to use (cuda or cpu, default: cuda)
    --mlflow-uri URI        MLflow tracking URI
    --skip-phase PHASE      Skip a phase (can be repeated)
    --help                  Show this help message

Phases:
    phase1    Time warp sigma ablation (4 experiments)
    phase2    Architecture search (8 experiments)
    phase3    Pre-training depth search (3 experiments)
    promote   Model promotion and comparison

Environment Variables:
    MLFLOW_TRACKING_URI     MLflow server URI
    AWS_ACCESS_KEY_ID       AWS credentials
    AWS_SECRET_ACCESS_KEY   AWS credentials
    SLACK_WEBHOOK_URL       Slack notifications

Example:
    $0 --device cuda --skip-phase phase1
EOF
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

log_info "========================================================================="
log_info "                 LSTM OPTIMIZATION PIPELINE"
log_info "========================================================================="
log_info "Configuration:"
log_info "  Device: $DEVICE"
log_info "  MLflow URI: $MLFLOW_URI"
log_info "  MLflow Experiment: $MLFLOW_EXPERIMENT"
log_info "  Seed: $SEED"
log_info "  Skip Phases: ${SKIP_PHASES:-none}"
log_info "========================================================================="

export MLFLOW_TRACKING_URI="$MLFLOW_URI"

# Verify prerequisites
log_info "Verifying prerequisites..."

if ! command -v python &> /dev/null; then
    log_error "Python not found. Please install Python 3.10+"
    exit 1
fi

if [ "$DEVICE" == "cuda" ]; then
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        log_error "CUDA not available. Use --device cpu or install CUDA toolkit"
        exit 1
    fi
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    log_success "CUDA available with $GPU_COUNT GPU(s)"
else
    log_warning "Running on CPU (this will be slow)"
fi

# Create results tracking file
RESULTS_FILE="$PROJECT_ROOT/experiment_results.json"
echo "{\"experiments\": [], \"start_time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > "$RESULTS_FILE"

# ============================================================================
# Phase 1: Time Warp Sigma Ablation (4 experiments)
# ============================================================================
if [[ ! "$SKIP_PHASES" =~ "phase1" ]]; then
    log_info ""
    log_info "========================================================================="
    log_info "PHASE 1: Time Warp Sigma Ablation (4 experiments)"
    log_info "========================================================================="

    PHASE1_RESULTS=()
    TIME_WARP_SIGMAS=(0.10 0.12 0.15 0.20)

    for sigma in "${TIME_WARP_SIGMAS[@]}"; do
        log_info ""
        log_info "--- Experiment: time_warp_sigma=$sigma ---"

        EXP_START=$(date +%s)

        # Generate unlabeled data
        log_info "Step 1/3: Generating unlabeled data..."
        python scripts/generate_unlabeled_data.py \
            --output "data/raw/unlabeled_windows_tw_${sigma}.parquet" \
            --time-warp-sigma "$sigma" \
            --num-samples 10000 \
            --seed "$SEED" || {
            log_error "Failed to generate unlabeled data"
            continue
        }

        # Pre-train Masked LSTM
        log_info "Step 2/3: Pre-training Masked LSTM encoder (50 epochs)..."
        python scripts/pretrain_masked_lstm.py \
            --unlabeled-data "data/raw/unlabeled_windows_tw_${sigma}.parquet" \
            --output-dir "data/artifacts/pretrained/masked_lstm_tw_${sigma}" \
            --epochs 50 \
            --batch-size 256 \
            --learning-rate 0.001 \
            --device "$DEVICE" \
            --mlflow-experiment "$MLFLOW_EXPERIMENT" \
            --mlflow-run-name "phase1_tw_${sigma}" || {
            log_error "Pre-training failed for sigma=$sigma"
            continue
        }

        # Fine-tune SimpleLSTM
        log_info "Step 3/3: Fine-tuning SimpleLSTM (50 epochs)..."
        python -m moola.cli oof \
            --model simple_lstm \
            --device "$DEVICE" \
            --seed "$SEED" \
            --load-pretrained-encoder "data/artifacts/pretrained/masked_lstm_tw_${sigma}/encoder.pt" \
            --mlflow-tracking \
            --mlflow-experiment "$MLFLOW_EXPERIMENT" || {
            log_error "Fine-tuning failed for sigma=$sigma"
            continue
        }

        EXP_END=$(date +%s)
        EXP_DURATION=$((EXP_END - EXP_START))

        log_success "Completed experiment in ${EXP_DURATION}s"

        # Clean up large files
        rm -f "data/raw/unlabeled_windows_tw_${sigma}.parquet"
    done

    log_success "Phase 1 complete!"
else
    log_warning "Skipping Phase 1"
fi

# Select Phase 1 winner
if [[ ! "$SKIP_PHASES" =~ "phase1" ]]; then
    log_info ""
    log_info "Selecting Phase 1 winner..."
    python scripts/select_phase_winner.py \
        --phase 1 \
        --experiment-name "$MLFLOW_EXPERIMENT" \
        --output-file "$PROJECT_ROOT/phase1_winner.json"

    WINNER_SIGMA=$(jq -r '.time_warp_sigma' phase1_winner.json)
    WINNER_ACC=$(jq -r '.accuracy' phase1_winner.json)
    log_success "Phase 1 Winner: time_warp_sigma=$WINNER_SIGMA (accuracy=$WINNER_ACC)"
fi

# ============================================================================
# Phase 2: Architecture Search (8 experiments)
# ============================================================================
if [[ ! "$SKIP_PHASES" =~ "phase2" ]]; then
    log_info ""
    log_info "========================================================================="
    log_info "PHASE 2: Architecture Search (8 experiments)"
    log_info "========================================================================="

    if [ ! -f phase1_winner.json ]; then
        log_error "phase1_winner.json not found. Run Phase 1 first or provide it manually."
        exit 1
    fi

    WINNER_SIGMA=$(jq -r '.time_warp_sigma' phase1_winner.json)
    log_info "Using Phase 1 winner config: time_warp_sigma=$WINNER_SIGMA"

    HIDDEN_SIZES=(64 128)
    NUM_HEADS=(4 8)
    NUM_LAYERS=(1 2)

    EXPERIMENT_COUNT=0
    TOTAL_EXPERIMENTS=$((${#HIDDEN_SIZES[@]} * ${#NUM_HEADS[@]} * ${#NUM_LAYERS[@]}))

    for hidden in "${HIDDEN_SIZES[@]}"; do
        for heads in "${NUM_HEADS[@]}"; do
            for layers in "${NUM_LAYERS[@]}"; do
                EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
                log_info ""
                log_info "--- Experiment $EXPERIMENT_COUNT/$TOTAL_EXPERIMENTS: hidden=$hidden, heads=$heads, layers=$layers ---"

                EXP_START=$(date +%s)

                python -m moola.cli oof \
                    --model simple_lstm \
                    --device "$DEVICE" \
                    --seed "$SEED" \
                    --hidden-size "$hidden" \
                    --num-layers "$layers" \
                    --num-heads "$heads" \
                    --load-pretrained-encoder "data/artifacts/pretrained/masked_lstm_tw_${WINNER_SIGMA}/encoder.pt" \
                    --mlflow-tracking \
                    --mlflow-experiment "$MLFLOW_EXPERIMENT" || {
                    log_error "Training failed for config: h=$hidden, nh=$heads, l=$layers"
                    continue
                }

                EXP_END=$(date +%s)
                EXP_DURATION=$((EXP_END - EXP_START))
                log_success "Completed in ${EXP_DURATION}s"
            done
        done
    done

    log_success "Phase 2 complete!"
else
    log_warning "Skipping Phase 2"
fi

# Select Phase 2 winner
if [[ ! "$SKIP_PHASES" =~ "phase2" ]]; then
    log_info ""
    log_info "Selecting Phase 2 winner..."
    python scripts/select_phase_winner.py \
        --phase 2 \
        --experiment-name "$MLFLOW_EXPERIMENT" \
        --output-file "$PROJECT_ROOT/phase2_winner.json"

    WINNER_CONFIG=$(jq -c '.' phase2_winner.json)
    log_success "Phase 2 Winner: $WINNER_CONFIG"
fi

# ============================================================================
# Phase 3: Pre-training Depth Search (3 experiments)
# ============================================================================
if [[ ! "$SKIP_PHASES" =~ "phase3" ]]; then
    log_info ""
    log_info "========================================================================="
    log_info "PHASE 3: Pre-training Depth Search (3 experiments)"
    log_info "========================================================================="

    if [ ! -f phase2_winner.json ]; then
        log_error "phase2_winner.json not found. Run Phase 2 first."
        exit 1
    fi

    WINNER_HIDDEN=$(jq -r '.hidden_size' phase2_winner.json)
    WINNER_HEADS=$(jq -r '.num_heads' phase2_winner.json)
    WINNER_LAYERS=$(jq -r '.num_layers' phase2_winner.json)

    log_info "Using Phase 2 winner architecture: h=$WINNER_HIDDEN, nh=$WINNER_HEADS, l=$WINNER_LAYERS"

    PRETRAIN_EPOCHS=(50 75 100)

    for epochs in "${PRETRAIN_EPOCHS[@]}"; do
        log_info ""
        log_info "--- Experiment: pretrain_epochs=$epochs ---"

        EXP_START=$(date +%s)

        # Pre-train with different epoch counts
        log_info "Step 1/2: Pre-training for $epochs epochs..."
        python scripts/pretrain_masked_lstm.py \
            --unlabeled-data "data/raw/unlabeled_windows_tw_${WINNER_SIGMA}.parquet" \
            --output-dir "data/artifacts/pretrained/masked_lstm_phase3_e${epochs}" \
            --epochs "$epochs" \
            --batch-size 256 \
            --learning-rate 0.001 \
            --device "$DEVICE" \
            --mlflow-experiment "$MLFLOW_EXPERIMENT" \
            --mlflow-run-name "phase3_pretrain_e${epochs}" || {
            log_error "Pre-training failed for epochs=$epochs"
            continue
        }

        # Fine-tune with winner architecture
        log_info "Step 2/2: Fine-tuning with winner architecture..."
        python -m moola.cli oof \
            --model simple_lstm \
            --device "$DEVICE" \
            --seed "$SEED" \
            --hidden-size "$WINNER_HIDDEN" \
            --num-layers "$WINNER_LAYERS" \
            --num-heads "$WINNER_HEADS" \
            --load-pretrained-encoder "data/artifacts/pretrained/masked_lstm_phase3_e${epochs}/encoder.pt" \
            --mlflow-tracking \
            --mlflow-experiment "$MLFLOW_EXPERIMENT" || {
            log_error "Fine-tuning failed for epochs=$epochs"
            continue
        }

        EXP_END=$(date +%s)
        EXP_DURATION=$((EXP_END - EXP_START))
        log_success "Completed in ${EXP_DURATION}s"
    done

    log_success "Phase 3 complete!"
else
    log_warning "Skipping Phase 3"
fi

# ============================================================================
# Final: Promote Best Model
# ============================================================================
if [[ ! "$SKIP_PHASES" =~ "promote" ]]; then
    log_info ""
    log_info "========================================================================="
    log_info "MODEL PROMOTION: Selecting best model from all 13 experiments"
    log_info "========================================================================="

    python scripts/select_best_model.py \
        --experiment-name "$MLFLOW_EXPERIMENT" \
        --output-report "$PROJECT_ROOT/comparison_report.md" \
        --min-class1-accuracy 0.30

    if [ -f best_model.json ]; then
        BEST_RUN_ID=$(jq -r '.run_id' best_model.json)
        BEST_ACC=$(jq -r '.accuracy' best_model.json)
        BEST_CLASS1=$(jq -r '.class_1_accuracy' best_model.json)

        log_success "Best Model Selected!"
        log_success "  Run ID: $BEST_RUN_ID"
        log_success "  Overall Accuracy: $BEST_ACC"
        log_success "  Class 1 Accuracy: $BEST_CLASS1"

        # Tag in MLflow
        python -c "
import mlflow
client = mlflow.MlflowClient()
client.set_tag('$BEST_RUN_ID', 'promotion_status', 'production_candidate')
client.set_tag('$BEST_RUN_ID', 'promoted_at', '$(date -u +%Y-%m-%dT%H:%M:%SZ)')
client.set_tag('$BEST_RUN_ID', 'promoted_by', 'automated_pipeline')
print('Tagged run as production candidate')
"
    else
        log_error "Failed to select best model"
    fi
fi

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))

log_info ""
log_info "========================================================================="
log_success "PIPELINE COMPLETE!"
log_info "========================================================================="
log_info "Total Duration: ${HOURS}h ${MINUTES}m"
log_info ""
log_info "Artifacts:"
log_info "  Comparison Report: $PROJECT_ROOT/comparison_report.md"
log_info "  Phase 1 Winner: $PROJECT_ROOT/phase1_winner.json"
log_info "  Phase 2 Winner: $PROJECT_ROOT/phase2_winner.json"
log_info "  Best Model: $PROJECT_ROOT/best_model.json"
log_info ""
log_info "Next Steps:"
log_info "  1. View MLflow UI: mlflow ui --port 5000"
log_info "  2. Review comparison report: cat comparison_report.md"
log_info "  3. Deploy best model: python -m moola.api.serve"
log_info "========================================================================="

# Send Slack notification if webhook configured
if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
    python scripts/send_slack_notification.py \
        --webhook-url "$SLACK_WEBHOOK_URL" \
        --channel "#ml-experiments" \
        --title "LSTM Optimization Pipeline Complete" \
        --message "Duration: ${HOURS}h ${MINUTES}m | Best accuracy: $BEST_ACC" \
        --report "$PROJECT_ROOT/comparison_report.md" || true
fi
