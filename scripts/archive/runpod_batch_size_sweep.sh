#!/bin/bash
# Batch Size Sweep for Jade Pre-training on RunPod
#
# This script runs 3 pre-training experiments with different batch sizes
# to determine the optimal batch size for 5-year NQ data pre-training.
#
# Usage:
#   # SSH to RunPod
#   ssh -i ~/.ssh/runpod_key ubuntu@YOUR_RUNPOD_IP
#   cd /workspace/moola
#
#   # Run sweep
#   bash scripts/runpod_batch_size_sweep.sh

set -e  # Exit on error

echo "========================================================================"
echo "JADE PRE-TRAINING: BATCH SIZE SWEEP"
echo "========================================================================"
echo "Testing batch sizes: 512, 768, 1024"
echo "Expected time: ~3-4 hours total on RTX 4090"
echo ""

# Configuration
DATA_PATH="data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet"
CONFIG_PATH="configs/windowed.yaml"
EPOCHS=50
SEED=42

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    echo "Please upload the 5-year NQ parquet file to RunPod first."
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Create output directory
mkdir -p artifacts/batch_size_sweep

# Function to run one experiment
run_experiment() {
    local BATCH_SIZE=$1
    local OUTPUT_DIR="artifacts/batch_size_sweep/batch_${BATCH_SIZE}"

    echo ""
    echo "========================================================================"
    echo "EXPERIMENT: Batch Size $BATCH_SIZE"
    echo "========================================================================"
    echo "Output directory: $OUTPUT_DIR"
    echo "Start time: $(date)"
    echo ""

    # Run pre-training
    python3 scripts/train_jade_pretrain.py \
        --config "$CONFIG_PATH" \
        --data "$DATA_PATH" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR"

    echo ""
    echo "Experiment completed: Batch size $BATCH_SIZE"
    echo "End time: $(date)"
    echo ""
}

# Run experiments
echo "Starting batch size sweep..."
echo ""

run_experiment 512
run_experiment 768
run_experiment 1024

echo ""
echo "========================================================================"
echo "BATCH SIZE SWEEP COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: artifacts/batch_size_sweep/"
echo ""
echo "Compare results:"
echo "  cat artifacts/batch_size_sweep/batch_512/training_results.json | jq '.best_val_loss'"
echo "  cat artifacts/batch_size_sweep/batch_768/training_results.json | jq '.best_val_loss'"
echo "  cat artifacts/batch_size_sweep/batch_1024/training_results.json | jq '.best_val_loss'"
echo ""
echo "Transfer results back to Mac:"
echo "  scp -i ~/.ssh/runpod_key -r ubuntu@YOUR_IP:/workspace/moola/artifacts/batch_size_sweep ./artifacts/"
echo ""
