#!/bin/bash
# Master script to run both experiments in sequence
# Total runtime: ~30 minutes on RTX 4090 GPU

set -e  # Exit on error

echo "=============================================================================="
echo "PARALLEL EXPERIMENTS: THRESHOLD TUNING + DATA AUGMENTATION"
echo "=============================================================================="
echo ""
echo "Experiment A: Threshold Precision Tuning (5 min)"
echo "Experiment B: Data Augmentation Strategy (20 epochs, ~25 min)"
echo ""
echo "Total expected runtime: ~30 minutes"
echo ""

# Set paths
DATA_PATH="data/processed/labeled/train_latest_overlaps_v2.parquet"
CHECKPOINT_PATH="artifacts/baseline_100ep/best_model.pt"
OUTPUT_DIR="results/parallel_experiments_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    echo ""
    echo "Please run baseline training first:"
    echo "  python3 scripts/train_baseline_100ep.py \\"
    echo "      --data $DATA_PATH \\"
    echo "      --output artifacts/baseline_100ep/ \\"
    echo "      --epochs 100 \\"
    echo "      --device cuda"
    exit 1
fi

echo "✓ Found checkpoint: $CHECKPOINT_PATH"
echo ""

# ============================================================================
# EXPERIMENT A: Threshold Precision Tuning
# ============================================================================

echo "=============================================================================="
echo "EXPERIMENT A: THRESHOLD PRECISION TUNING"
echo "=============================================================================="
echo ""

python3 scripts/experiment_a_threshold_grid.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --data "$DATA_PATH" \
    --output "$OUTPUT_DIR/threshold_grid.csv" \
    --min-threshold 0.30 \
    --max-threshold 0.40 \
    --step 0.02 \
    --device cuda

echo ""
echo "✅ Experiment A complete"
echo ""

# ============================================================================
# EXPERIMENT B: Data Augmentation Strategy
# ============================================================================

echo "=============================================================================="
echo "EXPERIMENT B: DATA AUGMENTATION STRATEGY"
echo "=============================================================================="
echo ""

python3 scripts/experiment_b_augmentation.py \
    --data "$DATA_PATH" \
    --output "$OUTPUT_DIR/augmentation" \
    --epochs 20 \
    --n-augment 2 \
    --sigma 0.03 \
    --pos-weight 13.1 \
    --batch-size 32 \
    --lr 1e-3 \
    --device cuda

echo ""
echo "✅ Experiment B complete"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "=============================================================================="
echo "EXPERIMENTS COMPLETE"
echo "=============================================================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Experiment A outputs:"
echo "  - $OUTPUT_DIR/threshold_grid.csv"
echo "  - $OUTPUT_DIR/threshold_summary.txt"
echo ""
echo "Experiment B outputs:"
echo "  - $OUTPUT_DIR/augmentation/training_history.csv"
echo "  - $OUTPUT_DIR/augmentation/training_curves.png"
echo "  - $OUTPUT_DIR/augmentation/best_model.pt"
echo ""
echo "Next steps:"
echo "  1. Review threshold_grid.csv to find optimal threshold"
echo "  2. Compare augmentation F1 vs baseline to assess effectiveness"
echo "  3. If F1 > 0.25, deploy augmentation strategy to production"
echo ""
