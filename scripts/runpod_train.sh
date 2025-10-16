#!/bin/bash
# RunPod Training Script - Executes GPU-intensive tasks
# Run this after runpod_setup.sh completes

set -e
cd /workspace/moola

echo "=================================="
echo "Moola GPU Training Pipeline"
echo "=================================="

# Configuration
DEVICE="cuda"
EXPERIMENT="runpod-gpu-training"

# Step 1: Generate SimpleLSTM and CNN-Transformer OOF (if needed)
echo -e "\n[STEP 1] Generating GPU Model OOF Predictions..."
echo "Models: simple_lstm, cnn_transformer"
echo "Device: $DEVICE"
echo ""

python3 scripts/regenerate_oof_phase2.py \
  --mode clean \
  --device $DEVICE \
  --data-file data/processed/train_clean.parquet \
  --output-dir data/oof \
  --no-mlflow

echo "✅ Step 1 Complete - OOF predictions generated"
echo ""

# Step 4: TS-TCC Pre-training (if unlabeled data exists)
if [ -f "data/raw/unlabeled_windows.parquet" ]; then
    echo -e "\n[STEP 4] TS-TCC Pre-training on Unlabeled Data..."
    echo "Unlabeled samples: $(python3 -c 'import pandas as pd; print(len(pd.read_parquet(\"data/raw/unlabeled_windows.parquet\")))')"
    echo ""

    # Create output directory
    mkdir -p models/ts_tcc

    # Pre-train encoder
    python3 -m moola.cli pretrain-tcc \
      --unlabeled data/raw/unlabeled_windows.parquet \
      --output models/ts_tcc/pretrained_encoder.pt \
      --device $DEVICE \
      --epochs 100 \
      --patience 15 \
      --batch-size 512 \
      --learning-rate 0.001

    echo "✅ Step 4 Complete - TS-TCC encoder pre-trained"
    echo "Saved: models/ts_tcc/pretrained_encoder.pt"
    echo ""

    PRETRAINED_ENCODER="models/ts_tcc/pretrained_encoder.pt"
else
    echo -e "\n⚠️  Skipping Step 4 - No unlabeled data found"
    echo "Upload data/raw/unlabeled_windows.parquet to enable TS-TCC pre-training"
    echo ""
    PRETRAINED_ENCODER=""
fi

# Step 5: Train with Augmentation (if CleanLab data exists)
if [ -f "data/processed/train_clean_v2.parquet" ]; then
    echo -e "\n[STEP 5] Training with Augmentation + Cleaned Data..."

    # Build pretrained encoder flag
    PRETRAIN_FLAG=""
    if [ ! -z "$PRETRAINED_ENCODER" ] && [ -f "$PRETRAINED_ENCODER" ]; then
        PRETRAIN_FLAG="--pretrained-encoder $PRETRAINED_ENCODER"
        echo "Using pre-trained encoder: $PRETRAINED_ENCODER"
    fi

    python3 scripts/regenerate_oof_phase2.py \
      --mode augmented \
      --use-cleaned-data \
      --device $DEVICE \
      --output-dir data/oof \
      --no-mlflow \
      $PRETRAIN_FLAG

    echo "✅ Step 5 Complete - Augmented training finished"
    echo ""
else
    echo -e "\n⚠️  Skipping Step 5 - No cleaned data found"
    echo "Run CleanLab locally first to create train_clean_v2.parquet"
    echo ""
fi

# Summary
echo "=================================="
echo "GPU Training Complete! 🎉"
echo "=================================="
echo ""
echo "Generated files:"
ls -lh data/oof/*.npy 2>/dev/null | tail -5 || echo "No OOF files found"
echo ""
if [ -f "$PRETRAINED_ENCODER" ]; then
    echo "Pre-trained encoder:"
    ls -lh $PRETRAINED_ENCODER
    echo ""
fi
echo "Download these files to your local machine:"
echo "  scp -P 27424 -i ~/.ssh/id_ed25519 root@213.173.102.99:/workspace/moola/data/oof/*.npy ./data/oof/"
if [ -f "$PRETRAINED_ENCODER" ]; then
    echo "  scp -P 27424 -i ~/.ssh/id_ed25519 root@213.173.102.99:/workspace/moola/$PRETRAINED_ENCODER ./models/ts_tcc/"
fi
echo ""
