#!/bin/bash
# Quick Fix Script for Pre-training Data Issue
# Run this to upload correct data and restart pre-training

set -e

POD_HOST="213.173.110.220"
POD_PORT="36832"
POD_SSH="root@${POD_HOST}"

echo "=========================================="
echo "FIX PRE-TRAINING DATA ISSUE"
echo "=========================================="
echo ""
echo "Issue: Pre-training used 98 samples instead of 11,873"
echo "Fix: Upload correct unlabeled_windows.parquet"
echo ""

# Step 1: Create raw data directory
echo "[1/5] Creating /workspace/data/raw on RunPod..."
ssh -p ${POD_PORT} ${POD_SSH} "mkdir -p /workspace/data/raw"
echo "✅ Directory created"
echo ""

# Step 2: Upload correct data file
echo "[2/5] Uploading unlabeled_windows.parquet (2.2 MB)..."
scp -P ${POD_PORT} \
    ./data/raw/unlabeled_windows.parquet \
    ${POD_SSH}:/workspace/data/raw/
echo "✅ File uploaded"
echo ""

# Step 3: Verify upload
echo "[3/5] Verifying file integrity..."
ssh -p ${POD_PORT} ${POD_SSH} "cd /workspace && python3 -c '
import pandas as pd
import numpy as np

df = pd.read_parquet(\"data/raw/unlabeled_windows.parquet\")
assert len(df) == 11873, f\"Expected 11873 samples, got {len(df)}\"

X_sample = np.stack([np.stack(f) for f in df[\"features\"].head(10)])
assert X_sample.shape == (10, 105, 4), f\"Expected shape (10, 105, 4), got {X_sample.shape}\"

print(f\"✅ Verified: {len(df)} samples with correct shape\")
'"
echo ""

# Step 4: Start pre-training
echo "[4/5] Starting pre-training (expect 20-40 minutes)..."
ssh -p ${POD_PORT} ${POD_SSH} "cd /workspace && \
    nohup python3 scripts/pretrain_tcc_unlabeled.py \
        --unlabeled-path data/raw/unlabeled_windows.parquet \
        --output-path models/ts_tcc/pretrained_encoder.pt \
        --device cuda \
        --epochs 100 \
        --batch-size 512 \
        --patience 15 \
    > pretrain_correct.log 2>&1 &"
echo "✅ Pre-training started in background"
echo ""

# Step 5: Instructions for monitoring
echo "[5/5] Monitor progress:"
echo ""
echo "  # Watch logs in real-time:"
echo "  ssh -p ${POD_PORT} ${POD_SSH} 'tail -f /workspace/pretrain_correct.log'"
echo ""
echo "  # Check GPU usage:"
echo "  ssh -p ${POD_PORT} ${POD_SSH} 'watch -n 1 nvidia-smi'"
echo ""
echo "  # Expected indicators:"
echo "    - Loaded 11,873 unlabeled samples (not 98)"
echo "    - 24 batches per epoch (not 1)"
echo "    - ~5-10 seconds per epoch"
echo "    - Total time: 20-40 minutes"
echo ""
echo "=========================================="
echo "Fix script complete!"
echo "Pre-training is now running with CORRECT data"
echo "=========================================="
