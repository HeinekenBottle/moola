#!/bin/bash
# Fast 2-Class Training Script (Optimized for RTX 4090)
# Usage: bash fast-train.sh

set -e

echo "⚡ FAST 2-CLASS TRAINING (RTX 4090)"
echo "=================================="

# Environment
cd /workspace/moola
source /tmp/moola-venv/bin/activate
export PYTHONPATH="/workspace/moola/src:$PYTHONPATH"
export MOOLA_DATA_DIR="/workspace/data"
export MOOLA_ARTIFACTS_DIR="/workspace/artifacts"

# Verify setup
echo "🔍 Quick verification..."
python -c "
import torch
import pandas as pd
import numpy as np

print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

df = pd.read_parquet('/workspace/data/processed/train.parquet')
print(f'Data: {df.shape}, Classes: {sorted(df[\"label\"].unique())}')
print('✅ Ready to train')
"

echo ""

# Phase 1: Quick Baseline (5 minutes)
echo "📊 Phase 1: Quick Baselines (CPU)"
echo "================================="

# LogReg (no torch dependency)
echo "Training LogReg..."
python -m moola.cli oof --model logreg --device cpu --seed 1337

# XGB with engineered features
echo "Training XGB..."
python -m moola.cli oof --model xgb --device cpu --seed 1337

# RF
echo "Training RF..."
python -m moola.cli oof --model rf --device cpu --seed 1337

echo ""

# Phase 2: Deep Learning (15-20 minutes)
echo "🧠 Phase 2: Deep Learning (GPU)"
echo "=============================="

# RWKV-TS (faster, less memory)
echo "Training RWKV-TS (25 epochs)..."
python -m moola.cli oof --model rwkv_ts --device cuda --seed 1337 --epochs 25

# CNN-Transformer
echo "Training CNN-Transformer (25 epochs)..."
python -m moola.cli oof --model cnn_transformer --device cuda --seed 1337 --epochs 25

echo ""

# Phase 3: Stack (2 minutes)
echo "🎯 Phase 3: Stacking"
echo "===================="

python -m moola.cli stack-train --seed 1337

echo ""

# Results
echo "📈 RESULTS SUMMARY"
echo "==================="
python -c "
import json
import glob

# Load stack metrics
stack_metrics = json.load(open('/workspace/artifacts/models/stack/metrics.json'))
print(f'Stack Accuracy: {stack_metrics[\"accuracy\"]:.3f}')
print(f'Stack F1: {stack_metrics[\"f1\"]:.3f}')
print(f'Stack ECE: {stack_metrics[\"ece\"]:.3f}')

print('')
print('📊 OOF Results:')
for model in ['logreg', 'rf', 'xgb', 'rwkv_ts', 'cnn_transformer']:
    metrics_file = f'/workspace/artifacts/oof/{model}/v1/metrics.json'
    try:
        with open(metrics_file) as f:
            m = json.load(f)
        print(f'{model:15s}: acc={m[\"accuracy\"]:.3f}, f1={m[\"f1\"]:.3f}')
    except:
        print(f'{model:15s}: Not found')
"

echo ""
echo "🎉 TRAINING COMPLETE!"
echo ""
echo "💾 Artifacts saved to: /workspace/artifacts/"
echo "📥 Download to local: bash .runpod/sync-from-storage.sh artifacts"
echo ""
echo "Expected results: 60-70% stack accuracy on 2-class problem"