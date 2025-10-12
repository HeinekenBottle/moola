#!/bin/bash
# Run full training pipeline
set -e

# Activate environment
source /root/venv/bin/activate

# Set paths
export MOOLA_ARTIFACTS_DIR="/workspace/artifacts"
export MOOLA_DATA_DIR="/workspace/data"
export MOOLA_LOG_DIR="/workspace/logs"

cd /root/moola

echo "🚀 Starting Training Pipeline"
echo "=============================="
echo ""

# OOF for all models
echo "📊 Generating OOF predictions..."
python -m moola.cli oof --model logreg --device cpu --seed 1337
python -m moola.cli oof --model rf --device cpu --seed 1337
python -m moola.cli oof --model xgb --device cpu --seed 1337
python -m moola.cli oof --model rwkv_ts --device cuda --seed 1337
python -m moola.cli oof --model cnn_transformer --device cuda --seed 1337

# Train stacker
echo ""
echo "🎯 Training stacker..."
python -m moola.cli stack-train --stacker rf --seed 1337

# Audit
echo ""
echo "✅ Running audit..."
python -m moola.cli audit

echo ""
echo "🎉 Training complete!"
echo "📊 Results in: /workspace/artifacts/"
