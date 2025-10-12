#!/bin/bash
# RunPod training script for Moola with Network Storage
# This script runs on RunPod pods with persistent network storage
# Usage: ./runpod-train.sh

set -e

echo "🚀 Moola RunPod Training Script (Network Storage Edition)"
echo "=========================================================="

# Detect network storage location
NETWORK_STORAGE=""
if [[ -d "/runpod-volume" ]]; then
    NETWORK_STORAGE="/runpod-volume"
elif [[ -d "/workspace/storage" ]]; then
    NETWORK_STORAGE="/workspace/storage"
elif [[ -d "/workspace/scripts" && -d "/workspace/data" ]]; then
    # Network volume mounted directly at /workspace
    NETWORK_STORAGE="/workspace"
else
    echo "❌ Network storage not found. Please check your RunPod volume mount."
    exit 1
fi

echo "✅ Network storage found at: $NETWORK_STORAGE"

# Setup paths
WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/moola"
ARTIFACTS_DIR="$NETWORK_STORAGE/artifacts"
DATA_DIR="$NETWORK_STORAGE/data"
LOGS_DIR="$NETWORK_STORAGE/logs"

# Create directories on network storage
mkdir -p "$ARTIFACTS_DIR" "$DATA_DIR" "$LOGS_DIR"

# Environment setup
echo "📦 Setting up environment..."
export PYTHONPATH="$REPO_DIR/src:$PYTHONPATH"
export MOOLA_ARTIFACTS_DIR="$ARTIFACTS_DIR"
export MOOLA_DATA_DIR="$DATA_DIR"
export MOOLA_LOG_DIR="$LOGS_DIR"

# Clone or update repository
if [[ ! -d "$REPO_DIR" ]]; then
    echo "📥 Cloning repository..."
    cd "$WORKSPACE"
    git clone https://github.com/HeinekenBottle/moola.git
else
    echo "🔄 Updating repository..."
    cd "$REPO_DIR"
    git pull
fi

cd "$REPO_DIR"

# Install dependencies (skip if already in venv on network storage)
VENV_DIR="$NETWORK_STORAGE/venv"
if [[ -d "$VENV_DIR" ]]; then
    echo "♻️  Using cached virtual environment from network storage..."
    source "$VENV_DIR/bin/activate"
else
    echo "📦 Installing dependencies (this will be cached on network storage)..."
    uv venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    uv pip install -e .
fi

# Verify CUDA setup
echo "🔍 Verifying GPU setup..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('❌ CUDA not available - training will be slow!')
"

# Check if data exists
if [[ ! -f "$DATA_DIR/processed/window105_train.parquet" ]]; then
    echo "⚠️  Training data not found on network storage"
    echo "Copying from repository..."
    mkdir -p "$DATA_DIR/processed"
    cp data/processed/window105_train.parquet "$DATA_DIR/processed/"
fi

# Run training pipeline
echo "🏃 Starting training pipeline..."

# 1. Verify setup (skip, no doctor command yet)
# echo "📋 1. Verifying setup..."
# python -m moola.cli doctor

# 2. OOF generation - Classical models (CPU)
echo "🤖 2. Generating OOF for classical models (CPU)..."
python -m moola.cli oof --model logreg --device cpu --seed 1337
python -m moola.cli oof --model rf --device cpu --seed 1337
python -m moola.cli oof --model xgb --device cpu --seed 1337

# 3. OOF generation - Deep learning models (GPU)
echo "🧠 3. Generating OOF for deep learning models (GPU)..."
python -m moola.cli oof --model cnn_transformer --device cuda --seed 1337
python -m moola.cli oof --model rwkv_ts --device cuda --seed 1337

# 4. Train RF stacker
echo "🎯 4. Training RandomForest meta-learner..."
python -m moola.cli stack-train --stacker rf --seed 1337

# 5. Full pipeline audit
echo "✅ 5. Running full pipeline audit..."
python -m moola.cli audit

# 6. Test predictions
echo "🔮 6. Testing predictions..."
mkdir -p "$ARTIFACTS_DIR/predictions"
python -m moola.cli predict --model stack \
    --input "$DATA_DIR/processed/window105_train.parquet" \
    --output "$ARTIFACTS_DIR/predictions/stack_test.csv"

echo ""
echo "🎉 Training pipeline completed!"
echo "📊 Results saved to: $ARTIFACTS_DIR"
echo "🔍 Logs saved to: $LOGS_DIR"
echo ""
echo "💡 These files are on persistent network storage and will survive pod termination!"
