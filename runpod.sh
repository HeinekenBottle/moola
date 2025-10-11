#!/bin/bash
# RunPod training script for Moola Stage 1→2
# Usage: ./runpod.sh

set -e

echo "🚀 Moola RunPod Training Script"
echo "================================"

# Check if we're on RunPod
if [[ ! -f /proc/driver/nvidia/version ]]; then
    echo "⚠️  Warning: NVIDIA driver not detected. Not on RunPod?"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Environment setup
echo "📦 Setting up environment..."
export PYTHONPATH="/workspace/moola/src:$PYTHONPATH"

# Build GPU Docker image
echo "🔨 Building GPU Docker image..."
docker build -f docker/Dockerfile.gpu -t moola:gpu .

# Verify CUDA setup
echo "🔍 Verifying CUDA setup..."
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi

# Verify PyTorch CUDA
echo "🔍 Verifying PyTorch CUDA..."
docker run --rm --gpus all moola:gpu python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'PyTorch version: {torch.__version__}')"

# Training pipeline
echo "🏃 Starting training pipeline..."

# 1. Verify setup
echo "📋 1. Verifying setup..."
docker run --rm -v ~/moola:/workspace/moola -w /workspace/moola moola:gpu uv run -m moola.cli doctor

# 2. OOF generation - Classical models (CPU)
echo "🤖 2. Generating OOF for classical models..."
docker run --rm -v ~/moola:/workspace/moola -w /workspace/moola moola:gpu uv run -m moola.cli oof --model logreg --device cpu --seed 1337
docker run --rm -v ~/moola:/workspace/moola -w /workspace/moola moola:gpu uv run -m moola.cli oof --model rf --device cpu --seed 1337  
docker run --rm -v ~/moola:/workspace/moola -w /workspace/moola moola:gpu uv run -m moola.cli oof --model xgb --device cpu --seed 1337

# 3. OOF generation - Deep learning models (GPU)
echo "🧠 3. Generating OOF for deep learning models..."
docker run --rm --gpus all -v ~/moola:/workspace/moola -w /workspace/moola moola:gpu uv run -m moola.cli oof --model cnn_transformer --device cuda --seed 1337
docker run --rm --gpus all -v ~/moola:/workspace/moola -w /workspace/moola moola:gpu uv run -m moola.cli oof --model rwkv_ts --device cuda --seed 1337

# 4. Train RF stacker
echo "🎯 4. Training RandomForest meta-learner..."
docker run --rm --gpus all -v ~/moola:/workspace/moola -w /workspace/moola moola:gpu uv run -m moola.cli stack-train --stacker rf --seed 1337

# 5. Full pipeline audit
echo "✅ 5. Running full pipeline audit..."
docker run --rm --gpus all -v ~/moola:/workspace/moola -w /workspace/moola moola:gpu uv run -m moola.cli audit

# 6. Test predictions
echo "🔮 6. Testing predictions..."
docker run --rm -v ~/moola:/workspace/moola -w /workspace/moola moola:gpu uv run -m moola.cli predict --model stack --input data/processed/window105_train.parquet --output data/artifacts/predictions/stack_test.csv

echo "🎉 Training pipeline completed!"
echo "📊 Results saved to data/artifacts/"
echo "🔍 Check logs in data/logs/"
