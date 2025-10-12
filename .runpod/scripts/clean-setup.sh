#!/bin/bash
# Clean RunPod Setup - Use this instead of pod-startup.sh
# Usage: bash clean-setup.sh

set -e

echo "🧹 CLEAN MOOLA SETUP"
echo "===================="

# Verify network storage
if [[ ! -d "/workspace/data" ]]; then
    echo "❌ Network storage not mounted at /workspace"
    echo "Check volume mount: hg878tp14w"
    exit 1
fi

echo "✅ Network storage found at /workspace"

# Clean up any existing broken installs
echo "🧹 Cleaning up previous installs..."
rm -rf /workspace/venv 2>/dev/null || true
rm -rf /workspace/moola 2>/dev/null || true

# Clone fresh repo
echo "📥 Cloning fresh repo..."
cd /workspace
git clone https://github.com/HeinekenBottle/moola.git
cd moola

# Verify we have our commits
echo "🔍 Checking commits..."
git log --oneline -3
echo ""

# Create clean virtual environment
echo "📦 Creating virtual environment..."
python -m venv /workspace/venv
source /workspace/venv/bin/activate

# Install critical dependencies first
echo "📦 Installing NumPy <2.0 (critical for RunPod)..."
pip install "numpy>=1.26,<2.0"

# Install stable PyTorch (NOT latest!)
echo "📦 Installing PyTorch 2.1.2 with CUDA 11.8..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install rest of dependencies
echo "📦 Installing project dependencies..."
pip install --no-cache-dir -e .

# Verify critical packages
echo "🔍 Verifying installations..."
python -c "
import sys
print(f'Python: {sys.version}')

import numpy as np
print(f'NumPy: {np.__version__}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'CUDA Version: {torch.version.cuda}')

print('✅ All critical imports successful')
"

# Set environment variables
echo "🔧 Setting environment variables..."
export PYTHONPATH="/workspace/moola/src:$PYTHONPATH"
export MOOLA_ARTIFACTS_DIR="/workspace/artifacts"
export MOOLA_DATA_DIR="/workspace/data"
export MOOLA_LOG_DIR="/workspace/logs"

# Add to bashrc
cat >> ~/.bashrc <<EOF

# Moola Environment
export PYTHONPATH="/workspace/moola/src:\$PYTHONPATH"
export MOOLA_ARTIFACTS_DIR="/workspace/artifacts"
export MOOLA_DATA_DIR="/workspace/data"
export MOOLA_LOG_DIR="/workspace/logs"

# Auto-activate venv
source /workspace/venv/bin/activate

# Quick commands
alias moola-train='cd /workspace/moola && python -m moola.cli'
alias moola-test='cd /workspace/moola && python -m moola.cli oof --model logreg --device cpu --seed 1337'
alias moola-status='ls -la /workspace/artifacts/'
EOF

# Test data loading
echo "🔍 Testing data loading..."
python -c "
import pandas as pd
df = pd.read_parquet('/workspace/data/processed/train.parquet')
print(f'Data loaded: {df.shape}')
print(f'Classes: {df[\"label\"].value_counts().to_dict()}')
print('✅ Data loading works')
"

echo ""
echo "🎉 SETUP COMPLETE!"
echo ""
echo "Quick test commands:"
echo "  moola-test    # Test with LogReg (CPU, fast)"
echo "  python -m moola.cli oof --model xgb --device cuda --seed 1337"
echo ""
echo "Training commands:"
echo "  python -m moola.cli oof --model xgb --device cuda --seed 1337"
echo "  python -m moola.cli oof --model rwkv_ts --device cuda --seed 1337 --epochs 50"
echo "  python -m moola.cli stack-train --seed 1337"
echo ""
echo "💡 All artifacts will save to /workspace/artifacts/ and persist!"