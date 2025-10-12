#!/bin/bash
# Robust RunPod Setup with python3 + pip-tools
# Usage: bash robust-setup.sh

set -e

echo "🔧 ROBUST MOOLA SETUP (python3 + pip-tools)"
echo "=========================================="

# Verify Python version
echo "🐍 Checking Python version..."
python3 --version
if [[ $? -ne 0 ]]; then
    echo "❌ python3 not found!"
    exit 1
fi

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
echo "📦 Creating virtual environment with python3..."
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate

# Upgrade pip and install pip-tools
echo "📦 Installing pip-tools..."
python3 -m pip install --upgrade pip
python3 -m pip install pip-tools

# Install NumPy first (critical for RunPod)
echo "📦 Installing NumPy <2.0 (critical for RunPod)..."
python3 -m pip install "numpy>=1.26,<2.0"

# Install PyTorch with explicit CUDA version
echo "📦 Installing PyTorch 2.1.2 with CUDA 11.8..."
python3 -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Create requirements.in for pip-compile
echo "📝 Creating requirements.in..."
cat > requirements.in <<'EOF'
# Core dependencies
numpy>=1.26,<2.0
pandas>=2.2
pydantic>=2.8
pyyaml>=6.0
rich>=13.7
loguru>=0.7
click>=8.1
hydra-core>=1.3
python-dotenv>=1.0
typer>=0.12
scikit-learn>=1.3
xgboost>=2.0
pyarrow>=14.0
torch>=2.0,<2.2.3
pandera>=0.26.1
openai>=2.3.0

# Additional utilities
tqdm>=4.66
requests>=2.31
joblib>=1.3
seaborn>=0.12
matplotlib>=3.7
EOF

# Compile exact requirements
echo "🔧 Compiling exact requirements with pip-compile..."
python3 -m pip-compile requirements.in --output-file=requirements.txt

# Install exact requirements
echo "📦 Installing exact requirements with pip-sync..."
python3 -m pip-sync requirements.txt

# Install project in development mode
echo "📦 Installing project in development mode..."
python3 -m pip install -e .

# Verify critical packages
echo "🔍 Verifying installations..."
python3 -c "
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

import pandas as pd
import sklearn
import xgboost
print(f'Pandas: {pd.__version__}')
print(f'Sklearn: {sklearn.__version__}')
print(f'XGBoost: {xgboost.__version__}')

print('✅ All critical imports successful')
"

# Show dependency tree
echo "🌳 Showing dependency tree..."
pip-tree | head -20

# Set environment variables
echo "🔧 Setting environment variables..."
export PYTHONPATH="/workspace/moola/src:$PYTHONPATH"
export MOOLA_ARTIFACTS_DIR="/workspace/artifacts"
export MOOLA_DATA_DIR="/workspace/data"
export MOOLA_LOG_DIR="/workspace/logs"

# Add to bashrc
cat >> ~/.bashrc <<EOF

# Moola Environment (robust setup)
export PYTHONPATH="/workspace/moola/src:\$PYTHONPATH"
export MOOLA_ARTIFACTS_DIR="/workspace/artifacts"
export MOOLA_DATA_DIR="/workspace/data"
export MOOLA_LOG_DIR="/workspace/logs"

# Auto-activate venv
source /workspace/venv/bin/activate

# Python3 aliases for consistency
alias pip='python3 -m pip'
alias python='python3'

# Quick commands
alias moola-test='cd /workspace/moola && python3 -m moola.cli oof --model logreg --device cpu --seed 1337'
alias moola-status='ls -la /workspace/artifacts/'
alias moola-deps='cd /workspace/moola && pip-tree'
EOF

# Test data loading
echo "🔍 Testing data loading..."
python3 -c "
import pandas as pd
df = pd.read_parquet('/workspace/data/processed/train.parquet')
print(f'Data loaded: {df.shape}')
print(f'Classes: {df[\"label\"].value_counts().to_dict()}')
print('✅ Data loading works')
"

# Test CLI
echo "🔍 Testing CLI..."
python3 -m moola.cli --help | head -5

echo ""
echo "🎉 ROBUST SETUP COMPLETE!"
echo ""
echo "Environment info:"
echo "  Python: $(python3 --version)"
echo "  Pip: $(python3 -m pip --version)"
echo "  Pip-tools: $(python3 -m pip show pip-tools | grep Version)"
echo ""
echo "Quick test commands:"
echo "  moola-test    # Test with LogReg (CPU, fast)"
echo "  moola-deps    # Show dependency tree"
echo "  python3 -m moola.cli oof --model xgb --device cuda --seed 1337"
echo ""
echo "Training commands:"
echo "  python3 -m moola.cli oof --model xgb --device cuda --seed 1337"
echo "  python3 -m moola.cli oof --model rwkv_ts --device cuda --seed 1337 --epochs 50"
echo "  python3 -m moola.cli stack-train --seed 1337"
echo ""
echo "💡 All artifacts will save to /workspace/artifacts/ and persist!"
echo "🔍 Dependencies locked in requirements.txt for reproducibility"