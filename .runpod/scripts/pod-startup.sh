#!/bin/bash
# RunPod Pod Startup Script
# Run this when you first SSH into a new RunPod pod
# Usage: bash pod-startup.sh

set -e

echo "🚀 Moola RunPod Pod Startup"
echo "============================"

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
    echo "❌ Network storage not detected!"
    echo "Please mount your RunPod network volume first."
    echo "Volume ID: hg878tp14w"
    exit 1
fi

echo "✅ Network storage detected at: $NETWORK_STORAGE"

# Setup workspace
WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/moola"

# Clone repository if not exists
if [[ ! -d "$REPO_DIR" ]]; then
    echo "📥 Cloning Moola repository..."
    cd "$WORKSPACE"
    git clone https://github.com/HeinekenBottle/moola.git
    cd "$REPO_DIR"
else
    echo "♻️  Repository already exists, updating..."
    cd "$REPO_DIR"
    git pull
fi

# Setup Python environment
VENV_DIR="$NETWORK_STORAGE/venv"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "📦 Creating virtual environment on network storage (one-time setup)..."
    cd "$REPO_DIR"

    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        echo "Installing uv..."
        pip install uv
    fi

    uv venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    uv pip install -e .
    echo "✅ Virtual environment created and cached on network storage"
else
    echo "♻️  Using cached virtual environment from network storage"
    source "$VENV_DIR/bin/activate"
fi

# Create symlinks for artifacts/data to network storage
echo "🔗 Creating symlinks to network storage..."
ln -sf "$NETWORK_STORAGE/artifacts" "$REPO_DIR/data/artifacts"
ln -sf "$NETWORK_STORAGE/data" "$REPO_DIR/data/network-data"
ln -sf "$NETWORK_STORAGE/logs" "$REPO_DIR/data/logs"

# Verify GPU
echo "🔍 Checking GPU availability..."
nvidia-smi || echo "⚠️  nvidia-smi failed, but may still work in Python"

python -c "
import torch
print()
if torch.cuda.is_available():
    print('✅ GPU Available:')
    print(f'   Device: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'   CUDA Version: {torch.version.cuda}')
else:
    print('❌ GPU NOT available - check your pod configuration!')
print()
"

# Set environment variables for this session
export PYTHONPATH="$REPO_DIR/src:$PYTHONPATH"
export MOOLA_ARTIFACTS_DIR="$NETWORK_STORAGE/artifacts"
export MOOLA_DATA_DIR="$NETWORK_STORAGE/data"
export MOOLA_LOG_DIR="$NETWORK_STORAGE/logs"

# Add to .bashrc for future sessions
cat >> ~/.bashrc <<EOF

# Moola environment
export PYTHONPATH="$REPO_DIR/src:\$PYTHONPATH"
export MOOLA_ARTIFACTS_DIR="$NETWORK_STORAGE/artifacts"
export MOOLA_DATA_DIR="$NETWORK_STORAGE/data"
export MOOLA_LOG_DIR="$NETWORK_STORAGE/logs"

# Activate virtual environment
if [[ -d "$VENV_DIR" ]]; then
    source $VENV_DIR/bin/activate
fi

# Quick commands
alias moola-train='bash $NETWORK_STORAGE/scripts/runpod-train.sh'
alias moola-status='ls -lh $NETWORK_STORAGE/artifacts/'
alias moola-logs='tail -f $NETWORK_STORAGE/logs/*.log'
EOF

echo ""
echo "✅ Pod setup complete!"
echo ""
echo "📝 Quick reference:"
echo "   - Repository: $REPO_DIR"
echo "   - Network storage: $NETWORK_STORAGE"
echo "   - Virtual env: $VENV_DIR"
echo ""
echo "🚀 To start training:"
echo "   cd $REPO_DIR"
echo "   bash $NETWORK_STORAGE/scripts/runpod-train.sh"
echo ""
echo "   Or use the alias: moola-train"
echo ""
echo "💡 All artifacts will be saved to network storage and persist after pod termination!"
