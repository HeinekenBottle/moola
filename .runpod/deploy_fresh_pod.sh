#!/bin/bash
# Deploy moola to fresh RunPod instance
# Usage: bash deploy_fresh_pod.sh <pod_host> <pod_port>

set -e

POD_HOST=${1:-"213.173.110.220"}
POD_PORT=${2:-36832}
SSH_KEY=~/.ssh/id_ed25519
WORKSPACE=/workspace/moola

echo "========================================================================"
echo "🚀 DEPLOYING MOOLA TO FRESH RUNPOD"
echo "========================================================================"
echo "Pod: $POD_HOST:$POD_PORT"
echo "SSH Key: $SSH_KEY"
echo ""

# Helper functions
log() { echo -e "\n[$(date +%H:%M:%S)] $1"; }
check() { echo "  ✓ $1"; }
error() { echo -e "  ✗ ERROR: $1"; exit 1; }

# SSH helper
ssh_cmd() {
    ssh -p "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no "root@$POD_HOST" "$@"
}

scp_upload() {
    scp -P "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no "$@"
}

# Phase 1: Verify PyTorch template
log "Verifying PyTorch 2.4 template..."
PYTORCH_VERSION=$(ssh_cmd "python3 -c 'import torch; print(torch.__version__)'")
[[ $PYTORCH_VERSION == *"2.4"* ]] || error "PyTorch 2.4 not found (got: $PYTORCH_VERSION)"
check "PyTorch $PYTORCH_VERSION"

# Phase 2: Setup workspace
log "Setting up workspace..."
ssh_cmd "mkdir -p $WORKSPACE /workspace/data/processed /workspace/artifacts/pretrained"
check "Directories created"

# Phase 3: Clone/update repository
log "Setting up repository..."
ssh_cmd "cd /workspace && git clone https://github.com/HeinekenBottle/moola.git 2>/dev/null || (cd moola && git pull origin main)" || true
check "Repository ready"

# Phase 4: Upload fixed code
log "Uploading fixed code..."
scp_upload -q src/moola/models/cnn_transformer.py "root@$POD_HOST:$WORKSPACE/src/moola/models/cnn_transformer.py"
check "cnn_transformer.py"

scp_upload -q src/moola/config/training_config.py "root@$POD_HOST:$WORKSPACE/src/moola/config/training_config.py"
check "training_config.py"

scp_upload -q src/moola/validation/training_validator.py "root@$POD_HOST:$WORKSPACE/src/moola/validation/training_validator.py"
check "training_validator.py"

# Phase 5: Upload data and encoder
log "Uploading data and pre-trained encoder..."
[[ -f "data/processed/train_pivot_134.parquet" ]] && {
    scp_upload -q data/processed/train_pivot_134.parquet "root@$POD_HOST:/workspace/data/processed/train_pivot_134.parquet"
    check "Training data"
} || echo "  ⚠ data/processed/train_pivot_134.parquet not found"

[[ -f "data/artifacts/pretrained/encoder_weights.pt" ]] && {
    scp_upload -q data/artifacts/pretrained/encoder_weights.pt "root@$POD_HOST:/workspace/artifacts/pretrained/encoder_weights.pt"
    check "Pre-trained encoder"
} || echo "  ⚠ data/artifacts/pretrained/encoder_weights.pt not found"

# Phase 6: Setup symlinks and verify
log "Verifying setup..."
ssh_cmd "cd /workspace/data/processed && ln -sf train_pivot_134.parquet train.parquet 2>/dev/null || true"
ssh_cmd "python3 -c 'import torch; import pandas as pd; df = pd.read_parquet(\"/workspace/data/processed/train.parquet\"); print(f\"Data: {df.shape[0]} samples, {len(df.columns)} features\"); print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\")'"
check "Environment verified"

# Phase 7: Create venv and install packages
log "Creating virtual environment..."
ssh_cmd "python3 -m venv /tmp/moola-venv --system-site-packages"
ssh_cmd "source /tmp/moola-venv/bin/activate && pip install --no-cache-dir -q xgboost imbalanced-learn pytorch-lightning pyarrow pandera click typer hydra-core pydantic pydantic-settings python-dotenv loguru rich mlflow joblib"
ssh_cmd "cd $WORKSPACE && source /tmp/moola-venv/bin/activate && pip install --no-cache-dir -q -e . --no-deps"
check "Environment ready"

# Phase 8: Run training
log "Starting CNN-Transformer training with pre-trained encoder..."
echo "========================================================================"

ssh_cmd "bash -s" << 'EOF'
cd /workspace/moola
source /tmp/moola-venv/bin/activate
export PYTHONPATH="/workspace/moola/src:$PYTHONPATH"
export MOOLA_DATA_DIR="/workspace/data"
export MOOLA_ARTIFACTS_DIR="/workspace/artifacts"

echo "🚀 CNN-Transformer with Pre-trained Encoder + FIXES"
echo "======================================================"
echo ""

python3 -m moola.cli oof \
    --model cnn_transformer \
    --device cuda \
    --seed 1337 \
    --load-pretrained-encoder /workspace/artifacts/pretrained/encoder_weights.pt

echo ""
echo "✅ Training complete!"
echo "Results: /workspace/artifacts/oof/cnn_transformer/v1/seed_1337.npy"
EOF

# Phase 9: Download results
log "Downloading results..."
mkdir -p /tmp/cnn_fresh_pod_results
scp_upload -q "root@$POD_HOST:/workspace/artifacts/oof/cnn_transformer/v1/seed_1337.npy" /tmp/cnn_fresh_pod_results/seed_1337.npy
check "Results downloaded to /tmp/cnn_fresh_pod_results/"

# Summary
echo ""
echo "========================================================================"
echo "✅ DEPLOYMENT COMPLETE"
echo "========================================================================"
echo ""
echo "📊 Results:"
echo "   Location: /tmp/cnn_fresh_pod_results/"
echo "   File: seed_1337.npy"
echo ""
echo "🔗 Pod Connection:"
echo "   SSH: ssh root@$POD_HOST -p $POD_PORT -i $SSH_KEY"
echo "   Workspace: $WORKSPACE"
echo ""
echo "📥 Download All Artifacts:"
echo "   scp -P $POD_PORT -i $SSH_KEY -r root@$POD_HOST:/workspace/artifacts ~/"
echo ""
