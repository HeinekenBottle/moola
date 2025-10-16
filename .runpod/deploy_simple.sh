#!/bin/bash
# Simple, robust deployment to fresh RunPod

set -e

POD_HOST=${1:-"213.173.110.220"}
POD_PORT=${2:-36832}
SSH_KEY=~/.ssh/id_ed25519

echo "========================================================================"
echo "🚀 DEPLOYING MOOLA - SIMPLE VERSION"
echo "========================================================================"
echo ""

ssh_run() {
    ssh -p "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no "root@$POD_HOST" "$@"
}

# 1. Setup workspace
echo "[1/6] Setting up workspace..."
ssh_run "mkdir -p /workspace/data/processed /workspace/artifacts/pretrained"

# 2. Clone/pull latest code (includes all our fixes)
echo "[2/6] Pulling latest moola code..."
ssh_run "
    if [[ ! -d /workspace/moola ]]; then
        cd /workspace && git clone https://github.com/HeinekenBottle/moola.git
    else
        cd /workspace/moola && git pull origin main
    fi
"

# 3. Upload data and encoder
echo "[3/6] Uploading training data and encoder..."
scp -P "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no data/processed/train_pivot_134.parquet "root@$POD_HOST:/workspace/data/processed/train_pivot_134.parquet"
scp -P "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no data/artifacts/pretrained/encoder_weights.pt "root@$POD_HOST:/workspace/artifacts/pretrained/encoder_weights.pt"

# 4. Setup environment
echo "[4/6] Setting up environment..."
ssh_run "bash -s" << 'SETUP_EOF'
set -e
cd /workspace/data/processed && ln -sf train_pivot_134.parquet train.parquet

python3 -m venv /tmp/moola-venv --system-site-packages
source /tmp/moola-venv/bin/activate

pip install --no-cache-dir -q xgboost imbalanced-learn pytorch-lightning pyarrow pandera click typer hydra-core pydantic pydantic-settings python-dotenv loguru rich mlflow joblib

cd /workspace/moola
pip install --no-cache-dir -q -e . --no-deps

echo "✓ Environment ready"
SETUP_EOF
"

# 5. Verify setup
echo "[5/6] Verifying setup..."
ssh_run "bash -s" << 'VERIFY_EOF'
source /tmp/moola-venv/bin/activate
cd /workspace/moola

python3 -c 'import torch; print(f"✓ PyTorch: {torch.__version__}"); print(f"✓ CUDA: {torch.cuda.is_available()}")'
python3 -c 'from moola.models import CnnTransformerModel; print("✓ Models imported")'
python3 -c 'import pandas as pd; df = pd.read_parquet("/workspace/data/processed/train.parquet"); print(f"✓ Data: {df.shape[0]} samples")'
echo "✓ All checks passed"
VERIFY_EOF
"

# 6. Run training
echo "[6/6] Starting training..."
echo "========================================================================"

ssh_run "bash -s" << 'TRAIN_EOF'
cd /workspace/moola
source /tmp/moola-venv/bin/activate
export PYTHONPATH="/workspace/moola/src:$PYTHONPATH"
export MOOLA_DATA_DIR="/workspace/data"
export MOOLA_ARTIFACTS_DIR="/workspace/artifacts"

echo ""
echo "🚀 CNN-Transformer with Pre-trained Encoder (ENCODER FREEZING FIX APPLIED)"
echo "=========================================================================="
echo ""

python3 -m moola.cli oof \
    --model cnn_transformer \
    --device cuda \
    --seed 1337 \
    --load-pretrained-encoder /workspace/artifacts/pretrained/encoder_weights.pt

echo ""
echo "✅ Training complete!"
TRAIN_EOF

# 7. Download results
echo ""
echo "========================================================================"
echo "[DONE] Downloading results..."

mkdir -p /tmp/cnn_fresh_results
scp -P "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no "root@$POD_HOST:/workspace/artifacts/oof/cnn_transformer/v1/seed_1337.npy" /tmp/cnn_fresh_results/seed_1337.npy

echo ""
echo "========================================================================"
echo "✅ DEPLOYMENT COMPLETE"
echo "========================================================================"
echo ""
echo "📊 Results saved to: /tmp/cnn_fresh_results/seed_1337.npy"
echo ""
