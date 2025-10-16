#!/bin/bash
# Pre-training deployment for Masked LSTM on RunPod RTX 4090
#
# This script deploys and executes masked LSTM pre-training on unlabeled data.
# Pre-training learns robust OHLC representations via masked reconstruction,
# which can then be transferred to SimpleLSTM for improved classification.
#
# Expected Performance (RTX 4090, 24GB VRAM):
# - Pre-training time: ~30-40 minutes (5000 sequences, 50 epochs)
# - Memory usage: ~8-10GB VRAM
# - Batch size: 512 (optimal for RTX 4090)

set -e

POD_HOST=${1:-"213.173.110.220"}
POD_PORT=${2:-36832}
SSH_KEY=~/.ssh/id_ed25519

echo "========================================================================"
echo "🚀 MASKED LSTM PRE-TRAINING DEPLOYMENT"
echo "========================================================================"
echo "  Target: RTX 4090 (24GB VRAM)"
echo "  Unlabeled data: 5000 sequences (1000 base + 4x augmentation)"
echo "  Expected time: 30-40 minutes"
echo "========================================================================"
echo ""

ssh_run() {
    ssh -p "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no "root@$POD_HOST" "$@"
}

# 1. Setup workspace
echo "[1/8] Setting up workspace..."
ssh_run "mkdir -p /workspace/data/processed /workspace/artifacts/pretrained"

# 2. Clone/pull latest code
echo "[2/8] Pulling latest moola code..."
ssh_run "
    if [[ ! -d /workspace/moola ]]; then
        cd /workspace && git clone https://github.com/HeinekenBottle/moola.git
    else
        cd /workspace/moola && git pull origin main
    fi
"

# 3. Generate unlabeled data locally (faster than remote generation)
echo "[3/8] Generating unlabeled data locally..."
if [[ ! -f data/processed/unlabeled_pretrain.parquet ]]; then
    python scripts/generate_unlabeled_data.py \
        --input data/processed/train_pivot_134.parquet \
        --output data/processed/unlabeled_pretrain.parquet \
        --target-count 1000 \
        --augment-factor 4 \
        --seed 1337
    echo "  ✓ Generated: data/processed/unlabeled_pretrain.parquet"
else
    echo "  ✓ Using existing: data/processed/unlabeled_pretrain.parquet"
fi

# 4. Upload unlabeled data
echo "[4/8] Uploading unlabeled data..."
scp -P "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    data/processed/unlabeled_pretrain.parquet \
    "root@$POD_HOST:/workspace/data/processed/unlabeled_pretrain.parquet"

echo "  ✓ Upload complete"

# 5. Setup environment
echo "[5/8] Setting up Python environment..."
ssh_run "bash -s" << 'SETUP_EOF'
set -e

# Create venv with system site-packages (PyTorch already installed)
python3 -m venv /tmp/moola-venv --system-site-packages
source /tmp/moola-venv/bin/activate

# Install dependencies
pip install --no-cache-dir -q xgboost imbalanced-learn pytorch-lightning pyarrow pandera click typer hydra-core pydantic pydantic-settings python-dotenv loguru rich mlflow joblib tqdm

# Install moola in editable mode
cd /workspace/moola
pip install --no-cache-dir -q -e . --no-deps

echo "✓ Environment ready"
SETUP_EOF

# 6. Verify setup
echo "[6/8] Verifying setup..."
ssh_run "bash -s" << 'VERIFY_EOF'
source /tmp/moola-venv/bin/activate
cd /workspace/moola

python3 -c '
import torch
import pandas as pd
from moola.pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Check data
df = pd.read_parquet("/workspace/data/processed/unlabeled_pretrain.parquet")
print(f"✓ Unlabeled data: {len(df)} sequences")

print("✓ All checks passed")
'
VERIFY_EOF

# 7. Run pre-training
echo "[7/8] Starting masked LSTM pre-training..."
echo "========================================================================"

ssh_run "bash -s" << 'TRAIN_EOF'
cd /workspace/moola
source /tmp/moola-venv/bin/activate
export PYTHONPATH="/workspace/moola/src:$PYTHONPATH"
export MOOLA_DATA_DIR="/workspace/data"
export MOOLA_ARTIFACTS_DIR="/workspace/artifacts"

echo ""
echo "🧠 MASKED LSTM PRE-TRAINING (RTX 4090)"
echo "=========================================================================="
echo ""

python3 -c '
import numpy as np
import pandas as pd
from pathlib import Path
from moola.pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer
from moola.config.training_config import (
    MASKED_LSTM_HIDDEN_DIM,
    MASKED_LSTM_NUM_LAYERS,
    MASKED_LSTM_DROPOUT,
    MASKED_LSTM_MASK_RATIO,
    MASKED_LSTM_MASK_STRATEGY,
    MASKED_LSTM_PATCH_SIZE,
    MASKED_LSTM_N_EPOCHS,
    MASKED_LSTM_LEARNING_RATE,
    MASKED_LSTM_BATCH_SIZE,
    MASKED_LSTM_VAL_SPLIT,
    MASKED_LSTM_PATIENCE,
)

# Load unlabeled data
print("[LOADING UNLABELED DATA]")
df = pd.read_parquet("/workspace/data/processed/unlabeled_pretrain.parquet")
X_unlabeled = np.stack(df["ohlc_sequence"].values)
print(f"  Loaded: {len(X_unlabeled)} sequences")
print(f"  Shape: {X_unlabeled.shape}")
print()

# Initialize pre-trainer
print("[INITIALIZING PRE-TRAINER]")
pretrainer = MaskedLSTMPretrainer(
    input_dim=4,
    hidden_dim=MASKED_LSTM_HIDDEN_DIM,
    num_layers=MASKED_LSTM_NUM_LAYERS,
    dropout=MASKED_LSTM_DROPOUT,
    mask_ratio=MASKED_LSTM_MASK_RATIO,
    mask_strategy=MASKED_LSTM_MASK_STRATEGY,
    patch_size=MASKED_LSTM_PATCH_SIZE,
    learning_rate=MASKED_LSTM_LEARNING_RATE,
    batch_size=MASKED_LSTM_BATCH_SIZE,
    device="cuda",
    seed=1337
)
print(f"  Model: BiLSTM-Masked-Autoencoder")
print(f"  Hidden dim: {MASKED_LSTM_HIDDEN_DIM}")
print(f"  Layers: {MASKED_LSTM_NUM_LAYERS}")
print(f"  Mask strategy: {MASKED_LSTM_MASK_STRATEGY}")
print(f"  Mask ratio: {MASKED_LSTM_MASK_RATIO}")
print()

# Run pre-training
save_path = Path("/workspace/artifacts/pretrained/bilstm_encoder.pt")
history = pretrainer.pretrain(
    X_unlabeled=X_unlabeled,
    n_epochs=MASKED_LSTM_N_EPOCHS,
    val_split=MASKED_LSTM_VAL_SPLIT,
    patience=MASKED_LSTM_PATIENCE,
    save_path=save_path,
    verbose=True
)

print()
print("="*70)
print("✅ PRE-TRAINING COMPLETE")
print("="*70)
print(f"  Best val loss: {min(history[\"val_loss\"]):.4f}")
print(f"  Encoder saved: {save_path}")
print("="*70)
'

echo ""
echo "✅ Pre-training complete!"
TRAIN_EOF

# 8. Download pre-trained encoder
echo ""
echo "========================================================================"
echo "[8/8] Downloading pre-trained encoder..."

mkdir -p data/artifacts/pretrained
scp -P "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    "root@$POD_HOST:/workspace/artifacts/pretrained/bilstm_encoder.pt" \
    data/artifacts/pretrained/bilstm_encoder.pt

echo ""
echo "========================================================================"
echo "✅ MASKED LSTM PRE-TRAINING DEPLOYMENT COMPLETE"
echo "========================================================================"
echo ""
echo "📊 Results:"
echo "  Pre-trained encoder: data/artifacts/pretrained/bilstm_encoder.pt"
echo ""
echo "Next steps:"
echo "  1. Fine-tune SimpleLSTM with pre-trained encoder"
echo "  2. Compare to baseline SimpleLSTM (no pre-training)"
echo "  3. Run full pipeline: .runpod/full_pipeline_masked_lstm.sh"
echo ""
echo "========================================================================"
