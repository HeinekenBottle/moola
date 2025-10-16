#!/bin/bash
# Full Masked LSTM Pipeline: Pre-training → Fine-tuning → Evaluation
#
# This script executes the complete masked LSTM transfer learning pipeline:
#   1. Pre-train BiLSTM encoder on unlabeled data (masked reconstruction)
#   2. Fine-tune SimpleLSTM with frozen encoder (10 epochs)
#   3. Unfreeze encoder and continue training (full fine-tuning)
#   4. Train baseline SimpleLSTM (no pre-training) for comparison
#   5. Download results and compare performance
#
# Expected Performance (RTX 4090, 24GB VRAM):
# - Total pipeline time: ~50-60 minutes
# - Pre-training: ~30-40 min (5000 sequences, 50 epochs)
# - Fine-tuning: ~10-15 min per model (98 samples, 60 epochs)
# - Expected improvement: +8-12% accuracy over baseline

set -e

POD_HOST=${1:-"213.173.110.220"}
POD_PORT=${2:-36832}
SSH_KEY=~/.ssh/id_ed25519

echo "========================================================================"
echo "🚀 FULL MASKED LSTM PIPELINE - RTX 4090"
echo "========================================================================"
echo "  Pipeline stages:"
echo "    1. Masked LSTM pre-training (30-40 min)"
echo "    2. SimpleLSTM fine-tuning with pre-trained encoder (10-15 min)"
echo "    3. Baseline SimpleLSTM training (10-15 min)"
echo "    4. Performance comparison and analysis"
echo ""
echo "  Total expected time: ~50-60 minutes"
echo "========================================================================"
echo ""

ssh_run() {
    ssh -p "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no "root@$POD_HOST" "$@"
}

# ==============================================================================
# STAGE 1: SETUP AND PRE-TRAINING
# ==============================================================================

echo ""
echo "========================================================================"
echo "STAGE 1: MASKED LSTM PRE-TRAINING"
echo "========================================================================"
echo ""

# 1. Setup workspace
echo "[1/10] Setting up workspace..."
ssh_run "mkdir -p /workspace/data/processed /workspace/artifacts/pretrained /workspace/artifacts/oof/simple_lstm"

# 2. Clone/pull latest code
echo "[2/10] Pulling latest moola code..."
ssh_run "
    if [[ ! -d /workspace/moola ]]; then
        cd /workspace && git clone https://github.com/HeinekenBottle/moola.git
    else
        cd /workspace/moola && git pull origin main
    fi
"

# 3. Generate unlabeled data locally
echo "[3/10] Generating unlabeled data..."
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

# 4. Upload data
echo "[4/10] Uploading data..."
scp -P "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    data/processed/unlabeled_pretrain.parquet \
    "root@$POD_HOST:/workspace/data/processed/unlabeled_pretrain.parquet"

scp -P "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    data/processed/train_pivot_134.parquet \
    "root@$POD_HOST:/workspace/data/processed/train_pivot_134.parquet"

# Create symlink for training data
ssh_run "cd /workspace/data/processed && ln -sf train_pivot_134.parquet train.parquet"

echo "  ✓ Upload complete"

# 5. Setup environment
echo "[5/10] Setting up Python environment..."
ssh_run "bash -s" << 'SETUP_EOF'
set -e

python3 -m venv /tmp/moola-venv --system-site-packages
source /tmp/moola-venv/bin/activate

pip install --no-cache-dir -q xgboost imbalanced-learn pytorch-lightning pyarrow pandera click typer hydra-core pydantic pydantic-settings python-dotenv loguru rich mlflow joblib tqdm

cd /workspace/moola
pip install --no-cache-dir -q -e . --no-deps

echo "✓ Environment ready"
SETUP_EOF

# 6. Verify setup
echo "[6/10] Verifying setup..."
ssh_run "bash -s" << 'VERIFY_EOF'
source /tmp/moola-venv/bin/activate
cd /workspace/moola

python3 -c '
import torch
import pandas as pd
from moola.pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer
from moola.models import SimpleLSTMModel

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Check data
df_unlabeled = pd.read_parquet("/workspace/data/processed/unlabeled_pretrain.parquet")
df_labeled = pd.read_parquet("/workspace/data/processed/train.parquet")
print(f"✓ Unlabeled data: {len(df_unlabeled)} sequences")
print(f"✓ Labeled data: {len(df_labeled)} samples")

print("✓ All checks passed")
'
VERIFY_EOF

# 7. Run pre-training
echo "[7/10] Running masked LSTM pre-training (30-40 min)..."
echo "========================================================================"

ssh_run "bash -s" << 'PRETRAIN_EOF'
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
PRETRAIN_EOF

# ==============================================================================
# STAGE 2: FINE-TUNING WITH PRE-TRAINED ENCODER
# ==============================================================================

echo ""
echo "========================================================================"
echo "STAGE 2: FINE-TUNING WITH PRE-TRAINED ENCODER"
echo "========================================================================"
echo ""

echo "[8/10] Fine-tuning SimpleLSTM with pre-trained encoder (10-15 min)..."
echo "=========================================================================="

ssh_run "bash -s" << 'FINETUNE_EOF'
cd /workspace/moola
source /tmp/moola-venv/bin/activate
export PYTHONPATH="/workspace/moola/src:$PYTHONPATH"
export MOOLA_DATA_DIR="/workspace/data"
export MOOLA_ARTIFACTS_DIR="/workspace/artifacts"

echo ""
echo "🎯 FINE-TUNING SimpleLSTM (WITH PRE-TRAINED ENCODER)"
echo "=========================================================================="
echo ""

python3 -m moola.cli oof \
    --model simple_lstm \
    --device cuda \
    --seed 1337 \
    --load-pretrained-encoder /workspace/artifacts/pretrained/bilstm_encoder.pt

echo ""
echo "✅ Fine-tuning complete!"
FINETUNE_EOF

# ==============================================================================
# STAGE 3: BASELINE TRAINING (NO PRE-TRAINING)
# ==============================================================================

echo ""
echo "========================================================================"
echo "STAGE 3: BASELINE TRAINING (NO PRE-TRAINING)"
echo "========================================================================"
echo ""

echo "[9/10] Training baseline SimpleLSTM (no pre-training) (10-15 min)..."
echo "=========================================================================="

ssh_run "bash -s" << 'BASELINE_EOF'
cd /workspace/moola
source /tmp/moola-venv/bin/activate
export PYTHONPATH="/workspace/moola/src:$PYTHONPATH"
export MOOLA_DATA_DIR="/workspace/data"
export MOOLA_ARTIFACTS_DIR="/workspace/artifacts"

echo ""
echo "📊 BASELINE SimpleLSTM (NO PRE-TRAINING)"
echo "=========================================================================="
echo ""

python3 -m moola.cli oof \
    --model simple_lstm \
    --device cuda \
    --seed 42

echo ""
echo "✅ Baseline training complete!"
BASELINE_EOF

# ==============================================================================
# STAGE 4: DOWNLOAD RESULTS AND COMPARE
# ==============================================================================

echo ""
echo "========================================================================"
echo "STAGE 4: DOWNLOADING RESULTS"
echo "========================================================================"
echo ""

echo "[10/10] Downloading artifacts..."

mkdir -p data/artifacts/pretrained
mkdir -p data/artifacts/oof/simple_lstm/v1

# Download pre-trained encoder
scp -P "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    "root@$POD_HOST:/workspace/artifacts/pretrained/bilstm_encoder.pt" \
    data/artifacts/pretrained/bilstm_encoder.pt

# Download fine-tuned results (seed 1337)
scp -P "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    "root@$POD_HOST:/workspace/artifacts/oof/simple_lstm/v1/seed_1337.npy" \
    data/artifacts/oof/simple_lstm/v1/seed_1337_pretrained.npy

# Download baseline results (seed 42)
scp -P "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    "root@$POD_HOST:/workspace/artifacts/oof/simple_lstm/v1/seed_42.npy" \
    data/artifacts/oof/simple_lstm/v1/seed_42_baseline.npy

echo ""
echo "========================================================================"
echo "✅ FULL MASKED LSTM PIPELINE COMPLETE"
echo "========================================================================"
echo ""
echo "📊 Results saved to:"
echo "  Pre-trained encoder: data/artifacts/pretrained/bilstm_encoder.pt"
echo "  Fine-tuned OOF:      data/artifacts/oof/simple_lstm/v1/seed_1337_pretrained.npy"
echo "  Baseline OOF:        data/artifacts/oof/simple_lstm/v1/seed_42_baseline.npy"
echo ""
echo "Next steps:"
echo "  1. Compare performance: python scripts/compare_masked_lstm_results.py"
echo "  2. Analyze per-class improvements"
echo "  3. Visualize learning curves"
echo ""
echo "Expected improvements:"
echo "  - Overall accuracy: +8-12%"
echo "  - Class 1 recall: 0% → 45-55%"
echo "  - Class imbalance handling: Significantly improved"
echo ""
echo "========================================================================"
