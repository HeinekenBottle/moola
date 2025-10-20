#!/usr/bin/env bash
#
# RunPod Baseline Training - Master Control Script
#
# ONE COMMAND to:
# 1. Upload code + data to RunPod
# 2. Setup environment (dependencies)
# 3. Run baseline training
# 4. Download results
#
# Usage:
#   ./scripts/runpod_baseline_workflow.sh <RUNPOD_IP>
#
# Example:
#   ./scripts/runpod_baseline_workflow.sh 44.201.123.45
#

set -euo pipefail

# Configuration
RUNPOD_IP="${1:-}"
SSH_KEY="${HOME}/.ssh/runpod_key"
RUNPOD_USER="ubuntu"
REMOTE_DIR="/workspace/moola"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Validate inputs
if [[ -z "$RUNPOD_IP" ]]; then
    log_error "Usage: $0 <RUNPOD_IP>"
    log_error "Example: $0 44.201.123.45"
    exit 1
fi

if [[ ! -f "$SSH_KEY" ]]; then
    log_error "SSH key not found: $SSH_KEY"
    exit 1
fi

log_info "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
log_info "RunPod Baseline Training Workflow"
log_info "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
log_info "RunPod IP: $RUNPOD_IP"
log_info "Local directory: $LOCAL_DIR"
log_info "Remote directory: $REMOTE_DIR"
echo

# Step 1: Test SSH connection
log_info "[1/5] Testing SSH connection..."
if ! ssh -i "$SSH_KEY" -o ConnectTimeout=10 "${RUNPOD_USER}@${RUNPOD_IP}" echo "Connected" &>/dev/null; then
    log_error "Cannot connect to RunPod at $RUNPOD_IP"
    log_error "Check:"
    log_error "  1. RunPod instance is running"
    log_error "  2. IP address is correct"
    log_error "  3. SSH key permissions: chmod 600 ~/.ssh/runpod_key"
    exit 1
fi
log_info "✓ SSH connection successful"
echo

# Step 2: Upload code and data
log_info "[2/5] Uploading code and data to RunPod..."

# Create remote directory
ssh -i "$SSH_KEY" "${RUNPOD_USER}@${RUNPOD_IP}" "mkdir -p $REMOTE_DIR"

# Upload src/ directory (code)
log_info "  Uploading src/..."
rsync -avz --delete -e "ssh -i $SSH_KEY" \
    "${LOCAL_DIR}/src/" \
    "${RUNPOD_USER}@${RUNPOD_IP}:${REMOTE_DIR}/src/"

# Upload data/ directory (only what's needed)
log_info "  Uploading data/processed/train_latest.parquet..."
ssh -i "$SSH_KEY" "${RUNPOD_USER}@${RUNPOD_IP}" "mkdir -p ${REMOTE_DIR}/data/processed ${REMOTE_DIR}/data/splits"

scp -i "$SSH_KEY" \
    "${LOCAL_DIR}/data/processed/train_latest.parquet" \
    "${RUNPOD_USER}@${RUNPOD_IP}:${REMOTE_DIR}/data/processed/"

log_info "  Uploading data/splits/fwd_chain_v3.json..."
scp -i "$SSH_KEY" \
    "${LOCAL_DIR}/data/splits/fwd_chain_v3.json" \
    "${RUNPOD_USER}@${RUNPOD_IP}:${REMOTE_DIR}/data/splits/"

# Upload requirements
log_info "  Uploading requirements.txt..."
scp -i "$SSH_KEY" \
    "${LOCAL_DIR}/requirements.txt" \
    "${RUNPOD_USER}@${RUNPOD_IP}:${REMOTE_DIR}/"

log_info "✓ Upload complete"
echo

# Step 3: Setup environment
log_info "[3/5] Setting up RunPod environment..."
ssh -i "$SSH_KEY" "${RUNPOD_USER}@${RUNPOD_IP}" "bash -s" << 'REMOTE_SETUP'
set -euo pipefail

cd /workspace/moola

echo "[RunPod Setup] Installing Python dependencies..."

# Check if we're in a venv, if not create one
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ ! -d "venv" ]]; then
        echo "[RunPod Setup] Creating virtual environment..."
        python3 -m venv venv
    fi
    source venv/bin/activate
fi

# Upgrade pip
pip install --upgrade pip --quiet

# Install requirements
pip install -r requirements.txt --quiet

# Verify key packages
echo "[RunPod Setup] Verifying installations..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import pandas; print(f'Pandas: {pandas.__version__}')"

echo "[RunPod Setup] ✓ Environment ready"
REMOTE_SETUP

log_info "✓ Environment setup complete"
echo

# Step 4: Run baseline training
log_info "[4/5] Running baseline training..."
log_info "  Model: enhanced_simple_lstm"
log_info "  Split: fwd_chain_v3.json"
log_info "  Device: cuda"
log_info "  Augmentation: disabled"
echo

ssh -i "$SSH_KEY" "${RUNPOD_USER}@${RUNPOD_IP}" "bash -s" << 'REMOTE_TRAIN'
set -euo pipefail

cd /workspace/moola

# Activate venv if exists
if [[ -d "venv" ]]; then
    source venv/bin/activate
fi

# Create artifacts directory
mkdir -p artifacts/runs artifacts/models

# Run training
echo "[RunPod Training] Starting EnhancedSimpleLSTM baseline..."
python3 -m moola.cli train \
    --model enhanced_simple_lstm \
    --split data/splits/fwd_chain_v3.json \
    --device cuda \
    --augment-data false \
    --seed 17 \
    2>&1 | tee "artifacts/runs/baseline_$(date +%Y%m%d_%H%M%S).log"

echo "[RunPod Training] ✓ Training complete"
REMOTE_TRAIN

log_info "✓ Training complete"
echo

# Step 5: Download results
log_info "[5/5] Downloading results from RunPod..."

# Create local results directory
mkdir -p "${LOCAL_DIR}/artifacts/runs"

# Download logs
log_info "  Downloading training logs..."
scp -i "$SSH_KEY" \
    "${RUNPOD_USER}@${RUNPOD_IP}:${REMOTE_DIR}/artifacts/runs/baseline_*.log" \
    "${LOCAL_DIR}/artifacts/runs/" || log_warn "No log files found"

# Download experiment results
log_info "  Downloading experiment_results.jsonl..."
scp -i "$SSH_KEY" \
    "${RUNPOD_USER}@${RUNPOD_IP}:${REMOTE_DIR}/experiment_results.jsonl" \
    "${LOCAL_DIR}/" || log_warn "No experiment results found"

# Download any saved models
log_info "  Downloading saved models..."
scp -i "$SSH_KEY" \
    "${RUNPOD_USER}@${RUNPOD_IP}:${REMOTE_DIR}/artifacts/models/*baseline*.pt" \
    "${LOCAL_DIR}/artifacts/models/" 2>/dev/null || log_warn "No model files found"

log_info "✓ Results downloaded"
echo

# Summary
log_info "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
log_info "Baseline Training Complete!"
log_info "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
log_info "Results are in:"
log_info "  - artifacts/runs/ (training logs)"
log_info "  - experiment_results.jsonl (metrics)"
log_info "  - artifacts/models/ (saved models)"
echo
log_info "Next steps:"
log_info "  1. Analyze results: cat experiment_results.jsonl | tail -1 | python3 -m json.tool"
log_info "  2. Review logs: tail -100 artifacts/runs/baseline_*.log"
log_info "  3. Compare metrics: PR-AUC, Brier, ECE"
echo
