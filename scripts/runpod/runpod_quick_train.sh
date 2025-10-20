#!/usr/bin/env bash
#
# RunPod Quick Training - Skip Upload, Just Train
#
# Use this when code/data is already on RunPod and you just want to:
# 1. Run a training experiment
# 2. Download results
#
# Usage:
#   ./scripts/runpod_quick_train.sh <RUNPOD_IP> [ADDITIONAL_ARGS]
#
# Examples:
#   # Basic baseline
#   ./scripts/runpod_quick_train.sh 44.201.123.45
#
#   # With pre-training
#   ./scripts/runpod_quick_train.sh 44.201.123.45 --pretrained-encoder models/bilstm_encoder.pt
#
#   # With augmentation
#   ./scripts/runpod_quick_train.sh 44.201.123.45 --augment-data true --augmentation-ratio 0.5
#

set -euo pipefail

RUNPOD_IP="${1:-}"
shift || true  # Remove first arg, keep rest for training args
TRAIN_ARGS="$*"

SSH_KEY="${HOME}/.ssh/runpod_key"
RUNPOD_USER="ubuntu"
REMOTE_DIR="/workspace/moola"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

if [[ -z "$RUNPOD_IP" ]]; then
    echo "Usage: $0 <RUNPOD_IP> [ADDITIONAL_ARGS]"
    exit 1
fi

log_info "Quick Training on RunPod: $RUNPOD_IP"
log_info "Additional args: ${TRAIN_ARGS:-none}"
echo

# Run training
log_info "Running training..."
ssh -i "$SSH_KEY" "${RUNPOD_USER}@${RUNPOD_IP}" "bash -s" << REMOTE_TRAIN
set -euo pipefail
cd /workspace/moola
source venv/bin/activate 2>/dev/null || true

mkdir -p artifacts/runs

RUN_ID="baseline_\$(date +%Y%m%d_%H%M%S)"

echo "[Training] Starting \$RUN_ID..."
python3 -m moola.cli train \
    --model enhanced_simple_lstm \
    --split data/splits/fwd_chain_v3.json \
    --device cuda \
    --seed 17 \
    ${TRAIN_ARGS} \
    2>&1 | tee "artifacts/runs/\${RUN_ID}.log"

echo "[Training] ✓ Complete: \$RUN_ID"
REMOTE_TRAIN

# Download results
log_info "Downloading results..."
mkdir -p "${LOCAL_DIR}/artifacts/runs"

scp -i "$SSH_KEY" \
    "${RUNPOD_USER}@${RUNPOD_IP}:${REMOTE_DIR}/artifacts/runs/baseline_*.log" \
    "${LOCAL_DIR}/artifacts/runs/" 2>/dev/null || true

scp -i "$SSH_KEY" \
    "${RUNPOD_USER}@${RUNPOD_IP}:${REMOTE_DIR}/experiment_results.jsonl" \
    "${LOCAL_DIR}/" 2>/dev/null || true

log_info "✓ Done! Check artifacts/runs/ and experiment_results.jsonl"
