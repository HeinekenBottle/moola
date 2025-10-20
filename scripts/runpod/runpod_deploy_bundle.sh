#!/usr/bin/env bash
#
# Deploy RunPod Bundle - File Transfer Only
#
# Uploads bundle, runs training, downloads results.
# NO SSH commands except ONE to trigger bootstrap.
#
# Usage:
#   ./scripts/runpod_deploy_bundle.sh <RUNPOD_IP> <BUNDLE_FILE>
#
# Example:
#   ./scripts/runpod_deploy_bundle.sh 44.201.123.45 runpod_bundle_20251019_143022.tar.gz
#

set -euo pipefail

RUNPOD_IP="${1:-}"
BUNDLE_FILE="${2:-}"

SSH_KEY="${HOME}/.ssh/runpod_key"
RUNPOD_USER="ubuntu"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Validate
if [[ -z "$RUNPOD_IP" ]] || [[ -z "$BUNDLE_FILE" ]]; then
    log_error "Usage: $0 <RUNPOD_IP> <BUNDLE_FILE>"
    log_error "Example: $0 44.201.123.45 runpod_bundle_20251019_143022.tar.gz"
    exit 1
fi

if [[ ! -f "$BUNDLE_FILE" ]]; then
    log_error "Bundle file not found: $BUNDLE_FILE"
    log_error "Run ./scripts/build_runpod_bundle.sh first"
    exit 1
fi

if [[ ! -f "$SSH_KEY" ]]; then
    log_error "SSH key not found: $SSH_KEY"
    exit 1
fi

BUNDLE_NAME=$(basename "$BUNDLE_FILE" .tar.gz)
BUNDLE_SIZE=$(du -h "$BUNDLE_FILE" | cut -f1)

log_info "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
log_info "RunPod Deployment - File Transfer Only"
log_info "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
log_info "RunPod IP: $RUNPOD_IP"
log_info "Bundle: $BUNDLE_FILE ($BUNDLE_SIZE)"
echo

# Step 1: Test connection
log_info "[1/4] Testing SSH connection..."
if ! ssh -i "$SSH_KEY" -o ConnectTimeout=10 "${RUNPOD_USER}@${RUNPOD_IP}" echo "OK" &>/dev/null; then
    log_error "Cannot connect to RunPod at $RUNPOD_IP"
    exit 1
fi
log_info "  ✓ Connected"
echo

# Step 2: Upload bundle
log_info "[2/4] Uploading bundle to RunPod..."
log_info "  Size: $BUNDLE_SIZE (this may take a few minutes)"

scp -i "$SSH_KEY" -C "$BUNDLE_FILE" "${RUNPOD_USER}@${RUNPOD_IP}:~/"

log_info "  ✓ Upload complete"
echo

# Step 3: Extract and run bootstrap (ONE SSH command)
log_info "[3/4] Running training on RunPod..."
log_info "  Extracting bundle and starting bootstrap..."
echo

ssh -i "$SSH_KEY" "${RUNPOD_USER}@${RUNPOD_IP}" "bash -s" << REMOTE_CMD
set -euo pipefail

cd ~
echo "[RunPod] Extracting bundle..."
rm -rf runpod_workspace  # Clean old runs
mkdir -p runpod_workspace
tar -xzf "${BUNDLE_NAME}.tar.gz" -C runpod_workspace

cd runpod_workspace
echo "[RunPod] Starting bootstrap..."
bash runpod_bootstrap.sh --augment-data false

echo "[RunPod] ✓ Training complete"
REMOTE_CMD

log_info "  ✓ Training finished"
echo

# Step 4: Download results
log_info "[4/4] Downloading results..."

mkdir -p "${LOCAL_DIR}/artifacts/runs"

# Download experiment results
scp -i "$SSH_KEY" \
    "${RUNPOD_USER}@${RUNPOD_IP}:~/runpod_workspace/moola/experiment_results.jsonl" \
    "${LOCAL_DIR}/" 2>/dev/null || log_warn "  No experiment_results.jsonl found"

# Download training logs
scp -i "$SSH_KEY" \
    "${RUNPOD_USER}@${RUNPOD_IP}:~/runpod_workspace/moola/artifacts/runs/*.log" \
    "${LOCAL_DIR}/artifacts/runs/" 2>/dev/null || log_warn "  No log files found"

# Download saved models
scp -i "$SSH_KEY" \
    "${RUNPOD_USER}@${RUNPOD_IP}:~/runpod_workspace/moola/artifacts/models/*baseline*.pt" \
    "${LOCAL_DIR}/artifacts/models/" 2>/dev/null || log_warn "  No model files found"

log_info "  ✓ Results downloaded"
echo

# Summary
log_info "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
log_info "✓ RunPod Training Complete!"
log_info "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
log_info "Results in:"
log_info "  - experiment_results.jsonl (metrics)"
log_info "  - artifacts/runs/ (logs)"
log_info "  - artifacts/models/ (trained models)"
echo
log_info "View metrics:"
log_info "  cat experiment_results.jsonl | tail -1 | python3 -m json.tool"
echo
log_info "Cleanup RunPod (optional):"
log_info "  ssh -i ~/.ssh/runpod_key ubuntu@${RUNPOD_IP} 'rm -rf ~/runpod_workspace ~/runpod_bundle_*.tar.gz'"
echo
