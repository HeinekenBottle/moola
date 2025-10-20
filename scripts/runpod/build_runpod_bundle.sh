#!/usr/bin/env bash
#
# Build RunPod Deployment Bundle
#
# Creates a self-contained package with:
# - Pre-downloaded Python wheels (for FAST offline install)
# - Code
# - Data
# - Single bootstrap script
#
# This eliminates the 15-20 minute "pip install" problem!
#
# Usage:
#   ./scripts/build_runpod_bundle.sh
#
# Output:
#   runpod_bundle_YYYYMMDD_HHMMSS.tar.gz
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUNDLE_DIR="${PROJECT_DIR}/runpod_bundle_build"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BUNDLE_NAME="runpod_bundle_${TIMESTAMP}"
OUTPUT_FILE="${PROJECT_DIR}/${BUNDLE_NAME}.tar.gz"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

cd "$PROJECT_DIR"

log_info "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
log_info "Building RunPod Deployment Bundle"
log_info "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
echo

# Clean old build
if [[ -d "$BUNDLE_DIR" ]]; then
    log_info "Cleaning old build directory..."
    rm -rf "$BUNDLE_DIR"
fi

mkdir -p "$BUNDLE_DIR"

# Step 1: Download Python wheels for offline installation
log_info "[1/6] Downloading Python dependencies as wheels..."
log_info "  (This takes a few minutes but saves 15+ minutes on RunPod!)"

mkdir -p "$BUNDLE_DIR/wheels"

# Download for Linux + CUDA (RunPod environment)
# Use Python 3.10 for compatibility with most RunPod PyTorch images
pip3 download \
    --only-binary :all: \
    --platform manylinux2014_x86_64 \
    --python-version 310 \
    --implementation cp \
    --abi cp310 \
    -r requirements.txt \
    -d "$BUNDLE_DIR/wheels" 2>&1 | grep -v "Collecting\|Downloading" || true

# Also download platform-independent packages
pip3 download \
    --no-binary :none: \
    -r requirements.txt \
    -d "$BUNDLE_DIR/wheels" 2>&1 | grep -v "Collecting\|Downloading\|Requirement already satisfied" || true

WHEEL_COUNT=$(find "$BUNDLE_DIR/wheels" -name "*.whl" -o -name "*.tar.gz" | wc -l | tr -d ' ')
log_info "  ✓ Downloaded $WHEEL_COUNT packages"

# Step 2: Copy code
log_info "[2/6] Copying source code..."
mkdir -p "$BUNDLE_DIR/moola/src"
rsync -a --exclude='__pycache__' --exclude='*.pyc' \
    "$PROJECT_DIR/src/" "$BUNDLE_DIR/moola/src/"
log_info "  ✓ Copied src/"

# Step 3: Copy data
log_info "[3/6] Copying training data..."
mkdir -p "$BUNDLE_DIR/moola/data/processed" \
         "$BUNDLE_DIR/moola/data/splits"

cp "$PROJECT_DIR/data/processed/train_latest.parquet" \
   "$BUNDLE_DIR/moola/data/processed/"

cp "$PROJECT_DIR/data/splits/fwd_chain_v3.json" \
   "$BUNDLE_DIR/moola/data/splits/"

log_info "  ✓ Copied data files"

# Step 4: Copy requirements
log_info "[4/6] Copying requirements.txt..."
cp "$PROJECT_DIR/requirements.txt" "$BUNDLE_DIR/moola/"
log_info "  ✓ Copied requirements.txt"

# Step 5: Create bootstrap script
log_info "[5/6] Creating bootstrap script..."
cat > "$BUNDLE_DIR/runpod_bootstrap.sh" << 'BOOTSTRAP_SCRIPT'
#!/usr/bin/env bash
#
# RunPod Bootstrap Script
#
# Single command to:
# 1. Install dependencies (FAST - from local wheels)
# 2. Run training
# 3. Signal completion
#
# Usage on RunPod:
#   bash runpod_bootstrap.sh [TRAIN_ARGS]
#
# Example:
#   bash runpod_bootstrap.sh --augment-data false
#

set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR"

echo "[Bootstrap] Starting RunPod training workflow..."
echo "[Bootstrap] Working directory: $WORK_DIR"
echo

# Step 1: Install dependencies from local wheels (FAST!)
echo "[Bootstrap] [1/3] Installing dependencies from wheels..."
echo "[Bootstrap]   (This should take ~30 seconds instead of 15+ minutes)"

# Create venv if it doesn't exist
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install from local wheels directory (NO network needed!)
pip install --no-index --find-links=wheels -r moola/requirements.txt --quiet

echo "[Bootstrap]   ✓ Dependencies installed"
echo

# Verify key packages
echo "[Bootstrap] Verifying installations..."
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}')"
python3 -c "import pandas; print(f'  Pandas: {pandas.__version__}')"
python3 -c "import numpy; print(f'  NumPy: {numpy.__version__}')"
python3 -c "import sklearn; print(f'  Scikit-learn: {sklearn.__version__}')"
echo

# Step 2: Run training
echo "[Bootstrap] [2/3] Running training..."
cd moola

mkdir -p artifacts/runs artifacts/models

RUN_ID="baseline_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="artifacts/runs/${RUN_ID}.log"

echo "[Bootstrap]   Run ID: $RUN_ID"
echo "[Bootstrap]   Log: $LOG_FILE"
echo

# Run training with all passed arguments
python3 -m moola.cli train \
    --model enhanced_simple_lstm \
    --split data/splits/fwd_chain_v3.json \
    --device cuda \
    --seed 17 \
    "$@" \
    2>&1 | tee "$LOG_FILE"

echo
echo "[Bootstrap] [3/3] Training complete!"
echo "[Bootstrap]   Results: experiment_results.jsonl"
echo "[Bootstrap]   Log: $LOG_FILE"
echo
echo "[Bootstrap] ✓ DONE - Ready for download"
BOOTSTRAP_SCRIPT

chmod +x "$BUNDLE_DIR/runpod_bootstrap.sh"
log_info "  ✓ Created runpod_bootstrap.sh"

# Step 6: Create README
cat > "$BUNDLE_DIR/README.txt" << 'README'
RunPod Deployment Bundle
========================

This bundle contains everything needed to run training on RunPod:
- Pre-downloaded Python wheels (for fast offline installation)
- Moola source code
- Training data
- Bootstrap script

Files:
  runpod_bootstrap.sh   - Single command to setup + train
  moola/                - Project code and data
  wheels/               - Pre-downloaded Python packages

Usage on RunPod:
  1. Upload this directory via SCP
  2. SSH in and run: bash runpod_bootstrap.sh
  3. Download results

Bundle created: 2025-10-19
README

log_info "  ✓ Created README.txt"

# Step 7: Create tarball
log_info "[6/6] Creating compressed bundle..."
cd "$(dirname "$BUNDLE_DIR")"
tar -czf "$OUTPUT_FILE" -C "$BUNDLE_DIR" .

BUNDLE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
log_info "  ✓ Created bundle: $BUNDLE_SIZE"

# Cleanup
rm -rf "$BUNDLE_DIR"

echo
log_info "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
log_info "✓ Bundle created successfully!"
log_info "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
log_info "Output: $OUTPUT_FILE"
log_info "Size: $BUNDLE_SIZE"
echo
log_info "Next steps:"
log_info "  1. Upload to RunPod: ./scripts/runpod_deploy_bundle.sh <IP> $OUTPUT_FILE"
log_info "  2. Or manually:"
log_info "     scp -i ~/.ssh/runpod_key $OUTPUT_FILE ubuntu@<IP>:~/"
log_info "     ssh -i ~/.ssh/runpod_key ubuntu@<IP>"
log_info "     tar -xzf ${BUNDLE_NAME}.tar.gz && bash runpod_bootstrap.sh"
echo
