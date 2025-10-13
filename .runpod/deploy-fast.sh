#!/bin/bash
# FAST RunPod Deployment - No Slow Cleanup
# Optimized version that skips the slow S3 deletion step

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RUNPOD_VOLUME="22uv11rdjk"
RUNPOD_S3_ENDPOINT="https://s3api-eu-ro-1.runpod.io"
RUNPOD_S3_REGION="eu-ro-1"
S3_BUCKET="s3://22uv11rdjk"

# Local paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Helper functions
log() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# AWS credentials - assumes they're already exported in the shell
# No credential check - user has keys exported in another shell

# Deploy everything to RunPod (FAST VERSION)
deploy_to_runpod_fast() {
    log "🚀 FAST Deploy to RunPod (skipping slow cleanup)"

    # No credential check - user has keys exported in shell
    log "📋 Using existing AWS credentials from shell environment"

    # Create deployment package
    log "📦 Creating deployment package..."
    DEPLOY_DIR="/tmp/moola-deploy-$(date +%s)"
    mkdir -p "$DEPLOY_DIR"

    # Copy essential files (minimal set)
    log "📋 Packing essential files..."

    # 1. Configuration
    cp -r "$PROJECT_ROOT/configs" "$DEPLOY_DIR/"

    # 2. Data (only what we need)
    mkdir -p "$DEPLOY_DIR/data/processed"
    cp "$PROJECT_ROOT/data/processed/train_2class.parquet" "$DEPLOY_DIR/data/processed/"
    cp "$PROJECT_ROOT/data/processed/reversals_archive.parquet" "$DEPLOY_DIR/data/processed/" 2>/dev/null || true
    cd "$DEPLOY_DIR/data/processed/"
    ln -sf train_2class.parquet train.parquet
    cd "$PROJECT_ROOT"

    # 3. Project source
    mkdir -p "$DEPLOY_DIR/src"
    cp -r "$PROJECT_ROOT/src" "$DEPLOY_DIR/"

    # 4. Dependencies
    cp "$PROJECT_ROOT/pyproject.toml" "$DEPLOY_DIR/"

    # 5. Deployment scripts
    mkdir -p "$DEPLOY_DIR/scripts"
    # Copy optimized setup script
    cp "$SCRIPT_DIR/scripts/optimized-setup.sh" "$DEPLOY_DIR/scripts/"
    cp "$SCRIPT_DIR/scripts/fast-train.sh" "$DEPLOY_DIR/scripts/"
    cp "$SCRIPT_DIR/scripts/precise-train.sh" "$DEPLOY_DIR/scripts/"
    chmod +x "$DEPLOY_DIR/scripts/"*.sh

    success "Package created at $DEPLOY_DIR"

    # Create unified deployment script
    cat > "$DEPLOY_DIR/scripts/start.sh" <<'DEPLOY_EOF'
#!/bin/bash
# Unified RunPod Startup Script - Handles everything automatically
set -e

echo "🚀 Moola Auto-Setup & Training (FAST VERSION)"
echo "============================================"

# Detect storage location
STORAGE_PATH=""
if [[ -d "/runpod-volume" ]]; then
    STORAGE_PATH="/runpod-volume"
elif [[ -d "/workspace" ]]; then
    STORAGE_PATH="/workspace"
else
    echo "❌ No storage found!"
    exit 1
fi

echo "✅ Storage found at: $STORAGE_PATH"

# Setup workspace
WORKSPACE="$STORAGE_PATH"
cd "$WORKSPACE"

# Use optimized venv (template already has PyTorch!)
if [[ ! -d "/tmp/moola-venv" ]]; then
    echo "📦 Creating lightweight venv (template has PyTorch)..."
    # Use --system-site-packages to inherit torch/numpy/pandas from template
    python3 -m venv /tmp/moola-venv --system-site-packages
    source /tmp/moola-venv/bin/activate

    # Install ONLY extras not in template (~50MB, 30-60 seconds)
    echo "📦 Installing extras (loguru, xgboost, etc.)..."
    pip install --no-cache-dir \
        loguru click rich typer xgboost pandera pyarrow \
        pydantic pyyaml hydra-core python-dotenv

    # Install moola package (editable, no deps)
    echo "📦 Installing moola..."
    pip install --no-cache-dir -e . --no-deps

    echo "✅ Lightweight venv created (~50MB vs 4GB)"
    echo "💾 Storage saved: 4GB (no PyTorch duplication)"
else
    echo "♻️  Using existing lightweight venv..."
    source /tmp/moola-venv/bin/activate
fi

# Verify setup
echo "🔍 Verifying setup..."
python -c "
import torch
import pandas as pd
import sklearn
print(f'✅ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'✅ Pandas: {pd.__version__}')
print(f'✅ Sklearn: {sklearn.__version__}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
"

# Set environment variables
export PYTHONPATH="$WORKSPACE/src:$PYTHONPATH"
export MOOLA_ARTIFACTS_DIR="$WORKSPACE/artifacts"
export MOOLA_DATA_DIR="$WORKSPACE/data"
export MOOLA_LOG_DIR="$WORKSPACE/logs"

# Create directories
mkdir -p artifacts logs

# Check if we should train automatically
if [[ "$1" == "--train" ]] || [[ "$1" == "-t" ]]; then
    echo "🏃 Starting automatic training..."

    # Quick test first
    echo "🧪 Running quick test..."
    python -m moola.cli oof --model logreg --device cpu --seed 1337

    # Full training pipeline
    echo "🤖 Classical models (CPU)..."
    python -m moola.cli oof --model rf --device cpu --seed 1337
    python -m moola.cli oof --model xgb --device cpu --seed 1337

    if torch.cuda.is_available(); then
        echo "🧠 Deep learning models (GPU)..."
        python -m moola.cli oof --model cnn_transformer --device cuda --seed 1337
        python -m moola.cli oof --model rwkv_ts --device cuda --seed 1337
    else
        echo "⚠️  No GPU available, skipping deep learning models"
    fi

    echo "🎯 Training meta-learner..."
    python -m moola.cli stack-train --stacker rf --seed 1337

    echo "✅ Training complete!"
    echo "📊 Results saved to: $WORKSPACE/artifacts"
else
    echo "✅ Setup complete!"
    echo "🚀 Start training with: python -m moola.cli oof --model xgb --device cuda --seed 1337"
fi
DEPLOY_EOF

    chmod +x "$DEPLOY_DIR/scripts/start.sh"

    # Upload to S3 (NO CLEANUP - just overwrite)
    log "📤 Uploading to RunPod network storage (FAST - no cleanup)..."

    # Upload new deployment (will overwrite existing files)
    aws s3 sync "$DEPLOY_DIR/" "$S3_BUCKET/" \
        --region "$RUNPOD_S3_REGION" \
        --endpoint-url "$RUNPOD_S3_ENDPOINT" \
        --exclude "*.tmp" \
        --exclude ".DS_Store" \
        --delete

    success "✅ FAST deployment complete!"

    # Cleanup
    rm -rf "$DEPLOY_DIR"

    echo ""
    echo "🎉 FAST DEPLOYMENT SUCCESSFUL!"
    echo "============================="
    echo ""
    echo "🚀 Next steps:"
    echo "   1. ssh runpod"
    echo "   2. cd /workspace  # or /runpod-volume"
    echo "   3. bash scripts/start.sh --train"
    echo ""
    echo "📊 Your files are now available at:"
    echo "   Data: /workspace/data/processed/"
    echo "   Configs: /workspace/configs/"
    echo "   Scripts: /workspace/scripts/"
    echo ""
    echo "💡 This deployment skipped the slow cleanup step!"
}

# Force wipe everything (if needed)
force_wipe() {
    log "🧹 Force-wiping RunPod storage..."

    # No credential check - user has keys exported in shell
    log "📋 Using existing AWS credentials from shell environment"

    warn "⚠️  This will attempt a fast bucket recreation..."

    # Try to recreate bucket (this is usually instant)
    if aws s3 rb "$S3_BUCKET" --force \
        --region "$RUNPOD_S3_REGION" \
        --endpoint-url "$RUNPOD_S3_ENDPOINT" 2>/dev/null; then
        success "✅ Bucket wiped successfully!"
    else
        warn "Bucket recreation failed, trying alternative method..."

        # Alternative: delete major directories in parallel
        (
            aws s3 rm "$S3_BUCKET/venv/" --recursive \
                --region "$RUNPOD_S3_REGION" \
                --endpoint-url "$RUNPOD_S3_ENDPOINT" 2>/dev/null &
            aws s3 rm "$S3_BUCKET/moola/" --recursive \
                --region "$RUNPOD_S3_REGION" \
                --endpoint-url "$RUNPOD_S3_ENDPOINT" 2>/dev/null &
            wait
        )
        success "✅ Major directories deleted!"
    fi
}

# Main command handling
case "${1:-deploy}" in
    deploy)
        deploy_to_runpod_fast
        ;;
    wipe)
        force_wipe
        ;;
    *)
        echo "Usage: $0 {deploy|wipe}"
        echo ""
        echo "Commands:"
        echo "  deploy  - FAST deploy (skips cleanup, just overwrites)"
        echo "  wipe    - Force wipe everything (use if needed)"
        echo ""
        echo "Quick start:"
        echo "  1. $0 deploy"
        echo "  2. ssh runpod"
        echo "  3. cd /workspace && bash scripts/start.sh --train"
        exit 1
        ;;
esac