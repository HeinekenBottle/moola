#!/bin/bash
# Streamlined RunPod Deployment - One Command Solution
# Replaces ALL existing .runpod scripts with a single, reliable workflow
# Usage: ./deploy.sh [deploy|train|status|logs|cleanup]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RUNPOD_VOLUME="hg878tp14w"
RUNPOD_S3_ENDPOINT="https://s3api-eu-ro-1.runpod.io"
RUNPOD_S3_REGION="eu-ro-1"
S3_BUCKET="s3://hg878tp14w"

# Local paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Helper functions
log() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check AWS credentials
check_credentials() {
    if [[ -z "$AWS_ACCESS_KEY_ID" ]] || [[ -z "$AWS_SECRET_ACCESS_KEY" ]]; then
        error "AWS credentials not set!"
        echo "Get them from: https://www.runpod.io/console/user/settings"
        echo "Then run:"
        echo "  export AWS_ACCESS_KEY_ID='your-key'"
        echo "  export AWS_SECRET_ACCESS_KEY='your-secret'"
        exit 1
    fi
}

# Deploy everything to RunPod
deploy_to_runpod() {
    log "🚀 Deploying Moola to RunPod (one-command solution)"

    check_credentials

    # Create deployment package
    log "📦 Creating deployment package..."
    DEPLOY_DIR="/tmp/moola-deploy-$(date +%s)"
    mkdir -p "$DEPLOY_DIR"

    # Copy essential files
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

    # 5. Deployment script (will be created below)
    mkdir -p "$DEPLOY_DIR/scripts"

    success "Package created at $DEPLOY_DIR"

    # Create unified deployment script
    cat > "$DEPLOY_DIR/scripts/start.sh" <<'DEPLOY_EOF'
#!/bin/bash
# Unified RunPod Startup Script - Handles everything automatically
set -e

echo "🚀 Moola Auto-Setup & Training"
echo "=============================="

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

# Create virtual environment if it doesn't exist
if [[ ! -d "venv" ]]; then
    echo "📦 Creating Python environment..."
    python3 -m venv venv
    source venv/bin/activate

    # Install system dependencies
    pip install --upgrade pip setuptools wheel

    # Install PyTorch first (critical)
    echo "🧠 Installing PyTorch..."
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu118

    # Install project dependencies
    echo "📦 Installing project dependencies..."
    pip install -e .
else
    echo "♻️  Using existing environment..."
    source venv/bin/activate
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

    # Upload to S3 with error handling
    log "📤 Uploading to RunPod network storage..."

    # Clear existing deployment
    aws s3 rm "$S3_BUCKET/" --recursive \
        --region "$RUNPOD_S3_REGION" \
        --endpoint-url "$RUNPOD_S3_ENDPOINT" 2>/dev/null || true

    # Upload new deployment
    aws s3 sync "$DEPLOY_DIR/" "$S3_BUCKET/" \
        --region "$RUNPOD_S3_REGION" \
        --endpoint-url "$RUNPOD_S3_ENDPOINT" \
        --exclude "*.tmp" \
        --exclude ".DS_Store"

    success "✅ Deployment complete!"

    # Cleanup
    rm -rf "$DEPLOY_DIR"

    echo ""
    echo "🎉 DEPLOYMENT SUCCESSFUL!"
    echo "========================"
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
    echo "💡 The setup script handles everything automatically!"
}

# Start training on RunPod
start_training() {
    log "🏃 Starting training on RunPod..."

    # Check if we're on RunPod
    if ! hostname | grep -q "runpod"; then
        error "This command must be run on RunPod!"
        echo "First deploy with: $0 deploy"
        echo "Then SSH to RunPod: ssh runpod"
        echo "Then run: $0 train"
        exit 1
    fi

    # Find and run the startup script
    if [[ -f "/workspace/scripts/start.sh" ]]; then
        bash /workspace/scripts/start.sh --train
    elif [[ -f "/runpod-volume/scripts/start.sh" ]]; then
        bash /runpod-volume/scripts/start.sh --train
    else
        error "Startup script not found!"
        echo "Please deploy first: $0 deploy"
        exit 1
    fi
}

# Check status
check_status() {
    log "📊 Checking RunPod status..."

    check_credentials

    # List deployment
    echo "📁 Deployment contents:"
    aws s3 ls "$S3_BUCKET/" --recursive \
        --region "$RUNPOD_S3_REGION" \
        --endpoint-url "$RUNPOD_S3_ENDPOINT" \
        --human-readable

    # Check if we're on RunPod and can check local status
    if hostname | grep -q "runpod"; then
        echo ""
        echo "🖥️  Local RunPod status:"

        # Check storage
        for path in "/workspace" "/runpod-volume"; do
            if [[ -d "$path" ]]; then
                echo "✅ Storage at: $path"
                if [[ -d "$path/artifacts" ]]; then
                    echo "  📊 Artifacts: $(find "$path/artifacts" -type f | wc -l) files"
                fi
                if [[ -d "$path/venv" ]]; then
                    echo "  🐍 Python env: $(ls "$path/venv/bin" | wc -l) packages"
                fi
                break
            fi
        done
    fi
}

# Show logs
show_logs() {
    log "📋 Showing recent logs..."

    if hostname | grep -q "runpod"; then
        # On RunPod - show local logs
        for path in "/workspace/logs" "/runpod-volume/logs"; do
            if [[ -d "$path" ]]; then
                echo "📁 Logs in $path:"
                find "$path" -name "*.log" -exec tail -10 {} + 2>/dev/null | head -50
                break
            fi
        done
    else
        echo "ℹ️  Connect to RunPod to see logs: ssh runpod"
    fi
}

# Cleanup everything
cleanup_all() {
    log "🧹 Cleaning up RunPod deployment..."

    check_credentials

    read -p "❗ This will delete ALL files from network storage. Continue? (type 'DELETE'): " confirm
    if [[ "$confirm" != "DELETE" ]]; then
        echo "Cancelled."
        exit 0
    fi

    # Delete everything from S3
    aws s3 rm "$S3_BUCKET/" --recursive \
        --region "$RUNPOD_S3_REGION" \
        --endpoint-url "$RUNPOD_S3_ENDPOINT"

    success "✅ Cleanup complete!"
}

# Main command handling
case "${1:-deploy}" in
    deploy)
        deploy_to_runpod
        ;;
    train)
        start_training
        ;;
    status)
        check_status
        ;;
    logs)
        show_logs
        ;;
    cleanup)
        cleanup_all
        ;;
    *)
        echo "Usage: $0 {deploy|train|status|logs|cleanup}"
        echo ""
        echo "Commands:"
        echo "  deploy  - Deploy everything to RunPod"
        echo "  train   - Start training (run on RunPod)"
        echo "  status  - Check deployment status"
        echo "  logs    - Show recent logs"
        echo "  cleanup - Clean up all files"
        echo ""
        echo "Quick start:"
        echo "  1. $0 deploy"
        echo "  2. ssh runpod"
        echo "  3. $0 train"
        exit 1
        ;;
esac