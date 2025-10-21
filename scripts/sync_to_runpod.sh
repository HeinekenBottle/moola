#!/bin/bash
# Sync moola project to RunPod for training
# Usage: ./scripts/sync_to_runpod.sh <RUNPOD_IP> [USER]

set -e

# Default user
USER=${2:-ubuntu}

# Check for required arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <RUNPOD_IP> [USER]"
    echo "Example: $0 123.45.67.89 ubuntu"
    exit 1
fi

RUNPOD_IP=$1
SSH_KEY="$HOME/.ssh/id_ed25519_runpod"

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key not found: $SSH_KEY"
    echo "Please ensure your RunPod SSH key is properly configured"
    exit 1
fi

# Test SSH connection
echo "🔍 Testing SSH connection to $USER@$RUNPOD_IP..."
if ! ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$USER@$RUNPOD_IP" "echo 'Connection successful'" 2>/dev/null; then
    echo "❌ Cannot connect to RunPod instance"
    echo "Please check:"
    echo "  1. RunPod instance is running"
    echo "  2. IP address is correct: $RUNPOD_IP"
    echo "  3. SSH key is added to RunPod settings"
    echo "  4. Security group allows SSH (port 22)"
    exit 1
fi

echo "✅ SSH connection successful"

# Create remote directory
echo "📁 Creating remote directory..."
ssh -i "$SSH_KEY" "$USER@$RUNPOD_IP" "mkdir -p /workspace/moola"

# Sync files (excluding heavy data and artifacts)
echo "📦 Syncing project files..."
rsync -avz --progress \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    --exclude 'data/raw' \
    --exclude 'data/processed' \
    --exclude 'artifacts' \
    --exclude '.dvc' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude 'logs' \
    --exclude '.mypy_cache' \
    --exclude '.coverage' \
    --exclude 'htmlcov' \
    ./ "$USER@$RUNPOD_IP:/workspace/moola/"

# Sync essential data files (small ones only)
echo "📊 Syncing essential data..."
rsync -avz --progress \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    --include 'data/processed/labeled/train_latest.parquet' \
    --include 'data/processed/labeled/metadata/' \
    --include 'data/processed/labeled/metadata/*' \
    --exclude 'data/processed/labeled/*' \
    --exclude 'data/processed/*' \
    --exclude 'data/*' \
    ./ "$USER@$RUNPOD_IP:/workspace/moola/"

# Create requirements file for RunPod
echo "📋 Creating RunPod requirements..."
cat > requirements-runpod.txt << 'EOF'
torch>=2.0.0
numpy<2.0.0
pandas
scipy
scikit-learn
xgboost
imbalanced-learn
pytorch-lightning
loguru
click
typer
hydra-core
pydantic
pyarrow
pandera
rich
seaborn
matplotlib
tqdm
EOF

# Sync requirements
rsync -avz --progress \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    requirements-runpod.txt "$USER@$RUNPOD_IP:/workspace/moola/"

echo ""
echo "✅ Sync completed successfully!"
echo ""
echo "Next steps on RunPod:"
echo "  ssh -i $SSH_KEY $USER@$RUNPOD_IP"
echo "  cd /workspace/moola"
echo "  pip install --no-cache-dir -r requirements-runpod.txt"
echo "  python3 -m moola.cli train --model enhanced_simple_lstm --device cuda"
echo ""
echo "To retrieve results:"
echo "  scp -i $SSH_KEY $USER@$RUNPOD_IP:/workspace/moola/experiment_results.jsonl ./"