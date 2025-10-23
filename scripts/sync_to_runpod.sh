#!/bin/bash
# Sync moola project to RunPod for training
# Usage: ./scripts/sync_to_runpod.sh <RUNPOD_IP> [USER]

set -e

# Default user
USER=${2:-root}
# Default port
PORT=${3:-19301}

# Check for required arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <RUNPOD_IP> [USER] [PORT]"
    echo "Example: $0 123.45.67.89 root 19301"
    exit 1
fi

RUNPOD_IP=$1
SSH_KEY="$HOME/.ssh/id_ed25519"

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key not found: $SSH_KEY"
    echo "Please ensure your RunPod SSH key is properly configured"
    exit 1
fi

# Test SSH connection
echo "🔍 Testing SSH connection to $USER@$RUNPOD_IP:$PORT..."
if ! ssh -i "$SSH_KEY" -p "$PORT" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$USER@$RUNPOD_IP" "echo 'Connection successful'" 2>/dev/null; then
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
ssh -i "$SSH_KEY" -p "$PORT" "$USER@$RUNPOD_IP" "mkdir -p /workspace/moola"

# Sync files (excluding heavy data and artifacts)
echo "📦 Syncing project files..."
rsync -avz --progress --delete --compress \
    -e "ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no" \
    --exclude-from='.rsyncignore' \
    --exclude 'archive/' \
    --exclude 'scratch/' \
    --exclude '.factory/' \
    --exclude '*.md' \
    --exclude 'training_summary.txt' \
    --exclude 'temporal_split.json' \
    --exclude 'test_features.parquet' \
    --exclude 'uv.lock' \
    --exclude '.dvcignore' \
    --exclude '_final_inventory.md' \
    --exclude 'DEPENDENCY_SECURITY_AUDIT.md' \
    --exclude 'COMPLIANCE_REPORT.md' \
    --exclude 'AGENTS_COMPLIANCE_REPORT.md' \
    --exclude 'DUPLICATES_FIXED.md' \
    --exclude 'PATCH_REPORT.md' \
    --exclude 'STONES_*' \
    --exclude 'STAGE4_*' \
    --exclude 'HEAVY_UNTRACKED.md' \
    --exclude 'CLEAN_STRUCTURE.md' \
    --exclude 'PERFORMANCE_*' \
    --exclude 'GPU_UTILIZATION_FIX_REPORT.md' \
    --exclude 'JADE_PRETRAINING_GUIDE.md' \
    --exclude 'RUNPOD_COMMANDS.md' \
    --exclude 'runpod_training_commands.md' \
    --exclude 'REAL_TRAINING_GUIDE.md' \
    --exclude 'DEPLOYMENT_SUMMARY.md' \
    --exclude 'FULL_TRAINING_DEPLOYMENT.md' \
    ./ "$USER@$RUNPOD_IP:/workspace/moola/"

# Skip separate data sync to save disk space
echo "📊 Essential data will be generated on-demand..."

# Use existing requirements.txt for consistency
echo "📋 Using existing requirements.txt for paper-strict dependencies..."

echo ""
echo "✅ Sync completed successfully!"
echo ""
echo "Next steps on RunPod:"
echo "  ssh -i $SSH_KEY -p $PORT $USER@$RUNPOD_IP"
echo "  cd /workspace/moola"
echo "  pip install --no-cache-dir -r requirements.txt"
echo "  pip install -e ."
echo "  python3 -m moola.cli train --model jade --device cuda"
echo ""
echo "To retrieve results:"
echo "  scp -i $SSH_KEY -P $PORT $USER@$RUNPOD_IP:/workspace/moola/experiment_results.jsonl ./"