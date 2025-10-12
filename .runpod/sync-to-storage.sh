#!/bin/bash
# Sync files from local machine to RunPod Network Storage
# Usage: ./sync-to-storage.sh [all|scripts|data|artifacts]

set -e

# Load network storage configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/network-storage.env"

# Check for AWS credentials
if [[ -z "$AWS_ACCESS_KEY_ID" ]] || [[ -z "$AWS_SECRET_ACCESS_KEY" ]]; then
    echo "❌ Error: AWS credentials not set"
    echo "Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    echo "Get them from: https://www.runpod.io/console/user/settings"
    exit 1
fi

# Base S3 path
S3_BASE="$RUNPOD_S3_BUCKET"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Function to sync with progress
sync_to_s3() {
    local src="$1"
    local dest="$2"
    local description="$3"

    echo "📤 Syncing $description..."
    aws s3 sync "$src" "$dest" \
        --region "$RUNPOD_S3_REGION" \
        --endpoint-url "$RUNPOD_S3_ENDPOINT" \
        --exclude ".git/*" \
        --exclude "__pycache__/*" \
        --exclude "*.pyc" \
        --exclude ".venv/*" \
        --exclude "node_modules/*"
    echo "✅ $description synced"
}

# Parse command
COMMAND="${1:-all}"

case "$COMMAND" in
    all)
        echo "🚀 Syncing all to RunPod Network Storage..."
        sync_to_s3 "$SCRIPT_DIR/scripts" "$S3_BASE/scripts/" "deployment scripts"
        sync_to_s3 "$PROJECT_ROOT/data/processed" "$S3_BASE/data/processed/" "processed datasets"
        sync_to_s3 "$PROJECT_ROOT/configs" "$S3_BASE/configs/" "configuration files"
        echo "🎉 All files synced to network storage"
        ;;

    scripts)
        sync_to_s3 "$SCRIPT_DIR/scripts" "$S3_BASE/scripts/" "deployment scripts"
        ;;

    data)
        sync_to_s3 "$PROJECT_ROOT/data/processed" "$S3_BASE/data/processed/" "processed datasets"
        ;;

    artifacts)
        if [[ -d "$PROJECT_ROOT/data/artifacts" ]]; then
            sync_to_s3 "$PROJECT_ROOT/data/artifacts" "$S3_BASE/artifacts/" "training artifacts"
        else
            echo "⚠️  No artifacts directory found (this is normal if you haven't trained yet)"
        fi
        ;;

    configs)
        sync_to_s3 "$PROJECT_ROOT/configs" "$S3_BASE/configs/" "configuration files"
        ;;

    *)
        echo "Usage: $0 [all|scripts|data|artifacts|configs]"
        exit 1
        ;;
esac

echo ""
echo "📊 Network storage contents:"
aws s3 ls --region "$RUNPOD_S3_REGION" --endpoint-url "$RUNPOD_S3_ENDPOINT" "$S3_BASE/" --recursive --human-readable
