#!/bin/bash
# Sync files from RunPod Network Storage to local machine
# Usage: ./sync-from-storage.sh [all|artifacts|logs|models]

set -e

# Load network storage configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/network-storage.env"

# Check for AWS credentials
if [[ -z "$AWS_ACCESS_KEY_ID" ]] || [[ -z "$AWS_SECRET_ACCESS_KEY" ]]; then
    echo "❌ Error: AWS credentials not set"
    echo "Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    exit 1
fi

# Base S3 path
S3_BASE="$RUNPOD_S3_BUCKET"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Function to sync from S3
sync_from_s3() {
    local src="$1"
    local dest="$2"
    local description="$3"

    echo "📥 Downloading $description..."
    mkdir -p "$dest"
    # Use cp with --recursive for better reliability vs sync with pagination issues
    aws s3 cp "$src" "$dest" \
        --recursive \
        --region "$RUNPOD_S3_REGION" \
        --endpoint-url "$RUNPOD_S3_ENDPOINT" \
        --no-progress 2>&1 | grep -v "Completed" || true
    echo "✅ $description downloaded"
}

# Parse command
COMMAND="${1:-all}"

case "$COMMAND" in
    all)
        echo "🚀 Syncing all from RunPod Network Storage..."
        sync_from_s3 "$S3_BASE/artifacts/" "$PROJECT_ROOT/data/artifacts/" "training artifacts"
        sync_from_s3 "$S3_BASE/logs/" "$PROJECT_ROOT/data/logs/" "training logs"
        echo "🎉 All files synced from network storage"
        ;;

    artifacts)
        sync_from_s3 "$S3_BASE/artifacts/" "$PROJECT_ROOT/data/artifacts/" "training artifacts"
        ;;

    logs)
        sync_from_s3 "$S3_BASE/logs/" "$PROJECT_ROOT/data/logs/" "training logs"
        ;;

    models)
        sync_from_s3 "$S3_BASE/artifacts/models/" "$PROJECT_ROOT/data/artifacts/models/" "trained models"
        ;;

    oof)
        sync_from_s3 "$S3_BASE/artifacts/oof/" "$PROJECT_ROOT/data/artifacts/oof/" "OOF predictions"
        ;;

    *)
        echo "Usage: $0 [all|artifacts|logs|models|oof]"
        exit 1
        ;;
esac
