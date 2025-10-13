#!/bin/bash
# Robust Script Sync to RunPod Network Storage
# Handles S3 pagination errors by using simpler commands
# Usage: bash sync-scripts-robust.sh

set -e

echo "📤 ROBUST SCRIPT SYNC"
echo "===================="

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
SCRIPTS_DIR="$SCRIPT_DIR/scripts"

echo "🔧 Configuration:"
echo "  S3 Bucket: $S3_BASE"
echo "  Endpoint: $RUNPOD_S3_ENDPOINT"
echo "  Region: $RUNPOD_S3_REGION"
echo "  Scripts: $SCRIPTS_DIR"
echo ""

# Check if scripts directory exists
if [[ ! -d "$SCRIPTS_DIR" ]]; then
    echo "❌ Scripts directory not found: $SCRIPTS_DIR"
    exit 1
fi

echo "📋 Scripts to sync:"
ls -la "$SCRIPTS_DIR/"
echo ""

# Function to sync individual files (more reliable than directory sync)
sync_file() {
    local local_file="$1"
    local s3_path="$2"

    if [[ -f "$local_file" ]]; then
        echo "  📤 Uploading $(basename "$local_file")..."
        aws s3 cp "$local_file" "$s3_path" \
            --region "$RUNPOD_S3_REGION" \
            --endpoint-url "$RUNPOD_S3_ENDPOINT"
        echo "    ✅ $(basename "$local_file") uploaded"
    else
        echo "  ⚠️  File not found: $local_file"
    fi
}

# Sync each script file individually
echo "🚀 Uploading script files..."

# List of scripts to sync
scripts=(
    "clean-network-storage.sh"
    "network-storage-cleanup.sh"
    "network-storage-repopulate.sh"
    "robust-setup.sh"
    "precise-train.sh"
    "pod-startup.sh"
    "runpod-train.sh"
    "setup.sh"
    "train.sh"
    "clean-storage.sh"
    "sync-from-storage.sh"
    "sync-to-storage.sh"
    "fresh-start.sh"
    "sync-scripts-robust.sh"
)

uploaded=0
for script in "${scripts[@]}"; do
    if [[ -f "$SCRIPTS_DIR/$script" ]]; then
        sync_file "$SCRIPTS_DIR/$script" "$S3_BASE/scripts/$script"
        ((uploaded++))
    else
        echo "  ⚠️  Script not found: $script"
    fi
done

echo ""
echo "📊 Sync Summary:"
echo "  Scripts uploaded: $uploaded"
echo "  Target: $S3_BASE/scripts/"
echo ""

# Verify upload by listing (with error handling)
echo "🔍 Verifying upload..."
echo "Files now in network storage scripts/:"

# Simple listing without pagination issues
aws s3 ls "$S3_BASE/scripts/" \
    --region "$RUNPOD_S3_REGION" \
    --endpoint-url "$RUNPOD_S3_ENDPOINT" \
    2>/dev/null || echo "  ⚠️  Could not list files (permissions issue)"

echo ""
echo "✅ Script sync complete!"
echo ""
echo "💡 On RunPod pod, these scripts will be available at:"
echo "   /runpod-volume/scripts/ (or /workspace/storage/scripts/)"
echo ""
echo "🚀 Next steps on RunPod:"
echo "   1. ssh runpod"
echo "   2. bash /runpod-volume/scripts/network-storage-cleanup.sh"
echo "   3. bash /runpod-volume/scripts/network-storage-repopulate.sh"