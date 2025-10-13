#!/bin/bash
# Simple File-by-File Sync (no pagination)
# Uploads essential files one by one to avoid S3 pagination errors
# Usage: bash simple-sync.sh

set -e

echo "📤 SIMPLE FILE SYNC"
echo "==================="

# Configuration (you may need to update these)
S3_BUCKET="s3://hg878tp14w"
S3_ENDPOINT="https://s3api-eu-ro-1.runpod.io"
S3_REGION="eu-ro-1"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts"

# Check credentials
if [[ -z "$AWS_ACCESS_KEY_ID" ]]; then
    echo "❌ AWS_ACCESS_KEY_ID not set"
    echo "Please set your RunPod credentials"
    exit 1
fi

echo "🔧 Configuration:"
echo "  Bucket: $S3_BUCKET"
echo "  Endpoint: $S3_ENDPOINT"
echo "  Scripts: $SCRIPTS_DIR"
echo ""

# Essential files to upload
echo "📋 Uploading essential files..."

# Key scripts for network storage management
key_files=(
    "network-storage-cleanup.sh"
    "network-storage-repopulate.sh"
    "robust-setup.sh"
    "precise-train.sh"
)

success=0
total=${#key_files[@]}

for script in "${key_files[@]}"; do
    local_file="$SCRIPTS_DIR/$script"
    s3_file="$S3_BUCKET/scripts/$script"

    if [[ -f "$local_file" ]]; then
        echo "  📤 $script"
        aws s3 cp "$local_file" "$s3_file" \
            --endpoint-url "$S3_ENDPOINT" \
            --region "$S3_REGION" \
            --no-progress

        if [[ $? -eq 0 ]]; then
            echo "    ✅ Success"
            ((success++))
        else
            echo "    ❌ Failed"
        fi
    else
        echo "  ❌ File not found: $local_file"
    fi
done

echo ""
echo "📊 Upload Summary:"
echo "  Successful: $success/$total"
echo ""

# Quick verification
echo "🔍 Quick verification..."
aws s3 ls "$S3_BUCKET/scripts/" \
    --endpoint-url "$S3_ENDPOINT" \
    --region "$S3_REGION" \
    2>/dev/null | head -5

echo ""
if [[ $success -eq $total ]]; then
    echo "✅ All essential scripts uploaded!"
    echo ""
    echo "🚀 Ready for RunPod:"
    echo "   ssh runpod"
    echo "   bash /runpod-volume/scripts/network-storage-cleanup.sh"
    echo "   bash /runpod-volume/scripts/network-storage-repopulate.sh"
else
    echo "⚠️  Some uploads failed - check AWS credentials"
fi