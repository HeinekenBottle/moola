#!/bin/bash
# Clean all old files from network storage
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/network-storage.env"

echo "🧹 Cleaning network storage..."

# Delete everything
aws s3 rm s3://hg878tp14w/ --recursive \
    --region "$RUNPOD_S3_REGION" \
    --endpoint-url "$RUNPOD_S3_ENDPOINT"

echo "✅ Network storage cleaned"
