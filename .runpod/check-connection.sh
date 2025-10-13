#!/bin/bash
# Quick script to test RunPod network storage connection

# Configuration (matches deploy scripts)
RUNPOD_VOLUME="22uv11rdjk"
RUNPOD_S3_ENDPOINT="https://s3api-eu-ro-1.runpod.io"
RUNPOD_S3_REGION="eu-ro-1"
S3_BUCKET="s3://22uv11rdjk"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🔍 Testing RunPod Network Storage Connection"
echo "=========================================="

# Check if credentials are set
if [[ -z "$AWS_ACCESS_KEY_ID" ]] || [[ -z "$AWS_SECRET_ACCESS_KEY" ]]; then
    echo -e "${RED}❌ AWS credentials not set!${NC}"
    echo ""
    echo "Please set your credentials:"
    echo "  export AWS_ACCESS_KEY_ID='your-access-key'"
    echo "  export AWS_SECRET_ACCESS_KEY='your-secret-key'"
    echo ""
    echo "Get them from: https://www.runpod.io/console/user/settings"
    exit 1
fi

echo -e "${GREEN}✅ AWS credentials found${NC}"
echo "  Access Key ID: ${AWS_ACCESS_KEY_ID:0:10}..."
echo "  Secret Key: ${AWS_SECRET_ACCESS_KEY:0:10}..."
echo ""

# Test connection
echo "🔌 Testing connection to $S3_BUCKET..."
echo ""

if aws s3 ls "$S3_BUCKET/" \
    --region "$RUNPOD_S3_REGION" \
    --endpoint-url "$RUNPOD_S3_ENDPOINT" 2>/dev/null; then
    echo -e "${GREEN}✅ Connection successful!${NC}"
    echo "  Bucket is accessible and ready for deployment"
else
    echo -e "${RED}❌ Connection failed!${NC}"
    echo "  Please check:"
    echo "  1. AWS_ACCESS_KEY_ID is correct"
    echo "  2. AWS_SECRET_ACCESS_KEY is correct"
    echo "  3. Bucket ID is correct: $RUNPOD_VOLUME"
    echo "  4. Region is correct: $RUNPOD_S3_REGION"
    echo ""
    echo "  Test with:"
    echo "    aws s3 ls --region $RUNPOD_S3_REGION --endpoint-url $RUNPOD_S3_ENDPOINT $S3_BUCKET"
    exit 1
fi

echo ""
echo "🚀 Ready to deploy! Run: bash deploy-fast.sh deploy"