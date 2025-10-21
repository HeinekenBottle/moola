#!/bin/bash
# Sync results and artifacts from RunPod back to local
# Usage: ./scripts/sync_from_runpod.sh <RUNPOD_IP> [USER]

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
    exit 1
fi

# Test SSH connection
echo "🔍 Testing SSH connection to $USER@$RUNPOD_IP..."
if ! ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$USER@$RUNPOD_IP" "echo 'Connection successful'" 2>/dev/null; then
    echo "❌ Cannot connect to RunPod instance"
    exit 1
fi

echo "✅ SSH connection successful"

# Create local artifacts directory if it doesn't exist
mkdir -p artifacts_runpod/{models,encoders,logs}

# Sync results file
echo "📊 Syncing experiment results..."
if ssh -i "$SSH_KEY" "$USER@$RUNPOD_IP" "test -f /workspace/moola/experiment_results.jsonl" 2>/dev/null; then
    scp -i "$SSH_KEY" "$USER@$RUNPOD_IP:/workspace/moola/experiment_results.jsonl" ./artifacts_runpod/
    echo "✅ Results synced to artifacts_runpod/experiment_results.jsonl"
else
    echo "⚠️  No experiment_results.jsonl found on RunPod"
fi

# Sync model artifacts
echo "🤖 Syncing model artifacts..."
rsync -avz --progress \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    --include 'artifacts/models/' \
    --include 'artifacts/models/*.pt' \
    --include 'artifacts/models/*.pkl' \
    --include 'artifacts/encoders/' \
    --include 'artifacts/encoders/*.pt' \
    --exclude 'artifacts/**' \
    --exclude 'artifacts/*' \
    "$USER@$RUNPOD_IP:/workspace/moola/" ./artifacts_runpod/

# Sync logs
echo "📝 Syncing logs..."
if ssh -i "$SSH_KEY" "$USER@$RUNPOD_IP" "test -d /workspace/moola/logs" 2>/dev/null; then
    rsync -avz --progress \
        -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
        "$USER@$RUNPOD_IP:/workspace/moola/logs/" ./artifacts_runpod/logs/
    echo "✅ Logs synced to artifacts_runpod/logs/"
fi

# Show summary
echo ""
echo "✅ Sync from RunPod completed!"
echo ""
echo "Downloaded files:"
if [ -f "./artifacts_runpod/experiment_results.jsonl" ]; then
    echo "  📊 experiment_results.jsonl ($(wc -l < ./artifacts_runpod/experiment_results.jsonl) lines)"
fi

find ./artifacts_runpod/models -name "*.pt" -o -name "*.pkl" 2>/dev/null | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "  🤖 $(basename "$file") ($size)"
done

find ./artifacts_runpod/encoders -name "*.pt" 2>/dev/null | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "  🔧 $(basename "$file") ($size)"
done

echo ""
echo "To view results:"
if [ -f "./artifacts_runpod/experiment_results.jsonl" ]; then
    echo "  python3 -c \"import json; [print(f'{r[\\\"experiment_id\\\"]}: {r[\\\"metrics\\\"][\\\"accuracy\\\"]:.4f}') for r in [json.loads(l) for l in open('artifacts_runpod/experiment_results.jsonl')]]\""
fi