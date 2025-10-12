#!/bin/bash
# Network Storage Cleanup (for RunPod pod)
# This script cleans the S3-backed network storage
# Usage: bash network-storage-cleanup.sh

set -e

echo "🧹 NETWORK STORAGE CLEANUP (RunPod Pod)"
echo "====================================="

# Detect network storage mount point
STORAGE_PATH=""
if [[ -d "/runpod-volume" ]]; then
    STORAGE_PATH="/runpod-volume"
    echo "✅ Found network storage at: /runpod-volume"
elif [[ -d "/workspace/storage" ]]; then
    STORAGE_PATH="/workspace/storage"
    echo "✅ Found network storage at: /workspace/storage"
elif [[ -d "/workspace" && -d "/workspace/data" ]]; then
    STORAGE_PATH="/workspace"
    echo "✅ Found network storage at: /workspace"
else
    echo "❌ Network storage not found!"
    echo "Expected locations:"
    echo "  - /runpod-volume (S3 network volume)"
    echo "  - /workspace/storage (alternative mount)"
    echo "  - /workspace (if volume mounted directly)"
    echo ""
    echo "Check your RunPod pod configuration."
    echo "Volume ID: hg878tp14w"
    exit 1
fi

echo ""
echo "⚠️  WARNING: This will DELETE ALL FILES in network storage!"
echo "   Volume: hg878tp14w"
echo "   Path: $STORAGE_PATH"
echo ""

# Safety check
read -p "Type 'DELETE-ALL' to confirm: " confirm
if [[ "$confirm" != "DELETE-ALL" ]]; then
    echo "❌ Cleanup cancelled"
    exit 1
fi

echo ""
echo "🔥 Starting cleanup of $STORAGE_PATH..."

# Show what will be deleted
echo "📋 Current contents:"
find "$STORAGE_PATH" -type f 2>/dev/null | wc -l
echo "files will be deleted"

# Create backup of critical files (if they exist)
echo ""
echo "💾 Creating backup..."
backup_dir="/tmp/network-storage-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$backup_dir"

critical_files=(
    "$STORAGE_PATH/data/processed/train_2class.parquet"
    "$STORAGE_PATH/data/processed/reversals_archive.parquet"
    "$STORAGE_PATH/configs/default.yaml"
)

backed_up=0
for file in "${critical_files[@]}"; do
    if [[ -f "$file" ]]; then
        filename=$(basename "$file")
        cp "$file" "$backup_dir/"
        echo "  ✅ Backed up $filename"
        ((backed_up++))
    fi
done

if [[ $backed_up -eq 0 ]]; then
    echo "  ℹ️  No critical files found to backup"
fi

echo "Backup created at: $backup_dir"
echo ""

# Delete contents (preserve directory structure)
echo "🗑️  Deleting contents..."

# Artifacts and models
if [[ -d "$STORAGE_PATH/artifacts" ]]; then
    rm -rf "$STORAGE_PATH/artifacts/"*
    echo "  ✅ Cleared artifacts/"
fi

# Logs
if [[ -d "$STORAGE_PATH/logs" ]]; then
    rm -rf "$STORAGE_PATH/logs/"*
    echo "  ✅ Cleared logs/"
fi

# Data files (keep directories)
if [[ -d "$STORAGE_PATH/data" ]]; then
    find "$STORAGE_PATH/data" -type f -delete 2>/dev/null || true
    echo "  ✅ Cleared data/ files"
fi

# Virtual environments
if [[ -d "$STORAGE_PATH/venv" ]]; then
    rm -rf "$STORAGE_PATH/venv"
    echo "  ✅ Deleted venv/"
fi

# Temp files
find "$STORAGE_PATH" -name "*.tmp" -delete 2>/dev/null || true
find "$STORAGE_PATH" -name "*.log" -delete 2>/dev/null || true
find "$STORAGE_PATH" -name ".build.lock" -delete 2>/dev/null || true
find "$STORAGE_PATH" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "  ✅ Deleted temp files"

# Old repo copies
if [[ -d "$STORAGE_PATH/moola" ]]; then
    rm -rf "$STORAGE_PATH/moola"
    echo "  ✅ Deleted old repo copies"
fi

# Recreate minimal directory structure
echo ""
echo "📁 Recreating directory structure..."
mkdir -p "$STORAGE_PATH"/{data/{raw,processed},artifacts/{models,oof,predictions},logs,configs,scripts}
echo "  ✅ Directory structure created"

# Show final state
echo ""
echo "📊 Network storage after cleanup:"
echo "================================="
find "$STORAGE_PATH" -type d | sort

echo ""
echo "🎉 Network storage cleanup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. bash $STORAGE_PATH/scripts/network-storage-repopulate.sh"
echo "   2. Set up training environment"
echo ""
echo "💡 Backup location: $backup_dir"
echo "   (restore if needed: cp $backup_dir/* $STORAGE_PATH/data/processed/)"