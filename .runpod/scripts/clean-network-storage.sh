#!/bin/bash
# Complete Network Storage Cleanup
# WARNING: This will DELETE everything in network storage
# Usage: bash clean-network-storage.sh

set -e

echo "🧹 NETWORK STORAGE CLEANUP"
echo "========================"
echo ""
echo "⚠️  WARNING: This will DELETE ALL FILES in network storage!"
echo "   Volume: hg878tp14w (mounted at /workspace)"
echo ""

# Safety check
read -p "Are you sure you want to delete ALL network storage files? (type 'DELETE' to confirm): " confirm
if [[ "$confirm" != "DELETE" ]]; then
    echo "❌ Cleanup cancelled"
    exit 1
fi

echo ""
echo "🔥 Starting cleanup..."

# Show what will be deleted
echo "📋 Current network storage contents:"
if [[ -d "/workspace" ]]; then
    find /workspace -type f -name "*.parquet" -o -name "*.npy" -o -name "*.pkl" -o -name "*.json" -o -name "*.csv" 2>/dev/null | wc -l
    echo "files will be deleted"
else
    echo "Network storage not accessible from this location"
    echo "Run this script from the RunPod pod (ssh runpod)"
    exit 1
fi

echo ""

# Backup critical files before cleanup (optional)
echo "💾 Creating backup of critical configs..."
backup_dir="/tmp/network-storage-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$backup_dir"

if [[ -f "/workspace/data/processed/train_2class.parquet" ]]; then
    cp /workspace/data/processed/train_2class.parquet "$backup_dir/"
    echo "  ✅ Backed up train_2class.parquet"
fi

if [[ -f "/workspace/data/processed/reversals_archive.parquet" ]]; then
    cp /workspace/data/processed/reversals_archive.parquet "$backup_dir/"
    echo "  ✅ Backed up reversals_archive.parquet"
fi

if [[ -f "/workspace/configs/default.yaml" ]]; then
    cp /workspace/configs/default.yaml "$backup_dir/"
    echo "  ✅ Backed up default.yaml"
fi

echo "Backup created at: $backup_dir"
echo ""

# Delete artifacts (models, OOF predictions, logs)
echo "🗑️  Deleting training artifacts..."
rm -rf /workspace/artifacts/
echo "  ✅ Deleted artifacts/"

# Delete logs
rm -rf /workspace/logs/
echo "  ✅ Deleted logs/"

# Delete data (but keep directory structure)
echo "🗑️  Deleting processed data..."
find /workspace/data -name "*.parquet" -delete 2>/dev/null || true
find /workspace/data -name "*.csv" -delete 2>/dev/null || true
find /workspace/data -name "*.npy" -delete 2>/dev/null || true
echo "  ✅ Deleted processed data files"

# Delete virtual environments
echo "🗑️  Deleting virtual environments..."
rm -rf /workspace/venv/
rm -rf /workspace/.venv/
echo "  ✅ Deleted virtual environments"

# Delete temporary files
echo "🗑️  Deleting temporary files..."
find /workspace -name "*.tmp" -delete 2>/dev/null || true
find /workspace -name "*.log" -delete 2>/dev/null || true
find /workspace -name ".build.lock" -delete 2>/dev/null || true
find /workspace -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "  ✅ Deleted temporary files"

# Delete old scripts/repo copies (keep our official ones)
echo "🗑️  Cleaning up old script copies..."
rm -rf /workspace/moola/ 2>/dev/null || true
echo "  ✅ Deleted old repo copies"

# Create minimal directory structure
echo "📁 Creating minimal directory structure..."
mkdir -p /workspace/{data/{raw,processed},artifacts/{models,oof,predictions},logs,configs,scripts}
echo "  ✅ Created directory structure"

# Show what's left
echo ""
echo "📊 Network storage after cleanup:"
echo "================================"
tree /workspace -L 2 2>/dev/null || find /workspace -type d | sort

echo ""
echo "🎉 Network storage cleanup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Run: bash /workspace/scripts/repopulate-storage.sh"
echo "   2. Run: bash /workspace/scripts/robust-setup.sh"
echo ""
echo "💡 Backup available at: $backup_dir"
echo "   (contains train_2class.parquet, reversals_archive.parquet, default.yaml)"