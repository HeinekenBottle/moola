#!/bin/bash
# Network Storage Repopulation (for RunPod pod)
# This script restores essential files to network storage
# Usage: bash network-storage-repopulate.sh

set -e

echo "📦 NETWORK STORAGE REPOPULATION (RunPod Pod)"
echo "==========================================="

# Detect network storage mount point
STORAGE_PATH=""
if [[ -d "/runpod-volume" ]]; then
    STORAGE_PATH="/runpod-volume"
    echo "✅ Using network storage: /runpod-volume"
elif [[ -d "/workspace/storage" ]]; then
    STORAGE_PATH="/workspace/storage"
    echo "✅ Using network storage: /workspace/storage"
elif [[ -d "/workspace" && -d "/workspace/data" ]]; then
    STORAGE_PATH="/workspace"
    echo "✅ Using network storage: /workspace"
else
    echo "❌ Network storage not found!"
    exit 1
fi

echo ""
echo "📥 Repopulating network storage at: $STORAGE_PATH"

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p "$STORAGE_PATH"/{data/{raw,processed,splits/v1},artifacts/{models,oof,predictions},logs,configs,scripts}
echo "  ✅ Directory structure created"

# Clone fresh repository to get essential files
echo "📥 Cloning repository..."
cd /tmp  # Use temp directory to avoid conflicts
rm -rf moola-repo 2>/dev/null || true
git clone https://github.com/HeinekenBottle/moola.git moola-repo

if [[ ! -d "/tmp/moola-repo" ]]; then
    echo "❌ Failed to clone repository"
    exit 1
fi

echo "  ✅ Repository cloned"

# Copy essential files to network storage
echo "📋 Copying essential files..."

# Scripts
cp /tmp/moola-repo/.runpod/scripts/* "$STORAGE_PATH/scripts/"
chmod +x "$STORAGE_PATH/scripts/"*.sh
echo "  ✅ Scripts copied"

# Configs
cp /tmp/moola-repo/configs/* "$STORAGE_PATH/configs/"
echo "  ✅ Configs copied"

# Data files (if they exist in repo)
if [[ -f "/tmp/moola-repo/data/processed/train_2class.parquet" ]]; then
    cp /tmp/moola-repo/data/processed/train_2class.parquet "$STORAGE_PATH/data/processed/"
    echo "  ✅ train_2class.parquet copied (115 samples)"
else
    echo "  ⚠️  train_2class.parquet not found in repo"
fi

if [[ -f "/tmp/moola-repo/data/processed/reversals_archive.parquet" ]]; then
    cp /tmp/moola-repo/data/processed/reversals_archive.parquet "$STORAGE_PATH/data/processed/"
    echo "  ✅ reversals_archive.parquet copied (19 samples)"
else
    echo "  ⚠️  reversals_archive.parquet not found in repo"
fi

if [[ -f "/tmp/moola-repo/data/processed/train_3class.parquet" ]]; then
    cp /tmp/moola-repo/data/processed/train_3class.parquet "$STORAGE_PATH/data/processed/"
    echo "  ✅ train_3class.parquet copied (backup)"
else
    echo "  ⚠️  train_3class.parquet not found in repo"
fi

# Create symlink for train.parquet
if [[ -f "$STORAGE_PATH/data/processed/train_2class.parquet" ]]; then
    cd "$STORAGE_PATH/data/processed/"
    ln -sf train_2class.parquet train.parquet
    echo "  ✅ Created symlink: train.parquet → train_2class.parquet"
else
    echo "  ❌ Cannot create symlink - train_2class.parquet missing"
fi

# Copy examples if they exist
if [[ -d "/tmp/moola-repo/examples" ]]; then
    cp -r /tmp/moola-repo/examples "$STORAGE_PATH/"
    echo "  ✅ Examples copied"
fi

# Copy README if exists
if [[ -f "/tmp/moola-repo/README.md" ]]; then
    cp /tmp/moola-repo/README.md "$STORAGE_PATH/"
    echo "  ✅ README copied"
fi

# Create storage manifest
echo "📋 Creating storage manifest..."
cat > "$STORAGE_PATH/storage-manifest.json" <<EOF
{
  "created": "$(date -Iseconds)",
  "purpose": "Moola 2-class training network storage",
  "volume_id": "hg878tp14w",
  "mount_point": "$STORAGE_PATH",
  "directory_structure": {
    "data/": "Training datasets (2-class and archived)",
    "configs/": "Configuration files (2-class enabled)",
    "scripts/": "Deployment and training scripts",
    "artifacts/": "Training artifacts (models, OOF, predictions)",
    "logs/": "Training logs"
  },
  "key_files": {
    "data/processed/train.parquet": "Symlink to train_2class.parquet (115 samples)",
    "data/processed/train_2class.parquet": "2-class training data (consolidation vs retracement)",
    "data/processed/reversals_archive.parquet": "Archived reversal samples (19 samples)",
    "configs/default.yaml": "2-class configuration with documentation"
  },
  "next_steps": [
    "Clone repo to pod: git clone https://github.com/HeinekenBottle/moola.git",
    "Setup environment: python3 -m venv venv && source venv/bin/activate",
    "Install dependencies: pip install -e .",
    "Start training: python3 -m moola.cli oof --model xgb --device cuda --seed 1337"
  ],
  "expected_performance": {
    "target_accuracy": "> 0.65",
    "target_f1": "> 0.60",
    "baseline_accuracy": "> 0.55"
  }
}
EOF

echo "  ✅ Storage manifest created"

# Verify essential files exist
echo ""
echo "🔍 Verification check..."
essential_files=(
    "$STORAGE_PATH/scripts/robust-setup.sh"
    "$STORAGE_PATH/scripts/precise-train.sh"
    "$STORAGE_PATH/configs/default.yaml"
    "$STORAGE_PATH/data/processed/train.parquet"
    "$STORAGE_PATH/storage-manifest.json"
)

missing_files=()
for file in "${essential_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "  ✅ $(basename "$file")"
    else
        echo "  ❌ $(basename "$file") - MISSING"
        missing_files+=("$file")
    fi
done

# Clean up temporary repo
echo ""
echo "🧹 Cleaning up temporary files..."
rm -rf /tmp/moola-repo
echo "  ✅ Temporary repo removed"

# Error if essential files missing
if [[ ${#missing_files[@]} -gt 0 ]]; then
    echo ""
    echo "⚠️  Some essential files are missing!"
    echo "Missing files:"
    for file in "${missing_files[@]}"; do
        echo "   $file"
    done
    echo ""
    echo "The repository might not have the data files committed."
    echo "You may need to manually copy them or restore from backup."
    exit 1
fi

# Set proper permissions
echo ""
echo "🔐 Setting permissions..."
chmod -R 755 "$STORAGE_PATH/scripts/"
chmod -R 755 "$STORAGE_PATH/configs/"
chmod 644 "$STORAGE_PATH/data/processed/"*.parquet 2>/dev/null || true
echo "  ✅ Permissions set"

# Show final structure
echo ""
echo "📊 Final network storage structure:"
echo "=================================="
find "$STORAGE_PATH" -type f \( -name "*.sh" -o -name "*.yaml" -o -name "*.parquet" -o -name "*.json" \) | sort

echo ""
echo "🎉 Network storage repopulation complete!"
echo ""
echo "📁 Storage location: $STORAGE_PATH"
echo "📋 Manifest: $STORAGE_PATH/storage-manifest.json"
echo ""
echo "🚀 Ready for training setup:"
echo "   1. Clone repo to pod location (e.g., /workspace/moola)"
echo "   2. Setup python environment"
echo "   3. Configure paths to use network storage data"
echo ""
echo "💡 All essential files are now in place for 2-class training"