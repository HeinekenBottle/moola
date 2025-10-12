#!/bin/bash
# Repopulate Network Storage with Required Files
# This script restores the essential files after cleanup
# Usage: bash repopulate-storage.sh

set -e

echo "📦 REPOPULATING NETWORK STORAGE"
echo "=============================="

# Check if network storage is accessible
if [[ ! -d "/workspace" ]]; then
    echo "❌ Network storage not accessible at /workspace"
    echo "Run this script from the RunPod pod"
    exit 1
fi

echo "✅ Network storage accessible at /workspace"
echo ""

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p /workspace/{data/{raw,processed},artifacts/{models,oof,predictions},logs,configs,scripts}
mkdir -p /workspace/data/splits/v1
echo "  ✅ Directory structure created"

# Clone fresh repository to get scripts and configs
echo "📥 Cloning fresh repository..."
cd /workspace
rm -rf moola-repo 2>/dev/null || true
git clone https://github.com/HeinekenBottle/moola.git moola-repo

echo "  ✅ Repository cloned"

# Copy essential files from repo to network storage
echo "📋 Copying essential files..."

# Copy scripts
cp -r /workspace/moola-repo/.runpod/scripts/* /workspace/scripts/
chmod +x /workspace/scripts/*.sh
echo "  ✅ Deployment scripts copied"

# Copy configs
cp -r /workspace/moola-repo/configs/* /workspace/configs/
echo "  ✅ Configuration files copied"

# Check if we have data files in repo
echo "🔍 Checking for data files in repo..."
if [[ -f "/workspace/moola-repo/data/processed/train_2class.parquet" ]]; then
    cp /workspace/moola-repo/data/processed/train_2class.parquet /workspace/data/processed/
    echo "  ✅ train_2class.parquet copied (115 samples, 2-class)"
else
    echo "  ⚠️  train_2class.parquet not found in repo"
fi

if [[ -f "/workspace/moola-repo/data/processed/reversals_archive.parquet" ]]; then
    cp /workspace/moola-repo/data/processed/reversals_archive.parquet /workspace/data/processed/
    echo "  ✅ reversals_archive.parquet copied (19 samples)"
else
    echo "  ⚠️  reversals_archive.parquet not found in repo"
fi

if [[ -f "/workspace/moola-repo/data/processed/train_3class.parquet" ]]; then
    cp /workspace/moola-repo/data/processed/train_3class.parquet /workspace/data/processed/
    echo "  ✅ train_3class.parquet copied (134 samples, backup)"
else
    echo "  ⚠️  train_3class.parquet not found in repo"
fi

# Create symlink for train.parquet -> train_2class.parquet
if [[ -f "/workspace/data/processed/train_2class.parquet" ]]; then
    cd /workspace/data/processed/
    ln -sf train_2class.parquet train.parquet
    echo "  ✅ Created symlink: train.parquet → train_2class.parquet"
else
    echo "  ❌ Cannot create symlink - train_2class.parquet missing"
fi

# Copy additional utility files if they exist
echo "🔍 Checking for additional files..."

# Copy any example files
if [[ -d "/workspace/moola-repo/examples" ]]; then
    cp -r /workspace/moola-repo/examples /workspace/
    echo "  ✅ Examples copied"
fi

# Copy documentation (optional)
if [[ -f "/workspace/moola-repo/README.md" ]]; then
    cp /workspace/moola-repo/README.md /workspace/
    echo "  ✅ README copied"
fi

# Create a storage manifest
echo "📋 Creating storage manifest..."
cat > /workspace/storage-manifest.json <<EOF
{
  "created": "$(date -Iseconds)",
  "purpose": "Moola 2-class training network storage",
  "volume_id": "hg878tp14w",
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
    "bash /workspace/scripts/robust-setup.sh",
    "bash /workspace/scripts/precise-train.sh"
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
    "/workspace/scripts/robust-setup.sh"
    "/workspace/scripts/precise-train.sh"
    "/workspace/configs/default.yaml"
    "/workspace/data/processed/train.parquet"
    "/workspace/storage-manifest.json"
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

if [[ ${#missing_files[@]} -gt 0 ]]; then
    echo ""
    echo "⚠️  Some essential files are missing!"
    echo "Missing files:"
    for file in "${missing_files[@]}"; do
        echo "   $file"
    done
    exit 1
fi

# Clean up temporary repo
echo ""
echo "🧹 Cleaning up temporary repo..."
rm -rf /workspace/moola-repo
echo "  ✅ Temporary repo removed"

# Set proper permissions
echo "🔐 Setting proper permissions..."
chmod -R 755 /workspace/scripts/
chmod -R 755 /workspace/configs/
chmod 644 /workspace/data/processed/*.parquet 2>/dev/null || true
echo "  ✅ Permissions set"

# Show final structure
echo ""
echo "📊 Final network storage structure:"
echo "=================================="
find /workspace -type f -name "*.sh" -o -name "*.yaml" -o -name "*.parquet" -o -name "*.json" | sort

echo ""
echo "🎉 Network storage repopulation complete!"
echo ""
echo "📝 Storage ready for training"
echo "   Volume: hg878tp14w"
echo "   Location: /workspace"
echo ""
echo "🚀 Next steps:"
echo "   1. bash /workspace/scripts/robust-setup.sh"
echo "   2. bash /workspace/scripts/precise-train.sh"
echo ""
echo "📋 Storage manifest: /workspace/storage-manifest.json"
echo "💡 All essential files are now in place for 2-class training"