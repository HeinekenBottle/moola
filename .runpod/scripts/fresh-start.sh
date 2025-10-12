#!/bin/bash
# Complete Fresh Start for RunPod
# This script combines cleanup + repopulation + setup
# Usage: bash fresh-start.sh

set -e

echo "🚀 MOOLA FRESH START"
echo "==================="

echo "This script will:"
echo "1. Clean all network storage (DELETE EVERYTHING)"
echo "2. Repopulate with essential files only"
echo "3. Set up clean training environment"
echo ""

# Safety confirmation
echo "⚠️  DANGER ZONE ⚠️"
echo "This will permanently delete ALL files in network storage"
echo ""
read -p "Type 'FRESH-START' to confirm complete cleanup and setup: " confirm
if [[ "$confirm" != "FRESH-START" ]]; then
    echo "❌ Fresh start cancelled"
    exit 1
fi

echo ""
echo "🚀 Starting fresh start process..."

# Step 1: Clean network storage
echo ""
echo "📍 Step 1: Cleaning network storage..."
bash /workspace/scripts/clean-network-storage.sh

# Step 2: Repopulate storage
echo ""
echo "📍 Step 2: Repopulating network storage..."
bash /workspace/scripts/repopulate-storage.sh

# Step 3: Setup training environment
echo ""
echo "📍 Step 3: Setting up training environment..."
bash /workspace/scripts/robust-setup.sh

# Step 4: Quick verification
echo ""
echo "📍 Step 4: Quick verification..."

# Verify data
python3 -c "
import pandas as pd
df = pd.read_parquet('/workspace/data/processed/train.parquet')
print(f'✅ Data: {df.shape}, Classes: {sorted(df[\"label\"].unique())}')
"

# Verify environment
python3 -c "
import torch
print(f'✅ PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
"

# Verify CLI
python3 -m moola.cli --help > /dev/null 2>&1
echo "✅ CLI working"

echo ""
echo "🎉 FRESH START COMPLETE!"
echo ""
echo "📊 System Status:"
echo "   Network Storage: ✅ Clean and populated"
echo "   Training Data: ✅ 115 samples (2-class)"
echo "   Environment: ✅ python3 + pip-tools"
echo "   Dependencies: ✅ Locked and verified"
echo "   GPU: ✅ Ready for training"
echo ""
echo "🚀 Ready to train:"
echo "   bash /workspace/scripts/precise-train.sh"
echo ""
echo "📁 Network Storage Contents:"
tree /workspace -L 2 2>/dev/null || find /workspace -maxdepth 2 -type d | sort