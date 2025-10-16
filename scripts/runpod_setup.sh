#!/bin/bash
# RunPod Setup Script for Moola ML Pipeline
# Run this on RunPod instance after uploading files

set -e  # Exit on error

echo "=================================="
echo "Moola RunPod Setup"
echo "=================================="

# 1. Verify CUDA
echo -e "\n[1/6] Verifying CUDA..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✅ CUDA: {torch.cuda.get_device_name(0)}')"

# 2. Install dependencies (skip blinker if it fails)
echo -e "\n[2/6] Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install scikit-learn pandas numpy xgboost --quiet || true
pip install cleanlab --quiet || true
pip install scipy --quiet || true
echo "✅ Dependencies installed"

# 3. Verify moola imports
echo -e "\n[3/6] Verifying Moola imports..."
cd /workspace/moola
python3 -c "
from moola.models import SimpleLSTMModel, CnnTransformerModel
from moola.models.ts_tcc import TSTCC
print('✅ All models import successfully')
"

# 4. Verify data files exist
echo -e "\n[4/6] Checking data files..."
if [ ! -f "data/processed/train_clean.parquet" ]; then
    echo "❌ Missing: data/processed/train_clean.parquet"
    echo "Please upload this file first!"
    exit 1
fi
echo "✅ Training data found: $(ls -lh data/processed/train_clean.parquet | awk '{print $5}')"

# 5. Check existing OOF files
echo -e "\n[5/6] Checking existing OOF predictions..."
for model in logreg rf xgb; do
    if [ -f "data/oof/${model}_clean.npy" ]; then
        echo "✅ Found: ${model}_clean.npy"
    else
        echo "⚠️  Missing: ${model}_clean.npy (will need to generate)"
    fi
done

# 6. Show GPU info
echo -e "\n[6/6] GPU Information..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo -e "\n=================================="
echo "Setup Complete! ✅"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Upload missing data files (if any)"
echo "2. Run: bash scripts/runpod_train.sh"
echo ""
