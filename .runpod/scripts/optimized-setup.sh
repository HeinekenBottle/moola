#!/bin/bash
# Bulletproof RunPod Setup - Handles all edge cases
set -e

echo "🚀 Moola Bulletproof Setup"
echo "=========================="
echo ""

# 1. Clone from GitHub
echo "📥 Step 1/6: Cloning repository from GitHub..."
if [[ ! -d "/workspace/moola" ]]; then
    cd /workspace
    git clone https://github.com/HeinekenBottle/moola.git
    echo "✅ Repository cloned"
else
    cd /workspace/moola
    git pull origin main || echo "⚠️ Git pull failed (offline?), using existing code"
    echo "✅ Repository ready"
fi
echo ""

# 2. Create symlink for data (fix data path issues)
echo "🔗 Step 2/6: Setting up data symlinks..."
cd /workspace/moola
if [[ -L "data" ]] && [[ -e "data" ]]; then
    echo "✅ Data symlink exists"
elif [[ -d "data" ]]; then
    # Backup existing data dir, create symlink
    mv data data.backup 2>/dev/null || true
    ln -sf /workspace/data data
    echo "✅ Data directory linked to network storage"
else
    ln -sf /workspace/data data
    echo "✅ Data linked to network storage"
fi

# Ensure train.parquet symlink exists
if [[ ! -f "/workspace/moola/data/processed/train.parquet" ]]; then
    if [[ -f "/workspace/moola/data/processed/train_2class.parquet" ]]; then
        cd /workspace/moola/data/processed/
        ln -sf train_2class.parquet train.parquet
        echo "✅ Created train.parquet symlink"
    else
        echo "⚠️ Warning: training data not found"
    fi
fi
echo ""

# 3. Create venv with system packages
echo "📦 Step 3/6: Creating virtual environment..."
if [[ ! -d "/tmp/moola-venv" ]]; then
    python3 -m venv /tmp/moola-venv --system-site-packages
    source /tmp/moola-venv/bin/activate

    echo "   Installing required packages..."
    # Install packages (torch already in template)
    pip install --no-cache-dir \
        "numpy>=1.26,<2.0" \
        "pandas>=2.2" \
        "scikit-learn>=1.3" \
        packaging \
        hatchling \
        loguru \
        click \
        rich \
        typer \
        xgboost \
        pandera \
        pyarrow \
        pydantic \
        pyyaml \
        hydra-core \
        python-dotenv

    echo "✅ Packages installed"
else
    source /tmp/moola-venv/bin/activate
    echo "✅ Using existing venv"
fi
echo ""

# 4. Set PYTHONPATH permanently
echo "⚙️  Step 4/6: Configuring environment..."
export PYTHONPATH="/workspace/moola/src:$PYTHONPATH"

# Add to .bashrc for future sessions
if ! grep -q "PYTHONPATH.*moola" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<'EOF'

# Moola environment
export PYTHONPATH="/workspace/moola/src:$PYTHONPATH"

# Activate venv automatically
if [[ -d "/tmp/moola-venv" ]]; then
    source /tmp/moola-venv/bin/activate
fi

# Quick commands
alias moola-train='cd /workspace/moola && bash /workspace/scripts/fast-train.sh'
alias moola-status='ls -lh /workspace/moola/data/artifacts/oof/*/v1/ 2>/dev/null || echo "No training done yet"'
EOF
    echo "✅ Environment configured in .bashrc"
else
    echo "✅ Environment already configured"
fi
echo ""

# 5. Verify GPU
echo "🔍 Step 5/6: Verifying GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'   CUDA: {torch.version.cuda}')
else:
    print('❌ No GPU detected!')
" || echo "⚠️ GPU check failed"
echo ""

# 6. Verify data and imports
echo "📊 Step 6/6: Final verification..."
python3 -c "
import sys
sys.path.insert(0, '/workspace/moola/src')

# Check imports
try:
    from moola.models import get_model
    print('✅ Moola imports work')
except Exception as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)

# Check data
import pandas as pd
from pathlib import Path
train_path = Path('/workspace/moola/data/processed/train.parquet')
if train_path.exists():
    df = pd.read_parquet(train_path)
    print(f'✅ Data: {df.shape[0]} samples, {len(df[\"label\"].unique())} classes')
    print(f'   Classes: {sorted(df[\"label\"].unique())}')
else:
    print('⚠️ Training data not found at', train_path)
"
echo ""

# Summary
echo "✅ SETUP COMPLETE!"
echo "================="
echo ""
echo "📊 Ready to train:"
echo "   cd /workspace/moola"
echo "   bash /workspace/scripts/fast-train.sh"
echo ""
echo "   Or use alias: moola-train"
echo ""
echo "⏱️  Total setup time: ~2-3 minutes"
