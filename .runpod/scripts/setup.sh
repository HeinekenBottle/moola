#!/bin/bash
# Simple RunPod setup - works with /workspace/ mount
set -e

echo "🚀 Moola Pod Setup"
echo "=================="

# Clone repo
if [[ ! -d "/root/moola" ]]; then
    cd /root
    git clone https://github.com/HeinekenBottle/moola.git
fi

# Create venv on local storage (fast)
if [[ ! -d "/root/venv" ]]; then
    cd /root/moola
    python -m venv /root/venv
    source /root/venv/bin/activate

    # Install dependencies
    pip install "numpy<2"
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install --no-cache-dir -e .
else
    source /root/venv/bin/activate
fi

# Set environment to use network storage for artifacts
export MOOLA_ARTIFACTS_DIR="/workspace/artifacts"
export MOOLA_DATA_DIR="/workspace/data"
export MOOLA_LOG_DIR="/workspace/logs"

# Verify GPU
echo ""
echo "🔍 GPU Check:"
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('❌ No GPU detected')
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Run training:"
echo "  bash /workspace/scripts/train.sh"
