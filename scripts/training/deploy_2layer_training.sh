#!/bin/bash
# Deploy and train 2-layer BiLSTM encoder on RunPod

set -e

RUNPOD_IP="103.196.86.56"
RUNPOD_PORT="12774"
SSH_KEY="~/.ssh/id_ed25519"
REMOTE_DIR="/workspace/moola"

echo "=== Deploying 2-Layer BiLSTM Training to RunPod ==="

# Test SSH connection
echo "Testing SSH connection..."
ssh -i $SSH_KEY -p $RUNPOD_PORT root@$RUNPOD_IP "echo 'Connection successful'"

# Create bundle
echo "Creating training bundle..."
tar -czf runpod_2layer_bundle.tar.gz \
    scripts/train_compatible_encoder.py \
    src/moola/models/bilstm_masked_autoencoder.py \
    src/moola/models/enhanced_simple_lstm.py \
    src/moola/pretraining/masked_lstm_pretrain.py \
    src/moola/utils/early_stopping.py \
    src/moola/utils/seeds.py \
    src/moola/data/load.py \
    data/raw/unlabeled_windows.parquet \
    requirements-runpod.txt

# Upload to RunPod
echo "Uploading bundle to RunPod..."
scp -i $SSH_KEY -P $RUNPOD_PORT runpod_2layer_bundle.tar.gz root@$RUNPOD_IP:/workspace/

# Extract and setup on RunPod
echo "Setting up training environment on RunPod..."
ssh -i $SSH_KEY -P $RUNPOD_PORT root@$RUNPOD_IP << 'EOF'
cd /workspace
tar -xzf runpod_2layer_bundle.tar.gz

# Install dependencies
pip3 install -r requirements-runpod.txt

# Create directories
mkdir -p data/artifacts/pretrained

# Check GPU
echo "GPU Status:"
nvidia-smi

echo "Environment setup complete"
EOF

# Start training
echo "Starting 2-layer BiLSTM training..."
ssh -i $SSH_KEY -P $RUNPOD_PORT root@$RUNPOD_IP << 'EOF'
cd /workspace
export PYTHONPATH=/workspace/src:$PYTHONPATH

# Run training
python3 scripts/train_compatible_encoder.py

echo "Training completed"
EOF

# Retrieve results
echo "Retrieving trained encoder..."
scp -i $SSH_KEY -P $RUNPOD_PORT root@$RUNPOD_IP:/workspace/data/artifacts/pretrained/bilstm_encoder_2layer.pt ./

echo "=== 2-Layer Training Complete ==="
echo "Encoder saved as: bilstm_encoder_2layer.pt"

# Cleanup
rm runpod_2layer_bundle.tar.gz