#!/bin/bash

# Deploy and train Enhanced SimpleLSTM on 174 annotated samples with pretrained encoder
# Complete pipeline: Pretraining → Model Training → Evaluation

echo "🚀 DEPLOYING 174-SAMPLE TRAINING PIPELINE"
echo "📅 Date: $(date)"
echo "🖥️  Local Host: $(hostname)"

# Configuration
RUNPOD_IP="103.196.86.56"
SSH_PORT="12774"
SSH_KEY="~/.ssh/id_ed25519"
SSH_USER="root"
PROJECT_DIR="/root/moola"

# Step 1: Create deployment bundle
echo "📦 Creating deployment bundle..."
cd /Users/jack/projects/moola

# Create bundle directory
BUNDLE_DIR="runpod_bundle_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BUNDLE_DIR"

# Copy essential files
echo "📋 Copying project files..."
cp -r src/ "$BUNDLE_DIR/"
cp -r configs/ "$BUNDLE_DIR/"
cp -r data/ "$BUNDLE_DIR/"
cp requirements*.txt "$BUNDLE_DIR/"
cp pyproject.toml "$BUNDLE_DIR/"
cp Makefile "$BUNDLE_DIR/"
cp -r scripts/ "$BUNDLE_DIR/"

# Copy training scripts
cp scripts/train_174_with_pretrained.sh "$BUNDLE_DIR/scripts/"

# Create deployment info
cat > "$BUNDLE_DIR/DEPLOYMENT_INFO.txt" << EOF
Deployment: 174-Sample Training Pipeline
Date: $(date)
Purpose: Train Enhanced SimpleLSTM on 174 annotated samples
Features: BiLSTM pretrained encoder + engineered features
Data: train_combined_174.parquet (174 samples)
Split: train_174_temporal.json (139 train, 35 val)
EOF

# Create tar bundle
echo "🗜️  Creating tar bundle..."
tar -czf "${BUNDLE_DIR}.tar.gz" "$BUNDLE_DIR"
echo "✅ Bundle created: ${BUNDLE_DIR}.tar.gz"

# Step 2: Deploy to RunPod
echo "🚀 Deploying to RunPod..."
if [ "$RUNPOD_IP" != "YOUR_RUNPOD_IP" ]; then
    # Copy bundle to RunPod
    scp -P "$SSH_PORT" -i "$SSH_KEY" "${BUNDLE_DIR}.tar.gz" "${SSH_USER}@${RUNPOD_IP}:/root/"
    
    # Extract and setup
    ssh -p "$SSH_PORT" -i "$SSH_KEY" "${SSH_USER}@${RUNPOD_IP}" << EOF
        cd /root
        tar -xzf "${BUNDLE_DIR}.tar.gz"
        mv "$BUNDLE_DIR" moola
        cd moola
        
        # Setup environment
        pip3 install -r requirements.txt
        
        # Check data
        echo "📊 Checking training data..."
        python3 -c "
import pandas as pd
try:
    df = pd.read_parquet('data/processed/train_combined_174.parquet')
    print(f'✅ Training data: {len(df)} samples')
    print(f'🏷️  Labels: {df[\"label\"].value_counts().to_dict()}')
except Exception as e:
    print(f'❌ Error loading data: {e}')
"
        
        # Run pretraining (if needed)
        echo "🧠 Starting BiLSTM pretraining..."
        python3 -m moola.cli pretrain-bilstm \\
            --input data/raw/unlabeled_windows.parquet \\
            --device cuda \\
            --epochs 50 \\
            --output artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt
        
        # Train model
        echo "🏃‍♂️ Training Enhanced SimpleLSTM..."
        bash scripts/train_174_with_pretrained.sh
        
        echo "✅ Training pipeline completed!"
EOF
    
    echo "🎉 Deployment completed successfully!"
    echo "📊 Check results on RunPod in experiment_results.jsonl"
    
else
    echo "⚠️  Please update RUNPOD_IP in this script with your actual RunPod IP address"
    echo "📦 Bundle ready: ${BUNDLE_DIR}.tar.gz"
    echo "🔧 Manual deployment steps:"
    echo "   1. scp -P 12774 -i ~/.ssh/id_ed25519 ${BUNDLE_DIR}.tar.gz root@103.196.86.56:/root/"
    echo "   2. ssh -p 12774 -i ~/.ssh/id_ed25519 root@103.196.86.56"
    echo "   3. cd /root && tar -xzf ${BUNDLE_DIR}.tar.gz && mv $BUNDLE_DIR moola"
    echo "   4. cd moola && pip3 install -r requirements.txt"
    echo "   5. Run: bash scripts/train_174_with_pretrained.sh"
fi

# Cleanup
rm -rf "$BUNDLE_DIR"
echo "🧹 Cleaned up local bundle directory"