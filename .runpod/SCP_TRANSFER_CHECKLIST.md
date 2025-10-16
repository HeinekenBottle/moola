# RunPod SCP Transfer Checklist - Masked LSTM Pre-training

**Pod Info**: 213.173.98.6:14385
**Status**: Environment ready (PyTorch 2.4.1+cu124, CUDA available) ✅

---

## Required Files to SCP

### **1. Training Data** (REQUIRED)
```bash
# Local → RunPod
scp -P 14385 -i ~/.ssh/id_ed25519 \
    data/processed/train_pivot_134.parquet \
    root@213.173.98.6:/workspace/data/processed/train_pivot_134.parquet

# Create symlink on RunPod
ssh root@213.173.98.6 -p 14385 -i ~/.ssh/id_ed25519 \
    "cd /workspace/data/processed && ln -sf train_pivot_134.parquet train.parquet"
```

### **2. Unlabeled Pre-training Data** (REQUIRED for masked LSTM)
```bash
# Option A: Generate locally first
python scripts/generate_unlabeled_data.py \
    --input data/processed/train_pivot_134.parquet \
    --output data/processed/unlabeled_pretrain.parquet \
    --target-count 1000 \
    --augment-factor 4

# Then SCP
scp -P 14385 -i ~/.ssh/id_ed25519 \
    data/processed/unlabeled_pretrain.parquet \
    root@213.173.98.6:/workspace/data/processed/unlabeled_pretrain.parquet
```

**OR**

```bash
# Option B: Generate on RunPod (after train data uploaded)
ssh root@213.173.98.6 -p 14385 -i ~/.ssh/id_ed25519 'bash -s' << 'EOF'
source /tmp/moola-venv/bin/activate
cd /workspace/moola
python scripts/generate_unlabeled_data.py \
    --input /workspace/data/processed/train_pivot_134.parquet \
    --output /workspace/data/processed/unlabeled_pretrain.parquet \
    --target-count 1000 \
    --augment-factor 4
EOF
```

### **3. Code Updates** (if not using git pull)

**Option A: Git Pull (RECOMMENDED)**
```bash
ssh root@213.173.98.6 -p 14385 -i ~/.ssh/id_ed25519 \
    "cd /workspace/moola && git pull origin main"
```

**Option B: SCP Individual Files**
```bash
# Core masked LSTM files
scp -P 14385 -i ~/.ssh/id_ed25519 \
    src/moola/models/bilstm_masked_autoencoder.py \
    root@213.173.98.6:/workspace/moola/src/moola/models/

scp -P 14385 -i ~/.ssh/id_ed25519 \
    src/moola/pretraining/masked_lstm_pretrain.py \
    root@213.173.98.6:/workspace/moola/src/moola/pretraining/

scp -P 14385 -i ~/.ssh/id_ed25519 \
    src/moola/pretraining/data_augmentation.py \
    root@213.173.98.6:/workspace/moola/src/moola/pretraining/

scp -P 14385 -i ~/.ssh/id_ed25519 \
    src/moola/models/simple_lstm.py \
    root@213.173.98.6:/workspace/moola/src/moola/models/

scp -P 14385 -i ~/.ssh/id_ed25519 \
    src/moola/config/training_config.py \
    root@213.173.98.6:/workspace/moola/src/moola/config/

scp -P 14385 -i ~/.ssh/id_ed25519 \
    src/moola/cli.py \
    root@213.173.98.6:/workspace/moola/src/moola/

# Scripts
scp -P 14385 -i ~/.ssh/id_ed25519 \
    scripts/generate_unlabeled_data.py \
    root@213.173.98.6:/workspace/moola/scripts/
```

---

## Quick Transfer Commands

### **Minimal Setup (if code already in GitHub)**
```bash
# 1. Upload training data
scp -P 14385 -i ~/.ssh/id_ed25519 \
    data/processed/train_pivot_134.parquet \
    root@213.173.98.6:/workspace/data/processed/train_pivot_134.parquet

# 2. Git pull latest code
ssh root@213.173.98.6 -p 14385 -i ~/.ssh/id_ed25519 \
    "cd /workspace/moola && git pull origin main"

# 3. Generate unlabeled data on RunPod
ssh root@213.173.98.6 -p 14385 -i ~/.ssh/id_ed25519 'bash -s' << 'EOF'
source /tmp/moola-venv/bin/activate
cd /workspace/moola
python scripts/generate_unlabeled_data.py \
    --input /workspace/data/processed/train_pivot_134.parquet \
    --output /workspace/data/processed/unlabeled_pretrain.parquet \
    --target-count 1000 \
    --augment-factor 4
EOF
```

---

## Pre-training Commands

### **Step 1: Pre-train Encoder**
```bash
ssh root@213.173.98.6 -p 14385 -i ~/.ssh/id_ed25519 'bash -s' << 'EOF'
source /tmp/moola-venv/bin/activate
cd /workspace/moola
export PYTHONPATH="/workspace/moola/src:$PYTHONPATH"
export MOOLA_DATA_DIR="/workspace/data"
export MOOLA_ARTIFACTS_DIR="/workspace/artifacts"

echo "🚀 Starting Masked LSTM Pre-training on RTX 4090"
echo "=============================================="

python -m moola.cli pretrain-bilstm \
    --input /workspace/data/processed/unlabeled_pretrain.parquet \
    --output /workspace/artifacts/pretrained/bilstm_encoder.pt \
    --device cuda \
    --epochs 50 \
    --batch-size 512 \
    --mask-strategy patch \
    --augment \
    --seed 1337

echo "✅ Pre-training complete!"
ls -lh /workspace/artifacts/pretrained/
EOF
```

**Expected time**: ~30-40 minutes on RTX 4090

### **Step 2: Download Pre-trained Encoder**
```bash
mkdir -p data/artifacts/pretrained

scp -P 14385 -i ~/.ssh/id_ed25519 \
    root@213.173.98.6:/workspace/artifacts/pretrained/bilstm_encoder.pt \
    data/artifacts/pretrained/bilstm_encoder.pt
```

### **Step 3: Fine-tune SimpleLSTM**
```bash
ssh root@213.173.98.6 -p 14385 -i ~/.ssh/id_ed25519 'bash -s' << 'EOF'
source /tmp/moola-venv/bin/activate
cd /workspace/moola
export PYTHONPATH="/workspace/moola/src:$PYTHONPATH"
export MOOLA_DATA_DIR="/workspace/data"
export MOOLA_ARTIFACTS_DIR="/workspace/artifacts"

echo "🚀 Fine-tuning SimpleLSTM with Pre-trained Encoder"
echo "================================================"

python -m moola.cli oof \
    --model simple_lstm \
    --device cuda \
    --seed 1337 \
    --load-pretrained-encoder /workspace/artifacts/pretrained/bilstm_encoder.pt

echo "✅ Fine-tuning complete!"
EOF
```

**Expected time**: ~10-15 minutes on RTX 4090

### **Step 4: Download Results**
```bash
mkdir -p /tmp/masked_lstm_results

scp -P 14385 -i ~/.ssh/id_ed25519 \
    root@213.173.98.6:/workspace/artifacts/oof/simple_lstm/v1/seed_1337.npy \
    /tmp/masked_lstm_results/seed_1337.npy
```

---

## Verification Checklist

### **Before Pre-training**
```bash
# Check files exist
ssh root@213.173.98.6 -p 14385 -i ~/.ssh/id_ed25519 'bash -s' << 'EOF'
echo "📁 Checking files..."
ls -lh /workspace/data/processed/train_pivot_134.parquet
ls -lh /workspace/data/processed/unlabeled_pretrain.parquet
ls -lh /workspace/moola/src/moola/models/bilstm_masked_autoencoder.py
ls -lh /workspace/moola/src/moola/pretraining/masked_lstm_pretrain.py
echo "✅ All files present"
EOF
```

### **After Pre-training**
```bash
# Check encoder created
ssh root@213.173.98.6 -p 14385 -i ~/.ssh/id_ed25519 \
    "ls -lh /workspace/artifacts/pretrained/bilstm_encoder.pt"
```

---

## File Size Reference

| File | Size | Transfer Time (estimate) |
|------|------|-------------------------|
| `train_pivot_134.parquet` | ~500 KB | <1 second |
| `unlabeled_pretrain.parquet` | ~2.5 MB | <5 seconds |
| `bilstm_encoder.pt` | ~3-5 MB | <5 seconds |
| All Python files | ~100 KB | <1 second |

**Total transfer time**: ~10-15 seconds for all files

---

## Troubleshooting

### **Missing training data**
```bash
# Error: FileNotFoundError: Missing /workspace/data/processed/train.parquet
# Solution: Create symlink
ssh root@213.173.98.6 -p 14385 -i ~/.ssh/id_ed25519 \
    "cd /workspace/data/processed && ln -sf train_pivot_134.parquet train.parquet"
```

### **Module not found errors**
```bash
# Reinstall moola package
ssh root@213.173.98.6 -p 14385 -i ~/.ssh/id_ed25519 'bash -s' << 'EOF'
source /tmp/moola-venv/bin/activate
cd /workspace/moola
pip install -e . --no-deps
EOF
```

### **CUDA out of memory**
```bash
# Reduce batch size in command
--batch-size 256  # Instead of 512
```

---

## Expected Results

**Pre-training metrics** (after 50 epochs):
```
Train Loss: ~0.012-0.015
Val Loss: ~0.015-0.020
```

**Fine-tuning accuracy** (with pre-trained encoder):
```
Overall: 65-69% (vs 57% baseline)
Class 0: 75-80% (vs 100% baseline)
Class 1: 45-55% (vs 0% baseline) ← CLASS COLLAPSE BROKEN!
```

---

## Status
- [x] Environment setup complete (PyTorch 2.4.1+cu124, CUDA)
- [ ] Training data uploaded
- [ ] Unlabeled data generated/uploaded
- [ ] Code updated (git pull or SCP)
- [ ] Pre-training started
- [ ] Encoder downloaded
- [ ] Fine-tuning complete
- [ ] Results analyzed
