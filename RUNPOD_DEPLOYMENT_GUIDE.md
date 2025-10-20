# RunPod Deployment Guide - 11D Training Pipeline
**Date:** 2025-10-20
**Branch:** `refactor/architecture-cleanup`
**Pipeline:** 11D RelativeTransform + BiLSTM MAE Pretraining

---

## ⚠️ Philosophy: Manual SSH/SCP Workflow

**Based on your experience:**
- ❌ Shell scripts break easily on RunPod (one error → total failure)
- ✅ Manual SSH/SCP workflow is most reliable
- ✅ Verify each step before proceeding

**This guide follows the proven workflow:**
1. **Verify** RunPod environment (Python, PyTorch, CUDA)
2. **Check** required packages installed
3. **SCP** project files to RunPod
4. **SSH** into RunPod and execute commands manually
5. **Monitor** progress and handle errors immediately
6. **Retrieve** results when complete

---

## 📋 Pre-Flight Checklist (On Mac)

### 1. Verify Local Files Ready
```bash
# Check 11D dataset exists
ls -lh data/processed/labeled/train_latest_relative.parquet
# Expected: 639 KB

# Check split file
ls -lh data/splits/fwd_chain_v3.json
# Expected: 1.7 KB

# Verify dataset shape
python3 << 'EOF'
import pandas as pd
import numpy as np
df = pd.read_parquet('data/processed/labeled/train_latest_relative.parquet')
feats = df['features'].iloc[0]
stacked = np.array([f for f in feats])
print(f"✅ Dataset shape: (174, {stacked.shape[0]}, {stacked.shape[1]})")
EOF
# Expected: ✅ Dataset shape: (174, 105, 11)
```

### 2. Prepare SSH Key
```bash
# Verify RunPod SSH key exists
ls -lh ~/.ssh/runpod_key
chmod 600 ~/.ssh/runpod_key
```

### 3. Create Deployment Bundle (Optional)
```bash
# Create a tarball for faster transfer
tar -czf moola_11d_bundle.tar.gz \
  src/ \
  data/processed/labeled/train_latest_relative.parquet \
  data/splits/fwd_chain_v3.json \
  pyproject.toml \
  setup.py

# Verify bundle
ls -lh moola_11d_bundle.tar.gz
```

---

## 🚀 RunPod Setup (Manual Steps)

### Step 1: Start RunPod Instance

**GPU Requirements:**
- Recommended: RTX 4090 (24GB VRAM)
- Minimum: RTX 3090 (24GB VRAM)
- Template: PyTorch 2.x with CUDA 11.8+

**Get RunPod IP:**
- Save IP address from RunPod dashboard
- Example: `216.xxx.xxx.xxx`

### Step 2: Verify RunPod Environment

**SSH into RunPod:**
```bash
# Replace YOUR_RUNPOD_IP with actual IP
export RUNPOD_IP="YOUR_RUNPOD_IP"
ssh -i ~/.ssh/runpod_key ubuntu@$RUNPOD_IP
```

**Run verification commands (on RunPod):**
```bash
# Check Python version (need 3.10+)
python3 --version

# Check PyTorch and CUDA
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
EOF
```

**Expected output:**
```
PyTorch version: 2.x.x
CUDA available: True
CUDA version: 11.8 (or 12.x)
GPU count: 1
GPU name: NVIDIA GeForce RTX 4090
GPU memory: 24.00 GB
```

### Step 3: Check Required Packages

```bash
# Check if packages exist (don't install yet)
python3 -c "import pandas; print(f'pandas {pandas.__version__}')"
python3 -c "import numpy; print(f'numpy {numpy.__version__}')"
python3 -c "import sklearn; print(f'scikit-learn {sklearn.__version__}')"

# If any missing, install individually
pip3 install pandas numpy scikit-learn

# Verify no errors
echo "✅ Environment verified"
```

### Step 4: Create Project Directory

```bash
# Create workspace
cd /workspace
mkdir -p moola
cd moola

# Verify location
pwd
# Expected: /workspace/moola
```

---

## 📦 File Transfer (SCP from Mac)

### Option A: Transfer Entire Project (Recommended)

**From Mac terminal:**
```bash
# Navigate to project root
cd /Users/jack/projects/moola

# SCP entire project (exclude unnecessary files)
rsync -avz --progress \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='data/raw/unlabeled' \
  --exclude='artifacts/models' \
  --exclude='experiment_results.jsonl' \
  -e "ssh -i ~/.ssh/runpod_key" \
  . ubuntu@$RUNPOD_IP:/workspace/moola/

# Verify transfer
ssh -i ~/.ssh/runpod_key ubuntu@$RUNPOD_IP "ls -lh /workspace/moola/data/processed/labeled/train_latest_relative.parquet"
```

### Option B: Transfer Bundle (Faster)

**From Mac:**
```bash
# Transfer bundle
scp -i ~/.ssh/runpod_key moola_11d_bundle.tar.gz ubuntu@$RUNPOD_IP:/workspace/moola/

# SSH and extract
ssh -i ~/.ssh/runpod_key ubuntu@$RUNPOD_IP
cd /workspace/moola
tar -xzf moola_11d_bundle.tar.gz
rm moola_11d_bundle.tar.gz

# Verify
ls -lh data/processed/labeled/train_latest_relative.parquet
```

---

## 🎯 Training Execution (Manual Commands)

**All commands run on RunPod via SSH**

### Pre-Execution Checks

```bash
# Navigate to project
cd /workspace/moola

# Verify files exist
ls -lh data/processed/labeled/train_latest_relative.parquet
ls -lh data/splits/fwd_chain_v3.json

# Create output directories
mkdir -p artifacts/encoders/pretrained
mkdir -p artifacts/runs
mkdir -p data/logs

# Verify Python path
export PYTHONPATH=/workspace/moola/src:$PYTHONPATH
echo $PYTHONPATH
```

### Step 1: Pretrain 11D BiLSTM Encoder (~20-30 min)

```bash
# Run pretraining
python3 -m moola.cli pretrain-bilstm \
  --input data/processed/labeled/train_latest_relative.parquet \
  --input-dim 11 \
  --hidden-dim 128 \
  --num-layers 2 \
  --mask-ratio 0.15 \
  --mask-strategy patch \
  --epochs 50 \
  --batch-size 256 \
  --device cuda \
  --output artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt \
  --seed 17
```

**Monitor progress:**
- Watch for loss decreasing
- GPU memory usage should be ~4-6 GB
- Expected time: 20-30 minutes

**Verify output:**
```bash
ls -lh artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt
# Expected: ~500 KB file
```

**If error occurs:**
- Check error message carefully
- Verify input_dim is 11 (not 4)
- Check CUDA memory (nvidia-smi)
- Reduce batch_size if OOM

### Step 2: Fine-tune with Frozen Encoder (~5-10 min)

```bash
# Run fine-tuning with frozen encoder
python3 -m moola.cli train \
  --model enhanced_simple_lstm \
  --data data/processed/labeled/train_latest_relative.parquet \
  --split data/splits/fwd_chain_v3.json \
  --pretrained-encoder artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt \
  --freeze-encoder \
  --input-dim 11 \
  --epochs 5 \
  --augment-data false \
  --device cuda \
  --seed 17 \
  --save-run true
```

**Monitor progress:**
- Quick convergence expected (5 epochs)
- Accuracy should reach ~80-85%
- Note the RUN_ID from output

**Verify output:**
```bash
ls -lh artifacts/runs/
# Should show new run directory with timestamp
```

### Step 3: Fine-tune with Unfrozen Encoder (~15-20 min)

```bash
# Run fine-tuning with unfrozen encoder
python3 -m moola.cli train \
  --model enhanced_simple_lstm \
  --data data/processed/labeled/train_latest_relative.parquet \
  --split data/splits/fwd_chain_v3.json \
  --pretrained-encoder artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt \
  --no-freeze-encoder \
  --input-dim 11 \
  --epochs 25 \
  --augment-data false \
  --device cuda \
  --seed 17 \
  --save-run true
```

**Monitor progress:**
- Full model optimization (25 epochs)
- Expected improvement: +5-8% accuracy over baseline
- Note the final RUN_ID

**Verify output:**
```bash
ls -lh artifacts/runs/
# Should show another run directory
```

---

## 📊 Retrieve Results (SCP to Mac)

### Option 1: Get Specific Files

**From Mac terminal:**
```bash
# Get encoder
scp -i ~/.ssh/runpod_key \
  ubuntu@$RUNPOD_IP:/workspace/moola/artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt \
  artifacts/encoders/pretrained/

# Get run artifacts (replace <RUN_ID>)
scp -i ~/.ssh/runpod_key -r \
  ubuntu@$RUNPOD_IP:/workspace/moola/artifacts/runs/<RUN_ID> \
  artifacts/runs/

# Get logs
scp -i ~/.ssh/runpod_key -r \
  ubuntu@$RUNPOD_IP:/workspace/moola/data/logs/ \
  data/
```

### Option 2: Get Everything

```bash
# Sync all artifacts and logs
rsync -avz --progress \
  -e "ssh -i ~/.ssh/runpod_key" \
  ubuntu@$RUNPOD_IP:/workspace/moola/artifacts/ \
  artifacts/

rsync -avz --progress \
  -e "ssh -i ~/.ssh/runpod_key" \
  ubuntu@$RUNPOD_IP:/workspace/moola/data/logs/ \
  data/logs/
```

---

## 🔍 Evaluation (On Mac)

```bash
# Find latest run
ls -lt artifacts/runs/ | head -5

# Evaluate (replace <RUN_ID> with actual ID)
export RUN_ID="<RUN_ID>"
python3 -m moola.cli eval \
  --run artifacts/runs/$RUN_ID \
  --metrics pr_auc brier ece accuracy f1_macro f1_per_class \
  --event-metrics hit_at_pm3 lead_lag pointer_f1 \
  --save-reliability-plot artifacts/plots/reliability_enh_rel_11d.png \
  --save-metrics artifacts/metrics/enh_rel_11d.json
```

**Promotion Rule:**
- ✅ PR-AUC ↑ (higher than baseline)
- ✅ Brier ↓ (lower than baseline)
- ✅ ECE ≤ baseline + 0.02

**If gates pass:**
```bash
cp artifacts/runs/$RUN_ID/model.pkl \
   artifacts/models/supervised/enhanced_baseline_v2_relative_11d.pt

echo "✅ Model promoted to production baseline"
```

---

## 🐛 Troubleshooting

### Issue: Module not found errors
```bash
# Fix PYTHONPATH
export PYTHONPATH=/workspace/moola/src:$PYTHONPATH
```

### Issue: CUDA out of memory
```bash
# Reduce batch size
# In pretrain-bilstm: --batch-size 128 (instead of 256)
# In train: default is 32, try 16
```

### Issue: Input dimension mismatch
```bash
# Verify --input-dim 11 is used in ALL commands
# Check dataset shape:
python3 << 'EOF'
import pandas as pd
import numpy as np
df = pd.read_parquet('data/processed/labeled/train_latest_relative.parquet')
feats = df['features'].iloc[0]
stacked = np.array([f for f in feats])
print(f"Shape: {stacked.shape}")  # Should be (105, 11)
EOF
```

### Issue: Seed not consistent
```bash
# Always pass --seed 17 to ALL commands
# Verify in config/training_config.yaml that default seed is 17
```

### Issue: Training not improving
```bash
# Check augmentation is disabled: --augment-data false
# Check class balance in dataset
# Check learning rate (default should be fine)
```

---

## 📝 Command Summary (Copy-Paste Ready)

**All commands assume you're SSH'd into RunPod at `/workspace/moola`**

```bash
# === SETUP ===
cd /workspace/moola
export PYTHONPATH=/workspace/moola/src:$PYTHONPATH
mkdir -p artifacts/encoders/pretrained artifacts/runs data/logs

# === STEP 1: PRETRAIN ===
python3 -m moola.cli pretrain-bilstm \
  --input data/processed/labeled/train_latest_relative.parquet \
  --input-dim 11 --hidden-dim 128 --num-layers 2 \
  --mask-ratio 0.15 --mask-strategy patch --epochs 50 \
  --batch-size 256 --device cuda --seed 17 \
  --output artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt

# === STEP 2: FINE-TUNE FROZEN ===
python3 -m moola.cli train \
  --model enhanced_simple_lstm \
  --data data/processed/labeled/train_latest_relative.parquet \
  --split data/splits/fwd_chain_v3.json \
  --pretrained-encoder artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt \
  --freeze-encoder --input-dim 11 --epochs 5 \
  --augment-data false --device cuda --seed 17 --save-run true

# === STEP 3: FINE-TUNE UNFROZEN ===
python3 -m moola.cli train \
  --model enhanced_simple_lstm \
  --data data/processed/labeled/train_latest_relative.parquet \
  --split data/splits/fwd_chain_v3.json \
  --pretrained-encoder artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt \
  --no-freeze-encoder --input-dim 11 --epochs 25 \
  --augment-data false --device cuda --seed 17 --save-run true
```

---

## ✅ Success Criteria

### Build Phase (Completed)
- ✅ 11D dataset created and validated
- ✅ CLI enhanced with all required parameters
- ✅ All files ready for deployment

### RunPod Phase (Execute Now)
- ⏳ Environment verified (Python, PyTorch, CUDA)
- ⏳ Files transferred successfully
- ⏳ Encoder pretrained without errors
- ⏳ Fine-tuning completed (frozen + unfrozen)
- ⏳ Results retrieved to Mac

### Evaluation Phase (After Training)
- ⏳ Metrics computed and gates checked
- ⏳ Model promoted if gates pass
- ⏳ Ready for next phase (CPSA 0.25 → 0.5)

---

## 🎯 Expected Timeline

| Phase | Time | Status |
|-------|------|--------|
| RunPod setup + file transfer | 10-15 min | ⏳ Ready |
| Step 1: Pretrain encoder | 20-30 min | ⏳ Ready |
| Step 2: Fine-tune frozen | 5-10 min | ⏳ Ready |
| Step 3: Fine-tune unfrozen | 15-20 min | ⏳ Ready |
| Retrieve results | 5 min | ⏳ Ready |
| **TOTAL** | **55-80 min** | ⏳ Ready |

---

## 🚀 Quick Start (When Ready)

```bash
# 1. Start RunPod instance (RTX 4090)
# 2. Set environment variable
export RUNPOD_IP="YOUR_IP_HERE"

# 3. Transfer files
rsync -avz --progress --exclude='.git' --exclude='__pycache__' \
  -e "ssh -i ~/.ssh/runpod_key" \
  . ubuntu@$RUNPOD_IP:/workspace/moola/

# 4. SSH in
ssh -i ~/.ssh/runpod_key ubuntu@$RUNPOD_IP

# 5. Run training (copy commands from "Command Summary" above)

# 6. Monitor progress and handle any errors immediately

# 7. Retrieve results when complete
```

**You're ready to train! 🎉**
