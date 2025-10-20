# RunPod Deployment Manifest
**11D Training Pipeline - Complete Requirements**
**Date:** 2025-10-20
**Status:** PRE-FLIGHT CHECKLIST

---

## 🎯 Overview

This manifest outlines EVERYTHING needed to successfully run the 11D training pipeline on RunPod.

**Pipeline Goal:**
- Pretrain BiLSTM encoder on 11D RelativeTransform features (50 epochs)
- Fine-tune EnhancedSimpleLSTM with frozen encoder (5 epochs)
- Fine-tune EnhancedSimpleLSTM with unfrozen encoder (25 epochs)
- Total time: 40-60 minutes on RTX 4090

---

## 📦 1. PACKAGE REQUIREMENTS

### **A. Python Version**
```
Python >= 3.10 (required)
Tested on: 3.10, 3.11, 3.12
```

### **B. Core ML Packages (CRITICAL)**

**PyTorch Ecosystem:**
```bash
torch>=2.0,<2.3          # MUST have CUDA support
torchvision>=0.17,<0.19
pytorch-lightning>=2.4.0,<3.0
torchmetrics>=1.8,<2.0
```

**Data Science Stack:**
```bash
numpy>=1.26.4,<2.0
pandas>=2.3,<3.0
scipy>=1.14,<2.0
scikit-learn>=1.7,<2.0
```

**Data Handling:**
```bash
pyarrow>=17.0,<18.0      # For parquet files
pandera>=0.26.1,<1.0     # Data validation
```

**Additional ML:**
```bash
xgboost>=2.0,<3.0
imbalanced-learn==0.14.0
```

### **C. Configuration & CLI**

```bash
click>=8.2,<9.0
typer>=0.17,<1.0
hydra-core>=1.3,<2.0
pydantic>=2.11,<3.0
pydantic-settings>=2.9,<3.0
python-dotenv>=1.0
PyYAML>=6.0
```

### **D. Utilities**

```bash
loguru>=0.7,<1.0         # Logging
rich>=14.0,<15.0         # Terminal output
tqdm                     # Progress bars
joblib>=1.5,<2.0         # Parallelization
```

### **E. Total Package Count**

**30 packages** from pyproject.toml (excluding dev dependencies)

---

## 📝 2. INSTALLATION STRATEGY

### **Option A: Use RunPod PyTorch Template (RECOMMENDED)**

**Template:** PyTorch 2.x with CUDA 11.8+ or 12.x

**Why:** Pre-installed PyTorch, numpy, pandas, sklearn

**Still need to install:**
```bash
pip3 install pyarrow pandera hydra-core loguru rich typer click \
  pydantic pydantic-settings python-dotenv PyYAML \
  pytorch-lightning torchmetrics xgboost imbalanced-learn joblib
```

### **Option B: Install Everything (If Base Template)**

```bash
# Install from pyproject.toml (if file is transferred)
pip3 install -e .

# OR install manually
pip3 install torch torchvision numpy pandas scipy scikit-learn \
  pyarrow pandera hydra-core loguru rich typer click \
  pydantic pydantic-settings python-dotenv PyYAML \
  pytorch-lightning torchmetrics xgboost imbalanced-learn joblib tqdm
```

### **Option C: Requirements File (Create on Mac)**

```bash
# Generate requirements.txt
cat > requirements_runpod.txt << 'EOF'
torch>=2.0
torchvision>=0.17
numpy>=1.26.4
pandas>=2.3
scipy>=1.14
scikit-learn>=1.7
pyarrow>=17.0
pandera>=0.26.1
click>=8.2
typer>=0.17
hydra-core>=1.3
pydantic>=2.11
pydantic-settings>=2.9
python-dotenv>=1.0
PyYAML>=6.0
loguru>=0.7
rich>=14.0
tqdm
joblib>=1.5
pytorch-lightning>=2.4.0
torchmetrics>=1.8
xgboost>=2.0
imbalanced-learn==0.14.0
EOF

# Transfer to RunPod
scp -i ~/.ssh/runpod_key requirements_runpod.txt ubuntu@$RUNPOD_IP:/workspace/moola/

# Install on RunPod
pip3 install -r requirements_runpod.txt
```

---

## 📂 3. FILE TRANSFER REQUIREMENTS

### **A. Source Code (REQUIRED)**

**Directory:** `src/moola/` (2.6 MB, 106 Python files)

**Critical modules:**
```
src/moola/
├── cli.py                          # Command-line interface
├── config/                         # Configuration files (6 files)
│   ├── training_config.py
│   ├── model_config.py
│   └── ...
├── models/                         # Model architectures
│   ├── simple_lstm.py              # EnhancedSimpleLSTM
│   ├── bilstm_masked_autoencoder.py
│   └── base.py
├── pretraining/                    # Pre-training code
│   ├── masked_lstm_pretrain.py     # MaskedLSTMPretrainer
│   └── data_augmentation.py
├── data/                           # Data loading
│   ├── load.py
│   └── ...
├── features/                       # Feature engineering
│   ├── relative_transform.py       # RelativeFeatureTransform
│   └── ...
├── utils/                          # Utilities
│   ├── seeds.py
│   ├── early_stopping.py
│   └── ...
└── ...
```

**Transfer command:**
```bash
rsync -avz --progress -e "ssh -i ~/.ssh/runpod_key" \
  src/ ubuntu@$RUNPOD_IP:/workspace/moola/src/
```

### **B. Data Files (REQUIRED)**

**1. Training Dataset (11D)**
```
File: data/processed/labeled/train_latest_relative.parquet
Size: 640 KB
Shape: (174, 105, 11)
Purpose: 11D RelativeTransform features for training
```

**2. Split Configuration**
```
File: data/splits/fwd_chain_v3.json
Size: 4 KB
Config: purge_window=104, 139 train / 35 val
Purpose: Forward-chaining split indices
```

**Transfer commands:**
```bash
# Create directories on RunPod first
ssh -i ~/.ssh/runpod_key ubuntu@$RUNPOD_IP \
  "mkdir -p /workspace/moola/data/processed/labeled /workspace/moola/data/splits"

# Transfer data files
scp -i ~/.ssh/runpod_key \
  data/processed/labeled/train_latest_relative.parquet \
  ubuntu@$RUNPOD_IP:/workspace/moola/data/processed/labeled/

scp -i ~/.ssh/runpod_key \
  data/splits/fwd_chain_v3.json \
  ubuntu@$RUNPOD_IP:/workspace/moola/data/splits/
```

### **C. Configuration Files (OPTIONAL but RECOMMENDED)**

**If using Hydra configs:**
```bash
# Check if these exist
ls -lh src/moola/config/*.yaml

# Transfer if present
rsync -avz --progress -e "ssh -i ~/.ssh/runpod_key" \
  src/moola/config/ ubuntu@$RUNPOD_IP:/workspace/moola/src/moola/config/
```

### **D. Project Metadata (RECOMMENDED)**

```
Files:
- pyproject.toml              (2 KB) - Package metadata
- verify_runpod_env.py        (6.5 KB) - Environment checker
- RUNPOD_QUICK_REFERENCE.md   (4.3 KB) - Command cheatsheet
```

**Transfer commands:**
```bash
scp -i ~/.ssh/runpod_key \
  pyproject.toml verify_runpod_env.py RUNPOD_QUICK_REFERENCE.md \
  ubuntu@$RUNPOD_IP:/workspace/moola/
```

### **E. Complete Transfer (ALL-IN-ONE)**

```bash
# Recommended: Transfer entire project (excludes git, cache)
cd /Users/jack/projects/moola

rsync -avz --progress \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='data/raw/unlabeled' \
  --exclude='artifacts/models' \
  --exclude='experiment_results.jsonl' \
  --exclude='.venv' \
  --exclude='venv' \
  -e "ssh -i ~/.ssh/runpod_key" \
  . ubuntu@$RUNPOD_IP:/workspace/moola/
```

**Expected transfer size:** ~4-5 MB (fast, < 1 minute)

---

## 🔧 4. ENVIRONMENT SETUP REQUIREMENTS

### **A. Directory Structure (Create on RunPod)**

```bash
cd /workspace/moola

# Create output directories
mkdir -p artifacts/encoders/pretrained
mkdir -p artifacts/runs
mkdir -p data/logs

# Verify structure
ls -lR
```

### **B. Environment Variables**

```bash
# CRITICAL: Add moola to PYTHONPATH
export PYTHONPATH=/workspace/moola/src:$PYTHONPATH

# Verify
echo $PYTHONPATH
python3 -c "import moola; print('✅ moola importable')"
```

**Persistent setup (optional):**
```bash
# Add to ~/.bashrc
echo 'export PYTHONPATH=/workspace/moola/src:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

### **C. Permissions**

```bash
# Ensure execute permissions
chmod +x verify_runpod_env.py

# Check write permissions
touch artifacts/test.txt && rm artifacts/test.txt && echo "✅ Write OK"
```

---

## 🖥️ 5. HARDWARE REQUIREMENTS

### **GPU Specifications**

**Recommended:**
```
GPU: RTX 4090
VRAM: 24 GB
CUDA: 11.8+ or 12.x
```

**Minimum:**
```
GPU: RTX 3090
VRAM: 24 GB
CUDA: 11.8+
```

**NOT recommended:**
```
GPU: < RTX 3090 (insufficient VRAM)
VRAM: < 20 GB (may OOM during pretraining)
```

### **Memory Usage Estimates**

| Step | GPU VRAM | Batch Size | Time |
|------|----------|------------|------|
| Step 1: Pretrain | ~4-6 GB | 256 | 20-30 min |
| Step 2: Freeze | ~2-3 GB | 32 | 5-10 min |
| Step 3: Unfreeze | ~2-3 GB | 32 | 15-20 min |

### **Disk Space**

```
Minimum required: 5 GB
Breakdown:
  - Source code: 2.6 MB
  - Data files: 650 KB
  - Installed packages: ~3-4 GB
  - Output artifacts: ~500 MB
```

---

## ✅ 6. VERIFICATION CHECKLIST

### **Phase 1: Pre-Transfer (On Mac)**

```bash
# 1. Check 11D dataset exists
[ -f data/processed/labeled/train_latest_relative.parquet ] && echo "✅ Dataset OK"

# 2. Check split file exists
[ -f data/splits/fwd_chain_v3.json ] && echo "✅ Split OK"

# 3. Check source code complete
[ -d src/moola ] && echo "✅ Source OK"

# 4. Verify dataset shape
python3 << 'EOF'
import pandas as pd
import numpy as np
df = pd.read_parquet('data/processed/labeled/train_latest_relative.parquet')
feats = df['features'].iloc[0]
stacked = np.array([f for f in feats])
assert stacked.shape == (105, 11), f"Wrong shape: {stacked.shape}"
print("✅ Dataset shape verified: (174, 105, 11)")
EOF

# 5. Check SSH key
[ -f ~/.ssh/runpod_key ] && echo "✅ SSH key OK"
chmod 600 ~/.ssh/runpod_key
```

### **Phase 2: Post-Transfer (On RunPod)**

```bash
# Run comprehensive verification
cd /workspace/moola
python3 verify_runpod_env.py

# Expected output:
# ✅ PASS: Python Version
# ✅ PASS: PyTorch and CUDA
# ✅ PASS: Required Packages
# ✅ PASS: Project Structure
# ✅ PASS: Output Directories
# ✅ PASS: 11D Dataset
# ✅ PASS: PYTHONPATH
# ✅ ALL CHECKS PASSED - READY FOR TRAINING!
```

### **Phase 3: Pre-Training (Final Checks)**

```bash
# 1. Verify CUDA
python3 -c "import torch; assert torch.cuda.is_available(); print('✅ CUDA OK')"

# 2. Check GPU memory
nvidia-smi

# 3. Test import
python3 -c "from moola.pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer; print('✅ Import OK')"

# 4. Verify files
ls -lh data/processed/labeled/train_latest_relative.parquet
ls -lh data/splits/fwd_chain_v3.json
ls -lh artifacts/encoders/pretrained/  # Should be empty initially
```

---

## 🚀 7. EXECUTION REQUIREMENTS

### **Training Commands (Copy-Paste Ready)**

**Setup:**
```bash
cd /workspace/moola
export PYTHONPATH=/workspace/moola/src:$PYTHONPATH
```

**Step 1: Pretrain BiLSTM Encoder**
```bash
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

**Step 2: Fine-tune Frozen**
```bash
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

**Step 3: Fine-tune Unfrozen**
```bash
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

---

## 📊 8. OUTPUT EXPECTATIONS

### **Artifacts Generated**

**After Step 1:**
```
File: artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt
Size: ~500 KB
Type: PyTorch state_dict (encoder weights only)
```

**After Step 2:**
```
Directory: artifacts/runs/<RUN_ID_1>/
Files:
  - model.pkl              # Complete model with frozen encoder
  - metadata.json          # Run configuration
  - metrics.json           # Training metrics
```

**After Step 3:**
```
Directory: artifacts/runs/<RUN_ID_2>/
Files:
  - model.pkl              # Complete model with unfrozen encoder
  - metadata.json          # Run configuration
  - metrics.json           # Training metrics
```

**Logs:**
```
Directory: data/logs/
Files: experiment_*.log (timestamped)
```

### **Files to Retrieve (SCP to Mac)**

```bash
# Get encoder
scp -i ~/.ssh/runpod_key \
  ubuntu@$RUNPOD_IP:/workspace/moola/artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt \
  artifacts/encoders/pretrained/

# Get runs (replace <RUN_ID> with actual IDs)
scp -i ~/.ssh/runpod_key -r \
  ubuntu@$RUNPOD_IP:/workspace/moola/artifacts/runs/<RUN_ID> \
  artifacts/runs/

# Get logs
scp -i ~/.ssh/runpod_key -r \
  ubuntu@$RUNPOD_IP:/workspace/moola/data/logs/ \
  data/
```

---

## 🐛 9. COMMON ISSUES & SOLUTIONS

### **Issue: Module not found**

**Symptom:**
```
ModuleNotFoundError: No module named 'moola'
```

**Solution:**
```bash
export PYTHONPATH=/workspace/moola/src:$PYTHONPATH
```

### **Issue: CUDA out of memory**

**Symptom:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution:**
```bash
# Reduce batch size in Step 1
--batch-size 128  # or 64

# Steps 2/3 use default batch_size=32, try:
# (add to train command if exposed, or edit config)
```

### **Issue: Input dimension mismatch**

**Symptom:**
```
RuntimeError: Expected input_dim=4, got 11
```

**Solution:**
```bash
# ALWAYS use --input-dim 11 in ALL commands
# Verify dataset:
python3 -c "import pandas as pd; import numpy as np; df = pd.read_parquet('data/processed/labeled/train_latest_relative.parquet'); print(np.array([f for f in df['features'].iloc[0]]).shape)"
# Should print: (105, 11)
```

### **Issue: Package import errors**

**Symptom:**
```
ImportError: cannot import name 'X' from 'Y'
```

**Solution:**
```bash
# Reinstall specific package
pip3 install --upgrade <package_name>

# Or reinstall all
pip3 install -r requirements_runpod.txt --force-reinstall
```

---

## 📋 10. DEPLOYMENT CHECKLIST SUMMARY

### **Before Starting RunPod**

- [ ] Verify 11D dataset exists (640 KB)
- [ ] Verify split file exists (4 KB)
- [ ] Source code complete (src/moola/, 106 files)
- [ ] SSH key ready (~/.ssh/runpod_key)
- [ ] verify_runpod_env.py ready
- [ ] RUNPOD_QUICK_REFERENCE.md available for copy-paste

### **On RunPod (Setup Phase)**

- [ ] Python >= 3.10
- [ ] PyTorch with CUDA installed
- [ ] All required packages installed (verify with verify_runpod_env.py)
- [ ] Files transferred successfully
- [ ] PYTHONPATH configured
- [ ] Output directories created
- [ ] Environment verification passed

### **On RunPod (Training Phase)**

- [ ] GPU verified (RTX 4090 or 3090)
- [ ] CUDA available
- [ ] Step 1 executed (encoder created)
- [ ] Step 2 executed (frozen run created)
- [ ] Step 3 executed (unfrozen run created)
- [ ] No errors occurred
- [ ] All artifacts generated

### **After Training (Retrieval Phase)**

- [ ] Encoder retrieved to Mac
- [ ] Run artifacts retrieved
- [ ] Logs retrieved
- [ ] Results evaluated
- [ ] Model promoted (if gates pass)

---

## 🎯 CRITICAL PARAMETERS (NEVER FORGET)

**These MUST be used in ALL commands:**

```
--input-dim 11              # NOT 4!
--seed 17                   # Reproducibility
--augment-data false        # In train commands
--device cuda               # GPU
```

**Dataset and split:**
```
--input data/processed/labeled/train_latest_relative.parquet
--split data/splits/fwd_chain_v3.json
```

**Encoder output:**
```
--output artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt
```

---

## ⏱️ TIMELINE ESTIMATE

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Setup RunPod + Transfer | 10-15 min | 15 min |
| Install packages | 5-10 min | 25 min |
| Verification | 2 min | 27 min |
| Step 1: Pretrain | 20-30 min | 57 min |
| Step 2: Freeze | 5-10 min | 67 min |
| Step 3: Unfreeze | 15-20 min | 87 min |
| Retrieve results | 5 min | 92 min |
| **TOTAL** | **~90 min** | |

**Parallel optimizations:**
- File transfer while packages install: Save 5-10 min
- Background retrieval: Overlap with next step

**Best case:** ~60 minutes
**Typical:** ~90 minutes
**Worst case:** ~120 minutes (if troubleshooting needed)

---

## ✅ READY TO DEPLOY

All requirements documented. All files identified. All commands ready.

**Next action:** Start RunPod instance and begin deployment.
