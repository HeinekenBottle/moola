# 11D Training Pipeline - READY TO RUN
**Date:** 2025-10-20  
**Status:** ✅ ALL FIXES APPLIED - READY FOR EXECUTION  
**Branch:** `refactor/architecture-cleanup`

---

## 🎯 ChatGPT's Requirements - ALL MET

### **Fixes Applied** ✅

1. ✅ **Added --input-dim 11 to train command**
   - Parameter added to CLI
   - Auto-detected if not specified
   - Passed to model kwargs

2. ✅ **Added --seed parameter**
   - Overrides config seed
   - Ensures seed=17 end-to-end

3. ✅ **Added --save-run flag**
   - Saves run artifacts to artifacts/runs/
   - Includes metadata, model, metrics
   - Generates unique run_id for tracking

4. ✅ **--augment-data false enforced**
   - Prevents augmentation in val/test
   - Explicit in all commands

---

## 🚀 READY TO EXECUTE

### **Option 1: Run Complete Pipeline (Recommended)**

```bash
./RUN_11D_TRAINING.sh
```

This script executes all 3 training steps sequentially:
1. Pretrain 11D BiLSTM encoder (~20-30 min)
2. Fine-tune with frozen encoder (~5-10 min)
3. Fine-tune with unfrozen encoder (~15-20 min)

**Total time:** ~40-60 minutes on GPU

---

### **Option 2: Run Steps Individually**

#### **Step 1: Pretrain 11D BiLSTM Encoder**

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

**Output:** `artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt`

---

#### **Step 2: Fine-tune (Freeze Encoder)**

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

**Output:** Run artifacts in `artifacts/runs/<RUN_ID>/`

---

#### **Step 3: Fine-tune (Unfreeze Encoder)**

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

**Output:** Run artifacts in `artifacts/runs/<RUN_ID>/`

---

#### **Step 4: Evaluate and Gate**

```bash
# Replace <RUN_ID> with the actual run ID from Step 3
python3 -m moola.cli eval \
  --run artifacts/runs/<RUN_ID> \
  --metrics pr_auc brier ece accuracy f1_macro f1_per_class \
  --event-metrics hit_at_pm3 lead_lag pointer_f1 \
  --save-reliability-plot artifacts/plots/reliability_enh_rel.png \
  --save-metrics artifacts/metrics/enh_rel.json
```

**Promotion Rule:**
- ✅ PR-AUC ↑ (higher than baseline)
- ✅ Brier ↓ (lower than baseline)
- ✅ ECE ≤ baseline + 0.02

**If gates pass:**
```bash
cp artifacts/runs/<RUN_ID>/model.pkl \
   artifacts/models/supervised/enhanced_baseline_v2_relative.pt
```

---

## 📋 Verification Checklist

### **Prerequisites** ✅
- ✅ 11D dataset created: `train_latest_relative.parquet` (639 KB, 174 samples)
- ✅ Split file exists: `fwd_chain_v3.json` (purge_window=104)
- ✅ CLI enhanced: --input-dim, --seed, --save-run added
- ✅ Conversion script: `make_relative_parquet.py` working
- ✅ All validations passed: no NaN/Inf values

### **Pitfalls Avoided** ✅
- ✅ Feature-dim mismatch: --input-dim 11 everywhere
- ✅ Hidden seeds: --seed 17 end-to-end
- ✅ Aug in val/test: --augment-data false enforced

### **Components Ready** ✅
- ✅ RelativeFeatureTransform (4D → 11D)
- ✅ MaskedLSTMPretrainer (11D support)
- ✅ EnhancedSimpleLSTM (pretrained encoder support)
- ✅ CLI train command (all parameters added)
- ✅ CLI pretrain-bilstm (11D support)

---

## 🎯 What We're Working Toward

### **Short Term**
EnhancedSimpleLSTM + RelativeTransform + MAE encoder → `enhanced_baseline_v2_relative.pt`

### **Then**
CPSA 0.25 → 0.5 (val/test real)

### **Later**
TS2Vec pretrain for stronger adapter

---

## 📊 Expected Results

### **Pretraining (Step 1)**
- Encoder learns 11D feature structure
- Validation loss should decrease steadily
- Final encoder: ~500 KB file

### **Fine-tuning Freeze (Step 2)**
- Quick convergence (5 epochs)
- Classifier head adapts to task
- Baseline performance established

### **Fine-tuning Unfreeze (Step 3)**
- Full model optimization (25 epochs)
- Expected improvement: +5-8% accuracy
- Better calibration and PR-AUC

### **Evaluation (Step 4)**
- PR-AUC should improve over baseline
- Brier score should decrease
- ECE should remain low (≤ baseline + 0.02)

---

## 📁 Files Created/Modified

### **Created (This Session)**
1. `CHATGPT_TRAJECTORY_ANALYSIS.md` - Analysis of trajectory requirements
2. `scripts/data/make_relative_parquet.py` - 4D→11D conversion script
3. `data/processed/labeled/train_latest_relative.parquet` - 11D dataset
4. `11D_PIPELINE_BUILD_SUMMARY.md` - Build documentation
5. `RUN_11D_TRAINING.sh` - Complete execution script
6. `READY_TO_RUN.md` - This document

### **Modified (This Session)**
1. `src/moola/cli.py` - Enhanced pretrain-bilstm and train commands

---

## 📝 Git Commits

1. **`feat: Add 11D RelativeTransform support for training pipeline`**
   - CLI pretrain-bilstm: --input-dim, --seed
   - make_relative_parquet script
   - 11D dataset created

2. **`docs: Add 11D pipeline build summary`**
   - Complete build documentation

3. **`feat: Add --input-dim, --seed, and --save-run to train command`**
   - CLI train: --input-dim, --seed, --save-run
   - Run artifact tracking

---

## 🚀 Quick Start

### **1. Verify Prerequisites**
```bash
# Check 11D dataset exists
ls -lh data/processed/labeled/train_latest_relative.parquet

# Check split file exists
cat data/splits/fwd_chain_v3.json | grep purge_window

# Check CUDA available
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### **2. Run Training**
```bash
# Option A: Complete pipeline
./RUN_11D_TRAINING.sh

# Option B: Individual steps (see above)
```

### **3. Monitor Progress**
```bash
# Watch logs
tail -f data/logs/*.log

# Check encoder after Step 1
ls -lh artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt

# Check runs after Steps 2 & 3
ls -lh artifacts/runs/
```

### **4. Evaluate and Promote**
```bash
# Find latest run
ls -lt artifacts/runs/ | head -2

# Evaluate (replace <RUN_ID>)
python3 -m moola.cli eval --run artifacts/runs/<RUN_ID> ...

# Promote if gates pass
cp artifacts/runs/<RUN_ID>/model.pkl \
   artifacts/models/supervised/enhanced_baseline_v2_relative.pt
```

---

## ✅ READY TO EXECUTE

All components built, all fixes applied, all validations passed.

**To start training:**
```bash
./RUN_11D_TRAINING.sh
```

**Estimated time:** 40-60 minutes on GPU

**Expected outcome:** Enhanced baseline model with 11D RelativeTransform features and pretrained encoder, ready for promotion if gates pass.

---

## 🎉 SUCCESS CRITERIA

### **Build Phase** ✅ COMPLETE
- ✅ All components built
- ✅ All fixes applied
- ✅ All validations passed

### **Training Phase** ⏳ READY TO RUN
- ⏳ Encoder pretrains successfully
- ⏳ Fine-tuning converges
- ⏳ Metrics improve over baseline
- ⏳ Gates pass for promotion

**Status:** READY FOR EXECUTION 🚀

