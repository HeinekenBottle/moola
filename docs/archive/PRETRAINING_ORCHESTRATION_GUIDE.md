# LSTM Pre-Training Orchestration Guide

## Executive Summary

**Status:** Configuration fixes applied ✓ | Orchestration script ready ✓ | Awaiting RunPod SSH ⏳

This guide documents the comprehensive ML operations campaign to execute BiLSTM masked autoencoder pre-training with parallel experiment orchestration.

---

## 🔧 Configuration Fixes Applied (Committed: `1044318`)

### Critical Changes to `src/moola/models/simple_lstm.py`:

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `hidden_size` | 64 | **128** | Match BiLSTM encoder forward direction |
| `num_heads` | 4 | **8** | 128/8 = 16 dims per head (optimal) |
| `time_warp_sigma` | 0.2 | **0.12** | Compatible with 15% masking (spec) |

**Impact:** These fixes enable proper weight transfer from pre-trained BiLSTM encoder and reduce temporal distortion.

**Expected Improvement:** +8-12% accuracy, Class 1 recovery from 0% to 40-60%

---

## 🏗️ Architecture Strategy (Clarification)

### Two-Stage Transfer Learning:

```
Stage 1: PRE-TRAINING (BiLSTM Masked Autoencoder)
├─ Architecture: Bidirectional LSTM encoder
│  ├─ Forward direction: 128 hidden units
│  └─ Backward direction: 128 hidden units (total output: 256)
├─ Dataset: 11,873 unlabeled samples
├─ Augmentation: 4x (time warp, jitter, scaling) → 59,365 samples
├─ Task: Masked reconstruction (15% patch masking)
└─ Output: Pre-trained encoder weights (forward + backward)

Stage 2: FINE-TUNING (Unidirectional SimpleLSTM)
├─ Architecture: Unidirectional LSTM (128 hidden, CAUSAL)
├─ Dataset: 89 labeled samples (consolidation vs retracement)
├─ Transfer: Load ONLY forward direction weights from BiLSTM
├─ Freeze: Encoder frozen initially, unfreeze after N epochs
└─ Output: Production-ready classifier
```

**Key Point:** SimpleLSTM remains **unidirectional** for causal inference. We only use the BiLSTM for pre-training to learn better features.

---

## 🚀 Parallel Experiment Matrix

Five experiments deployed in parallel to optimize hyperparameters:

| Experiment | Epochs | Mask Strategy | Time Warp σ | Aug Multiplier | Duration | Description |
|------------|--------|---------------|-------------|----------------|----------|-------------|
| **exp1_baseline** | 75 | patch | 0.12 | 4x | 35-40 min | **Recommended spec** |
| exp2_conservative | 50 | patch | 0.10 | 3x | 25-30 min | Safe, faster |
| exp3_aggressive | 100 | patch | 0.15 | 5x | 45-50 min | Maximum learning |
| exp4_block_mask | 75 | block | 0.12 | 4x | 35-40 min | Contiguous masking |
| exp5_random_mask | 75 | random | 0.12 | 4x | 35-40 min | Random masking |

**Total parallelization:** 5 experiments × ~40 min = **45-50 minutes total** (limited by slowest)

---

## 📋 Execution Instructions

### When RunPod SSH Reconnects:

```bash
cd /Users/jack/projects/moola

# Execute orchestration script
./scripts/orchestrate_pretraining_experiments.sh

# Or manually specify connection details:
./scripts/orchestrate_pretraining_experiments.sh root@213.173.108.43 15395 ~/.ssh/id_ed25519
```

### What the Script Does:

1. ✅ Verify SSH connection to RunPod
2. ✅ Upload `unlabeled_windows.parquet` (2.2 MB, 11,873 samples)
3. ✅ Sync updated codebase (with configuration fixes)
4. ✅ Launch 5 parallel pre-training experiments
5. ✅ Provide monitoring commands

### Validation Checklist:

After execution, verify logs show:

- [ ] **Dataset:** "Loaded 11,873 unlabeled samples" (not 89!)
- [ ] **Batches:** ~24 batches per epoch (not 1)
- [ ] **Duration:** 25-50 minutes (not 2-3)
- [ ] **Convergence:** Final val loss < 0.01
- [ ] **File size:** Encoder weights 10-20 MB
- [ ] **GPU usage:** RTX 4090 utilized properly

---

## 📊 Monitoring Commands

### Real-time Log Monitoring:

```bash
# Monitor all experiments
ssh -p 15395 root@213.173.108.43 -i ~/.ssh/id_ed25519 \
    'tail -f /workspace/logs/pretraining/*.log'

# Monitor baseline experiment (recommended)
ssh -p 15395 root@213.173.108.43 -i ~/.ssh/id_ed25519 \
    'tail -f /workspace/logs/pretraining/exp1_baseline.log'

# Monitor GPU utilization
ssh -p 15395 root@213.173.108.43 -i ~/.ssh/id_ed25519 \
    'watch -n 1 nvidia-smi'
```

### Check Completion Status:

```bash
# List completed checkpoints
ssh -p 15395 root@213.173.108.43 -i ~/.ssh/id_ed25519 \
    'ls -lh /workspace/data/artifacts/pretrained/*.pt'

# Expected output:
# -rw-r--r-- 1 root root 15M exp1_baseline_encoder.pt
# -rw-r--r-- 1 root root 15M exp2_conservative_encoder.pt
# -rw-r--r-- 1 root root 15M exp3_aggressive_encoder.pt
# -rw-r--r-- 1 root root 15M exp4_block_mask_encoder.pt
# -rw-r--r-- 1 root root 15M exp5_random_mask_encoder.pt
```

---

## 🎯 Post-Training: Encoder Selection & Fine-Tuning

### Step 1: Select Best Encoder

Based on validation loss from logs:

```bash
# Extract final validation losses
ssh -p 15395 root@213.173.108.43 -i ~/.ssh/id_ed25519 \
    'grep "Final val loss" /workspace/logs/pretraining/*.log'
```

**Selection Criteria:**
1. Lowest validation loss (< 0.01 preferred)
2. Stable convergence (no divergence)
3. File size 10-20 MB (architecture verification)

**Recommended:** Start with `exp1_baseline_encoder.pt` (matches spec exactly)

### Step 2: Fine-Tune SimpleLSTM

```bash
# Run OOF pipeline with pre-trained encoder
ssh -p 15395 root@213.173.108.43 -i ~/.ssh/id_ed25519 "
cd /workspace/moola && \
python3 -m moola.cli oof \
    --model simple_lstm \
    --device cuda \
    --seed 1337 \
    --load-pretrained-encoder data/artifacts/pretrained/exp1_baseline_encoder.pt
"
```

### Step 3: Validate Performance

**Expected Results:**

| Metric | Baseline (Current) | Target (Post-Pretraining) | Improvement |
|--------|-------------------|---------------------------|-------------|
| **Overall Accuracy** | 57.1% | **64-72%** | +7-15 points |
| **Class 0 (Consolidation)** | 100.0% | 75-90% | Normalized |
| **Class 1 (Retracement)** | **0.0%** | **40-60%** | **+40-60 points** |

**Success Criteria:**
- ✅ Class 1 accuracy > 40% (recovery from collapse)
- ✅ Overall accuracy > 64%
- ✅ No class collapse (both classes > 30%)

---

## 🔬 Experimental Design Rationale

### Why 5 Parallel Experiments?

1. **exp1_baseline:** Validate spec implementation (high confidence)
2. **exp2_conservative:** Insurance policy if baseline overfits
3. **exp3_aggressive:** Push limits for maximum performance
4. **exp4_block_mask:** Test contiguous masking hypothesis
5. **exp5_random_mask:** Baseline masking strategy comparison

### Hyperparameter Justification:

| Parameter | Range Tested | Rationale |
|-----------|--------------|-----------|
| **Epochs** | 50-100 | Balance convergence vs overfitting |
| **Time Warp σ** | 0.10-0.15 | Goldilocks zone (spec: 0.12) |
| **Mask Strategy** | random/block/patch | Pivot point preservation |
| **Augmentation** | 3x-5x | Dataset expansion (11,873 → 35K-59K) |

---

## 📈 Expected Training Dynamics

### Healthy Pre-Training Signals:

```
Epoch 1:  Train Loss: 0.025  Val Loss: 0.023  [Initial high loss]
Epoch 10: Train Loss: 0.012  Val Loss: 0.013  [Rapid descent]
Epoch 25: Train Loss: 0.007  Val Loss: 0.008  [Convergence zone]
Epoch 50: Train Loss: 0.003  Val Loss: 0.004  [Fine-tuning]
Epoch 75: Train Loss: 0.001  Val Loss: 0.002  [Near-perfect reconstruction]
```

**Red Flags:**
- ❌ Val loss > 0.05 after 25 epochs (not learning)
- ❌ Val loss diverges from train loss (overfitting)
- ❌ Only 1 batch per epoch (wrong dataset)
- ❌ Completes in < 20 minutes (insufficient data)

---

## 🛠️ Troubleshooting

### Issue: "Connection refused" to RunPod

**Solution:** Wait for RunPod instance to start, then re-run orchestration script.

### Issue: "Shape mismatch" when loading encoder

**Check:**
```bash
# Verify SimpleLSTM hidden_size=128
grep "hidden_size.*128" src/moola/models/simple_lstm.py

# Verify BiLSTM encoder hidden_dim=128
ssh -p 15395 root@213.173.108.43 -i ~/.ssh/id_ed25519 \
    "python3 -c 'import torch; ckpt = torch.load(\"/workspace/data/artifacts/pretrained/exp1_baseline_encoder.pt\"); print(ckpt[\"hyperparams\"])'"
```

### Issue: Pre-training uses wrong data (89 samples)

**Fix:** Verify unlabeled data uploaded:
```bash
ssh -p 15395 root@213.173.108.43 -i ~/.ssh/id_ed25519 \
    "python3 -c 'import pandas as pd; df = pd.read_parquet(\"/workspace/data/raw/unlabeled_windows.parquet\"); print(f\"Samples: {len(df)}\")'"

# Expected: "Samples: 11873"
```

---

## 📚 Related Documentation

- **MLOps Audit Report:** Comprehensive findings from specialized agents (see Claude Code session)
- **BiLSTM Integration Guide:** `docs/BILSTM_MASKED_AUTOENCODER_INTEGRATION_GUIDE.md`
- **LSTM Optimization Spec:** `docs/LSTM_OPTIMIZATION_ANALYSIS_PHASE_IV.md`
- **Configuration Code:** `src/moola/config/training_config.py` (lines 210-276)
- **Pre-training CLI:** `src/moola/cli.py` (lines 532-685)

---

## 🎯 Success Metrics

### Immediate (Post-Pretraining):
- [ ] All 5 encoders trained successfully
- [ ] Best encoder validation loss < 0.01
- [ ] Encoder weights load into SimpleLSTM without errors
- [ ] Fine-tuning shows improved Class 1 accuracy

### Long-term (Production):
- [ ] Consistent 64%+ accuracy across seeds
- [ ] No class collapse on new data
- [ ] Encoder reusable for future experiments
- [ ] 30-40 minute pre-training pipeline established

---

## 🚀 Next Steps After Orchestration Completes

1. **Download all encoder checkpoints** to local machine for archival
2. **Run ablation study** comparing all 5 encoders
3. **Document best configuration** for production deployment
4. **Archive logs and metrics** in MLflow
5. **Update production pipeline** to use pre-trained encoder by default

---

**Generated:** 2025-10-17
**Status:** Ready for execution (awaiting RunPod SSH)
**Orchestration Script:** `scripts/orchestrate_pretraining_experiments.sh`
**Contact:** Claude Code MLOps Orchestration
