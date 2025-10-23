# Moola Validation Plan - Pre-Computation & Training

**Date:** 2025-10-23
**Status:** Ready for pre-computation → Training implementation needed

---

## ✅ COMPLETED: Critical Fixes (P0)

### 1. **Expansion Proxy Feature** ✅
**Location:** `src/moola/features/relativity.py`

**What changed:**
- Added 11th feature: `expansion_proxy = range_z × leg_dir × body_pct`
- Bounded to `[-2, 2]` as per Grok's specification
- Updated all metadata and documentation

**Impact:**
- **REQUIRES RE-PRECOMPUTATION** of 5-year NQ data
- Old `features_10d.npy` is now obsolete
- New output: `features_11d.npy`

---

### 2. **Phantom Models Removed** ✅
**Location:** `src/moola/cli.py:127`

**What changed:**
- Removed "sapphire" and "opal" from CLI choices
- Only "jade" model remains (matches registry)

**Impact:**
- Prevents AI confusion and user errors
- Clean, unambiguous model selection

---

### 3. **Uncertainty Weighting Parameters** ✅
**Location:** `src/moola/models/jade_core.py:105-114`

**What changed:**
- Added learnable `log_sigma_ptr` and `log_sigma_type` parameters
- Initialized: `log_sigma_ptr=-0.30` (σ ≈ 0.74), `log_sigma_type=0.00` (σ ≈ 1.0)
- Returns σ values in forward pass for loss computation

**Implementation:**
```python
# In model forward():
output["sigma_ptr"] = torch.exp(self.log_sigma_ptr)
output["sigma_type"] = torch.exp(self.log_sigma_type)
```

**Loss formula (Kendall et al., CVPR 2018):**
```
L_total = (1/2σ_ptr²)L_ptr + (1/2σ_type²)L_type + log(σ_ptr × σ_type)
```

**Impact:**
- Automatic task balancing (no manual λ tuning)
- Prevents pointer task collapse
- **REQUIRES:** Training script to use these σ values in loss computation

---

### 4. **Pre-computation Script Updated** ✅
**Location:** `scripts/precompute_nq_features.py`

**What changed:**
- Updated output filename: `features_10d.npy` → `features_11d.npy`
- Added expansion_proxy to feature list documentation
- Added performance notes (CPU-bound, 1-3 hours, sequential zigzag)

**Hardware recommendation:**
- **CPU ONLY** (not GPU-compatible due to pandas/numpy operations)
- High clock speed (4-5 GHz) > Many cores
- 16-32 GB RAM
- Est. 1-3 hours on good CPU

---

## ⚠️ PENDING: Critical Gaps (Must Fix Before Training)

### 1. **WeightedRandomSampler NOT Connected** ❌
**Problem:** `scripts/run_supervised_train.py` imports non-existent utility functions

**Missing files:**
- `src/moola/utils/training/training_utils.py` (doesn't exist)
- `setup_optimized_dataloader()` function

**Grok requirement:**
```python
from torch.utils.data import WeightedRandomSampler

# Compute class weights
class_counts = torch.bincount(y_train)
weights = 1.0 / class_counts[y_train]

# Create sampler
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

# Use in DataLoader
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

**Status:** Implementation needed before training

---

### 2. **Input Size Default Still 10** ⚠️
**Location:** Multiple files

**Need to update:**
- `src/moola/models/jade_core.py:41` - Change `input_size: int = 10` → `input_size: int = 11`
- Any config files with `input_size: 10`
- Registry build functions

**Impact:** Model will fail if not updated before using 11-feature data

---

## 📋 VALIDATION CHECKLIST

### Phase 1: Pre-Computation (CPU Job - 1-3 hours)

```bash
# On RunPod or local Mac
python3 scripts/precompute_nq_features.py \
  --data data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet \
  --output data/processed/nq_features \
  --config configs/windowed.yaml
```

**Expected outputs:**
- `data/processed/nq_features/features_11d.npy` (7.1+ GB)
- `data/processed/nq_features/valid_mask.npy`
- `data/processed/nq_features/metadata.json`
- `data/processed/nq_features/splits.json`

**Validation:**
```bash
python3 scripts/verify_precomputed_features.py \
  --features data/processed/nq_features/features_11d.npy \
  --metadata data/processed/nq_features/metadata.json
```

**Check:**
- Feature shape: `[N_windows, 105, 11]` (was `[..., 10]`)
- Feature names include `expansion_proxy`
- Valid mask ratio >90%
- Non-zero density >50% (Grok's target)

---

### Phase 2: Training Implementation (Before RunPod)

**Must implement:**

1. **Create `src/moola/utils/training/training_utils.py`** with:
   - `setup_optimized_dataloader()` with WeightedRandomSampler support
   - `initialize_model_biases()` function
   - `validate_batch_schema()` function

2. **Update training script** (`scripts/run_supervised_train.py`):
   - Connect WeightedRandomSampler
   - Use uncertainty-weighted loss from model σ parameters
   - Implement head-only training → unfreeze schedule

3. **Update model defaults:**
   - Change `input_size=10` → `input_size=11` in jade_core.py
   - Update any configs with old feature count

---

### Phase 3: Sanity Checks (Local - 5 minutes)

```bash
# Test model instantiation
python3 -c "
from moola.models.jade_core import JadeCompact
import torch

model = JadeCompact(input_size=11, predict_pointers=True)
x = torch.randn(4, 105, 11)
output = model(x)

assert 'logits' in output
assert 'pointers' in output
assert 'sigma_ptr' in output
assert 'sigma_type' in output
print('✅ Model with uncertainty weighting works!')
"

# Test feature loading
python3 -c "
import numpy as np
X = np.load('data/processed/nq_features/features_11d.npy')
assert X.shape[2] == 11, f'Expected 11 features, got {X.shape[2]}'
print(f'✅ Features loaded: {X.shape}')
"
```

---

## 🚀 DEPLOYMENT SEQUENCE

### 1. Pre-Compute (RunPod CPU)
```bash
# SSH to RunPod
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP

# Sync code
cd /workspace
rsync -avz --exclude data --exclude artifacts \
  jack@YOUR_MAC_IP:/Users/jack/projects/moola/ ./moola/

# Run pre-computation (1-3 hours)
cd moola
python3 scripts/precompute_nq_features.py \
  --data data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet \
  --output data/processed/nq_features
```

### 2. Validate Features
```bash
python3 scripts/verify_precomputed_features.py \
  --features data/processed/nq_features/features_11d.npy
```

### 3. SCP Results Back to Mac
```bash
# From Mac
scp -i ~/.ssh/runpod_key -r \
  ubuntu@YOUR_IP:/workspace/moola/data/processed/nq_features/ \
  /Users/jack/projects/moola/data/processed/
```

### 4. Implement Training (Next Session)
- Fix training utilities
- Connect WeightedRandomSampler
- Implement uncertainty-weighted loss
- Test locally with small data
- Deploy to RunPod GPU for full training

---

## 📊 SUCCESS CRITERIA

### Pre-Computation:
- ✅ features_11d.npy exists and loads
- ✅ Shape matches `[N_windows, 105, 11]`
- ✅ `expansion_proxy` in feature names
- ✅ Non-zero density >50%
- ✅ Valid mask ratio >90%

### Training (After Implementation):
- ✅ No class collapse (both classes >30% predicted)
- ✅ F1_macro ≥60% (Grok's target)
- ✅ Hit@±3 ≥60% on validation
- ✅ σ_ptr, σ_type values converge (σ ~0.5-2.0)
- ✅ No NaN/Inf in losses

---

## 🔧 NEXT STEPS

**Immediate (Before RunPod):**
1. Review this validation plan
2. Decide: Start pre-computation now, or implement training first?
3. If pre-compute first: Run on RunPod CPU (1-3 hours)
4. If training first: Implement missing utilities, test locally

**After Pre-Computation:**
1. Implement training utilities (WeightedRandomSampler, loss)
2. Test training on 10% of data locally
3. Deploy full training to RunPod GPU
4. Monitor for collapse, validate metrics

**Questions?**
- Which path do you prefer? (Pre-compute now vs. implement training first)
- Do you want help implementing the missing training utilities?
- Should we archive the old training scripts and start fresh?
