# ChatGPT Trajectory Analysis - What Exists in Codebase
**Date:** 2025-10-20  
**Purpose:** Verify which components exist for the proposed training trajectory

---

## 🎯 ChatGPT's Proposed Trajectory

**Goal:** Train EnhancedSimpleLSTM with 11D RelativeTransform features + pretrained BiLSTM encoder

**Steps:**
1. Build 11D dataset from 4D OHLC
2. Pretrain BiLSTM encoder on 11D features (masked autoencoder)
3. Fine-tune EnhancedSimpleLSTM (freeze → unfreeze)
4. Evaluate and gate promotion

---

## ✅ Clarification Items - Status Check

### 1. Split File Path and Purge Window ✅ **EXISTS**

**File:** `data/splits/fwd_chain_v3.json`

**Status:** ✅ **CONFIRMED**
- File exists at correct location
- Contains `"purge_window": 104` ✅
- Forward-chaining 80/20 split (139 train, 35 val)
- Created: 2025-10-19
- Dataset: train_latest.parquet (174 samples)

**Verdict:** Ready to use as-is

---

### 2. Encoder Name ❌ **DOES NOT EXIST**

**Expected:** `artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt`

**Status:** ❌ **MISSING**

**What exists:**
- `artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt` ✅ (546 KB, 4D OHLC)
- `artifacts/encoders/pretrained/tstcc_encoder_v1.pt` ✅ (3.5 MB, TS-TCC)

**What's missing:**
- `bilstm_mae_11d_v1.pt` - 11D RelativeTransform encoder

**Verdict:** Need to create this encoder (Step 2 of trajectory)

---

### 3. CLI Entry Point ✅ **EXISTS**

**Command:** `python -m moola.cli pretrain-bilstm`

**Status:** ✅ **CONFIRMED**

**Location:** `src/moola/cli.py` line 823-949

**Current Implementation:**
```python
@app.command()
def pretrain_bilstm(
    cfg_dir, over, input_path, output_path, device, epochs, patience,
    mask_ratio, mask_strategy, patch_size, hidden_dim, batch_size,
    augment, num_augmentations
):
    """Pre-train bidirectional masked LSTM autoencoder on unlabeled OHLC data."""
    
    # Initialize pre-trainer
    pretrainer = MaskedLSTMPretrainer(
        input_dim=4,  # OHLC ⚠️ HARDCODED TO 4D
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.2,
        mask_ratio=mask_ratio,
        mask_strategy=mask_strategy,
        patch_size=patch_size,
        learning_rate=1e-3,
        batch_size=batch_size,
        device=device,
        seed=cfg.seed,
    )
```

**Issue:** ⚠️ **HARDCODED TO 4D**
- Line 919: `input_dim=4` is hardcoded
- Need to add `--input-dim` CLI option to support 11D

**Alternative:** `scripts/pretraining/runpod_pretrain_bilstm.py`
- Also hardcoded to 4D (line 103)
- Would need modification

**Verdict:** CLI exists but needs modification to support 11D

---

### 4. Feature Shape ❌ **DOES NOT EXIST**

**Expected:** `data/processed/labeled/train_latest_relative.parquet` with (105, 11)

**Status:** ❌ **MISSING**

**What exists:**
- `data/processed/labeled/train_latest.parquet` ✅ (170 KB, 4D OHLC)

**What's missing:**
- `train_latest_relative.parquet` - 11D RelativeTransform version

**Verdict:** Need to create this file (Step 1 of trajectory)

---

### 5. Seeds ⚠️ **PARTIALLY CORRECT**

**Expected:** Global seed 17 only, no hidden per-module seeds

**Status:** ⚠️ **NEEDS VERIFICATION**

**Current Implementation:**
- CLI uses `cfg.seed` from config (default: 1337)
- `MaskedLSTMPretrainer` accepts `seed` parameter
- `set_seed()` utility exists in `src/moola/utils/seeds.py`

**Issues:**
- Default seed is 1337, not 17
- Need to verify no hidden seeds in modules

**Verdict:** Need to change default seed to 17 and audit for hidden seeds

---

## 📋 What Exists in Codebase

### ✅ **Core Components (Ready to Use)**

1. **RelativeFeatureTransform** ✅
   - Location: `src/moola/features/relative_transform.py`
   - Converts [N, 105, 4] OHLC → [N, 105, 11] relative features
   - Features: 4 log returns + 3 candle ratios + 4 rolling z-scores
   - Ready to use

2. **MaskedLSTMPretrainer** ✅
   - Location: `src/moola/pretraining/masked_lstm_pretrain.py`
   - Supports masked autoencoder pretraining
   - Configurable input_dim (currently hardcoded to 4 in CLI)
   - Ready to use (with CLI modification)

3. **EnhancedSimpleLSTM** ✅
   - Location: `src/moola/models/simple_lstm.py`
   - Supports pretrained encoder loading
   - Supports freeze/unfreeze
   - Ready to use

4. **CLI train command** ✅
   - Location: `src/moola/cli.py` line 113-400
   - Supports `--pretrained-encoder` option
   - Supports `--freeze-encoder` flag
   - Ready to use

5. **Feature11DIntegrator** ✅
   - Location: `src/moola/data/feature_11d_integration.py`
   - Utility for loading and transforming to 11D
   - Functions: `load_11d_features()`, `create_enhanced_dataset()`
   - Ready to use

### ⚠️ **Components Needing Modification**

1. **CLI pretrain-bilstm** ⚠️
   - Hardcoded to `input_dim=4`
   - Need to add `--input-dim` CLI option
   - **Fix:** Add CLI parameter and pass to MaskedLSTMPretrainer

2. **Default seed** ⚠️
   - Currently 1337, should be 17
   - **Fix:** Change default in config or pass `--seed 17` to all commands

### ❌ **Missing Components**

1. **make_relative_parquet script** ❌
   - ChatGPT suggests: `python -m scripts.make_relative_parquet`
   - **Does not exist** in codebase
   - **Alternative:** Use `Feature11DIntegrator` directly

2. **11D pretrained encoder** ❌
   - File: `artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt`
   - **Does not exist** - need to create via pretraining

3. **11D training dataset** ❌
   - File: `data/processed/labeled/train_latest_relative.parquet`
   - **Does not exist** - need to create from train_latest.parquet

---

## 🔧 Required Fixes Before Execution

### **Fix 1: Add --input-dim to CLI pretrain-bilstm**

**File:** `src/moola/cli.py`

**Change:**
```python
# Add CLI option
@click.option("--input-dim", default=4, type=int, help="Input feature dimension (4 for OHLC, 11 for RelativeTransform)")

# Update function signature
def pretrain_bilstm(
    cfg_dir, over, input_path, output_path, device, epochs, patience,
    mask_ratio, mask_strategy, patch_size, hidden_dim, batch_size,
    augment, num_augmentations, input_dim  # ADD THIS
):
    # ...
    
    # Update MaskedLSTMPretrainer initialization
    pretrainer = MaskedLSTMPretrainer(
        input_dim=input_dim,  # CHANGE FROM 4 TO input_dim
        hidden_dim=hidden_dim,
        # ...
    )
```

### **Fix 2: Create make_relative_parquet utility**

**Option A:** Create new script
```python
# scripts/data/make_relative_parquet.py
import pandas as pd
import numpy as np
from pathlib import Path
from moola.features.relative_transform import RelativeFeatureTransform

def main():
    # Load 4D OHLC data
    df = pd.read_parquet("data/processed/labeled/train_latest.parquet")
    
    # Extract OHLC features
    ohlc_data = np.array(df['features'].tolist(), dtype=np.float32)
    
    # Transform to 11D
    transformer = RelativeFeatureTransform()
    relative_data = transformer.transform(ohlc_data)
    
    # Create new dataframe
    df_relative = df.copy()
    df_relative['features'] = list(relative_data)
    
    # Save
    df_relative.to_parquet("data/processed/labeled/train_latest_relative.parquet")
    print(f"✅ Created train_latest_relative.parquet with shape {relative_data.shape}")

if __name__ == "__main__":
    main()
```

**Option B:** Use existing Feature11DIntegrator
```python
from moola.data.feature_11d_integration import Feature11DIntegrator

integrator = Feature11DIntegrator()
df = integrator.create_dual_input_dataset(
    input_path="data/processed/labeled/train_latest.parquet",
    output_path="data/processed/labeled/train_latest_relative.parquet"
)
```

---

## 📝 Corrected Execution Plan

### **Step 0: Fix CLI and Create Utilities**

```bash
# 1. Add --input-dim to CLI pretrain-bilstm (manual code change)
# 2. Create make_relative_parquet.py script
```

### **Step 1: Build 11D Dataset**

```bash
# Option A: Use new script
python scripts/data/make_relative_parquet.py

# Option B: Use Feature11DIntegrator
python -c "
from moola.data.feature_11d_integration import Feature11DIntegrator
integrator = Feature11DIntegrator()
integrator.create_dual_input_dataset(
    'data/processed/labeled/train_latest.parquet',
    'data/processed/labeled/train_latest_relative.parquet'
)
"
```

### **Step 2: Pretrain BiLSTM Encoder on 11D**

```bash
python -m moola.cli pretrain-bilstm \
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

### **Step 3: Fine-tune EnhancedSimpleLSTM (Freeze)**

```bash
python -m moola.cli train \
  --model enhanced_simple_lstm \
  --data data/processed/labeled/train_latest_relative.parquet \
  --split data/splits/fwd_chain_v3.json \
  --pretrained-encoder artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt \
  --freeze-encoder \
  --epochs 5 \
  --device cuda \
  --seed 17
```

### **Step 4: Fine-tune EnhancedSimpleLSTM (Unfreeze)**

```bash
python -m moola.cli train \
  --model enhanced_simple_lstm \
  --data data/processed/labeled/train_latest_relative.parquet \
  --split data/splits/fwd_chain_v3.json \
  --pretrained-encoder artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt \
  --no-freeze-encoder \
  --epochs 25 \
  --device cuda \
  --seed 17
```

---

## 🎯 Summary

### **What Exists:**
✅ RelativeFeatureTransform (4D → 11D)  
✅ MaskedLSTMPretrainer (configurable input_dim)  
✅ EnhancedSimpleLSTM (pretrained encoder support)  
✅ CLI train command (freeze/unfreeze support)  
✅ Split file (fwd_chain_v3.json with purge_window=104)  
✅ Feature11DIntegrator utility  

### **What's Missing:**
❌ 11D pretrained encoder (bilstm_mae_11d_v1.pt)  
❌ 11D training dataset (train_latest_relative.parquet)  
❌ make_relative_parquet script  

### **What Needs Fixing:**
⚠️ CLI pretrain-bilstm hardcoded to input_dim=4  
⚠️ Default seed is 1337 (should be 17)  

### **Action Items:**
1. Add `--input-dim` parameter to CLI pretrain-bilstm
2. Create make_relative_parquet.py script (or use Feature11DIntegrator)
3. Change default seed to 17 (or always pass --seed 17)
4. Execute corrected trajectory

---

## 🚀 Ready to Execute?

**Prerequisites:**
- [ ] Fix CLI pretrain-bilstm (add --input-dim)
- [ ] Create make_relative_parquet.py
- [ ] Verify seed handling

**Then execute:**
1. Build 11D dataset
2. Pretrain encoder on 11D
3. Fine-tune with freeze
4. Fine-tune with unfreeze
5. Evaluate and gate

