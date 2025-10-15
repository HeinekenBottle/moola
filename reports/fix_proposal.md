# Fix Proposal: CNN-Transformer Mode Collapse
**Date**: 2025-10-14
**Issue**: CNN-Transformer stuck at 56.5% with mode-collapsed predictions

---

## Critical Findings

### 1. Mode Collapse Confirmed
The CNN-Transformer has **completely stopped learning** and outputs nearly identical predictions for all samples:

```
Prediction Statistics:
- Always predicts class 0: 115/115 (100%)
- Class 0 prob range: [0.5663, 0.5731] (only 0.0068 variation!)
- Standard deviation: 0.0024 (near zero)
- Unique values: Only 4 distinct predictions across 115 samples
```

**Comparison to XGBoost**:
```
XGBoost (still learning despite issues):
- Class 0 prob range: [0.0684, 0.9650] (0.8966 variation)
- Standard deviation: 0.2785 (116x higher than CNN-Transformer!)
- Predictions: 70 favor class 0, 45 favor class 1 (diverse)
```

### 2. Root Cause: Data Corruption
**18/115 samples (15.7%) have invalid expansion indices**:
- 9 samples: zero-length expansions (start == end)
- 3 samples: out-of-bounds start positions
- 8 samples: out-of-bounds end positions

These corrupted samples create **conflicting gradient signals** in the multi-task learning:
- Classification loss tries to learn consolidation vs retracement
- Pointer losses receive invalid targets (zero-length or clipped positions)
- Gradients pull in contradictory directions
- Model defaults to constant prediction to minimize loss

### 3. Training Dynamics
- Early stopping triggered at **epoch 48** (patience=20)
- Validation loss stopped improving after ~28 epochs
- All 5 folds: exactly **56.5% accuracy** (majority class baseline)
- No fold shows any learning signal

---

## Proposed Fixes (Priority Order)

### Fix 1: Clean Training Data (CRITICAL - Do This First)

**Action**: Remove 18 problematic samples before training

**Implementation**:
```python
# File: src/moola/data/load.py or preprocessing script
import pandas as pd

def load_clean_train_data(path="data/processed/train.parquet"):
    """Load training data with invalid expansion indices removed."""
    df = pd.read_parquet(path)

    # Remove problematic samples
    problematic_indices = [2, 14, 22, 27, 40, 42, 44, 50, 66, 72, 78, 80, 83, 85, 103, 107, 112, 114]
    df_clean = df.drop(df.index[problematic_indices])

    print(f"[DATA CLEAN] Removed {len(problematic_indices)} problematic samples")
    print(f"[DATA CLEAN] Clean dataset: {len(df_clean)} samples")

    return df_clean

# Or use validation-based filtering:
def validate_expansions(df):
    """Remove samples with invalid expansion indices."""
    start = df['expansion_start'].values
    end = df['expansion_end'].values

    valid_mask = (
        (start < end) &  # No zero-length
        (start >= 30) & (start <= 74) &  # Valid start range
        (end >= 30) & (end <= 74)  # Valid end range
    )

    print(f"[DATA CLEAN] Valid samples: {valid_mask.sum()}/{len(df)}")
    return df[valid_mask]
```

**Update data loading in cli.py or pipelines**:
```python
# Before:
train_df = pd.read_parquet(cfg.train_path)

# After:
from moola.data.load import validate_expansions
train_df = pd.read_parquet(cfg.train_path)
train_df = validate_expansions(train_df)  # Clean data
```

**Expected Impact**:
- Clean dataset: 97 samples (53 consolidation, 44 retracement)
- Baseline accuracy shifts to 54.6% (from 56.5%)
- Multi-task learning should start working
- Pointer heads receive valid, consistent targets

---

### Fix 2: Adjust Loss Weights (Do After Fix 1)

**Current weights**: α=0.5 (classification), β=0.25 (each pointer)

**Issue**: With corrupted data, pointer losses dominate and mislead training

**Proposed adjustment**:
```python
# File: src/moola/models/cnn_transformer.py (line ~485)

# Option A: Reduce pointer weight initially
self.alpha = 0.7  # Increase classification weight
self.beta = 0.15  # Reduce pointer weight

# Option B: Progressive weight scheduling
def get_loss_weights(epoch):
    """Gradually increase pointer task importance."""
    if epoch < 10:
        return 0.9, 0.05  # Focus on classification early
    elif epoch < 30:
        return 0.7, 0.15  # Balance tasks
    else:
        return 0.5, 0.25  # Full multi-task learning
```

**Rationale**: Even with clean data, pointer prediction is harder than classification. Start with easier task, then introduce multi-task complexity.

---

### Fix 3: Add Pointer Loss Monitoring (Debugging)

**Action**: Log individual loss components to understand training dynamics

**Implementation**:
```python
# In cnn_transformer.py train() method, around line 550-570

# Add after computing total loss:
if batch_idx % 10 == 0:  # Log every 10 batches
    print(f"[LOSS] Batch {batch_idx} | "
          f"cls={cls_loss:.4f} | "
          f"ptr_start={pointer_start_loss:.4f} | "
          f"ptr_end={pointer_end_loss:.4f} | "
          f"total={total_loss:.4f}")
```

**Purpose**: Identify which task is causing gradient instability

---

### Fix 4: Reduce Model Complexity (If Still Stuck)

**Current architecture**:
- 4 transformer layers
- 8 attention heads
- 256 embedding dimension
- ~2.1M parameters

**Issue**: 115 samples → 97 after cleaning = **21,649 parameters per sample!**

**Proposed reduction**:
```python
# File: src/moola/models/cnn_transformer.py

# Reduce to minimal viable architecture:
self.embedding_dim = 128  # Was 256
self.num_transformer_layers = 2  # Was 4
self.num_heads = 4  # Was 8

# This reduces parameters from 2.1M to ~500K
# New ratio: ~5,154 parameters per sample (still high but better)
```

**Or**: Add stronger regularization
```python
# In TransformerBlock:
self.dropout = nn.Dropout(0.3)  # Increase from 0.1
```

---

### Fix 5: Verify Masking is Active (Quick Check)

**Action**: Confirm attention mask is working as expected

**Current debug log** (line 405-409):
```python
if not hasattr(self, '_mask_verified'):
    blocked_positions = (attention_mask == float('-inf')).sum().item()
    print(f"[MASK] Attention mask applied | shape={attention_mask.shape} | blocked={blocked_positions}/11025")
    self._mask_verified = True
```

**Expected output**: `blocked=2700/11025` (verified from previous logs)

**If not appearing**: Masking may not be applied, breaking multi-task learning

---

### Fix 6: Investigate Feature Quality (After Fixes 1-3)

**If model still stuck after data cleaning**:

**Hypothesis**: HopSketch features may not be discriminative for this task

**Quick test**:
```python
# Add feature visualization before training
from sklearn.decomposition import PCA

X_2d = PCA(n_components=2).fit_transform(X_train.reshape(len(X_train), -1))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train)
plt.title("Feature separability")
plt.savefig("feature_viz.png")
```

**If features are not separable**:
- Features may be too low-level (geometric only)
- Need higher-level features (trend, momentum, volume)
- Consider alternative feature extraction

---

## Implementation Priority

### Phase 1: Data Cleaning (Immediate)
1. ✅ Create `validate_expansions()` function
2. ✅ Integrate into data loading pipeline
3. ✅ Re-run OOF evaluation with clean data
4. ✅ Compare accuracy: expect >56.5% if learning works

### Phase 2: Loss Monitoring (If Phase 1 Helps)
1. Add loss component logging
2. Run single fold to observe loss dynamics
3. Identify if pointer losses are converging

### Phase 3: Architecture Tuning (If Still Stuck)
1. Reduce model complexity (128 dim, 2 layers)
2. Increase regularization (dropout 0.3)
3. Adjust loss weights (0.7/0.15)

### Phase 4: Feature Investigation (Last Resort)
1. Visualize feature separability
2. Test alternative features (technical indicators)
3. Consider ensemble with XGBoost only

---

## Success Metrics

### After Fix 1 (Data Cleaning):
- ✅ **Prediction diversity**: Std dev > 0.1 (currently 0.0024)
- ✅ **Learning signal**: Accuracy varies across folds (not all exactly 56.5%)
- ✅ **Class balance**: Some predictions favor class 1 (currently 0/115)
- ✅ **Unique predictions**: More than 4 distinct values (currently 4)

### After Fix 2 (Loss Tuning):
- ✅ **Pointer convergence**: Pointer losses decrease during training
- ✅ **No gradient explosion**: Loss remains stable (not NaN/Inf)
- ✅ **Improved accuracy**: >54.6% new baseline (97 samples, 53:44 ratio)

### Stretch Goal:
- ✅ **Ensemble improvement**: CNN-Transformer + XGBoost > XGBoost alone
- ✅ **Multi-task signal**: Pointer predictions correlate with classification
- ✅ **Target accuracy**: 60-63% (per project goals)

---

## Next Steps

**Immediate Action (Start Here)**:
1. Implement `validate_expansions()` function
2. Update data loading in `cli.py` or training pipeline
3. Re-run OOF with clean data: `python3 -m moola.cli oof --model cnn_transformer`
4. Check if predictions show diversity (std > 0.1, not all class 0)

**If successful**:
5. Add loss monitoring to understand training dynamics
6. Fine-tune loss weights if needed
7. Proceed with ensemble stacking

**If still stuck**:
8. Reduce model complexity
9. Investigate feature quality
10. Consider alternative approaches

---

## Code Changes Required

### File 1: `src/moola/data/load.py` (NEW FILE)
```python
"""Data loading utilities with validation."""
import pandas as pd

def validate_expansions(df):
    """Remove samples with invalid expansion indices.

    Args:
        df: DataFrame with 'expansion_start' and 'expansion_end' columns

    Returns:
        Cleaned DataFrame with valid expansion indices
    """
    start = df['expansion_start'].values
    end = df['expansion_end'].values

    valid_mask = (
        (start < end) &  # No zero-length expansions
        (start >= 30) & (start <= 74) &  # Valid start range [30, 74]
        (end >= 30) & (end <= 74)  # Valid end range [30, 74]
    )

    n_removed = (~valid_mask).sum()
    print(f"[DATA CLEAN] Removed {n_removed} samples with invalid expansions")
    print(f"[DATA CLEAN] Clean dataset: {valid_mask.sum()}/{len(df)} samples")

    return df[valid_mask].reset_index(drop=True)
```

### File 2: `src/moola/cli.py` (MODIFY)
```python
# Around line 420-430 in oof() function

# Before:
train_df = pd.read_parquet(cfg.train_path)

# After:
from moola.data.load import validate_expansions
train_df = pd.read_parquet(cfg.train_path)
train_df = validate_expansions(train_df)  # Remove problematic samples
```

### File 3: `src/moola/models/cnn_transformer.py` (OPTIONAL - for monitoring)
```python
# Around line 560 in train() method

# Add loss component logging:
if batch_idx % 10 == 0:
    print(f"[LOSS] Batch {batch_idx:3d} | "
          f"cls={cls_loss:.4f} | "
          f"ptr_start={pointer_start_loss:.4f} | "
          f"ptr_end={pointer_end_loss:.4f} | "
          f"total={total_loss:.4f}")
```

---

## Conclusion

The CNN-Transformer mode collapse is caused by **15.7% corrupted training data** creating conflicting gradient signals in multi-task learning. The model has learned to output constant predictions (~57% class 0) to minimize these contradictory losses.

**The fix is straightforward**: Remove the 18 problematic samples and retrain. This should restore learning and enable the multi-task approach to work as designed.

**Expected timeline**:
- Fix 1 implementation: 5 minutes
- OOF re-run: ~7 minutes
- Results verification: 2 minutes
- **Total: ~15 minutes to validate the fix**

If data cleaning works, proceed with loss tuning and ensemble stacking. If still stuck, escalate to architecture simplification and feature investigation.
