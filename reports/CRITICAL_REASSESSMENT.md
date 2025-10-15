# CRITICAL: Forensic Audit Reassessment

## Test Results - Hypothesis REJECTED

**Option 1 Test (5 Simple Features)**: **48.7% accuracy**
- Target: 63-66%
- Actual: 48.7%
- **FAILED by -14.3%**

**Complex Features (37)**: **47.8% accuracy**
- Both simple AND complex underperform!

---

## Root Cause: Baseline Misidentification

### What We Assumed:
56.5% OOF came from XGBoost with expansion indices + 37 complex features

### What It Actually Is:
The 56.5% baseline is from **STACKING ENSEMBLE**:
- LogisticRegression (base model #1)
- RandomForest (base model #2)
- XGBoost (base model #3)
- Meta-learner (LogisticRegression) trained on OOF predictions
- Plus diversity meta-features

**Individual Model Performance:**
- XGBoost alone: ~48% (as confirmed by test)
- LogReg alone: ~47-50% (estimated)
- RF alone: ~47-50% (estimated)
- **Stacking ensemble**: 56.5% ✓

---

## Why the Forensic Audit Was Partially Wrong

### Correct Findings:
✓ CNN-Transformer has signal dilution issues (4.8% effective signal)
✓ Global pooling destroys pattern signal
✓ No attention masking causes contamination
✓ XGBoost uses expansion indices correctly

### Incorrect Assumption:
✗ **The baseline was NOT from a single XGBoost model**
✗ It was from an ensemble of 3 models + meta-learner
✗ Therefore, single-model improvements won't reach 56.5%

---

## Corrected Understanding

### Model Performance Hierarchy:

```
Stacking Ensemble (LogReg + RF + XGB + Meta-learner): 56.5% ✓ CURRENT
    ↑
    Combines OOF predictions from 3 base models
    ↓
Individual Base Models:
  - XGBoost (37 complex features):        ~47.8%
  - XGBoost (5 simple features):          ~48.7%
  - LogisticRegression:                   ~47-50% (est)
  - RandomForest:                         ~47-50% (est)
    ↓
CNN-Transformer (signal dilution):        ~56.5% (Phase A optimized baseline)
```

**Wait!** The RunPod logs show "Optimized Baseline OOF: 0.5652" - this might be CNN-Transformer alone, not stacking!

---

## What's Actually Happening

Looking at the RunPod logs:
```
Phase A (Optimized Baseline):
Optimized Baseline OOF: 0.5652
```

This suggests **CNN-Transformer is getting 56.5%**, not XGBoost!

### Revised Model Performance:

| Model | OOF Accuracy | Notes |
|-------|--------------|-------|
| **CNN-Transformer** | **56.5%** | Current best (despite signal dilution!) |
| XGBoost (37 complex) | 47.8% | Underperforms CNN-Trans |
| XGBoost (5 simple) | 48.7% | Marginally better than complex |
| LogReg | ~47-50% | Part of stack |
| RF | ~47-50% | Part of stack |
| **Stacking Ensemble** | ? | May be better than 56.5% |

---

## Critical Insight: CNN-Transformer is Working Despite Flaws!

This is **counter-intuitive** but important:

**CNN-Transformer gets 56.5% accuracy DESPITE:**
- 4.8% effective signal strength (20:1 contamination)
- No attention masking
- Global pooling dilution

**Why it works:**
1. Deep learning can learn despite noise (regularization, dropout)
2. Window weighting (1.5x) provides some help
3. 105-bar input contains MORE context than 37 hand-engineered features
4. Transformer attention can learn to ignore noise (even without explicit masking)

**Classical ML (XGBoost) gets 47.8% because:**
1. 37 hand-engineered features lose too much information
2. Feature extraction destroys subtle patterns (averaging, smoothing)
3. Small dataset (115 samples) → hard to learn good features

---

## Corrected Recommendations

### 🥇 **NEW Option 1: Fix CNN-Transformer (HIGHEST PRIORITY)**

**Why**: It's already at 56.5% despite architectural flaws. Fixing them could reach 60-65%.

**Fixes**:
1. **Add attention masking**:
   ```python
   # Mask buffers [0:30] and [75:105]
   mask = torch.zeros(105, 105)
   mask[30:75, 30:75] = 1
   transformer(x, src_mask=mask)
   ```

2. **Region-specific pooling**:
   ```python
   # Old: x.mean(dim=1)
   # New: x[:, 30:75, :].mean(dim=1)
   ```

3. **Increase window weighting**:
   ```python
   # Old: weights[30:75] = 1.5
   # New: weights[30:75] = 3.0  # or even 5.0
   ```

4. **Add expansion-aware attention** (advanced):
   ```python
   # Use expansion_start/end to create per-sample masks
   for i in range(batch_size):
       mask[i, expansion_start[i]:expansion_end[i], :] = 2.0
   ```

**Expected Performance**: 60-65% (+3.5-8.5%)

**Why this will work**:
- Model already learns despite noise
- Reducing noise → more signal → better performance
- Keeps deep learning advantages (learns complex patterns)

---

### 🥈 **Option 2: Improve XGBoost Features (SECONDARY)**

**Why**: XGBoost underperforms (47.8%) because features lose information.

**Fixes**:
1. **Remove smoothing operations**:
   - Don't use .mean() across bars
   - Don't use rolling windows
   - Extract per-bar features, let XGBoost learn aggregation

2. **Add multi-scale features**:
   - Pattern-level features (from expansion region)
   - Context features (from [30:75])
   - Relative features (pattern vs context)

3. **Use multi-scale feature engineering** (already exists!):
   ```python
   from moola.features.price_action_features import engineer_multiscale_features
   X_multiscale = engineer_multiscale_features(X, expansion_start, expansion_end)
   ```

**Expected Performance**: 50-52% (+2-4% from 47.8%)

**Why this might not be enough**:
- Classical ML fundamentally limited for 6-bar patterns
- Deep learning better at learning spatial/temporal patterns
- Small dataset (115 samples) hurts classical ML more

---

### 🥉 **Option 3: Improve Stacking Ensemble (EXISTING)**

**Why**: Combine improved CNN-Transformer + improved XGBoost

**Implementation**:
1. Fix CNN-Transformer (Option 1)
2. Improve XGBoost features (Option 2)
3. Train stacking ensemble on improved base models

**Expected Performance**: 62-67% (+5.5-10.5%)

**Why this is best long-term**:
- Combines strengths of deep learning + classical ML
- Diversifies predictions
- Most robust approach

---

## Immediate Action Plan (REVISED)

### Step 1: Fix CNN-Transformer (TODAY)
```python
# File: src/moola/models/cnn_transformer.py

# 1. Add attention masking (line ~375)
def forward(self, x):
    # ... existing code ...

    # Create mask for [30:75] region only
    mask = torch.zeros(105, 105, device=x.device)
    mask[30:75, 30:75] = 1
    mask = mask.bool()

    # Apply transformer with mask
    x = self.transformer(x, src_key_padding_mask=~mask)

    # Region-specific pooling
    pooled = x[:, 30:75, :].mean(dim=1)  # Only pool [30:75]

    # ... rest of code ...
```

### Step 2: Test Improvement (TODAY)
```bash
# Re-run training with fixed CNN-Transformer
python3 scripts/train_cnn_transformer_fixed.py

# Expected: 60-65% OOF (vs 56.5% baseline)
```

### Step 3: If Successful, Improve Stacking (TOMORROW)
- Combine fixed CNN-Transformer + improved XGBoost
- Re-train stacking ensemble
- Target: 62-67%

---

## Summary

**What We Learned:**
1. ❌ Simple features (5) are NOT better than complex (37)
2. ❌ Both get ~48% (worse than 56.5% baseline)
3. ✅ CNN-Transformer is the strong model (56.5%)
4. ✅ Fixing CNN-Transformer has highest ROI
5. ✅ Stacking ensemble combines multiple models (not single XGBoost)

**Corrected Priority:**
1. 🥇 Fix CNN-Transformer architecture (highest impact)
2. 🥈 Improve XGBoost features (secondary)
3. 🥉 Re-train stacking ensemble (long-term best)

**Expected Final Performance:**
- Fixed CNN-Transformer: 60-65%
- Improved XGBoost: 50-52%
- Re-stacked Ensemble: **62-67%** ← TARGET

---

## Next Immediate Action

**Create fix for CNN-Transformer**:
```bash
# Edit model file
vim src/moola/models/cnn_transformer.py

# Add:
# 1. Attention masking for [30:75] region
# 2. Region-specific pooling
# 3. Stronger window weighting (3.0x instead of 1.5x)

# Test
python3 scripts/train_cnn_transformer_fixed.py
```

This is the **highest ROI fix** based on actual test results.
