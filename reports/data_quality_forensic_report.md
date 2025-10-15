# Data Quality Forensic Report
**Date**: 2025-10-14
**Analysis**: CNN-Transformer stuck at 56.5% baseline accuracy
**Root Cause**: Invalid expansion indices in training data

---

## Executive Summary

**Critical Finding**: 15.7% (18/115) of training samples have invalid or out-of-bounds expansion indices that prevent the CNN-Transformer from learning the pointer prediction task.

### Impact on Model Performance
- **CNN-Transformer**: Stuck at 56.5% (majority class baseline) across all 5 folds
- **XGBoost**: 48.7% average (worse than baseline)
- **Multi-task learning**: Enabled but ineffective due to corrupted pointer targets

---

## Data Quality Issues

### 1. Invalid Expansions (Zero-Length)
**Count**: 9 samples (7.8%)
**Issue**: `expansion_start == expansion_end` creates zero-length regions

**Affected Samples**:
```
Sample   2: start=57, end=57, length=0, label=consolidation
Sample  22: start=49, end=49, length=0, label=consolidation
Sample  44: start=49, end=49, length=0, label=consolidation
Sample  66: start=57, end=57, length=0, label=retracement
Sample  72: start=57, end=57, length=0, label=retracement
Sample  80: start=57, end=57, length=0, label=consolidation
Sample 107: start=57, end=57, length=0, label=consolidation
Sample 112: start=49, end=49, length=0, label=consolidation
Sample 114: start=57, end=57, length=0, label=consolidation
```

**Model Impact**:
- Cross-entropy loss for pointer heads receives same index for start and end
- Model cannot learn to distinguish expansion boundaries
- Gradient signal is corrupted

### 2. Out-of-Bounds Start Positions
**Count**: 3 samples (2.6%)
**Issue**: `expansion_start < 30` or `expansion_start > 74` (valid range: [30, 74])

**Affected Samples**:
```
Sample  40: start=81, end=83, label=retracement  [start > 74]
Sample  83: start=23, end=31, label=consolidation [start < 30]
Sample  85: start=85, end=86, label=retracement  [start > 74]
```

**Model Impact**:
- After clipping (line 527 in cnn_transformer.py), these become boundary cases
- Sample 83: start=23 → clipped to 0 (relative position in inner window)
- Sample 40, 85: start > 74 → clipped to 44 (maximum relative position)
- Creates artificial clustering at boundary positions

### 3. Out-of-Bounds End Positions
**Count**: 8 samples (7.0%)
**Issue**: `expansion_end < 30` or `expansion_end > 74`

**Affected Samples**:
```
Sample  14: start=73, end=82, label=consolidation
Sample  27: start=70, end=75, label=retracement
Sample  40: start=81, end=83, label=retracement
Sample  42: start=70, end=75, label=consolidation
Sample  50: start=70, end=75, label=consolidation
Sample  78: start=70, end=75, label=consolidation
Sample  85: start=85, end=86, label=retracement
Sample 103: start=70, end=75, label=consolidation
```

**Model Impact**:
- End positions clipped to valid range [0, 44]
- 5 samples with end=75 get clipped to end=44 (maximum)
- Creates artificial preference for maximum end position

### 4. Total Problematic Samples
**Unique Count**: 18/115 (15.7%)

**Class Distribution**:
- Problematic: 12 consolidation, 6 retracement
- Clean: 53 consolidation, 44 retracement
- **Impact**: Removing problematic samples reduces class imbalance from 65:50 to 53:44 (54.6% vs 56.5% baseline)

---

## Expansion Statistics

### All Samples (including problematic)
- Total: 115 samples
- Length range: [0, 23] bars
- Mean: 6.09 ± 3.94 bars
- Consolidation: 5.43 bars average
- Retracement: 6.94 bars average

### Valid Samples Only
- Total: 97 samples (84.3%)
- Length range: [1, 17] bars
- Mean: 6.46 ± 3.24 bars
- Consolidation: 5.87 bars average
- Retracement: 7.18 bars average
- **Statistical difference**: Retracement expansions are 1.31 bars longer on average

---

## Why Multi-Task Learning Failed

### Expected Behavior
1. Classification head learns consolidation vs retracement patterns
2. Pointer heads learn to identify expansion start/end positions
3. Shared features from transformer attend to expansion region
4. Multi-task loss provides richer gradient signal

### Actual Behavior
1. **18 samples (15.7%) have corrupted pointer targets**
   - 9 samples: pointer_start == pointer_end (no valid target)
   - 11 samples: positions outside valid range get clipped

2. **Pointer loss is noisy and misleading**
   - Zero-length expansions provide contradictory gradients
   - Clipped positions create artificial boundary bias
   - Model cannot distinguish signal from noise

3. **Classification task dominates**
   - Loss weights: α=0.5 (classification), β=0.25 each (pointers)
   - With 15.7% corrupted pointer targets, pointer loss is unreliable
   - Model defaults to classification-only learning
   - Classification achieves 56.5% (majority class baseline)

4. **No attention guidance from pointer task**
   - Pointer heads should guide attention to expansion regions
   - With corrupted targets, attention remains unfocused
   - Transformer fails to learn discriminative features

---

## Recommended Fixes

### Option 1: Clean Data (Recommended)
**Action**: Remove 18 problematic samples from training data

**Implementation**:
```python
# In data preprocessing pipeline
valid_mask = (
    (expansion_start < expansion_end) &  # No zero-length
    (expansion_start >= 30) & (expansion_start <= 74) &  # Valid start
    (expansion_end >= 30) & (expansion_end <= 74)  # Valid end
)
df_clean = df[valid_mask]
```

**Pros**:
- Eliminates all corrupted pointer targets
- Provides clean gradient signal
- Reduces class imbalance slightly (53:44 vs 65:50)

**Cons**:
- Reduces dataset size from 115 to 97 samples (15.7% reduction)
- May impact cross-validation fold balance

**Expected Impact**:
- Multi-task learning should start working
- Pointer heads can learn meaningful patterns
- Classification accuracy should improve beyond 56.5% baseline

### Option 2: Clip and Validate
**Action**: Keep all samples but clip positions and add minimum length constraint

**Implementation**:
```python
expansion_start = np.clip(expansion_start, 30, 74)
expansion_end = np.clip(expansion_end, 30, 74)
# Ensure minimum length of 1
expansion_end = np.maximum(expansion_end, expansion_start + 1)
```

**Pros**:
- Keeps all 115 samples
- Maintains cross-validation fold structure

**Cons**:
- Artificially modifies expansion lengths
- May introduce systematic bias
- Doesn't address root cause

### Option 3: Investigate Data Generation Pipeline
**Action**: Trace back to source of invalid indices

**Investigation Points**:
1. Where do `expansion_start` and `expansion_end` come from?
2. Are these manually labeled or algorithmically detected?
3. Why are some expansions zero-length or out-of-bounds?
4. Should we re-label these samples or exclude them?

**Next Steps**:
- Examine data generation scripts
- Check for labeling errors vs algorithmic bugs
- Validate against source market data

---

## Conclusion

The CNN-Transformer is stuck at 56.5% baseline accuracy because **15.7% of training samples have corrupted expansion indices** that prevent the pointer prediction task from learning meaningful patterns. The multi-task learning approach is sound, but it requires clean, valid pointer targets to function.

**Immediate Action Required**: Clean the training data by removing the 18 problematic samples identified in this report, then re-run OOF evaluation to measure improvement.

**Long-term Action**: Investigate the data generation pipeline to prevent future corruption and potentially re-label the problematic samples if they represent valid market patterns.

---

## Appendix: Problematic Sample Indices

**All 18 problematic samples**: 2, 14, 22, 27, 40, 42, 44, 50, 66, 72, 78, 80, 83, 85, 103, 107, 112, 114
