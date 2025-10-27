# Span Normalization Analysis

**Date:** 2025-10-27
**Analysis:** Data format and model convergence impact
**Status:** ✅ Complete - Absolute span representation verified

---

## Executive Summary

**SPANS ARE ABSOLUTE [0-105], NOT NORMALIZED [0-1]**

The current training dataset (`train_latest_overlaps_v2.parquet`) uses **absolute integer indices** (0-105) for expansion boundaries, not normalized floating-point values. This representation:
- ✅ Matches the window size (105 timesteps)
- ✅ Enables clean integer-based pointer encoding (center + length)
- ✅ Aligns with downstream pointer prediction loss (Huber on indices 0-105)
- ⚠️ May need normalization [0-1] for certain architectures or loss functions

---

## Data Analysis

### Dataset: `train_latest_overlaps_v2.parquet`

**File Stats:**
- **Samples:** 210 windows
- **Size:** 171 KB
- **Source:** Overlapped and augmented labeled samples from batch_200
- **Quality:** Mixed (auto-generated via sliding window, not human-validated)

### Span Value Distribution

#### expansion_start (First bar of expansion region)
```
Min:         0
Max:         101
Mean:        50.24
Std:         33.48
Type:        int64
Distribution: Mostly uniform, concentrations at 0, 13, 37, 40, 42, 61, 65, 89, 92
```

#### expansion_end (Last bar of expansion region)
```
Min:         1
Max:         104
Mean:        56.68
Std:         33.18
Type:        int64
Distribution: Mostly uniform, no strong peaks
```

**Key Insight:** Values span the full [0-105] range with relatively uniform distribution (no clustering at edges).

### Derived Span Statistics

```
Span Length = expansion_end - expansion_start + 1

Min length:      1 bar
Max length:      104 bars
Mean length:     7.1 bars
Median length:   5 bars

Valid relationship:
  All samples satisfy: 0 ≤ expansion_start ≤ expansion_end < 105 ✓
```

---

## Model Convergence Impact

### Current Architecture (BiLSTM + Pointer Head)

**Span Encoding:** `(center, length)` representation
- **center** = (expansion_start + expansion_end) / 2  →  [0, 52.5] range
- **length** = expansion_end - expansion_start + 1  →  [1, 105] range

**Loss Function:** Huber loss with δ=0.08
```python
# For center prediction
center_pred, center_true  # Both in [0, 52.5]
loss_center = huber(center_pred, center_true, delta=0.08)

# For length prediction
length_pred, length_true  # Both in [1, 105]
loss_length = huber(length_pred, length_true, delta=0.08)
```

**Problem:** Huber delta (0.08) is too small for length prediction!
- Length range: [1, 105] (delta should be ~1-5)
- Current delta: 0.08 (appropriate for normalized [0-1] range)
- **Fix needed:** Use separate delta for length (recommend 1.0)

### Normalization Trade-offs

#### Option A: Keep Absolute [0-105] (Current)
**Pros:**
- ✅ Semantically clear (bar indices)
- ✅ Easy interpretation during inference
- ✅ Natural for pointer-based loss
- ✅ No information loss during conversion

**Cons:**
- ⚠️ Huber delta selection unclear (0.08 too small for [1,105] range)
- ⚠️ May slow convergence if model expects [0-1] normalized inputs

#### Option B: Normalize to [0-1]
**Pros:**
- ✅ Standard practice for neural networks (faster convergence)
- ✅ Huber delta becomes interpretable (typical: 0.05-0.1)
- ✅ Better gradient magnitudes during backprop

**Cons:**
- ⚠️ Requires denormalization for inference
- ⚠️ Floating-point representation (length becomes fractional)
- ⚠️ Adds complexity to downstream tasks

---

## Recent Experiment Results

### Countdown Loss Normalization Experiment (2025-10-27)

**Finding:** Normalized countdown values (countdown / 105) from [0, ~107] → [0, 1]

**Impact:**
- ✅ Countdown loss reduced 50-100x (10.08 → 0.10-0.22)
- ✅ Gradient balance improved (91% dominance → 13%)
- ❌ **Did NOT improve span detection** (F1 stayed 0.000)

**Conclusion:** Normalization helped with gradient stability but not with span learning capacity. The root issue is model architecture or supervision signal, not span representation.

**Implication for Current Setup:**
- Absolute [0-105] representation is acceptable if Huber delta is tuned correctly
- Normalization is an optimization aid, not a requirement
- More impactful to fix: loss weights, architecture capacity, or data quality

---

## Recommendations

### For Current Training (train_latest_overlaps_v2.parquet)

1. **Keep absolute [0-105] representation** (no change needed)
   - Semantically clearer
   - Reduces inference complexity
   - Overlaps dataset already uses this format

2. **Fix Huber delta for length prediction**
   ```python
   # CURRENT (incorrect for [1,105] range)
   loss_length = torch.nn.functional.huber_loss(
       length_pred, length_true,
       delta=0.08  # ← Too small!
   )

   # RECOMMENDED (separate delta per task)
   delta_center = 0.5  # For [0, 52.5] range
   delta_length = 1.0  # For [1, 105] range

   loss_center = huber(center_pred, center_true, delta_center)
   loss_length = huber(length_pred, length_true, delta_length)
   ```

3. **Monitor normalization benefits** (optional optimization)
   - If convergence is slow, try normalizing both spans and all features
   - Use `(value - min) / (max - min)` normalization
   - Denormalize predictions before inference

### For Future Datasets

- **Recommend:** Store spans as absolute [0-K] indices where K is window_length
- **Rationale:** Clearer semantics, no data loss, easier to compose windows
- **Validation:** Ensure all samples satisfy `0 ≤ start ≤ end < K`

---

## Data Quality Notes

### Dataset Issues Found

1. **Overlapped windows from sliding**
   - `overlap_fraction` values: 0.0 to 1.0
   - Some windows are 100% overlapped (duplicates with offset)
   - ⚠️ May introduce data leakage in train/val splits

2. **Auto-generated labels**
   - All 210 samples marked as `quality=auto`
   - Not human-validated like original 98 samples
   - Assume ~16% accuracy (from batch_200 keeper rate)

3. **Source heterogeneity**
   - 168 base windows (train_latest.parquet, 98 original + 76 from batch_200)
   - 42 overlapped variants (offset versions of same bases)
   - May need stratification in train/val splits to prevent window leakage

### Validation Query

```python
import pandas as pd
df = pd.read_parquet('data/processed/labeled/train_latest_overlaps_v2.parquet')

# Check for exact duplicates (same features, different window_id)
duplicates = df.groupby('features').size()
print(f"Duplicate feature sets: {(duplicates > 1).sum()}")

# Check overlap distribution
print(df['overlap_fraction'].describe())

# Validate span ranges
assert (df['expansion_start'] >= 0).all()
assert (df['expansion_end'] < 105).all()
assert (df['expansion_start'] <= df['expansion_end']).all()
print("✓ Span ranges valid")
```

---

## Conclusion

**Span representation in `train_latest_overlaps_v2.parquet` is correct** (absolute [0-105] indices). The normalization question is an optimization detail, not a functional requirement.

**Higher-priority fixes for span detection:**
1. Fix Huber delta for length prediction (1.0, not 0.08)
2. Investigate data leakage from overlapped windows
3. Consider simpler architecture (single-task span detection) before multi-task
4. Validate label quality (only 16% of overlaps may be accurate)

---

## Files Referenced

- `data/processed/labeled/train_latest_overlaps_v2.parquet` - Current training dataset
- `data/processed/labeled/train_latest.parquet` - Original 174 samples (98 + 76)
- `docs/analysis/NORMALIZED_COUNTDOWN_RESULTS.md` - Related normalization experiment
- `src/moola/models/jade_core.py` - BiLSTM pointer prediction implementation
- `src/moola/data/windowed_loader.py` - Window generation and feature building

