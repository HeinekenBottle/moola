# Baseline 100-Epoch Training Results

**Date**: 2025-10-26
**Run ID**: `baseline_100ep`
**Dataset**: 210 labeled expansion windows (174 base + 36 overlaps)
**Duration**: ~20 minutes on RTX 4090

---

## Executive Summary

✅ **Training completed successfully** with comprehensive metric logging
❌ **Span detection FAILED** - Model learned to predict no expansions (F1 = 0.0000)
⚠️ **Root cause identified**: Countdown loss scale mismatch dominates training

### Key Metrics

| Metric | Initial (Epoch 1) | Final (Epoch 100) | Change |
|--------|-------------------|-------------------|--------|
| **Validation Loss** | 17.79 | 11.05 | ↓ **37.9%** ✅ |
| **Span F1** | 0.0000 | 0.0000 | No change ❌ |
| **Span Precision** | 0.0000 | 0.0000 | No change ❌ |
| **Span Recall** | 0.0000 | 0.0000 | No change ❌ |
| **Probability Separation** | -0.0008 | +0.0036 | Minimal improvement ⚠️ |

**Conclusion**: Model is learning (loss improves), but **not the right task** (span detection fails).

---

## What Went Wrong: Technical Diagnosis

### 1. **Countdown Loss Dominates Training**

Despite uncertainty weighting giving countdown only **4.3% task weight**, it accounts for **91.2% of total loss**:

```
Final Loss Breakdown (Epoch 100, Validation):
  Classification:  0.68 (6.2%)
  Pointers:        0.01 (0.1%)
  Span:            0.28 (2.5%)
  Countdown:      10.08 (91.2%)  ← PROBLEM
  -------------------------
  Total:          11.05
```

### 2. **Countdown Loss Scale Mismatch**

Countdown is computed as: `countdown[i] = -(i - expansion_start)`, clipped to [-20, 20]

**Example**: For expansion_start=50 in a 105-bar window:
- Bar 0: countdown = -(0 - 50) = 50 → clipped to 20
- Bar 50: countdown = 0 (expansion starts)
- Bar 104: countdown = -(104 - 50) = -54 → clipped to -20

**Huber loss (δ=1.0) on [-20, 20] range produces HUGE loss values**:
- Countdown loss: 16.1 → 10.08 (still 10x larger than other losses)
- Other losses: 0.01 - 0.68 (reasonable scale)

**Why uncertainty weighting can't compensate**:
- σ_countdown = 1.47 (high uncertainty) reduces task weight to 4.3%
- But absolute loss magnitude (10.08) still dominates gradient flow
- Model focuses on minimizing countdown error at expense of span detection

### 3. **Probability Collapse**

Model learned the "safe" strategy: **predict low probabilities everywhere**

| Metric | Epoch 1 | Epoch 100 | Interpretation |
|--------|---------|-----------|----------------|
| In-span mean | 0.423 | 0.065 | Collapsed ↓ |
| Out-span mean | 0.424 | 0.061 | Collapsed ↓ |
| Separation | -0.001 | 0.004 | Near-zero (random) |

**Why this happens**:
1. Countdown loss is overwhelming → model can't focus on span detection
2. Class imbalance (expansions are minority) → predicting "no expansion" is safe
3. Soft span loss can't learn when probabilities collapse to ~0.06

---

## What Went Right: Positive Findings

### ✅ 1. **Training Infrastructure Works**

- All 6 CSV metrics files generated correctly
- 11 checkpoint files saved (every 10 epochs + best model)
- Comprehensive logging captured all intended metrics
- No NaN losses, no crashes, no GPU OOM

### ✅ 2. **Gradient Health: Excellent**

```
Total Gradient Norm Evolution:
  Epoch  10: 15.28
  Epoch  50: 29.96
  Epoch 100:  5.99
```

- No vanishing gradients (norm > 0.001)
- No exploding gradients (norm < 100)
- Healthy gradient flow throughout training

### ✅ 3. **Feature Stability**

All 12 features remain stable throughout training (no drift):
- expansion_proxy: mean=0.0151, std=0.6339 (constant across 100 epochs)
- No feature saturation or collapse
- Input distribution is healthy

### ✅ 4. **Uncertainty Weighting Adapts**

Task weights evolved as model learned:

| Task | Initial Weight | Final Weight | σ (Final) |
|------|----------------|--------------|-----------|
| Pointers | 35.7% | **56.1%** | 0.41 (confident) |
| Span | 23.4% | **26.0%** | 0.60 (moderate) |
| Classification | 26.8% | **13.7%** | 0.83 (uncertain) |
| Countdown | 14.1% | **4.3%** | 1.47 (very uncertain) |

**Interpretation**:
- Uncertainty weighting is WORKING (countdown gets downweighted)
- But absolute loss magnitude still overwhelms the weighting

---

## Loss Evolution Analysis

### Validation Loss Components (5 Key Epochs)

| Epoch | Type | Pointer | Span | Countdown | Total |
|-------|------|---------|------|-----------|-------|
| 1 | 1.08 | 0.025 | 0.57 | **16.12** | 17.79 |
| 10 | 0.69 | 0.011 | 0.25 | **15.25** | 16.20 |
| 20 | 0.70 | 0.010 | 0.27 | **12.88** | 13.86 |
| 50 | 0.74 | 0.011 | 0.29 | **11.00** | 12.04 |
| 100 | 0.68 | 0.009 | 0.28 | **10.08** | 11.05 |

**Observations**:
1. **All losses improve** (classification, pointers, span, countdown)
2. **Countdown improves slowest** (16.12 → 10.08 = 37% vs. span 57% → 0.28 = 51%)
3. **Countdown absolute magnitude prevents other tasks from learning**

---

## Recommendations for Next Steps

### Option 1: **Remove Countdown Task Entirely** (Fastest)

**Why**: Countdown is not helping span detection and actively hurting it.

**Action**:
```python
# Modify training script to exclude countdown
# Keep only: classification + pointers + span
loss = (
    (1.0 / (2 * sigma_type**2)) * loss_type + torch.log(sigma_type)
    + (1.0 / (2 * sigma_ptr**2)) * loss_ptr + torch.log(sigma_ptr)
    + (1.0 / (2 * sigma_span**2)) * loss_span + torch.log(sigma_span)
)
```

**Expected outcome**:
- Span loss can improve without countdown interference
- F1 > 0 (model will predict some expansions)
- Probability separation > 0.05 (viable for thresholding)

---

### Option 2: **Normalize Countdown to [0, 1] Range** (Medium complexity)

**Why**: Reduce countdown loss magnitude to match other losses.

**Action**:
```python
def create_expansion_labels(expansion_start, expansion_end, window_length=105):
    countdown = np.arange(window_length, dtype=np.float32) - expansion_start
    countdown = -countdown
    # CHANGE: Normalize to [0, 1] instead of clipping to [-20, 20]
    countdown = (countdown + 20) / 40.0  # Map [-20, 20] → [0, 1]
    countdown = np.clip(countdown, 0, 1)
    return binary_mask, countdown
```

**Expected outcome**:
- Countdown loss magnitude: 10.08 → ~0.5 (comparable to other losses)
- Uncertainty weighting can balance tasks properly
- Span detection has room to learn

---

### Option 3: **Train Span Loss Only** (Simplest baseline)

**Why**: Isolate span detection to understand its ceiling performance.

**Action**:
```python
# Single-task training: just soft span loss
loss = soft_span_loss(output["expansion_probs"], binary_mask)
```

**Expected outcome**:
- Pure baseline for span detection capability
- No multi-task confusion
- Clear signal if features are sufficient

---

### Option 4: **Keep Everything, Just Re-scale Countdown Delta**

**Why**: Huber loss delta=1.0 may be too small for [-20, 20] range.

**Action**:
```python
# CHANGE: Use larger delta to reduce countdown loss magnitude
loss_countdown = F.huber_loss(
    output["expansion_countdown"],
    countdown,
    delta=10.0  # Was 1.0, increase to 10.0
)
```

**Expected outcome**:
- Countdown loss becomes more forgiving for large errors
- Magnitude reduces to ~2-3 instead of 10.08
- Other tasks can compete

---

## Data Artifacts Generated

All artifacts saved to `artifacts/baseline_100ep/`:

### Model Checkpoints (11 files, 13.2 MB total)
- `best_model.pt` - Best validation loss (epoch 68)
- `checkpoint_epoch_10.pt` through `checkpoint_epoch_100.pt`

### Metrics CSVs (6 files)
1. **epoch_metrics.csv** (7 KB)
   - Per-epoch: train_loss, val_loss, span_f1, precision, recall, epoch_time

2. **loss_components.csv** (21 KB)
   - Per-epoch per-phase: loss_type, loss_ptr, loss_span, loss_countdown, total_loss

3. **uncertainty_params.csv** (3 KB)
   - Every 5 epochs: σ_ptr, σ_type, σ_span, σ_countdown, derived task weights

4. **probability_stats.csv** (14 KB)
   - Per-epoch per-phase: in_span_mean, out_span_mean, separation, std

5. **feature_stats.csv** (31 KB)
   - Every 5 epochs per-feature (12 features): mean, std, min, max, median

6. **gradient_stats.csv** (5 KB)
   - Every 10 epochs: total_grad_norm, per-layer gradient norms

### Metadata
- `metadata.json` - Training config, dataset size, feature names, model params

---

## Success Criteria Check

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Training completes | ✅ | ✅ | **PASS** |
| No crashes/NaN | ✅ | ✅ | **PASS** |
| Val loss improves ≥20% | ≥20% | 37.9% | **PASS** ✅ |
| Span F1 > 0 | >0 | 0.0000 | **FAIL** ❌ |
| Probability separation > 0.05 | >0.05 | 0.0036 | **FAIL** ❌ |

**Overall Assessment**: Infrastructure and training work, but model architecture needs adjustment to enable span detection learning.

---

## Key Insights for "Reverse Engineering and Collaborating" Next Steps

### 1. **Multi-task Learning Works (But Not for This Task Combination)**

Uncertainty weighting successfully adapted task priorities:
- Countdown downweighted from 14% → 4.3% (model detected it's hard)
- Pointers upweighted from 36% → 56% (model found it useful)

**Lesson**: Uncertainty weighting identifies problematic tasks but can't fix scale mismatches.

### 2. **Soft Span Loss Needs Viable Probabilities**

Probability collapse (0.42 → 0.06) prevents soft span loss from learning:
- Soft span loss requires probability separation > 0.05 to be effective
- Collapsed probabilities (~0.06) can't encode expansion boundaries
- Need to remove interference (countdown) to let probabilities rise

### 3. **Features Are Stable (Good Sign)**

No feature drift or collapse during 100 epochs:
- expansion_proxy: mean=0.015 ± 0.634 (constant)
- All 12 features maintain healthy distributions

**Lesson**: Features are not the problem - model architecture is.

### 4. **Countdown as Auxiliary Task: Bad Idea**

Despite intuition that "counting down to expansion" should help:
- Countdown loss is too large (10.08 vs 0.28 for span)
- Model can't balance tasks even with uncertainty weighting
- Simpler is better: remove countdown

---

## Next Experiment Recommendations

**Immediate (Tonight/Tomorrow)**:
1. **Run Option 1**: Remove countdown, keep 3 tasks (type, ptr, span)
2. **Expected runtime**: 20 minutes on RTX 4090
3. **Success criteria**: Span F1 > 0.10, Separation > 0.05

**If Option 1 succeeds** (F1 > 0.10):
- Try adding pre-training (Jade encoder on unlabeled data)
- Expected boost: +3-5% F1

**If Option 1 fails** (F1 still ~0):
- Run Option 3: Span-only training (isolate the problem)
- Investigate features further (are they expressive enough?)

**For "Reverse Engineering"**:
- Use `loss_components.csv` to identify which tasks conflict
- Use `uncertainty_params.csv` to see which tasks model finds easy/hard
- Use `probability_stats.csv` to track when collapse happens

---

**Created**: 2025-10-26
**Training completed**: 2025-10-26 23:10 UTC
**Total training time**: ~20 minutes
**Artifacts location**: `artifacts/baseline_100ep/`
