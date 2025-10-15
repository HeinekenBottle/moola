# FixMatch Pipeline Status Report

## Current Situation

### Phase A: Optimized Baseline (FAILED ⚠️)
- **Expected**: 62.5%+ (quality gate)
- **Actual**: 56.52% (±0.00%)
- **Status**: REGRESSION from previous 60.9% baseline

### Critical Issues Discovered

#### 1. Model Performance Degradation
```
ALL FOLDS IDENTICAL: 56.52%
- Fold 1: 56.52%
- Fold 2: 56.52%
- Fold 3: 56.52%
- Fold 4: 56.52%
- Fold 5: 56.52%
- Variance: 0.0000 (🚨 SUSPICIOUS)
```

**Root Cause**: Model is predicting essentially one class constantly. This indicates:
- Severe overfitting or
- Training instability with current settings or
- Data loading issue

#### 2. GPU Underutilization
```
Current State:
- GPU: 0.02-0.03 GB / 24 GB (0.1% utilization)
- Batch size: 512
- Training samples: 115
- Batches per epoch: 115/512 = 0.2 batches

Problem: With only 115 training samples, batch_size=512 means less than 1 batch per epoch!
```

**Solution Needed**: batch_size=512 is TOO LARGE for 115 samples. Should use 16-32.

#### 3. FixMatch Pseudo-Label Generation (BLOCKED)
```
Pseudo-label candidates:
- Consolidation (threshold 0.92): 0 candidates
- Retracement (threshold 0.85): 0 candidates
- Total pseudo-labels: 0

Reason: Teacher model has ~56% accuracy and low confidence
```

**Cannot proceed with Phase B until Phase A is fixed.**

---

## Diagnosis

### Why Did Performance Drop from 60.9% to 56.5%?

**Hypothesis 1: Batch Size Too Large**
- Previous baseline (60.9%): likely used batch_size=32
- Current (56.5%): batch_size=512
- With only 115 samples:
  - batch_size=32 → ~4 batches per epoch (reasonable gradient updates)
  - batch_size=512 → 0.2 batches per epoch (almost no learning)

**Hypothesis 2: Mixup/CutMix Interaction**
- Mixup already in baseline (mixup_alpha=0.2)
- Increased to mixup_alpha=0.3 in optimized version
- May be too aggressive for 115-sample dataset

**Hypothesis 3: FP16 Numerical Instability**
- Mixed precision enabled (use_amp=True)
- Small dataset may amplify FP16 issues
- Binary cross-entropy loss may underflow

---

## Recommended Fixes

### Immediate Action: Fix Phase A Baseline

**Option 1: Revert to Proven Settings** (SAFEST)
```python
model = CnnTransformerModel(
    seed=seed,
    device='cuda',
    n_epochs=60,
    batch_size=32,         # REVERT from 512
    num_workers=4,         # REVERT from 16 (overkill for 115 samples)
    mixup_alpha=0.2,       # REVERT from 0.3
    early_stopping_patience=20,
    use_amp=False,         # DISABLE FP16 for stability
)
```

**Option 2: Optimize for Small Dataset** (EXPERIMENTAL)
```python
model = CnnTransformerModel(
    seed=seed,
    device='cuda',
    n_epochs=100,          # More epochs since batches are smaller
    batch_size=16,         # Smaller batch for 115 samples
    num_workers=2,         # Lower workers for small dataset
    mixup_alpha=0.1,       # Gentle mixup
    early_stopping_patience=30,
    use_amp=False,         # Disable for stability
)
```

### Phase B Adjustments

**If Phase A succeeds (>62.5%):**
1. Lower pseudo-label thresholds dramatically:
   - `tau_consolidation`: 0.92 → **0.65**
   - `tau_retracement`: 0.85 → **0.60**
2. Accept lower-confidence pseudo-labels initially
3. Gradually increase thresholds if distribution skews

**Target**: 50-100 pseudo-labels per class (reduced from 150-200)

---

## GPU Utilization Strategy

### Reality Check: Small Dataset Problem

**Current Dataset**:
- Labeled: 115 samples
- Unlabeled: 11,873 samples

**Proper GPU Utilization**:
| Stage | Samples | Batch Size | Workers | GPU Util |
|-------|---------|------------|---------|----------|
| Teacher Training | 115 | 16-32 | 2-4 | 10-20% (expected!) |
| Pseudo-label Gen | 11,873 | 512 | 16 | 80-95% ✅ |
| Student Training | 115+200 | 64 | 8 | 30-50% |

**Key Insight**: GPU will be underutilized during labeled training (115 samples). That's OKAY.
Focus GPU optimization on:
1. Pseudo-label generation (11K samples)
2. Student training with combined data (300+ samples)

---

## Next Steps

### Step 1: Reproduce Previous Baseline (URGENT)
```bash
# Run with original proven settings
python3 oof_baseline.py \
  --batch-size 32 \
  --num-workers 4 \
  --mixup-alpha 0.2 \
  --no-amp  # Disable FP16
```

**Expected**: 60-61% accuracy (matching previous baseline)

### Step 2: Debug Performance Regression
If Step 1 fails:
1. Check data loading (verify train.parquet hasn't changed)
2. Verify model architecture (no unintended changes)
3. Test without Mixup (set mixup_alpha=0.0)
4. Compare with previous successful run's hyperparameters

### Step 3: Proceed with FixMatch (Once Phase A > 62.5%)
```bash
python3 -m moola.pipelines.fixmatch \
  --tau-consolidation 0.65 \  # LOWERED
  --tau-retracement 0.60 \     # LOWERED
  --target-per-class 75 \      # HALVED
  --max-per-class 100          # HALVED
```

---

## Timeline Adjustment

**Original Estimate**: 4-5 hours
**Current Status**: Phase A blocked (1 hour spent)

**Revised Timeline**:
- Debug & Fix Phase A: 1-2 hours
- Run corrected baseline: 30 min
- Phase B (if Phase A succeeds): 3-4 hours
- **Total**: 5-7 hours

**GPU Cost**: ~$15-20 (increased due to debugging)

---

## Quality Gates

### Phase A (Baseline OOF)
- ✅ Accuracy > 62.5%
- ✅ Fold variance < 8%
- ❌ **Current**: 56.5% ± 0.0% (FAILED BOTH)

### Phase B (FixMatch)
- ✅ Pseudo-labels: 50-100 per class (revised down)
- ✅ Distribution: 40-60% each class (wider tolerance)
- ✅ Self-consistency > 75%
- ✅ Final OOF > 66%

---

## Files Status

### Created
- ✅ `src/moola/pipelines/fixmatch.py` (480 lines)
- ✅ `/tmp/oof_optimized.py` (baseline with optimized settings)
- ✅ `/tmp/run_fixmatch_pipeline.sh` (full pipeline runner)

### Uploaded to RunPod
- ✅ All files uploaded
- ⚠️ Results: baseline FAILED (56.5%)

### Results
- ⚠️ `/workspace/logs/oof_optimized_results.txt` (56.5%)
- ❌ FixMatch blocked (0 pseudo-labels generated)
