# Stones Pre-training Reset - Analysis & Next Steps

**Date:** 2025-10-25
**Status:** Pre-training completed, fine-tuning results BELOW TARGET

---

## Executive Summary

Pre-training completed successfully (20 epochs, MAE 0.00998), but fine-tuning results **dramatically underperformed** Stones specifications:

| Metric | Target | Actual | Gap |
|--------|--------|--------|-----|
| F1 Macro | >0.60 | 0.355 | **-41%** |
| Recall Consolidation | >0.40 | 0.212 | **-47%** |
| Recall Retracement | >0.40 | 0.800 | ✓ 2x target |
| ECE | <0.10 | 0.179 | **+79%** |

**Critical Issue:** Severe consolidation class collapse (only 21% recall) suggests **negative transfer** from pre-training.

---

## What Happened

### Pre-Training (Successful)
- **Data:** 1.8M bars of NQ futures (5 years, 2020-09 to 2025-09)
- **Model:** BiLSTM masked autoencoder (128 hidden × 2 directions)
- **Features:** 12D relativity pipeline (6 candle + 4 swing + 2 proxy)
- **Training:** 20 epochs, batch 1024, 10 minutes total
- **Result:** MAE 0.00998, checkpoint saved to `artifacts/jade_pretrain/checkpoint_best.pt`

### Fine-Tuning (Failed Targets)
- **Data:** 174 labeled samples (93 retracement, 81 consolidation)
- **Config:** Frozen encoder, batch 29, PCGrad enabled, 20 epochs max
- **Early Stop:** Epoch 5-6 (patience=5 triggered)
- **Results:**
  - Accuracy: 52.8% (barely better than random)
  - F1 Consolidation: 0.150 (model ignoring class)
  - F1 Retracement: 0.561 (model defaulting to this class)
  - Pointer MAE: 10 bars center, 42.5 bars length
  - Joint success@±3: 7.5% (multi-task failure)

### Technical Issues Identified

1. **Class Collapse**
   - Model learned to predict retracement as safe default (80% recall)
   - Consolidation severely under-predicted (21% recall)
   - Confusion matrix: Nearly all predictions → retracement class

2. **Loss Imbalance**
   - Classification loss: 1.049 (dominant)
   - Pointer loss: 0.017 (nearly silenced)
   - Uncertainty weighting failed to balance tasks

3. **PCGrad Ineffective**
   - 63% gradient conflicts detected
   - Only 3% projection rate (threshold -0.3 too strict)
   - Task conflicts not being resolved

4. **Poor Calibration**
   - ECE 0.179 >> 0.10 target
   - Confidence scores mismatched to actual performance

5. **JSON Export Crash**
   - Training completed all 5 folds successfully
   - Crashed during final JSON export (float32 not CPU-converted)

---

## Root Cause Hypothesis: Negative Transfer

### The Distribution Mismatch Problem

**Unlabeled Data (Pre-training):**
- 1.8M raw bars including ALL market conditions
- Noisy periods, ambiguous patterns, low-volatility zones
- No quality filtering or curation
- MAE objective: Reconstruct ANY pattern, including noise

**Labeled Data (Fine-tuning):**
- 174 carefully curated samples
- Human-annotated with quality grades (A/B/C only, D blacklisted)
- Represents CLEAN, UNAMBIGUOUS patterns only
- 166 D-grade windows explicitly excluded

**Result:** Pre-trained encoder learned to represent **noise** (what's common in raw data), not **signal** (what's discriminative for our task).

### Why Freezing Made It Worse

- Frozen encoder: 256 hidden × 2 directions = **512 params locked**
- Trainable head: Only **16,775 params** to compensate
- Encoder has ~3% of total model capacity, but **controls 100% of feature representation**
- Head cannot override biased features learned during pre-training

---

## Critical Questions to Answer

### Q1: Is pre-training helping or hurting?
**Need:** Baseline training WITHOUT pre-training for direct comparison

**Hypothesis:**
- If baseline F1 > 0.50: Pre-training is **negative transfer**
- If baseline F1 < 0.30: Architecture itself is the issue
- If baseline 0.30-0.50: Pre-training neutral, tuning needed

### Q2: Are features actually discriminative?
**Need:** Feature correlation analysis with labels

**Key Checks:**
- consol_proxy correlation with consolidation class (expect >0.2)
- expansion_proxy correlation with retracement class
- Which features have strongest signal?
- Are swing features behaving correctly?

### Q3: Is the loss configuration optimal?
**Current:** Uncertainty weighting (Kendall et al.)
- σ_ptr and σ_type learned during training
- Should automatically balance tasks

**Problem:** Pointer task silenced (0.017 vs 1.049)
- Uncertainty weights might be collapsing due to small dataset
- Manual λ weights might work better for 174 samples

### Q4: Is PCGrad actually needed?
**Current:** 63% conflicts, 3% projections
- Threshold -0.3 seems too strict
- Might be adding noise without helping

**Alternative:** Simple weighted sum with manual λ values

---

## Action Plan

### Phase 1: Baseline Comparison (HIGHEST PRIORITY)

**Run training WITHOUT pre-training:**
```bash
python3 scripts/finetune_jade.py \
  --data data/processed/labeled/train_latest.parquet \
  --epochs 20 \
  --batch-size 29 \
  --use-pcgrad \
  --device cuda
```

**Expected outcomes:**
1. If F1 > 0.50: **Confirms negative transfer**, abandon current pre-training approach
2. If F1 0.30-0.50: Pre-training neutral, focus on architecture/loss tuning
3. If F1 < 0.30: Deeper architectural issues, need fundamental rethink

**Time:** ~10 minutes on RTX 4090

### Phase 2: Feature Quality Audit

**Correlation Analysis:**
```python
# Check feature-label correlations
# Rebuild 12D features from OHLC for sample of 50-100 windows
# Compute Pearson correlation with consolidation label
# Flag features with |corr| < 0.15 as weak discriminators
```

**SHAP Analysis (post-training):**
```python
# After baseline training completes
# Run SHAP explainer on validation set
# Identify which features actually drive predictions
# Compare to expected: consol_proxy should rank high
```

### Phase 3: Loss Configuration Experiments

**If baseline shows promise (F1 > 0.40):**

1. **Manual λ weights** (instead of uncertainty):
   ```python
   loss = λ_type * loss_type + λ_ptr * (loss_center + loss_length)
   ```
   Try: λ_type=1.0, λ_ptr=0.5, 0.7, 1.0

2. **Drop PCGrad** (if causing issues):
   - Simpler weighted sum might work better for small dataset
   - PCGrad adds overhead and potential instability

3. **Increase patience** to 10-15:
   - Current patience=5 might be stopping too early
   - Small dataset needs more epochs to converge

### Phase 4: Architecture Variations

**If baseline F1 < 0.40:**

1. **Unfreeze encoder gradually:**
   - Epochs 1-10: Freeze encoder
   - Epochs 11-20: Unfreeze last LSTM layer only
   - Epochs 21-30: Unfreeze entire encoder with low LR (1e-5)

2. **Simplify to single-task:**
   - Train classification only (no pointers)
   - Establish ceiling performance without multi-task complexity
   - Add pointers back only if F1 > 0.55

3. **Try different architectures:**
   - Simpler 1-layer LSTM (reduce capacity for small dataset)
   - Add dropout 0.5-0.7 (Stones spec suggests 0.65)
   - Batch normalization between layers

---

## Immediate Next Steps

1. ✅ **Document current state** (this file)
2. ⏳ **Spin up new RunPod instance**
3. ⏳ **Run baseline (no pre-training)** - BLOCKING for all other decisions
4. ⏳ **Feature correlation analysis** - Can run locally while baseline trains
5. ⏳ **Compare baseline vs pre-trained** - Determines entire strategy

**Timeline:**
- Baseline training: 10 minutes
- Analysis: 15 minutes
- Decision point: 30 minutes from now

---

## Code Fixes Needed

### 1. JSON Export Crash
**File:** `scripts/finetune_jade.py` line ~1255

**Issue:** numpy.float32 not JSON serializable

**Fix:**
```python
# Before json.dump(), convert numpy types:
summary = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
           for k, v in summary.items()}
json.dump(summary, f, indent=2)
```

### 2. Patience Too Aggressive
**Current:** `patience=5`
**Fix:** `patience=15` (for <200 sample dataset)

### 3. PCGrad Threshold
**Current:** `threshold=-0.3`
**Fix:** `threshold=-0.1` (activate more aggressively)

---

## Data Quality Notes

**Labeled Data Structure:**
- File: `data/processed/labeled/train_latest.parquet`
- Samples: 174 (93 retracement, 81 consolidation)
- Storage: 105 OHLC arrays per window (raw prices)
- Features: Rebuilt on-the-fly during training using `relativity.py`
- Pointers: expansion_start ∈ [23,85], expansion_end ∈ [31,95]

**Feature Pipeline:**
- Input: Raw OHLC (4D per timestep)
- Output: 12D relativity features per timestep
- Features: open_norm, close_norm, body_pct, upper_wick, lower_wick, range_z, dist_prev_SH, dist_prev_SL, bars_since_SH, bars_since_SL, expansion_proxy, consol_proxy
- Bug fix: Single-timestep overwrite bug FIXED in commit 2d986ef

**Pre-training Data:**
- File: `data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet`
- Bars: 1.8M (5 years)
- Windows: ~34K (stride 52, window 105)
- Features: Same 12D pipeline used in fine-tuning
- Quality: UNFILTERED (includes all market conditions)

---

## Stones Specifications (Reference)

### Non-Negotiable Requirements:
- Batch size: 29 (small dataset optimization)
- Dropout: 0.65 (high regularization)
- Uncertainty weighting: REQUIRED (not manual λ)
- PCGrad: REQUIRED for multi-task conflicts

### Target Metrics:
- F1 Macro: >0.60
- Per-class recall: >0.40 (both classes)
- ECE: <0.10 (calibration)
- Center MAE: <0.02 (normalized)
- Hit@±3: >60%

### Monitoring (every 5 epochs):
- Classification: F1_macro, per-class recall
- Pointers: Center/length MAE, Hit@±3
- Uncertainty: σ_ptr/σ_type ratio ∈ [0.5, 2.0]
- Conflicts: PCGrad activation rate
- Calibration: ECE, AUCPR

---

## Files Changed This Session

1. `artifacts/jade_pretrain/checkpoint_best.pt` - Pre-trained encoder (6.3MB)
2. `artifacts/jade_pretrain/training_results.json` - Pre-training metrics
3. `/workspace/moola/finetune.log` - Fine-tuning output (RunPod, lost after termination)

---

## Decisions Pending

1. **Continue with pre-training approach?**
   - ⏳ WAIT for baseline results
   - If negative transfer: Abandon MAE pre-training, try contrastive learning or supervised pre-training
   - If neutral: Focus on loss/architecture tuning

2. **Unfreeze encoder?**
   - ⏳ WAIT for baseline results
   - If baseline > 0.50: Try gradual unfreezing
   - If baseline < 0.40: Fundamental architecture rethink needed

3. **Switch to single-task?**
   - ⏳ WAIT for baseline results
   - If multi-task causing collapse: Separate classification and pointer models
   - If baseline single-task F1 > 0.60: Add pointers back carefully

---

## Context for Next Session

**What worked:**
- Pre-training pipeline (feature computation, training loop)
- Feature bug fix (timestep_idx hardcoding resolved)
- Fine-tuning infrastructure (CV, metrics, monitoring)

**What didn't work:**
- Frozen encoder transfer learning (negative transfer suspected)
- Uncertainty-weighted multi-task learning (loss imbalance)
- PCGrad conflict resolution (low activation rate)
- Early stopping (patience too aggressive)

**What's unknown:**
- Baseline performance without pre-training
- Feature discriminative power (correlation analysis pending)
- Optimal loss configuration for 174 samples
- Whether 12D features are actually helping vs 4D OHLC

**Next session priorities:**
1. Run baseline comparison
2. Feature correlation analysis
3. Decide on pre-training strategy going forward
4. Fix JSON export crash
5. Tune hyperparameters based on baseline results

---

**Last updated:** 2025-10-25 23:20 UTC
**RunPod instance:** 103.196.86.97:15251 (terminated)
**Commit:** 2d986ef (bug fix), e919b8c (docs)
