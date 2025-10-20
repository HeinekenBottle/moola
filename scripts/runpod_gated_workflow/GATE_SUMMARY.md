# Gated Workflow Summary

## Quick Reference

| Gate | Name | Duration | Purpose | Failure Impact |
|------|------|----------|---------|----------------|
| 0 | Environment Verification | <1 min | Validate CUDA, data, splits | ABORT - Environment not ready |
| 1 | Smoke Test (Enhanced) | ~2 min | Baseline metrics (3 epochs) | ABORT - Basic training fails |
| 2 | Control (MiniRocket) | ~3 min | Validate DL > classical | ABORT - Deep learning adds no value |
| 3 | Pretrain BiLSTM | ~20 min | Learn representations | ABORT - Encoder quality < 55% |
| 4 | Finetune Enhanced | ~45 min | Transfer learning (60 epochs) | ABORT - No improvement over baseline |
| 5 | Train with Augmentation | ~45 min | Boost with synthetic data | ABORT - Validation degradation |
| 6 | Baseline SimpleLSTM | ~40 min | Validate pretraining benefit | WARNING - Pretraining ineffective |
| 7 | Ensemble | <1 min | Calibrated model fusion | WARNING - Ensemble < best individual |

**Total Runtime**: ~2.6 hours on RTX 4090

---

## Gate Decision Tree

```
┌─────────────────────┐
│  Gate 0: Verify Env │
└──────┬──────────────┘
       │ PASS
       ▼
┌─────────────────────┐
│  Gate 1: Smoke Test │ ← Establish baseline (3 epochs)
└──────┬──────────────┘
       │ PASS (record baseline F1)
       ▼
┌─────────────────────────┐
│  Gate 2: MiniRocket     │ ← Control test
└──────┬──────────────────┘
       │ PASS (MiniRocket < Enhanced)
       │ FAIL → ABORT: Deep learning not helping
       ▼
┌──────────────────────────┐
│  Gate 3: Pretrain BiLSTM │ ← Learn on unlabeled data
└──────┬───────────────────┘
       │ PASS (linear probe ≥ 55%)
       │ FAIL → ABORT: Encoder too weak
       ▼
┌────────────────────────────┐
│  Gate 4: Finetune Enhanced │ ← Transfer learning (60 epochs)
└──────┬─────────────────────┘
       │ PASS (F1 > Gate 1 baseline)
       │ FAIL → ABORT: No improvement from pretraining
       ▼
┌────────────────────────────┐
│  Gate 5: Train with Aug    │ ← Synthetic data boost
└──────┬─────────────────────┘
       │ PASS (no validation degradation)
       │ FAIL → ABORT: Augmentation hurts performance
       ▼
┌─────────────────────────────┐
│  Gate 6: SimpleLSTM Baseline│ ← Validate pretraining
└──────┬──────────────────────┘
       │ PASS (SimpleLSTM < Enhanced)
       │ WARN → Pretraining not helping
       ▼
┌──────────────────────┐
│  Gate 7: Ensemble    │ ← Final fusion
└──────┬───────────────┘
       │ PASS (ensemble ≈ best individual)
       ▼
┌──────────────────────┐
│  WORKFLOW COMPLETE   │
└──────────────────────┘
```

---

## Critical Gates (Hard Failures)

### Gate 2: Control Test
**Why**: If MiniRocket (classical) beats EnhancedSimpleLSTM (deep learning), something is fundamentally wrong.

**Possible Causes**:
- Pretrained encoder is poor quality
- Deep learning overfitting on small dataset
- Data quality issues
- Incorrect split (data leakage)

**Action**: ABORT workflow, fix root cause

---

### Gate 3: Pretrain Quality
**Why**: Linear probe < 55% means encoder learned nothing useful.

**Possible Causes**:
- Too few unlabeled samples (need >10K)
- Poor masking strategy
- Insufficient pretraining epochs
- Data corruption

**Action**: ABORT workflow, improve pretraining

---

### Gate 4: Transfer Learning
**Why**: If pretraining doesn't improve over baseline, it's worthless.

**Possible Causes**:
- Poor pretrained encoder (Gate 3 passed but barely)
- Finetuning strategy incorrect
- Encoder architecture mismatch
- Learning rate too high/low

**Action**: ABORT workflow, re-inspect Gates 3-4

---

## Warning Gates (Soft Failures)

### Gate 6: Pretraining Validation
**Why**: If SimpleLSTM matches Enhanced, pretraining adds no value.

**Possible Causes**:
- Dataset too small for pretraining to help
- Pretrained encoder overfitted to unlabeled data
- SimpleLSTM architecture better suited for this task

**Action**: Continue workflow, but flag for investigation

---

### Gate 7: Ensemble Performance
**Why**: Ensemble should improve or match best individual.

**Possible Causes**:
- Only one strong model (ensemble degenerates to single model)
- Poor calibration
- Models too correlated

**Action**: Continue workflow, but use best individual instead of ensemble

---

## Metrics to Track

### Per-Gate Metrics

| Gate | Primary Metric | Threshold | Comparison |
|------|----------------|-----------|------------|
| 0 | Environment checks | All pass | N/A |
| 1 | Val F1 (baseline) | Record | N/A |
| 2 | Val F1 (MiniRocket) | Any | Must be < Gate 1 |
| 3 | Linear probe accuracy | ≥ 55% | Absolute threshold |
| 4 | Val F1 (finetuned) | Any | Must be > Gate 1 |
| 5 | Val F1 (augmented) | Any | Must not degrade > 2pp vs Gate 4 |
| 6 | Val F1 (SimpleLSTM) | Any | Should be < Gate 4 |
| 7 | Val F1 (ensemble) | Any | Should be ≥ best individual |

### Cross-Gate Comparisons

```
Gate 1 (baseline)  <  Gate 2 (MiniRocket)  → ABORT
Gate 1 (baseline)  <  Gate 4 (finetuned)   → Required
Gate 4 (finetuned) >  Gate 5 (augmented) + 2pp → ABORT (degradation)
Gate 4 (finetuned) >  Gate 6 (SimpleLSTM) → Expected
Gate 7 (ensemble)  ≈  max(Gate 4, 5, 6)   → Expected
```

---

## Results File Format

**File**: `/workspace/moola/gated_workflow_results.jsonl`

**Format**: JSON Lines (one JSON object per line)

**Example**:
```json
{
  "gate": "1_smoke_enhanced",
  "timestamp": "2025-10-18T12:34:56.000000Z",
  "model": "enhanced_simple_lstm",
  "config": {
    "epochs": 3,
    "pretrained": true,
    "frozen_encoder": true,
    "augmentation": false
  },
  "metrics": {
    "train_acc": 0.850,
    "val_acc": 0.720,
    "val_f1": 0.715
  },
  "train_time_sec": 120.5,
  "status": "passed"
}
```

**Analysis**:
```bash
# Best F1 across all gates
cat gated_workflow_results.jsonl | jq -s 'max_by(.metrics.val_f1 // 0)'

# Compare Gates 1 vs 4 (baseline vs finetuned)
cat gated_workflow_results.jsonl | jq 'select(.gate | startswith("1_") or startswith("4_")) | {gate: .gate, f1: .metrics.val_f1}'

# Check if any gate failed
cat gated_workflow_results.jsonl | jq 'select(.status == "failed")'
```

---

## Common Failure Scenarios

### Scenario 1: Gate 2 Fails (MiniRocket wins)

**Symptom**:
```
Gate 1 Enhanced F1: 0.650
Gate 2 MiniRocket F1: 0.680
✗ GATE FAILED: MiniRocket >= Enhanced
```

**Root Cause**: Pretrained encoder from Gate 3 is poor quality OR Gate 1 used pretrained encoder that doesn't exist yet.

**Fix**:
1. Check if Gate 3 ran successfully
2. Verify encoder quality (linear probe ≥ 55%)
3. If Gate 1 ran before Gate 3, encoder might not exist
4. Run workflow in order: 0 → 1 → 2 → 3 → 4

---

### Scenario 2: Gate 3 Fails (Linear probe < 55%)

**Symptom**:
```
Linear probe CV accuracy: 0.480
✗ GATE FAILED: Linear probe accuracy 0.480 < 55%
```

**Root Cause**: Encoder didn't learn useful representations.

**Fix**:
1. Increase pretraining epochs (50 → 100)
2. Try different masking strategy (patch → random)
3. Check unlabeled data quality
4. Verify sufficient unlabeled samples (need >5K, have 11,873)

---

### Scenario 3: Gate 4 Fails (No improvement)

**Symptom**:
```
Baseline F1: 0.720
Finetuned F1: 0.715
✗ GATE FAILED: No improvement over baseline
```

**Root Cause**: Pretrained encoder not helping, or overfitting during finetuning.

**Fix**:
1. Check Gate 3 linear probe (might be barely passing at 55%)
2. Reduce learning rate (5e-4 → 1e-4)
3. Increase freeze phase (3 → 10 epochs)
4. Try different finetuning strategy

---

### Scenario 4: Gate 5 Fails (Augmentation degrades)

**Symptom**:
```
Finetuned F1: 0.750
Augmented F1: 0.720
Degradation: +0.030
✗ GATE FAILED: Validation performance degraded
```

**Root Cause**: Synthetic data introduces distribution shift.

**Fix**:
1. Increase quality threshold (0.85 → 0.90)
2. Reduce augmentation ratio (2.0 → 1.5)
3. Use safer augmentation strategies only
4. Skip Gate 5 if augmentation not critical

---

## Re-run Strategies

### Full Re-run (from scratch)
```bash
# Delete old results
rm /workspace/moola/gated_workflow_results.jsonl

# Run all gates
python3 scripts/runpod_gated_workflow/run_all.py
```

### Partial Re-run (from specific gate)
```bash
# Keep old results, resume from Gate 4
python3 scripts/runpod_gated_workflow/run_all.py --start-gate 4
```

### Single Gate Re-run (debugging)
```bash
# Run specific gate only
python3 scripts/runpod_gated_workflow/3_pretrain_bilstm.py
```

### Modified Config Re-run
```bash
# Edit gate script to change hyperparameters
vim scripts/runpod_gated_workflow/3_pretrain_bilstm.py
# Change epochs from 50 to 100

# Re-run gate
python3 scripts/runpod_gated_workflow/3_pretrain_bilstm.py
```

---

## Success Criteria

### Minimum Viable Success
- All gates 0-4 pass
- Gate 4 F1 > Gate 1 F1 by at least 0.05 (5pp improvement)
- Gate 3 linear probe ≥ 60% (comfortable margin above 55%)

### Ideal Success
- All gates 0-7 pass
- Gate 4 F1 > Gate 1 F1 by 0.10+ (10pp improvement)
- Gate 6 SimpleLSTM F1 < Gate 4 F1 (validates pretraining benefit)
- Gate 7 ensemble F1 ≥ Gate 4 F1 (ensemble at least matches best)

### Red Flags
- Gate 2 fails (MiniRocket wins) → **Critical**
- Gate 3 linear probe < 60% → **Warning**
- Gate 4 improvement < 0.03 (3pp) → **Warning**
- Gate 6 SimpleLSTM > Gate 4 Enhanced → **Warning**

---

## Next Steps After Completion

### If All Gates Pass
1. Download results and models
2. Select best model (Gate 4 or 7)
3. Run additional validation on held-out test set
4. Deploy to production

### If Some Gates Fail
1. Analyze failure logs
2. Identify root cause using decision tree
3. Fix configuration/code
4. Re-run from failed gate

### If Multiple Gates Fail
1. Check data quality (Phase 0 analysis)
2. Verify temporal splits are correct
3. Review pretrained encoder quality
4. Consider simplifying workflow (skip augmentation)

---

**Remember**: The workflow is designed to fail fast. A failure is not a bug, it's validation working correctly.
