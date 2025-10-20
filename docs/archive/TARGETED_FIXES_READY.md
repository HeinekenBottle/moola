# Targeted Fixes Ready for RunPod Testing

**Date**: 2025-10-18
**Status**: ALL FIXES COMPLETE - Ready for RunPod Upload

---

## Summary

**Production Model**: Gate 1 EnhancedSimpleLSTM (Val Acc: 70.0%, Val F1: 57.6%)

**Targeted Experiment**: Layer-matched transfer learning (Gate 4)

**Control Test**: MiniRocket baseline (Gate 2)

---

## Files Modified (Ready for Upload)

### 1. Gate 4: Layer-Matched Transfer Learning
**File**: `scripts/runpod_gated_workflow/4_finetune_enhanced.py`

**Changes**:
- ✅ `num_layers=1 → 2` (match pretrained encoder)
- ✅ Added PR-AUC, Brier, ECE evaluation metrics
- ✅ Added ECE calculation function
- ✅ Metrics logged in results

**Expected Impact**:
- Tensor match ratio: 44.4% → ~80-90% (layer alignment)
- Pretrained load: PASS with ≥80% gate
- Zero shape mismatches

**Success Criteria**:
- Val F1 > 57.6% (improvement over baseline)
- PR-AUC ↑ (better precision-recall)
- Brier ↓ (better calibration)
- ECE ↓ (better expected calibration)

---

### 2. EnhancedSimpleLSTM: Strict Validation Restored
**File**: `src/moola/models/enhanced_simple_lstm.py`

**Changes**:
- ✅ `min_match_ratio=0.40 → 0.80` (restore strict gate)
- ✅ Updated comment: "Layer-matched should achieve ≥80%"

**Impact**:
- Strict encoder-scope validation enforced
- With num_layers=2, expect 80-90% match (vs 44.4% with mismatch)
- Zero shape mismatches required

---

### 3. SimpleLSTM: Device Bug Fixed
**File**: `src/moola/models/simple_lstm.py`

**Changes**:
- ✅ Line 292: `.to(old_layer.weight.device)` added
- ✅ Dynamic feature encoder now placed on correct device

**Impact**:
- Gate 6 should now pass (no CPU/CUDA mismatch)
- SimpleLSTM baseline ready to run

---

### 4. MiniRocket: Input Shape Fixed
**File**: `scripts/runpod_gated_workflow/2_control_minirocket.py`

**Changes**:
- ✅ Added transpose: `[N, 105, 4] → [N, 4, 105]`
- ✅ MiniRocket expects [N, n_channels, n_timepoints] format
- ✅ Logging added for shape validation

**Impact**:
- Gate 2 should now run without errors
- MiniRocket control test active
- If MiniRocket ≥ Enhanced → STOP and investigate labels/splits

---

## Upload Commands (When RunPod Available)

```bash
# Set RunPod connection details (user will provide)
RUNPOD_IP="<IP>"
RUNPOD_PORT="<PORT>"
SSH_KEY="~/.ssh/id_ed25519"

# Upload fixed files
scp -i $SSH_KEY -P $RUNPOD_PORT \
    scripts/runpod_gated_workflow/4_finetune_enhanced.py \
    root@$RUNPOD_IP:/workspace/moola/scripts/runpod_gated_workflow/

scp -i $SSH_KEY -P $RUNPOD_PORT \
    scripts/runpod_gated_workflow/2_control_minirocket.py \
    root@$RUNPOD_IP:/workspace/moola/scripts/runpod_gated_workflow/

scp -i $SSH_KEY -P $RUNPOD_PORT \
    src/moola/models/enhanced_simple_lstm.py \
    root@$RUNPOD_IP:/workspace/moola/src/moola/models/

scp -i $SSH_KEY -P $RUNPOD_PORT \
    src/moola/models/simple_lstm.py \
    root@$RUNPOD_IP:/workspace/moola/src/moola/models/
```

---

## Execution Plan (RunPod)

### Phase 1: MiniRocket Control (Gate 2)
```bash
ssh root@$RUNPOD_IP -p $RUNPOD_PORT -i $SSH_KEY
cd /workspace/moola
python3 scripts/runpod_gated_workflow/2_control_minirocket.py
```

**Expected Outcome**:
- MiniRocket Val F1 < 57.6% (baseline should win)
- If MiniRocket wins → STOP, investigate labels/splits

---

### Phase 2: Layer-Matched Transfer (Gate 4)
```bash
python3 scripts/runpod_gated_workflow/4_finetune_enhanced.py
```

**Expected Outcome**:
- Pretrained load: ✅ PASS with ~80-90% match ratio
- Val F1 > 57.6% (improvement over baseline)
- PR-AUC ↑, Brier ↓, ECE ↓

**Decision Tree**:
- If improves → Ship finetuned model
- If no improvement → Keep Gate 1 baseline, stop transfer learning

---

### Phase 3: SimpleLSTM Baseline (Gate 6) - Optional
```bash
python3 scripts/runpod_gated_workflow/6_baseline_simplelstm.py
```

**Expected Outcome**:
- No device mismatch error
- SimpleLSTM < EnhancedSimpleLSTM (sanity check)

---

## Guard Rails (Locked)

### Enforced Constraints
- ✅ Forward-chaining only (train: 0-77, val: 78-97)
- ✅ No augmentation in val/test
- ✅ Strict manifests
- ✅ Encoder-scope ≥80% match, 0 shape mismatches
- ✅ Temporal ordering validation

### Monitoring
- Log pretrained load statistics
- Track tensor match ratios
- Record PR-AUC, Brier, ECE for calibration
- Validate split indices

---

## Success Metrics

### Gate 2: MiniRocket Control
**Target**: MiniRocket Val F1 ≤ 57.6%
**If fails**: STOP and investigate labels/splits

### Gate 4: Layer-Matched Transfer
**Target**:
- Val F1 > 57.6% (baseline improvement)
- PR-AUC ↑ (better than baseline)
- Brier ↓ (better than baseline)
- ECE ↓ (better than baseline)

**Baseline Comparison** (Gate 1):
- Val Acc: 70.0%
- Val F1: 57.6%
- (No PR-AUC, Brier, ECE - add for comparison)

---

## Decision Logic

### Scenario A: Gate 4 Improves Metrics
**Outcome**: Ship finetuned model (Gate 4)
**Action**:
- Download model: `enhanced_finetuned_v1.pt`
- Log results in production
- Transfer learning validated

### Scenario B: Gate 4 No Improvement
**Outcome**: Keep Gate 1 baseline
**Action**:
- Document layer matching test results
- Stop transfer learning experiments
- Ship Gate 1 baseline

### Scenario C: MiniRocket Beats Baseline
**Outcome**: ABORT - Investigate data issues
**Action**:
- Review label quality
- Check for split leakage
- Validate data preprocessing
- Do NOT ship any model until resolved

---

## Deferred: TS2Vec Pretraining

**Rationale**:
- BiLSTM encoder already validated (69.8% linear probe)
- 78 samples too small for TS2Vec to help
- Layer matching is simpler and more targeted
- If layer matching works, TS2Vec unnecessary

**Revisit When**:
- Dataset grows to ≥500 samples, OR
- Layer-matched transfer shows clear benefit, OR
- After successful production deployment

---

## Files Changed This Session

1. `/Users/jack/projects/moola/scripts/runpod_gated_workflow/4_finetune_enhanced.py`
   - num_layers: 1 → 2
   - Added PR-AUC, Brier, ECE
   - Added calculate_ece() function

2. `/Users/jack/projects/moola/src/moola/models/enhanced_simple_lstm.py`
   - min_match_ratio: 0.40 → 0.80

3. `/Users/jack/projects/moola/src/moola/models/simple_lstm.py`
   - Line 292: Added .to(device) for dynamic layer

4. `/Users/jack/projects/moola/scripts/runpod_gated_workflow/2_control_minirocket.py`
   - Added reshape: [N, 105, 4] → [N, 4, 105]

---

## Git Commit (When Ready)

```bash
git add scripts/runpod_gated_workflow/4_finetune_enhanced.py
git add scripts/runpod_gated_workflow/2_control_minirocket.py
git add src/moola/models/enhanced_simple_lstm.py
git add src/moola/models/simple_lstm.py

git commit -m "fix: targeted improvements for transfer learning validation

- Gate 4: Match encoder layers (num_layers=2) for proper weight transfer
- Gate 4: Add PR-AUC, Brier, ECE evaluation metrics
- Gate 4: Restore strict validation (min_match_ratio=0.80)
- Gate 2: Fix MiniRocket input shape ([N, 4, 105])
- SimpleLSTM: Fix device placement bug in dynamic feature encoder

Expected impact:
- Transfer learning tensor match: 44.4% → 80-90%
- MiniRocket control test now runnable
- SimpleLSTM baseline ready (Gate 6)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Ready for RunPod Testing

**Status**: ✅ ALL FIXES COMPLETE

**Waiting for**:
- User to provide new RunPod SSH connection details
- Upload files to RunPod
- Execute gates 2, 4, (6)
- Download results

**Expected Duration**:
- Gate 2 (MiniRocket): ~30-60 seconds
- Gate 4 (Finetune): ~60-90 seconds (60 epochs)
- Gate 6 (SimpleLSTM): ~60-90 seconds (60 epochs)
- Total: ~3-5 minutes

**Files Ready**:
- [x] Gate 4 layer matching + metrics
- [x] Gate 2 MiniRocket fix
- [x] SimpleLSTM device fix
- [x] EnhancedSimpleLSTM strict validation
- [x] Production model documented
- [x] Decision logic documented

**Next Step**: User provides RunPod SSH details, then upload and execute.
