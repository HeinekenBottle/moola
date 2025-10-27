# RunPod Deployment Plan - Expansion-Focused Training

**Date:** 2025-10-25
**Status:** Local test passed, ready for deployment

---

## What Was Tested Locally

✅ **Architecture**: 97,547 parameters (expansion heads enabled)
✅ **Loss normalization**: Working (handles 80x scale difference)
✅ **Training convergence**: Loss 0.97 → 0.51 over 5 epochs
✅ **Label generation**: Binary mask and countdown validated

---

## Recommended Improvement Before Full Run

**Issue:** Countdown loss scale too large (26.9)
- Current: Countdown ranges from -54 to +50 (full window)
- Better: Clip to ±20 bars for stability

**Quick fix in train_expansion_local.py:**
```python
def create_expansion_labels(expansion_start, expansion_end, window_length=105):
    binary_mask = np.zeros(window_length, dtype=np.float32)
    binary_mask[expansion_start:expansion_end+1] = 1.0

    countdown = np.arange(window_length, dtype=np.float32) - expansion_start
    countdown = -countdown
    countdown = np.clip(countdown, -20, 20)  # ADD THIS LINE

    return binary_mask, countdown
```

---

## RunPod Deployment Command

```bash
# 1. Sync files to RunPod
rsync -avz scripts/train_expansion_local.py root@IP:/workspace/moola/scripts/
rsync -avz data/processed/labeled/train_latest.parquet root@IP:/workspace/moola/data/processed/labeled/
rsync -avz src/moola/models/jade_core.py root@IP:/workspace/moola/src/moola/models/

# 2. SSH to RunPod
ssh root@IP -p PORT -i ~/.ssh/id_ed25519

# 3. Run training (modify script params for full dataset)
cd /workspace/moola
PYTHONPATH=/workspace/moola python3 scripts/train_expansion_local.py \
  --epochs 20 \
  --batch-size 29 \
  --device cuda \
  --max-samples 174  # Use all samples
```

---

## Expected Results

**Primary (expansion detection):**
- Binary Hit@±3: >60%
- Countdown MAE: <5 bars (after clipping)
- Pointer center MAE: <0.02 (<2 bars)

**Secondary (pattern type):**
- F1 macro: >0.50
- Per-class recall: >0.30

**Training time:** ~5-10 minutes (RTX 4090, 174 samples, 20 epochs)

---

## What to Monitor

1. **Loss contributions** (should stabilize):
   - Type: ~10%
   - Pointers: ~70%
   - Binary: ~10%
   - Countdown: ~10%

2. **Loss normalizer running means**:
   - All tasks should contribute fairly
   - No task silencing (all values >0)

3. **Training convergence**:
   - Train loss should decrease smoothly
   - Val loss should not explode

---

## Files Needed on RunPod

1. `scripts/train_expansion_local.py` - Training script
2. `src/moola/models/jade_core.py` - Model with expansion heads
3. `src/moola/features/relativity.py` - Feature pipeline
4. `data/processed/labeled/train_latest.parquet` - 174 samples

---

## Next Steps

1. ✅ Local test passed
2. ⏳ Add countdown clipping (optional but recommended)
3. ⏳ Sync to RunPod
4. ⏳ Run full training
5. ⏳ Analyze results vs baseline (F1 0.355)

---

**Commit:** d83075b (expansion heads architecture)
**Branch:** reset/stones-only
