# Jade Sanity Checks - Quick Start

## TL;DR

Run this before every expensive GPU training run:

```bash
python3 scripts/jade_sanity_checks.py
```

- ✅ Exit code 0 → All checks passed, safe to train
- ❌ Exit code 1 → Fix issues before training

**Runtime:** ~3 minutes on CPU, ~1 minute on GPU

---

## What It Checks

1. **Feature Statistics** - No NaN/Inf, proper normalization
2. **Class Balance** - Batches are balanced (ratio > 0.3)
3. **Gradient Flow** - All parameters receive gradients
4. **Scale Invariance** - Features are scale-invariant
5. **Tiny Train Overfit** - Model can learn (>65% on 50 samples)
6. **Shuffle Labels** - Model doesn't achieve high accuracy with random labels

---

## CLI Options

```bash
# Basic usage
python3 scripts/jade_sanity_checks.py

# Run on GPU (faster)
python3 scripts/jade_sanity_checks.py --device cuda

# Custom data/splits
python3 scripts/jade_sanity_checks.py \
    --data data/processed/labeled/train_latest_jade.parquet \
    --splits data/processed/labeled/splits_temporal.json

# Quiet mode (only show failures)
python3 scripts/jade_sanity_checks.py --quiet
```

---

## Python API

```python
from scripts.jade_sanity_checks import run_all_sanity_checks

# Run all checks programmatically
success = run_all_sanity_checks(
    data_path="data/processed/labeled/train_latest_jade.parquet",
    splits_path="data/processed/labeled/splits_temporal.json",
    device="cpu",
    verbose=True
)

if success:
    print("Safe to train!")
else:
    print("Fix issues before training")
```

---

## CI/CD Integration

```bash
# Fail fast pattern
python3 scripts/jade_sanity_checks.py || exit 1
python3 scripts/train_jade_finetune.py --epochs 60 --device cuda
```

---

## Expected Output

```
================================================================================
JADE FINE-TUNING SANITY CHECKS
================================================================================
Data: data/processed/labeled/train_latest_jade.parquet
Splits: data/processed/labeled/splits_temporal.json
Device: cpu
================================================================================

Loading data...
Train samples: 115
Val samples: 26
Features shape: (115, 105, 10)
Number of classes: 2

================================================================================
CHECK 1: Feature Statistics
================================================================================
✅ No NaN values
✅ No Inf values
✅ Reasonable scale: max std=0.106

================================================================================
CHECK 2: Class Balance in Batches
================================================================================
✅ Batches well-balanced: ratio=0.77

================================================================================
CHECK 3: Gradient Flow
================================================================================
✅ Gradient flow check passed: all 18 trainable params have gradients

================================================================================
CHECK 4: Tiny Train Overfit
================================================================================
⚠️  Tiny train test passed (marginal): 68.00% accuracy (expected >65.00%)
    Note: Sparse features (99% zeros) limit overfitting capacity

================================================================================
CHECK 5: Shuffle Labels (Ceiling Check)
================================================================================
✅ Shuffle test passed: 30.77% accuracy

================================================================================
CHECK 6: Scale Invariance
================================================================================
✅ Feature scale check passed: mean_abs=0.004

================================================================================
RESULTS: 6/6 checks passed
================================================================================
```

---

## Interpreting Results

### All Pass (6/6) ✅

```bash
RESULTS: 6/6 checks passed
```

**Action:** Proceed with full 60-epoch training on GPU

### Some Fail ❌

```bash
RESULTS: 4/6 checks passed
Failed tests: Gradient Flow, Tiny Train Overfit
```

**Action:** Fix failing checks before training. See troubleshooting in `SANITY_CHECKS_README.md`.

### Marginal Pass ⚠️

```bash
⚠️  Tiny train test passed (marginal): 68.00% accuracy (expected >65.00%)
```

**Meaning:** Passed but close to threshold. This is normal for sparse features (99% zeros).

**Action:** Proceed with training. Monitor if drops below 60%.

---

## Common Issues

| Check | Failure | Fix |
|-------|---------|-----|
| Feature Statistics | NaN/Inf found | Check feature engineering pipeline |
| Class Balance | Ratio < 0.3 | Fix WeightedRandomSampler config |
| Gradient Flow | Missing gradients | Check model architecture |
| Tiny Train Overfit | Acc < 65% | Increase LR or epochs |
| Shuffle Labels | Acc > 70% | Check for data leakage |
| Scale Invariance | mean_abs > 5 | Fix feature normalization |

---

## Full Documentation

- **Complete Guide:** `scripts/SANITY_CHECKS_README.md` (detailed checks, troubleshooting)
- **Implementation Details:** `scripts/SANITY_CHECKS_SUMMARY.md` (technical details, limitations)
- **Source Code:** `scripts/jade_sanity_checks.py` (700+ lines, fully documented)

---

## Integration with Workflow

### Before Training

```bash
# 1. Run sanity checks (3 min, CPU)
python3 scripts/jade_sanity_checks.py || exit 1

# 2. SSH to RunPod
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP

# 3. Train on GPU (10-15 min)
cd /workspace/moola
python3 scripts/train_jade_finetune.py --epochs 60 --device cuda

# 4. Exit RunPod
exit

# 5. SCP results back
scp -i ~/.ssh/runpod_key ubuntu@YOUR_IP:/workspace/moola/artifacts/models/jade_finetuned.pt ./
```

### After Changing Code

```bash
# Feature engineering changes
python3 scripts/build_jade_features.py
python3 scripts/jade_sanity_checks.py  # Re-validate

# Model architecture changes
# Edit src/moola/models/jade_core.py
python3 scripts/jade_sanity_checks.py  # Re-validate

# Data splits changes
# Edit data/processed/labeled/splits_temporal.json
python3 scripts/jade_sanity_checks.py  # Re-validate
```

---

## Performance

| Environment | Runtime |
|-------------|---------|
| MacBook Pro M1 Max (CPU) | ~3 minutes |
| RunPod RTX 4090 (GPU) | ~1 minute |

**Breakdown:**
- Feature Statistics: < 1s
- Class Balance: < 1s
- Gradient Flow: < 1s
- Scale Invariance: < 1s
- Tiny Train Overfit: ~90s (100 epochs)
- Shuffle Labels: ~45s (20 epochs)

---

## When to Run

✅ **Always run before:**
- Full 60-epoch training on GPU ($$$)
- Submitting results to experiment tracking
- Deploying model to production

⚠️ **Consider running after:**
- Changing feature engineering
- Modifying model architecture
- Updating data splits
- Debugging training failures

❌ **Don't need to run:**
- Every code commit (too slow)
- After documentation changes
- After config file updates (unless training-related)

---

## Exit Codes

```bash
python3 scripts/jade_sanity_checks.py
echo $?
```

- `0` - All checks passed ✅
- `1` - One or more checks failed ❌

---

## Questions?

See full documentation:
- `scripts/SANITY_CHECKS_README.md` - Complete guide
- `scripts/SANITY_CHECKS_SUMMARY.md` - Implementation details
- `scripts/jade_sanity_checks.py` - Source code with inline docs

