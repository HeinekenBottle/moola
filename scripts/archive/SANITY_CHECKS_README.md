# Jade Fine-tuning Sanity Checks

Fast validation suite to catch bugs before expensive 60-epoch training runs.

## Purpose

Before running full Jade fine-tuning experiments (10-15 minutes on GPU), run these sanity checks (2-3 minutes on CPU) to verify:

1. Features are properly normalized
2. Data loaders produce balanced batches
3. Gradients flow correctly
4. Model can learn (overfit check)
5. Model isn't just memorizing (shuffle check)
6. Features are scale-invariant

## Usage

```bash
# Run all checks (takes ~3 minutes on CPU)
python3 scripts/jade_sanity_checks.py

# Run on GPU (faster for training tests)
python3 scripts/jade_sanity_checks.py --device cuda

# Run with custom data/splits
python3 scripts/jade_sanity_checks.py \
    --data data/processed/labeled/train_latest_jade.parquet \
    --splits data/processed/labeled/splits_temporal.json

# Quiet mode (only show failures)
python3 scripts/jade_sanity_checks.py --quiet
```

## Exit Codes

- `0` - All checks passed ✅
- `1` - One or more checks failed ❌

## Sanity Checks Explained

### 1. Feature Statistics

**What it checks:**
- No NaN or Inf values
- Features are reasonably scaled (std < 10)
- All 10 relativity features present

**Why it matters:**
- NaN/Inf will cause training to crash
- Extreme scales can cause gradient instability
- Missing features indicate data pipeline bug

**Expected output:**
```
✅ No NaN values
✅ No Inf values
✅ Reasonable scale: max std=0.106

Feature statistics (10 relativity features):
  open_norm           : mean=  0.005, std=  0.064, min=  0.000, max=  1.000
  close_norm          : mean=  0.004, std=  0.054, min=  0.000, max=  1.000
  ...
```

---

### 2. Class Balance in Batches

**What it checks:**
- WeightedRandomSampler produces balanced batches
- Both classes appear in batches
- Balance ratio > 0.3 (within 30% of each other)

**Why it matters:**
- Imbalanced batches can cause class collapse
- WeightedRandomSampler is critical for minority class learning
- Verifies sampler configuration is correct

**Expected output:**
```
Mean class distribution over 10 batches: [8.0 8.0]
Balance ratio (min/max): 1.00
✅ Batches well-balanced: ratio=1.00
```

**⚠️ Warning:** If ratio < 0.7, batches are moderately imbalanced but acceptable.

---

### 3. Gradient Flow

**What it checks:**
- All trainable parameters receive gradients
- No gradient vanishing/blocking
- Gradient norms are reasonable

**Why it matters:**
- Parameters without gradients won't learn
- Indicates architecture bugs (missing connections)
- Early detection of gradient issues

**Expected output:**
```
✅ Gradient flow check passed: all 18 trainable params have gradients
   Gradient norms: mean=0.0427, min=0.0003, max=0.1716
```

---

### 4. Tiny Train Overfit

**What it checks:**
- Model can overfit 50 samples to >80% accuracy
- Training loop works correctly
- Model has sufficient capacity

**Why it matters:**
- If model can't overfit tiny dataset, it won't generalize to full dataset
- Validates optimizer, loss function, and training loop
- Catches architecture bugs early

**Expected output:**
```
Training on 50 samples for 100 epochs...
Data sparsity: 99.1% zeros (expected ~99% for relativity features)
  Epoch  20/100: loss=0.5234, acc=75.00%
  Epoch  40/100: loss=0.3567, acc=82.00%
  Epoch  60/100: loss=0.2341, acc=86.00%
  Epoch  80/100: loss=0.1789, acc=88.00%
  Epoch 100/100: loss=0.1234, acc=90.00%
✅ Tiny train test passed: 90.00% accuracy (expected >80%)
```

**⚠️ Note:** Relativity features are 99% zero (warmup period), so overfitting ceiling is ~85-90% (not 99%).

---

### 5. Shuffle Labels (Ceiling Check)

**What it checks:**
- Model with shuffled labels plateaus at random accuracy
- Random accuracy for 2 classes: ~50%
- Random accuracy for 3 classes: ~33%

**Why it matters:**
- If model achieves high accuracy with random labels, it's overfitting to noise
- Validates that model is learning signal, not memorizing artifacts
- Detects data leakage bugs

**Expected output:**
```
Training with shuffled labels for 20 epochs...
  Epoch   5/20: val_acc=48.00%
  Epoch  10/20: val_acc=50.00%
  Epoch  15/20: val_acc=52.00%
  Epoch  20/20: val_acc=51.00%
✅ Shuffle test passed: 50.50% accuracy (expected 35.00%-65.00%, random ~50.00%)
```

**Acceptable range:** Random ± 15% (e.g., 35-65% for 2 classes)

---

### 6. Scale Invariance

**What it checks:**
- Relativity features are normalized (mean absolute value < 5)
- Features are invariant to price scaling

**Why it matters:**
- Relativity features should not leak absolute price information
- Validates feature engineering pipeline
- Ensures model can generalize across price regimes

**Expected output:**
```
✅ Feature scale check passed: mean_abs=0.004 (expected < 5)
   All features have mean absolute value < 5
```

---

## Interpreting Results

### All Checks Pass ✅

```
================================================================================
RESULTS: 6/6 checks passed
================================================================================
```

**Action:** Proceed with full 60-epoch training run on GPU.

### Some Checks Fail ❌

```
================================================================================
RESULTS: 3/6 checks passed
Failed tests: Class Balance in Batches, Tiny Train Overfit
================================================================================
```

**Action:** Fix failing checks before running expensive training.

**Common failures:**

| Failure | Likely Cause | Fix |
|---------|--------------|-----|
| Feature Statistics | NaN/Inf in data | Check feature engineering pipeline |
| Class Balance | WeightedRandomSampler config | Fix sampler weights |
| Gradient Flow | Architecture bug | Check model connections |
| Tiny Train Overfit | Learning rate too low | Increase LR or epochs |
| Shuffle Labels | Data leakage | Check for future information |
| Scale Invariance | Features not normalized | Fix feature normalization |

---

## Integration with Training Workflow

### Recommended Workflow

```bash
# 1. Run sanity checks (3 minutes on CPU)
python3 scripts/jade_sanity_checks.py

# 2. If all pass, run full training (10-15 minutes on GPU)
python3 scripts/train_jade_finetune.py \
    --config configs/jade_finetune.yaml \
    --device cuda \
    --epochs 60

# 3. Evaluate results
python3 scripts/evaluate_jade.py
```

### When to Re-run Sanity Checks

- After changing feature engineering pipeline
- After modifying model architecture
- After updating data splits
- After changing training hyperparameters
- When debugging training failures

---

## Technical Details

### Data Format

Expects parquet file with columns:
- `features`: Array of shape (105, 10) for each sample
- `label`: String labels (e.g., "consolidation", "retracement")
- `expansion_start`, `expansion_end`: Integer indices for pointers

### Relativity Features (10 dimensions)

1. `open_norm` - Normalized open position in candle range
2. `close_norm` - Normalized close position in candle range
3. `body_pct` - Body size as percentage of range
4. `upper_wick_pct` - Upper wick size as percentage of range
5. `lower_wick_pct` - Lower wick size as percentage of range
6. `range_z` - Range z-score relative to EMA
7. `dist_to_prev_SH` - Distance to previous swing high (ATR-normalized)
8. `dist_to_prev_SL` - Distance to previous swing low (ATR-normalized)
9. `bars_since_SH_norm` - Bars since swing high (normalized by window length)
10. `bars_since_SL_norm` - Bars since swing low (normalized by window length)

### Warmup Period

**Important:** First 15 timesteps are zeros (warmup for ATR and zigzag detection).

- Only last ~90 timesteps contain signal
- This is expected behavior for causal features
- Reduces overfitting ceiling in tiny train test

---

## Troubleshooting

### "Failed to load data" Error

**Cause:** Data file not found or corrupted.

**Fix:**
```bash
# Check file exists
ls -lh data/processed/labeled/train_latest_jade.parquet

# Re-run feature engineering if needed
python3 scripts/build_jade_features.py
```

### Tiny Train Overfit Fails (< 80%)

**Cause:** Model can't learn from sparse features.

**Try:**
1. Increase learning rate (3e-3 → 5e-3)
2. Increase epochs (100 → 200)
3. Check if data is all zeros

**Verify data has signal:**
```bash
python3 -c "
import pandas as pd
import numpy as np
df = pd.read_parquet('data/processed/labeled/train_latest_jade.parquet')
X = np.stack([np.stack(row) for row in df['features'].values])
print('Non-zero %:', (X != 0).sum() / X.size * 100)
"
```

### Shuffle Labels Accuracy Too High (> 65% for 2 classes)

**Cause:** Possible data leakage or feature memorization.

**Fix:**
1. Check for future information in features
2. Verify labels are actually shuffled
3. Check for repeated samples in validation set

---

## Performance Benchmarks

**Hardware:** MacBook Pro (M1 Max), CPU only

| Check | Duration | Notes |
|-------|----------|-------|
| Feature Statistics | < 1s | Fast array operations |
| Class Balance | < 1s | 10 batches × 16 samples |
| Gradient Flow | < 1s | Single forward/backward pass |
| Scale Invariance | < 1s | Fast array operations |
| Tiny Train Overfit | ~90s | 100 epochs × 50 samples |
| Shuffle Labels | ~45s | 20 epochs × 115 samples |
| **Total** | **~3 minutes** | All checks sequentially |

**On GPU:** Total runtime ~1 minute (training tests much faster).

---

## Exit Codes

- `0` - All checks passed, safe to train
- `1` - One or more checks failed, fix before training

Use in CI/CD:
```bash
python3 scripts/jade_sanity_checks.py || exit 1
python3 scripts/train_jade_finetune.py
```

---

## Related Files

- `scripts/jade_sanity_checks.py` - Main sanity check suite
- `scripts/train_jade_finetune.py` - Full training script
- `src/moola/features/relativity.py` - Feature engineering pipeline
- `src/moola/models/jade_core.py` - Jade model architecture
- `tests/test_jade_model.py` - Unit tests for Jade model

---

## Contributing

When adding new sanity checks:

1. Add function to `jade_sanity_checks.py`
2. Document expected behavior in this README
3. Add to test list in `run_all_sanity_checks()`
4. Ensure check runs in < 30 seconds

**Naming convention:** `test_<check_name>()`

**Return type:** `bool` (True if passed, raise AssertionError if failed)
