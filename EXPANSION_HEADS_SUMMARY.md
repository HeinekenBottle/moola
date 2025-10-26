# Expansion Detection Heads - Implementation Summary

**Date:** 2025-10-25
**Status:** Architecture implemented and validated ✅

---

## What Was Built

### 1. Model Architecture (jade_core.py)

Added two new per-timestep prediction heads to `JadeCompact`:

**Binary Head:**
- **Input**: BiLSTM outputs `[batch, 105, 192]`
- **Output**: `[batch, 105]` sigmoid probabilities
- **Purpose**: Classify each bar as "inside expansion" (1) or "outside expansion" (0)
- **Loss**: Binary Cross Entropy

**Countdown Head:**
- **Input**: BiLSTM outputs `[batch, 105, 192]`
- **Output**: `[batch, 105]` regression values
- **Purpose**: Predict "bars until expansion starts" for each timestep
- **Loss**: Huber loss (δ=1.0)

**Enabled via:**
```python
model = JadeCompact(
    input_size=12,
    predict_pointers=True,  # Existing: center/length pointers
    predict_expansion_sequence=True,  # NEW: binary + countdown heads
)
```

###2. Label Generation Functions

**Binary Mask:**
```python
def create_expansion_labels(expansion_start, expansion_end, window_length=105):
    binary_mask = np.zeros(window_length)
    binary_mask[expansion_start:expansion_end+1] = 1.0  # Mark expansion region
    return binary_mask
```

**Example:**
- expansion_start=50, expansion_end=65
- Binary: `[0,0,...,0,1,1,1,...,1,0,0,...,0]`
- Marks 16 bars (bars 50-65 inclusive)

**Countdown:**
```python
    countdown = np.arange(window_length) - expansion_start
    countdown = -countdown  # Flip to count down
```

**Example:**
- expansion_start=50
- Countdown at bar 45: `+5` (5 bars until expansion)
- Countdown at bar 50: `0` (expansion starts here)
- Countdown at bar 55: `-5` (5 bars into expansion)

### 3. Loss Configuration

**Target weights (70/20/10):**
- Pointers: 70% (expansion center/length - PRIMARY)
- Binary + Countdown: 20% (10% each - SECONDARY expansion detection)
- Classification: 10% (consolidation/retracement type - AUXILIARY)

**Formula:**
```python
total_loss = (
    0.70 * loss_pointers +
    0.10 * loss_binary +
    0.10 * loss_countdown +
    0.10 * loss_classification
)
```

### 4. Validation Results ✅

**Test Suite:** `scripts/test_expansion_heads.py`

**Results:**
- ✅ Label generation: Binary mask correctly marks 16 bars (50-65)
- ✅ Countdown: Correctly counts to zero at expansion_start
- ✅ Model architecture: 97,547 parameters (added ~10K for new heads)
- ✅ Forward pass: All output shapes correct
- ✅ Real data: Processed actual labeled sample successfully

**Parameter Count:**
- Original model: ~52K params
- With pointers: ~65K params
- **With expansion heads: ~98K params** (still appropriate for 174 samples)

---

## Critical Issue Discovered: Loss Scale Mismatch

**Problem:** Raw loss values have very different scales:
- Classification loss: 1.22 (large)
- Pointer loss: 0.015 (tiny)
- Binary loss: 0.84 (medium)
- Countdown loss: 0.72 (medium)

**Impact:** Even with λ=0.7 on pointers, classification still dominates:
```
Actual contributions (should be 70/10/10/10):
- Classification: 42.3% (should be 10%)
- Pointers: 3.6% (should be 70%!)
- Binary: 29.1% (should be 10%)
- Countdown: 25.0% (should be 10%)
```

**This is the SAME problem that caused the original fine-tuning failure!**

---

## Solutions

### Option 1: Loss Normalization (Recommended)
Normalize each loss by its initial value:

```python
# Track initial losses (first batch)
if not hasattr(self, 'loss_norms'):
    self.loss_norms = {
        'ptr': loss_ptr.item(),
        'type': loss_type.item(),
        'binary': loss_binary.item(),
        'countdown': loss_countdown.item(),
    }

# Normalize and weight
total_loss = (
    0.70 * (loss_ptr / self.loss_norms['ptr']) +
    0.10 * (loss_type / self.loss_norms['type']) +
    0.10 * (loss_binary / self.loss_norms['binary']) +
    0.10 * (loss_countdown / self.loss_norms['countdown'])
)
```

### Option 2: Adaptive Weighting (Kendall Method)
Use learnable uncertainty parameters (already implemented):

```python
# Already in model:
self.log_sigma_ptr = nn.Parameter(torch.tensor(-0.30))
self.log_sigma_type = nn.Parameter(torch.tensor(0.00))
self.log_sigma_binary = nn.Parameter(torch.tensor(0.00))
self.log_sigma_countdown = nn.Parameter(torch.tensor(0.00))

# Loss:
total_loss = (
    (1 / (2 * sigma_ptr**2)) * loss_ptr + log(sigma_ptr) +
    (1 / (2 * sigma_type**2)) * loss_type + log(sigma_type) +
    (1 / (2 * sigma_binary**2)) * loss_binary + log(sigma_binary) +
    (1 / (2 * sigma_countdown**2)) * loss_countdown + log(sigma_countdown)
)
```

**But:** This failed in original training (pointer task silenced). Need constraints:
- Minimum σ_ptr to prevent collapse
- Maximum σ_type to prevent type dominance

### Option 3: Manual Scale Adjustment
Empirically adjust weights based on observed loss scales:

```python
# Observed: loss_type ≈ 1.2, loss_ptr ≈ 0.015 (ratio: 80x)
# To get 70% contribution from pointers, need weight ≈ 56x (0.7 / 0.015 ≈ 47)

total_loss = (
    50.0 * loss_ptr +      # Boost pointer importance
    0.1 * loss_type +      # Reduce type importance
    1.0 * loss_binary +    # Keep binary as-is
    1.0 * loss_countdown   # Keep countdown as-is
)
```

---

## Next Steps

### Immediate (Before Training)

1. **Choose loss normalization strategy** (recommend Option 1)
2. **Integrate into finetune_jade.py**:
   - Add `create_expansion_labels()` function
   - Update dataset to return binary + countdown targets
   - Modify loss computation with chosen normalization
3. **Update model instantiation**:
   ```python
   model = JadeCompact(
       predict_pointers=True,
       predict_expansion_sequence=True,  # Enable new heads
   )
   ```

### Testing

**Local test (5 minutes):**
```bash
python3 scripts/finetune_jade_expansion.py \
  --data data/processed/labeled/train_latest.parquet \
  --epochs 5 \
  --batch-size 8 \
  --device cpu \
  --use-expansion-heads
```

**Full training (RunPod, 10 minutes):**
```bash
python3 scripts/finetune_jade_expansion.py \
  --data data/processed/labeled/train_latest.parquet \
  --pretrained-encoder artifacts/jade_pretrain/checkpoint_best.pt \
  --freeze-encoder \
  --epochs 20 \
  --batch-size 29 \
  --device cuda \
  --use-expansion-heads
```

### Success Metrics

**Primary (expansion detection):**
- Binary Hit@±3: >60% (bar-level expansion detection accuracy)
- Countdown MAE: <5 bars (accurate timing prediction)
- Pointer center MAE: <0.02 normalized (<2 bars)

**Secondary (pattern classification):**
- F1 macro: >0.50 (relaxed from 0.60 since it's auxiliary)
- Per-class recall: >0.30 (relaxed from 0.40)

**Loss monitoring:**
- Pointer contribution: 60-80% of total loss
- Type contribution: <20% of total loss
- No task silencing (all losses actively contributing)

---

## Implementation Status

✅ **Completed:**
- Model architecture with expansion heads
- Label generation functions
- Validation test suite
- Parameter counting

⏳ **In Progress:**
- Loss normalization strategy selection
- Integration into finetune_jade.py

📋 **TODO:**
- Modify finetune_jade.py with new heads and losses
- Local dry-run on 10k bars
- Full training run on RunPod
- Metric analysis and comparison to baseline

---

## Files Modified

1. **src/moola/models/jade_core.py**:
   - Added `predict_expansion_sequence` parameter
   - Added `expansion_binary_head` (per-timestep binary classifier)
   - Added `expansion_countdown_head` (per-timestep regression)
   - Added uncertainty parameters for new tasks
   - Updated forward pass to output expansion predictions

2. **scripts/test_expansion_heads.py** (NEW):
   - Label generation functions
   - Architecture validation
   - Loss computation tests
   - Real data integration test

---

## Key Insights

`★ Insight ─────────────────────────────────────`
**Why This Approach Should Work:**

1. **Objective Alignment**: Expansion detection (binary + countdown) directly targets the user's goal, not a proxy task
2. **Per-Timestep Learning**: BiLSTM naturally good at sequence modeling - countdown teaches temporal progression
3. **Pre-training Compatibility**: MAE learned per-timestep feature reconstruction, similar to countdown regression
4. **Reduced Complexity**: Classification becomes auxiliary (10% weight), reducing pressure on ambiguous consol/retr distinction

**Critical Success Factor:** Loss normalization must be implemented correctly, or pointers will be silenced again (same as before).
`─────────────────────────────────────────────────`

---

## References

**Pointer Semantics (User confirmed):**
- `expansion_start`: First bar of expansion (overlaps with last bar of consol/retr)
- `expansion_end`: Last bar of expansion
- Pattern type (consol/retr): Describes the pre-expansion state

**Loss Weighting Research:**
- Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018)
- Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing" (ICML 2018)

---

**Last Updated:** 2025-10-25 23:30 UTC
**Next Session:** Implement loss normalization and run local test
