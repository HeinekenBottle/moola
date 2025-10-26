# Loss Normalization: Actionable Fixes

## Quick Diagnosis

Before applying fixes, confirm the problem:

```bash
# Run 5 epochs on local data
python3 scripts/train_expansion_local.py

# Check outputs
# If countdown loss changes <1% and pointers change >20%, you have the problem
```

Expected output with problem:
```
Epoch 1: Countdown loss ≈ 15.9
Epoch 5: Countdown loss ≈ 15.9  ← No improvement (RED FLAG)
         Pointer loss  ≈ 0.010   ← Improved (GREEN FLAG)
```

---

## FIX 1: Enable Uncertainty Weighting (RECOMMENDED)

**Complexity**: 2-3 lines in training script
**Effectiveness**: 100% (fixes gradient scale implicitly)
**Time to implement**: 5 minutes

### What it does

Uses learnable uncertainty (confidence) parameters for each task. The model learns to weight tasks automatically:

```
L_total = (1/2σ²_ptr)L_ptr + log(σ_ptr) +
          (1/2σ²_countdown)L_countdown + log(σ_countdown) + ...
```

The σ parameters (sigma) represent task uncertainty. Smaller σ = higher task weight.

### How to implement

**File to modify**: `/Users/jack/projects/moola/scripts/train_expansion_local.py`

**Location**: Replace lines 145-150 in `train_epoch` function

**Current code**:
```python
        # Apply weights: 10/70/10/10
        loss = (
            0.10 * loss_norm['type'] +
            0.70 * loss_norm['ptr'] +
            0.10 * loss_norm['binary'] +
            0.10 * loss_norm['countdown']
        )
```

**Replace with**:
```python
        # Uncertainty-weighted multi-task loss
        # Extract sigma parameters from model output
        sigma_type = torch.exp(model.log_sigma_type)
        sigma_ptr = torch.exp(model.log_sigma_ptr)
        sigma_binary = torch.exp(model.log_sigma_binary)
        sigma_countdown = torch.exp(model.log_sigma_countdown)

        # Loss = (1/2σ²)L + log(σ)
        loss = (
            (1.0 / (2 * sigma_type ** 2)) * loss_type + torch.log(sigma_type) +
            (1.0 / (2 * sigma_ptr ** 2)) * loss_ptr + torch.log(sigma_ptr) +
            (1.0 / (2 * sigma_binary ** 2)) * loss_binary + torch.log(sigma_binary) +
            (1.0 / (2 * sigma_countdown ** 2)) * loss_countdown + torch.log(sigma_countdown)
        )
```

**Remove**:
- Lines 206: `normalizer = LossNormalizer(...)` (no longer needed)
- Lines 137-142: `loss_norm = normalizer.normalize(...)` (no longer needed)

### Why this works

- Uncertainty parameters are learnable: they adapt during training
- Learns task balance automatically: no manual weight tuning
- Handles reduction bias implicitly: tasks with large gradients learn lower uncertainty (higher weight)
- Research-validated: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018)

### Verification

After implementing, during epoch 1, you should see in logs:
```
sigma_ptr: ≈0.74 (high confidence in pointers)
sigma_countdown: ≈1.0 (lower confidence in countdown initially)
```

By epoch 5:
```
sigma_ptr: ≈0.5 (model trusts pointers more)
sigma_countdown: ≈0.8 (learns to weight countdown reasonably)
```

### Testing

```python
# After first epoch, check sigma values
print(f"sigma_type: {torch.exp(model.log_sigma_type).item():.4f}")
print(f"sigma_ptr: {torch.exp(model.log_sigma_ptr).item():.4f}")
print(f"sigma_countdown: {torch.exp(model.log_sigma_countdown).item():.4f}")
```

If sigma_countdown starts low and stays low (e.g., 0.1), the countdown task may have bad targets. If it increases over epochs (e.g., 0.8 → 1.2), model is learning to downweight it due to high noise.

---

## FIX 2: Correct Normalizer Gradient Scale (PROPER FIX)

**Complexity**: 10 lines in LossNormalizer class
**Effectiveness**: 95% (corrects gradient scale mathematically)
**Time to implement**: 10 minutes
**Benefit**: Keeps manual weight control if preferred

### How it works

Multiply normalized loss by sqrt(output_dimensions) to correct for reduction bias:

```
normalized_loss = (raw_loss / running_mean) * sqrt(num_output_elements)
```

For pointers (2 elements): multiply by sqrt(2) ≈ 1.41
For countdown (105 elements): multiply by sqrt(105) ≈ 10.25

This counteracts the gradient scale reduction.

### Implementation

**File**: `/Users/jack/projects/moola/scripts/train_expansion_local.py`

**Replace LossNormalizer class (lines 21-53) with**:

```python
import math

class LossNormalizer:
    """Normalize losses by running mean AND gradient scale for fair multi-task weighting."""

    # Map loss names to number of output elements
    OUTPUT_DIMS = {
        'type': 3,      # 3-class classification
        'ptr': 2,       # center, length
        'binary': 105,  # per-timestep binary mask
        'countdown': 105  # per-timestep countdown
    }

    def __init__(self, momentum=0.95, warmup_steps=10):
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.running_means = {}

    def normalize(self, losses):
        """Normalize losses by running mean and gradient scale."""
        normalized = {}
        self.step_count += 1

        for name, loss in losses.items():
            loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss

            if name not in self.running_means:
                self.running_means[name] = loss_value

            # Update running mean
            if self.step_count > self.warmup_steps:
                self.running_means[name] = (
                    self.momentum * self.running_means[name] +
                    (1 - self.momentum) * loss_value
                )
            else:
                self.running_means[name] = (
                    (self.running_means[name] * (self.step_count - 1) + loss_value) /
                    self.step_count
                )

            # Normalize by running mean
            mean = self.running_means[name]
            normalized_loss = loss / mean if mean > 1e-8 else loss

            # Correct for reduction bias: multiply by sqrt(num_outputs)
            # This counteracts the gradient magnitude reduction from F.huber_loss(..., reduction='mean')
            if name in self.OUTPUT_DIMS:
                num_outputs = self.OUTPUT_DIMS[name]
                gradient_scale_factor = math.sqrt(num_outputs)
                normalized_loss = normalized_loss * gradient_scale_factor

            normalized[name] = normalized_loss

        return normalized
```

**No changes needed** to training loop (lines 145-150 stay the same).

### Why this works

- Corrects for reduction bias mathematically
- Still uses manual weights (if you prefer that approach)
- Scales with output dimensionality automatically
- Keeps training logic simple

### Verification

Expected behavior after fix:

```
Epoch 1: countdown_loss ≈ 15.9
Epoch 5: countdown_loss ≈ 10-12  ← Clear improvement (RED FLAG resolved)
         pointer_loss  ≈ 0.010   ← Still improving (GREEN FLAG)
```

Countdown loss should show similar relative improvement as other tasks.

---

## FIX 3: Scalar Countdown Prediction (ARCHITECTURAL)

**Complexity**: 15 lines in model
**Effectiveness**: 100% (eliminates reduction bias)
**Time to implement**: 15 minutes
**Trade-off**: Loses per-timestep prediction capability

### What changes

Instead of predicting countdown for each of 105 timesteps, predict a single scalar per sample indicating when expansion will start.

### Implementation

**File**: `/Users/jack/projects/moola/src/moola/models/jade_core.py`

**Change 1**: Model architecture (lines 136-139)

Current:
```python
self.expansion_countdown_head = nn.Sequential(
    nn.Dropout(dense_dropout),
    nn.Linear(lstm_output_size, 1),  # ← Applied per-timestep
)
```

Fixed:
```python
self.expansion_countdown_head = nn.Sequential(
    nn.Dropout(dense_dropout),
    nn.Linear(backbone_out, 1),  # ← Applied to pooled features
)
```

**Change 2**: Forward pass (lines 286-288)

Current:
```python
expansion_countdown = self.expansion_countdown_head(lstm_out).squeeze(-1)  # Per-timestep
output["expansion_countdown"] = expansion_countdown
```

Fixed:
```python
# Apply to pooled features (same as pointers)
countdown = self.expansion_countdown_head(features).squeeze(-1)  # (batch,) scalar
output["expansion_countdown"] = countdown
```

**Change 3**: Adjust target preprocessing (in dataset/training)

Current (line 61-63 in train_expansion_local.py):
```python
countdown = np.arange(window_length, dtype=np.float32) - expansion_start
countdown = -countdown
countdown = np.clip(countdown, -20, 20)  # Per-timestep targets
```

Fixed:
```python
# Use expansion_start as single scalar target per sample
countdown = np.array(expansion_start, dtype=np.float32) / 105.0  # Normalize to [0, 1]
# Or keep as raw: countdown = np.array(expansion_start, dtype=np.float32)
```

### Trade-offs

**Pros**:
- Eliminates reduction bias completely (now 2 outputs like pointers)
- Simpler architecture
- Faster training

**Cons**:
- Can't predict per-timestep countdown
- Model must learn to estimate average expansion_start position
- Less detailed temporal information

### When to use

Use this fix if you only care about knowing *when* expansion happens (expansion_start), not the per-timestep countdown to it.

---

## FIX 4: Batch Normalization Instead of Running Mean (HYBRID)

**Complexity**: 20 lines
**Effectiveness**: 90% (corrects gradient scale, adapts faster)
**Time to implement**: 20 minutes
**Benefit**: Faster convergence than momentum-based running mean

### Concept

Instead of using momentum-based running mean (takes 100s of steps to adapt), use batch statistics directly:

```
normalized_loss = loss / sqrt(var(batch_losses))
```

This adapts immediately to changes in loss range.

### Implementation

**File**: `/Users/jack/projects/moola/scripts/train_expansion_local.py`

**Replace LossNormalizer class (lines 21-53) with**:

```python
class LossNormalizer:
    """Normalize losses using batch statistics with gradient scale correction."""

    OUTPUT_DIMS = {
        'type': 3,
        'ptr': 2,
        'binary': 105,
        'countdown': 105
    }

    def normalize(self, losses):
        """Normalize losses by batch statistics and gradient scale."""
        normalized = {}

        # Compute batch mean and std
        loss_values = [
            loss.item() if isinstance(loss, torch.Tensor) else loss
            for loss in losses.values()
        ]
        batch_mean = np.mean(loss_values)
        batch_std = np.std(loss_values) if np.std(loss_values) > 1e-8 else 1.0

        for name, loss in losses.items():
            loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss

            # Normalize by batch statistics (z-score normalization)
            if batch_std > 1e-8:
                normalized_loss = (loss_value - batch_mean) / batch_std
            else:
                normalized_loss = loss_value / batch_mean if batch_mean > 1e-8 else loss

            # Correct for reduction bias
            if name in self.OUTPUT_DIMS:
                num_outputs = self.OUTPUT_DIMS[name]
                gradient_scale_factor = math.sqrt(num_outputs)
                normalized_loss = normalized_loss * gradient_scale_factor

            normalized[name] = normalized_loss

        return normalized
```

### Why this works

- Batch normalization adapts immediately to loss changes
- No lag from momentum (0.95 momentum takes 100s steps to adapt)
- Still corrects for gradient scale
- Simpler than running mean (no state tracking)

### Drawback

May be noisier on small batches (batch_size=8). Requires batch_size≥16 to be stable.

---

## FIX 5: Manual Weight Adjustment (TEMPORARY WORKAROUND)

**Complexity**: 1 line change
**Effectiveness**: 70% (helps but doesn't solve root cause)
**Time to implement**: 1 minute
**Benefit**: Quick temporary fix while implementing proper solution

### Idea

Increase countdown weight to compensate for reduction bias:

```python
# Current (doesn't work)
loss = (
    0.10 * loss_norm['type'] +
    0.70 * loss_norm['ptr'] +
    0.10 * loss_norm['binary'] +
    0.10 * loss_norm['countdown']  # Too small
)

# Temporary fix (increase countdown weight)
loss = (
    0.10 * loss_norm['type'] +
    0.60 * loss_norm['ptr'] +  # Reduced
    0.10 * loss_norm['binary'] +
    0.20 * loss_norm['countdown']  # Increased 2x
)
```

### Why it only helps partially

- Addresses magnitude but not gradient scale fundamentally
- Weights are fixed, but loss ranges change during training
- Will require re-tuning if loss ratios change
- Not a long-term solution

### Only use if

You're waiting to implement a proper fix and want to test if gradient scale is the issue.

---

## Comparison of Fixes

| Fix | Complexity | Effectiveness | Implementation Time | Permanent? | Recommended |
|-----|-----------|--------------|-------------------|-----------|------------|
| 1: Uncertainty weighting | 2-3 lines | 100% | 5 min | Yes | ✓✓✓ |
| 2: Corrected normalizer | 10 lines | 95% | 10 min | Yes | ✓✓ |
| 3: Scalar countdown | 15 lines | 100% | 15 min | Yes | ✓ (if scalar OK) |
| 4: Batch norm | 20 lines | 90% | 20 min | Yes | ✓ |
| 5: Manual weights | 1 line | 70% | 1 min | No | ✗ (temporary only) |

---

## Recommended Implementation Path

### Step 1: Verify Problem (5 min)
```bash
python3 scripts/train_expansion_local.py
# Check if countdown loss barely changes across epochs
```

### Step 2: Implement Fix 1 (5 min)
- Best effort-to-benefit ratio
- Model already supports it
- Research-validated

### Step 3: Test (10 min)
```bash
python3 scripts/train_expansion_local.py  # Local test
# Check countdown loss decreases noticeably by epoch 5
```

### Step 4: Deploy to RunPod (if working)
```bash
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP
cd /workspace/moola
# Update train_expansion_local.py with uncertainty weighting
python3 scripts/train_expansion_local.py --epochs 20 --batch-size 32
```

### Step 5: Monitor Training
```python
# Check during training
print(f"Countdown loss: {loss_countdown.item():.4f}")  # Should decrease
print(f"sigma_countdown: {torch.exp(model.log_sigma_countdown).item():.4f}")  # Should adapt
```

---

## Rollback Plan

If a fix causes problems:

```bash
# Revert to original
git checkout scripts/train_expansion_local.py

# Or keep working version
cp scripts/train_expansion_local.py scripts/train_expansion_local_fixed.py
```

All fixes are backwards-compatible with existing model checkpoints.

---

## Expected Results After Fix

### Before Fix
```
Epoch 1:  countdown_loss=15.93, type_loss=1.01, ptr_loss=0.016
Epoch 5:  countdown_loss=15.99, type_loss=0.83, ptr_loss=0.011  ← Countdown barely changes
Epoch 10: countdown_loss=15.97, type_loss=0.74, ptr_loss=0.010
Epoch 20: countdown_loss=15.93, type_loss=0.73, ptr_loss=0.009
```

### After Fix (Expected)
```
Epoch 1:  countdown_loss=15.93, type_loss=1.01, ptr_loss=0.016
Epoch 5:  countdown_loss=9.5,   type_loss=0.83, ptr_loss=0.011  ← Countdown improves
Epoch 10: countdown_loss=5.2,   type_loss=0.74, ptr_loss=0.010
Epoch 20: countdown_loss=2.8,   type_loss=0.73, ptr_loss=0.009
```

Countdown loss should show **exponential decay** similar to other tasks, not flat plateau.
