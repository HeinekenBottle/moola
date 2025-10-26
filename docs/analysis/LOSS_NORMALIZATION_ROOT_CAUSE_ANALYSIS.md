# Loss Normalization Root Cause Analysis

## Executive Summary

The loss normalization implementation in `scripts/train_expansion_local.py` has a **fundamental architectural flaw**: it normalizes loss *magnitudes* but fails to account for *gradient flow scale differences* between dense outputs (pointers, 2 values) and sequential outputs (countdown, 105 timesteps).

**Result**: Countdown task is trained ~0.18x the gradient scale of pointer task, despite equal normalized loss weights.

## Observed Failure

### RunPod Training Results (210 windows, 20 epochs)

```
Final loss normalizer running means:
  type:     0.7270
  ptr:      0.0095      ← Small absolute value
  binary:   0.2617
  countdown: 15.9264    ← 1600x larger than pointers!
```

**Critical observation**: Despite loss normalization and target weights (10/70/10/10), countdown remains 1600x larger in absolute value than pointers.

### Why This Indicates Failure

If normalization were working correctly:
```
normalized_ptr      = 0.012 / 0.0095  ≈ 1.26
normalized_countdown = 15.5 / 15.93   ≈ 0.97

weighted_ptr        = 0.70 × 1.26 ≈ 0.88
weighted_countdown  = 0.10 × 0.97 ≈ 0.10
```

The weights *look* balanced in loss space. But training doesn't behave this way.

## Root Cause: Reduction Bias in Multi-Task Learning

### The Core Problem

The `LossNormalizer` class normalizes losses by dividing by running mean:

```python
# Line 51 in scripts/train_expansion_local.py
normalized[name] = loss / mean if mean > 1e-8 else loss
```

This works for **loss magnitude** but completely ignores **gradient magnitude**.

In PyTorch's `F.huber_loss(..., reduction='mean')`:
- Element-wise loss is computed
- All elements are summed
- **Divided by number of elements**

This division causes **reduction bias**: outputs computed on more elements have proportionally smaller gradients.

### Mathematical Proof

For a Huber loss with `reduction='mean'`:

```
L = (1/N) * sum(huber_loss(pred_i, target_i))
dL/dpred = (1/N) * dhuber/dpred

So gradient magnitude ∝ 1/N
```

**For pointers**:
- N = 2 (center, length)
- Gradient magnitude ∝ 1/2

**For countdown**:
- N = 105 × batch_size (per-timestep predictions)
- Gradient magnitude ∝ 1/(105 × batch_size)
- ~0.18x the gradient scale of pointers (sqrt(2/105) ≈ 0.138)

### Empirical Validation

Direct gradient flow test:

```python
# Pointers (batch=8, 2 outputs)
ptr_pred.requires_grad = True
loss_ptr = F.huber_loss(ptr_pred, ptr_true, delta=0.08, reduction='mean')
loss_ptr.backward()
ptr_grad_avg = 0.004906  ← Measured

# Countdown (batch=8, 105 timesteps)
cd_pred.requires_grad = True
loss_cd = F.huber_loss(cd_pred, cd_true, delta=1.0, reduction='mean')
loss_cd.backward()
cd_grad_avg = 0.000873   ← Measured

Ratio = cd_grad_avg / ptr_grad_avg = 0.1780
Expected reduction factor = sqrt(2/105) = 0.1380
```

**Result**: Countdown gradients are ~18% the scale of pointer gradients, despite equal normalized loss weights.

## Detailed Code Analysis

### Current Implementation (FAILS)

**File**: `/Users/jack/projects/moola/scripts/train_expansion_local.py`

**LossNormalizer class (lines 21-53)**:
```python
class LossNormalizer:
    """Normalize losses by running mean for fair multi-task weighting."""
    def __init__(self, momentum=0.95, warmup_steps=10):
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.running_means = {}

    def normalize(self, losses):
        normalized = {}
        self.step_count += 1

        for name, loss in losses.items():
            loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss

            if name not in self.running_means:
                self.running_means[name] = loss_value

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

            mean = self.running_means[name]
            normalized[name] = loss / mean if mean > 1e-8 else loss  # ← FAILURE POINT

        return normalized
```

**Problem**: Line 51 only normalizes scalar loss values. It doesn't account for:
1. Number of elements contributing to each loss
2. Resulting gradient scale differences
3. Actual parameter update magnitudes

**Training loop (lines 113-159)**:
```python
def train_epoch(model, loader, optimizer, normalizer, device):
    """Train one epoch with normalized multi-task loss."""
    model.train()
    total_loss = 0

    for batch in loader:
        # ... load data ...

        # Compute raw losses (lines 129-134)
        loss_type = F.cross_entropy(output['logits'], labels)
        loss_ptr = F.huber_loss(output['pointers'], pointers, delta=0.08)
        loss_binary = F.binary_cross_entropy_with_logits(
            output['expansion_binary_logits'], binary
        )
        loss_countdown = F.huber_loss(output['expansion_countdown'], countdown, delta=1.0)

        # Normalize (lines 137-142)
        loss_norm = normalizer.normalize({
            'type': loss_type,
            'ptr': loss_ptr,
            'binary': loss_binary,
            'countdown': loss_countdown,
        })

        # Apply weights: 10/70/10/10 (lines 145-150)
        loss = (
            0.10 * loss_norm['type'] +
            0.70 * loss_norm['ptr'] +
            0.10 * loss_norm['binary'] +
            0.10 * loss_norm['countdown']  # ← Effectively gets ~0.018x gradient weight
        )
```

**Issue**: Even though `loss_norm['countdown']` is weighted with 0.10, its gradient contribution is ~0.018x (0.10 × 0.18 reduction factor) of the pointer contribution.

### Model Architecture (Contributes to Problem)

**File**: `/Users/jack/projects/moola/src/moola/models/jade_core.py`

**Pointer head (lines 106-110)**:
```python
if predict_pointers:
    self.pointer_head = nn.Sequential(
        nn.Dropout(dense_dropout),
        nn.Linear(backbone_out, 2),  # ← 2 outputs
    )
```

**Countdown head (lines 136-139)**:
```python
self.expansion_countdown_head = nn.Sequential(
    nn.Dropout(dense_dropout),
    nn.Linear(lstm_output_size, 1),  # ← But applied per-timestep!
)
```

**Forward pass (lines 280-288)**:
```python
if self.predict_expansion_sequence:
    # Binary head: (batch, 105, 1) -> (batch, 105)
    expansion_binary_logits = self.expansion_binary_head(lstm_out).squeeze(-1)
    output["expansion_binary_logits"] = expansion_binary_logits
    output["expansion_binary"] = torch.sigmoid(expansion_binary_logits)

    # Countdown head: (batch, 105, 1) -> (batch, 105)
    expansion_countdown = self.expansion_countdown_head(lstm_out).squeeze(-1)
    output["expansion_countdown"] = expansion_countdown
```

**The mismatch**:
- `lstm_out` has shape `(batch, 105, lstm_output_size)`
- Linear layer is applied per-timestep, producing `(batch, 105, 1)`
- This creates 105 separate predictions per sample
- Loss computed across all 105 values, creating reduction bias

## Why Countdown Loss Stays at ~16

### The Vicious Cycle

1. **Initialization**: Countdown predictions start random ∈ [-some_init, +some_init]
2. **First batch**: Loss computed on 105 random values, clipped targets ∈ [-20, +20]
3. **Running mean update**: mean ≈ 15-16 (typical Huber loss on ±20 range)
4. **Normalization**: countdown_loss / 16 → normalized value ≈ 1.0
5. **Weighting**: 0.10 × 1.0 = 0.10 contribution
6. **Gradient flow**: 0.10 × 0.18 (reduction bias) ≈ 0.018 effective weight
7. **Parameter updates**: Very small updates to countdown head weights
8. **Next epoch**: Countdown predictions barely improve → loss stays ~15-16
9. **Repeats**: Loop continues, countdown task never learns

### Why It Doesn't Improve

Even if countdown loss decreases slightly (e.g., 15.9 → 15.5):
- Reduction bias of 0.18x remains constant
- Effective gradient weight: 0.10 × 0.18 = 0.018
- Pointer effective weight: 0.70 × 1.0 = 0.70
- Ratio: 0.018 / 0.70 = 0.026
- **Countdown task gets ~2.6% the learning rate of pointer task**

This explains why countdown loss converges to a constant value around 15-16 despite normalization.

## Why Standard Fixes Fail

### Why Uncertainty Weighting Doesn't Help

The project documentation suggests uncertainty-weighted loss (Kendall et al., CVPR 2018):

```
L = (1/2σ_ptr²)L_ptr + log(σ_ptr) + (1/2σ_countdown²)L_countdown + log(σ_countdown)
```

**The model supports this** (lines 112-117 in jade_core.py):
```python
self.log_sigma_ptr = nn.Parameter(torch.tensor(-0.30, dtype=torch.float32))
self.log_sigma_type = nn.Parameter(torch.tensor(0.00, dtype=torch.float32))
```

**But it still fails because**:
- Uncertainty weighting helps balance *different types of losses* (CE vs Huber)
- It does NOT solve *reduction bias from tensor shape differences*
- The σ_countdown parameter can learn to downweight, but then countdown task stops learning entirely

### Why Smaller Huber Delta Doesn't Help

If we use `delta=0.08` instead of `delta=1.0` for countdown:
- Huber loss becomes steeper for small errors
- More penalties for predictions outside [-0.08, 0.08]
- But reduction bias persists: still 0.18x gradient scale
- Just makes convergence worse (exploding gradients on small differences)

### Why Clipping Targets to [-20, 20] Helps... But Doesn't Solve It

Current code (line 63):
```python
countdown = np.clip(countdown, -20, 20)
```

This limits the loss to reasonable values. **But**:
- Countdown loss range: ~0-20 (reasonable)
- Pointer loss range: ~0-0.1 (much smaller range)
- Running mean settles to: countdown ≈ 15-16, pointers ≈ 0.01
- Normalization brings both to ~1.0, but...
- **Reduction bias still causes 0.18x gradient flow**

## Specific Failures in Current Implementation

### Failure 1: Ignores Output Dimensionality

**Location**: `LossNormalizer.normalize()` (line 51)

```python
normalized[name] = loss / mean if mean > 1e-8 else loss
```

**Problem**: Treats all losses equally, ignoring that:
- `loss_ptr` is scalar result of loss on 2 values (gradient ∝ 1/2)
- `loss_countdown` is scalar result of loss on 105 values (gradient ∝ 1/105)

**Fix needed**: Normalize by `loss_magnitude / sqrt(num_contributing_elements)`

### Failure 2: Assumes All Losses Have Similar Gradient Properties

**Location**: `train_epoch()` (lines 145-150)

```python
loss = (
    0.10 * loss_norm['type'] +
    0.70 * loss_norm['ptr'] +
    0.10 * loss_norm['binary'] +
    0.10 * loss_norm['countdown']
)
```

**Problem**: Applies equal weights after normalization, but doesn't account for:
- CE loss on 3-way classification (sparse one-hot target)
- Huber loss on 2 continuous values (dense predictions)
- Huber loss on 105 continuous values (dense predictions, per-timestep)
- BCE loss on 105 binary values (dense binary targets)

Each has different gradient properties due to:
- Target encoding (one-hot vs continuous vs binary)
- Output dimensionality (2 vs 105 vs 105)
- Loss function (CE vs Huber vs BCE)

### Failure 3: Momentum-Based Running Mean Is Too Slow

**Location**: `LossNormalizer.__init__()` and `normalize()` (lines 23-48)

```python
self.momentum = 0.95  # Very high momentum
# ...
self.running_means[name] = (
    self.momentum * self.running_means[name] +
    (1 - self.momentum) * loss_value
)
```

**Problem**: With momentum=0.95:
- Running mean updates very slowly
- Takes ~20 batches to converge to new average
- During training, as model learns, loss ranges change
- Running mean lags behind, causing stale normalization

**Example**: If countdown loss naturally decreases from 16 → 14, with momentum=0.95:
- Step 1: 0.95 × 16 + 0.05 × 14 = 15.9
- Step 2: 0.95 × 15.9 + 0.05 × 14 = 15.81
- Step 20: Would still be ~14.5
- Running mean takes 100+ steps to fully adapt

**Result**: Normalization constantly lags behind actual loss range.

## Recommendations

### Recommendation 1: Remove Running Mean Normalization (SIMPLEST)

**Rationale**: The root problem is architectural (output shape), not magnitude.

**Action**: Use uncertainty-weighted loss instead:

```python
# In train_epoch, replace lines 145-150 with:
sigma_ptr = torch.exp(model.log_sigma_ptr)
sigma_countdown = torch.exp(model.log_sigma_countdown)

loss = (
    (1 / (2 * sigma_ptr ** 2)) * loss_ptr + torch.log(sigma_ptr) +
    (1 / (2 * sigma_countdown ** 2)) * loss_countdown + torch.log(sigma_countdown) +
    # ... similar for type and binary
)
```

**Why**: Uncertainty weighting learns task-specific confidence, adapts online, no running mean needed.

### Recommendation 2: Fix Reduction Bias (CORRECT FIX)

**Rationale**: Normalize by both magnitude AND gradient scale.

**Action**: Modify LossNormalizer:

```python
class LossNormalizerFixed:
    def __init__(self, output_dims=None, momentum=0.95, warmup_steps=10):
        """
        output_dims: dict mapping loss name to number of predicted elements
            e.g., {'ptr': 2, 'countdown': 105, 'binary': 105, 'type': 3}
        """
        self.output_dims = output_dims or {}
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.running_means = {}

    def normalize(self, losses):
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

            # Normalize by running mean AND reduction factor
            mean = self.running_means[name]
            normalized_loss = loss / mean if mean > 1e-8 else loss

            # Correct for reduction bias
            if name in self.output_dims:
                num_outputs = self.output_dims[name]
                reduction_factor = math.sqrt(num_outputs)
                normalized_loss = normalized_loss * reduction_factor

            normalized[name] = normalized_loss

        return normalized
```

**Usage**:
```python
normalizer = LossNormalizerFixed(
    output_dims={
        'type': 3,      # 3-class classification
        'ptr': 2,       # center, length
        'binary': 105,  # per-timestep binary
        'countdown': 105  # per-timestep regression
    }
)
```

### Recommendation 3: Don't Use Per-Timestep Predictions (ARCHITECTURAL)

**Rationale**: The fundamental issue is architecture: dense sequential output (105 values) vs scalar output (2 values).

**Action**: Change countdown prediction to be scalar per sample, not per-timestep:

```python
# CURRENT (WRONG)
self.expansion_countdown_head = nn.Sequential(
    nn.Dropout(dense_dropout),
    nn.Linear(lstm_output_size, 1),  # Applied per timestep → (batch, 105, 1)
)

# FIXED
self.expansion_countdown_head = nn.Sequential(
    nn.Dropout(dense_dropout),
    nn.Linear(backbone_out, 1),  # Applied to global pooled features
)
```

**Impact**:
- Countdown becomes scalar output per sample (like pointers)
- Gradient scale becomes comparable
- Loss normalization actually works
- Trade-off: Model can't learn per-timestep patterns, only sample-level prediction

### Recommendation 4: Use Separate Batch Norm for Dense Tasks (HYBRID)

**Rationale**: Keep per-timestep architecture but normalize differently.

**Action**: Use batch norm internally:

```python
class LossBatchNorm:
    """Normalize losses using batch statistics, not running mean."""

    def normalize(self, losses_dict):
        """
        losses_dict: {name: tensor}
        Returns: {name: normalized tensor}
        """
        normalized = {}

        for name, loss in losses_dict.items():
            # Compute reduction factor for this loss type
            if loss.dim() > 0:
                num_elements = loss.numel()
            else:
                num_elements = 1

            # Normalize: divide by sqrt(num_elements) to correct gradient scale
            reduction_factor = math.sqrt(max(num_elements, 1))
            normalized[name] = loss / reduction_factor

        return normalized
```

**Usage**:
```python
normalizer = LossBatchNorm()
normalized = normalizer.normalize({
    'type': loss_type,
    'ptr': loss_ptr,
    'binary': loss_binary,
    'countdown': loss_countdown,
})
```

### Recommendation 5: Enable Uncertainty Weighting in CLI (EASIEST)

**Status**: Model already supports uncertainty weighting (jade_core.py lines 112-143)

**Action**: Add `--use-uncertainty-weighting` CLI flag to `src/moola/cli.py:train` function

**Why**:
- Uncertainty weighting handles reduction bias automatically
- Model learns task-specific confidence online
- Proven method (Kendall et al., CVPR 2018)
- Just needs CLI plumbing

## Summary Table

| Issue | Location | Current Behavior | Root Cause | Fix |
|-------|----------|------------------|-----------|-----|
| Loss normalization ignores output shape | `LossNormalizer.normalize()` line 51 | Countdown loss stays 15-16 despite normalization | Normalizes magnitude but not gradient scale | Add sqrt(num_elements) factor |
| Reduction bias causes 0.18x gradient | `train_epoch()` lines 145-150 | Countdown gets ~2.6% learning rate of pointers | Per-timestep output (105) vs dense (2) has different gradient magnitude | Use uncertainty weighting or batch norm |
| Running mean too slow | `LossNormalizer.__init__()` momentum=0.95 | Stale normalization during training | High momentum (0.95) requires 100+ steps to adapt | Use batch norm or uncertainty weighting |
| Architecture mismatch | `jade_core.py` lines 136-139, 287 | Dense vs sequential outputs conflict | Countdown applied per-timestep, pointers applied globally | Change countdown to scalar or use separate normalizers |
| Countdown doesn't converge | Multiple factors combine | Loss plateaus at 15-16, training doesn't improve countdown task | Effective gradient weight is 0.018 due to reduction bias | Apply Recommendation 2 or 5 |

## Conclusion

The loss normalization implementation fails because it normalizes loss **magnitudes** but not **gradient magnitudes**. The countdown task, which makes 105 predictions per sample, experiences ~0.18x the gradient scale of the pointer task (which makes 2 predictions). Even with equal normalized loss weights, countdown receives ~2.6% the learning rate.

This is not a bug in the normalizer code itself, but a **fundamental architectural incompatibility** between:
- Dense outputs (pointers: 2 values)
- Sequential outputs (countdown: 105 values)

**Best fixes** (in order of simplicity & effectiveness):
1. Enable uncertainty weighting (already in model, just needs CLI flag)
2. Normalize by gradient scale: `loss / running_mean / sqrt(num_elements)`
3. Change countdown to scalar prediction per sample
4. Use batch normalization instead of running mean

The uncertainty weighting approach (Recommendation 5) is recommended because:
- Model architecture already supports it
- Learns task balance automatically
- Handles reduction bias implicitly
- Only needs CLI flag addition
