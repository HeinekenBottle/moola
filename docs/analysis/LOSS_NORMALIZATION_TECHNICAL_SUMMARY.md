# Loss Normalization: Technical Summary

## The Problem in One Sentence

Normalizing loss *values* doesn't normalize gradient *magnitude* when losses are computed on different numbers of elements (2 values vs 105 values).

## Key Metrics

| Metric | Value | Impact |
|--------|-------|--------|
| Countdown loss magnitude | 15.93 | 1600x larger than pointers |
| Countdown gradient scale | 0.18x | Only 18% of pointer gradients |
| Effective countdown learning rate | 0.018 | 2.6% of pointer task learning rate |
| Required reduction factor | sqrt(2/105) = 0.138 | Missing from current normalizer |

## Mathematical Root Cause

PyTorch's `F.huber_loss(..., reduction='mean')` computes:

```
L = (1/N) * sum(huber_loss(pred_i, target_i))
dL/dpred = (1/N) * dhuber/dpred
```

The gradient is inversely proportional to N (number of elements).

**For pointers**: N = 2, gradient ∝ 1/2
**For countdown**: N = 105, gradient ∝ 1/105
**Ratio**: (1/105) / (1/2) = 2/105 ≈ 0.018

## Why Current Approach Fails

### Code Location
`/Users/jack/projects/moola/scripts/train_expansion_local.py` line 51:

```python
normalized[name] = loss / mean if mean > 1e-8 else loss
```

This only divides by the running mean, ignoring gradient scale.

### The Failure Chain

1. Raw countdown loss ≈ 15.5
2. Running mean ≈ 15.9
3. Normalized: 15.5 / 15.9 ≈ 0.97 ✓ (looks balanced!)
4. Apply weight: 0.10 × 0.97 = 0.097
5. But gradient magnitude: 0.097 × 0.18 ≈ 0.018 (reduction bias!)
6. Compare to pointers: 0.70 × 1.0 = 0.70 (pointer gradient gets 0.70)
7. Ratio: 0.018 / 0.70 ≈ 0.026 = 2.6%

**Result**: Countdown task trained at 2.6% the rate of pointer task, despite normalized loss weights.

## Architecture Mismatch

### Pointers (Working)
- Global average pooling: (batch, hidden) → (batch, hidden)
- Fully connected: (batch, hidden) → (batch, 2)
- Loss on 2 values per sample
- Gradient scale: baseline

### Countdown (Broken)
- Per-timestep application: (batch, 105, hidden) → (batch, 105, 1)
- Loss on 105 values per sample
- Gradient scale: 0.18x baseline
- Stays at constant 15-16 loss despite training

## Specific Code Failures

### Failure 1: Missing Gradient Scale Correction
**File**: `scripts/train_expansion_local.py`, lines 21-53 (LossNormalizer class)
**Issue**: Normalizes magnitude only
**Fix**: Multiply by sqrt(output_dimensions)

### Failure 2: Architectural Incompatibility
**File**: `src/moola/models/jade_core.py`, lines 136-139, 287
**Issue**: Different output shapes (2 vs 105) not handled
**Fix**: Use uncertainty weighting or make countdown scalar

### Failure 3: Slow Convergence
**File**: `scripts/train_expansion_local.py`, line 24
**Issue**: momentum=0.95 is too high, running mean lags
**Fix**: Use batch norm instead or disable momentum during learning

## Recommended Fixes (Priority Order)

### 1. Enable Uncertainty Weighting (EASIEST)
- Already implemented in model (jade_core.py:112-143)
- Just needs CLI flag: `--use-uncertainty-weighting`
- Learns task balance automatically
- Handles reduction bias implicitly

### 2. Fix Normalizer Gradient Scale (CORRECT)
```python
# Add to LossNormalizer.normalize()
reduction_factor = math.sqrt(output_dims.get(name, 1))
normalized[name] = (loss / mean) * reduction_factor
```

### 3. Change to Scalar Countdown (ARCHITECTURAL)
- Predict countdown per-sample, not per-timestep
- Loses temporal information but fixes gradient bias
- Simpler training dynamics

### 4. Use Batch Norm Instead (HYBRID)
- Normalize by actual batch statistics
- Adapts faster than momentum-based running mean
- Still per-timestep but corrects gradient scale

## Why Other Approaches Fail

| Approach | Reason It Fails |
|----------|-----------------|
| Smaller Huber delta | Doesn't fix reduction bias, just makes gradients explodier |
| Clipping targets [-20,20] | Helps loss magnitude but not gradient scale |
| Higher weight on countdown | Makes it worse (compounds the problem) |
| Manual lambda tuning | Subjective, doesn't adapt during training |
| Standard uncertainty weighting (without correction) | Works well with correct implementation |

## Verification

To verify the fix works:

1. **Check gradient magnitude**:
```python
# During training, after loss.backward()
ptr_grad_norm = model.pointer_head[1].weight.grad.norm().item()
cd_grad_norm = model.expansion_countdown_head[1].weight.grad.norm().item()
# Should be approximately equal (both ~0.001-0.01 range)
# Currently: cd_grad_norm ≈ 0.18x ptr_grad_norm
```

2. **Check loss convergence**:
```python
# After 20 epochs, countdown loss should decrease noticeably
# Currently: stays constant at 15-16
# With fix: should decrease to ~5-10
```

3. **Check task performance**:
```python
# Model should improve countdown predictions
# Currently: random predictions (loss barely decreases)
# With fix: should show learning (prediction MAE decreases)
```

## Timeline

| Time | Symptom |
|------|---------|
| Epoch 1 | Countdown loss ≈ 16.0 (random predictions) |
| Epoch 5 | Countdown loss ≈ 15.9 (minimal change) |
| Epoch 10 | Countdown loss ≈ 15.7 (still minimal) |
| Epoch 20 | Countdown loss ≈ 15.5 (plateaued at 15+) |

This constant plateau is the key diagnostic: countdown task is not learning, despite 20 epochs of training.

## Next Steps

1. Apply Recommendation 1 (uncertainty weighting) or 2 (gradient scale)
2. Monitor countdown gradient magnitude during training
3. Verify countdown loss decreases below 10
4. Confirm countdown predictions converge to targets
5. Retest expansion prediction accuracy

The fix should be simple (2-3 lines of code) and immediately observable in training curves.
