# Gradient Monitoring Quick Start Guide

## TL;DR

Enable gradient monitoring to detect task collapse in multi-task training:

```bash
python3 -m moola.cli train \
    --model enhanced_simple_lstm \
    --predict-pointers \
    --monitor-gradients \
    --gradient-log-freq 10
```

## What It Does

1. **Monitors gradient flow** - Tracks gradient norms every N epochs
2. **Detects task imbalance** - Checks if pointer/type tasks have balanced gradients
3. **Warns about collapse** - Alerts when one task dominates training
4. **Provides summary** - Shows gradient statistics at end of training

## Example Output

### Healthy Training

```
Epoch 10 Gradient Stats:
  Mean norm: 0.2134
  Max norm:  1.4567

[PHASE 4] Task balance: Tasks balanced (ratio=1.23)

[PHASE 4] Gradient Monitoring Summary:
  Mean gradient norm: 0.2034 ± 0.0245
  Max gradient norm: 1.3156 (peak: 1.8234)
  Task balance: Tasks balanced (ratio=1.34)
```

### Task Collapse Detected

```
Epoch 15 Gradient Stats:
  Mean norm: 0.4567
  Max norm:  2.8901

[PHASE 4] Task imbalance detected at epoch 15!
Pointer/Type gradient ratio: 5.23 (healthy range: 0.5-2.0)

[PHASE 4] TASK COLLAPSE: Pointer task dominating (ratio=6.34 > 5.0)
```

## When to Use

- **Always use** when training multi-task models (pointer + classification)
- **Use during development** when debugging training issues
- **Use when tuning** loss weights (alpha/beta)
- **Optional in production** once training is stable

## Interpreting Results

### Task Balance Ratio

| Ratio | Status | Action |
|-------|--------|--------|
| 0.5-2.0 | ✅ Healthy | No change needed |
| 2.0-3.0 | ⚠️ Minor imbalance | Monitor closely |
| 3.0-5.0 | ⚠️ Significant imbalance | Adjust loss weights |
| > 5.0 | ❌ Task collapse | Urgent: rebalance weights |

### Gradient Norms

| Mean Norm | Status | Action |
|-----------|--------|--------|
| 0.1-0.5 | ✅ Healthy | No change needed |
| < 0.01 | ⚠️ Vanishing | Increase learning rate |
| > 1.0 | ⚠️ Large | Check gradient clipping |
| > 5.0 | ❌ Exploding | Reduce learning rate |

## Fixing Task Collapse

If ratio > 5.0 (pointer dominates):

```bash
# Increase classification task weight
python3 -m moola.cli train \
    --model enhanced_simple_lstm \
    --predict-pointers \
    --loss-alpha 2.0 \
    --loss-beta 1.0 \
    --monitor-gradients
```

If ratio < 0.2 (type dominates):

```bash
# Increase pointer task weight
python3 -m moola.cli train \
    --model enhanced_simple_lstm \
    --predict-pointers \
    --loss-alpha 1.0 \
    --loss-beta 2.0 \
    --monitor-gradients
```

## Performance Impact

- Monitoring overhead: ~1-2% (runs every N epochs, not every batch)
- Can disable in production: `--no-monitor-gradients`

## Full Documentation

See `PHASE4_GRADIENT_MONITORING_SUMMARY.md` for complete details.
