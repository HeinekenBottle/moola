# Joint Metrics Quick Reference Card

**Phase 4: Joint Hit-F1 Metrics for Dual-Task Models**

---

## Quick Start

```bash
# Train with joint metrics (automatically computed with --predict-pointers)
python3 -m moola.cli train --model enhanced_simple_lstm --predict-pointers --device cuda
```

---

## Metrics at a Glance

| Metric | Formula | Target | Use Case |
|--------|---------|--------|----------|
| **Joint Accuracy** | `(pointer_hit AND type_correct) / N` | >50% | Overall dual-task success rate |
| **Joint F1** | `f1_score((type, pointer_hit))` | >0.60 | Model ranking and selection |
| **Pointer Hit Rate** | `(start_hit AND end_hit) / N` | >65% | Pointer regression performance |
| **Type Accuracy** | `(pred_type == true_type) / N` | >70% | Type classification performance |

---

## Task Contribution Breakdown

| Category | Meaning | Ideal | If High |
|----------|---------|-------|---------|
| **Both Correct** | Pointer hit + type correct | >50% | ✅ Good! |
| **Pointer Only** | Pointer hit + type wrong | <15% | ⚠️ Decrease loss_beta |
| **Type Only** | Pointer miss + type correct | <15% | ⚠️ Increase loss_beta |
| **Both Wrong** | Both tasks incorrect | <20% | 🚨 Model capacity issue |

---

## Hyperparameter Tuning Flowchart

```
1. Train with loss_beta=0.5 (baseline)
   ↓
2. Check task contribution breakdown
   ↓
3. ┌─────────────────┬──────────────────┬──────────────────┐
   │ Type Only >30%  │ Pointer Only >30%│ Both Wrong >40%  │
   │                 │                  │                  │
   │ Increase beta   │ Decrease beta    │ Not beta issue!  │
   │ → 0.6, 0.7, 0.8 │ → 0.4, 0.3, 0.2  │ → Check model    │
   └─────────────────┴──────────────────┴──────────────────┘
   ↓
4. Select model with highest joint_f1 (meeting minimum thresholds)
```

---

## Common Patterns

### ✅ Healthy (Balanced)
```
Joint Accuracy:     58%
Pointer Hit:        68%
Type Accuracy:      72%

Both Correct:       58%  ← Majority
Pointer Only:       10%
Type Only:          14%
Both Wrong:         18%
```
**Action**: Keep current settings

---

### ⚠️ Pointer Bottleneck
```
Joint Accuracy:     32%
Pointer Hit:        39%  ← Low!
Type Accuracy:      79%

Both Correct:       32%
Pointer Only:       7%
Type Only:          47%  ← High!
Both Wrong:         14%
```
**Action**: `--loss-beta 0.7` (increase from 0.5)

---

### ⚠️ Type Bottleneck
```
Joint Accuracy:     41%
Pointer Hit:        72%
Type Accuracy:      52%  ← Low!

Both Correct:       41%
Pointer Only:       31%  ← High!
Type Only:          11%
Both Wrong:         17%
```
**Action**: `--loss-beta 0.3` (decrease from 0.5)

---

### 🚨 Model Capacity Issue
```
Joint Accuracy:     18%
Pointer Hit:        35%  ← Both low
Type Accuracy:      39%  ← Both low

Both Correct:       18%
Pointer Only:       17%
Type Only:          21%
Both Wrong:         44%  ← Very high!
```
**Action**: Not a balance issue! Check:
- Model architecture (increase hidden_dim)
- Learning rate (try 1e-4 or 5e-4)
- Data quality
- Training stability

---

## Model Selection Example

```python
from moola.utils.metrics.joint_metrics import select_best_model_by_joint_metric

results = [
    {'joint_f1': 0.45, 'pointer_hit_rate': 0.78, 'type_accuracy': 0.51},
    {'joint_f1': 0.62, 'pointer_hit_rate': 0.68, 'type_accuracy': 0.72},  # Best
    {'joint_f1': 0.48, 'pointer_hit_rate': 0.52, 'type_accuracy': 0.81},
]

best_result, best_idx = select_best_model_by_joint_metric(
    results,
    metric_name='joint_f1',
    min_pointer_hit=0.4,   # Minimum acceptable pointer performance
    min_type_acc=0.5       # Minimum acceptable type performance
)

# Returns: index=1 (joint_f1=0.62, balanced performance)
```

---

## CLI Output Location

```
Train accuracy: 0.842 | Test accuracy: 0.793

============================================================
POINTER REGRESSION METRICS (Validation Set)
============================================================
  hit@±3:      65.5%
  hit@±5:      78.2%
  ...

============================================================
JOINT TASK METRICS (Pointer + Type)        ← NEW! Phase 4
============================================================
Joint Performance (both tasks correct):
  Joint Accuracy:     48.3%
  Pointer Hit Rate:   65.5%
  Type Accuracy:      72.4%
  ...

Task Contribution Breakdown:
  Both Correct:       48.3% (28 samples)
  Pointer Only:       17.2% (10 samples)
  Type Only:          24.1% (14 samples)
  Both Wrong:         10.3% (6 samples)
============================================================

MC DROPOUT UNCERTAINTY ESTIMATION (if enabled)
...
```

---

## Mathematical Properties

1. **Upper Bound**: `joint_accuracy ≤ min(pointer_hit_rate, type_accuracy)`

2. **Independence Check**:
   - If errors independent: `joint_acc ≈ pointer_hit × type_acc`
   - Gap indicates correlation

3. **Class Encoding** (for joint F1):
   ```
   Class 0: type=0, pointer_miss
   Class 1: type=0, pointer_hit
   Class 2: type=1, pointer_miss
   Class 3: type=1, pointer_hit
   ```

---

## Files Reference

| File | Purpose |
|------|---------|
| `src/moola/utils/metrics/joint_metrics.py` | Core implementation |
| `tests/utils/test_joint_metrics.py` | Unit tests (12 tests) |
| `configs/phase4_joint_metrics.json` | Configuration and targets |
| `docs/JOINT_METRICS_EXAMPLE.md` | Detailed examples |
| `examples/phase4_joint_metrics_demo.py` | Runnable demo |
| `PHASE4_JOINT_METRICS_SUMMARY.md` | Full documentation |

---

## Quick Demo

```bash
# Run demonstration script
python3 examples/phase4_joint_metrics_demo.py

# Output shows:
# - Perfect predictions
# - Pointer bottleneck scenario
# - Type bottleneck scenario
# - Model selection example
```

---

## Key Formulas

```python
# Joint accuracy
pointer_hit = (abs(pred_start - true_start) <= tol) & (abs(pred_end - true_end) <= tol)
type_correct = (pred_type == true_type)
joint_accuracy = (pointer_hit & type_correct).mean()

# Joint F1
pred_joint = pred_type * 2 + pointer_hit.astype(int)
true_joint = true_type * 2 + 1  # Always hit for ground truth
joint_f1 = f1_score(true_joint, pred_joint, average='weighted')
```

---

## Warnings

### ⚠️ Joint Accuracy < 30%
```
WARNING: Joint accuracy < 30% - tasks may be competing!
Consider: (1) Adjust loss_beta, (2) Enable uncertainty weighting
```

### ⚠️ Pointer Hit < 10%
```
WARNING: hit@±3 < 10% - pointer head may be silenced!
Consider: (1) Increase loss_beta, (2) Enable uncertainty weighting
```

---

## When to Use Joint Metrics

✅ **Use for**:
- Model selection (rank by joint_f1)
- Hyperparameter tuning (guide loss_beta)
- Error diagnosis (identify bottleneck task)
- Production monitoring (track joint performance)

❌ **Don't use for**:
- Single-task models (use standard metrics)
- Pre-training (no type labels available)
- Pure pointer regression (no type classification)

---

**Last Updated**: 2025-10-21 | **Phase**: 4 | **Status**: ✅ Production Ready
