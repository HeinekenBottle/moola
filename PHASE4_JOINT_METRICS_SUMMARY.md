# Phase 4: Joint Hit-F1 Metrics Implementation Summary

**Date**: 2025-10-21
**Status**: ✅ Complete
**Purpose**: Evaluate dual-task model performance where BOTH pointer localization AND type classification must be correct

---

## Overview

Implemented comprehensive joint metrics for the dual-task model (pointer regression + type classification). These metrics provide a single score that requires both tasks to be correct, enabling:

1. **Fair model comparison** - Single metric (joint_f1) for ranking models
2. **Error diagnosis** - Breakdown showing which task is bottleneck
3. **Hyperparameter tuning** - Guide for adjusting loss_beta to balance tasks

---

## Implementation Summary

### Files Created

1. **`src/moola/utils/metrics/joint_metrics.py`** (216 lines)
   - `compute_joint_hit_accuracy()` - Joint accuracy where both tasks correct
   - `compute_joint_hit_f1()` - F1 score treating (type, pointer_hit) as classes
   - `compute_task_contribution_analysis()` - Breakdown of error sources
   - `select_best_model_by_joint_metric()` - Model selection with thresholds

2. **`tests/utils/test_joint_metrics.py`** (229 lines)
   - 12 comprehensive unit tests
   - All tests passing ✅
   - Coverage: perfect predictions, partial correctness, tolerance, averaging modes

3. **`configs/phase4_joint_metrics.json`** (184 lines)
   - Metric definitions and formulas
   - Expected value ranges (baseline, good, excellent)
   - Interpretation guide for task contribution patterns
   - Hyperparameter tuning recommendations

4. **`docs/JOINT_METRICS_EXAMPLE.md`** (287 lines)
   - Example CLI output
   - 4 common scenarios with interpretations
   - Model selection examples
   - Hyperparameter tuning guide

### CLI Integration

Modified **`src/moola/cli.py`** (lines 716-797):
- Automatically computes joint metrics when `--predict-pointers` enabled
- Displays after pointer regression metrics, before MC Dropout
- Includes warning if joint accuracy < 30% (task competition)

---

## Key Metrics Explained

### 1. Joint Accuracy
```python
joint_accuracy = (pointer_hit AND type_correct) / total_samples
```

- **Range**: [0.0, 1.0]
- **Target**: >0.50
- **Property**: Always ≤ min(pointer_hit_rate, type_accuracy)

### 2. Joint F1
```python
# Classes: (type * 2 + pointer_hit)
# 0: type=0, pointer_miss
# 1: type=0, pointer_hit
# 2: type=1, pointer_miss
# 3: type=1, pointer_hit
joint_f1 = f1_score(true_joint, pred_joint, average='weighted')
```

- **Range**: [0.0, 1.0]
- **Target**: >0.60
- **Advantage**: Handles class imbalance, provides single ranking metric

### 3. Task Contribution Breakdown
- **Both Correct**: Pointer hit + type correct (ideal: >50%)
- **Pointer Only**: Pointer hit + type wrong (if high: decrease loss_beta)
- **Type Only**: Pointer miss + type correct (if high: increase loss_beta)
- **Both Wrong**: Both tasks incorrect (if high: model capacity issue)

---

## Example Output

```
============================================================
JOINT TASK METRICS (Pointer + Type)
============================================================
Joint Performance (both tasks correct):
  Joint Accuracy:     48.3%
  Pointer Hit Rate:   65.5%
  Type Accuracy:      72.4%
  Joint Correct:      28 / 58

Joint F1 Metrics:
  Joint F1:           0.592
  Joint Precision:    0.634
  Joint Recall:       0.556

Task Contribution Breakdown:
  Both Correct:       48.3% (28 samples)
  Pointer Only:       17.2% (10 samples)
  Type Only:          24.1% (14 samples)
  Both Wrong:         10.3% (6 samples)
============================================================
```

---

## Usage Examples

### Training with Joint Metrics

```bash
# On RunPod GPU
python3 -m moola.cli train \
  --model enhanced_simple_lstm \
  --predict-pointers \
  --device cuda \
  --epochs 60

# Joint metrics automatically displayed after pointer metrics
```

### Model Selection

```python
from moola.utils.metrics.joint_metrics import select_best_model_by_joint_metric

results = [
    {'joint_f1': 0.45, 'pointer_hit_rate': 0.78, 'type_accuracy': 0.51},  # Imbalanced
    {'joint_f1': 0.62, 'pointer_hit_rate': 0.68, 'type_accuracy': 0.72},  # Balanced
    {'joint_f1': 0.48, 'pointer_hit_rate': 0.52, 'type_accuracy': 0.81},  # Imbalanced
]

best_result, best_idx = select_best_model_by_joint_metric(
    results,
    metric_name='joint_f1',
    min_pointer_hit=0.4,
    min_type_acc=0.5
)

# Returns: index=1 (balanced model with joint_f1=0.62)
```

### Hyperparameter Tuning

```bash
# If "Type Only" category is high (>30%), pointer needs more weight
python3 -m moola.cli train --loss-beta 0.7  # Increase from 0.5

# If "Pointer Only" category is high (>30%), type needs more weight
python3 -m moola.cli train --loss-beta 0.3  # Decrease from 0.5

# If "Both Wrong" is high (>40%), not a balance issue
# → Check model capacity, learning rate, data quality
```

---

## Common Patterns

### ✅ Healthy Performance
```
Both Correct:       65.2%  ← Majority fully correct
Pointer Only:       7.6%   ← Low pointer-only errors
Type Only:          15.8%  ← Moderate type-only errors
Both Wrong:         11.4%  ← Low complete failures
```
**Action**: Continue with current hyperparameters

### ⚠️ Pointer Bottleneck
```
Both Correct:       32.1%
Pointer Only:       6.4%
Type Only:          46.5%  ← Type correct but pointer wrong
Both Wrong:         15.0%
```
**Action**: Increase `loss_beta` to 0.7-0.8

### ⚠️ Type Bottleneck
```
Both Correct:       41.4%
Pointer Only:       31.0%  ← Pointer correct but type wrong
Type Only:          10.7%
Both Wrong:         16.9%
```
**Action**: Decrease `loss_beta` to 0.3-0.4

### 🚨 Model Capacity Issue
```
Both Correct:       18.2%
Pointer Only:       16.3%
Type Only:          20.7%
Both Wrong:         44.8%  ← Nearly half completely wrong
```
**Action**: Not a balance issue - check architecture, learning rate, data

---

## Integration with Existing Metrics

### Metric Hierarchy
1. **Component Metrics** (individual tasks)
   - Pointer: hit@±3, hit@±5, MAE (start/end/center/length)
   - Type: accuracy, precision, recall, F1

2. **Joint Metrics** (both tasks together)
   - Joint accuracy, joint F1
   - Task contribution breakdown

3. **Uncertainty Metrics** (optional, Phase 3)
   - MC Dropout, Temperature Scaling, Calibration

### Display Order in CLI
```
1. Basic validation accuracy
2. Pointer regression metrics
3. Joint task metrics          ← NEW (Phase 4)
4. MC Dropout (if enabled)
5. Temperature scaling (if enabled)
6. Calibration metrics (if enabled)
```

---

## Testing

```bash
# Run joint metrics tests
python3 -m pytest tests/utils/test_joint_metrics.py -v

# Results: 12/12 passed ✅
# Coverage:
#   - Perfect predictions
#   - Partial correctness
#   - Tolerance handling
#   - F1 averaging modes (weighted, macro)
#   - Model selection with thresholds
#   - Task contribution breakdown
```

---

## Expected Value Ranges

| Performance Level | Joint Accuracy | Joint F1 | Pointer Hit | Type Accuracy |
|-------------------|----------------|----------|-------------|---------------|
| Baseline (untrained) | 5% | 0.10 | 10% | 50% |
| Good | 50% | 0.60 | 65% | 70% |
| Excellent | 70% | 0.75 | 80% | 85% |

---

## Mathematical Properties

1. **Joint Accuracy Upper Bound**
   ```
   joint_accuracy ≤ min(pointer_hit_rate, type_accuracy)
   ```

2. **Task Independence**
   - If errors are independent: `joint_acc ≈ pointer_hit × type_acc`
   - If errors are correlated: Gap between actual and product indicates correlation

3. **F1 Class Encoding**
   - Ground truth always has pointer_hit=1 (classes 1 or 3)
   - Predictions can be any of 4 classes (0, 1, 2, 3)
   - Weighted averaging handles class imbalance

---

## Next Steps

### Immediate Use
1. Train dual-task model with `--predict-pointers`
2. Review joint metrics to identify bottleneck task
3. Adjust `loss_beta` based on task contribution breakdown
4. Iterate until joint_f1 > 0.60

### Future Enhancements (Optional)
1. **Tolerance Sweep**: Compute joint metrics at multiple tolerances (±1, ±3, ±5)
2. **Per-Class Breakdown**: Separate joint metrics for consolidation vs. retracement
3. **Visualization**: Plot joint accuracy vs. component accuracies
4. **Time Series**: Track joint metrics across training epochs

---

## Documentation References

- **Implementation**: `src/moola/utils/metrics/joint_metrics.py`
- **Tests**: `tests/utils/test_joint_metrics.py`
- **Configuration**: `configs/phase4_joint_metrics.json`
- **Examples**: `docs/JOINT_METRICS_EXAMPLE.md`
- **CLI Integration**: `src/moola/cli.py` (lines 716-797)

---

## Conclusion

Phase 4 joint metrics provide a principled way to:
1. Evaluate dual-task models with a single metric (joint_f1)
2. Diagnose which task is limiting performance (task contribution)
3. Guide hyperparameter tuning (loss_beta adjustment)

The implementation is fully tested, documented, and integrated into the CLI. Joint metrics automatically display when training with `--predict-pointers` enabled.

**Status**: ✅ Ready for production use on RunPod
