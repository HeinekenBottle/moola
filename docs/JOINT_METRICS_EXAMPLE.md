# Joint Metrics Example Output

This document shows example output from Phase 4 joint metrics implementation.

## CLI Output Example

When training a dual-task model with `--predict-pointers` enabled:

```
============================================================
POINTER REGRESSION METRICS (Validation Set)
============================================================
Pointer Regression Performance:
  hit@±3:      65.5% (both start AND end within ±3)
  hit@±5:      78.2% (both start AND end within ±5)
  Center MAE:  3.45 bars (expansion center position)
  Length MAE:  4.12 bars (expansion span length)
  Start MAE:   4.23 bars
  End MAE:     3.87 bars
  Exact Match: 12.1% (both pointers perfect)
============================================================
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

## Interpretation Guide

### Scenario 1: Healthy Dual-Task Performance

```
Joint Accuracy:     65.2%
Pointer Hit Rate:   72.8%
Type Accuracy:      81.0%

Task Contribution:
  Both Correct:       65.2%  ← Good! Majority of predictions fully correct
  Pointer Only:       7.6%   ← Low - pointer doesn't dominate errors
  Type Only:          15.8%  ← Moderate - some type errors
  Both Wrong:         11.4%  ← Low - few complete failures
```

**Analysis**: Model is well-balanced. Joint accuracy is close to the minimum of component accuracies (65.2% is close to 72.8%), indicating tasks are not competing significantly.

**Action**: Continue with current hyperparameters. Consider slight increase in loss_beta to further improve pointer head.

---

### Scenario 2: Pointer Bottleneck

```
Joint Accuracy:     32.1%
Pointer Hit Rate:   38.5%  ← Low pointer performance!
Type Accuracy:      78.6%

Task Contribution:
  Both Correct:       32.1%  ← Low
  Pointer Only:       6.4%   ← Very low
  Type Only:          46.5%  ← High! Type correct but pointer wrong
  Both Wrong:         15.0%
```

**Analysis**: Pointer regression is the bottleneck. Type classifier performs well (78.6%) but pointer head is struggling (38.5%). High "Type Only" category confirms pointer is limiting joint performance.

**Action**:
1. Increase `loss_beta` to give more weight to pointer task
2. Consider enabling uncertainty weighting
3. Check if pointer head is being silenced (hit@±3 < 10%)
4. Verify pointer targets are properly normalized

---

### Scenario 3: Type Classification Bottleneck

```
Joint Accuracy:     41.4%
Pointer Hit Rate:   72.4%
Type Accuracy:      52.1%  ← Low type performance!

Task Contribution:
  Both Correct:       41.4%  ← Moderate
  Pointer Only:       31.0%  ← High! Pointer correct but type wrong
  Type Only:          10.7%  ← Low
  Both Wrong:         16.9%
```

**Analysis**: Type classification is the bottleneck. Pointer regression performs well (72.4%) but type classifier is weak (52.1%). High "Pointer Only" category confirms type is limiting joint performance.

**Action**:
1. Decrease `loss_beta` to give more weight to type classification
2. Check for class imbalance in validation set
3. Consider increasing hidden_dim for type classifier
4. Review type classification head architecture

---

### Scenario 4: Task Competition (Both Low)

```
Joint Accuracy:     18.2%  ← Very low!
Pointer Hit Rate:   34.5%  ← Both tasks performing poorly
Type Accuracy:      38.9%

Task Contribution:
  Both Correct:       18.2%
  Pointer Only:       16.3%
  Type Only:          20.7%
  Both Wrong:         44.8%  ← Nearly half are completely wrong!
```

**Analysis**: Both tasks performing poorly, with high "Both Wrong" category. This indicates fundamental model capacity or training issues, not just task balance.

**Action**:
1. Check for training instability (loss curves)
2. Verify data quality and preprocessing
3. Consider larger model architecture
4. Review learning rate and optimizer settings
5. Check if gradients are flowing properly to both heads

---

## Using Joint Metrics for Model Selection

### Example: Comparing Multiple Models

```python
# Model A: High pointer, low type
results_a = {
    'joint_f1': 0.45,
    'joint_accuracy': 0.42,
    'pointer_hit_rate': 0.78,
    'type_accuracy': 0.51
}

# Model B: Balanced performance
results_b = {
    'joint_f1': 0.62,
    'joint_accuracy': 0.58,
    'pointer_hit_rate': 0.68,
    'type_accuracy': 0.72
}

# Model C: High type, low pointer
results_c = {
    'joint_f1': 0.48,
    'joint_accuracy': 0.44,
    'pointer_hit_rate': 0.52,
    'type_accuracy': 0.81
}

from moola.utils.metrics.joint_metrics import select_best_model_by_joint_metric

best_result, best_idx = select_best_model_by_joint_metric(
    results=[results_a, results_b, results_c],
    metric_name='joint_f1',
    min_pointer_hit=0.4,
    min_type_acc=0.5
)

# Returns: results_b (index=1) with joint_f1=0.62
# Model A and C meet thresholds but have lower joint_f1
```

**Interpretation**: Model B is selected because it has the highest joint F1 (0.62) among models that meet minimum thresholds. Model A and C have imbalanced performance that limits joint task success.

---

## Joint F1 Class Encoding

The joint F1 metric treats (type, pointer_hit) combinations as separate classes:

| Class | Type | Pointer Hit | Description |
|-------|------|-------------|-------------|
| 0 | 0 (consolidation) | Miss | Predicted consolidation, pointer wrong |
| 1 | 0 (consolidation) | Hit | Predicted consolidation, pointer correct |
| 2 | 1 (retracement) | Miss | Predicted retracement, pointer wrong |
| 3 | 1 (retracement) | Hit | Predicted retracement, pointer correct |

**Note**: Ground truth labels are always class 1 or 3 (hits), since we're comparing against correct pointers.

The weighted F1 accounts for class imbalance and provides a single metric that:
- Rewards models that get BOTH tasks correct
- Penalizes models that only get one task correct
- Handles class imbalance gracefully

---

## Hyperparameter Tuning Guide

Based on task contribution breakdown:

| Pattern | loss_beta Action | Reasoning |
|---------|-----------------|-----------|
| Type Only > 30% | Increase beta (e.g., 0.5 → 0.7) | Pointer needs more weight |
| Pointer Only > 30% | Decrease beta (e.g., 0.5 → 0.3) | Type needs more weight |
| Both Correct > 60% | Keep current beta | Well-balanced |
| Both Wrong > 40% | Not beta issue | Model capacity problem |

**Example tuning sequence**:

1. Start with `loss_beta=0.5` (equal weight)
2. Train and evaluate joint metrics
3. If Type Only is high: Try `loss_beta=0.6, 0.7, 0.8`
4. If Pointer Only is high: Try `loss_beta=0.4, 0.3, 0.2`
5. Select model with highest joint_f1 that meets minimum thresholds
