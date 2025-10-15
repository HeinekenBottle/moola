# Phase 3: Multi-Task Pointer Prediction - Quick Reference

## TL;DR

Phase 3 adds multi-task learning to predict:
1. Classification (consolidation/retracement/reversal)
2. Pointer start (expansion start within inner window)
3. Pointer end (expansion end within inner window)

**Status**: ✅ COMPLETE and TESTED

---

## Quick Start

### Single-Task Mode (Backward Compatible)

```python
from moola.models.cnn_transformer import CnnTransformerModel

# Works exactly as before
model = CnnTransformerModel(predict_pointers=False)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Multi-Task Mode (New)

```python
# Enable pointer prediction
model = CnnTransformerModel(
    predict_pointers=True,
    loss_alpha=0.5,    # Classification weight
    loss_beta=0.25     # Each pointer weight
)

# Train with pointer labels
model.fit(
    X_train, y_train,
    pointer_starts=starts,  # [N] indices in [0, 44]
    pointer_ends=ends       # [N] indices in [0, 44]
)

# Get all predictions
results = model.predict_with_pointers(X_test)
```

---

## Modified Files

1. **src/moola/schemas/canonical_v1.py**
   - Added: `pointer_start` (optional)
   - Added: `pointer_end` (optional)

2. **src/moola/utils/losses.py** (NEW)
   - Function: `compute_multitask_loss()`

3. **src/moola/utils/metrics.py**
   - Added: `compute_pointer_metrics()`

4. **src/moola/models/cnn_transformer.py**
   - Added: Multi-task architecture
   - Added: `predict_with_pointers()` method
   - Modified: Training loop
   - Modified: Save/load

---

## API Reference

### Model Initialization

```python
CnnTransformerModel(
    # Existing parameters
    seed=1337,
    n_epochs=60,
    batch_size=32,
    device='cpu',

    # NEW: Multi-task parameters
    predict_pointers=False,  # Enable multi-task
    loss_alpha=0.5,          # Classification weight
    loss_beta=0.25,          # Each pointer weight
)
```

### Training

```python
model.fit(
    X,                        # [N, 105, 4] or [N, 420]
    y,                        # [N] labels
    pointer_starts=None,      # [N] indices [0, 44] (optional)
    pointer_ends=None         # [N] indices [0, 44] (optional)
)
```

### Prediction

```python
# Single-task (existing)
labels = model.predict(X)           # [N] strings
probs = model.predict_proba(X)      # [N, 3]

# Multi-task (new)
results = model.predict_with_pointers(X)
# Returns dict:
# {
#   'labels': [N],                   # Classification
#   'probabilities': [N, 3],
#   'start_probabilities': [N, 45],  # Pointer start
#   'end_probabilities': [N, 45],
#   'start_predictions': [N],
#   'end_predictions': [N]           # Pointer end
# }
```

### Evaluation

```python
from moola.utils.metrics import compute_pointer_metrics

metrics = compute_pointer_metrics(
    start_preds,    # [N, 45] probabilities
    end_preds,      # [N, 45] probabilities
    start_true,     # [N] ground truth
    end_true,       # [N] ground truth
    k=3             # Top-k for precision@k
)

# Returns 8 metrics:
# - start_auc, end_auc
# - start_precision_at_k, end_precision_at_k
# - start_exact_accuracy, end_exact_accuracy
# - avg_start_error, avg_end_error
```

---

## Architecture

```
Input [B, 105, 4]
    ↓
CNN → Transformer (Shared Backbone)
    ↓
[B, 105, d_model]
    ↓
    ├─ Classification Head → [B, 3]
    ├─ Start Head → [B, 45]
    └─ End Head → [B, 45]
```

---

## Loss Function

```
total = alpha * L_class + beta * (L_start + L_end)

L_class = CrossEntropy(class_logits, class_labels)
L_start = BCEWithLogits(start_logits, start_one_hot)
L_end = BCEWithLogits(end_logits, end_one_hot)

Default: alpha=0.5, beta=0.25 → total weight = 1.0
```

---

## Metrics Explained

| Metric | Description | Random Baseline |
|--------|-------------|-----------------|
| AUC | Ranking quality | 50% |
| Precision@3 | True pointer in top-3? | 6.7% |
| Exact Accuracy | Predicted exact timestep? | 2.2% |
| Avg Error | Mean distance (timesteps) | 15 |

---

## Common Patterns

### Pattern 1: Train Multi-Task, Predict Classification Only

```python
model = CnnTransformerModel(predict_pointers=True)
model.fit(X, y, pointer_starts=starts, pointer_ends=ends)

# Still works - just extracts classification
labels = model.predict(X_test)
```

### Pattern 2: Evaluate Pointer Quality

```python
results = model.predict_with_pointers(X_test)
metrics = compute_pointer_metrics(
    results['start_probabilities'],
    results['end_probabilities'],
    y_test_starts,
    y_test_ends
)

print(f"Start AUC: {metrics['start_auc']:.3f}")
print(f"End Precision@3: {metrics['end_precision_at_k']:.1%}")
```

### Pattern 3: Tune Loss Weights

```python
for alpha in [0.3, 0.5, 0.7]:
    beta = (1.0 - alpha) / 2
    model = CnnTransformerModel(
        predict_pointers=True,
        loss_alpha=alpha,
        loss_beta=beta
    )
    model.fit(X, y, pointer_starts=starts, pointer_ends=ends)
    # Evaluate...
```

---

## Performance Targets

### Classification
- Baseline (single-task): 44%
- Target (multi-task): 50-55%
- Hypothesis: Shared reps improve both tasks

### Pointer Prediction
- Start AUC: 55-65%
- End AUC: 55-65%
- Precision@3: 40-50%
- Exact: 5-10%

---

## Troubleshooting

### Error: "Model not trained with pointer prediction"

**Cause**: Called `predict_with_pointers()` on single-task model

**Fix**: Initialize with `predict_pointers=True`

### Error: "predict_pointers=True but pointer labels not provided"

**Cause**: Enabled pointer prediction but didn't pass labels

**Fix**: Call `fit(X, y, pointer_starts=..., pointer_ends=...)`

### Warning: "Pointer labels provided but predict_pointers=False"

**Cause**: Passed pointer labels but didn't enable flag

**Fix**: Set `predict_pointers=True` or remove pointer labels

### Pointer indices out of range

**Cause**: Pointer indices not in [0, 44]

**Fix**: Indices must be relative to inner window [30:75]
- Absolute index 30 → relative index 0
- Absolute index 74 → relative index 44

---

## Testing

Run example script:
```bash
python3 examples/multitask_pointer_training.py
```

Expected: All 4 examples complete successfully

---

## Next Steps

1. **Load Real Data**: Get actual pointer labels
2. **Hyperparameter Tuning**: Grid search loss weights
3. **Compare Models**: Single-task vs multi-task
4. **Production Deployment**: Integrate with pipeline

---

## Resources

- **Full Documentation**: `PHASE3_IMPLEMENTATION_SUMMARY.md`
- **Example Code**: `examples/multitask_pointer_training.py`
- **Schema**: `src/moola/schemas/canonical_v1.py`
- **Loss Function**: `src/moola/utils/losses.py`
- **Metrics**: `src/moola/utils/metrics.py`
- **Model**: `src/moola/models/cnn_transformer.py`

---

**Last Updated**: 2025-10-12
**Version**: Phase 3 Complete
