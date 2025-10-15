# Phase 3: Migration Guide - Updating to Multi-Task System

## Overview

This guide helps you migrate existing code to use the new multi-task pointer prediction system.

**Good News**: Phase 3 is **fully backward compatible**. Your existing code will continue to work without changes.

---

## Migration Paths

### Path 1: Keep Using Single-Task (No Changes Needed)

If you're happy with classification-only:

```python
# Your existing code works as-is
model = CnnTransformerModel(seed=1337)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**No action required!**

---

### Path 2: Enable Multi-Task (Minimal Changes)

If you want to add pointer prediction:

#### Before (Single-Task)
```python
model = CnnTransformerModel(
    seed=1337,
    n_epochs=60,
    batch_size=32,
    device='cpu'
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

#### After (Multi-Task)
```python
model = CnnTransformerModel(
    seed=1337,
    n_epochs=60,
    batch_size=32,
    device='cpu',
    predict_pointers=True,      # NEW: Enable multi-task
    loss_alpha=0.5,              # NEW: Classification weight
    loss_beta=0.25               # NEW: Pointer weight
)

# NEW: Pass pointer labels
model.fit(
    X_train, y_train,
    pointer_starts=train_starts,  # [N] in [0, 44]
    pointer_ends=train_ends        # [N] in [0, 44]
)

# NEW: Get multi-task predictions
results = model.predict_with_pointers(X_test)
y_pred = results['labels']          # Same as before
start_pred = results['start_predictions']  # NEW
end_pred = results['end_predictions']      # NEW
```

**Changes**:
1. Add `predict_pointers=True` to constructor
2. Pass `pointer_starts` and `pointer_ends` to `fit()`
3. Use `predict_with_pointers()` instead of `predict()`

---

## Data Preparation

### Step 1: Prepare Pointer Labels

Pointer labels must be **relative to inner window** [0, 45), not absolute [30, 75).

#### If you have absolute indices:
```python
# Legacy system gives absolute indices
absolute_starts = [35, 42, 50, ...]  # Range [30, 75)
absolute_ends = [55, 60, 68, ...]

# Convert to relative indices
relative_starts = absolute_starts - 30  # Range [0, 45)
relative_ends = absolute_ends - 30

# Validate
assert np.all((relative_starts >= 0) & (relative_starts < 45))
assert np.all((relative_ends >= 0) & (relative_ends < 45))
```

#### If you have data in DataFrame:
```python
import pandas as pd
from moola.schemas.canonical_v1 import validate_training_data

# Add pointer columns
df['pointer_start'] = relative_starts
df['pointer_end'] = relative_ends

# Validate schema
validated_df = validate_training_data(df)
```

### Step 2: Handle Missing Pointers

If some samples don't have pointer labels:

```python
# Option A: Fill with dummy values (model will learn to ignore)
df['pointer_start'] = df['pointer_start'].fillna(22)  # Middle of window
df['pointer_end'] = df['pointer_end'].fillna(22)

# Option B: Filter to samples with pointers
df_with_pointers = df.dropna(subset=['pointer_start', 'pointer_end'])

# Option C: Train single-task on full data, multi-task on subset
model_single = CnnTransformerModel(predict_pointers=False)
model_single.fit(X_all, y_all)

model_multi = CnnTransformerModel(predict_pointers=True)
model_multi.fit(X_with_pointers, y_with_pointers, starts, ends)
```

---

## Evaluation Updates

### Before (Classification Only)
```python
from sklearn.metrics import accuracy_score
from moola.utils.metrics import calculate_metrics

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
metrics = calculate_metrics(y_test, y_pred, y_proba)

print(f"Accuracy: {accuracy:.1%}")
print(f"F1: {metrics['f1']:.3f}")
```

### After (Multi-Task)
```python
from sklearn.metrics import accuracy_score
from moola.utils.metrics import calculate_metrics, compute_pointer_metrics

# Get multi-task predictions
results = model.predict_with_pointers(X_test)

# Classification metrics (same as before)
y_pred = results['labels']
y_proba = results['probabilities']
accuracy = accuracy_score(y_test, y_pred)
metrics = calculate_metrics(y_test, y_pred, y_proba)

# Pointer metrics (NEW)
pointer_metrics = compute_pointer_metrics(
    results['start_probabilities'],
    results['end_probabilities'],
    test_starts,
    test_ends,
    k=3
)

print(f"Classification Accuracy: {accuracy:.1%}")
print(f"F1: {metrics['f1']:.3f}")
print(f"Start AUC: {pointer_metrics['start_auc']:.3f}")
print(f"End AUC: {pointer_metrics['end_auc']:.3f}")
print(f"Start Precision@3: {pointer_metrics['start_precision_at_k']:.1%}")
```

---

## Save/Load Updates

### Before
```python
from pathlib import Path

# Save
model.save(Path("models/classifier.pt"))

# Load
model = CnnTransformerModel(device='cpu')
model.load(Path("models/classifier.pt"))
```

### After (No Changes!)

Same code works for both single-task and multi-task models:

```python
# Save multi-task model
model.save(Path("models/multitask.pt"))

# Load automatically detects mode
loaded_model = CnnTransformerModel(device='cpu')
loaded_model.load(Path("models/multitask.pt"))

# Check mode
if loaded_model.predict_pointers:
    print("Loaded multi-task model")
    results = loaded_model.predict_with_pointers(X_test)
else:
    print("Loaded single-task model")
    y_pred = loaded_model.predict(X_test)
```

---

## Training Loop Updates

### Before (Basic Training)
```python
model = CnnTransformerModel(n_epochs=60, device='cpu')
model.fit(X_train, y_train)
```

### After (Multi-Task with Logging)
```python
model = CnnTransformerModel(
    n_epochs=60,
    device='cpu',
    predict_pointers=True,
    loss_alpha=0.5,
    loss_beta=0.25
)

# Training automatically logs per-task losses
model.fit(X_train, y_train, pointer_starts=starts, pointer_ends=ends)

# Output includes:
# [MULTI-TASK] Training with pointer prediction enabled
# [MULTI-TASK] Loss weights: alpha=0.5, beta=0.25
# Batch 0 | Class: 1.57 | Start: 0.42 | End: 0.84
```

---

## Hyperparameter Tuning

### Loss Weight Grid Search

```python
import numpy as np
from sklearn.model_selection import GridSearchCV

# Define grid
alphas = [0.3, 0.5, 0.7]
results = []

for alpha in alphas:
    beta = (1.0 - alpha) / 2  # Ensure total = 1.0

    model = CnnTransformerModel(
        seed=1337,
        n_epochs=60,
        predict_pointers=True,
        loss_alpha=alpha,
        loss_beta=beta,
        device='cpu'
    )

    model.fit(X_train, y_train, pointer_starts=starts, pointer_ends=ends)

    # Evaluate
    preds = model.predict_with_pointers(X_val)
    class_acc = accuracy_score(y_val, preds['labels'])

    pointer_metrics = compute_pointer_metrics(
        preds['start_probabilities'],
        preds['end_probabilities'],
        val_starts,
        val_ends
    )

    results.append({
        'alpha': alpha,
        'beta': beta,
        'class_acc': class_acc,
        'start_auc': pointer_metrics['start_auc'],
        'end_auc': pointer_metrics['end_auc']
    })

# Find best
best = max(results, key=lambda x: x['class_acc'])
print(f"Best: alpha={best['alpha']}, beta={best['beta']}")
```

---

## Common Migration Issues

### Issue 1: Pointer Indices Out of Range

**Error**:
```
AssertionError: pointer_starts must be in range [0, 44]
```

**Cause**: Using absolute indices instead of relative

**Fix**:
```python
# Wrong (absolute)
pointer_starts = [30, 35, 40, ...]

# Correct (relative)
pointer_starts = [0, 5, 10, ...]  # Subtract 30
```

### Issue 2: Shape Mismatch

**Error**:
```
pointer_starts shape mismatch: expected (100,), got (100, 1)
```

**Fix**:
```python
# Wrong
pointer_starts = np.array([[5], [10], ...])  # Shape (N, 1)

# Correct
pointer_starts = np.array([5, 10, ...])  # Shape (N,)
# Or
pointer_starts = pointer_starts.flatten()
```

### Issue 3: Forgot to Enable Flag

**Error**:
```
ValueError: Model not trained with pointer prediction
```

**Cause**: Called `predict_with_pointers()` on single-task model

**Fix**:
```python
# Add flag when creating model
model = CnnTransformerModel(predict_pointers=True)
```

### Issue 4: Missing Pointer Labels

**Error**:
```
ValueError: predict_pointers=True but pointer labels not provided
```

**Fix**:
```python
# Pass pointer labels to fit()
model.fit(
    X_train, y_train,
    pointer_starts=starts,  # Don't forget!
    pointer_ends=ends       # Don't forget!
)
```

---

## Rollback Plan

If you encounter issues, rollback is simple:

### Option 1: Revert to Single-Task

```python
# Just set predict_pointers=False
model = CnnTransformerModel(predict_pointers=False)
model.fit(X_train, y_train)  # No pointer labels needed
```

### Option 2: Use Old Checkpoints

Old model checkpoints work with new code:

```python
# Load old single-task model
model = CnnTransformerModel(device='cpu')
model.load(Path("old_model.pt"))

# Automatically detected as single-task
print(model.predict_pointers)  # False
```

---

## Performance Comparison

Track performance before/after migration:

```python
# Before migration (single-task)
model_before = CnnTransformerModel(predict_pointers=False)
model_before.fit(X_train, y_train)
acc_before = accuracy_score(y_test, model_before.predict(X_test))

# After migration (multi-task)
model_after = CnnTransformerModel(predict_pointers=True)
model_after.fit(X_train, y_train, pointer_starts=starts, pointer_ends=ends)
results = model_after.predict_with_pointers(X_test)
acc_after = accuracy_score(y_test, results['labels'])

print(f"Before: {acc_before:.1%}")
print(f"After:  {acc_after:.1%}")
print(f"Change: {(acc_after - acc_before)*100:+.1f}pp")

# Expected: +2-8pp improvement from multi-task learning
```

---

## Migration Checklist

- [ ] **Prepare pointer labels** (relative to inner window)
- [ ] **Update model initialization** (add `predict_pointers=True`)
- [ ] **Update training call** (add `pointer_starts` and `pointer_ends`)
- [ ] **Update prediction code** (use `predict_with_pointers()`)
- [ ] **Update evaluation** (add pointer metrics)
- [ ] **Test on validation set** (verify metrics improve)
- [ ] **Update logging/monitoring** (track per-task losses)
- [ ] **Update documentation** (note multi-task mode)
- [ ] **Train baseline** (single-task for comparison)
- [ ] **Deploy to production** (after validation)

---

## Example: Full Migration

### Before (Single-Task Pipeline)
```python
# data_loading.py
def load_data():
    df = pd.read_parquet("data/processed/train.parquet")
    X = np.stack(df['features'].values)
    y = df['label'].values
    return X, y

# train.py
X_train, y_train = load_data()

model = CnnTransformerModel(
    seed=1337,
    n_epochs=60,
    device='cuda'
)

model.fit(X_train, y_train)
model.save(Path("models/classifier.pt"))

# evaluate.py
model = CnnTransformerModel(device='cuda')
model.load(Path("models/classifier.pt"))

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.1%}")
```

### After (Multi-Task Pipeline)
```python
# data_loading.py
def load_data():
    df = pd.read_parquet("data/processed/train.parquet")
    X = np.stack(df['features'].values)
    y = df['label'].values

    # NEW: Load pointer labels
    starts = df['pointer_start'].values
    ends = df['pointer_end'].values

    # Convert to relative indices if needed
    if starts.max() >= 45:  # Absolute indices detected
        starts = starts - 30
        ends = ends - 30

    return X, y, starts, ends

# train.py
X_train, y_train, starts, ends = load_data()

model = CnnTransformerModel(
    seed=1337,
    n_epochs=60,
    device='cuda',
    predict_pointers=True,    # NEW
    loss_alpha=0.5,            # NEW
    loss_beta=0.25             # NEW
)

model.fit(
    X_train, y_train,
    pointer_starts=starts,     # NEW
    pointer_ends=ends          # NEW
)
model.save(Path("models/multitask.pt"))

# evaluate.py
model = CnnTransformerModel(device='cuda')
model.load(Path("models/multitask.pt"))

results = model.predict_with_pointers(X_test)  # NEW
y_pred = results['labels']

# Classification metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.1%}")

# NEW: Pointer metrics
pointer_metrics = compute_pointer_metrics(
    results['start_probabilities'],
    results['end_probabilities'],
    test_starts,
    test_ends
)
print(f"Start AUC: {pointer_metrics['start_auc']:.3f}")
print(f"End AUC: {pointer_metrics['end_auc']:.3f}")
```

---

## Support

If you encounter issues during migration:

1. **Check example script**: `examples/multitask_pointer_training.py`
2. **Review quick reference**: `PHASE3_QUICK_REFERENCE.md`
3. **Read full docs**: `PHASE3_IMPLEMENTATION_SUMMARY.md`
4. **Test with mock data** before using real data
5. **Compare single-task vs multi-task** on same data

---

**Last Updated**: 2025-10-12
**Version**: Phase 3 Complete
