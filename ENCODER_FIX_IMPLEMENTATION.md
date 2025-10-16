# Pre-trained Encoder Fix Implementation Guide

**Priority**: 🔴 CRITICAL
**Estimated Time**: 2-3 hours
**Expected Impact**: +5-10% accuracy, break class collapse

---

## Fix #1: Encoder Freezing (CRITICAL)

### File: `src/moola/models/cnn_transformer.py`

#### Change 1: Update `load_pretrained_encoder` method signature

**Location**: Line 1126

**Before**:
```python
def load_pretrained_encoder(self, encoder_path: Path) -> "CnnTransformerModel":
```

**After**:
```python
def load_pretrained_encoder(self, encoder_path: Path, freeze_encoder: bool = True) -> "CnnTransformerModel":
    """Load pre-trained encoder weights from SSL pre-training.

    Args:
        encoder_path: Path to pre-trained encoder weights (.pt file)
        freeze_encoder: If True, freeze encoder weights during initial training (default: True)

    Returns:
        Self with pre-trained encoder weights loaded
    """
```

#### Change 2: Add freezing logic after weight loading

**Location**: After line 1197 (`self.model.load_state_dict(model_state_dict)`)

**Add**:
```python
    # Freeze encoder weights if requested
    if freeze_encoder:
        print(f"[SSL] Freezing encoder weights for initial training")
        frozen_count = 0
        for name, param in self.model.named_parameters():
            # Freeze CNN blocks, Transformer encoder, and positional encoding
            if any(prefix in name for prefix in ["cnn_blocks", "transformer", "rel_pos_enc", "window_pos_weight"]):
                param.requires_grad = False
                frozen_count += 1
                print(f"[SSL]   Frozen: {name}")

        print(f"[SSL] Frozen {frozen_count} encoder parameters")
        print(f"[SSL] Classification head and pointer heads remain trainable")
```

#### Change 3: Add unfreezing scheduler to `fit()` method

**Location**: Line 488, update method signature

**Before**:
```python
def fit(
    self,
    X: np.ndarray,
    y: np.ndarray,
    expansion_start: Optional[np.ndarray] = None,
    expansion_end: Optional[np.ndarray] = None,
) -> "CnnTransformerModel":
```

**After**:
```python
def fit(
    self,
    X: np.ndarray,
    y: np.ndarray,
    expansion_start: Optional[np.ndarray] = None,
    expansion_end: Optional[np.ndarray] = None,
    unfreeze_encoder_after: int = 10,
) -> "CnnTransformerModel":
    """Train CNN→Transformer model with optional multi-task pointer prediction.

    Args:
        X: Feature matrix of shape [N, D] or [N, T, D]
        y: Target labels of shape [N]
        expansion_start: Optional expansion start indices
        expansion_end: Optional expansion end indices
        unfreeze_encoder_after: Unfreeze encoder after N epochs (default: 10, 0=never unfreeze)

    Returns:
        Self for method chaining
    """
```

#### Change 4: Add unfreezing logic in training loop

**Location**: Line 681, inside training loop (after `for epoch in range(self.n_epochs):`)

**Add** (before progressive loss weighting section):
```python
        # Unfreeze encoder after warm-up period
        if epoch == unfreeze_encoder_after and hasattr(self, '_pretrained_encoder_path'):
            print(f"\n[SSL] Unfreezing encoder weights at epoch {epoch}")
            unfrozen_count = 0
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen_count += 1
                    print(f"[SSL]   Unfrozen: {name}")
            print(f"[SSL] Unfrozen {unfrozen_count} encoder parameters")
            print(f"[SSL] Entering full fine-tuning mode\n")
```

---

## Fix #2: Disable Multi-Task Learning (HIGH PRIORITY)

### File: `src/moola/cli.py`

#### Change 1: Add `--no-predict-pointers` flag

**Location**: Line 388, `oof()` command parameters

**Add**:
```python
@app.command()
def oof(
    cfg_dir: Path = typer.Option(...),
    over: bool = typer.Option(False),
    model: str = typer.Option(...),
    seed: int = typer.Option(1337),
    device: str = typer.Option("cpu"),
    load_pretrained_encoder: str = typer.Option(None),
    freeze_encoder: bool = typer.Option(True, help="Freeze encoder during initial training"),  # NEW
    unfreeze_after: int = typer.Option(10, help="Unfreeze encoder after N epochs"),  # NEW
    no_predict_pointers: bool = typer.Option(False, help="Disable pointer prediction (classification only)"),  # NEW
):
```

#### Change 2: Pass flags to model

**Location**: Line 450-451

**Before**:
```python
if load_pretrained_encoder:
    model_kwargs["load_pretrained_encoder"] = load_pretrained_encoder
```

**After**:
```python
if load_pretrained_encoder:
    model_kwargs["load_pretrained_encoder"] = load_pretrained_encoder
    model_kwargs["freeze_encoder"] = freeze_encoder
    model_kwargs["unfreeze_encoder_after"] = unfreeze_after

# Override predict_pointers if flag set
if no_predict_pointers:
    model_kwargs["predict_pointers"] = False
```

### File: `src/moola/models/__init__.py`

#### Change: Update `get_model()` to handle new parameters

**Location**: Line 59-72

**Before**:
```python
# Extract load_pretrained_encoder parameter (only for cnn_transformer)
load_pretrained_encoder = kwargs.pop("load_pretrained_encoder", None)

# Instantiate model
model = model_class(**kwargs)

# Load pre-trained encoder if specified (only for cnn_transformer)
if load_pretrained_encoder and name == "cnn_transformer":
    from pathlib import Path
    encoder_path = Path(load_pretrained_encoder)
    # Note: load_pretrained_encoder will be called during fit() after model is built
    # Store the path for later use in fit()
    model._pretrained_encoder_path = encoder_path
```

**After**:
```python
# Extract SSL-related parameters (only for cnn_transformer)
load_pretrained_encoder = kwargs.pop("load_pretrained_encoder", None)
freeze_encoder = kwargs.pop("freeze_encoder", True)
unfreeze_encoder_after = kwargs.pop("unfreeze_encoder_after", 10)

# Instantiate model
model = model_class(**kwargs)

# Store SSL parameters if pre-trained encoder specified
if load_pretrained_encoder and name == "cnn_transformer":
    from pathlib import Path
    encoder_path = Path(load_pretrained_encoder)
    model._pretrained_encoder_path = encoder_path
    model._freeze_encoder = freeze_encoder
    model._unfreeze_encoder_after = unfreeze_encoder_after
```

### File: `src/moola/models/cnn_transformer.py` (update)

#### Change: Use stored SSL parameters during fit

**Location**: Line 581-583

**Before**:
```python
if hasattr(self, '_pretrained_encoder_path'):
    print(f"[SSL] Loading pre-trained encoder from {self._pretrained_encoder_path}")
    self.load_pretrained_encoder(self._pretrained_encoder_path)
```

**After**:
```python
if hasattr(self, '_pretrained_encoder_path'):
    print(f"[SSL] Loading pre-trained encoder from {self._pretrained_encoder_path}")
    freeze_encoder = getattr(self, '_freeze_encoder', True)
    self.load_pretrained_encoder(self._pretrained_encoder_path, freeze_encoder=freeze_encoder)
```

---

## Fix #3: Add Per-Class Monitoring (MEDIUM PRIORITY)

### File: `src/moola/models/cnn_transformer.py`

#### Change: Add per-class metrics during validation

**Location**: Line 869-879, inside validation phase

**After line 869** (`avg_val_loss = val_loss / len(val_dataloader)`):

**Add**:
```python
            # Compute per-class validation metrics
            if (epoch + 1) % 5 == 0:
                from collections import Counter
                val_preds_by_class = Counter()
                val_targets_by_class = Counter()

                for batch_data in val_dataloader:
                    if has_pointers:
                        batch_X, batch_y = batch_data[:2]
                    else:
                        batch_X, batch_y = batch_data

                    batch_X = batch_X.to(self.device, non_blocking=True)
                    outputs = self.model(batch_X)
                    if isinstance(outputs, dict):
                        logits = outputs['classification']
                    else:
                        logits = outputs

                    _, predicted = torch.max(logits, 1)
                    for pred in predicted.cpu().numpy():
                        val_preds_by_class[pred] += 1
                    for target in batch_y.cpu().numpy():
                        val_targets_by_class[target] += 1

                print(f"[CLASS DIST] Epoch {epoch+1} Validation:")
                print(f"  Targets: {dict(val_targets_by_class)}")
                print(f"  Predictions: {dict(val_preds_by_class)}")

                # Alert if class collapse detected
                if len(val_preds_by_class) == 1:
                    collapsed_class = list(val_preds_by_class.keys())[0]
                    print(f"[WARNING] Class collapse detected! Only predicting class {collapsed_class}")
```

---

## Fix #4: Improve Early Stopping for SSL

### File: `src/moola/models/cnn_transformer.py`

#### Change: Adjust early stopping patience for SSL

**Location**: Line 212

**Before**:
```python
early_stopping_patience: int = 20,
```

**After**:
```python
early_stopping_patience: int = 30,  # Increased for SSL transfer learning
```

**Rationale**: SSL transfer learning requires more epochs to converge. Increase patience from 20 to 30.

---

## Testing Checklist

### Test 1: Verify Encoder Freezing
```bash
# Run with frozen encoder
python -m moola.cli oof \
    --model cnn_transformer \
    --device cpu \
    --seed 1337 \
    --load-pretrained-encoder data/artifacts/pretrained/encoder_weights.pt \
    --freeze-encoder \
    --unfreeze-after 5 \
    --no-predict-pointers

# Expected output:
# [SSL] Loaded 74 pre-trained layers
# [SSL] Freezing encoder weights for initial training
# [SSL]   Frozen: cnn_blocks.0.convs.0.conv.weight
# [SSL]   Frozen: cnn_blocks.0.convs.0.conv.bias
# ... (many more frozen layers)
# [SSL] Frozen 72 encoder parameters
# [SSL] Classification head and pointer heads remain trainable
#
# Epoch [5/60] ...
# [SSL] Unfreezing encoder weights at epoch 5
# [SSL]   Unfrozen: cnn_blocks.0.convs.0.conv.weight
# ... (many more unfrozen layers)
# [SSL] Unfrozen 72 encoder parameters
# [SSL] Entering full fine-tuning mode
```

### Test 2: Verify Multi-Task Disabled
```bash
python -m moola.cli oof \
    --model cnn_transformer \
    --device cpu \
    --seed 1337 \
    --load-pretrained-encoder data/artifacts/pretrained/encoder_weights.pt \
    --no-predict-pointers

# Expected output:
# [CLASS BALANCE] Class distribution: {0: 44, 1: 34}
# [LOSS] Using Focal Loss (gamma=2.0) WITHOUT class weights
# (NO multi-task logs - pointer tasks should be disabled)
```

### Test 3: Verify Per-Class Monitoring
```bash
# Run training and watch for per-class metrics
python -m moola.cli oof \
    --model cnn_transformer \
    --device cpu \
    --seed 1337 \
    --load-pretrained-encoder data/artifacts/pretrained/encoder_weights.pt

# Expected output (every 5 epochs):
# [CLASS DIST] Epoch 5 Validation:
#   Targets: {0: 12, 1: 8}
#   Predictions: {0: 15, 1: 5}
#
# [CLASS DIST] Epoch 10 Validation:
#   Targets: {0: 12, 1: 8}
#   Predictions: {0: 13, 1: 7}
#
# (If class collapse occurs:)
# [WARNING] Class collapse detected! Only predicting class 0
```

### Test 4: Full Pipeline Test
```bash
# Run complete OOF generation with all fixes
python -m moola.cli oof \
    --model cnn_transformer \
    --device cuda \
    --seed 1337 \
    --load-pretrained-encoder data/artifacts/pretrained/encoder_weights.pt \
    --freeze-encoder \
    --unfreeze-after 10 \
    --no-predict-pointers

# Expected results:
# - Overall OOF accuracy > 60% (vs 57.14% baseline)
# - Class 1 (retracement) accuracy > 30% (vs 0% baseline)
# - No class collapse warning
```

---

## Validation Criteria

### Success Criteria
- ✅ Encoder weights freeze correctly (logged output confirms)
- ✅ Encoder unfreezes at specified epoch
- ✅ Multi-task can be disabled via flag
- ✅ Per-class metrics logged every 5 epochs
- ✅ Overall accuracy > 60%
- ✅ Class 1 accuracy > 0% (breaks class collapse)

### Failure Criteria
- ❌ Encoder weights not frozen (no freeze logs)
- ❌ Class collapse persists (class 1 accuracy = 0%)
- ❌ Accuracy remains at 57.14% (no improvement)
- ❌ Training crashes or produces errors

---

## Rollback Plan

If fixes cause regressions or errors:

1. **Revert changes**: `git checkout src/moola/models/cnn_transformer.py src/moola/cli.py src/moola/models/__init__.py`
2. **Test baseline**: Verify original behavior works
3. **Apply fixes incrementally**: Start with Fix #1 only, test, then add Fix #2, etc.

---

## Next Steps After Implementation

1. **Run ablation study** (see `ENCODER_AUDIT_SUMMARY.md`)
2. **Implement Bi-LSTM AE pre-training** if improvement < 5%
3. **Tune multi-task loss weights** if classification improves with fixes
4. **Investigate data quality** if class collapse persists

---

**End of Implementation Guide**
