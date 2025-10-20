# Utility Modules Quick Reference Guide

## Overview
This guide provides quick reference for the refactored training utility modules. All utilities are stateless and can be used across any model implementation.

---

## TrainingSetup (`src/moola/utils/training_utils.py`)

### Purpose
Handles training infrastructure setup: DataLoaders, mixed precision, data splitting.

### API

#### `create_dataloader(X, y, batch_size, shuffle, num_workers, device, prefetch_factor=2)`
Creates optimized DataLoader with device-aware configuration.

**Example**:
```python
from moola.utils.training_utils import TrainingSetup
import torch

X = torch.randn(100, 105, 4)
y = torch.randint(0, 3, (100,))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloader = TrainingSetup.create_dataloader(
    X=X,
    y=y,
    batch_size=32,
    shuffle=True,
    num_workers=16,
    device=device,
)
```

**Features**:
- Automatic CPU/CUDA worker configuration (0 workers for CPU)
- Pin memory for CUDA devices
- Persistent workers for efficiency
- Configurable prefetch factor

---

#### `setup_mixed_precision(use_amp, device)`
Configures automatic mixed precision (FP16) training.

**Example**:
```python
scaler = TrainingSetup.setup_mixed_precision(
    use_amp=True,
    device=torch.device("cuda"),
)

# Use in training loop:
with torch.cuda.amp.autocast():
    loss = model(inputs)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Returns**:
- `torch.cuda.amp.GradScaler` if AMP enabled and CUDA available
- `None` otherwise

---

#### `split_data(X, y, val_split, seed, stratify=True)`
Splits data into train/validation sets.

**Example**:
```python
X_train, X_val, y_train, y_val = TrainingSetup.split_data(
    X=X,
    y=y,
    val_split=0.15,
    seed=42,
    stratify=True,  # Preserve class distribution
)
```

**Returns**:
- `(X_train, X_val, y_train, y_val)` if val_split > 0
- `(X, None, y, None)` if val_split == 0

---

## ModelDiagnostics (`src/moola/utils/model_diagnostics.py`)

### Purpose
Provides standardized logging for model diagnostics and monitoring.

### API

#### `log_model_info(model, n_samples)`
Logs parameter counts and parameter-to-sample ratio.

**Example**:
```python
from moola.utils.model_diagnostics import ModelDiagnostics

info = ModelDiagnostics.log_model_info(model, n_samples=100)
# Logs:
# [MODEL] Total parameters: 36,867
# [MODEL] Trainable parameters: 36,867
# [MODEL] Parameter-to-sample ratio: 368.7:1
```

**Returns**:
```python
{
    "total_params": 36867,
    "trainable_params": 36867,
    "frozen_params": 0,
    "param_to_sample_ratio": 368.67,
}
```

---

#### `log_gpu_info(device, use_amp)`
Logs GPU/CUDA diagnostic information.

**Example**:
```python
gpu_info = ModelDiagnostics.log_gpu_info(device, use_amp=True)
# Logs:
# [GPU] Training on: NVIDIA GeForce RTX 3090
# [GPU] Memory allocated: 0.14 GB
# [GPU] Memory reserved: 0.20 GB
# [GPU] Mixed precision (FP16): True
```

**Returns**:
```python
{
    "device": "cuda",
    "gpu_name": "NVIDIA GeForce RTX 3090",
    "memory_allocated_gb": 0.14,
    "memory_reserved_gb": 0.20,
    "use_amp": True,
}
```

---

#### `log_gpu_memory(device, prefix="GPU")`
Logs current GPU memory usage during training.

**Example**:
```python
ModelDiagnostics.log_gpu_memory(device, prefix="EPOCH 10")
# Logs: [EPOCH 10] Memory: 2.34 GB
```

---

#### `log_class_distribution(y, label_to_idx=None)`
Logs class distribution for imbalance analysis.

**Example**:
```python
class_dist = ModelDiagnostics.log_class_distribution(y)
# Logs: [CLASS BALANCE] Class distribution: {0: 50, 1: 30, 2: 20}
```

---

#### `count_frozen_parameters(model)`
Counts frozen vs trainable parameters.

**Example**:
```python
trainable, frozen = ModelDiagnostics.count_frozen_parameters(model)
print(f"Trainable: {trainable}, Frozen: {frozen}")
```

---

## DataValidator (`src/moola/utils/data_validation.py`)

### Purpose
Input validation, reshaping, and label mapping for training pipelines.

### API

#### `reshape_input(X, expected_features=4)`
Normalizes input to 3D format [N, T, F].

**Example**:
```python
from moola.utils.data_validation import DataValidator
import numpy as np

# 2D input: 100 samples, 420 features (105 timesteps * 4 features)
X_2d = np.random.randn(100, 420)
X_3d = DataValidator.reshape_input(X_2d, expected_features=4)
# Result: (100, 105, 4)
```

**Handles**:
- 2D → 3D reshaping
- Validates divisibility by expected features
- Already 3D inputs (pass-through)

---

#### `create_label_mapping(y)`
Creates bidirectional label-to-index mapping.

**Example**:
```python
y = np.array(['bull', 'bear', 'neutral', 'bull', 'bear'])
label_to_idx, idx_to_label, n_classes = DataValidator.create_label_mapping(y)

# label_to_idx: {'bull': 0, 'bear': 1, 'neutral': 2}
# idx_to_label: {0: 'bull', 1: 'bear', 2: 'neutral'}
# n_classes: 3
```

---

#### `convert_labels_to_indices(y, label_to_idx)`
Converts string/int labels to continuous indices.

**Example**:
```python
y = np.array(['bull', 'bear', 'neutral'])
label_to_idx = {'bull': 0, 'bear': 1, 'neutral': 2}
y_indices = DataValidator.convert_labels_to_indices(y, label_to_idx)
# Result: array([0, 1, 2])
```

---

#### `prepare_training_data(X, y, expected_features=4)`
Complete data preparation pipeline (one-call setup).

**Example**:
```python
X = np.random.randn(100, 420)
y = np.array(['bull', 'bear', 'neutral'] * 33 + ['bull'])

X_3d, y_indices, label_to_idx, idx_to_label, n_classes = (
    DataValidator.prepare_training_data(X, y, expected_features=4)
)

# Ready for training:
# X_3d: (100, 105, 4)
# y_indices: array([0, 1, 2, ..., 0])
# label_to_idx: {'bull': 0, 'bear': 1, 'neutral': 2}
# idx_to_label: {0: 'bull', 1: 'bear', 2: 'neutral'}
# n_classes: 3
```

---

## Configuration (`src/moola/config/training_config.py`)

### Purpose
Centralized hyperparameters for all models.

### SimpleLSTM Constants

```python
from moola.config.training_config import (
    SIMPLE_LSTM_HIDDEN_SIZE,        # 64
    SIMPLE_LSTM_NUM_LAYERS,         # 1
    SIMPLE_LSTM_NUM_HEADS,          # 4
    SIMPLE_LSTM_DROPOUT,            # 0.4
    SIMPLE_LSTM_N_EPOCHS,           # 60
    SIMPLE_LSTM_LEARNING_RATE,      # 5e-4
    SIMPLE_LSTM_BATCH_SIZE,         # 512
    SIMPLE_LSTM_WEIGHT_DECAY,       # 1e-4
    SIMPLE_LSTM_VAL_SPLIT,          # 0.15
    SIMPLE_LSTM_MIXUP_ALPHA,        # 0.4
    SIMPLE_LSTM_CUTMIX_PROB,        # 0.5
)
```

**Usage**:
```python
model = SimpleLSTMModel(
    hidden_size=SIMPLE_LSTM_HIDDEN_SIZE,
    n_epochs=SIMPLE_LSTM_N_EPOCHS,
    learning_rate=SIMPLE_LSTM_LEARNING_RATE,
    batch_size=SIMPLE_LSTM_BATCH_SIZE,
)
```

---

## Complete Training Example

```python
import numpy as np
import torch
from moola.models.simple_lstm import SimpleLSTMModel
from moola.utils.data_validation import DataValidator
from moola.utils.training_utils import TrainingSetup
from moola.utils.model_diagnostics import ModelDiagnostics
from moola.config.training_config import (
    SIMPLE_LSTM_HIDDEN_SIZE,
    SIMPLE_LSTM_N_EPOCHS,
    SIMPLE_LSTM_LEARNING_RATE,
    SIMPLE_LSTM_BATCH_SIZE,
)

# 1. Prepare data
X = np.random.randn(100, 420)  # 100 samples, 420 features
y = np.array([0, 1, 2] * 33 + [0])  # 3 classes

X_3d, y_indices, label_to_idx, idx_to_label, n_classes = (
    DataValidator.prepare_training_data(X, y, expected_features=4)
)

# 2. Initialize model with config
model = SimpleLSTMModel(
    hidden_size=SIMPLE_LSTM_HIDDEN_SIZE,
    n_epochs=SIMPLE_LSTM_N_EPOCHS,
    learning_rate=SIMPLE_LSTM_LEARNING_RATE,
    batch_size=SIMPLE_LSTM_BATCH_SIZE,
)

# 3. Train model (utilities used internally)
model.fit(X_3d, y)

# 4. Log diagnostics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ModelDiagnostics.log_model_info(model.model, len(X_3d))
ModelDiagnostics.log_gpu_info(device, use_amp=True)

# 5. Predict
predictions = model.predict(X_3d[:10])
```

---

## Design Principles

### Stateless Utilities
All utilities use static methods - no instance state required.

```python
# ✓ Good: Direct static method call
loader = TrainingSetup.create_dataloader(X, y, ...)

# ✗ Bad: Don't instantiate
# setup = TrainingSetup()  # Not needed!
```

### Type Safety
All methods include type hints for IDE support.

```python
def create_dataloader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    prefetch_factor: int = 2,
) -> DataLoader:
    ...
```

### Device Awareness
Utilities automatically optimize for CPU vs CUDA.

```python
# Automatically configures:
# - num_workers: 16 for CUDA, 0 for CPU
# - pin_memory: True for CUDA, False for CPU
# - persistent_workers: True if workers > 0
```

---

## Migration Guide (For Other Models)

### Before (in model file):
```python
# Scattered throughout fit() method
X = X.reshape(N, T, 4)  # Reshaping
train_loader = DataLoader(...)  # Manual DataLoader setup
scaler = GradScaler() if use_amp else None  # AMP setup
print(f"Parameters: {total_params}")  # Manual logging
```

### After (using utilities):
```python
# In fit() method
X, y_indices, label_map, ... = DataValidator.prepare_training_data(X, y)
ModelDiagnostics.log_model_info(self.model, N)
ModelDiagnostics.log_gpu_info(self.device, self.use_amp)
train_loader = TrainingSetup.create_dataloader(X, y, ...)
scaler = TrainingSetup.setup_mixed_precision(self.use_amp, self.device)
```

---

## Testing

Run verification script:
```bash
PYTHONPATH=src python3 scripts/verify_refactoring.py
```

Expected output:
```
✓ ALL VERIFICATION TESTS PASSED
```

---

## Future Enhancements

### Planned Utilities
- `CheckpointUtils`: Model save/load standardization
- `MetricsTracker`: Training metrics collection
- `AugmentationPipeline`: Composable augmentation strategies

### Usage in Other Models
These utilities are ready for:
- CNN-Transformer (`src/moola/models/cnn_transformer.py`)
- RWKV-TS (`src/moola/models/rwkv_ts.py`)
- MaskedLSTM pre-training (`src/moola/models/masked_lstm.py`)

---

## Troubleshooting

### Import Errors
```python
# ✓ Correct
from moola.utils.training_utils import TrainingSetup

# ✗ Incorrect
from src.moola.utils.training_utils import TrainingSetup
```

### CUDA Out of Memory
Reduce batch size or num_workers:
```python
loader = TrainingSetup.create_dataloader(
    X, y,
    batch_size=256,  # Reduce from 512
    num_workers=8,   # Reduce from 16
    device=device,
)
```

### Stratified Split Errors
Ensure enough samples per class (minimum 2):
```python
# ✗ Bad: Only 1 sample of class 2
y = np.array([0, 1, 2])

# ✓ Good: At least 2 samples per class
y = np.array([0, 0, 1, 1, 2, 2])
```

---

## References

- Architecture Documentation: `/docs/ARCHITECTURE_IMPROVEMENTS.md`
- Verification Script: `/scripts/verify_refactoring.py`
- Configuration File: `/src/moola/config/training_config.py`
- Source Code:
  - `/src/moola/utils/training_utils.py`
  - `/src/moola/utils/model_diagnostics.py`
  - `/src/moola/utils/data_validation.py`
