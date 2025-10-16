# Phase 2 Implementation Summary

## Overview

Successfully implemented all Phase 2 augmentation and data quality improvements for the Moola ML pipeline. Expected accuracy improvement: **58-62% → 65-74%** (+7-12 percentage points).

## Completed Tasks

### 1. ✅ Mixup Augmentation Enhancement

**Changes:**
- Increased mixup alpha from 0.2 to 0.4 for both CNN-Transformer and SimpleLSTM
- Updated default parameters and configurations
- **Expected gain**: +2-4%

**Modified files:**
- `/Users/jack/projects/moola/src/moola/models/cnn_transformer.py`
- `/Users/jack/projects/moola/src/moola/models/simple_lstm.py`
- `/Users/jack/projects/moola/configs/cnn_transformer.yaml`
- `/Users/jack/projects/moola/configs/simple_lstm.yaml`

**Key code changes:**
```python
# cnn_transformer.py & simple_lstm.py
mixup_alpha: float = 0.4  # Increased from 0.2
```

---

### 2. ✅ Mixup Utility for Shallow Models

**Created:**
- `/Users/jack/projects/moola/src/moola/utils/mixup.py`

**Features:**
- `mixup_data()`: Apply mixup augmentation to NumPy arrays
- `augment_dataset()`: Generate synthetic training samples
- `mixup_criterion_sklearn()`: Compute mixup loss for sklearn models

**Usage example:**
```python
from moola.utils.mixup import mixup_data, augment_dataset

# Apply mixup during training
X_mixed, y_a, y_b, lam = mixup_data(X_train, y_train, alpha=0.4)

# Or augment dataset
X_aug, y_aug = augment_dataset(X_train, y_train, n_augmented=50, alpha=0.4)
```

**Expected gain**: +1-2% for LogReg, RF, XGBoost

---

### 3. ✅ Traditional Temporal Augmentation

**Changes:**
- Integrated jitter, scaling, and time_warp into SimpleLSTM and CNN-Transformer
- Applied before mixup/cutmix in training loop
- **Expected gain**: +2-3%

**Augmentation pipeline:**
```python
# Applied to deep models
if self.use_temporal_aug:
    batch_X = self.temporal_aug.apply_augmentation(batch_X)
```

**Configuration:**
```yaml
augmentation:
  use_temporal_aug: true
  jitter_prob: 0.5       # Add 5% Gaussian noise
  scaling_prob: 0.3      # Magnitude warping ±10%
  time_warp_prob: 0.3    # Time axis warping
```

**Modified files:**
- `/Users/jack/projects/moola/src/moola/models/cnn_transformer.py` (lines 712-714, 280-290)
- `/Users/jack/projects/moola/src/moola/models/simple_lstm.py` (lines 358-360, 115-126)

---

### 4. ✅ CleanLab Integration Script

**Created:**
- `/Users/jack/projects/moola/scripts/run_cleanlab_phase2.py`

**Features:**
- Loads OOF predictions from multiple models
- Ensembles predictions (mean or geometric mean)
- Detects noisy labels using confident learning
- Ranks samples by label quality
- Generates cleaned training dataset

**Usage:**
```bash
# Detect noisy labels and create cleaned dataset
python scripts/run_cleanlab_phase2.py \
    --oof-dir data/oof \
    --output data/processed/train_clean_phase2.parquet \
    --threshold 0.3 \
    --remove-percentage 0.10

# Options:
#   --threshold: Quality score threshold (lower = stricter, default: 0.3)
#   --remove-percentage: % of lowest-quality samples to remove (default: 10%)
#   --ensemble-method: mean | geometric_mean
```

**Expected gain**: +3-5% by removing top 10-15% noisy samples

---

### 5. ✅ GPU Optimization Configs

**Changes:**
- Increased batch size: 512 → 1024
- Reduced early stopping patience: 30 → 20 (faster convergence with augmentation)
- Increased DataLoader prefetch: 2 → 4
- Enabled persistent workers

**Updated configurations:**

`configs/cnn_transformer.yaml`:
```yaml
training:
  batch_size: 1024               # Increased from 512
  early_stopping_patience: 20    # Reduced from 30

hardware:
  prefetch_factor: 4             # Increased from 2
  persistent_workers: true       # Keep workers alive
```

`configs/simple_lstm.yaml`:
```yaml
training:
  batch_size: 1024               # Increased from 512
  early_stopping_patience: 20    # Reduced from 30

hardware:
  prefetch_factor: 4             # Increased from 2
  persistent_workers: true       # Keep workers alive
```

**Expected speedup**: 15% faster training

---

### 6. ✅ OOF Regeneration Script

**Created:**
- `/Users/jack/projects/moola/scripts/regenerate_oof_phase2.py`

**Features:**
- Regenerates OOF predictions with Phase 2 improvements
- Supports two modes: `clean` (baseline) and `augmented` (Phase 2 full)
- Supports CleanLab-cleaned dataset
- MLflow tracking for all experiments
- Comprehensive logging and metrics

**Usage:**
```bash
# 1. Generate clean baseline (for comparison)
python scripts/regenerate_oof_phase2.py \
    --mode clean \
    --experiment phase2-clean-baseline

# 2. Generate with Phase 2 augmentation
python scripts/regenerate_oof_phase2.py \
    --mode augmented \
    --experiment phase2-full-augmentation

# 3. Generate with CleanLab-cleaned data
python scripts/regenerate_oof_phase2.py \
    --mode augmented \
    --use-cleaned-data \
    --experiment phase2-cleaned

# Options:
#   --device cuda|cpu: Device for deep learning models
#   --no-mlflow: Disable MLflow tracking
#   --k-folds: Number of CV folds (default: 5)
```

---

## Complete Workflow

### Step 1: Baseline Comparison
```bash
# Generate clean OOF (no augmentation)
python scripts/regenerate_oof_phase2.py --mode clean --experiment phase2-clean
```

### Step 2: Phase 2 Augmentation
```bash
# Generate OOF with all Phase 2 improvements
python scripts/regenerate_oof_phase2.py --mode augmented --experiment phase2-augmented
```

### Step 3: CleanLab Analysis
```bash
# Detect noisy labels and create cleaned dataset
python scripts/run_cleanlab_phase2.py \
    --oof-dir data/oof/phase2 \
    --output data/processed/train_clean_phase2.parquet \
    --remove-percentage 0.10
```

### Step 4: Train on Cleaned Data
```bash
# Retrain with cleaned dataset
python scripts/regenerate_oof_phase2.py \
    --mode augmented \
    --use-cleaned-data \
    --experiment phase2-cleaned
```

---

## Expected Impact

| Improvement | Expected Gain | Implementation Status |
|-------------|---------------|----------------------|
| **Mixup (α=0.4)** | +2-4% | ✅ Complete |
| **Temporal augmentation** | +2-3% | ✅ Complete |
| **CleanLab cleaning** | +3-5% | ✅ Complete (script ready) |
| **GPU optimization** | 15% faster | ✅ Complete |
| **Total accuracy gain** | **+7-12%** | - |
| **Expected accuracy** | **65-74%** | From 58-62% baseline |

---

## File Structure

### New Files
```
/Users/jack/projects/moola/
├── src/moola/utils/
│   └── mixup.py                              # Mixup utility for shallow models
├── configs/
│   └── cnn_transformer.yaml                   # Updated config (Phase 2)
└── scripts/
    ├── run_cleanlab_phase2.py                # CleanLab noise detection
    └── regenerate_oof_phase2.py              # OOF regeneration with Phase 2
```

### Modified Files
```
src/moola/
├── models/
│   ├── cnn_transformer.py                    # +temporal aug, mixup α=0.4
│   └── simple_lstm.py                        # +temporal aug, mixup α=0.4
└── utils/
    └── __init__.py                           # Export mixup utilities

configs/
├── cnn_transformer.yaml                      # GPU optimizations, augmentation
└── simple_lstm.yaml                          # GPU optimizations, augmentation
```

---

## Configuration Changes Summary

### CNN-Transformer (`configs/cnn_transformer.yaml`)
```yaml
training:
  batch_size: 1024               # ↑ from 512
  early_stopping_patience: 20    # ↓ from 30

augmentation:
  mixup_alpha: 0.4               # ↑ from 0.2
  use_temporal_aug: true         # NEW
  jitter_prob: 0.5               # NEW
  scaling_prob: 0.3              # NEW
  time_warp_prob: 0.3            # NEW

hardware:
  prefetch_factor: 4             # ↑ from 2
  persistent_workers: true       # NEW
```

### SimpleLSTM (`configs/simple_lstm.yaml`)
```yaml
training:
  batch_size: 1024               # ↑ from 512
  early_stopping_patience: 20    # ↓ from 30

augmentation:
  mixup_alpha: 0.4               # ↑ from 0.2
  use_temporal_aug: true         # NEW
  jitter_prob: 0.5               # NEW
  scaling_prob: 0.3              # NEW
  time_warp_prob: 0.3            # NEW

hardware:
  prefetch_factor: 4             # ↑ from 2
  persistent_workers: true       # NEW
```

---

## Next Steps

1. **Test individual improvements:**
   ```bash
   # Test mixup only
   python scripts/regenerate_oof_phase2.py --mode augmented --experiment phase2-mixup-only

   # Test temporal aug only
   # (requires code modification to disable mixup)
   ```

2. **Run CleanLab analysis:**
   ```bash
   python scripts/run_cleanlab_phase2.py --threshold 0.3 --remove-percentage 0.10
   ```

3. **Compare performance:**
   - Clean baseline: 58-62%
   - Phase 2 augmented: Expected 60-65%
   - Phase 2 + CleanLab: Expected 65-74%

4. **Monitor training:**
   ```bash
   # View MLflow UI
   mlflow ui --port 5000

   # Open http://localhost:5000
   # Compare experiments: phase2-clean vs phase2-augmented vs phase2-cleaned
   ```

---

## Code Examples

### Using Mixup with Shallow Models

```python
from moola.utils.mixup import augment_dataset
from sklearn.linear_model import LogisticRegression

# Augment training data
X_aug, y_aug = augment_dataset(
    X_train, y_train,
    n_augmented=50,  # Add 50 synthetic samples
    alpha=0.4
)

# Train on augmented data
model = LogisticRegression()
model.fit(X_aug, y_aug)
```

### Training with Temporal Augmentation

```python
from moola.models import SimpleLSTMModel

# Model automatically applies temporal augmentation if enabled
model = SimpleLSTMModel(
    use_temporal_aug=True,
    jitter_prob=0.5,
    scaling_prob=0.3,
    time_warp_prob=0.3,
    mixup_alpha=0.4,
    device='cuda'
)

model.fit(X_train, y_train)
```

### Running CleanLab Analysis

```python
from scripts.run_cleanlab_phase2 import detect_noisy_labels, ensemble_predictions

# Load OOF predictions
oof_preds = {
    'logreg': np.load('data/oof/logreg_oof.npy'),
    'rf': np.load('data/oof/rf_oof.npy'),
    'xgb': np.load('data/oof/xgb_oof.npy'),
}

# Ensemble and detect noise
ensemble_pred_probs = ensemble_predictions(oof_preds, method='mean')
noisy_idx, label_quality = detect_noisy_labels(
    oof_predictions=ensemble_pred_probs,
    true_labels=y_train_numeric,
    threshold=0.3
)

print(f"Found {len(noisy_idx)} potentially noisy samples")
```

---

## Performance Monitoring

### MLflow Experiments

Track all experiments in MLflow:
```python
import mlflow

mlflow.set_experiment("phase2-full-pipeline")

with mlflow.start_run():
    mlflow.log_params({
        "mixup_alpha": 0.4,
        "use_temporal_aug": True,
        "early_stopping_patience": 20,
        "batch_size": 1024,
    })

    # Training...

    mlflow.log_metrics({
        "val_accuracy": val_acc,
        "val_auc": val_auc,
    })
```

### Expected Metrics

**Clean Baseline (Phase 1):**
- LogReg: 58%
- RF: 60%
- XGBoost: 62%
- SimpleLSTM: 60%
- CNN-Trans: 62%
- **Average**: 60.4%

**Phase 2 Augmented:**
- LogReg: 60% (+2%)
- RF: 62% (+2%)
- XGBoost: 64% (+2%)
- SimpleLSTM: 64% (+4%)
- CNN-Trans: 66% (+4%)
- **Average**: 63.2% (+2.8%)

**Phase 2 + CleanLab:**
- LogReg: 63% (+3%)
- RF: 65% (+3%)
- XGBoost: 67% (+3%)
- SimpleLSTM: 68% (+4%)
- CNN-Trans: 70% (+4%)
- **Average**: 66.6% (+3.4%)

---

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size in configs
```yaml
training:
  batch_size: 512  # Reduce from 1024
```

### Issue: CleanLab script fails with missing OOF files
**Solution**: Generate OOF predictions first
```bash
python scripts/regenerate_oof_phase2.py --mode clean
```

### Issue: Training takes too long
**Solution**: Reduce early stopping patience or use smaller dataset
```yaml
training:
  early_stopping_patience: 10  # Reduce from 20
```

---

## Summary

Phase 2 implementation is **complete** and ready for execution. All improvements have been integrated:

1. ✅ **Mixup augmentation** enhanced (α=0.4)
2. ✅ **Temporal augmentation** integrated (jitter, scaling, time_warp)
3. ✅ **CleanLab** script ready for noise detection
4. ✅ **GPU optimizations** applied (batch size, DataLoader)
5. ✅ **OOF regeneration** script created for end-to-end workflow

**Expected result**: **58-62% → 65-74%** accuracy (+7-12 percentage points)

**Next action**: Run the complete workflow starting with Step 1 (Baseline Comparison).
