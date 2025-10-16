# CNN-Transformer Encoder Fixes - Implementation Guide

## 🎯 Problem Summary

**Critical Bug:** Pre-trained encoder weights were loaded but NOT frozen, causing them to be destroyed during fine-tuning. This resulted in:
- Class 1 (Retracement) accuracy: **0%** (complete collapse)
- Overall accuracy: **57%** (below baseline)
- Training stopping early at epoch 21-28

## ✅ Fixes Implemented

### 1. Encoder Freezing & Gradual Unfreezing

**File:** `src/moola/models/cnn_transformer.py`

Added methods:
- `freeze_encoder()`: Freeze CNN blocks + Transformer layers
- `unfreeze_encoder_gradual(stage)`: Progressive unfreezing schedule

**Schedule:**
- Epochs 0-10: Encoder frozen, train classification head only
- Epoch 10: Unfreeze last transformer layer
- Epoch 20: Unfreeze all transformer layers
- Epoch 30: Unfreeze CNN blocks (full fine-tuning)

### 2. Configuration Updates

**File:** `src/moola/config/training_config.py`

New constants:
```python
CNNTR_N_EPOCHS = 80                    # Up from 60
CNNTR_EARLY_STOPPING_PATIENCE = 30    # Up from 20
CNNTR_FREEZE_EPOCHS = 10               # Freeze duration
CNNTR_GRADUAL_UNFREEZE = True          # Enable schedule
CNNTR_LOSS_ALPHA_CLASSIFICATION = 1.0  # Full weight to classification
CNNTR_LOSS_BETA_POINTER = 0.0          # Disable multi-task initially
```

### 3. Validation Utilities

**File:** `src/moola/validation/training_validator.py`

New functions:
- `validate_encoder_loading()`: Verify weights loaded correctly
- `detect_class_collapse()`: Early warning for class imbalance
- `verify_gradient_flow()`: Check frozen/trainable params

### 4. Enhanced Training Loop

**File:** `src/moola/models/cnn_transformer.py` (fit method)

Added:
- Automatic encoder freezing after loading pre-trained weights
- Gradual unfreezing at specified epochs
- Per-class accuracy tracking (every 10 epochs)
- Class collapse detection and warnings

### 5. New Training Script

**File:** `src/moola/scripts/train_cnn_pretrained_fixed.py`

Comprehensive training pipeline with:
- Stratified K-fold cross-validation
- Encoder loading verification
- Gradient flow monitoring
- Per-class accuracy reporting
- OOF predictions generation

### 6. Testing Script

**File:** `src/moola/scripts/test_encoder_fixes.py`

Validation tests:
- Encoder weight loading verification
- Freezing correctness check
- Gradual unfreezing progression
- Training with frozen encoder

## 🚀 Quick Start

### Step 1: Verify Encoder Exists

```bash
# Check if pre-trained encoder is available
ls -lh data/artifacts/pretrained/encoder_weights.pt

# If not, run TS-TCC pre-training first:
python -m moola.cli pretrain-tcc --device cuda --epochs 100
```

### Step 2: Run Tests (Recommended)

```bash
# Quick validation (no training test)
python -m moola.scripts.test_encoder_fixes \
    --encoder-path data/artifacts/pretrained/encoder_weights.pt \
    --device cuda \
    --skip-training-test

# Full validation (includes training test, slower)
python -m moola.scripts.test_encoder_fixes \
    --encoder-path data/artifacts/pretrained/encoder_weights.pt \
    --device cuda
```

**Expected output:**
```
TEST SUMMARY
================================================================================
  Encoder Loading: ✓ PASSED
  Encoder Freezing: ✓ PASSED
  Gradual Unfreezing: ✓ PASSED
  Training with Frozen Encoder: ✓ PASSED

Total: 4/4 tests passed

🎉 ALL TESTS PASSED! Encoder fixes are working correctly.
```

### Step 3: Train with Fixed Pipeline

```bash
# Train on GPU with all fixes
python -m moola.scripts.train_cnn_pretrained_fixed \
    --data-path data/processed/train_clean.parquet \
    --encoder-path data/artifacts/pretrained/encoder_weights.pt \
    --output-dir data/artifacts/models/cnn_transformer_fixed \
    --device cuda \
    --seed 1337 \
    --folds 5 \
    --max-epochs 80 \
    --patience 30

# Train on CPU (slower)
python -m moola.scripts.train_cnn_pretrained_fixed \
    --data-path data/processed/train_clean.parquet \
    --encoder-path data/artifacts/pretrained/encoder_weights.pt \
    --output-dir data/artifacts/models/cnn_transformer_fixed \
    --device cpu
```

## 📊 Expected Results

### Before Fixes
- Overall Accuracy: **57%**
- Class 0 (Consolidation): 100% (predicting everything as Class 0)
- Class 1 (Retracement): **0%** (complete collapse)
- Training: Stops at epoch 21-28

### After Fixes
- Overall Accuracy: **62-67%**
- Class 0 (Consolidation): 70-80%
- Class 1 (Retracement): **30-45%** (up from 0%!)
- Training: 40-50 epochs

## 🔍 Debugging Guide

### Issue: Encoder weights not loading

```bash
# Verify encoder file format
python -c "import torch; print(torch.load('data/artifacts/pretrained/encoder_weights.pt').keys())"

# Should print: dict_keys(['encoder_state_dict', 'hyperparams', ...])
```

### Issue: Class collapse still occurring

Check training logs for:
```
[CLASS BALANCE] Epoch 15 - Per-class Accuracy:
  ⚠️  Class 1 (retracement): 0.0% (0/15) - COLLAPSE DETECTED!
```

Solutions:
1. Increase freeze epochs: `--freeze-epochs 20`
2. Use class weighting (modify `FOCAL_LOSS_ALPHA` in config)
3. More pre-training data/epochs

### Issue: Unfreezing not working

Run gradient flow check:
```python
from moola.validation.training_validator import verify_gradient_flow
verify_gradient_flow(model.model, phase="epoch_15")
```

Should show increasing trainable param count across epochs.

## 📁 Generated Artifacts

After training, you'll find:

```
data/artifacts/models/cnn_transformer_fixed/
├── fold_1.pt                    # Model weights for fold 1
├── fold_2.pt                    # Model weights for fold 2
├── fold_3.pt                    # Model weights for fold 3
├── fold_4.pt                    # Model weights for fold 4
├── fold_5.pt                    # Model weights for fold 5
├── oof_predictions.parquet      # Out-of-fold predictions
└── training_summary.json        # Overall results and config
```

## 🔬 Advanced Usage

### Custom Freezing Schedule

Edit `src/moola/config/training_config.py`:

```python
CNNTR_UNFREEZE_SCHEDULE = {
    "stage1_epoch": 15,  # Delay unfreezing
    "stage2_epoch": 30,
    "stage3_epoch": 45,
}
```

### Enable Multi-Task Learning

**Warning:** Not recommended for small datasets (< 200 samples)

```bash
python -m moola.scripts.train_cnn_pretrained_fixed \
    --enable-multitask \
    --max-epochs 100
```

### Integration with Existing Pipeline

Add to `scripts/train_full_pipeline.py`:

```python
# Phase 2: Train CNN-Transformer with pre-trained encoder
run_command([
    sys.executable, "-m", "moola.scripts.train_cnn_pretrained_fixed",
    "--data-path", str(PROCESSED_DIR / "train_clean.parquet"),
    "--encoder-path", str(MODELS_DIR / "ts_tcc" / "pretrained_encoder.pt"),
    "--output-dir", str(MODELS_DIR / "cnn_transformer_fixed"),
    "--device", args.device,
    "--seed", str(args.seed),
], "CNN-Transformer training with encoder fixes")
```

## 📚 Code Architecture

### Training Flow

```
1. Load pre-trained encoder weights
   ↓
2. Freeze encoder (CNN + Transformer)
   ↓
3. Train classification head (epochs 0-10)
   ↓
4. Unfreeze last transformer layer (epoch 10)
   ↓
5. Unfreeze all transformer layers (epoch 20)
   ↓
6. Unfreeze CNN blocks (epoch 30)
   ↓
7. Full fine-tuning (epochs 30+)
   ↓
8. Early stopping or max epochs reached
```

### Key Classes and Methods

**CnnTransformerModel:**
- `load_pretrained_encoder(path)`: Load SSL weights
- `freeze_encoder()`: Freeze encoder layers
- `unfreeze_encoder_gradual(stage)`: Progressive unfreezing
- `fit(X, y)`: Training with automatic freezing/unfreezing

**Validation:**
- `validate_encoder_loading(model, path)`: Verify weight loading
- `detect_class_collapse(preds, labels, epoch)`: Class imbalance detection
- `verify_gradient_flow(model, phase)`: Gradient debugging

## 🐛 Common Issues

### 1. CUDA Out of Memory

```bash
# Reduce batch size
# Edit src/moola/config/training_config.py
DEFAULT_BATCH_SIZE = 256  # Down from 512
```

### 2. Slow Training on CPU

```bash
# Reduce workers
# Edit training config
DEFAULT_NUM_WORKERS = 0  # Down from 16
```

### 3. Early Stopping Too Aggressive

```bash
# Increase patience
python -m moola.scripts.train_cnn_pretrained_fixed --patience 50
```

## 📈 Monitoring Training

Watch for these log messages:

```
[SSL] Loaded 142 pre-trained layers              # ✓ Encoder loaded
[FREEZE] Encoder frozen: 142 frozen params       # ✓ Freezing active
[UNFREEZE] Stage 1: Last transformer layer...    # ✓ Unfreezing started
[CLASS BALANCE] Class 1: 35.2% (5/14)           # ✓ Class 1 learning!
```

## 🎓 Next Steps

1. **Verify fixes work:** Run test script
2. **Train model:** Use fixed training script
3. **Analyze results:** Check per-class accuracy improvement
4. **If Class 1 still low:**
   - Collect more retracement samples
   - Try different SSL pre-training data
   - Experiment with class weighting
   - Increase freeze duration

## 📞 Support

If issues persist:
1. Check logs for error messages
2. Run validation script with `--skip-training-test`
3. Verify encoder architecture matches (CNN channels/kernels)
4. Review training logs for class collapse warnings

## 🔗 References

- SSL Pre-training: `src/moola/models/ts_tcc.py`
- Training config: `src/moola/config/training_config.py`
- Original model: `src/moola/models/cnn_transformer.py`
- Validation utils: `src/moola/validation/training_validator.py`
