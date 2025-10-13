# Moola

ML trading pattern recognition system using expansion indices for precise feature extraction and stacking ensemble for robust predictions.

## Features

- **Expansion Index Integration**: Extracts features from variable-length pattern regions (5-23 bars)
- **ICT-Aligned Features**: Market structure, liquidity zones, fair value gaps, order blocks
- **5-Model Stacking Ensemble**: XGBoost, Random Forest, Logistic Regression, CNN-Transformer, RWKV-TS with Random Forest meta-learner
- **2-Class Binary Classification**: Consolidation vs Retracement pattern detection
- **RunPod GPU Deployment**: Optimized for cloud training with RTX 4090 support

## Quick Start

### Local Training
```bash
# Train individual models
python -m moola.cli oof --model xgb --device cpu --seed 1337
python -m moola.cli oof --model logreg --device cpu --seed 1337

# Train stacking ensemble
python -m moola.cli stack-train --stacker rf --seed 1337
```

### RunPod Deployment
```bash
# Deploy from local machine
.runpod/deploy-fast.sh

# Train on RunPod
ssh runpod
cd /workspace
bash scripts/start.sh --train
```

## Dataset

**Current (2-Class)**: 115 CleanLab-cleaned samples with expansion indices
- Consolidation: 65 samples (56.5%)
- Retracement: 50 samples (43.5%)
- **Archived**: 19 reversal samples → `data/processed/reversal_holdout.parquet`
- **Backup**: 3-class dataset (134 samples) → `data/processed/train_3class_backup.parquet`

## Performance

**Stacking Ensemble (2-Class)**:
- **Accuracy**: 60.9% (21.8% above 50% binary baseline)
- **F1 Score**: 0.592 (+50% improvement over 3-class baseline of 0.394)
- **ECE**: 0.128 (reasonable calibration)
- **Log Loss**: 0.716
- **Cross-Validation**: 5-fold stratified (folds: 65.2%, 65.2%, 52.2%, 52.2%, 69.6%)

**Base Models (Average Performance)**:
- Logistic Regression: ~53% (best individual)
- RWKV-TS: ~50%
- XGBoost: ~44% (high variance)
- Random Forest: ~43%
- CNN-Transformer: 43.5% (perfect stability)

**Key Achievement**: Stacking ensemble provides +7.9% improvement over best individual model

## Architecture

```
data/processed/train.parquet (115 samples with expansion_start/end)
    ↓
src/moola/features/price_action_features.py (extract from expansion regions)
    ↓
src/moola/models/ (5 base models: logreg, rf, xgb, rwkv_ts, cnn_transformer)
    ↓
data/artifacts/oof/ (Out-of-Fold predictions: 115 × 2 per model)
    ↓
src/moola/pipelines/stack_train.py (Random Forest meta-learner)
    ↓
data/artifacts/models/stack/stack.pkl (Stacking ensemble)
```

### Deployment Verification

```bash
# Verify deployment integrity
python scripts/verify_deployment.py

# Output:
# ✓ Data configuration (2-class, 115 samples)
# ✓ Model artifacts (5 OOF + stack model)
# ✓ Inference pipeline (predictions functional)
```

## Requirements

- Python 3.10+
- PyTorch (for deep learning models)
- XGBoost 2.0+
- See `requirements-runpod.txt` for minimal deployment dependencies

## License

MIT
