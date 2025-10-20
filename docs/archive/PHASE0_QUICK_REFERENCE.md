# Phase 0 Quick Reference - Model & Split Strategy

## Report Location
`/Users/jack/projects/moola/PHASE0_CODE_SURVEY.md` (661 lines, comprehensive analysis)

## TL;DR - Active vs Experimental

### PRODUCTION ACTIVE (Keep in models/)
- **SimpleLSTM** (921 params) - baseline, well-integrated
- **LogReg, RF, XGBoost** - stacking base learners
- **Stack** - meta-learner
- **EnhancedSimpleLSTM** - variant (note: NOT registered in __init__.py!)

### EXPERIMENTAL/PRE-TRAINING (Move to models_extras/)
- BiLSTM Masked Autoencoder (pre-training only)
- Feature-Aware BiLSTM MA (experimental variant)
- BiLSTM Autoencoder (deprecated)
- TS-TCC (contrastive pre-training)
- CNN-Transformer (1321 LOC, 56K params - over-parameterized)
- RWKV-TS (639 LOC, 409K params - severely over-parameterized)

## CRITICAL ISSUE: Split Strategy

**Current State:** RANDOM STRATIFIED K-FOLD (bad for time series)
```python
# Location: src/moola/utils/splits.py line 40
StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
```

**Problem:** LOOK-AHEAD BIAS in financial time series
- Random shuffling breaks temporal coherence
- Past prices should not be mixed with future prices
- This is especially dangerous for financial data

**Solution Needed:** Forward-chaining with embargo
```python
# Should use:
TimeSeriesSplit()  # or custom forward-chaining
# Plus: 10% embargo period between train/test
```

**Affected Code Locations:**
1. `src/moola/utils/splits.py` - make_splits()
2. `src/moola/cli.py` line 254-262 - train/test split
3. `src/moola/cli.py` line 441-442 - K-fold evaluation
4. `src/moola/models/simple_lstm.py` line 444-450 - validation split

## File Organization Recommendations

### Keep in src/moola/models/
```
base.py
simple_lstm.py           [ACTIVE]
enhanced_simple_lstm.py  [ACTIVE variant, needs registry]
logreg.py               [ACTIVE baseline]
rf.py                   [ACTIVE baseline]
xgb.py                  [ACTIVE baseline]
stack.py                [ACTIVE meta-learner]
__init__.py
```

### Move to src/moola/models_extras/ OR src/moola/pretraining/
```
bilstm_autoencoder.py
bilstm_masked_autoencoder.py
feature_aware_bilstm_masked_autoencoder.py
ts_tcc.py
cnn_transformer.py          [if moving experimental models]
rwkv_ts.py                  [if moving experimental models]
```

## Data Pipeline Quick Overview

```
Raw Data
  ↓
data/processed/train.parquet
  ↓
DualInputDataProcessor
  ├─ OHLC: 105×4 (always)
  ├─ Features: 25-30 optimized + 21 multi-scale (optional)
  └─ Augmentation: pseudo-samples (optional)
  ↓
Model Selection
  ├─ simple_lstm (DEFAULT)
  ├─ logreg/rf/xgb (baselines)
  ├─ cnn_transformer (experimental)
  └─ rwkv_ts (experimental)
  ↓
artifacts/models/{model}/model.pkl
```

## Key Decision Points

1. **SimpleLSTM vs EnhancedSimpleLSTM**
   - SimpleLSTM: registered, widely used (Oct 18 01:42 last modified)
   - EnhancedSimpleLSTM: not registered, variant with feature fusion (Oct 18 01:35)
   - DECISION: Which should be production baseline?

2. **CNN-Transformer & RWKV-TS**
   - Both in registry but over-parameterized (56K and 409K vs 921)
   - DECISION: Keep for experimentation or move to models_extras/?

3. **Pre-training Strategy**
   - BiLSTM masked autoencoder working
   - TS-TCC as alternative
   - DECISION: Continue investing in transfer learning or focus on feature engineering?

## Pre-training Models (Not for Inference)

These models are **ONLY used for pre-training**, never for inference:
- BiLSTMMaskedAutoencoder
- FeatureAwareBiLSTMMaskedAutoencoder
- TS-TCC

They should be accessible via CLI but separate from the inference model registry.

## CLI Usage

### Training
```bash
python -m moola.cli train --model simple_lstm --device cpu
python -m moola.cli train --model simple_lstm --device cuda --use-engineered-features
```

### Pre-training
```bash
python -m moola.cli pretrain-bilstm --device cuda --epochs 50
python -m moola.cli pretrain-tcc --device cuda --epochs 100
```

### Evaluation
```bash
python -m moola.cli evaluate --model simple_lstm --device cuda
```

## Recent Development Focus

Last 7 days of commits show focus on:
1. SimpleLSTM optimization and variants
2. Pseudo-sample augmentation (controlled)
3. Pre-trained encoder integration
4. Feature engineering for 98-sample dataset
5. Monitoring infrastructure

This is the RIGHT direction - focusing on SimpleLSTM maturity.

## Next Steps (Recommended)

### Phase 0 Complete
✓ Model inventory created
✓ Split strategy issue identified
✓ File organization plan drafted

### Phase 1 (Split Strategy Fix)
1. [ ] Update splits.py to use forward-chaining
2. [ ] Add embargo logic
3. [ ] Update CLI train/evaluate commands
4. [ ] Test with existing data

### Phase 2 (Model Organization)
1. [ ] Create models_extras/ directory
2. [ ] Move experimental models
3. [ ] Update imports
4. [ ] Update CLI to reference new paths

### Phase 3 (Registry Cleanup)
1. [ ] Register EnhancedSimpleLSTM (or decide to deprecate)
2. [ ] Remove pre-training models from production registry
3. [ ] Create separate pretraining_models registry

---

**Report Generated:** 2025-10-18
**Completeness:** Comprehensive (11 sections + 4 appendices)
**Focus Areas:** Model classification, split strategy analysis, refactoring roadmap
