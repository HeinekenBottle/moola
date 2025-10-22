# Stones-Only Reset Progress

**Date:** 2025-10-22  
**Branch:** `reset/stones-only`  
**Tag:** `pre-clean-legacy` (recovery point)

## Objective

Rebuild Moola into a minimal, production-focused training repository centered on the Stones doctrine (Jade/Opal/Sapphire models). Remove all legacy code, experimental pipelines, and unused models.

## Progress Summary

### ✅ Stage 1: Protect Existing Work (COMPLETE)

Created safety checkpoints before destructive operations:

- ✅ Created feature branch: `reset/stones-only`
- ✅ Tagged current state: `pre-clean-legacy`
- ✅ Created git bundle backup: `archive/pre-clean-legacy.bundle` (1.9M)
- ✅ Archived RunPod artifacts: `archive/artifacts_runpod/`

**Recovery command if needed:**
```bash
git checkout pre-clean-legacy
# or
git bundle unbundle archive/pre-clean-legacy.bundle
```

### ✅ Stage 2: Trace Dependencies (COMPLETE)

Identified code actually used by Stones models:

**Jade/Opal/Sapphire Dependencies:**
- `src/moola/models/jade.py` - Production model with full training logic
- `src/moola/models/base.py` - Model interface
- `src/moola/models/registry.py` - Model selection
- `src/moola/utils/augmentation.py` - Mixup/CutMix
- `src/moola/utils/data_validation.py` - Data validation
- `src/moola/utils/early_stopping.py` - Early stopping
- `src/moola/utils/focal_loss.py` - Focal loss
- `src/moola/utils/model_diagnostics.py` - Model diagnostics
- `src/moola/utils/seeds.py` - Reproducibility
- `src/moola/utils/temporal_augmentation.py` - Temporal augmentation
- `src/moola/metrics/hit_metrics.py` - Hit@K metrics
- `src/moola/data/pointer_transforms.py` - Pointer encoding
- `src/moola/data/splits.py` - Data splitting
- `src/moola/data/load.py` - Data loading
- `src/moola/data/storage_11d.py` - 11D feature storage
- `src/moola/data/feature_11d_integration.py` - Feature integration
- `src/moola/pretraining/masked_lstm_pretrain.py` - BiLSTM MAE (for Opal/Sapphire)
- `src/moola/pretraining/data_augmentation.py` - Pretraining augmentation

**Opal/Sapphire Specific:**
- Require pretrained encoder: `artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt`
- Use transfer learning with frozen/adaptive fine-tuning

### ✅ Stage 3: Purge Unused Code (COMPLETE)

Deleted 30 files:

**Root-level scripts (6 files):**
- ❌ `four_stage_pipeline.py`
- ❌ `full_training_pipeline.py`
- ❌ `full_training_real.py`
- ❌ `real_training_pipeline.py`
- ❌ `run_pointer_favoring_training.py`
- ❌ `simple_train.py`

**Model files (8 files):**
- ❌ `src/moola/models/enhanced_simple_lstm.py`
- ❌ `src/moola/models/simple_lstm.py`
- ❌ `src/moola/models/logreg.py`
- ❌ `src/moola/models/rf.py`
- ❌ `src/moola/models/xgb.py`
- ❌ `src/moola/models/stack.py`
- ❌ `src/moola/models/stones_ensemble.py`
- ❌ `src/moola/models/jade_compact.py` (redundant with jade.py)

**Data pipeline files (5 files):**
- ❌ `src/moola/data/dual_input_pipeline.py`
- ❌ `src/moola/data/enhanced_pipeline.py`
- ❌ `src/moola/data/latent_mixup.py`
- ❌ `src/moola/data/optimized_pipeline.py`
- ❌ `src/moola/data/pretrain_pipeline.py`

**Pretraining files (2 files):**
- ❌ `src/moola/pretraining/feature_aware_masked_lstm_pretrain.py`
- ❌ `src/moola/pretraining/multitask_pretrain.py`

**CLI files (1 file):**
- ❌ `src/moola/cli_registry_patch.py`

**Test files (8 files):**
- ❌ `tests/test_augmentation.py`
- ❌ `tests/test_bilstm_masked_autoencoder.py`
- ❌ `tests/test_data_infra.py`
- ❌ `tests/test_metrics.py`
- ❌ `tests/test_pipeline.py`
- ❌ `tests/test_relative_transform.py`
- ❌ `tests/test_uncertainty_integration.py`
- ❌ `tests/test_uncertainty_weighted_loss.py`

**Kept tests:**
- ✅ `tests/test_jade_model.py` - Jade model tests
- ✅ `tests/test_stones_augmentation.py` - Stones augmentation tests
- ✅ `tests/test_import.py` - Import validation

### 🔄 Stage 4: Verify Integrity (IN PROGRESS)

**Current Status:**

1. ✅ Import validation: `from moola.models import JadeModel` works
2. ⚠️ Tests failing: `test_jade_model.py` has failures due to:
   - `JadeModel` is `nn.Module`, not `BaseModel` (no `seed` parameter)
   - `UncertaintyWeightedLoss` initialization values changed
   - Architecture tests expect different interface

**Issues Found:**

1. **CLI references to deleted models:**
   - `src/moola/cli.py` has 18+ references to `enhanced_simple_lstm`
   - `src/moola/config/feature_aware_config.py` references `enhanced_simple_lstm`
   - `src/moola/utils/feature_aware_utils.py` imports `EnhancedSimpleLSTMModel`

2. **Test compatibility:**
   - Tests expect `BaseModel` interface
   - `JadeModel` in `jade.py` is `nn.Module` (not `BaseModel`)
   - Need to decide: update tests or update model interface

**Next Steps:**

1. Clean up CLI references to deleted models
2. Fix or update tests to match current `JadeModel` interface
3. Verify Jade dry-run works
4. Check for remaining broken references

### ⏳ Stage 5: Document New Structure (NOT STARTED)

Will update:
- `README.md` - Stones-only training pipeline
- `DATA_NORMALIZATION.md` (NEW) - Price relevance scaling
- `.gitignore` - Exclude archive/

## Current Repository State

**Stones Models:**
- ✅ Jade (`src/moola/models/jade.py`) - Production BiLSTM
- ✅ Opal (config: `configs/model/opal.yaml`) - Adaptive fine-tuning
- ✅ Sapphire (config: `configs/model/sapphire.yaml`) - Frozen encoder

**Essential Files Kept:**
```
src/moola/
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── jade.py
│   └── registry.py
├── data/
│   ├── load.py
│   ├── splits.py
│   ├── pointer_transforms.py
│   ├── storage_11d.py
│   ├── feature_11d_integration.py
│   └── temporal_augmentation.py
├── pretraining/
│   ├── masked_lstm_pretrain.py
│   └── data_augmentation.py
├── utils/
│   ├── augmentation.py
│   ├── data_validation.py
│   ├── early_stopping.py
│   ├── focal_loss.py
│   ├── model_diagnostics.py
│   ├── seeds.py
│   └── temporal_augmentation.py
├── metrics/
│   └── hit_metrics.py
└── cli.py (needs cleanup)
```

**Tests Kept:**
```
tests/
├── test_jade_model.py (needs fixes)
├── test_stones_augmentation.py
└── test_import.py
```

## Rollback Instructions

If you need to revert all changes:

```bash
# Option 1: Reset to tagged state
git reset --hard pre-clean-legacy
git clean -fd

# Option 2: Restore from bundle
git bundle unbundle archive/pre-clean-legacy.bundle
git checkout main

# Option 3: Delete branch and start over
git checkout main
git branch -D reset/stones-only
```

## Next Actions

1. **Clean up CLI references:**
   - Remove `enhanced_simple_lstm` references from `cli.py`
   - Remove `simple_lstm` references
   - Update model selection to only allow Jade/Opal/Sapphire

2. **Fix tests:**
   - Update `test_jade_model.py` to match current `JadeModel` interface
   - Or update `JadeModel` to extend `BaseModel`

3. **Verify integrity:**
   - Run remaining tests
   - Test Jade training dry-run
   - Check for broken imports

4. **Document:**
   - Update README.md
   - Create DATA_NORMALIZATION.md
   - Update .gitignore

## Files Changed (30 deletions, 1 modification)

```
D  four_stage_pipeline.py
D  full_training_pipeline.py
D  full_training_real.py
D  real_training_pipeline.py
D  run_pointer_favoring_training.py
D  simple_train.py
D  src/moola/cli_registry_patch.py
D  src/moola/data/dual_input_pipeline.py
D  src/moola/data/enhanced_pipeline.py
D  src/moola/data/latent_mixup.py
D  src/moola/data/optimized_pipeline.py
D  src/moola/data/pretrain_pipeline.py
D  src/moola/models/enhanced_simple_lstm.py
D  src/moola/models/jade_compact.py
D  src/moola/models/logreg.py
D  src/moola/models/rf.py
D  src/moola/models/simple_lstm.py
D  src/moola/models/stack.py
D  src/moola/models/stones_ensemble.py
D  src/moola/models/xgb.py
D  src/moola/pretraining/feature_aware_masked_lstm_pretrain.py
D  src/moola/pretraining/multitask_pretrain.py
D  tests/test_augmentation.py
D  tests/test_bilstm_masked_autoencoder.py
D  tests/test_data_infra.py
D  tests/test_metrics.py
D  tests/test_pipeline.py
D  tests/test_relative_transform.py
D  tests/test_uncertainty_integration.py
D  tests/test_uncertainty_weighted_loss.py
M  src/moola/models/__init__.py (updated import)
?? archive/ (git-ignored)
```

