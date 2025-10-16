# Moola Codebase Cleanup & Modernization Plan

**Status:** ACTIVE
**Date:** 2025-10-16
**Scope:** Context coherence, model-data integrity, documentation consolidation

---

## Executive Summary

The moola codebase has accumulated significant bloat and context issues from multiple iterations:
- **5,638 lines** of overlapping RunPod documentation in `.runpod/` alone
- **16+ root-level markdown files** creating documentation noise
- **Scattered magic numbers** across model/training files
- **No centralized configuration** for reproducibility
- **Missing data/model validation guardrails** that allowed past issues

This plan delivers systematic cleanup in 5 phases with clear success criteria.

---

## Phase 1: DOCUMENTATION CONSOLIDATION (Priority: CRITICAL)

### Problem
- `.runpod/AUDIT_SUMMARY.md` (378 lines) vs `.runpod/CRITICAL_INFRASTRUCTURE_AUDIT.md` (1014 lines) - duplicate content
- `.runpod/README.md` + `.runpod/README_AUDIT.md` - overlapping scope
- Root level docs scattered: `PRETRAINED_ENCODER_*.md`, `ENCODER_*.md`, `IMPLEMENTATION_*.md`, etc.
- Maintenance burden: 16 markdown files in root, 16 in `.runpod/`

### Solution

#### A. Consolidate `.runpod/` into single deployments guide
**Target structure:**
```
.runpod/
├── DEPLOYMENT_GUIDE.md       (NEW: How to deploy/run)
│   - Merge: README.md, QUICKSTART.md, QUICK_START.md, SIMPLE_WORKFLOW.md
│   - Keep: Essential runpod setup, docker, worker config, examples
│   - Delete duplicates
│
├── TROUBLESHOOTING.md         (NEW: Known issues & fixes)
│   - Merge: CRITICAL_INFRASTRUCTURE_AUDIT.md, AUDIT_SUMMARY.md, README_AUDIT.md
│   - Keep: Real issues, root causes, solutions (not audit status updates)
│   - Delete: Timestamps, historical status
│
├── QUICK_REFERENCE.md         (RENAME: Infrastructure reference)
│   - Merge: STORAGE_BREAKDOWN.md, TEMPLATE_PACKAGES.md
│   - Keep: Architecture, sizing, package specs
│   - Update: Remove stale info
│
└── DELETE (REDUNDANT):
    - AUDIT_SUMMARY.md (2025-10-16 update, subsumed into TROUBLESHOOTING)
    - README_AUDIT.md (subsumed into TROUBLESHOOTING)
    - DEPLOYMENT_AUDIT_REPORT.md (stale, subsumed)
    - MIGRATION_GUIDE.md (stale, runpod setup has moved to DEPLOYMENT_GUIDE)
    - OPTIMIZED_DEPLOYMENT.md (outdated optimization notes)
    - WORKFLOW_OPTIMIZATION.md (outdated)
    - README.md (consolidate into DEPLOYMENT_GUIDE)
    - QUICKSTART.md (consolidate into DEPLOYMENT_GUIDE)
    - QUICK_START.md (consolidate into DEPLOYMENT_GUIDE)
    - SIMPLE_WORKFLOW.md (consolidate into DEPLOYMENT_GUIDE)
    - QUICK_FIX_CHECKLIST.md (subsumed into TROUBLESHOOTING)
    - RUNPOD_FIX_SUMMARY.md (stale, subsumed into TROUBLESHOOTING)
    - TEMPLATE_PACKAGES.md (consolidate into QUICK_REFERENCE)

**Action:**
- [ ] Create `DEPLOYMENT_GUIDE.md` from `.runpod/` quickstart docs
- [ ] Create `TROUBLESHOOTING.md` from audit/fix docs
- [ ] Update `QUICK_REFERENCE.md` with merged content
- [ ] Delete 13 redundant files
- [ ] Verify all content moved (use regex search)

#### B. Consolidate root-level docs
**Target structure:**
```
Root:
├── README.md                 (Keep: Main entry point)
├── PRETRAINED_ENCODER_TRAINING.md (Keep: Technical deep-dive, recent & detailed)
├── docs/
│   ├── ARCHITECTURE.md       (NEW: High-level system design)
│   ├── TRAINING_PIPELINE.md  (NEW: How to train models)
│   ├── DEPLOYMENT.md         (Link to .runpod/DEPLOYMENT_GUIDE.md)
│   └── TROUBLESHOOTING.md    (Link to .runpod/TROUBLESHOOTING.md)
│
└── DELETE (STALE/REDUNDANT):
    - ENCODER_AUDIT_SUMMARY.md
    - ENCODER_FIX_IMPLEMENTATION.md (superseded by PRETRAINED_ENCODER_TRAINING.md)
    - CLEANLAB_ITERATION_2_GUIDE.md (historical iteration notes)
    - ENSEMBLE_OPTIMIZATION_ANALYSIS.md (historical analysis)
    - IMPLEMENTATION_COMPLETE.md (historical)
    - IMPLEMENTATION_SUMMARY.md (historical)
    - PHASE*.md files (all historical, 5 files)
    - SMOTE_EXPERIMENT_RESULTS.md (historical)
    - QUICKSTART_*.md (move to .runpod/DEPLOYMENT_GUIDE)
    - RUNPOD_README.md (consolidate into PRETRAINED_ENCODER_TRAINING)
    - PRETRAINED_ENCODER_AUDIT_REPORT.md (subsumed into PRETRAINED_ENCODER_TRAINING)

**Action:**
- [ ] Create `docs/ARCHITECTURE.md` (system overview)
- [ ] Create `docs/TRAINING_PIPELINE.md` (how-to guide)
- [ ] Keep only: `README.md`, `PRETRAINED_ENCODER_TRAINING.md`
- [ ] Delete 16 redundant root files
- [ ] Add symbolic links in root: `DEPLOYMENT.md -> .runpod/DEPLOYMENT_GUIDE.md`

### Success Criteria
- [ ] 13 `.runpod/` files deleted
- [ ] 16 root files deleted
- [ ] 3 new consolidated docs created
- [ ] All content preserved in new structure (grep verification)
- [ ] No broken links/references
- [ ] Total doc files: 32 → 8 (75% reduction)

---

## Phase 2: CENTRALIZED CONFIGURATION SYSTEM (Priority: HIGH)

### Problem
- Magic numbers scattered across model files (batch_size, learning_rate, dropout, etc.)
- Random seed not enforced consistently
- Hyperparameters vary between train/eval/oof pipelines
- No single source of truth for reproducibility

### Solution

Create `/Users/jack/projects/moola/src/moola/config/` directory:

#### A. `config/training_config.py` - Standard hyperparameters
```python
"""Centralized training hyperparameters for reproducibility."""

# RandomSeed Management
DEFAULT_SEED = 1337
SEED_REPRODUCIBLE = True  # Enforce deterministic CUDA ops

# Deep Learning - General
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 512
DEFAULT_NUM_WORKERS = 16
DEFAULT_PIN_MEMORY = True
USE_AMP = True  # Automatic mixed precision

# CNN-Transformer specific
CNNTR_CHANNELS = [64, 128, 128]
CNNTR_KERNELS = [3, 5, 9]
CNNTR_TRANSFORMER_LAYERS = 3
CNNTR_TRANSFORMER_HEADS = 4
CNNTR_DROPOUT = 0.25
CNNTR_N_EPOCHS = 60
CNNTR_LEARNING_RATE = 5e-4
CNNTR_EARLY_STOPPING_PATIENCE = 20
CNNTR_VAL_SPLIT = 0.15
CNNTR_MIXUP_ALPHA = 0.4
CNNTR_CUTMIX_PROB = 0.5

# RWKV-TS specific
RWKV_N_EPOCHS = 50
RWKV_LEARNING_RATE = 1e-3
RWKV_BATCH_SIZE = 256

# Loss functions
FOCAL_LOSS_GAMMA = 2.0
LOSS_ALPHA_CLASSIFICATION = 0.5  # Multi-task: classification weight
LOSS_BETA_POINTER = 0.25  # Multi-task: pointer weight
LOSS_PROGRESSIVE_WEIGHTING = True

# Data preprocessing
WINDOW_SIZE = 105
INNER_WINDOW_START = 30
INNER_WINDOW_END = 75
INNER_WINDOW_SIZE = 45
OHLC_DIMS = 4

# Temporal augmentation
TEMPORAL_AUG_JITTER_PROB = 0.5
TEMPORAL_AUG_JITTER_SIGMA = 0.05
TEMPORAL_AUG_SCALING_PROB = 0.3
TEMPORAL_AUG_SCALING_SIGMA = 0.1
TEMPORAL_AUG_TIME_WARP_PROB = 0.3
TEMPORAL_AUG_TIME_WARP_SIGMA = 0.2

# Cross-validation
DEFAULT_CV_FOLDS = 5
STRATIFIED_SPLIT = True

# SMOTE for data augmentation
SMOTE_TARGET_COUNT = 150
SMOTE_K_NEIGHBORS = 5
```

#### B. `config/model_config.py` - Model architecture definitions
```python
"""Model architecture specifications."""

MODEL_ARCHITECTURES = {
    "cnn_transformer": {
        "input_dim": 4,  # OHLC
        "cnn_channels": [64, 128, 128],
        "cnn_kernels": [3, 5, 9],
        "transformer_layers": 3,
        "transformer_heads": 4,
        "supports_pointers": True,
        "supports_multiclass": True,
    },
    "rwkv_ts": {
        "input_dim": 4,
        "hidden_dim": 128,
        "supports_pointers": False,
        "supports_multiclass": True,
    },
    "simple_lstm": {
        "input_dim": 4,
        "hidden_dim": 64,
        "supports_pointers": False,
        "supports_multiclass": True,
    },
}

# Model-to-device compatibility
MODEL_DEVICE_REQUIREMENTS = {
    "cnn_transformer": ["cpu", "cuda"],  # GPU-accelerated
    "rwkv_ts": ["cpu", "cuda"],
    "simple_lstm": ["cpu", "cuda"],
    "logreg": ["cpu"],
    "rf": ["cpu"],
    "xgb": ["cpu"],
}
```

#### C. `config/data_config.py` - Data integrity constants
```python
"""Data loading and validation specifications."""

# Expected data format
EXPECTED_FEATURES_PER_WINDOW = 4  # OHLC
EXPECTED_WINDOW_LENGTH = 105
EXPECTED_INNER_WINDOW = (30, 75)

# Validation ranges
EXPANSION_START_MIN = 30
EXPANSION_START_MAX = 74
EXPANSION_END_MIN = 30
EXPANSION_END_MAX = 74

# Data checksums (computed during initial load)
KNOWN_DATASET_CHECKSUMS = {
    # "dataset_name": "sha256_hash_of_features",
}

# Allowed label values
VALID_LABELS = ["consolidation", "retracement", "expansion"]  # Dynamically updated
```

**Action:**
- [ ] Create `config/__init__.py`
- [ ] Create `config/training_config.py`
- [ ] Create `config/model_config.py`
- [ ] Create `config/data_config.py`
- [ ] Replace magic numbers in `cnn_transformer.py` with imports
- [ ] Update `cli.py` to use config constants
- [ ] Update `oof.py` to use config constants

### Success Criteria
- [ ] 4 new config files created
- [ ] All hyperparameter magic numbers removed from model files
- [ ] Config imports don't break any tests
- [ ] Seed consistency enforced across all pipelines

---

## Phase 3: DATA/MODEL INTEGRITY GUARDRAILS (Priority: CRITICAL)

### Problem
- No validation before training starts
- Wrong datasets can silently train on wrong models
- Encoder loading not verified for weight transfer
- Class collapse (e.g., retracement=0% accuracy) only caught after training completes
- OOF prediction zeros not caught until audit

### Solution

Create `/Users/jack/projects/moola/src/moola/validation/` directory:

#### A. `validation/data_validator.py` - Data integrity checks
```python
"""Data validation and integrity checks."""

def validate_data_shape(X, y, expected_shape=None):
    """Ensure data has expected dimensions."""
    # Check: X is [N, T, F] or [N, D]
    # Check: y is [N]
    # Check: No NaN/Inf values
    # Check: Feature ranges reasonable (not all zeros)

def validate_labels(y, allowed_labels=None):
    """Ensure labels are valid."""
    # Check: All labels in allowed set
    # Check: No empty classes
    # Check: Min samples per class >= 2

def compute_data_checksum(X):
    """Compute SHA256 of data features."""
    # Used to detect data mutations between train/val splits

def validate_train_val_split_integrity(X_train, X_val, X_original):
    """Ensure no data leakage between splits."""
    # Check: No sample appears in both train and val
    # Check: checksums(X_train) + checksums(X_val) == checksum(X_original)
```

#### B. `validation/model_validator.py` - Model compatibility checks
```python
"""Model architecture validation."""

def validate_model_input_shape(model, X):
    """Check if X matches expected input shape for model."""
    # Check: Input dims match model.input_dim
    # Check: Sequence length correct for model

def validate_encoder_loading(model, encoder_path):
    """Verify encoder weights actually loaded and are frozen."""
    # Check: File exists and valid PyTorch checkpoint
    # Check: Weights transferred to model layers
    # Check: CNN/Transformer layer gradients are/aren't flowing as expected

def verify_weight_transfer(model_before, model_after, encoder_path):
    """Verify weights changed after encoder loading."""
    # Compare CNN layer weights before/after
    # Ensure at least 90% of encoder weights transferred

def detect_class_collapse(predictions, labels):
    """Warn if any class has <10% accuracy."""
    # Early warning: catches issues like "retracement=0%"
    # Raises warning, not error (allows debugging)
```

#### C. `validation/compatibility_matrix.py` - Model-dataset compatibility
```python
"""Model-dataset compatibility matrix."""

COMPATIBLE_CONFIGS = {
    "cnn_transformer": {
        "input_types": ["ohlc_timeseries"],
        "window_sizes": [105],
        "min_samples": 20,
        "supports_expansion_indices": True,
    },
    "rwkv_ts": {
        "input_types": ["ohlc_timeseries", "features"],
        "window_sizes": [105, None],
        "min_samples": 20,
        "supports_expansion_indices": False,
    },
}

def validate_model_data_compatibility(model_name, X, y, expansion_indices=None):
    """Check if model and data are compatible."""
    # Raise ValueError if incompatible
    # Log warnings for suboptimal combos
```

#### D. Update `cli.py` pre-training validation
```python
def oof(cfg_dir, over, model, seed, device, load_pretrained_encoder):
    """Generate out-of-fold predictions with validation gates."""
    ...
    # VALIDATION GATE 1: Data integrity
    from .validation.data_validator import validate_data_shape, validate_labels
    validate_data_shape(X, y)
    validate_labels(y, VALID_LABELS)

    # VALIDATION GATE 2: Model-data compatibility
    from .validation.compatibility_matrix import validate_model_data_compatibility
    validate_model_data_compatibility(model, X, y, expansion_start, expansion_end)

    # VALIDATION GATE 3: Encoder loading
    if load_pretrained_encoder:
        from .validation.model_validator import validate_encoder_loading
        validate_encoder_loading(encoder_path)  # Before training

    # ... rest of training ...

    # VALIDATION GATE 4: Class collapse detection
    from .validation.model_validator import detect_class_collapse
    detect_class_collapse(oof_predictions, y)
```

**Action:**
- [ ] Create `validation/__init__.py`
- [ ] Create `validation/data_validator.py`
- [ ] Create `validation/model_validator.py`
- [ ] Create `validation/compatibility_matrix.py`
- [ ] Add validation gates to `cli.py` in `train()`, `oof()`, `evaluate()` commands
- [ ] Add per-fold class balance logging in `oof.py`

### Success Criteria
- [ ] 4 new validation modules created
- [ ] All CLI commands include validation gates
- [ ] Class collapse detected in first epoch (via logging per-fold accuracies)
- [ ] Data shape mismatches caught before training
- [ ] Encoder loading verified before fine-tuning starts

---

## Phase 4: CODE REFACTORING FOR CLARITY (Priority: MEDIUM)

### A. `models/cnn_transformer.py` - Extract magic numbers
- Replace `0.25` dropout → `CNNTR_DROPOUT` from config
- Replace `60` epochs → `CNNTR_N_EPOCHS`
- Replace `5e-4` learning rate → `CNNTR_LEARNING_RATE`
- Extract positional encoding constants (30, 75, 105) to config
- Add logging verifier for encoder loading

### B. `pipelines/oof.py` - Add per-fold debugging
```python
# Log per-fold metrics immediately after prediction
for label_idx, label_name in enumerate(unique_labels):
    mask = y_val == label_idx
    class_acc = (val_pred[mask] == y_val[mask]).mean()
    logger.info(f"Fold {fold_idx} Class '{label_name}' accuracy: {class_acc:.4f}")
    if class_acc < 0.1:
        logger.warning(f"⚠️  Class '{label_name}' accuracy critically low!")
```

### C. `models/cnn_transformer.py` - Add encoder verification logging
```python
def load_pretrained_encoder(self, encoder_path):
    """Load pre-trained encoder with verification."""
    ...
    # VERIFY: Check weights actually changed
    encoder_layers_before = [p.clone() for p in self.model.cnn_blocks[0].parameters()]
    self.model.load_state_dict(...)
    encoder_layers_after = [p for p in self.model.cnn_blocks[0].parameters()]

    # Compare using cosine similarity
    sim = torch.nn.functional.cosine_similarity(
        encoder_layers_before[0].flatten(),
        encoder_layers_after[0].flatten(),
        dim=0
    )

    if sim > 0.99:
        logger.warning("[SSL] Encoder weights may not have been updated (cosine sim > 0.99)")
    else:
        logger.info(f"[SSL] Encoder weights updated (cosine sim: {sim:.4f})")
```

**Action:**
- [ ] Update `models/cnn_transformer.py` with config imports
- [ ] Add per-fold class accuracy logging in `pipelines/oof.py`
- [ ] Add encoder weight verification in `load_pretrained_encoder()`
- [ ] Add dataset checksum validation

### Success Criteria
- [ ] No magic numbers in model files (all use config)
- [ ] Class collapse visible in fold logs
- [ ] Encoder loading verified with weight similarity check
- [ ] Data mutations caught with checksums

---

## Phase 5: CLEANUP & VERIFICATION (Priority: HIGH)

### A. Delete redundant documentation
- [ ] Delete 13 `.runpod/` files
- [ ] Delete 16 root-level `.md` files
- [ ] Verify grep finds no broken references

### B. Run comprehensive tests
- [ ] All data validations pass
- [ ] Model-data compatibility checks pass
- [ ] CLI commands work with validation gates
- [ ] OOF generation catches class collapse early
- [ ] Encoder loading verifies weight transfer

### C. Git commit and document changes
- [ ] Commit: "refactor: consolidate documentation, extract config"
- [ ] Commit: "feat: add comprehensive data/model validation"
- [ ] Commit: "refactor: replace magic numbers with config constants"

### Success Criteria
- [ ] All redundant docs deleted
- [ ] No broken references in remaining docs
- [ ] All validations pass
- [ ] Codebase ready for MLOps audit

---

## Files Modified Summary

### New Files (9)
```
src/moola/config/
  ├── __init__.py
  ├── training_config.py
  ├── model_config.py
  └── data_config.py

src/moola/validation/
  ├── __init__.py
  ├── data_validator.py
  ├── model_validator.py
  └── compatibility_matrix.py

docs/
  ├── ARCHITECTURE.md
  └── TRAINING_PIPELINE.md
```

### Modified Files (4)
```
src/moola/models/cnn_transformer.py      (Import config, add verification)
src/moola/pipelines/oof.py               (Add per-fold logging, validation)
src/moola/cli.py                         (Add validation gates)
src/moola/data/load.py                   (Add checksums)
```

### Consolidated Documentation (3)
```
.runpod/DEPLOYMENT_GUIDE.md              (NEW: consolidate 6 files)
.runpod/TROUBLESHOOTING.md               (NEW: consolidate 6 files)
.runpod/QUICK_REFERENCE.md               (UPDATE: merge storage/template docs)
```

### Deleted Files (29)
```
.runpod/ (13 files)
root level (16 files)
```

---

## Timeline

- **Phase 1 (Docs):** 30 min - straightforward consolidation
- **Phase 2 (Config):** 1 hour - file creation + imports
- **Phase 3 (Validation):** 1.5 hours - validation logic + CLI integration
- **Phase 4 (Refactoring):** 45 min - magic number replacement
- **Phase 5 (Verification):** 30 min - testing + cleanup

**Total:** ~4 hours

---

## Success Metrics

1. **Documentation reduction:** 32 files → 8 files (75%)
2. **Magic number elimination:** 100% of hyperparameters in config
3. **Validation coverage:** All entry points have guardrails
4. **Data integrity:** Checksums prevent silent corruption
5. **Early warning:** Class collapse detected in fold 1, not after full training
6. **Reproducibility:** Random seeds enforced globally

---

## Rollback Plan

- All changes are backward compatible (config provides defaults)
- If validation fails: remove validation gates, revert config imports
- Git tags for checkpoint: `v-before-cleanup`, `v-config-added`, `v-validation-added`

---

## Next Steps

1. **Immediate:** Execute Phase 1 (documentation)
2. **Then:** Execute Phase 2 (config system)
3. **Then:** Execute Phase 3 (validation guardrails)
4. **Then:** Execute Phase 4 (code refactoring)
5. **Finally:** Execute Phase 5 (verification)

All phases follow strict git commit hygiene with descriptive messages.

---

*Last updated: 2025-10-16*
*Next review: After Phase 3 completion*
