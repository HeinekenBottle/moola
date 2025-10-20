# Mid-Level Architecture Audit
**Date:** 2025-10-20  
**Scope:** src/moola/, scripts/, data/, artifacts/  
**Purpose:** Ensure consistency with refactored architecture and project goals

---

## 🎯 Audit Objectives

1. **Consistency** - All directories follow same organizational principles
2. **Clarity** - No ambiguous or duplicate directories
3. **Consolidation** - Related functionality grouped together
4. **Naming** - Consistent naming conventions applied
5. **Cleanup** - Remove obsolete/duplicate files

---

## 📋 Part 1: src/moola/ Structure Audit

### ✅ **GOOD - Well Organized**

1. **`models/`** - Model architectures (14 files)
   - Clear separation of model types
   - Consistent naming
   - **Status:** ✅ Good

2. **`pretraining/`** - Self-supervised learning (4 files)
   - Clear purpose
   - Well organized
   - **Status:** ✅ Good

3. **`pipelines/`** - Training pipelines (4 files)
   - OOF, stacking, SSL, FixMatch
   - **Status:** ✅ Good

4. **`features/`** - Feature engineering (4 files)
   - RelativeTransform, price action features
   - **Status:** ✅ Good

5. **`config/`** - Configuration files (5 files)
   - Model, data, training configs
   - **Status:** ✅ Good

6. **`runpod/`** - RunPod orchestration (2 files)
   - SCP orchestrator, README
   - **Status:** ✅ Good

### ⚠️ **NEEDS REVIEW - Potential Issues**

7. **`data_infra/`** vs **`data/`** - Overlapping concerns
   - `data_infra/` - Schemas, validation, monitoring
   - `data/` - Data loading, splits, pipelines
   - **Issue:** Unclear boundary between the two
   - **Recommendation:** Consider consolidating or clarifying

8. **`schemas/`** vs **`data_infra/schemas.py`** - Duplicate schemas
   - `schemas/canonical_v1.py` - Canonical schema
   - `data_infra/schemas.py` - Data schemas
   - `data_infra/schemas_11d.py` - 11D schemas
   - **Issue:** Schema definitions scattered
   - **Recommendation:** Consolidate to `schemas/` directory

9. **`experiments/`** - Overlaps with `pipelines/`
   - Contains: benchmark.py, data_manager.py, validation.py
   - **Issue:** Unclear distinction from `pipelines/`
   - **Recommendation:** Move to `pipelines/` or clarify purpose

10. **`scripts/`** inside **`src/moola/`** - Confusing
    - Contains: deploy_fixes.py, test scripts, training scripts
    - **Issue:** Overlaps with top-level `scripts/` directory
    - **Recommendation:** Move to top-level `scripts/` or rename to `internal/`

11. **`utils/`** - Too many files (24 files)
    - Contains everything from augmentation to windowing
    - **Issue:** Catch-all directory, hard to navigate
    - **Recommendation:** Split into subcategories:
      - `utils/augmentation/` (augmentation, mixup, temporal_augmentation)
      - `utils/training/` (early_stopping, training_utils, training_pipeline_integration)
      - `utils/validation/` (data_validation, pseudo_sample_validation)
      - `utils/metrics/` (metrics, losses, focal_loss)
      - Keep core utils at top level

12. **`validation/`** - Only 2 files
    - training_monitor.py, training_validator.py
    - **Issue:** Could be in `utils/training/` or `pipelines/`
    - **Recommendation:** Consider consolidating

13. **`diagnostics/`** - Empty or minimal
    - **Recommendation:** Check if needed, delete if empty

14. **`optimization/`** - Empty or minimal
    - **Recommendation:** Check if needed, delete if empty

15. **`visualization/`** - Only 1 file (calibration.py)
    - **Issue:** Underutilized directory
    - **Recommendation:** Keep for future expansion or move to `utils/`

16. **`api/`** - Production API (2 files)
    - serve.py, __init__.py
    - **Issue:** Production code in ML training repo
    - **Recommendation:** Consider separate repo or clearly document

### 🔍 **Duplicate Files**

17. **`cli.py`** vs **`cli_feature_aware.py`**
    - **Issue:** Two CLI files
    - **Recommendation:** Consolidate or clarify purpose

18. **`schema.py`** vs **`schemas/`** directory
    - **Issue:** Duplicate schema definitions
    - **Recommendation:** Consolidate to `schemas/` directory

---

## 📋 Part 2: scripts/ Directory Audit

### 📊 **Statistics**
- **Total scripts:** 98 files
- **Archive:** ~30 files in `scripts/archive/`
- **Active:** ~68 files

### ⚠️ **Issues Found**

1. **No organization** - Flat structure with 98 files
   - Hard to find specific scripts
   - No categorization

2. **Naming inconsistency**
   - Some: `runpod_pretrain_bilstm.py`
   - Some: `deploy_2layer_training.sh`
   - Some: `fix_train_latest_features.py`
   - **Recommendation:** Consistent naming: `{category}_{action}_{target}.{ext}`

3. **Mixed purposes**
   - Data extraction
   - Model training
   - RunPod deployment
   - Debugging/testing
   - Utilities

### ✅ **Recommended Organization**

```
scripts/
├── data/                   # Data extraction and processing
│   ├── extract_batch_200.py
│   ├── extract_session_aware_batch.py
│   ├── merge_candlesticks_annotations.py
│   ├── fix_train_latest_features.py
│   └── generate_structure_labels.py
│
├── training/               # Model training scripts
│   ├── train_simple_lstm.py
│   ├── train_cnn_pretrained_fixed.py
│   ├── deploy_2layer_training.sh
│   └── deploy_174_training.sh
│
├── pretraining/            # Pre-training scripts
│   ├── runpod_pretrain_bilstm.py
│   └── pretrain_*.py
│
├── runpod/                 # RunPod deployment and orchestration
│   ├── build_runpod_bundle.sh
│   ├── deploy_to_fresh_pod.py
│   ├── runpod_baseline_workflow.sh
│   ├── runpod_quick_train.sh
│   └── verify_runpod_env.py
│
├── evaluation/             # Model evaluation and analysis
│   ├── generate_full_oof.py
│   ├── select_best_model.py
│   ├── aggregate_results.py
│   └── diagnose_transfer_learning.py
│
├── cleanlab/               # CleanLab integration
│   ├── export_for_cleanlab_studio.py
│   ├── convert_to_cleanlab_format.py
│   └── create_cleaned_train.py
│
├── utils/                  # Utility scripts
│   ├── experiment_configs.py
│   ├── send_slack_notification.py
│   └── export_prometheus_metrics.py
│
└── archive/                # Old/deprecated scripts (keep as-is)
```

---

## 📋 Part 3: data/ Directory Audit

### ⚠️ **Issues Found**

1. **`data/artifacts/`** - Duplicate of top-level `artifacts/`
   - Contains: models/, oof/, pretrained/, splits/
   - **Issue:** We just created top-level `artifacts/`
   - **Recommendation:** **DELETE** `data/artifacts/` and move contents to `artifacts/`

2. **`data/experiments/`** - Unclear purpose
   - Contains: verify_test/
   - **Issue:** Overlaps with experiments elsewhere
   - **Recommendation:** Review and consolidate or delete

3. **`data/interim/`** - Unclear purpose
   - **Recommendation:** Check if used, delete if empty

4. **`data/logs/`** - Duplicate of `artifacts/logs/`
   - **Issue:** We just moved logs to `artifacts/logs/`
   - **Recommendation:** **DELETE** `data/logs/` if empty

5. **`data/pretraining/`** - Should be in `data/processed/unlabeled/`
   - **Recommendation:** Move to `data/processed/unlabeled/`

6. **`data/reports/`** - Unclear purpose
   - **Recommendation:** Check if used, move to `artifacts/` or delete

7. **`data/training_cleaned/`** - Unclear purpose
   - **Recommendation:** Check if used, move to `data/processed/archived/` or delete

8. **`data/corrections/bespoke/`** and **`data/corrections/bespoke_annotations/`**
   - **Issue:** Duplicate directories?
   - **Recommendation:** Consolidate

### ✅ **Recommended Structure**

```
data/
├── raw/
│   ├── unlabeled/          # 2.2M unlabeled windows
│   └── labeled/            # (future: raw labeled data)
│
├── processed/
│   ├── unlabeled/          # Processed unlabeled (4D, 11D)
│   ├── labeled/            # Current training set (174 samples)
│   └── archived/           # Historical datasets
│
├── oof/
│   ├── supervised/         # OOF from supervised models
│   └── pretrained/         # OOF from pretrained models
│
├── splits/                 # Train/val/test splits
│
├── batches/                # Annotation batches
│
└── corrections/            # Human annotations
    ├── candlesticks_annotations/
    ├── review_corrections/
    └── window_blacklist.csv
```

**DELETE:**
- `data/artifacts/` (move to top-level `artifacts/`)
- `data/experiments/` (consolidate or delete)
- `data/interim/` (if empty)
- `data/logs/` (moved to `artifacts/logs/`)
- `data/pretraining/` (move to `data/processed/unlabeled/`)
- `data/reports/` (move to `artifacts/` or delete)
- `data/training_cleaned/` (move to `data/processed/archived/` or delete)

---

## 📋 Part 4: artifacts/ Directory Audit

### ✅ **GOOD - Well Organized**

Current structure is good after Phase 4 refactor:
```
artifacts/
├── encoders/
│   ├── pretrained/         # bilstm_mae_4d_v1.pt, tstcc_encoder_v1.pt
│   └── supervised/
├── models/
│   ├── supervised/
│   ├── pretrained/
│   └── ensemble/
├── oof/
├── metadata/
├── results/
├── logs/
├── runpod_bundles/
└── runpod_results/
```

### ⚠️ **Minor Issues**

1. **`artifacts/runs/`** - Unclear purpose
   - **Recommendation:** Check if TensorBoard runs, move to `artifacts/logs/tensorboard/`

2. **`artifacts/oof/`** - Not organized by supervised/pretrained
   - **Recommendation:** Create subdirectories like `data/oof/`

---

## 📊 Summary of Issues

### **HIGH PRIORITY - Duplicates**
1. ❌ `data/artifacts/` - Duplicate of top-level `artifacts/`
2. ❌ `data/logs/` - Duplicate of `artifacts/logs/`
3. ❌ `src/moola/schemas/` vs `src/moola/data_infra/schemas.py` - Duplicate schemas
4. ❌ `src/moola/schema.py` vs `src/moola/schemas/` - Duplicate schemas

### **MEDIUM PRIORITY - Organization**
5. ⚠️ `scripts/` - 98 files, no organization
6. ⚠️ `src/moola/utils/` - 24 files, needs subcategories
7. ⚠️ `src/moola/scripts/` - Overlaps with top-level `scripts/`
8. ⚠️ `src/moola/experiments/` - Overlaps with `pipelines/`

### **LOW PRIORITY - Cleanup**
9. 🔍 `data/experiments/` - Check if needed
10. 🔍 `data/interim/` - Check if empty
11. 🔍 `data/pretraining/` - Move to `data/processed/unlabeled/`
12. 🔍 `data/reports/` - Move to `artifacts/` or delete
13. 🔍 `data/training_cleaned/` - Move to `data/processed/archived/` or delete
14. 🔍 `src/moola/diagnostics/` - Check if empty
15. 🔍 `src/moola/optimization/` - Check if empty

---

## 🎯 Recommended Actions

### **Phase 8: Mid-Level Refactor**

**Step 1: Delete Duplicates**
- Delete `data/artifacts/` (move contents to `artifacts/`)
- Delete `data/logs/` (already moved to `artifacts/logs/`)
- Consolidate schemas to `src/moola/schemas/`

**Step 2: Organize scripts/**
- Create subdirectories: data/, training/, pretraining/, runpod/, evaluation/, cleanlab/, utils/
- Move 68 active scripts to appropriate subdirectories
- Keep archive/ as-is

**Step 3: Reorganize src/moola/utils/**
- Create subdirectories: augmentation/, training/, validation/, metrics/
- Move 24 files to appropriate subdirectories

**Step 4: Clean up data/**
- Move `data/pretraining/` to `data/processed/unlabeled/`
- Delete or move `data/experiments/`, `data/interim/`, `data/reports/`, `data/training_cleaned/`

**Step 5: Consolidate src/moola/**
- Move `src/moola/scripts/` to top-level `scripts/internal/`
- Consolidate `src/moola/experiments/` into `pipelines/` or clarify purpose
- Consolidate schemas

---

## ⏱️ Estimated Effort

- **Step 1:** 30 minutes (delete duplicates)
- **Step 2:** 1 hour (organize scripts/)
- **Step 3:** 30 minutes (reorganize utils/)
- **Step 4:** 30 minutes (clean up data/)
- **Step 5:** 30 minutes (consolidate src/moola/)

**Total:** 3 hours

---

## 📝 Next Steps

1. Review this audit
2. Confirm recommended actions
3. Execute Phase 8 refactor
4. Update documentation

