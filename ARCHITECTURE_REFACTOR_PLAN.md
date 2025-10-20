# Moola Architecture Refactor Plan
**Date:** 2025-10-20  
**Current Dataset:** 174 labeled samples (small dataset regime)  
**Goal:** Clean directory structure with clear data/model/encoder taxonomy

---

## 🎯 Executive Summary

This refactor addresses three critical issues:
1. **Root directory clutter** - AI agent configs, duplicate documentation, scattered artifacts
2. **Ambiguous data taxonomy** - Unclear distinction between unlabeled/labeled, 4D/11D, raw/processed
3. **Confusing model/encoder naming** - Mixed terminology for encoders, models, weights, checkpoints

**Key Principle:** Distinguish **encoders** (reusable feature extractors) from **models** (complete architectures) from **weights** (saved checkpoints).

---

## 📋 Part 1: Root Directory Cleanup

### Current Problems
```
/Users/jack/projects/moola/
├── .mcp.json                          # Empty, duplicates global config
├── .env                               # Contains GLM_API_KEY (not used by Claude Code)
├── .claude/                           # Local Claude Code cache (OK to keep)
├── .factory/                          # Local Factory cache (OK to keep)
├── claude_code_zai_env.sh             # Duplicate of global config
├── code_reviewer_agent.md             # OpenCode agent (belongs in ~/dotfiles)
├── data_scientist_agent.md            # OpenCode agent (belongs in ~/dotfiles)
├── ml_engineer_agent.md               # OpenCode agent (belongs in ~/dotfiles)
├── python_pro_agent.md                # OpenCode agent (belongs in ~/dotfiles)
├── code_review_command.md             # OpenCode command (belongs in ~/dotfiles)
├── data_analysis_command.md           # OpenCode command (belongs in ~/dotfiles)
├── ml_pipeline_command.md             # OpenCode command (belongs in ~/dotfiles)
├── python_dev_command.md              # OpenCode command (belongs in ~/dotfiles)
├── model_174_baseline.pkl             # Scattered model artifact
├── model_174_pretrained.pkl           # Scattered model artifact
├── feature_metadata_174.json          # Scattered metadata
├── test_oof.npy                       # Scattered OOF predictions
├── test_splits/                       # Duplicate of data/splits/
├── runpod_bundle_*.tar.gz             # Build artifacts (should be in artifacts/)
├── runpod_bundle_20251020_013740/     # Unpacked bundle (should be in artifacts/)
├── runpod_bundle_build/               # Build directory (should be in artifacts/)
├── runpod_results/                    # Duplicate of artifacts/runpod_results/
├── CLAUDE_DESKTOP_ML_TRAINING_PROMPT.md  # Duplicate documentation
├── CLEANUP_SUMMARY_2025-10-19.md      # Temporary documentation
├── IMPLEMENTATION_11D_GUIDE.md        # Should be in docs/
├── TRANSFER_LEARNING_PROGRESS_SUMMARY.md  # Should be in docs/
├── WELCOME_BACK.md                    # Temporary documentation
└── WORKFLOW_SSH_SCP_GUIDE.md          # Should be in docs/
```

### ✅ Files to KEEP at Root
```
/Users/jack/projects/moola/
├── .git/                              # Version control
├── .gitignore                         # Git ignore patterns
├── .pre-commit-config.yaml            # Pre-commit hooks
├── .env.example                       # Example environment variables (no secrets)
├── .claude/                           # Local Claude Code cache (auto-generated)
├── .factory/                          # Local Factory cache (auto-generated)
├── .dvc/                              # DVC version control
├── .dvcignore                         # DVC ignore patterns
├── .venv/                             # Python virtual environment
├── pyproject.toml                     # Python project configuration
├── uv.lock                            # UV lock file
├── requirements.txt                   # Python dependencies
├── requirements-runpod.txt            # RunPod-specific dependencies
├── requirements_production.txt        # Production dependencies
├── Makefile                           # Build automation
├── README.md                          # Project overview
├── CLAUDE.md                          # Claude Code context (main guide)
├── RUNPOD_QUICK_START.md              # RunPod workflow guide
└── dvc.yaml                           # DVC pipeline configuration
```

### ❌ Files to DELETE
```
# AI Agent Configs (duplicates of ~/dotfiles)
.mcp.json                              # Empty, use global config
.env                                   # Contains unused GLM_API_KEY
claude_code_zai_env.sh                 # Duplicate of global config
*_agent.md                             # 4 files - OpenCode agents (global)
*_command.md                           # 4 files - OpenCode commands (global)

# Temporary Documentation (archive or delete)
CLEANUP_SUMMARY_2025-10-19.md          # Temporary, archive if needed
WELCOME_BACK.md                        # Temporary, delete
CLAUDE_DESKTOP_ML_TRAINING_PROMPT.md   # Duplicate, delete

# Scattered Artifacts (move to artifacts/)
model_174_baseline.pkl                 # → artifacts/models/supervised/
model_174_pretrained.pkl               # → artifacts/models/supervised/
feature_metadata_174.json              # → artifacts/metadata/
test_oof.npy                           # → artifacts/oof/
test_splits/                           # → DELETE (duplicate of data/splits/)
runpod_bundle_*.tar.gz                 # → artifacts/runpod_bundles/
runpod_bundle_20251020_013740/         # → artifacts/runpod_bundles/
runpod_bundle_build/                   # → artifacts/runpod_bundles/
runpod_results/                        # → artifacts/runpod_results/ (already exists)
```

### 📁 Files to MOVE to docs/
```
IMPLEMENTATION_11D_GUIDE.md            # → docs/guides/
TRANSFER_LEARNING_PROGRESS_SUMMARY.md # → docs/progress/
WORKFLOW_SSH_SCP_GUIDE.md              # → docs/workflows/ (or keep at root)
```

---

## 📋 Part 2: Data Taxonomy Refactor

### Current Problems
- `data/processed/` contains 15+ parquet files with unclear naming
- `data/pretraining/` mixes unlabeled OHLC with features
- `data/raw/` only has unlabeled data, no labeled raw data
- Unclear which files are 4D OHLC vs 11D RelativeTransform
- No clear separation of train/val/test splits

### Proposed Data Structure
```
data/
├── raw/                               # Raw data (never modified)
│   ├── unlabeled/
│   │   └── unlabeled_windows.parquet  # 2.2M samples, 4D OHLC
│   └── labeled/
│       └── (future: raw labeled data before cleaning)
│
├── processed/                         # Processed datasets ready for training
│   ├── unlabeled/
│   │   ├── unlabeled_4d_ohlc.npy      # 2.2M samples, (N, 105, 4)
│   │   └── unlabeled_11d_relative.npy # 2.2M samples, (N, 105, 11)
│   │
│   ├── labeled/
│   │   ├── train_latest.parquet       # Current training set (174 samples)
│   │   ├── train_latest_4d.npy        # 4D OHLC version
│   │   ├── train_latest_11d.npy       # 11D RelativeTransform version
│   │   └── metadata/
│   │       ├── feature_metadata_174.json
│   │       └── dataset_manifest.json
│   │
│   └── archived/                      # Historical datasets
│       ├── train_clean.parquet        # 98 samples (before batch 200)
│       ├── train_combined_174.parquet
│       ├── train_combined_175.parquet
│       ├── train_combined_178.parquet
│       ├── train_pivot_134.parquet
│       ├── train_smote_300.parquet    # Synthetic augmentation
│       └── README.md                  # Explains each archived dataset
│
├── splits/                            # Train/val/test splits
│   ├── fwd_chain_v3.json              # Forward chaining split config
│   ├── fold_0_train.npy               # 5-fold CV splits
│   ├── fold_0_val.npy
│   ├── ...
│   └── fold_4_val.npy
│
├── oof/                               # Out-of-fold predictions
│   ├── supervised/                    # From supervised models
│   │   ├── simple_lstm_clean.npy
│   │   ├── simple_lstm_augmented.npy
│   │   ├── cnn_transformer_clean.npy
│   │   ├── logreg_clean.npy
│   │   ├── rf_clean.npy
│   │   └── xgb_clean.npy
│   └── pretrained/                    # From pretrained models
│       └── (future: OOF from pretrained encoders)
│
├── batches/                           # Annotation batches
│   ├── batch_200.parquet
│   ├── batch_200_clean_keepers.parquet
│   ├── batch_200_manifest.json
│   ├── batch_201.parquet
│   ├── ...
│   └── README.md                      # Batch extraction workflow
│
├── corrections/                       # Human annotations & quality control
│   ├── candlesticks_annotations/      # Annotation JSON files
│   ├── window_blacklist.csv           # D-grade windows (permanent exclusion)
│   ├── cleanlab_label_issues.csv      # Label quality analysis
│   └── README.md                      # Annotation workflow
│
└── artifacts/                         # Experiment artifacts (MOVE TO TOP-LEVEL)
    └── (see Part 3)
```

### Key Changes
1. **Separate unlabeled/labeled** - Clear distinction in `processed/`
2. **Separate 4D/11D** - Explicit naming for OHLC vs RelativeTransform
3. **Archive old datasets** - Move historical datasets to `processed/archived/`
4. **Separate OOF by source** - Supervised vs pretrained models
5. **Move artifacts/** - Promote to top-level (see Part 3)

---

## 📋 Part 3: Model/Encoder Taxonomy Refactor

### Current Problems
- `models/pretrained/bilstm_encoder.pt` - Is this an encoder or a model?
- `artifacts/models/enhanced_simple_lstm/model.pkl` - Is this pretrained or supervised?
- `artifacts/runpod_results/simple_lstm_with_pretrained_encoder.pkl` - Unclear naming
- No distinction between encoder weights, model checkpoints, and fine-tuned models

### Proposed Structure
```
artifacts/
├── encoders/                          # Reusable feature extraction blocks
│   ├── pretrained/                    # Self-supervised pretrained encoders
│   │   ├── bilstm_mae_4d_v1.pt        # BiLSTM Masked Autoencoder (4D OHLC)
│   │   ├── bilstm_mae_11d_v1.pt       # BiLSTM Masked Autoencoder (11D Relative)
│   │   ├── ts2vec_encoder_v1.pt       # TS2Vec contrastive encoder
│   │   └── tstcc_encoder_v1.pt        # TSTCC contrastive encoder
│   │
│   └── supervised/                    # Encoders from supervised training
│       └── (future: encoder blocks extracted from trained models)
│
├── models/                            # Complete model checkpoints
│   ├── supervised/                    # Supervised training (no pretraining)
│   │   ├── simple_lstm_baseline_174.pkl
│   │   ├── enhanced_simple_lstm_174.pkl
│   │   ├── cnn_transformer_174.pkl
│   │   ├── rwkv_ts_174.pkl
│   │   ├── logreg_174.pkl
│   │   ├── rf_174.pkl
│   │   └── xgb_174.pkl
│   │
│   ├── pretrained/                    # Fine-tuned from pretrained encoders
│   │   ├── simple_lstm_bilstm_mae_4d_174.pkl
│   │   ├── simple_lstm_bilstm_mae_11d_174.pkl
│   │   ├── enhanced_simple_lstm_ts2vec_174.pkl
│   │   └── cnn_transformer_tstcc_174.pkl
│   │
│   └── ensemble/                      # Stacking ensemble models
│       ├── stack_rf_meta_174.pkl      # Random Forest meta-learner
│       └── stack_logreg_meta_174.pkl  # Logistic Regression meta-learner
│
├── oof/                               # Out-of-fold predictions (MOVE FROM data/)
│   ├── supervised/
│   │   ├── simple_lstm_174.npy
│   │   ├── enhanced_simple_lstm_174.npy
│   │   ├── cnn_transformer_174.npy
│   │   ├── logreg_174.npy
│   │   ├── rf_174.npy
│   │   └── xgb_174.npy
│   │
│   └── pretrained/
│       ├── simple_lstm_bilstm_mae_4d_174.npy
│       └── simple_lstm_bilstm_mae_11d_174.npy
│
├── metadata/                          # Experiment metadata
│   ├── feature_metadata_174.json
│   ├── dataset_manifest.json
│   └── experiment_registry.json
│
├── runpod_bundles/                    # RunPod deployment bundles
│   ├── runpod_bundle_20251020_013740.tar.gz
│   ├── runpod_bundle_20251020_013740/
│   └── runpod_bundle_build/
│
└── runpod_results/                    # Results from RunPod training
    ├── phase2_results.csv
    └── oof/
```

### Naming Convention
```
Format: {architecture}_{pretraining}_{features}_{dataset_size}.{ext}

Examples:
- bilstm_mae_4d_v1.pt              # BiLSTM encoder, MAE pretrained, 4D OHLC, version 1
- simple_lstm_baseline_174.pkl     # SimpleLSTM, no pretraining, 174 samples
- simple_lstm_bilstm_mae_11d_174.pkl  # SimpleLSTM, BiLSTM MAE encoder, 11D, 174 samples
- enhanced_simple_lstm_ts2vec_174.pkl # EnhancedSimpleLSTM, TS2Vec encoder, 174 samples
```

---

## 📋 Part 4: Migration Plan

### Phase 1: Root Cleanup (Low Risk)
```bash
# 1. Delete AI agent configs (backed up in ~/dotfiles)
rm .mcp.json .env claude_code_zai_env.sh
rm *_agent.md *_command.md

# 2. Delete temporary documentation
rm CLEANUP_SUMMARY_2025-10-19.md WELCOME_BACK.md
rm CLAUDE_DESKTOP_ML_TRAINING_PROMPT.md

# 3. Create .env.example (no secrets)
echo "# Project-specific environment variables" > .env.example
echo "# Global API keys are in ~/dotfiles/.env" >> .env.example
```

### Phase 2: Move Scattered Artifacts (Medium Risk)
```bash
# 1. Create new artifact directories
mkdir -p artifacts/models/supervised
mkdir -p artifacts/metadata
mkdir -p artifacts/runpod_bundles

# 2. Move scattered model files
mv model_174_baseline.pkl artifacts/models/supervised/simple_lstm_baseline_174.pkl
mv model_174_pretrained.pkl artifacts/models/supervised/simple_lstm_pretrained_174.pkl
mv feature_metadata_174.json artifacts/metadata/

# 3. Move OOF predictions
mv test_oof.npy artifacts/oof/simple_lstm_174.npy

# 4. Move RunPod bundles
mv runpod_bundle_*.tar.gz artifacts/runpod_bundles/
mv runpod_bundle_20251020_013740 artifacts/runpod_bundles/
mv runpod_bundle_build artifacts/runpod_bundles/

# 5. Delete duplicate directories
rm -rf test_splits/  # Duplicate of data/splits/
rm -rf runpod_results/  # Duplicate of artifacts/runpod_results/
```

### Phase 3: Data Taxonomy Refactor (High Risk - Requires Testing)
```bash
# 1. Create new data structure
mkdir -p data/raw/unlabeled data/raw/labeled
mkdir -p data/processed/unlabeled data/processed/labeled/metadata
mkdir -p data/processed/archived
mkdir -p data/oof/supervised data/oof/pretrained

# 2. Move unlabeled data
# (Already in correct location: data/raw/unlabeled_windows.parquet)

# 3. Rename current training dataset
cp data/processed/train_latest.parquet data/processed/labeled/train_latest.parquet

# 4. Archive old datasets
mv data/processed/train_clean.parquet data/processed/archived/
mv data/processed/train_combined_*.parquet data/processed/archived/
mv data/processed/train_pivot_134.parquet data/processed/archived/
mv data/processed/train_smote_300.parquet data/processed/archived/

# 5. Move OOF predictions
mv data/oof/*.npy data/oof/supervised/

# 6. Create README files
# (Document each archived dataset)
```

### Phase 4: Model/Encoder Taxonomy Refactor (High Risk - Requires Code Changes)
```bash
# 1. Create new encoder/model structure
mkdir -p artifacts/encoders/pretrained artifacts/encoders/supervised
mkdir -p artifacts/models/supervised artifacts/models/pretrained artifacts/models/ensemble

# 2. Rename existing encoder
mv models/pretrained/bilstm_encoder.pt artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt

# 3. Move existing models
mv artifacts/models/enhanced_simple_lstm/model.pkl artifacts/models/supervised/enhanced_simple_lstm_174.pkl
mv artifacts/models/logreg/model.pkl artifacts/models/supervised/logreg_174.pkl

# 4. Update code references
# (Requires updating paths in src/moola/models/pretrained_utils.py, etc.)
```

---

## 📋 Part 5: Code Changes Required

### Files to Update
1. **src/moola/paths.py** - Update all artifact paths
2. **src/moola/models/pretrained_utils.py** - Update encoder loading paths
3. **src/moola/pretraining/masked_lstm_pretrain.py** - Update save paths
4. **src/moola/cli.py** - Update default paths
5. **scripts/*.py** - Update all hardcoded paths
6. **tests/*.py** - Update test fixture paths

### Example Path Updates
```python
# OLD
ENCODER_PATH = "models/pretrained/bilstm_encoder.pt"
MODEL_PATH = "artifacts/models/enhanced_simple_lstm/model.pkl"

# NEW
ENCODER_PATH = "artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt"
MODEL_PATH = "artifacts/models/supervised/enhanced_simple_lstm_174.pkl"
```

---

## 📋 Part 6: Documentation Updates

### Files to Create
1. **data/processed/archived/README.md** - Explain each archived dataset
2. **artifacts/encoders/README.md** - Explain encoder taxonomy
3. **artifacts/models/README.md** - Explain model taxonomy
4. **MIGRATION_GUIDE.md** - Document migration process

### Files to Update
1. **CLAUDE.md** - Update directory structure section
2. **README.md** - Update quick start paths
3. **docs/ARCHITECTURE.md** - Update architecture diagrams

---

## ✅ Success Criteria

1. **Root directory** - Only essential project files (≤15 files)
2. **Data taxonomy** - Clear separation of unlabeled/labeled, 4D/11D, raw/processed
3. **Model taxonomy** - Clear distinction between encoders, models, weights
4. **No duplicates** - No duplicate configs, artifacts, or documentation
5. **Tests pass** - All existing tests pass after migration
6. **Documentation** - All paths updated in docs and CLAUDE.md

---

## 🚨 Risks & Mitigation

### High Risk
- **Breaking existing code** - Mitigation: Update all path references before moving files
- **Losing experiment history** - Mitigation: Archive old datasets with README
- **RunPod workflow breaks** - Mitigation: Test bundle creation after refactor

### Medium Risk
- **Git history confusion** - Mitigation: Use `git mv` for tracked files
- **Broken symlinks** - Mitigation: Check for symlinks before moving

### Low Risk
- **Documentation out of sync** - Mitigation: Update docs in same PR as refactor

---

## 📅 Recommended Execution Order

1. **Phase 1: Root Cleanup** (30 min, low risk)
2. **Phase 5: Code Changes** (2 hours, prepare for migration)
3. **Phase 2: Move Scattered Artifacts** (30 min, medium risk)
4. **Phase 3: Data Taxonomy Refactor** (1 hour, high risk - test thoroughly)
5. **Phase 4: Model/Encoder Taxonomy Refactor** (1 hour, high risk - test thoroughly)
6. **Phase 6: Documentation Updates** (1 hour, low risk)

**Total Estimated Time:** 6 hours

---

## 🎯 Next Steps

1. **Review this plan** - Confirm approach with user
2. **Create backup branch** - `git checkout -b refactor/architecture-cleanup`
3. **Execute Phase 1** - Low-risk root cleanup
4. **Test after each phase** - Run tests, verify paths
5. **Create PR** - Review changes before merging to main

