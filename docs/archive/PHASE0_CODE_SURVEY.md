# Phase 0: Model & Code Survey
**Date:** 2025-10-18

## Executive Summary

The Moola project has a clear production focus around **SimpleLSTM** and **EnhancedSimpleLSTM** for 98-sample financial time series classification. The codebase includes several experimental/pre-training models (BiLSTM autoencoders, CNN-Transformer, RWKV-TS, TS-TCC) that support transfer learning but are not directly used in production inference. Data infrastructure uses **stratified random K-fold splits** (not temporal forward-chaining), and training uses simple **train_test_split** for 80/20 evaluation.

---

## 1. Active Models (Keep in models/)

### 1.1 SimpleLSTM
- **File:** `/Users/jack/projects/moola/src/moola/models/simple_lstm.py`
- **LOC:** 921 lines
- **Size:** 40 KB
- **Last Modified:** Oct 18, 01:42
- **Status:** ACTIVE baseline + production model
- **Registry:** YES (models/__init__.py line 24)
- **CLI Usage:** YES (train, evaluate, pretrain commands support it)
- **Used In:**
  - `cli.py` train command (default model option)
  - `cli.py` evaluate command
  - `cli.py` oof (out-of-fold) command
  - Tests and examples
- **Key Features:**
  - 921 parameters (production-grade for 98-sample dataset)
  - Dual-input support: OHLC sequences + engineered features
  - Bidirectional LSTM (BiLSTM) for temporal processing
  - Mixup + CutMix augmentation
  - Early stopping with patience=20
  - Pre-trained encoder support (transfers from BiLSTM masked autoencoder)
- **Architecture:** [B, 105, 4] OHLC → BiLSTM(128) → 256 → Classification head

### 1.2 EnhancedSimpleLSTM
- **File:** `/Users/jack/projects/moola/src/moola/models/enhanced_simple_lstm.py`
- **LOC:** 778 lines
- **Size:** 31 KB
- **Last Modified:** Oct 18, 01:35
- **Status:** ACTIVE (alternative/variant)
- **Registry:** NOT in models/__init__.py (!)
- **CLI Usage:** NO (not registered)
- **Note:** Exists as experimental variant but not actively wired into production pipeline
- **Key Difference:** Feature-aware transfer learning with combined OHLC+features input
- **Recommendation:** Clarify if this should replace SimpleLSTM or remain experimental

### 1.3 LogReg, RF, XGBoost
- **Files:**
  - `logreg.py` (109 LOC, 4.0 KB)
  - `rf.py` (138 LOC, 5.1 KB)
  - `xgb.py` (287 LOC, 11 KB)
- **Status:** ACTIVE baseline models (stacking ensemble base learners)
- **Registry:** YES
- **CLI Usage:** YES (train, evaluate, oof commands)
- **Last Modified:** Oct 14 (older, stable)
- **Purpose:** OOF stacking ensemble with deep learning models

### 1.4 Stack
- **File:** `/Users/jack/projects/moola/src/moola/models/stack.py`
- **LOC:** 87 lines
- **Status:** ACTIVE meta-learner for stacking
- **Registry:** YES
- **Last Modified:** Oct 12

---

## 2. Experimental/Pre-training Models (Candidates for models_extras/)

### 2.1 BiLSTM Masked Autoencoder
- **File:** `/Users/jack/projects/moola/src/moola/models/bilstm_masked_autoencoder.py`
- **LOC:** 359 lines
- **Size:** 12 KB
- **Status:** EXPERIMENTAL (pre-training only)
- **Registry:** NO
- **CLI Usage:** YES (pretrain-bilstm command line 713)
- **Last Modified:** Oct 16, 20:51
- **Purpose:** Self-supervised pre-training for SimpleLSTM encoder transfer learning
- **Note:** Not imported in models/__init__.py; directly instantiated in CLI for pre-training
- **Recommendation:** MOVE to models_extras/ - only used for pre-training, not inference

### 2.2 Feature-Aware BiLSTM Masked Autoencoder
- **File:** `/Users/jack/projects/moola/src/moola/models/feature_aware_bilstm_masked_autoencoder.py`
- **LOC:** 471 lines
- **Size:** 18 KB
- **Status:** EXPERIMENTAL (pre-training variant)
- **Registry:** NO
- **Last Modified:** Oct 18, 01:34 (recent work)
- **Purpose:** Pre-training with engineered features integration
- **Used By:** `utils/feature_aware_utils.py` only (experimental utils)
- **Recommendation:** MOVE to models_extras/ - pre-training only

### 2.3 BiLSTM Autoencoder (non-masked)
- **File:** `/Users/jack/projects/moola/src/moola/models/bilstm_autoencoder.py`
- **LOC:** 403 lines
- **Size:** 13 KB
- **Status:** EXPERIMENTAL (deprecated variant)
- **Registry:** NO
- **Last Modified:** Oct 16, 19:35
- **Purpose:** Earlier autoencoder implementation (replaced by masked version)
- **Recommendation:** MOVE to models_extras/ - superseded

### 2.4 TS-TCC (Time Series Contrastive Clustering)
- **File:** `/Users/jack/projects/moola/src/moola/models/ts_tcc.py`
- **LOC:** 591 lines
- **Size:** 21 KB
- **Status:** EXPERIMENTAL (pre-training alternative)
- **Registry:** NO
- **CLI Usage:** YES (pretrain-tcc command)
- **Last Modified:** Oct 16, 23:45
- **Purpose:** Contrastive pre-training alternative to masked autoencoder
- **Note:** Directly instantiated in CLI (TSTCCPretrainer), not in models registry
- **Recommendation:** MOVE to models_extras/ - alternative pre-training only

### 2.5 CNN-Transformer
- **File:** `/Users/jack/projects/moola/src/moola/models/cnn_transformer.py`
- **LOC:** 1321 lines (largest)
- **Size:** 56 KB
- **Status:** EXPERIMENTAL (alternative architecture)
- **Registry:** YES (models/__init__.py line 25)
- **CLI Usage:** YES (train, evaluate, oof support)
- **Last Modified:** Oct 16, 23:44
- **Purpose:** Alternative deep learning architecture (CNN feature extraction + Transformer)
- **Parameters:** 56K (vs 921 for SimpleLSTM)
- **Status:** Included in registry but appears to be experimental option
- **Recommendation:** KEEP or MOVE to models_extras/? Currently in registry but likely over-parameterized for 98-sample dataset. Recent evaluation suggests SimpleLSTM outperforms it.

### 2.6 RWKV-TS
- **File:** `/Users/jack/projects/moola/src/moola/models/rwkv_ts.py`
- **LOC:** 639 lines
- **Size:** 25 KB
- **Status:** EXPERIMENTAL (alternative RNN architecture)
- **Registry:** YES (models/__init__.py line 23)
- **CLI Usage:** YES (train, evaluate, oof support)
- **Last Modified:** Oct 16, 23:45
- **Parameters:** 409K (vs 921 for SimpleLSTM)
- **Purpose:** RWKV (RecurrentWKV) time series model - over-parameterized for small dataset
- **Recommendation:** MOVE to models_extras/ - significantly over-parameterized (409K params for 98 samples)

---

## 3. Model Registry Analysis

**Current Registry** (`models/__init__.py`):
```python
_MODEL_REGISTRY = {
    "logreg": LogRegModel,
    "rf": RFModel,
    "xgb": XGBModel,
    "rwkv_ts": RWKVTSModel,           # EXPERIMENTAL - over-parameterized
    "simple_lstm": SimpleLSTMModel,   # ACTIVE
    "cnn_transformer": CnnTransformerModel,  # EXPERIMENTAL - over-parameterized
    "stack": StackModel,              # ACTIVE
}
```

**Missing from Registry** (but exist):
- `EnhancedSimpleLSTM` - exists but not registered (!)
- `BiLSTMMaskedAutoencoder` - pre-training only, should not be in registry
- `FeatureAwareBiLSTMMaskedAutoencoder` - pre-training only

---

## 4. Data & Split Strategy Analysis

### 4.1 Split Implementation
- **Location:** `src/moola/utils/splits.py` (main utilities)
- **Strategy:** STRATIFIED K-FOLD (NOT temporal forward-chaining)
- **Function:** `make_splits()` uses `StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)`
- **Purge Logic:** NONE (no temporal gap between train/val)
- **Issue:** Using stratified random splits for FINANCIAL TIME SERIES is problematic:
  - Ignores temporal dependencies (past prices predict future prices)
  - No embargo/purge logic prevents look-ahead bias
  - Random shuffling breaks temporal coherence

### 4.2 Train/Test Split (CLI)
- **Location:** `src/moola/cli.py` lines 254-262
- **Strategy:** STRATIFIED RANDOM (80/20 split)
- **Code:**
  ```python
  X_train, X_test, y_train, y_test, ... = train_test_split(
      X, y, expansion_start, expansion_end, 
      test_size=0.2, 
      random_state=cfg.seed, 
      stratify=y  # Stratified by class
  )
  ```
- **Issue:** Random train/test split on time series data (even with class stratification) creates look-ahead bias

### 4.3 Validation Split (Models)
- **Location:** `src/moola/models/simple_lstm.py` lines 444-450
- **Strategy:** STRATIFIED RANDOM (15% validation split)
- **Code:**
  ```python
  X_train, X_val, y_train, y_val = train_test_split(
      X, y, test_size=0.15, random_state=self.seed, stratify=y
  )
  ```
- **Issue:** Same temporal bias issue

### 4.4 K-Fold Evaluation (CLI)
- **Location:** `src/moola/cli.py` lines 441-442
- **Strategy:** STRATIFIED K-FOLD (shuffle=True)
- **Code:**
  ```python
  skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.seed)
  ```

### 4.5 Split Persistence
- **Location:** `src/moola/utils/splits.py` lines 39-60
- **Feature:** Splits saved to `fold_{i}.json` with manifest (train_idx, val_idx, seed, fold number)
- **Advantage:** Reproducible, comparable across experiments

### Recommendation for Refactor
**CRITICAL:** Refactor splits to use **forward-chaining (TimeSeriesSplit) with embargo logic**:
1. Replace `StratifiedKFold(shuffle=True)` with `TimeSeriesSplit()`
2. Add embargo period (e.g., 10% gap between train and test)
3. For small datasets, consider:
   - Walk-forward validation (instead of K-fold)
   - Temporal anchoring (keep class distribution but respect temporal order)
4. This is essential for financial time series to avoid look-ahead bias

---

## 5. Data Loading & Processing Pipeline

### 5.1 Entry Points
1. **CLI Ingest:** `cli.py` lines 39-112
   - Loads raw data from parquet or generates synthetic
   - Validates schema
   - Outputs to `data/processed/train.parquet`

2. **CLI Train:** `cli.py` lines 115-317
   - Loads `data/processed/train.parquet`
   - Uses `DualInputDataProcessor` for feature engineering
   - Applies pseudo-sample augmentation (optional)
   - Prepares model inputs
   - Trains model and saves to `artifacts/models/{model}/model.pkl`

### 5.2 Dual-Input Pipeline
- **File:** `src/moola/data/dual_input_pipeline.py` (28 KB)
- **Purpose:** Combines raw OHLC (105×4) with engineered features
- **Config:** `FeatureConfig` dataclass with:
  - `use_raw_ohlc`: Always enabled
  - `use_small_dataset_features`: 25-30 optimized features
  - `use_price_action_features`: Multi-scale indicators
  - `use_hopsketch_features`: 1575 features (optional, for XGBoost)
  - `enable_augmentation`: Pseudo-sample generation (controlled)

### 5.3 Feature Engineering
- **Small Dataset Features:** `src/moola/features/small_dataset_features.py` (30 KB)
  - Optimized indicators for 98-sample dataset
  - 25-30 features
  
- **Price Action Features:** `src/moola/features/price_action_features.py` (31 KB)
  - Multi-scale technical indicators
  - 21 features from engineer_multiscale_features()
  - Optional HopSketch (1575 features)

### 5.4 Pseudo-Sample Augmentation
- **Location:** `src/moola/utils/pseudo_sample_generation.py`
- **Pipeline:** `PseudoSampleGenerationPipeline`
- **Strategies:** Temporal augmentation + pattern-based synthesis
- **Quality Control:** Accepts only samples above quality_threshold (default 0.7)
- **Config in CLI:** Lines 123-127
  ```
  --augment-data (enable/disable)
  --augmentation-ratio 2.0 (default: 2:1 synthetic:real)
  --max-synthetic-samples 210 (max generation)
  --quality-threshold 0.7 (minimum quality)
  ```

### 5.5 Expansion Indices
- **Purpose:** Marks the expansion window start/end for each sample
- **Validation:** `src/moola/data/load.py` validates expansion indices
- **Requirements:** expansion_start < expansion_end, in range [30, 74]

### 5.6 Data Validation
- **Location:** `src/moola/utils/data_validation.py`
- **Features:** Schema validation, drift detection
- **Schemas:** Located in `src/moola/schemas/canonical_v1.py`

---

## 6. Current Training Workflow

### 6.1 CLI Command Structure
**Training:**
```bash
python -m moola.cli train \
  --model simple_lstm \
  --device cpu \
  --use-engineered-features \
  --augment-data \
  --augmentation-ratio 2.0
```

**Available Flags:**
- `--model`: logreg, rf, xgb, simple_lstm, rwkv_ts, cnn_transformer, stack
- `--device`: cpu | cuda
- `--use-engineered-features`: Enable feature engineering
- `--max-engineered-features`: Limit to N features
- `--augment-data`: Enable pseudo-sample augmentation
- `--augmentation-ratio`: Synthetic:real ratio (default 2.0)
- `--max-synthetic-samples`: Max synthetic samples (default 210)
- `--quality-threshold`: Min quality score (default 0.7)

### 6.2 Pre-training Commands
**Masked BiLSTM Pre-training:**
```bash
python -m moola.cli pretrain-bilstm \
  --device cuda \
  --epochs 50 \
  --n-epochs 50
```

**TS-TCC Pre-training:**
```bash
python -m moola.cli pretrain-tcc \
  --device cuda \
  --epochs 100
```

### 6.3 Evaluation Command
**K-Fold Cross-Validation:**
```bash
python -m moola.cli evaluate \
  --model simple_lstm \
  --device cuda
```
- Runs stratified K-fold (default k=5)
- Logs: accuracy, precision, recall, f1, confusion matrix
- Persists: results, confusion matrix plots

### 6.4 Out-of-Fold (OOF) Validation
**Location:** `cli.py` lines 550-653
**Purpose:** Base learner training for stacking ensemble
- Trains multiple base models
- Generates OOF predictions for meta-learner
- Supports pre-trained encoder loading

### 6.5 Results Logging
- **Class:** `src/moola/utils/results_logger.py` (ResultsLogger)
- **Format:** JSON lines file (`experiment_results.jsonl`)
- **Records:** timestamp, phase, experiment_id, metrics, config
- **Location:** `experiment_results.jsonl` (project root)
- **NO manifest/run tracking** (simple append-only logging)

### 6.6 Model Persistence
- **Format:** Pickle (`.pkl`)
- **Location:** `artifacts/models/{model}/model.pkl`
- **Feature Metadata:** `artifacts/models/{model}/feature_metadata.json` (if features used)
- **Pre-trained Encoders:** `artifacts/pretrained/{encoder_name}.pt` (PyTorch)

---

## 7. File Inventory Summary

### Active Production Code
```
src/moola/models/
├── base.py (96 LOC, base interface)
├── simple_lstm.py (921 LOC, ACTIVE)
├── logreg.py (109 LOC, ACTIVE baseline)
├── rf.py (138 LOC, ACTIVE baseline)
├── xgb.py (287 LOC, ACTIVE baseline)
├── stack.py (87 LOC, ACTIVE meta-learner)
├── __init__.py (96 LOC, registry)
```

### Experimental/Pre-training Code (Candidates for models_extras/)
```
src/moola/models/
├── bilstm_masked_autoencoder.py (359 LOC, pre-training only)
├── feature_aware_bilstm_masked_autoencoder.py (471 LOC, experimental)
├── bilstm_autoencoder.py (403 LOC, deprecated)
├── ts_tcc.py (591 LOC, pre-training alternative)
├── cnn_transformer.py (1321 LOC, experimental, over-parameterized)
├── rwkv_ts.py (639 LOC, experimental, over-parameterized)
├── enhanced_simple_lstm.py (778 LOC, variant - NOT in registry!)
```

### Data Infrastructure
```
src/moola/data/
├── load.py (simple validation)
└── dual_input_pipeline.py (28 KB, feature processing)

src/moola/features/
├── small_dataset_features.py (30 KB)
├── price_action_features.py (31 KB)
└── feature_engineering.py (12 KB)

src/moola/utils/
├── splits.py (stratified K-fold + persistence)
├── data_validation.py (schema validation)
├── results_logger.py (JSON lines logging)
├── augmentation.py (Mixup, CutMix)
├── temporal_augmentation.py (jitter, scaling, time_warp)
├── pseudo_sample_generation.py (synthetic data)
```

---

## 8. Recent Development Focus

**Recent Commits (last 7 days):**
1. Oct 18: Monitoring infrastructure (prometheus, alerts)
2. Oct 18: Controlled pseudo-sample augmentation
3. Oct 18: Enhanced SimpleLSTM with BiLSTM encoder
4. Oct 18: Dual-input architecture with feature removal
5. Oct 17: Hyperparameter optimizations

**Pattern:** Heavy focus on:
- SimpleLSTM variants and optimization
- Pseudo-sample augmentation (controlled)
- Pre-trained encoder integration
- Feature engineering for small datasets

---

## 9. Refactor Recommendations

### Priority 1: Split Strategy (CRITICAL)
**Issue:** Current stratified random splits create look-ahead bias for time series
**Action:**
- [ ] Implement forward-chaining (TimeSeriesSplit)
- [ ] Add embargo logic (10% temporal gap between train/test)
- [ ] Update `src/moola/utils/splits.py`
- [ ] Update CLI train/evaluate to use temporal splits
- [ ] Update model internal validation splits

### Priority 2: Model Registry Cleanup
**Issue:** Inconsistent registration, experimental models in main registry
**Action:**
- [ ] Decide: Keep CNN-Transformer and RWKV-TS or move to models_extras/?
- [ ] Register EnhancedSimpleLSTM or remove it
- [ ] Remove pre-training models from main registry
- [ ] Add pre-training models to separate `pretraining_models/__init__.py`

### Priority 3: Code Organization
**Action:**
- [ ] Create `src/moola/models_extras/` for experimental/pre-training models
- [ ] Move: bilstm_autoencoder.py, bilstm_masked_autoencoder.py, feature_aware_bilstm_masked_autoencoder.py, ts_tcc.py
- [ ] Move: cnn_transformer.py and rwkv_ts.py (if not keeping for experimentation)
- [ ] Create separate import paths for pre-training vs. inference models

### Priority 4: Data Pipeline Documentation
**Action:**
- [ ] Document dual-input pipeline architecture
- [ ] Create example notebook for feature engineering
- [ ] Add validation checks for expansion indices
- [ ] Document pseudo-sample augmentation strategy

### Priority 5: Results Logging Enhancement (Optional)
**Action:**
- [ ] Add run manifest with model, data version, hyperparameters
- [ ] Link to git commit hash
- [ ] Add experiment tagging/grouping

---

## 10. Key Findings & Insights

### What's Working Well
1. **SimpleLSTM**: Clean, focused (921 params), well-integrated into CLI
2. **Pre-training Architecture**: BiLSTM masked autoencoder works for transfer learning
3. **Dual-Input Pipeline**: Flexible feature processing (OHLC + engineered)
4. **Pseudo-Sample Augmentation**: Controlled, quality-aware synthetic data generation
5. **Results Logging**: Simple JSON lines (no database overhead)
6. **Pre-commit Hooks**: Black, Ruff, isort enforcement working

### What Needs Improvement
1. **Split Strategy**: Random splits inappropriate for time series (look-ahead bias)
2. **Model Bloat**: CNN-Transformer (1321 LOC, 56K params) and RWKV-TS (639 LOC, 409K params) over-parameterized
3. **Registry Inconsistency**: EnhancedSimpleLSTM not registered, experimental models in production registry
4. **No Temporal Validation**: Training pipeline doesn't respect temporal ordering
5. **Pre-training Integration**: Pre-training models directly instantiated in CLI, not in registry

### Architecture Decision Points
1. **SimpleLSTM vs EnhancedSimpleLSTM**: Which is the production baseline? (EnhancedSimpleLSTM seems newer but not registered)
2. **Transfer Learning Strategy**: Continue pre-training or focus on feature engineering?
3. **Augmentation Philosophy**: Pseudo-sample generation effective? Should it be always-on or experimental?

---

## 11. Conclusion

**Current State:** Moola has clear production focus on SimpleLSTM with supporting infrastructure for pre-training and experimentation. Code is well-structured but needs refactoring for temporal correctness and model organization.

**Recommended Refactor Order:**
1. Fix split strategy (forward-chaining + embargo)
2. Move experimental models to models_extras/
3. Clean up registry (remove pre-training models)
4. Document data pipeline
5. Clarify SimpleLSTM vs EnhancedSimpleLSTM decision


---

## Appendix A: Model Dependency & Usage Graph

```
PRODUCTION INFERENCE FLOW:
=========================

train.parquet
    |
    v
DualInputDataProcessor
    |
    +-> raw OHLC (105×4)
    +-> engineered features (25-50)
    +-> expansion indices
    |
    v
Model Selection (CLI flag)
    |
    +---> simple_lstm (921 params)     [ACTIVE]
    |
    +---> logreg, rf, xgb             [ACTIVE baselines]
    |         |
    |         v
    |      Stack (meta-learner)       [ACTIVE]
    |
    +---> enhanced_simple_lstm        [EXPERIMENTAL - not wired]
    |
    +---> cnn_transformer (56K)       [EXPERIMENTAL - over-params]
    |
    +---> rwkv_ts (409K)              [EXPERIMENTAL - over-params]

PRE-TRAINING & TRANSFER LEARNING:
=================================

unlabeled OHLC (105×4)
    |
    v
PretrainBiLSTMMaskedAutoencoder (359 LOC)  [experimental]
    |
    v
Pretrained encoder saved to artifacts/pretrained/
    |
    v
Simple LSTM fit() loads encoder              [transfer learning]

ALTERNATIVE PRE-TRAINING:
TS-TCC (591 LOC)  [alternative, experimental]
    |
    v
Contrastive pretrained encoder


FEATURE ENGINEERING FLOW:
========================

OHLC sequences
    |
    +-> SmallDatasetFeatureEngineer
    |       |
    |       v
    |   25-30 optimized features
    |
    +-> PriceActionFeatures
    |       |
    |       v
    |   21 multi-scale indicators
    |
    +-> [Optional] HopSketch
            |
            v
        1575 features (for XGBoost)


EXPERIMENTAL COMPONENTS (candidates for models_extras/):
========================================================

bilstm_masked_autoencoder.py (359 LOC)
    └-> used only in pretrain-bilstm CLI command

feature_aware_bilstm_masked_autoencoder.py (471 LOC)
    └-> used in experimental utils/feature_aware_utils.py

bilstm_autoencoder.py (403 LOC)
    └-> deprecated, superseded by masked version

ts_tcc.py (591 LOC)
    └-> used only in pretrain-tcc CLI command

cnn_transformer.py (1321 LOC)
    └-> in registry but over-parameterized (56K params)

rwkv_ts.py (639 LOC)
    └-> in registry but severely over-parameterized (409K params)

enhanced_simple_lstm.py (778 LOC)
    └-> NOT in registry, variant of SimpleLSTM
```

---

## Appendix B: Code Metrics Summary

| Model | File | LOC | Size | Status | Params | Registry | CLI |
|-------|------|-----|------|--------|--------|----------|-----|
| SimpleLSTM | simple_lstm.py | 921 | 40K | ACTIVE | 921 | YES | YES |
| EnhancedSimpleLSTM | enhanced_simple_lstm.py | 778 | 31K | ACTIVE (variant) | ~17K | NO | NO |
| LogReg | logreg.py | 109 | 4K | ACTIVE | N/A | YES | YES |
| RandomForest | rf.py | 138 | 5K | ACTIVE | N/A | YES | YES |
| XGBoost | xgb.py | 287 | 11K | ACTIVE | N/A | YES | YES |
| Stack | stack.py | 87 | 3K | ACTIVE | N/A | YES | YES |
| BiLSTM MA | bilstm_masked_autoencoder.py | 359 | 12K | EXPERIMENTAL | N/A | NO | YES (pretrain) |
| Feature-Aware BiLSTM MA | feature_aware_bilstm_masked_autoencoder.py | 471 | 18K | EXPERIMENTAL | N/A | NO | NO |
| BiLSTM Autoencoder | bilstm_autoencoder.py | 403 | 13K | DEPRECATED | N/A | NO | NO |
| TS-TCC | ts_tcc.py | 591 | 21K | EXPERIMENTAL | N/A | NO | YES (pretrain) |
| CNN-Transformer | cnn_transformer.py | 1321 | 56K | EXPERIMENTAL | 56K | YES | YES |
| RWKV-TS | rwkv_ts.py | 639 | 25K | EXPERIMENTAL | 409K | YES | YES |

---

## Appendix C: CLI Commands Reference

**Training:**
```bash
python -m moola.cli train --model simple_lstm --device cpu
python -m moola.cli train --model simple_lstm --device cuda --use-engineered-features
python -m moola.cli train --model simple_lstm --augment-data --augmentation-ratio 2.0
```

**Evaluation:**
```bash
python -m moola.cli evaluate --model simple_lstm --device cuda
```

**Pre-training:**
```bash
python -m moola.cli pretrain-bilstm --device cuda --epochs 50
python -m moola.cli pretrain-tcc --device cuda --epochs 100
```

**Out-of-Fold (Stacking):**
```bash
python -m moola.cli oof --seed 1337 --device cuda
```

---

## Appendix D: Recent Git History (Model-Related)

```
7cf53ec feat: add comprehensive monitoring infrastructure with production alerts and workflows
224551b feat: implement controlled pseudo-sample augmentation for small dataset enhancement
da5cc2d feat: complete Enhanced SimpleLSTM with pre-trained BiLSTM encoder integration
70f6039 feat: enhanced SimpleLSTM with dual-input architecture and traditional indicator removal
b8d1a78 feat: verify hyperparameter optimizations with confirmed augmentation
fd99272 refactor: comprehensive project consolidation and SSH/SCP workflow transition
1044318 fix: update SimpleLSTM for pre-training compatibility
bf30628 fix: correct encoder weight key mapping for bidirectional LSTM to unidirectional SimpleLSTM
7c9b1bb feat: complete bidirectional Masked LSTM pre-training implementation
9c224e3 feat: complete ML pipeline audit, encoder fixes, and RunPod orchestration
```

Pattern: Heavy focus on SimpleLSTM optimization, augmentation, and pre-training integration.

