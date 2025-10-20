# Phase 0: Metrics & Validation Survey
**Date:** 2025-10-18
**Purpose:** Analyze current metrics computation, validation guards, logging infrastructure, and pretrained loading logic to inform Phase 3 refactor

---

## Executive Summary

**Current State:** Moola has **partial metrics infrastructure** with basic accuracy/F1 tracking but **lacks comprehensive validation guards** and **systematic calibration analysis**.

**Critical Findings:**
- ✅ Metrics computation exists (accuracy, F1, precision, recall, ECE)
- ❌ **NO reliability diagrams** (calibration plots) generated
- ❌ **NO validation guards** against data leakage (random splits, synthetic contamination)
- ✅ Pretrained encoder loading implemented with shape validation
- ⚠️ Logging infrastructure exists but lacks **mandatory lineage tracking**

**Recommendation:** Phase 3 must implement validation guards BEFORE metrics improvements. Data integrity > metrics quality.

---

## 1. Current Metrics Computation

### 1.1 Computed Metrics

| Metric | Status | Location | Saved? |
|--------|--------|----------|--------|
| **Accuracy** | ✅ YES | `/src/moola/utils/metrics.py:74` | YES (metrics.json) |
| **Class-wise F1** | ✅ YES | `/src/moola/utils/metrics.py:77` (macro-averaged) | YES (metrics.json) |
| **Precision** | ✅ YES | `/src/moola/utils/metrics.py:75` (macro-averaged) | YES (metrics.json) |
| **Recall** | ✅ YES | `/src/moola/utils/metrics.py:76` (macro-averaged) | YES (metrics.json) |
| **PR-AUC** | ❌ NO | Not implemented | N/A |
| **Brier Score** | ❌ NO | Not implemented | N/A |
| **ECE (Calibration)** | ✅ YES | `/src/moola/utils/metrics.py:7-53` | YES (inline, not saved to metrics.json) |
| **Confusion Matrix** | ✅ YES | `/src/moola/cli.py:485` (evaluate command) | YES (confusion_matrix.csv) |
| **Log Loss** | ✅ YES | `/src/moola/utils/metrics.py:83` | YES (metrics.json) |

**Key Function:** `calculate_metrics()` in `/src/moola/utils/metrics.py:56-88`
```python
def calculate_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Calculate comprehensive evaluation metrics.

    Returns:
        - accuracy: Overall accuracy
        - precision: Macro-averaged precision
        - recall: Macro-averaged recall
        - f1: Macro-averaged F1 score
        - ece: Expected calibration error (if y_proba provided)
        - logloss: Log loss (if y_proba provided)
    """
```

### 1.2 Visualizations

| Visualization | Status | Location | Saved? |
|---------------|--------|----------|--------|
| **Confusion Matrix Plot** | ❌ NO | Not implemented | N/A |
| **Reliability Diagram** | ❌ NO | Not implemented | N/A |
| **PR Curve** | ❌ NO | Not implemented | N/A |
| **ROC Curve** | ❌ NO | Not implemented | N/A |
| **Calibration Curve** | ❌ NO | Not implemented | N/A |

**Finding:** ECE is **computed** but **not visualized**. No reliability diagrams exist.

**Evidence:**
- ECE calculation exists: `/src/moola/utils/metrics.py:7-53`
- No matplotlib/seaborn plotting code for calibration in codebase
- Pseudo-sample validation has plots (`/src/moola/utils/pseudo_sample_validation.py:767-950`) but NOT for model calibration

### 1.3 Metrics Storage

**Format:** JSON files + CSV
```
artifacts/
├── models/{model_name}/
│   ├── metrics.json          # Accuracy, F1, precision, recall, CV fold details
│   ├── confusion_matrix.csv  # Raw confusion matrix
│   └── model.pkl            # Trained model
├── runs.csv                  # Run tracking (run_id, model, git_sha, accuracy, f1, duration)
└── manifest.json            # Artifact hashes + git SHA
```

**Example metrics.json structure (from `/src/moola/cli.py:494-503`):**
```json
{
  "model": "simple_lstm",
  "accuracy": 0.87,
  "f1": 0.85,
  "precision": 0.86,
  "recall": 0.84,
  "cv_folds": 5,
  "timestamp": "2025-10-18T12:34:56Z",
  "fold_details": [...]
}
```

**Missing from metrics.json:**
- ❌ ECE (computed but not saved)
- ❌ Brier score
- ❌ PR-AUC
- ❌ Class-wise metrics (only macro-averaged)

---

## 2. Validation & Guards

### 2.1 Existing Guards

| Guard | Status | Location | Action |
|-------|--------|----------|--------|
| **OHLC Integrity Check** | ✅ YES | `/src/moola/utils/pseudo_sample_validation.py:88-133` | WARN (quality score) |
| **KS Test (Synthetic Quality)** | ✅ YES | `/src/moola/utils/pseudo_sample_validation.py:161-163` | WARN (threshold check) |
| **Expansion Index Validation** | ✅ YES | `/src/moola/data/load.py:validate_expansions()` (referenced in CLI) | ABORT (removes invalid rows) |
| **Split Index Overlap Detection** | ❌ NO | Not implemented | N/A |
| **Schema Validation** | ✅ YES | `/src/moola/cli.py:67-70` (canonical_v1 check) | ABORT on invalid schema |

**Details:**

#### 2.1.1 OHLC Integrity Check (Pseudo-Sample Validation)
- **Location:** `/src/moola/utils/pseudo_sample_validation.py:88-133`
- **What it validates:**
  - OHLC relationships (O ≤ H, O ≥ L, C ≤ H, C ≥ L, H ≥ L)
  - Negative price detection
- **Action:** Returns integrity score (0-1), **does NOT abort**
- **Threshold:** 0.90-0.95 (configurable)
- **Issue:** This is only for pseudo-samples, NOT for main training data

#### 2.1.2 KS Test (Distribution Similarity)
- **Location:** `/src/moola/utils/pseudo_sample_validation.py:161-163`
- **What it validates:** Return distribution similarity between real and synthetic data
- **Metric:** `ks_similarity = 1.0 - ks_statistic`
- **Threshold:** 0.70-0.80
- **Action:** WARN (recommendations), does NOT abort

#### 2.1.3 Expansion Index Validation
- **Referenced in:** `/src/moola/cli.py:604-605`
```python
from moola.data.load import validate_expansions
df = validate_expansions(df)  # Removes invalid expansion indices
```
- **Action:** ABORT (removes invalid rows)

#### 2.1.4 Schema Validation
- **Location:** `/src/moola/cli.py:67-70`
```python
from .schemas.canonical_v1 import check_training_data
if check_training_data(df):
    log.info("✅ Dataset schema validation passed")
else:
    log.error("❌ Dataset schema validation failed")
    raise ValueError("Invalid dataset schema")
```
- **Action:** ABORT on failure

### 2.2 Missing Guards (CRITICAL)

| Guard | Priority | Risk | Implementation Needed |
|-------|----------|------|----------------------|
| **Forbid Random Splits** | 🔴 CRITICAL | Data leakage | Detect `train_test_split` without `random_state` or `stratify` |
| **Val/Test Synthetic Contamination** | 🔴 CRITICAL | Inflated metrics | Check `is_synthetic` flag, ABORT if synthetic in val/test |
| **Pretrained Load Validation (>80% match)** | 🟡 HIGH | Silent failure | Verify tensor match rate, ABORT if <80% |
| **Synthetic KS p-value Threshold** | 🟡 HIGH | Bad synthetic data | ABORT if KS p-value < 0.05 (distributions differ) |
| **Split Index Overlap Detection** | 🟢 MEDIUM | Train/val/test leak | Check set intersection, ABORT if overlap found |

**Evidence of Missing Guards:**

1. **No random split detection:**
   - `train_test_split` used in `/src/moola/cli.py:254-262` with `random_state=cfg.seed`
   - ✅ Good: Uses random_state
   - ❌ Bad: No validation that random_state is always set

2. **No synthetic contamination checks:**
   - Pseudo-sample code in `/src/moola/data/dual_input_pipeline.py` has augmentation
   - ❌ No code checks if synthetic samples leak into validation/test sets
   - ❌ No `is_synthetic` column tracking in final datasets

3. **Pretrained loading has partial validation:**
   - ✅ Shape validation exists: `/src/moola/models/simple_lstm.py:880-889`
   - ✅ Key matching exists: `/src/moola/models/simple_lstm.py:863-891`
   - ❌ NO enforcement of minimum match threshold (e.g., >80%)
   - ⚠️ Warning logged but no ABORT: Line 887-889

---

## 3. Logging & Manifests

### 3.1 Run Tracking

**Status:** ✅ YES (multiple formats)

**Formats:**
1. **experiment_results.jsonl** (JSON Lines)
   - Location: `/Users/jack/projects/moola/experiment_results.jsonl`
   - Format: One JSON object per line
   - Fields: `exp_id, name, dropout, heads, batch_size, test_accuracy, pred_hash, config_hash`
   - Example:
   ```json
   {"exp_id": 1, "name": "Baseline", "dropout": 0.1, "heads": 2, "batch_size": 16, "test_accuracy": 0.55, "pred_hash": "03c2cea35752b7bb36eebcf1bcc69998", "config_hash": "f3b7aab744a19d4e6c67cfc1aeb6692c"}
   ```

2. **runs.csv** (Tabular)
   - Location: `artifacts/runs.csv`
   - Fields: `run_id, model, git_sha, accuracy, f1, duration`
   - Written by: `/src/moola/cli.py:536-553`

3. **ResultsLogger (programmatic)**
   - Location: `/src/moola/utils/results_logger.py`
   - Format: JSON Lines
   - Fields: `timestamp, phase, experiment_id, metrics, config`
   - Usage: Phase-based experiment tracking

### 3.2 Manifest Format

**Status:** ✅ YES (comprehensive)

**Location:** `/Users/jack/projects/moola/data/artifacts/manifest.json`

**Structure:**
```json
{
  "created_at": "2025-10-16T16:29:12.473030Z",
  "git_sha": "9d1f686",
  "models": ["logreg", "rf", "xgb", "stack"],
  "artifacts": {
    "models/logreg/model.pkl": "4903f27774d46823fc006807106ad683b4917bac34aa68609a59604aba7b3364",
    "oof/logreg/v1/seed_1337.npy": "a1a8683bfd7ddcb6eec7cf8e7764c05801073ddd87d2c7ca000ba739b1fb9d42",
    ...
  }
}
```

**Features:**
- ✅ Git SHA logged
- ✅ SHA256 hashes for artifact integrity
- ✅ Timestamp
- ✅ Model list

**Utility Functions:**
- `create_manifest()` - `/src/moola/utils/manifest.py:60-90`
- `verify_manifest()` - `/src/moola/utils/manifest.py:121-150`

### 3.3 Missing Lineage Tracking

| Lineage Aspect | Status | Impact |
|----------------|--------|--------|
| **Data source path** | ❌ NO | Can't trace which data version was used |
| **Synthetic augmentation metadata** | ⚠️ PARTIAL | Logged during training, not in manifest |
| **Pretrained encoder source** | ❌ NO | Can't trace encoder provenance |
| **Feature engineering config** | ⚠️ PARTIAL | Saved in `feature_metadata.json`, not in manifest |
| **Split indices** | ✅ YES | Saved in `artifacts/splits/v1/fold_*.json` |

**Evidence:**
- Augmentation metadata logged: `/src/moola/cli.py:201-217`
- NOT added to manifest.json
- Feature metadata saved separately: `/src/moola/cli.py:302-314`

### 3.4 Current Artifacts Structure

```
artifacts/
├── manifest.json                     # Artifact hashes + git SHA
├── runs.csv                         # Run tracking table
├── confusion_matrix.csv             # Latest confusion matrix
├── models/
│   ├── {model_name}/
│   │   ├── model.pkl               # Trained model
│   │   ├── metrics.json            # Accuracy, F1, precision, recall
│   │   ├── confusion_matrix.csv    # Model-specific confusion matrix
│   │   └── feature_metadata.json   # Feature engineering config (if used)
├── oof/
│   ├── {model_name}/
│   │   └── v1/
│   │       └── seed_{seed}.npy     # Out-of-fold predictions
├── splits/
│   └── v1/
│       └── fold_{i}.json           # Split indices + metadata
└── pretrained/
    └── bilstm_encoder.pt           # Pretrained encoder weights
```

**Observations:**
- ✅ Well-organized structure
- ✅ Version control for splits/OOF (`v1/`)
- ❌ No plots/ subdirectory (no visualizations saved)
- ❌ No data_lineage/ subdirectory

---

## 4. Pretrained Loading

### 4.1 Implementation Status

**Status:** ✅ YES (comprehensive)

**Location:** `/src/moola/models/simple_lstm.py:806-921`

**Key Function:** `load_pretrained_encoder(encoder_path, freeze_encoder=True)`

### 4.2 Features

| Feature | Status | Details |
|---------|--------|---------|
| **Tensor name matching** | ✅ YES | Maps encoder keys to `ohlc_encoder.*` keys |
| **Mismatch detection** | ✅ YES | Logs warnings for missing/mismatched keys |
| **Shape validation** | ✅ YES | Checks tensor shapes match (line 880-889) |
| **Encoder freezing** | ✅ YES | Freezes LSTM parameters if `freeze_encoder=True` |
| **Match stats logging** | ✅ YES | Logs loaded/skipped keys (line 896-903) |
| **Layer count mismatch handling** | ✅ YES | Loads only first layer if pretrained has more layers (line 852-871) |

**Code Example (Tensor Matching):**
```python
# Line 863-891
for key in encoder_state_dict:
    # Only load layer 0 weights if layer count mismatch
    if pretrained_layers > self.num_layers:
        if "_l1" in key or "_l2" in key or "_l3" in key:
            skipped_keys.append(key)
            continue

    # Map encoder keys to Enhanced SimpleLSTM's OHLC encoder keys
    model_key = f"ohlc_encoder.{key}"

    if model_key in model_state_dict:
        # Verify shapes match
        encoder_shape = encoder_state_dict[key].shape
        model_shape = model_state_dict[model_key].shape

        if encoder_shape == model_shape:
            model_state_dict[model_key] = encoder_state_dict[key]
            loaded_keys.append(model_key)
        else:
            logger.warning(f"Shape mismatch for {model_key}: Expected {model_shape}, Got {encoder_shape}")
```

### 4.3 Validation Gaps

| Validation | Status | Impact |
|------------|--------|--------|
| **Minimum match threshold** | ❌ NO | Silent partial loading (e.g., 30% match is OK) |
| **Encoder architecture compatibility** | ⚠️ PARTIAL | Checks hidden_size, but not dropout/num_layers |
| **Checkpoint metadata validation** | ⚠️ PARTIAL | Loads hyperparams but doesn't enforce compatibility |
| **Frozen param verification** | ❌ NO | Doesn't verify params are actually frozen post-load |

**Critical Issue:**
```python
# Line 906-910
if len(loaded_keys) == 0:
    raise ValueError("Failed to load any weights from pre-trained encoder.")
```
- ✅ Detects total failure
- ❌ **Does NOT detect partial failure** (e.g., 30% match = silent success)

**Recommendation:** Add threshold check:
```python
match_rate = len(loaded_keys) / len(encoder_state_dict)
if match_rate < 0.80:  # Require 80% match
    raise ValueError(f"Insufficient weight match: {match_rate:.1%} < 80%")
```

### 4.4 Freezing Implementation

**Status:** ✅ YES (correct)

```python
# Line 913-919
if freeze_encoder:
    logger.info("Freezing OHLC encoder weights")
    for param in self.model.ohlc_encoder.parameters():
        param.requires_grad = False
    logger.info("OHLC encoder frozen. Only feature encoder and classifier will be trained initially.")
else:
    logger.info("OHLC encoder unfrozen. All parameters trainable.")
```

**Verification:** Frozen params are logged at training time (line 511-526)

---

## 5. Recommendations

### 5.1 Priority 1: Validation Guards (MUST IMPLEMENT)

**Before any metrics improvements, implement these guards:**

1. **Forbid Random Splits**
   - Location: Add to `/src/moola/utils/data_validation.py`
   - Logic: Check `train_test_split` has `random_state` and `stratify`
   - Action: ABORT if missing

2. **Val/Test Synthetic Contamination Prevention**
   - Location: Add to `/src/moola/data/dual_input_pipeline.py`
   - Logic: Track synthetic samples with `is_synthetic` column, ensure val/test are pure real
   - Action: ABORT if synthetic in val/test

3. **Pretrained Load Validation (>80% Match)**
   - Location: Enhance `/src/moola/models/simple_lstm.py:906-910`
   - Logic: `match_rate = len(loaded_keys) / len(encoder_state_dict)`
   - Action: ABORT if `match_rate < 0.80`

4. **Synthetic KS p-value Threshold**
   - Location: Enhance `/src/moola/utils/pseudo_sample_validation.py`
   - Logic: Compute KS test p-value, ABORT if p < 0.05
   - Action: ABORT (distributions significantly different)

### 5.2 Priority 2: Metrics Enhancements (AFTER Guards)

1. **Add Brier Score**
   - Implementation: `from sklearn.metrics import brier_score_loss`
   - Save to: `metrics.json`

2. **Add PR-AUC**
   - Implementation: `from sklearn.metrics import average_precision_score`
   - Save to: `metrics.json`

3. **Generate Reliability Diagrams**
   - Use: `sklearn.calibration.calibration_curve`
   - Save to: `artifacts/models/{model_name}/calibration_plot.png`

4. **Save Class-wise Metrics**
   - Compute F1/precision/recall per class (not just macro)
   - Save to: `metrics.json` under `class_metrics` key

### 5.3 Priority 3: Logging & Lineage

1. **Add Data Lineage to Manifest**
   - Fields: `data_source_path, data_sha256, synthetic_ratio, augmentation_config`

2. **Add Pretrained Encoder Provenance**
   - Fields: `pretrained_encoder_path, pretrained_encoder_sha256, match_rate`

3. **Create Plots Directory**
   - Structure: `artifacts/models/{model_name}/plots/`
   - Save: Confusion matrix heatmap, reliability diagram, PR curve, ROC curve

4. **Mandatory Git SHA Check**
   - ABORT training if `git status` shows uncommitted changes
   - Ensures reproducibility

---

## 6. Current vs. Ideal State

| Aspect | Current | Ideal |
|--------|---------|-------|
| **Metrics Computed** | Accuracy, F1, precision, recall, ECE | + Brier, PR-AUC, class-wise metrics |
| **Visualizations** | None | Confusion heatmap, reliability diagram, PR/ROC curves |
| **Validation Guards** | Schema validation only | + Random split check, synthetic contamination, pretrained match threshold |
| **Logging** | experiment_results.jsonl + runs.csv | + Comprehensive manifest with lineage |
| **Pretrained Loading** | Partial validation | + 80% match threshold enforcement |

---

## 7. Implementation Roadmap for Phase 3

### Week 1: Validation Guards
- [ ] Implement random split detector
- [ ] Implement synthetic contamination check
- [ ] Implement pretrained match threshold (>80%)
- [ ] Implement KS p-value threshold check
- [ ] Add unit tests for all guards

### Week 2: Metrics Enhancements
- [ ] Add Brier score computation
- [ ] Add PR-AUC computation
- [ ] Generate reliability diagrams (calibration plots)
- [ ] Generate confusion matrix heatmaps
- [ ] Save class-wise metrics to metrics.json

### Week 3: Logging & Lineage
- [ ] Extend manifest.json with data lineage
- [ ] Add pretrained encoder provenance
- [ ] Create plots/ directory structure
- [ ] Implement git SHA enforcement (ABORT on dirty state)
- [ ] Update documentation

---

## Appendix: Key File Locations

### Metrics
- `/src/moola/utils/metrics.py` - Main metrics computation
- `/src/moola/cli.py:334-555` - Evaluate command (K-fold CV + metrics)

### Validation
- `/src/moola/utils/pseudo_sample_validation.py` - Pseudo-sample quality checks
- `/src/moola/utils/data_validation.py` - Data validation utilities
- `/src/moola/schemas/canonical_v1.py` - Schema validation

### Logging
- `/src/moola/utils/results_logger.py` - Phase-based experiment tracking
- `/src/moola/utils/manifest.py` - Manifest creation/verification
- `/src/moola/cli.py:536-553` - runs.csv tracking

### Pretrained Loading
- `/src/moola/models/simple_lstm.py:806-921` - load_pretrained_encoder()
- `/src/moola/pretraining/masked_lstm_pretrain.py` - BiLSTM pretraining

### Splits
- `/src/moola/utils/splits.py` - Deterministic K-fold split generation

---

**END OF SURVEY**
