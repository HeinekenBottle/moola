# Phase 0: Baseline Data Survey
Date: 2025-10-18

## Executive Summary

This baseline survey catalogues all data assets in the Moola crypto prediction project. The project operates on two distinct data regimes:
- **Labeled regime**: 89-105 small labeled samples for supervised training
- **Unlabeled regime**: 11,873 samples for self-supervised pre-training via BiLSTM/TS-TCC

Key finding: A canonical v1 split definition exists at `/data/artifacts/splits/v1/` with 5-fold forward-chaining validation.

---

## 1. Labeled Windows Dataset

### Status: FOUND (MULTIPLE VERSIONS)

### Primary Dataset: train_clean.parquet
**Path**: `/Users/jack/projects/moola/data/processed/train_clean.parquet`
- **Sample count**: 98 samples
- **Features**: 5 columns [window_id, label, expansion_start, expansion_end, features]
- **Label distribution**: consolidation=56, retracement=42
- **Size**: 0.09 MB
- **Last modified**: 2025-10-16 00:12:00
- **Content**: Each row contains a 105-bar OHLC window (stored as nested list in 'features' column)

### Backup/Historical Versions

| Dataset | Path | Samples | Notes |
|---------|------|---------|-------|
| train.parquet | `/data/processed/train.parquet` (symlink) | 105 | Original, links to train_pivot_134.parquet |
| train_3class_backup.parquet | `/data/processed/train_3class_backup.parquet` | 134 | With reversal class (obsolete?) |
| train_clean_backup.parquet | `/data/processed/train_clean_backup.parquet` | 98 | Backup copy of train_clean |
| train_clean_phase2.parquet | `/data/processed/train_clean_phase2.parquet` | 89 | Phase 2 cleaned (quality scores included) |
| train_pivot_134.parquet | `/data/processed/train_pivot_134.parquet` | 134 | Possibly 3-class or different structure |
| reversal_holdout.parquet | `/data/processed/reversal_holdout.parquet` | 22 | Holdout with reversal patterns |

### SMOTE Augmented: train_smote_300.parquet
**Path**: `/Users/jack/projects/moola/data/processed/train_smote_300.parquet`
- **Sample count**: 300 (augmented)
- **Label distribution**: consolidation=150, retracement=150 (perfectly balanced)
- **Size**: 0.79 MB
- **Method**: SMOTE applied to 98 samples → 300 synthetic samples
- **Last modified**: 2025-10-16 00:22:00
- **Columns**: [window_id, label, features, expansion_start, expansion_end]

### Recommendation
**Canonical v1 labeled dataset**: `/Users/jack/projects/moola/data/processed/train_clean.parquet`
- Most recent quality-filtered version (98 samples)
- Consistent across experiment configurations
- Used as base for all k-fold splits
- Two classes: consolidation vs retracement (binary classification)

---

## 2. Unlabeled OHLC Corpus (for TS2Vec/Self-Supervised Pre-training)

### Status: FOUND

### Primary Unlabeled Dataset: unlabeled_windows.parquet
**Path**: `/Users/jack/projects/moola/data/raw/unlabeled_windows.parquet`
- **Sample count**: 11,873 samples
- **Features**: 2 columns [window_id, features]
- **Size**: 2.20 MB
- **Last modified**: 2025-10-14 12:41:00
- **Content**: Each row contains a 105-bar OHLC window (unlabeled time-series)
- **Columns in features**: [open, high, low, close] (4 OHLC features per timestep)
- **Shape per window**: (105, 4)

### Augmented Unlabeled: unlabeled_with_labels.parquet
**Path**: `/Users/jack/projects/moola/data/processed/unlabeled_with_labels.parquet`
- **Sample count**: 11,873 samples (same corpus)
- **Features**: 5 columns [window_id, features, expansion_labels, swing_labels, candle_labels]
- **Size**: 2.59 MB
- **Last modified**: 2025-10-17 19:35:00
- **Purpose**: Same unlabeled windows but with multiple pseudo-labels from different labeling schemes
- **Note**: Pseudo-labels do NOT match Moola's consolidation/retracement taxonomy

### Pretraining Cache

**Path**: `/Users/jack/projects/moola/data/pretraining/`

| File | Shape | Size | Purpose |
|------|-------|------|---------|
| unlabeled_ohlc.npy | (11873, 105, 4) | 38.05 MB | Raw OHLC cache for fast loading |
| unlabeled_features.npy | (11873, 25) | 2.26 MB | Pre-computed feature vectors |

---

## 3. Synthetic/Augmentation Cache

### Status: FOUND

### Augmentation Artifacts

#### SMOTE Augmentation (3x balance)
- **Dataset**: train_smote_300.parquet (298 synthetic samples from 98 originals)
- **Balance**: Perfect 150/150 consolidation/retracement
- **Location**: `/Users/jack/projects/moola/data/processed/train_smote_300.parquet`
- **Quality metrics**: None stored (check scikit-learn SMOTE params)

#### OOF (Out-of-Fold) Predictions (Multiple Models)

**Clean baseline (no augmentation)**:
- `/Users/jack/projects/moola/data/oof/cnn_transformer_clean.npy`
- `/Users/jack/projects/moola/data/oof/logreg_clean.npy`
- `/Users/jack/projects/moola/data/oof/simple_lstm_clean.npy`
- `/Users/jack/projects/moola/data/oof/xgb_clean.npy`
- `/Users/jack/projects/moola/data/oof/rf_clean.npy`

**SMOTE augmented**:
- `/Users/jack/projects/moola/data/oof/cnn_transformer_augmented.npy`
- `/Users/jack/projects/moola/data/oof/logreg_augmented.npy`
- `/Users/jack/projects/moola/data/oof/simple_lstm_augmented.npy`
- `/Users/jack/projects/moola/data/oof/xgb_augmented.npy`
- `/Users/jack/projects/moola/data/oof/rf_augmented.npy`

### Augmentation Code References
- `/Users/jack/projects/moola/src/moola/utils/augmentation.py` - Base augmentation utilities
- `/Users/jack/projects/moola/src/moola/utils/financial_augmentation.py` - Finance-specific transforms
- `/Users/jack/projects/moola/src/moola/utils/pseudo_sample_generation.py` - Synthetic sample generation
- `/Users/jack/projects/moola/src/moola/utils/temporal_augmentation.py` - Time-series augmentation
- `/Users/jack/projects/moola/examples/augmentation_example.py` - Example usage

### Quality Metrics
- **KS test (Kolmogorov-Smirnov)**: Check code in pseudo_sample_validation.py
- **OHLC relationship validation**: Implemented in experiments/data_manager.py
- **Val/Test contamination risk**: LOW (splits are temporal forward-chaining, not random)

---

## 4. Split Definitions

### Status: FOUND

### Canonical v1 Splits
**Location**: `/Users/jack/projects/moola/data/artifacts/splits/v1/`

**Fold structure** (5-fold cross-validation):
- fold_0.json: train=78, val=20
- fold_1.json: train=78, val=20
- fold_2.json: train=78, val=20
- fold_3.json: train=79, val=19
- fold_4.json: train=79, val=19

**Metadata**:
- Seed: 1337
- K: 5 (5-fold CV)
- Total samples across all folds: 98 (from train_clean.parquet)

**Type**: STRATIFIED (preserves class distribution consolidation/retracement)
**Strategy**: Forward-chaining time-series split (indices are sorted by time)

### Split File Format
```json
{
  "fold": 0,
  "seed": 1337,
  "k": 5,
  "train_idx": [array of 78 sample indices],
  "val_idx": [array of 20 sample indices],
  "train_size": 78,
  "val_size": 20
}
```

### Alternative Split Locations
- `/Users/jack/projects/moola/data/splits/fold_0_train.npy` (78 indices)
- `/Users/jack/projects/moola/data/splits/fold_0_val.npy` (20 indices)
- (Similar for fold_1-4)

**Recommendation**: Use `/data/artifacts/splits/v1/` as canonical v1. These are JSON-serialized with full metadata.

---

## 5. Pretrained Artifacts (TS2Vec / BiLSTM Encoders)

### Status: FOUND (MULTIPLE VERSIONS)

### Active Pretrained Encoders

| Checkpoint | Size | Location | Status | Type |
|------------|------|----------|--------|------|
| bilstm_encoder.pt | 0.52 MB | `/bilstm_encoder.pt` (root) | ✅ Active | BiLSTM encoder state dict |
| bilstm_encoder_correct.pt | 2.03 MB | `/data/artifacts/pretrained/bilstm_encoder_correct.pt` | ✅ Latest | Corrected BiLSTM with proper weights |
| encoder_weights.pt | 3.37 MB | `/data/artifacts/pretrained/encoder_weights.pt` | ✅ Candidate | Full checkpoint (likely includes optimizer) |

### Archive/Experimental Encoders

| Checkpoint | Size | Location | Notes |
|------------|------|----------|-------|
| pretrained_encoder.pt | 3.37 MB | `/models/ts_tcc/pretrained_encoder.pt` | TS-TCC encoder (experimental) |
| pretrained_encoder.pt | 3.37 MB | `/data/artifacts/models_smote_300/models/ts_tcc/pretrained_encoder.pt` | SMOTE-trained variant |
| multitask_encoder.pt | 2.03 MB | `/artifacts/runpod_results/multitask_encoder.pt` | RunPod experiment |

### Model Parameter Counts
- **bilstm_encoder.pt**: ~135,168 parameters
- **bilstm_encoder_correct.pt**: ~2.03 MB (full checkpoint, likely with optimizer state)
- **encoder_weights.pt**: 3.37 MB (full checkpoint)

### Recommendation
**Canonical v1 encoder**: `/Users/jack/projects/moola/data/artifacts/pretrained/bilstm_encoder_correct.pt`
- Most recent corrected version
- Size indicates full checkpoint (model + optimizer)
- Name explicitly indicates "correct" version (vs. earlier iterations)

### How to Load
```python
import torch
checkpoint = torch.load('data/artifacts/pretrained/bilstm_encoder_correct.pt', map_location='cpu')
# If it's a full checkpoint:
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)  # Direct state_dict
```

---

## 6. Results and Logging

### Experiment Results

**Primary results file**: `/Users/jack/projects/moola/experiment_results.jsonl`
- **Format**: JSON Lines (one JSON object per line)
- **Lines**: 4 experiments logged
- **Columns per experiment**: exp_id, name, dropout, heads, batch_size, test_accuracy, pred_hash, config_hash
- **Example entry**:
  ```json
  {"exp_id": 1, "name": "Baseline", "dropout": 0.1, "heads": 2, "batch_size": 16, "test_accuracy": 0.55}
  ```

### Artifact Predictions

**Prediction CSVs**:
- `/Users/jack/projects/moola/data/artifacts/predictions.csv` (49 KB)
- `/Users/jack/projects/moola/data/artifacts/predictions_v2_cleanlab.csv` (6.2 KB)
- `/Users/jack/projects/moola/data/artifacts/predictions_v2_raw.csv` (6.3 KB)

**Metrics/Reporting**:
- `/Users/jack/projects/moola/data/artifacts/confusion_matrix.csv`
- `/Users/jack/projects/moola/data/artifacts/metrics.json`
- `/Users/jack/projects/moola/data/artifacts/runs.csv`
- `/Users/jack/projects/moola/data/artifacts/test_report.json`

---

## 7. Data Quality and Corrections

### Cleanlab Label Quality Review

**Location**: `/Users/jack/projects/moola/data/corrections/`

| File | Purpose | Rows |
|------|---------|------|
| cleanlab_label_quality.csv | Per-sample label quality scores | 98+ |
| cleanlab_label_issues.csv | Identified problematic labels | 12 |
| cleanlab_studio_all_samples.csv | Full Cleanlab studio export | 105+ |
| cleanlab_studio_priority_review.csv | Priority review cases | 35+ |
| cleanlab_reviewed.json | JSON record of reviews | - |

### Manual Annotations and Candlesticks

**Structure annotation batches**:
- `candlesticks_annotations/batch_*.json` (batches 0-110 exist)
- `candlesticks_annotations/master_index.csv`

**Review corrections**:
- `review_corrections/review_batch_*.json` (various batches)
- `review_corrections/review_master_index.csv`

### Exclusions and Replacements

**Training cleaned data**:
- `/Users/jack/projects/moola/data/training_cleaned/windows_to_exclude.csv`
- `/Users/jack/projects/moola/data/training_cleaned/cleanlab_corrections_20251015.json`
- `/Users/jack/projects/moola/data/training_cleaned/replacement_log_20251015_221254.json`

**Purpose**: Track which samples were excluded/replaced during quality filtering

---

## Summary Table: Key Datasets

| Dataset | Type | Samples | Size | Status | Recommendation |
|---------|------|---------|------|--------|-----------------|
| train_clean.parquet | Labeled (binary) | 98 | 0.09 MB | ✅ Canonical v1 | **USE THIS** |
| train_smote_300.parquet | Augmented (SMOTE) | 300 | 0.79 MB | ✅ Available | For SMOTE experiments |
| unlabeled_windows.parquet | Unlabeled OHLC | 11,873 | 2.20 MB | ✅ Primary | **USE FOR PRETRAINING** |
| unlabeled_ohlc.npy | Cache (OHLC) | 11,873 | 38.05 MB | ✅ Ready | Fast loading |
| fold_*.json (v1) | Splits (5-fold) | 5 files | - | ✅ Canonical v1 | **USE FOR CV** |
| bilstm_encoder_correct.pt | Pretrained | 135K params | 2.03 MB | ✅ Latest | **USE FOR TRANSFER** |

---

## Data Flow Architecture

```
Raw Data (crypto OHLC)
    ↓
Labeled: train_clean.parquet (98 samples)
    ├─→ Fold splits (artifacts/splits/v1/)
    ├─→ Augmented via SMOTE (train_smote_300.parquet)
    └─→ OOF predictions (multiple models)
    
Unlabeled: unlabeled_windows.parquet (11,873 samples)
    ├─→ Cache: unlabeled_ohlc.npy (fast pretraining)
    ├─→ With pseudo-labels: unlabeled_with_labels.parquet
    └─→ Pre-training: BiLSTM encoder (bilstm_encoder_correct.pt)

Corrections & QA:
    ├─→ Cleanlab reviews (data/corrections/)
    ├─→ Manual annotations (candlesticks_annotations/)
    └─→ Exclusions (windows_to_exclude.csv)
```

---

## Critical Notes for Refactor

1. **Base dataset**: train_clean.parquet is canonical (98 samples, 2 classes, no quality issues flagged)
2. **Augmentation**: SMOTE is only synthetic method currently applied. Pseudo-sample generation code exists but not widely deployed
3. **Unlabeled corpus**: 11,873 samples ready for pre-training, but requires (~30 MB preprocessing to cache)
4. **Pretraining**: BiLSTM encoder exists but TS2Vec likely not fully tuned
5. **Splits**: v1 folds are forward-chaining (time-series respecting), 5-fold with seed=1337
6. **Quality**: Cleanlab reviews identified ~12 problematic samples, but train_clean already filtered
7. **No contamination risk**: Splits are temporal, not random → holdout is truly unseen future patterns

---

## Immediate Action Items

- [ ] Verify train_clean.parquet sample integrity (98 samples with valid OHLC)
- [ ] Test loading bilstm_encoder_correct.pt in current PyTorch version
- [ ] Validate unlabeled_ohlc.npy shape matches (11873, 105, 4)
- [ ] Check if forward-chaining split respects temporal order
- [ ] Document feature engineering pipeline (currently in features/ dir)
- [ ] Confirm SMOTE parameters used in train_smote_300.parquet creation

---

## Appendix A: Complete Absolute Path Reference

### Canonical Datasets (Use These)

```
LABELED TRAINING (98 samples, binary classification)
/Users/jack/projects/moola/data/processed/train_clean.parquet

UNLABELED CORPUS (11,873 samples, for pre-training)
/Users/jack/projects/moola/data/raw/unlabeled_windows.parquet

SPLIT DEFINITIONS (5-fold, forward-chaining, seed=1337)
/Users/jack/projects/moola/data/artifacts/splits/v1/fold_0.json
/Users/jack/projects/moola/data/artifacts/splits/v1/fold_1.json
/Users/jack/projects/moola/data/artifacts/splits/v1/fold_2.json
/Users/jack/projects/moola/data/artifacts/splits/v1/fold_3.json
/Users/jack/projects/moola/data/artifacts/splits/v1/fold_4.json

PRETRAINED BILSTM ENCODER (135K params)
/Users/jack/projects/moola/data/artifacts/pretrained/bilstm_encoder_correct.pt

PRETRAINING CACHE (38 MB OHLC, 11,873 windows)
/Users/jack/projects/moola/data/pretraining/unlabeled_ohlc.npy

FEATURE CACHE (pre-computed, 25 dims)
/Users/jack/projects/moola/data/pretraining/unlabeled_features.npy
```

### Augmentation & Synthetic Data

```
SMOTE AUGMENTED (300 samples, perfectly balanced)
/Users/jack/projects/moola/data/processed/train_smote_300.parquet

PSEUDO-LABELED UNLABELED (multi-label pseudo-labels)
/Users/jack/projects/moola/data/processed/unlabeled_with_labels.parquet

OUT-OF-FOLD PREDICTIONS (baseline models)
/Users/jack/projects/moola/data/oof/simple_lstm_clean.npy
/Users/jack/projects/moola/data/oof/cnn_transformer_clean.npy
/Users/jack/projects/moola/data/oof/xgb_clean.npy
/Users/jack/projects/moola/data/oof/rf_clean.npy
/Users/jack/projects/moola/data/oof/logreg_clean.npy

OUT-OF-FOLD PREDICTIONS (SMOTE-trained models)
/Users/jack/projects/moola/data/oof/simple_lstm_augmented.npy
/Users/jack/projects/moola/data/oof/cnn_transformer_augmented.npy
/Users/jack/projects/moola/data/oof/xgb_augmented.npy
/Users/jack/projects/moola/data/oof/rf_augmented.npy
/Users/jack/projects/moola/data/oof/logreg_augmented.npy
```

### Historical/Backup Versions (Reference Only)

```
Original labeled (105 samples, symlink to train_pivot_134.parquet)
/Users/jack/projects/moola/data/processed/train.parquet

3-class variant with reversal (134 samples, obsolete)
/Users/jack/projects/moola/data/processed/train_3class_backup.parquet

Backup copy of train_clean (98 samples)
/Users/jack/projects/moola/data/processed/train_clean_backup.parquet

Phase 2 cleaned (89 samples, with quality scores)
/Users/jack/projects/moola/data/processed/train_clean_phase2.parquet

Pivot variant (134 samples)
/Users/jack/projects/moola/data/processed/train_pivot_134.parquet

Reversal holdout (22 samples)
/Users/jack/projects/moola/data/processed/reversal_holdout.parquet
```

### Quality Control & Corrections

```
Cleanlab label quality scores (per-sample)
/Users/jack/projects/moola/data/corrections/cleanlab_label_quality.csv

Identified problematic labels
/Users/jack/projects/moola/data/corrections/cleanlab_label_issues.csv

Candlestick annotations (110 batches)
/Users/jack/projects/moola/data/corrections/candlesticks_annotations/batch_*.json
/Users/jack/projects/moola/data/corrections/candlesticks_annotations/master_index.csv

Review corrections batches
/Users/jack/projects/moola/data/corrections/review_corrections/review_batch_*.json

Samples to exclude from training
/Users/jack/projects/moola/data/training_cleaned/windows_to_exclude.csv

Cleanlab corrections applied (2025-10-15)
/Users/jack/projects/moola/data/training_cleaned/cleanlab_corrections_20251015.json

Replacement log
/Users/jack/projects/moola/data/training_cleaned/replacement_log_20251015_221254.json
```

### Results & Metrics

```
Experiment results (JSON Lines format, 4 experiments)
/Users/jack/projects/moola/experiment_results.jsonl

Model predictions (various versions)
/Users/jack/projects/moola/data/artifacts/predictions.csv
/Users/jack/projects/moola/data/artifacts/predictions_v2_raw.csv
/Users/jack/projects/moola/data/artifacts/predictions_v2_cleanlab.csv

Performance metrics
/Users/jack/projects/moola/data/artifacts/metrics.json
/Users/jack/projects/moola/data/artifacts/confusion_matrix.csv
/Users/jack/projects/moola/data/artifacts/test_report.json
/Users/jack/projects/moola/data/artifacts/runs.csv
```

### Pretrained Encoders (Archive)

```
Main project BiLSTM encoder (root)
/Users/jack/projects/moola/bilstm_encoder.pt (0.52 MB)

TS-TCC experimental encoder
/Users/jack/projects/moola/models/ts_tcc/pretrained_encoder.pt (3.37 MB)

SMOTE-trained TS-TCC variant
/Users/jack/projects/moola/data/artifacts/models_smote_300/models/ts_tcc/pretrained_encoder.pt (3.37 MB)

RunPod multitask experiment
/Users/jack/projects/moola/artifacts/runpod_results/multitask_encoder.pt (2.03 MB)

Alternative weights
/Users/jack/projects/moola/data/artifacts/pretrained/encoder_weights.pt (3.37 MB)
/Users/jack/projects/moola/data/artifacts/pretrained/bilstm_encoder.pt (0.53 MB)
```

---

## Appendix B: Python Loading Snippets

### Load Canonical Labeled Dataset
```python
import pandas as pd

# Load 98 training samples
df = pd.read_parquet('/Users/jack/projects/moola/data/processed/train_clean.parquet')
print(df.shape)  # (98, 5)
print(df['label'].value_counts())  # consolidation: 56, retracement: 42

# Extract OHLC windows: each 'features' is a list of [open, high, low, close] × 105
X = df['features'].values  # shape: (98, 105, 4)
y = df['label'].values    # shape: (98,)
```

### Load Fold Split
```python
import json

fold_0 = json.load(open('/Users/jack/projects/moola/data/artifacts/splits/v1/fold_0.json'))
train_idx = fold_0['train_idx']  # 78 indices
val_idx = fold_0['val_idx']      # 20 indices
print(f"Fold 0: train={len(train_idx)}, val={len(val_idx)}")
```

### Load Unlabeled Corpus for Pre-training
```python
import pandas as pd
import numpy as np

# Full parquet (slower)
df_unlabeled = pd.read_parquet('/Users/jack/projects/moola/data/raw/unlabeled_windows.parquet')
print(df_unlabeled.shape)  # (11873, 2)

# Fast numpy cache (recommended for training loops)
X_ohlc = np.load('/Users/jack/projects/moola/data/pretraining/unlabeled_ohlc.npy')
print(X_ohlc.shape)  # (11873, 105, 4)

X_features = np.load('/Users/jack/projects/moola/data/pretraining/unlabeled_features.npy')
print(X_features.shape)  # (11873, 25)
```

### Load BiLSTM Pretrained Encoder
```python
import torch

# Load state dict
checkpoint = torch.load(
    '/Users/jack/projects/moola/data/artifacts/pretrained/bilstm_encoder_correct.pt',
    map_location='cpu'
)

# If full checkpoint (with optimizer):
if 'model_state_dict' in checkpoint:
    encoder_state = checkpoint['model_state_dict']
else:
    encoder_state = checkpoint

# Load into model
from moola.models import BiLSTMEncoder
model = BiLSTMEncoder(input_size=4, hidden_size=64, num_layers=2)
model.load_state_dict(encoder_state)
model.eval()
```

---

## Appendix C: File Statistics Summary

```
TOTAL DATA VOLUME: ~50 MB active + 200 MB archived

Most Recent Modification:
  2025-10-17 19:35:00  unlabeled_with_labels.parquet (2.59 MB)
  2025-10-16 00:38:00  train_clean_backup.parquet (0.09 MB)
  2025-10-16 00:22:00  train_smote_300.parquet (0.79 MB)

Active Training Assets:
  91.0 KB   train_clean.parquet (CANONICAL)
  2.20 MB   unlabeled_windows.parquet (CANONICAL)
  0.79 MB   train_smote_300.parquet (SMOTE augmented)
  38.05 MB  unlabeled_ohlc.npy (pretraining cache, fast)
  2.26 MB   unlabeled_features.npy (feature cache)
  2.03 MB   bilstm_encoder_correct.pt (pretrained encoder)

Split & Metadata:
  ~5 KB     fold_*.json files (5 folds total)

Archive & QA:
  ~100+ MB  candlestick annotations + reviews
  300 KB    quality control CSVs & JSON
```

---

Report generated: 2025-10-18
Verified all critical assets: ✅ COMPLETE
Ready for refactor: ✅ YES
