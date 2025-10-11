# Baseline Training SOP

## Purpose
Establish a reproducible baseline for the ML pipeline using LogisticRegression on synthetic data.

## Prerequisites
- Python environment with all dependencies installed (`pip install -e .`)
- Git repository initialized
- Clean data directories (data/processed/, data/artifacts/, data/logs/)

## Baseline Specifications

### Data Contract
- **File**: `data/processed/train.parquet`
- **Schema**: TrainingDataRow (window_id: int, label: str, features: list[float])
- **Samples**: 1000 synthetic examples
- **Classes**: 2 (class_A, class_B) - balanced 50/50 split
- **Features**: 10 random features per sample (mean=0, std=1)

### Model Specifications
- **Algorithm**: LogisticRegression (scikit-learn)
- **Hyperparameters**:
  - max_iter=1000
  - random_state=1337 (from config)
- **Train/Test Split**: 80/20 stratified
- **Evaluation**: Stratified 5-fold cross-validation

### Reproducibility Requirements
- **Seed**: 1337 (set in configs/default.yaml)
- **Deterministic**: All random operations seeded (numpy, sklearn)
- **Version Control**: Track git SHA in runs.csv

## Procedure

### Step 1: Ingest Data
```bash
moola ingest
```

**Expected Output**:
- Creates `data/processed/train.parquet` (1000 rows × 3 columns)
- Validates schema for all rows
- Logs sample count and schema

**Acceptance Criteria**:
- Parquet file exists and is readable
- All rows pass TrainingDataRow validation
- No missing values

### Step 2: Train Baseline Model
```bash
moola train
```

**Expected Output**:
- Saves `data/artifacts/model.bin` (pickle)
- Logs train/test accuracy
- Train accuracy should be ~50-60% (random features baseline)

**Acceptance Criteria**:
- model.bin file exists
- Model is unpicklable and has `.predict()` method
- Train accuracy ≥ 50%
- Test accuracy ≥ 50%

### Step 3: Evaluate with K-Fold CV
```bash
moola evaluate
```

**Expected Output**:
- Creates `data/artifacts/metrics.json`
- Creates `data/artifacts/confusion_matrix.csv`
- Appends to `data/artifacts/runs.csv`
- Logs per-fold and mean metrics

**Acceptance Criteria**:
- metrics.json contains: accuracy, f1, precision, recall, timestamp
- Mean CV accuracy ≥ 50%
- Confusion matrix is 2×2 (for 2 classes)
- runs.csv appended with current run (run_id, git_sha, accuracy, f1, duration)

### Step 4: Verify Artifacts
```bash
ls -lh data/artifacts/
cat data/artifacts/metrics.json
cat data/artifacts/runs.csv
```

**Expected Artifacts**:
- model.bin (~5KB for LogisticRegression)
- metrics.json (JSON with mean + per-fold metrics)
- confusion_matrix.csv (2×2 matrix)
- runs.csv (header + ≥1 data row)

## Success Criteria
- ✅ All three commands (ingest, train, evaluate) complete without errors
- ✅ All expected artifacts are created
- ✅ Mean CV accuracy ≥ 50% (baseline for random features)
- ✅ Tests pass: `pytest tests/test_pipeline.py -v`
- ✅ Run is tracked in runs.csv with git SHA

## Troubleshooting

### Issue: Train accuracy < 50%
- **Cause**: Random features may occasionally produce lower accuracy
- **Fix**: Re-run with different seed or increase sample size

### Issue: Schema validation fails
- **Cause**: Data corruption or incompatible types
- **Fix**: Delete processed/train.parquet and re-run ingest

### Issue: Model.bin not created
- **Cause**: Training failed or insufficient permissions
- **Fix**: Check logs in data/logs/moola.log for error details

## Deviations Log
Document any deviations from this SOP below:

| Date | Run ID | Deviation | Reason | Impact |
|------|--------|-----------|--------|--------|
| - | - | - | - | - |

## Next Steps After Baseline
1. Replace synthetic data with real dataset
2. Implement feature engineering
3. Test XGBoost as alternative to LogisticRegression
4. Tune hyperparameters with grid search
5. Add GPU support for larger models (Dockerfile.gpu)
