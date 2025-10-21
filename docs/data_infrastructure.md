# Production Data Infrastructure for LSTM Pre-training

## Overview

This document describes the production-grade data infrastructure for the moola ML pipeline, specifically designed for financial time-series data and LSTM pre-training workflows.

### Key Components

1. **Data Validation Schemas** (Pydantic)
2. **Data Versioning** (DVC)
3. **Quality Framework** (Custom validators)
4. **Automated Pipelines** (Validation, monitoring, lineage)
5. **Storage Architecture** (Optimized for time-series)

---

## 1. Data Validation Schemas

### Location
- `src/moola/data_infra/schemas.py`

### Schemas

#### Time-Series Schemas
- **`OHLCBar`**: Single candlestick with OHLC validation
- **`TimeSeriesWindow`**: 105-timestep window (30 past + 45 prediction + 30 future)
- **`LabeledWindow`**: Window with pattern label and expansion indices

#### Dataset Schemas
- **`UnlabeledDataset`**: Collection for SSL pre-training (11,873 samples)
- **`LabeledDataset`**: Labeled training data (98 samples) with class balance validation

#### Quality & Lineage
- **`DataQualityReport`**: Quality metrics and validation results
- **`DataLineage`**: Transformation tracking and checksums
- **`DataVersion`**: DVC-integrated version metadata

### Example Usage

```python
from moola.data_infra.schemas import TimeSeriesWindow, LabeledWindow

# Validate unlabeled window
window = TimeSeriesWindow(
    window_id="window_0",
    features=ohlc_array.tolist()  # (105, 4) array
)

# Validate labeled window
labeled = LabeledWindow(
    window_id="train_0",
    features=ohlc_array.tolist(),
    label="consolidation",
    expansion_start=35,
    expansion_end=60
)
```

---

## 2. Data Versioning with DVC

### Configuration
- `.dvc/config`: Local storage configuration
- `dvc.yaml`: Pipeline stages and dependencies

### Pipeline Stages

```yaml
stages:
  - ingest_raw: Raw OHLC data ingestion
  - validate_raw: Raw data quality checks
  - create_windows: 105-timestep windowing
  - validate_windows: Window validation
  - process_labeled: Label encoding and processing
  - validate_labeled: Labeled data quality gates
  - version_datasets: Version metadata creation
  - monitor_drift: Data drift detection
```

### Version Control

```python
from moola.data_infra.lineage import DataVersionControl

vc = DataVersionControl()
version = vc.create_version(
    dataset_name="train",
    file_path=Path("data/processed/train.parquet"),
    version_id="v1.0.0",
    label_distribution={"consolidation": 60, "retracement": 45},
    notes="Initial labeled dataset after class collapse fix"
)
```

### DVC Commands

```bash
# Track datasets
dvc add data/raw/unlabeled_windows.parquet
dvc add data/processed/train.parquet

# Run pipeline
dvc repro

# Pull data version
dvc pull data/processed/train.parquet
```

---

## 3. Data Quality Framework

### Location
- `src/moola/data_infra/validators/quality_checks.py`

### Quality Checks

#### Completeness
- Missing values: Max 1% overall, 5% per column
- Null handling for OHLC data

#### Statistical Validation
- Outlier detection: Z-score (5σ) and IQR (3x) methods
- Price range validation: [0.001, 1M]
- Price jump detection: Max 200% per bar

#### Financial-Specific
- **OHLC Logic**: high ≥ low, high ≥ max(open, close), etc.
- **Temporal Consistency**: No duplicate timestamps, gap detection
- **Market Hours**: Symbol-specific trading hours validation

#### Class Balance
- Min 2 samples per class (SMOTE requirement)
- Max 10x class imbalance ratio
- Expected distributions for consolidation/retracement/expansion

### Example Usage

```python
from moola.data_infra.validators import TimeSeriesQualityValidator, QualityThresholds

validator = TimeSeriesQualityValidator(
    thresholds=QualityThresholds(
        max_missing_percent=1.0,
        outlier_zscore=5.0,
        max_price_jump_percent=200.0
    )
)

report = validator.validate_dataset(df, dataset_name="train_v1")

if report.passed_validation:
    print(f"✓ Quality score: {report.quality_score}/100")
else:
    print(f"✗ Validation failed: {report.validation_errors}")
```

---

## 4. Automated Validation Pipeline

### Location
- `src/moola/data_infra/pipelines/validate.py`

### Validation Stages

#### Stage 1: Raw Data
```bash
python -m moola.data_infra.pipelines.validate --stage raw
```

Validates:
- OHLC price ranges
- Temporal consistency
- Missing values
- Volume data (if present)

#### Stage 2: Windows
```bash
python -m moola.data_infra.pipelines.validate --stage windows
```

Validates:
- Window shape (105, 4)
- OHLC constraints per timestep
- Feature consistency across samples

#### Stage 3: Labeled Data
```bash
python -m moola.data_infra.pipelines.validate --stage labeled
```

Validates:
- Label values (consolidation/retracement/expansion)
- Expansion indices [30, 75)
- Class balance (min 2 samples, max 10x imbalance)
- Sample count requirements

### Output

Each stage produces:
- `validation_report.json`: Full quality report
- `quality_metrics.json`: Metrics for tracking
- Exit code 0 (pass) or 1 (fail) for CI/CD gates

---

## 5. Data Drift Monitoring

### Location
- `src/moola/data_infra/monitoring/drift_detector.py`

### Detection Methods

1. **Kolmogorov-Smirnov Test** (default)
   - Two-sample statistical test
   - Threshold: p-value < 0.05

2. **Population Stability Index (PSI)**
   - PSI < 0.1: No change
   - 0.1 ≤ PSI < 0.2: Moderate change
   - PSI ≥ 0.2: Significant drift

3. **Wasserstein Distance**
   - Earth Mover's Distance
   - Normalized by baseline range

### Usage

```bash
python -m moola.data_infra.monitoring.drift_detector \
    --baseline data/processed/train.parquet \
    --current data/processed/new_data.parquet \
    --method ks_test \
    --threshold 0.05
```

### Monitoring Output

```json
{
  "summary": {
    "total_features": 5,
    "drifted_features": 1,
    "drift_percentage": 20.0,
    "overall_drift_detected": false
  },
  "features": {
    "open": {"drift_score": 0.023, "p_value": 0.45, "drift_detected": false},
    "high": {"drift_score": 0.089, "p_value": 0.02, "drift_detected": true},
    ...
  }
}
```

---

## 6. Data Lineage Tracking

### Location
- `src/moola/data_infra/lineage/tracker.py`

### Lineage Tracker

Tracks:
- Parent datasets
- Transformation types
- Input/output checksums
- Row counts
- Execution metadata

```python
from moola.data_infra.lineage import LineageTracker

tracker = LineageTracker()
tracker.log_transformation(
    dataset_id="unlabeled_windows_v1",
    transformation_type="windowing",
    input_path=Path("data/raw/ohlc.parquet"),
    output_path=Path("data/raw/unlabeled_windows.parquet"),
    rows_in=50000,
    rows_out=11873,
    params={"window_length": 105, "stride": 1}
)

# Query lineage
ancestors = tracker.get_ancestors("unlabeled_windows_v1")
descendants = tracker.get_descendants("unlabeled_windows_v1")

# Visualize
tracker.visualize_lineage("unlabeled_windows_v1", output_path=Path("lineage.png"))
```

### Lineage Graph

The tracker maintains a DAG (Directed Acyclic Graph) of transformations:

```
raw_ohlc.parquet
    ↓ [windowing: stride=1]
unlabeled_windows.parquet (11,873 samples)
    ↓ [pre-training: BiLSTM]
pretrained_encoder.pt
    ↓ [fine-tuning: labeled data]
train.parquet (98 samples)
    ↓ [augmentation: SMOTE]
train_smote_300.parquet (300 samples)
```

---

## 7. Storage Architecture

### Directory Structure

```
data/
├── raw/                          # Raw and unlabeled data
│   ├── unlabeled_windows.parquet # 11,873 samples, 2.2 MB
│   ├── market_data_*.parquet     # Raw OHLC from exchanges
│   └── validation_report.json    # Quality reports
│
├── processed/                    # Labeled and processed data
│   ├── train.parquet             # 98 labeled samples → symlink to train_pivot_134.parquet
│   ├── train_pivot_134.parquet   # Current best version
│   ├── train_clean.parquet       # Cleaned version
│   ├── train_smote_300.parquet   # Augmented with SMOTE
│   └── validation_report.json
│
├── versions/                     # Version metadata
│   ├── v1.0.0/
│   │   ├── metadata.json
│   │   └── checksums.json
│   └── v1.1.0/
│       ├── metadata.json
│       └── checksums.json
│
├── lineage/                      # Lineage tracking
│   ├── unlabeled_windows_v1.json
│   ├── train_v1.json
│   └── lineage_report.json
│
├── monitoring/                   # Drift and quality monitoring
│   ├── drift_report.json
│   ├── drift_distribution.json
│   └── quality_metrics.json
│
├── artifacts/                    # Model artifacts
│   └── pretrained/
│       └── encoder_weights.pt
│
└── splits/                       # Train/val/test splits
    └── fold_*.parquet
```

### File Formats

| Data Type | Format | Rationale |
|-----------|--------|-----------|
| Time-series windows | Parquet | Columnar, compressed, schema enforcement |
| Model weights | PyTorch (.pt) | Native format, includes metadata |
| Metrics/Reports | JSON | Human-readable, version-controllable |
| Large arrays | NumPy (.npy) | Fast loading, memory-mapped |
| OOF predictions | NumPy (.npy) | Consistent with sklearn |

### Compression Strategy

```python
# Parquet with optimal compression
df.to_parquet(
    path,
    engine='pyarrow',
    compression='snappy',  # Fast compression for frequent access
    # Or 'gzip' for cold storage with better compression ratio
)
```

### Retention Policies

| Data Stage | Retention | Storage Tier |
|------------|-----------|--------------|
| Raw market data | 2 years | Cold (S3 Glacier) |
| Processed windows | 1 year | Warm (S3 Standard-IA) |
| Labeled datasets | Indefinite | Hot (S3 Standard) + DVC |
| Model artifacts | All versions | Hot (S3 Standard) + DVC |
| Monitoring data | 90 days | Warm |
| Logs | 30 days | Warm |

### Storage Costs (Estimated for AWS S3)

- **Unlabeled windows** (2.2 MB): ~$0.000051/month (Standard)
- **Labeled train** (94 KB): ~$0.0000022/month (Standard)
- **Model weights** (various): ~$0.023/GB/month (Standard)
- **Total monthly cost** (< 1 GB): ~$0.05/month

### Backup Strategy

```bash
# DVC remote for versioned data
dvc remote add backup s3://moola-ml-backups/data

# Automated daily backups
aws s3 sync data/ s3://moola-ml-backups/data-$(date +%Y%m%d) \
    --exclude "*.pyc" \
    --exclude "*.log" \
    --exclude "__pycache__/*"
```

---

## 8. Data Catalog

### Metadata Registry

The data catalog tracks all datasets with:

```json
{
  "unlabeled_windows": {
    "path": "data/raw/unlabeled_windows.parquet",
    "version": "v1.0.0",
    "format": "parquet",
    "schema": {
      "columns": ["window_id", "features"],
      "feature_shape": [105, 4],
      "feature_names": ["open", "high", "low", "close"]
    },
    "statistics": {
      "num_samples": 11873,
      "size_bytes": 2300000,
      "price_range": [19000, 21000],
      "timestamp_range": ["2024-01-01", "2024-10-14"]
    },
    "quality": {
      "score": 98.5,
      "validation_passed": true,
      "last_validated": "2024-10-16T22:00:00Z"
    },
    "lineage": {
      "source": "binance_btcusdt_1h",
      "transformations": ["windowing"],
      "parent_datasets": ["raw_ohlc_2024"]
    },
    "usage": {
      "purpose": "SSL pre-training",
      "used_by": ["ssl_pretrain.py", "ts_tcc.py"]
    }
  },
  "train_pivot_134": {
    "path": "data/processed/train_pivot_134.parquet",
    "version": "v1.0.0",
    "schema": {
      "columns": ["window_id", "label", "features", "expansion_start", "expansion_end"],
      "labels": ["consolidation", "retracement"],
      "feature_shape": [105, 4]
    },
    "statistics": {
      "num_samples": 98,
      "label_distribution": {
        "consolidation": 60,
        "retracement": 38
      },
      "class_imbalance_ratio": 1.58
    },
    "quality": {
      "score": 95.0,
      "validation_passed": true,
      "issues_fixed": ["class_collapse_resolved"]
    }
  }
}
```

---

## 9. Quality Gates and SLAs

### Data Quality SLAs

| Metric | Threshold | Action if Failed |
|--------|-----------|------------------|
| Missing values | < 1% | Block pipeline, investigate source |
| OHLC logic errors | 0 | Block pipeline, fix data |
| Outlier rate | < 5% | Warning, log for review |
| Class imbalance | < 10x | Warning, consider augmentation |
| Min samples/class | ≥ 2 | Block training, acquire more data |

### Data Freshness SLAs

| Dataset | Max Age | Update Frequency |
|---------|---------|-----------------|
| Raw market data | 1 hour | Continuous (streaming) |
| Unlabeled windows | 24 hours | Daily batch |
| Labeled data | N/A | Manual annotation |
| Drift metrics | 1 week | Weekly |

### Pipeline SLAs

| Stage | Max Duration | Timeout |
|-------|-------------|---------|
| Data ingestion | 5 min | 10 min |
| Window creation | 30 min | 1 hour |
| Validation | 10 min | 30 min |
| Pre-training | 4 hours | 8 hours |
| Fine-tuning | 30 min | 2 hours |

---

## 10. Integration with ML Pipeline

### Pre-training Workflow

```python
from moola.data_infra.schemas import UnlabeledDataset
from moola.data_infra.validators import TimeSeriesQualityValidator
from moola.data_infra.lineage import LineageTracker

# 1. Load and validate unlabeled data
df = pd.read_parquet("data/raw/unlabeled_windows.parquet")

validator = TimeSeriesQualityValidator()
report = validator.validate_dataset(df, "unlabeled_windows")

if not report.passed_validation:
    raise ValueError(f"Data quality check failed: {report.validation_errors}")

# 2. Convert to Pydantic schema
windows = [
    TimeSeriesWindow(window_id=row['window_id'], features=row['features'].tolist())
    for _, row in df.iterrows()
]
dataset = UnlabeledDataset(windows=windows, total_samples=len(windows))

# 3. Track lineage
tracker = LineageTracker()
tracker.log_transformation(
    dataset_id="pretrain_input",
    transformation_type="validation",
    input_path=Path("data/raw/unlabeled_windows.parquet"),
    output_path=None,
    rows_in=len(df),
    rows_out=len(df),
)

# 4. Run pre-training
X = dataset.to_numpy()  # Shape: (11873, 105, 4)
# ... SSL pre-training code ...
```

### Fine-tuning Workflow

```python
from moola.data_infra.schemas import LabeledDataset
from moola.data_infra.lineage import DataVersionControl

# 1. Version control
vc = DataVersionControl()
version = vc.create_version(
    dataset_name="train",
    file_path=Path("data/processed/train_pivot_134.parquet"),
    version_id="v1.0.0",
    label_distribution={"consolidation": 60, "retracement": 38},
    notes="Class collapse fixed with pre-training"
)

# 2. Validate labeled data
df = pd.read_parquet(version.file_path)
validator = TimeSeriesQualityValidator()
report = validator.validate_dataset(df, "train_v1")

# 3. Load with schema validation
windows = [
    LabeledWindow(
        window_id=row['window_id'],
        features=row['features'].tolist(),
        label=row['label'],
        expansion_start=row['expansion_start'],
        expansion_end=row['expansion_end']
    )
    for _, row in df.iterrows()
]

dataset = LabeledDataset(
    windows=windows,
    total_samples=len(windows),
    label_distribution=df['label'].value_counts().to_dict()
)

# 4. Extract for training
X, y = dataset.to_numpy()
# ... Fine-tuning code ...
```

---

## 11. Monitoring and Alerting

### Automated Monitoring

```bash
# Daily drift check (cron job)
0 0 * * * python -m moola.data_infra.monitoring.drift_detector \
    --baseline data/processed/train.parquet \
    --current data/processed/latest.parquet \
    --output data/monitoring/drift_$(date +%Y%m%d).json

# Send alert if drift detected
if [ $(jq '.summary.overall_drift_detected' data/monitoring/drift_latest.json) == "true" ]; then
    slack-notify "Data drift detected - review required"
fi
```

### Quality Metrics Dashboard

Track over time:
- Quality scores per dataset
- Drift metrics per feature
- Data volume trends
- Validation failure rates
- Lineage complexity

Tools: Prometheus + Grafana or custom dashboard

---

## 12. Future Enhancements

### Short-term (1-3 months)
- [ ] Implement Great Expectations integration for advanced validation
- [ ] Add Feast feature store for online/offline feature serving
- [ ] Set up automated data profiling with pandas-profiling
- [ ] Create manual batch processing scripts

### Medium-term (3-6 months)
- [ ] Implement data catalog with Amundsen or DataHub
- [ ] Add automated data labeling pipeline
- [ ] Set up real-time streaming ingestion with Kafka
- [ ] Implement A/B testing framework for data pipelines

### Long-term (6-12 months)
- [ ] Migrate to data lakehouse (Delta Lake or Iceberg)
- [ ] Implement federated learning for privacy
- [ ] Add synthetic data generation for augmentation
- [ ] Build data marketplace for feature sharing

---

## 13. Troubleshooting

### Common Issues

#### Issue: Validation Failing on OHLC Logic
```bash
# Check for corrupted bars
python -m moola.data_infra.validators.quality_checks \
    --check-ohlc-logic \
    --data data/raw/unlabeled_windows.parquet
```

#### Issue: Data Drift Detected
```bash
# Investigate drift sources
python -m moola.data_infra.monitoring.drift_detector \
    --baseline data/versions/v1.0.0/train.parquet \
    --current data/processed/train.parquet \
    --method psi
```

#### Issue: DVC Cache Growing Too Large
```bash
# Clean old cache entries
dvc gc --workspace --cloud

# Check cache size
du -sh .dvc/cache
```

#### Issue: Lineage Graph Too Complex
```bash
# Prune old lineage records
python -c "
from moola.data_infra.lineage import LineageTracker
tracker = LineageTracker()
# Keep only last 30 days
tracker.prune_old_lineage(days=30)
"
```

---

## References

- [DVC Documentation](https://dvc.org/doc)
- [Pydantic V2 Documentation](https://docs.pydantic.dev/latest/)
- [Great Expectations](https://greatexpectations.io/)
- [Data Drift Detection Methods](https://arxiv.org/abs/2004.03045)
- [MLOps Best Practices](https://ml-ops.org/)
