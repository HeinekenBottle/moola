# Data Infrastructure Module

Production-grade data infrastructure for financial time-series ML pipelines.

## 📦 Module Structure

```
data_infra/
├── __init__.py                  # Public API exports
├── schemas.py                   # Pydantic data models (500+ lines)
├── validators/
│   ├── __init__.py
│   └── quality_checks.py       # Quality validation (400+ lines)
├── pipelines/
│   ├── __init__.py
│   └── validate.py             # Automated validation (350+ lines)
├── monitoring/
│   ├── __init__.py
│   └── drift_detector.py       # Drift detection (400+ lines)
└── lineage/
    ├── __init__.py
    └── tracker.py              # Lineage & versioning (500+ lines)
```

## 🚀 Quick Start

### Installation

```python
# Already included in moola package
from moola.data_infra import (
    TimeSeriesWindow,
    LabeledWindow,
    TimeSeriesQualityValidator,
    LineageTracker,
    DataVersionControl,
    TimeSeriesDriftMonitor
)
```

### Basic Usage

#### 1. Validate Data

```python
from moola.data_infra import TimeSeriesQualityValidator
import pandas as pd

df = pd.read_parquet("data/raw/unlabeled_windows.parquet")

validator = TimeSeriesQualityValidator()
report = validator.validate_dataset(df, "my_dataset")

if report.passed_validation:
    print(f"✓ Quality score: {report.quality_score}/100")
else:
    print(f"✗ Errors: {report.validation_errors}")
```

#### 2. Version Dataset

```python
from moola.data_infra import DataVersionControl
from pathlib import Path

vc = DataVersionControl()
version = vc.create_version(
    dataset_name="train",
    file_path=Path("data/processed/train.parquet"),
    version_id="v1.0.0",
    notes="Initial production version"
)
```

#### 3. Track Lineage

```python
from moola.data_infra import LineageTracker

tracker = LineageTracker()
tracker.log_transformation(
    dataset_id="windows_v1",
    transformation_type="windowing",
    input_path=Path("data/raw/ohlc.parquet"),
    output_path=Path("data/raw/unlabeled_windows.parquet"),
    rows_in=50000,
    rows_out=11873,
    params={"window_length": 105}
)
```

#### 4. Monitor Drift

```python
from moola.data_infra import TimeSeriesDriftMonitor

monitor = TimeSeriesDriftMonitor(method="ks_test")
results = monitor.monitor_dataset_drift(baseline_df, current_df)

for feature, result in results.items():
    if result.drift_detected:
        print(f"⚠️  {feature}: DRIFT DETECTED")
```

## 📊 Schemas

### Time-Series Schemas

#### `OHLCBar`
Single candlestick with validation:
- ✅ high ≥ low
- ✅ high ≥ max(open, close)
- ✅ low ≤ min(open, close)
- ✅ Detects >200% price jumps

```python
from moola.data_infra import OHLCBar

bar = OHLCBar(open=100, high=105, low=95, close=102)
arr = bar.to_array()  # [100, 105, 95, 102]
```

#### `TimeSeriesWindow`
105-timestep OHLC window:
- 30 past bars (context)
- 45 prediction window
- 30 future bars (outcome)

```python
from moola.data_infra import TimeSeriesWindow

window = TimeSeriesWindow(
    window_id="window_0",
    features=ohlc_list  # List[List[float]], shape (105, 4)
)

arr = window.to_numpy()  # Shape: (105, 4)
```

#### `LabeledWindow`
Pattern-labeled window:

```python
from moola.data_infra import LabeledWindow, PatternLabel

window = LabeledWindow(
    window_id="train_0",
    features=ohlc_list,
    label=PatternLabel.CONSOLIDATION,
    expansion_start=35,  # In [30, 75)
    expansion_end=60     # In [30, 75)
)
```

### Dataset Schemas

#### `UnlabeledDataset`
For SSL pre-training:

```python
from moola.data_infra import UnlabeledDataset

dataset = UnlabeledDataset(
    windows=window_list,
    total_samples=len(window_list)
)

X = dataset.to_numpy()  # Shape: (N, 105, 4)
```

#### `LabeledDataset`
For supervised training:

```python
from moola.data_infra import LabeledDataset

dataset = LabeledDataset(
    windows=labeled_window_list,
    total_samples=len(labeled_window_list),
    label_distribution={"consolidation": 60, "retracement": 45}
)

X, y = dataset.to_numpy()  # X: (N, 105, 4), y: (N,)
```

Validates:
- ✅ Min 2 samples per class
- ✅ Max 10x class imbalance
- ✅ Label distribution consistency

## 🔍 Validators

### `TimeSeriesQualityValidator`

Comprehensive quality checks:

```python
from moola.data_infra import TimeSeriesQualityValidator, QualityThresholds

validator = TimeSeriesQualityValidator(
    thresholds=QualityThresholds(
        max_missing_percent=1.0,
        outlier_zscore=5.0,
        max_price_jump_percent=200.0,
        check_ohlc_logic=True
    )
)

report = validator.validate_dataset(df, "dataset_name")
```

**Checks:**
1. Completeness: Missing values
2. Statistical: Outliers (Z-score, IQR)
3. OHLC Logic: high ≥ low constraints
4. Price Jumps: >200% movements
5. Temporal: Duplicate timestamps, gaps

**Output:**
- Quality score: 0-100
- Validation errors
- Warnings
- Pass/fail status

### `FinancialDataValidator`

Financial-specific checks:
- Price range validation
- Volume validation
- Market hours checking

## 📈 Monitoring

### `DriftDetector`

Three methods available:

#### 1. Kolmogorov-Smirnov Test (default)
```python
from moola.data_infra import DriftDetector

detector = DriftDetector(method="ks_test", threshold=0.05)
result = detector.detect_drift(baseline, current, "feature_name")

if result.drift_detected:
    print(f"Drift score: {result.drift_score}, p-value: {result.p_value}")
```

#### 2. Population Stability Index (PSI)
```python
detector = DriftDetector(method="psi", threshold=0.2)
result = detector.detect_drift(baseline, current, "feature_name")

# PSI interpretation:
# < 0.1: No change
# 0.1-0.2: Moderate change
# >= 0.2: Significant drift
```

#### 3. Wasserstein Distance
```python
detector = DriftDetector(method="wasserstein", threshold=0.1)
result = detector.detect_drift(baseline, current, "feature_name")
```

### `TimeSeriesDriftMonitor`

Monitor OHLC features:

```python
from moola.data_infra import TimeSeriesDriftMonitor

monitor = TimeSeriesDriftMonitor(method="ks_test", threshold=0.05)
results = monitor.monitor_dataset_drift(baseline_df, current_df)

# Generates report
monitor.generate_drift_report(results, Path("drift_report.json"))
```

## 🔗 Lineage & Versioning

### `LineageTracker`

Track data transformations:

```python
from moola.data_infra import LineageTracker

tracker = LineageTracker()

# Log transformation
tracker.log_transformation(
    dataset_id="processed_v1",
    transformation_type="augmentation",
    input_path=Path("data/raw/train.parquet"),
    output_path=Path("data/processed/train_smote.parquet"),
    rows_in=98,
    rows_out=300,
    params={"method": "smote", "k_neighbors": 5},
    parent_datasets=["train_v1"]
)

# Query lineage
ancestors = tracker.get_ancestors("processed_v1")
descendants = tracker.get_descendants("train_v1")

# Visualize (requires graphviz)
tracker.visualize_lineage("processed_v1", output_path=Path("lineage.png"))

# Export report
tracker.export_lineage_report(Path("lineage_report.json"))
```

### `DataVersionControl`

Semantic versioning for datasets:

```python
from moola.data_infra import DataVersionControl

vc = DataVersionControl()

# Create version
version = vc.create_version(
    dataset_name="train",
    file_path=Path("data/processed/train.parquet"),
    version_id="v1.0.0",
    label_distribution={"consolidation": 60, "retracement": 45},
    parent_version=None,
    transformation=None,
    tags=["stable", "pretrained"],
    notes="Production-ready dataset with BiLSTM pre-training"
)

# Retrieve version
v = vc.get_version("train", "v1.0.0")
print(f"Samples: {v.num_samples}, Quality: {v.quality_score}/100")

# Get latest
latest = vc.get_latest_version("train")

# List all versions
versions = vc.list_versions("train")
```

## 🔧 CLI Tools

### Validation Pipeline

```bash
# Validate raw data
python -m moola.data_infra.pipelines.validate --stage raw

# Validate windows
python -m moola.data_infra.pipelines.validate --stage windows

# Validate labeled data
python -m moola.data_infra.pipelines.validate --stage labeled

# Custom path
python -m moola.data_infra.pipelines.validate \
    --stage labeled \
    --data-path data/processed/train.parquet \
    --output-dir data/reports
```

**Outputs:**
- `validation_report.json`: Full quality report
- `quality_metrics.json`: Metrics for tracking
- Exit code: 0 (pass) or 1 (fail)

### Drift Detection

```bash
python -m moola.data_infra.monitoring.drift_detector \
    --baseline data/processed/train_v1.parquet \
    --current data/processed/train_v2.parquet \
    --method ks_test \
    --threshold 0.05 \
    --output data/monitoring/drift_report.json
```

## 🧪 Testing

Run integration tests:

```bash
pytest tests/test_data_infra.py -v
```

Test coverage:
- ✅ Schema validation
- ✅ Quality checks
- ✅ Lineage tracking
- ✅ Version control
- ✅ Drift detection

## 📚 Documentation

- **Quick Start**: `docs/data_infra_quickstart.md`
- **Full Guide**: `docs/data_infrastructure.md`
- **Examples**: `examples/data_infra_example.py`
- **Summary**: `DATA_INFRASTRUCTURE_SUMMARY.md`

## 🎯 Use Cases

### Pre-training Pipeline

```python
from moola.data_infra import (
    UnlabeledDataset,
    TimeSeriesWindow,
    TimeSeriesQualityValidator,
    LineageTracker
)

# 1. Load and validate
df = pd.read_parquet("data/raw/unlabeled_windows.parquet")
validator = TimeSeriesQualityValidator()
report = validator.validate_dataset(df, "unlabeled_windows")

if not report.passed_validation:
    raise ValueError(f"Validation failed: {report.validation_errors}")

# 2. Convert to schema
windows = [
    TimeSeriesWindow(
        window_id=row['window_id'],
        features=row['features'].tolist()
    )
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
    rows_out=len(df)
)

# 4. Extract for training
X = dataset.to_numpy()  # (11873, 105, 4)
# Run SSL pre-training...
```

### Fine-tuning Pipeline

```python
from moola.data_infra import (
    LabeledDataset,
    LabeledWindow,
    DataVersionControl
)

# 1. Version control
vc = DataVersionControl()
version = vc.create_version(
    dataset_name="train",
    file_path=Path("data/processed/train.parquet"),
    version_id="v1.0.0",
    label_distribution={"consolidation": 60, "retracement": 45}
)

# 2. Load and validate
df = pd.read_parquet(version.file_path)

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

# 3. Extract for training
X, y = dataset.to_numpy()  # (98, 105, 4), (98,)
# Run fine-tuning...
```

### Production Monitoring

```python
from moola.data_infra import TimeSeriesDriftMonitor

def daily_drift_check():
    """Run daily drift monitoring."""
    monitor = TimeSeriesDriftMonitor(method="ks_test", threshold=0.05)

    baseline = pd.read_parquet("data/versions/v1.0.0/train.parquet")
    current = pd.read_parquet("data/processed/latest.parquet")

    results = monitor.monitor_dataset_drift(baseline, current)

    # Generate report
    monitor.generate_drift_report(
        results,
        Path(f"data/monitoring/drift_{datetime.now().date()}.json")
    )

    # Alert if drift
    drift_count = sum(1 for r in results.values() if r.drift_detected)
    if drift_count > len(results) * 0.5:
        send_alert("⚠️  Significant drift detected - consider retraining")

# Schedule daily
```

## ⚙️ Configuration

### Quality Thresholds

```python
from moola.data_infra import QualityThresholds

thresholds = QualityThresholds(
    # Completeness
    max_missing_percent=1.0,
    max_missing_per_column=5.0,

    # Outliers
    outlier_zscore=5.0,
    outlier_iqr_multiplier=3.0,

    # Price validation
    min_price=0.001,
    max_price=1_000_000.0,
    max_price_jump_percent=200.0,

    # Temporal
    allow_duplicate_timestamps=False,
    max_time_gap_minutes=None,

    # OHLC
    check_ohlc_logic=True
)
```

### DVC Integration

```yaml
# dvc.yaml
stages:
  validate_windows:
    cmd: python -m moola.data_infra.pipelines.validate --stage windows
    deps:
      - data/raw/unlabeled_windows.parquet
      - src/moola/data_infra/validators/quality_checks.py
    outs:
      - data/raw/window_validation.json:
          cache: false
```

## 🚨 Common Issues

### Issue: Validation Failing

```python
# Debug validation errors
report = validator.validate_dataset(df, "debug")

for error in report.validation_errors:
    print(error)

# Check specific windows
for idx, row in df.iterrows():
    try:
        window = TimeSeriesWindow(
            window_id=str(row['window_id']),
            features=row['features'].tolist()
        )
    except Exception as e:
        print(f"Window {idx} failed: {e}")
```

### Issue: Drift Detected

```python
# Investigate drift sources
for feature, result in results.items():
    if result.drift_detected:
        print(f"{feature}:")
        print(f"  Baseline: mean={result.baseline_stats['mean']:.2f}")
        print(f"  Current:  mean={result.current_stats['mean']:.2f}")
        print(f"  Drift score: {result.drift_score:.4f}")
```

## 📦 API Reference

### Public API

```python
from moola.data_infra import (
    # Schemas
    OHLCBar,
    TimeSeriesWindow,
    LabeledWindow,
    UnlabeledDataset,
    LabeledDataset,
    DataQualityReport,
    DataLineage,
    DataVersion,
    PatternLabel,
    DataStage,
    DataFormat,

    # Validators
    TimeSeriesQualityValidator,
    FinancialDataValidator,
    QualityThresholds,

    # Monitoring
    DriftDetector,
    DriftResult,
    TimeSeriesDriftMonitor,

    # Lineage
    LineageTracker,
    DataVersionControl,
)
```

## 🔗 Integration Points

- **DVC**: Version control integration
- **MLflow**: Metrics tracking (future)
- **Airflow**: Pipeline orchestration (future)
- **Great Expectations**: Advanced validation (future)
- **Feast**: Feature store (future)

## 📊 Metrics

- **Quality Score**: 0-100 based on multiple factors
- **Drift Score**: Method-specific (p-value, PSI, distance)
- **Lineage Depth**: Number of transformations
- **Version Count**: Number of dataset versions

## 🎓 Best Practices

1. ✅ Always validate before training
2. ✅ Version every dataset
3. ✅ Track all transformations
4. ✅ Monitor production data
5. ✅ Document quality issues
6. ✅ Automate quality gates
7. ✅ Use schemas for type safety
8. ✅ Set up CI/CD validation

## 🔮 Roadmap

- [ ] Great Expectations integration
- [ ] Feast feature store
- [ ] Airflow DAG templates
- [ ] Data catalog (Amundsen/DataHub)
- [ ] Real-time streaming validation
- [ ] A/B testing framework
- [ ] Synthetic data generation
- [ ] Federated learning support

---

**Version**: 1.0.0
**Status**: Production-Ready
**License**: MIT
