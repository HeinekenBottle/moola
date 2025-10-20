# Data Infrastructure Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Prerequisites

```bash
# Install required packages
pip install pydantic pandas pyarrow scipy loguru

# Optional: For DVC integration
pip install dvc dvc-s3

# Optional: For visualization
pip install graphviz
```

### Project Structure

The data infrastructure is organized under `src/moola/data_infra/`:

```
src/moola/data_infra/
├── __init__.py
├── schemas.py              # Pydantic data models
├── validators/
│   ├── __init__.py
│   └── quality_checks.py   # Quality validation
├── pipelines/
│   ├── __init__.py
│   └── validate.py         # Automated validation pipeline
├── monitoring/
│   ├── __init__.py
│   └── drift_detector.py   # Data drift detection
└── lineage/
    ├── __init__.py
    └── tracker.py          # Lineage and versioning
```

---

## 📊 Quick Examples

### 1. Validate Your Data (30 seconds)

```python
from moola.data_infra.validators import TimeSeriesQualityValidator
import pandas as pd

# Load your data
df = pd.read_parquet("data/raw/unlabeled_windows.parquet")

# Validate
validator = TimeSeriesQualityValidator()
report = validator.validate_dataset(df, "my_dataset")

# Check results
print(f"Quality Score: {report.quality_score}/100")
print(f"Passed: {report.passed_validation}")
```

**Output:**
```
Quality Score: 98.5/100
Passed: True
```

### 2. Version Your Dataset (1 minute)

```python
from moola.data_infra.lineage import DataVersionControl
from pathlib import Path

# Initialize version control
vc = DataVersionControl()

# Create version
version = vc.create_version(
    dataset_name="train",
    file_path=Path("data/processed/train.parquet"),
    version_id="v1.0.0",
    notes="Initial production version"
)

print(f"Version {version.version_id} created!")
print(f"Samples: {version.num_samples:,}")
```

**Output:**
```
Version v1.0.0 created!
Samples: 98
```

### 3. Track Data Lineage (2 minutes)

```python
from moola.data_infra.lineage import LineageTracker
from pathlib import Path

tracker = LineageTracker()

# Log transformation
tracker.log_transformation(
    dataset_id="windows_v1",
    transformation_type="windowing",
    input_path=Path("data/raw/ohlc.parquet"),
    output_path=Path("data/raw/unlabeled_windows.parquet"),
    rows_in=50000,
    rows_out=11873,
    params={"window_length": 105, "stride": 1}
)

# Query lineage
lineage = tracker.get_lineage("windows_v1")
print(f"Transformation: {lineage.transformation_type}")
print(f"Rows: {lineage.rows_in} → {lineage.rows_out}")
```

### 4. Detect Data Drift (3 minutes)

```python
from moola.data_infra.monitoring import TimeSeriesDriftMonitor
import pandas as pd

# Load datasets
baseline = pd.read_parquet("data/processed/train_v1.parquet")
current = pd.read_parquet("data/processed/train_v2.parquet")

# Monitor drift
monitor = TimeSeriesDriftMonitor(method="ks_test", threshold=0.05)
results = monitor.monitor_dataset_drift(baseline, current)

# Check results
for feature, result in results.items():
    if result.drift_detected:
        print(f"⚠️  {feature}: DRIFT DETECTED (score={result.drift_score:.4f})")
```

---

## 🔧 Common Use Cases

### Use Case 1: Validate Before Training

```python
from moola.data_infra.validators import TimeSeriesQualityValidator
import pandas as pd

def validate_before_training(data_path):
    """Run quality checks before expensive model training."""
    df = pd.read_parquet(data_path)

    validator = TimeSeriesQualityValidator()
    report = validator.validate_dataset(df, data_path.stem)

    if not report.passed_validation:
        raise ValueError(
            f"Data quality check failed:\n" +
            "\n".join(report.validation_errors)
        )

    if report.quality_score < 80:
        print(f"⚠️  Warning: Low quality score ({report.quality_score}/100)")

    return df

# Use it
df = validate_before_training(Path("data/processed/train.parquet"))
# Proceed with training...
```

### Use Case 2: Automated Pipeline with Quality Gates

```bash
# Run validation pipeline (CLI)
python -m moola.data_infra.pipelines.validate --stage labeled

# In CI/CD (exits with code 1 if validation fails)
```

**Example output:**
```
[2024-10-17 01:00:00] VALIDATION SUMMARY
Quality Score: 95.0/100
Validation Status: PASSED
Errors: 0
Warnings: 1
```

### Use Case 3: Monitor Production Data

```python
from moola.data_infra.monitoring import TimeSeriesDriftMonitor
from pathlib import Path

def check_production_drift():
    """Daily drift check for production data."""
    monitor = TimeSeriesDriftMonitor(method="ks_test", threshold=0.05)

    baseline = pd.read_parquet("data/versions/v1.0.0/train.parquet")
    current = pd.read_parquet("data/processed/latest.parquet")

    results = monitor.monitor_dataset_drift(baseline, current)

    # Generate report
    report_path = Path(f"data/monitoring/drift_{datetime.now().date()}.json")
    monitor.generate_drift_report(results, report_path)

    # Alert if drift detected
    drift_count = sum(1 for r in results.values() if r.drift_detected)
    if drift_count > len(results) * 0.5:
        send_alert("Significant data drift detected - consider retraining")

# Run daily (cron job)
check_production_drift()
```

### Use Case 4: Complete ETL with Lineage

```python
from moola.data_infra.lineage import LineageTracker
from moola.data_infra.validators import TimeSeriesQualityValidator
import pandas as pd

def etl_pipeline_with_lineage():
    """ETL pipeline with automated lineage tracking."""
    tracker = LineageTracker()
    validator = TimeSeriesQualityValidator()

    # Step 1: Load raw data
    raw_df = pd.read_parquet("data/raw/market_data.parquet")

    # Step 2: Create windows
    windows_df = create_windows(raw_df, window_length=105)
    windows_df.to_parquet("data/raw/unlabeled_windows.parquet")

    # Track transformation
    tracker.log_transformation(
        dataset_id="unlabeled_windows_v1",
        transformation_type="windowing",
        input_path=Path("data/raw/market_data.parquet"),
        output_path=Path("data/raw/unlabeled_windows.parquet"),
        rows_in=len(raw_df),
        rows_out=len(windows_df),
        params={"window_length": 105}
    )

    # Step 3: Validate
    report = validator.validate_dataset(windows_df, "unlabeled_windows")

    if not report.passed_validation:
        raise ValueError("Validation failed")

    return windows_df

# Run pipeline
df = etl_pipeline_with_lineage()
```

---

## 📋 CLI Reference

### Validate Data

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

### Detect Drift

```bash
# Basic drift detection
python -m moola.data_infra.monitoring.drift_detector \
    --baseline data/processed/train.parquet \
    --current data/processed/new_data.parquet

# With custom method and threshold
python -m moola.data_infra.monitoring.drift_detector \
    --baseline data/versions/v1.0.0/train.parquet \
    --current data/processed/latest.parquet \
    --method psi \
    --threshold 0.1 \
    --output data/monitoring/drift_report.json
```

### DVC Integration

```bash
# Initialize DVC (if not already done)
dvc init

# Track datasets
dvc add data/raw/unlabeled_windows.parquet
dvc add data/processed/train.parquet

# Commit DVC files
git add data/raw/.gitignore data/raw/unlabeled_windows.parquet.dvc
git commit -m "Track unlabeled windows dataset"

# Run DVC pipeline
dvc repro

# Push data to remote
dvc push
```

---

## 🎯 Best Practices

### 1. Always Validate Before Training

```python
# ✅ Good
df = pd.read_parquet(path)
validator = TimeSeriesQualityValidator()
report = validator.validate_dataset(df)
if report.passed_validation:
    train_model(df)

# ❌ Bad
df = pd.read_parquet(path)
train_model(df)  # No validation!
```

### 2. Version Every Dataset

```python
# ✅ Good
vc = DataVersionControl()
version = vc.create_version(
    dataset_name="train",
    file_path=path,
    version_id="v1.0.0",
    notes="What changed and why"
)

# ❌ Bad
# Overwriting files without versioning
df.to_parquet("data/train.parquet")  # Lost previous version!
```

### 3. Track All Transformations

```python
# ✅ Good
tracker = LineageTracker()
tracker.log_transformation(
    dataset_id="augmented_v1",
    transformation_type="smote",
    input_path=input_path,
    output_path=output_path,
    rows_in=len(input_df),
    rows_out=len(output_df),
    params={"k_neighbors": 5, "sampling_strategy": "auto"}
)

# ❌ Bad
# Transforming without tracking
output_df = apply_smote(input_df)
output_df.to_parquet(output_path)  # Lost lineage!
```

### 4. Monitor Production Data

```python
# ✅ Good - Regular drift monitoring
from apscheduler.schedulers.blocking import BlockingScheduler

scheduler = BlockingScheduler()

@scheduler.scheduled_job('cron', hour=0)  # Daily at midnight
def daily_drift_check():
    monitor = TimeSeriesDriftMonitor()
    # ... drift detection logic ...

scheduler.start()

# ❌ Bad - No monitoring
# Hope the data hasn't changed...
```

---

## 🐛 Troubleshooting

### Issue: "Validation failed - OHLC logic errors"

**Problem:** High values less than low values in your OHLC data.

**Solution:**
```python
# Check for corrupted bars
from moola.data_infra.validators import TimeSeriesQualityValidator

validator = TimeSeriesQualityValidator()
report = validator.validate_dataset(df, "debug")

# Print specific errors
for error in report.validation_errors:
    print(error)
```

### Issue: "Data drift detected"

**Problem:** Production data distribution has changed significantly.

**Solution:**
```python
# Investigate which features drifted
monitor = TimeSeriesDriftMonitor(method="psi", threshold=0.1)
results = monitor.monitor_dataset_drift(baseline, current)

for feature, result in results.items():
    if result.drift_detected:
        print(f"{feature}: baseline_mean={result.baseline_stats['mean']:.2f}, "
              f"current_mean={result.current_stats['mean']:.2f}")

# Action: Consider retraining or investigating data source
```

### Issue: "Pydantic validation error"

**Problem:** Data doesn't match schema.

**Solution:**
```python
from moola.data_infra.schemas import TimeSeriesWindow

# Debug individual windows
for idx, row in df.iterrows():
    try:
        window = TimeSeriesWindow(
            window_id=str(row['window_id']),
            features=row['features'].tolist()
        )
    except Exception as e:
        print(f"Window {idx} failed: {e}")
        print(f"Features shape: {np.array(row['features']).shape}")
        break
```

---

## 📚 Next Steps

1. **Read full documentation**: `docs/data_infrastructure.md`
2. **Run examples**: `python examples/data_infra_example.py`
3. **Set up DVC**: Follow DVC setup in main docs
4. **Integrate with your pipeline**: Start with validation, then add versioning and lineage
5. **Set up monitoring**: Configure daily drift checks

---

## 🔗 Key Files

| File | Purpose |
|------|---------|
| `src/moola/data_infra/schemas.py` | Pydantic schemas for validation |
| `src/moola/data_infra/validators/quality_checks.py` | Quality validation logic |
| `src/moola/data_infra/pipelines/validate.py` | Automated validation pipeline |
| `src/moola/data_infra/monitoring/drift_detector.py` | Drift detection |
| `src/moola/data_infra/lineage/tracker.py` | Lineage and version control |
| `dvc.yaml` | DVC pipeline configuration |
| `examples/data_infra_example.py` | Complete examples |

---

## 💡 Pro Tips

1. **Use schemas early**: Validate data as soon as it enters your pipeline
2. **Automate quality gates**: Fail fast if data quality is poor
3. **Version everything**: Datasets, models, and configurations
4. **Monitor continuously**: Set up daily drift checks
5. **Track lineage**: Know where your data came from
6. **Document assumptions**: Use version notes and metadata

---

## ✅ Checklist: Production Readiness

- [ ] All datasets validated with quality checks
- [ ] Datasets versioned with DVC
- [ ] Transformations tracked in lineage
- [ ] Drift monitoring configured
- [ ] Quality gates in CI/CD pipeline
- [ ] Documentation updated
- [ ] Team trained on data infrastructure
- [ ] Alerting configured for drift/quality issues

---

**Questions?** Check the full documentation or the example code for more details.
