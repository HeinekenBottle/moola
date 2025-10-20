# Batch 200 Window Extraction - Complete Index

## Quick Start

```bash
# 1. Extract 200 windows
python3 /Users/jack/projects/moola/scripts/extract_batch_200.py

# 2. Validate outputs
python3 /Users/jack/projects/moola/scripts/test_extract_batch_200.py

# 3. See usage examples
python3 /Users/jack/projects/moola/scripts/example_batch_200_usage.py
```

## File Directory

### Scripts
| File | Purpose | Lines |
|------|---------|-------|
| **extract_batch_200.py** | Main extraction script | ~600 |
| **test_extract_batch_200.py** | Validation test suite | ~350 |
| **example_batch_200_usage.py** | Usage examples | ~330 |

### Documentation
| File | Purpose |
|------|---------|
| **BATCH_200_DELIVERY_SUMMARY.md** | Complete delivery overview |
| **EXTRACT_BATCH_200_README.md** | Comprehensive user guide |
| **BATCH_EXTRACTION_QUICK_REF.md** | Quick reference for common tasks |
| **BATCH_200_INDEX.md** | This file - navigation hub |

## Output Files

### Data Files
| Path | Description |
|------|-------------|
| `data/batches/batch_200.parquet` | 200 windows with OHLC + metadata |
| `data/batches/batch_200_manifest.json` | Extraction statistics |
| `data/corrections/candlesticks_annotations/master_index.csv` | Updated with 200 rows |
| `data/corrections/candlesticks_annotations/rejections.json` | Rejected windows (if any) |

## Documentation by Use Case

### First Time User
1. Read: **BATCH_200_DELIVERY_SUMMARY.md** (Overview)
2. Run: `python3 scripts/extract_batch_200.py`
3. Validate: `python3 scripts/test_extract_batch_200.py`
4. Learn: `python3 scripts/example_batch_200_usage.py`

### Quick Reference
- **BATCH_EXTRACTION_QUICK_REF.md** - Copy/paste snippets

### Deep Dive
- **EXTRACT_BATCH_200_README.md** - Full documentation

### Troubleshooting
1. Check: **EXTRACT_BATCH_200_README.md** → Troubleshooting section
2. Check: **BATCH_EXTRACTION_QUICK_REF.md** → Common Issues
3. Run: `python3 scripts/test_extract_batch_200.py` for diagnostics

### Integration
- See: **example_batch_200_usage.py** for code patterns
- See: **EXTRACT_BATCH_200_README.md** → Example Analysis

## Key Specifications

| Spec | Value |
|------|-------|
| **Total Windows** | 200 |
| **Window Size** | 105 bars |
| **Columns per Bar** | 4 (open, high, low, close) |
| **Features Shape** | (105, 4) |
| **Sessions** | 4 (A, B, C, D) |
| **Volatility Buckets** | 2 (low, high) |
| **Cells** | 8 (4 sessions × 2 buckets) |
| **Samples per Cell** | ~25 |
| **Random Seed** | 17 (reproducible) |
| **Window ID Format** | `batch_YYYYMMDDHHMM_###` |

## Workflow

### Extraction
```
Raw Data (nq_1min_raw.parquet)
    ↓
[Load & Validate]
    ↓
[Compute Global Stats]
    ↓
[Generate All Candidates]
    ↓
[Filter Overlaps with Labeled]
    ↓
[Stratified Sampling]
    ↓
[Assign Window IDs]
    ↓
[Save Outputs]
    ↓
Batch Parquet + Manifest + Master Index
```

### Validation
```
[Test Batch Parquet]
    ↓
[Test Manifest JSON]
    ↓
[Test Master Index]
    ↓
[Test Traceability]
    ↓
All Tests Pass ✓
```

### Annotation (Future)
```
[Load Batch 200]
    ↓
[For Each Window]
    ↓
[Review OHLC Chart]
    ↓
[Assign Label]
    ↓
[Update Master Index]
    ↓
Labeled Dataset Complete
```

## Common Commands

### Load Batch
```python
import pandas as pd
df = pd.read_parquet("data/batches/batch_200.parquet")
```

### Load Manifest
```python
import json
with open("data/batches/batch_200_manifest.json") as f:
    manifest = json.load(f)
```

### Check Distribution
```python
import pandas as pd
df = pd.read_parquet("data/batches/batch_200.parquet")
print(df.groupby(["session", "volatility_bucket"]).size())
```

### Access Window OHLC
```python
import pandas as pd
df = pd.read_parquet("data/batches/batch_200.parquet")
window = df.iloc[0]
ohlc = window["features"]  # shape: (105, 4)
```

### Trace to Raw Data
```python
import pandas as pd
batch_df = pd.read_parquet("data/batches/batch_200.parquet")
raw_df = pd.read_parquet("candlesticks/data/raw/nq_1min_raw.parquet")
window = batch_df.iloc[0]
original = raw_df.iloc[window["raw_start_idx"]:window["raw_end_idx"]+1]
```

## Dependencies

### Python Packages
- pandas
- numpy
- pyarrow (for parquet)

All included in standard Moola environment.

### Input Files
- `candlesticks/data/raw/nq_1min_raw.parquet` (required)
- `data/processed/train_pivot_134.parquet` (optional, for overlap check)

### Output Directories
- `data/batches/` (auto-created)
- `data/corrections/candlesticks_annotations/` (auto-created)

## Quality Guarantees

✅ **Reproducible**: Same seed → same output
✅ **Traceable**: Every window maps to raw data
✅ **Validated**: No NaN, no invalid prices
✅ **Diverse**: Stratified across sessions and volatility
✅ **Unique**: No duplicate window IDs
✅ **Non-overlapping**: No intersection with labeled data
✅ **Documented**: Comprehensive guides and examples
✅ **Tested**: Automated validation suite

## Support

### Getting Help
1. Check documentation in order:
   - This index (navigation)
   - Quick reference (common tasks)
   - README (comprehensive guide)
   - Delivery summary (overview)

2. Run validation:
   ```bash
   python3 scripts/test_extract_batch_200.py
   ```

3. Review examples:
   ```bash
   python3 scripts/example_batch_200_usage.py
   ```

### Reporting Issues
Include:
- Error message (full traceback)
- Command run
- Output files present (ls data/batches/)
- Python version: `python3 --version`

## Version

**Script Version**: 1.0.0
**Documentation Date**: 2025-10-18
**Status**: Production Ready ✓

---

**Total Deliverables**: 8 files (3 scripts, 5 docs)
**Total Lines of Code**: ~1,280
**Total Documentation**: ~1,900 lines
