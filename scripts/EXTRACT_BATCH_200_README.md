# Extract Batch 200 - Production Window Extraction Script

## Overview

`extract_batch_200.py` is a production-grade script that extracts 200 diverse trading windows from raw OHLC data for manual annotation. The script implements stratified sampling across trading sessions and volatility regimes while ensuring no overlap with existing labeled data.

## Key Features

### Traceability
- Each window is traceable to original raw data via `raw_start_idx` and `raw_end_idx`
- Unique window IDs: `batch_YYYYMMDDHHMM_<seq>` (e.g., `batch_202410181430_001`)
- Timestamps preserved: `start_ts` and `end_ts` for each window

### Diversity Metrics
Three metrics ensure diverse coverage of market conditions:

1. **Realized Volatility (rv)**: Standard deviation of 1-bar log returns
2. **Normalized Range (range_norm)**: (high_max - low_min) / median_yearly_range
3. **Trend Magnitude (trend_mag)**: |close_end - close_start| / median_yearly_range

### Stratified Sampling
- 4 trading sessions (evening, overnight, midday, afternoon)
- 2 volatility buckets (low/high split by median RV)
- ~25 windows per cell (4 × 2 = 8 cells)
- Within each cell: spread across range_norm and trend_mag quantiles

### Quality Guards
1. No NaN or infinite values
2. All prices positive
3. High/low constraints: `high >= max(open, close)`, `low <= min(open, close)`
4. No overlap with existing labeled windows

## Usage

```bash
# Basic execution
python3 scripts/extract_batch_200.py

# Run from project root
cd /Users/jack/projects/moola
python3 scripts/extract_batch_200.py
```

## Outputs

### 1. Batch Parquet File
**Path**: `data/batches/batch_200.parquet`

**Schema**:
- `window_id`: str - Unique identifier (batch_YYYYMMDDHHMM_###)
- `features`: array(105, 4) - OHLC data [open, high, low, close]
- `raw_start_idx`: int - Start index in raw data
- `raw_end_idx`: int - End index in raw data (inclusive)
- `start_ts`: timestamp - First bar timestamp
- `end_ts`: timestamp - Last bar timestamp
- `session`: str - Trading session (A/B/C/D)
- `volatility_bucket`: str - "low" or "high"
- `rv`: float - Realized volatility
- `range_norm`: float - Normalized range
- `trend_mag`: float - Normalized trend magnitude

### 2. Manifest JSON
**Path**: `data/batches/batch_200_manifest.json`

```json
{
  "script_version": "1.0.0",
  "extraction_datetime": "2025-10-18T14:30:00Z",
  "seed": 17,
  "total_windows": 200,
  "source_file": "candlesticks/data/raw/nq_1min_raw.parquet",
  "global_stats": {
    "median_yearly_range": 1.23,
    "rv_median": 0.45
  },
  "session_counts": {
    "A_low": 25,
    "A_high": 25,
    ...
  },
  "rejected_count": 5,
  "rejection_reasons": {
    "overlaps_labeled_window": 5
  }
}
```

### 3. Master Index Update
**Path**: `data/corrections/candlesticks_annotations/master_index.csv`

Appends 200 rows with columns:
- `window_id`: Unique window identifier
- `batch_file`: "batch_200.parquet"
- `annotation_date`: Timestamp when window was extracted
- `quality_grade`: "" (empty, to be filled during annotation)
- `expansion_count`: 0 (to be updated during annotation)

### 4. Rejections Log
**Path**: `data/corrections/candlesticks_annotations/rejections.json`

Records any windows rejected due to overlap or quality issues:
```json
{
  "rejections": [
    {
      "id": "rejected_202410181430_001",
      "raw_start_idx": 1234,
      "reason": "overlaps_labeled_window",
      "timestamp": "2024-09-01T22:15:00+00:00"
    }
  ]
}
```

## Configuration

### Constants (in script)
```python
T = 105                 # Window size in bars
N_TOTAL = 200          # Total windows to extract
SEED = 17              # For reproducibility
N_PER_CELL = 25        # ~25 per session×volatility cell
```

### Session Definitions (UTC)
```python
SESSIONS = {
    "A": (22, 1),      # 22:00-01:00 (evening)
    "B": (1, 7),       # 01:00-07:00 (overnight)
    "C": (13, 17),     # 13:00-17:00 (midday)
    "D": (17, 22),     # 17:00-22:00 (afternoon/evening)
}
```

## Validation

Uncomment the validation section at the bottom of the script:

```python
if __name__ == "__main__":
    main()
    validate_outputs()  # Uncomment this line
```

This will check:
- Correct number of windows extracted
- No duplicate window IDs
- Features shape (105, 4)
- Session distribution
- No NaN values in features

## Example Analysis

```python
import pandas as pd
import numpy as np

# Load batch
df = pd.read_parquet("data/batches/batch_200.parquet")

# Distribution by session and volatility
print(df.groupby(["session", "volatility_bucket"]).size())

# Diversity metrics summary
print(df[["rv", "range_norm", "trend_mag"]].describe())

# Example: access a single window's OHLC data
window_0 = df.iloc[0]
ohlc = window_0["features"]  # shape: (105, 4)
print(f"Window ID: {window_0['window_id']}")
print(f"Session: {window_0['session']}, Vol: {window_0['volatility_bucket']}")
print(f"Time range: {window_0['start_ts']} to {window_0['end_ts']}")
print(f"OHLC shape: {ohlc.shape}")
print(f"First 3 bars:\n{ohlc[:3]}")
```

## Reproducibility

The script is fully reproducible:
- Fixed random seed (SEED=17)
- Deterministic sorting of candidates
- Identical outputs on repeated runs (same input data)

## Error Handling

The script fails fast with clear error messages:
- Missing input files
- Invalid data format
- Insufficient candidates after filtering
- File I/O errors

All errors are logged with timestamps.

## Performance

Expected runtime on typical hardware:
- Loading raw data: ~1-2 seconds
- Generating candidates: ~30-60 seconds (118K bars)
- Stratified sampling: <1 second
- Saving outputs: ~1 second

Total: ~35-65 seconds

## Troubleshooting

### "No valid candidate windows generated"
- Check raw data has sufficient bars (need at least T=105)
- Verify OHLC data quality (no NaN, positive prices)

### "Only X candidates available after filtering"
- Many windows overlap with labeled data
- Consider adjusting session definitions or expanding raw data range

### "Session distribution imbalanced"
- Some sessions may have fewer valid windows
- Script will sample as many as available per cell

## Next Steps

After extraction:
1. Review `data/batches/batch_200_manifest.json` for distribution stats
2. Load `data/batches/batch_200.parquet` for manual annotation
3. Use `window_id` to track annotations
4. Update `quality_grade` and `expansion_count` in `master_index.csv`
5. Use `raw_start_idx` and `raw_end_idx` for traceability back to source data

## Version History

- **1.0.0** (2025-10-18): Initial production release
  - 200-window extraction
  - Stratified sampling (session × volatility)
  - Diversity metrics (rv, range_norm, trend_mag)
  - Overlap checking with labeled data
  - Complete traceability to raw data
