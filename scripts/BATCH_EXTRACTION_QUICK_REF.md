# Batch Extraction - Quick Reference

## One-Line Execution
```bash
python3 /Users/jack/projects/moola/scripts/extract_batch_200.py
```

## Output Files
| File | Purpose |
|------|---------|
| `data/batches/batch_200.parquet` | 200 windows with OHLC + metadata |
| `data/batches/batch_200_manifest.json` | Extraction stats and distribution |
| `data/corrections/candlesticks_annotations/master_index.csv` | Updated with 200 new rows |
| `data/corrections/candlesticks_annotations/rejections.json` | Rejected windows (if any) |

## Load and Inspect
```python
import pandas as pd
import json

# Load batch
df = pd.read_parquet("data/batches/batch_200.parquet")
print(f"Total windows: {len(df)}")
print(df.columns.tolist())

# Load manifest
with open("data/batches/batch_200_manifest.json") as f:
    manifest = json.load(f)
print(f"Session distribution: {manifest['session_counts']}")

# Access a window
window = df.iloc[0]
print(f"ID: {window['window_id']}")
print(f"OHLC shape: {window['features'].shape}")  # (105, 4)
print(f"Session: {window['session']}, Vol: {window['volatility_bucket']}")
print(f"RV: {window['rv']:.6f}, Range: {window['range_norm']:.4f}")
```

## Window ID Format
```
batch_YYYYMMDDHHMM_<seq>
├─ batch_            : Prefix
├─ YYYYMMDDHHMM      : Extraction timestamp
└─ <seq>             : Zero-padded 3-digit (001-200)

Example: batch_202410181430_001
```

## Sessions (UTC)
| Session | Hours | Description |
|---------|-------|-------------|
| A | 22:00-01:00 | Evening |
| B | 01:00-07:00 | Overnight |
| C | 13:00-17:00 | Midday |
| D | 17:00-22:00 | Afternoon/Evening |

## Diversity Metrics
| Metric | Formula | Purpose |
|--------|---------|---------|
| `rv` | std(log returns) | Realized volatility |
| `range_norm` | (high_max - low_min) / median_yearly_range | Normalized range |
| `trend_mag` | abs(close_end - close_start) / median_yearly_range | Trend strength |

## Expected Distribution
- 8 cells total: 4 sessions × 2 volatility buckets
- ~25 windows per cell
- Spread across range_norm and trend_mag quantiles

## Quick Validation
```python
df = pd.read_parquet("data/batches/batch_200.parquet")

# Check total
assert len(df) == 200, f"Expected 200, got {len(df)}"

# Check features shape
assert df.iloc[0]["features"].shape == (105, 4), "Wrong features shape"

# Check no duplicates
assert df["window_id"].nunique() == 200, "Duplicate IDs found"

# Session distribution
print(df.groupby(["session", "volatility_bucket"]).size())
```

## Traceability
```python
# Trace a window back to raw data
window = df[df["window_id"] == "batch_202410181430_001"].iloc[0]
raw_df = pd.read_parquet("candlesticks/data/raw/nq_1min_raw.parquet")

# Extract original bars
original_bars = raw_df.iloc[window["raw_start_idx"]:window["raw_end_idx"]+1]
print(f"Original time range: {original_bars['timestamp'].min()} to {original_bars['timestamp'].max()}")
```

## Common Issues

### Script fails with "No valid candidates"
- Raw data too small or all windows overlap with labeled data
- Check: `len(pd.read_parquet("candlesticks/data/raw/nq_1min_raw.parquet"))` >= 105

### Session distribution imbalanced
- Normal - some sessions have fewer valid windows
- Script samples as many as available per cell

### "Overlaps labeled window" rejections
- Expected - script avoids reusing labeled data
- Check `rejections.json` for details

## Next Steps After Extraction

1. **Review manifest**: Check `batch_200_manifest.json` for stats
2. **Inspect distribution**: Verify session/volatility balance
3. **Manual annotation**: Add labels to windows
4. **Update master index**: Fill `quality_grade` and `expansion_count`
5. **Track progress**: Use `window_id` for annotation tracking

## Reproducibility
- Fixed seed: 17
- Deterministic output
- Same input → same output (guaranteed)
