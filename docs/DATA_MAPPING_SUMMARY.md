# Moola Data Window ID Mapping - Executive Summary

## Key Findings

### 1. Window ID Format
- **Pattern:** `{base_index}_{exp_num}` (e.g., "102_exp_1")
- **base_index:** Position in the sliding window extraction sequence (0-238 in labeled data)
- **exp_num:** Labeling iteration number (1, 2, etc.)
- **Important:** Multiple window IDs with same base_index have the SAME raw OHLC data but DIFFERENT pattern interpretations

### 2. OHLC Data Structure
- **Stored as:** Array of 105 numpy arrays, each containing [open, high, low, close]
- **Shape when loaded:** (105,) with dtype=object
- **Conversion:** `np.array([np.array(bar) for bar in features])` → (105, 4) array
- **105-bar breakdown:**
  - Bars 0-29: Past context (30 bars)
  - Bars 30-74: Prediction window (45 bars) ← Patterns occur here
  - Bars 75-104: Future outcome (30 bars)

### 3. Expansion Indices
- **Range:** [30, 74] (always within prediction window)
- **Meaning:** Where the labeled pattern (consolidation/retracement) actually occurs
- **Example:** expansion_start=40, expansion_end=42 → Pattern spans bars 40-42 (3 bars)
- **Average pattern length:** 5-23 bars (mean 6.6 bars)

### 4. Data Source
- **File:** `/Users/jack/projects/moola/data/processed/train_pivot_134.parquet`
- **Rows:** 105 labeled samples
- **Unique base indices:** 66 (range 0-238)
- **Label distribution:** 60 consolidation, 45 retracement (135 total samples from 105 rows)

---

## How to Extract for Candlesticks

### Option 1: Direct from Parquet (Simplest)
```python
import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/train_pivot_134.parquet')
row = df[df['window_id'] == "102_exp_1"].iloc[0]

# Get full OHLC
ohlc = np.array([np.array(bar) for bar in row['features']])  # [105, 4]

# Get pattern region
pattern = ohlc[row['expansion_start']:row['expansion_end']+1]
```

### Option 2: By Range
```python
df = pd.read_parquet('data/processed/train_pivot_134.parquet')
df['base_idx'] = df['window_id'].str.split('_').str[0].astype(int)
df_range = df[(df['base_idx'] >= 0) & (df['base_idx'] <= 100)]
```

### Option 3: By Label
```python
df = pd.read_parquet('data/processed/train_pivot_134.parquet')
consolidations = df[df['label'] == 'consolidation']
```

---

## Critical Validation Rules

1. **OHLC relationships must hold:**
   - High ≥ Low
   - High ≥ max(Open, Close)
   - Low ≤ min(Open, Close)

2. **Expansion indices must be valid:**
   - 30 ≤ expansion_start ≤ expansion_end ≤ 74
   - expansion_start < expansion_end (or equal for 1-bar patterns)

3. **Window ID format:**
   - Must match regex: `^\d+_exp_\d+$`
   - Base index ≤ 238
   - Exp number ≥ 1

---

## Integration Checklist

- [ ] Load parquet file
- [ ] Extract base_index from window_id string (split by "_", take first part)
- [ ] Convert features array to [105, 4] numpy array
- [ ] Validate OHLC relationships
- [ ] Validate expansion indices in [30, 74]
- [ ] Extract pattern region: `ohlc[expansion_start:expansion_end+1]`
- [ ] Highlight pattern bars in candlestick chart
- [ ] Display full 105-bar window with context indicators

---

## Reference Files

All documentation is in `/Users/jack/projects/moola/docs/`:

1. **WINDOW_ID_MAPPING.md** - Complete technical reference
   - Window creation pipeline
   - Data loading code locations
   - OHLC structure details
   - Extraction methods comparison

2. **CANDLESTICKS_INTEGRATION_GUIDE.md** - Practical integration guide
   - Complete Python code examples
   - Three extraction workflows
   - MoolaCandlestickExtractor class
   - Data validation functions
   - API endpoint examples

3. **DATA_MAPPING_SUMMARY.md** - This file
   - Executive summary
   - Quick reference
   - Validation rules

---

## Code Locations to Reference

| File | Purpose |
|------|---------|
| `/Users/jack/projects/moola/src/moola/data/dual_input_pipeline.py` | OHLC extraction logic (lines 231-244) |
| `/Users/jack/projects/moola/src/moola/data_infra/schemas.py` | Data validation schemas |
| `/Users/jack/projects/moola/src/moola/config/data_config.py` | Constants (window size, expansion ranges) |
| `/Users/jack/projects/moola/scripts/archive/extract_unlabeled_windows.py` | Window creation algorithm |
| `/Users/jack/projects/moola/scripts/archive/ingest_pivot_134_clean.py` | Labeled data ingestion |
| `/Users/jack/projects/moola/data/processed/train_pivot_134.parquet` | Actual labeled data |

---

## Quick Answers

**Q: Are window IDs sequential integers?**
A: No. They're strings like "102_exp_1", not integers. The base index (102) is sequential, but multiple samples can share the same base index with different exp numbers.

**Q: How do I know which raw bars correspond to a labeled sample?**
A: The base_index directly maps to the sliding window extraction sequence. If you need raw OHLC data, you already have it - it's in the parquet file as the `features` column. No need to go back to raw data.

**Q: Can I extract by sequential ID range?**
A: Yes, extract base_index from window_id and filter. See "Option 2: By Range" above.

**Q: What's the "expansion" region?**
A: The bars where the actual pattern (consolidation/retracement) occurs. Defined by expansion_start and expansion_end indices. This should be highlighted in candlestick charts.

**Q: How many bars per window?**
A: Always 105 bars. Structure: 30 past + 45 prediction + 30 future.

**Q: Is the data safe to use?**
A: Yes, if you validate:
  1. OHLC relationships hold
  2. Expansion indices in [30, 74]
  3. Window IDs match the format

---

## Example: Complete Extraction

```python
import pandas as pd
import numpy as np

# Load
df = pd.read_parquet('data/processed/train_pivot_134.parquet')

# Get sample
sample = df[df['window_id'] == "102_exp_1"].iloc[0]

# Extract OHLC
ohlc = np.array([np.array(bar) for bar in sample['features']])
pattern = ohlc[sample['expansion_start']:sample['expansion_end']+1]

# Result
print(f"Window: {sample['window_id']}")
print(f"Label: {sample['label']}")
print(f"Full bars: {ohlc.shape}")  # (105, 4)
print(f"Pattern bars: {pattern.shape}")  # (N, 4) where N = expansion_end - expansion_start + 1
print(f"Pattern location: bars {sample['expansion_start']}-{sample['expansion_end']}")
```

---

## Status: READY FOR CANDLESTICKS INTEGRATION

All data extraction paths verified and documented:
- ✓ Window ID format and mapping understood
- ✓ OHLC data structure confirmed
- ✓ Expansion indices validated
- ✓ Safe extraction methods documented
- ✓ Code examples provided
- ✓ Data integrity checks specified

See `CANDLESTICKS_INTEGRATION_GUIDE.md` for complete implementation code.

