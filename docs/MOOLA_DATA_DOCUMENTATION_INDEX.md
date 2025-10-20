# Moola Data Documentation Index

Complete guide to understanding window IDs, OHLC data structure, and extracting data for Candlesticks integration.

## Documents

### 1. DATA_MAPPING_SUMMARY.md (START HERE)
**File:** `/Users/jack/projects/moola/docs/DATA_MAPPING_SUMMARY.md`
**Length:** ~196 lines
**For:** Quick reference, executive overview

**Contains:**
- Key findings (window ID format, OHLC structure, expansion indices)
- Three extraction options with code
- Critical validation rules
- FAQ with quick answers
- Integration checklist
- Complete extraction example

**Read if:** You need a quick understanding of how data maps or want to see 3 simple code examples.

---

### 2. WINDOW_ID_MAPPING.md (DETAILED REFERENCE)
**File:** `/Users/jack/projects/moola/docs/WINDOW_ID_MAPPING.md`
**Length:** ~449 lines
**For:** Comprehensive technical reference

**Contains:**
- Detailed window ID mapping explanation
- Base index to raw data mapping
- OHLC data structure in parquet
- Complete 105-bar window structure
- Window creation pipeline details
- Data loading code locations
- Extraction methods comparison (sequential vs by ID)
- Data integrity checks
- File reference table

**Read if:** You need complete details about how windows are created, how data is stored, or need to reference code locations.

---

### 3. CANDLESTICKS_INTEGRATION_GUIDE.md (IMPLEMENTATION)
**File:** `/Users/jack/projects/moola/docs/CANDLESTICKS_INTEGRATION_GUIDE.md`
**Length:** ~441 lines
**For:** Practical implementation and code examples

**Contains:**
- Quick reference diagram
- Three extraction workflows with code:
  1. Extract labeled samples for visualization
  2. Extract by sequential range
  3. Group by base index (multiple experiments)
- Data validation functions
- Complete `MoolaCandlestickExtractor` class (production-ready)
- API endpoint example
- Integration points

**Read if:** You're implementing candlesticks integration and need working Python code.

---

## Quick Start Path

### For Implementation (Fast Track)
1. Read: **DATA_MAPPING_SUMMARY.md** (5 min)
2. Copy code from: **CANDLESTICKS_INTEGRATION_GUIDE.md** (class and examples)
3. Reference: **WINDOW_ID_MAPPING.md** (if validation issues arise)

### For Understanding (Deep Dive)
1. Start: **DATA_MAPPING_SUMMARY.md** (quick overview)
2. Then: **WINDOW_ID_MAPPING.md** (complete details)
3. Finally: **CANDLESTICKS_INTEGRATION_GUIDE.md** (implementation)

### For Specific Questions
- "What's a window ID?" → DATA_MAPPING_SUMMARY.md FAQ
- "How do I extract candlesticks?" → CANDLESTICKS_INTEGRATION_GUIDE.md
- "How is the data created?" → WINDOW_ID_MAPPING.md Section 4
- "How do I validate data?" → Both documents have validation sections

---

## Key Concepts at a Glance

### Window ID Format
```
"102_exp_1"
 ├─ 102 = base index (position in sliding window sequence)
 └─ exp_1 = experiment/labeling iteration
```

### OHLC Structure
- **Storage:** Array of 105 bars, each [open, high, low, close]
- **Shape:** (105,) object array → convert to (105, 4) float array
- **Bars:** 30 past context + 45 prediction + 30 future
- **Pattern:** Located in prediction window [30:75], marked by expansion_start/end

### Data Source
- **File:** `data/processed/train_pivot_134.parquet`
- **Rows:** 105 labeled samples
- **Columns:** window_id, label, expansion_start, expansion_end, features

---

## Code Examples by Use Case

### Use Case 1: Extract One Sample
```python
import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/train_pivot_134.parquet')
sample = df[df['window_id'] == "102_exp_1"].iloc[0]
ohlc = np.array([np.array(bar) for bar in sample['features']])
# ohlc.shape = (105, 4)
```
**From:** DATA_MAPPING_SUMMARY.md (Example: Complete Extraction)

---

### Use Case 2: Extract Multiple Samples by IDs
```python
def extract_labeled_candlesticks(sample_ids):
    df = pd.read_parquet('data/processed/train_pivot_134.parquet')
    results = []
    for wid in sample_ids:
        row = df[df['window_id'] == wid].iloc[0]
        ohlc = np.array([np.array(bar) for bar in row['features']])
        results.append({
            'window_id': wid,
            'ohlc': ohlc,
            'pattern_start': row['expansion_start'],
            'pattern_end': row['expansion_end']
        })
    return results
```
**From:** CANDLESTICKS_INTEGRATION_GUIDE.md (Workflow 1)

---

### Use Case 3: Extract by Base Index Range
```python
def extract_range_candlesticks(start_idx, end_idx):
    df = pd.read_parquet('data/processed/train_pivot_134.parquet')
    df['base_idx'] = df['window_id'].str.split('_').str[0].astype(int)
    filtered = df[(df['base_idx'] >= start_idx) & (df['base_idx'] <= end_idx)]
    # Process filtered...
```
**From:** CANDLESTICKS_INTEGRATION_GUIDE.md (Workflow 2)

---

### Use Case 4: Production-Ready Extraction Class
See: **CANDLESTICKS_INTEGRATION_GUIDE.md** → MoolaCandlestickExtractor class

Methods:
- `extract_by_window_id(window_ids)`
- `extract_range(start, end)`
- `extract_by_label(label, limit)`
- `extract_grouped()` (multiple experiments per window)
- `validate_sample(window_id)`

---

## Data Validation Checklist

From WINDOW_ID_MAPPING.md Section 7:

**OHLC relationships:**
```python
# Must be true:
High >= Low
High >= max(Open, Close)
Low <= min(Open, Close)
```

**Expansion indices:**
```python
# Must be true:
30 <= expansion_start <= expansion_end <= 74
```

**Window ID format:**
```python
# Must match regex:
^\d+_exp_\d+$
```

---

## File Reference

All documents point to these key source files:

| File | Purpose | Lines |
|------|---------|-------|
| `data/processed/train_pivot_134.parquet` | Labeled training data | N/A |
| `src/moola/data/dual_input_pipeline.py` | OHLC extraction (line 231-244) | 231 |
| `src/moola/data_infra/schemas.py` | Data schemas | Multiple |
| `src/moola/config/data_config.py` | Constants | Multiple |
| `scripts/archive/extract_unlabeled_windows.py` | Window creation | 77-138 |
| `scripts/archive/ingest_pivot_134_clean.py` | Labeled data ingestion | 59-133 |

---

## Status

All documentation complete and verified with actual parquet data:

- ✓ Window ID format documented and mapped
- ✓ OHLC data structure confirmed (105 x 4)
- ✓ Expansion indices explained and validated
- ✓ Three extraction approaches documented
- ✓ Production-ready code provided
- ✓ Data validation rules specified
- ✓ Code locations referenced
- ✓ FAQ answered

**Ready for:** Candlesticks integration implementation

---

## How to Use These Documents

### As a Developer:
1. Copy the `MoolaCandlestickExtractor` class from CANDLESTICKS_INTEGRATION_GUIDE.md
2. Use `extract_by_window_id()` or `extract_range()` methods
3. Reference validation methods as needed
4. See "Integration Points" section for API endpoint example

### As a Data Analyst:
1. Start with DATA_MAPPING_SUMMARY.md
2. Use extraction options in "How to Extract" section
3. Follow validation rules
4. Reference code locations for deeper understanding

### As an Architect:
1. Read WINDOW_ID_MAPPING.md Section 4 (creation pipeline)
2. Review all three documents to understand data flow
3. Check "Files to Reference" tables for code locations

---

## Questions?

For specific questions, use the "Quick Answers" section in DATA_MAPPING_SUMMARY.md

Common questions answered:
- Are window IDs sequential integers?
- How do I know which raw bars correspond to a sample?
- Can I extract by sequential ID range?
- What's the "expansion" region?
- How many bars per window?
- Is the data safe to use?

---

## Document Statistics

- **Total lines:** 1,086
- **Code examples:** 20+
- **Code blocks:** Python-ready
- **Data sources:** 1 parquet file
- **Documentation time:** Comprehensive
- **Status:** READ-ONLY (information gathering phase)

