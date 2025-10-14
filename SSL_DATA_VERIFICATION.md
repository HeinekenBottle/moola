# SSL Data Verification Checklist

**Date**: October 14, 2025
**Task**: Task 17 - Verify data requirements for SSL implementation

---

## Current Status: ⚠️ BLOCKER - Unlabeled Data Not Found

### Local Environment
- ✅ Labeled data exists: `data/processed/train.parquet` (115 samples)
- ❌ Unlabeled data: **NOT FOUND**
- ❌ Pivot-experiments repo: **NOT FOUND** at `../pivot-experiments/`
- ❌ Raw OHLC data: `data/raw/` is empty (only .gitkeep)

### Data Requirements for SSL

| Phase | Data Needed | Expected Location | Status |
|-------|-------------|-------------------|--------|
| **Phase 1** | 118k unlabeled windows | `data/raw/unlabeled_windows.parquet` | ❌ Missing |
| **Phase 2** | 115 labeled samples | `data/processed/train.parquet` | ✅ Exists |
| **Phase 3** | 118k unlabeled (reuse) | Same as Phase 1 | ❌ Missing |
| **Phase 4** | 115 + 185 pseudo | Phase 2 + Phase 3 output | ⏳ Pending |

---

## Verification Steps

### Step 1: Check RunPod for Existing Unlabeled Data

```bash
# SSH into RunPod
ssh -p 14147 root@213.173.108.148

# Check for unlabeled windows
ls -lh /workspace/data/raw/
ls -lh /workspace/data/processed/

# Search for any large parquet files (118k samples ≈ 50-200 MB)
find /workspace/data -name "*.parquet" -size +10M

# Check pivot-experiments if accessible
ls -lh /workspace/../pivot-experiments/data/ 2>/dev/null
```

**Expected Files**:
- `unlabeled_windows.parquet` (~118,000 rows, 105×4 OHLC per row)
- OR `all_windows.parquet` (if includes both labeled and unlabeled)
- OR individual window parquets in a directory

### Step 2: Verify Unlabeled Data Structure

If found, verify with:
```python
import pandas as pd
import numpy as np

# Load unlabeled data
df = pd.read_parquet('/workspace/data/raw/unlabeled_windows.parquet')

print(f"Total samples: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Sample shape: {df.iloc[0].shape if 'features' in df.columns else 'N/A'}")

# Expected structure
# Option 1: Pre-processed like labeled data
#   Columns: ['window_id', 'features']  # No 'label' column
#   features: [105, 4] OHLC array per row

# Option 2: Flattened
#   Columns: 420 columns (105 bars × 4 OHLC)

# Option 3: Raw OHLC list
#   Columns: ['ohlc']
#   ohlc: List of 105 [O, H, L, C] arrays
```

---

## Data Acquisition Options

### Option A: Unlabeled Data Already Exists on RunPod
**Best case**: Just need to locate it

```bash
# On RunPod, search comprehensively
find /workspace -name "*window*" -o -name "*unlabel*" 2>/dev/null
du -sh /workspace/data/raw/* /workspace/data/processed/*
```

If found → Proceed directly to SSL pipeline implementation ✅

### Option B: Extract from Pivot-Experiments Repo

If pivot-experiments is accessible on RunPod:

```bash
# Check if pivot-experiments exists
ls -lh /workspace/../pivot-experiments/

# Look for ALL windows (not just labeled 134)
find /workspace/../pivot-experiments/data -name "*.parquet" | wc -l
# Should see >> 134 files if unlabeled windows exist
```

**Likely locations**:
- `pivot-experiments/data/processed/windows105/all/` (all windows including unlabeled)
- `pivot-experiments/data/processed/windows105/unlabeled/` (explicitly unlabeled)
- `pivot-experiments/data/interim/sliding_windows/` (raw extracted windows)

### Option C: Generate from Raw OHLC Data

**If** you have access to raw futures price history:

Use the provided extraction script (see `scripts/extract_unlabeled_windows.py` below):

```bash
# Extract 118k unlabeled windows from raw OHLC
python scripts/extract_unlabeled_windows.py \
  --input /path/to/raw_ohlc.csv \
  --output data/raw/unlabeled_windows.parquet \
  --window-size 105 \
  --stride 1 \
  --exclude data/processed/train.parquet
```

This will:
1. Load raw OHLC time series
2. Apply sliding window (105 bars, stride 1)
3. Exclude the 134 labeled samples (based on timestamps/indices)
4. Save ~118k unlabeled windows

### Option D: Skip SSL, Use Mixup Instead (Fallback)

If unlabeled data is **not available**, implement the simpler Mixup augmentation:

```python
# In cnn_transformer.py training loop
# Already has mixup_cutmix() implemented!
# Just train normally without pre-training

# Expected gain: +2-4% accuracy (vs +5-8% with SSL)
```

This is already implemented in the existing `CnnTransformerModel`, so it's a zero-effort fallback.

---

## Extraction Script Created

Created `scripts/extract_unlabeled_windows.py` to generate unlabeled data from raw OHLC if needed.

**Usage**:
```bash
python scripts/extract_unlabeled_windows.py \
  --input /workspace/data/raw/ES_1min_2020-2024.csv \
  --output /workspace/data/raw/unlabeled_windows.parquet \
  --window-size 105 \
  --stride 10 \
  --max-samples 120000 \
  --exclude /workspace/data/processed/train.parquet
```

**Parameters**:
- `--input`: Raw OHLC CSV (columns: timestamp, open, high, low, close)
- `--output`: Output parquet file
- `--window-size`: Window size (default: 105)
- `--stride`: Sliding window stride (default: 10 for faster extraction)
- `--max-samples`: Maximum windows to extract (default: 120000)
- `--exclude`: Labeled data file to exclude windows from

---

## Decision Tree

```
Do you have unlabeled data?
├─ YES (on RunPod) → Proceed with SSL ✅
│  └─ Implement 4 pipeline scripts
│
├─ YES (can extract from raw OHLC) → Extract first, then SSL
│  ├─ Run extract_unlabeled_windows.py
│  └─ Then proceed with SSL
│
└─ NO (no raw data available) → Use Mixup fallback
   └─ Train cnn_transformer normally (already has Mixup)
   └─ Expected: +2-4% vs baseline (instead of +5-8%)
```

---

## Next Steps (Immediate)

### On Local Machine
1. ✅ Review this verification checklist
2. ⏳ Decide on data acquisition strategy
3. ⏳ If needed, run extraction script on RunPod

### On RunPod
1. ⏳ SSH into RunPod: `ssh -p 14147 root@213.173.108.148`
2. ⏳ Run verification commands (Step 1 above)
3. ⏳ Report findings:
   - Found existing unlabeled data? → Path and size
   - Found pivot-experiments? → Check for unlabeled windows
   - Need to extract? → Confirm raw OHLC data location
   - None available? → Fallback to Mixup

### After Data Verification
1. If data exists → Continue implementing SSL pipelines
2. If data needs extraction → Run extraction, then pipelines
3. If no data → Skip SSL, use existing Mixup (already implemented)

---

## Recommended Action

**Most Likely Scenario**: Unlabeled windows exist on RunPod in `/workspace/data/` somewhere

**Recommended Next Step**:
1. SSH into RunPod
2. Run comprehensive search:
   ```bash
   find /workspace -name "*.parquet" -size +10M 2>/dev/null
   ls -lh /workspace/data/raw/
   ls -lh /workspace/data/processed/
   ```
3. Report back with findings

If RunPod has the data, we can proceed immediately with SSL implementation!

---

**Status**: Awaiting RunPod data verification
**Blocker**: Need to confirm 118k unlabeled windows location
**Fallback**: Mixup augmentation (already implemented, +2-4% expected gain)
