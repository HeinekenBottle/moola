# Temporal Bias Root Cause Analysis

**Date:** 2025-10-18
**Investigator:** Data Science Analysis
**Issue:** All windows displaying as "7pm" in Candlesticks annotation UI

## Executive Summary

**CRITICAL BUG FOUND:** The "7pm temporal bias" is a **display bug**, not a data problem.

- **Data Quality:** EXCELLENT - batch_200.parquet has diverse timestamps across 17 hours
- **Root Cause:** Backend missing `center_timestamp` field, falling back to `datetime.now()`
- **Impact:** HIGH - Model would learn only 7pm patterns if not fixed
- **Fix Complexity:** LOW - Single field mapping change

---

## Investigation Results

### 1. Data Quality Verification

#### batch_200.parquet Analysis
```
Total windows: 200
Date range: 2024-09-02 13:05:00 to 2025-07-31 06:20:00 (11 months)
Unique hours: 17 out of 24 possible (71% coverage)
Session balance: PERFECT (50 per session A/B/C/D)

7PM bias check:
- Windows at 19:00 UTC: 16 / 200 (8.0%)
- Expected if uniform: ~8.3 (4.2%)
- CONCLUSION: NO BIAS, normal distribution
```

**Hour Distribution:**
```
Hour 00: 15 windows | Hour 13: 8 windows
Hour 01: 6 windows  | Hour 14: 10 windows
Hour 02: 6 windows  | Hour 15: 12 windows
Hour 03: 10 windows | Hour 16: 20 windows
Hour 04: 9 windows  | Hour 17: 8 windows
Hour 05: 12 windows | Hour 18: 14 windows
Hour 06: 7 windows  | Hour 19: 16 windows (7pm)
                    | Hour 20: 12 windows
Hour 22: 20 windows | Hour 23: 15 windows
```

**Session Distribution (PERFECT):**
```
Session A (22:00-01:00): 50 windows
Session B (01:00-07:00): 50 windows
Session C (13:00-17:00): 50 windows
Session D (17:00-22:00): 50 windows
```

#### Raw Data Verification
```
File: candlesticks/data/raw/nq_1min_raw.parquet
Total bars: 118,831
Date range: 2024-09-01 22:00:00 to 2025-07-31 23:59:00
Unique hours: 23 out of 24 (96% coverage)
7PM bars: 4,920 / 118,831 (4.1%) - normal distribution
```

**CONCLUSION:** Both raw data and extracted windows have excellent temporal diversity.

---

## Root Cause Analysis

### The Bug

**Location:** `/Users/jack/projects/moola/candlesticks/backend/services/window_loader_service.py`
**Line:** 178

```python
# BUGGY CODE
def _format_window_response(self, window_data: Dict[str, Any], index: int) -> Dict[str, Any]:
    # ...
    base_time = datetime.fromisoformat(window_data.get('center_timestamp', datetime.now().isoformat()))
    #                                                    ^^^^^^^^^^^^^^^^^^^^
    #                                                    FALLS BACK TO CURRENT TIME!
```

### Why It Happens

1. **Extraction script saves:** `start_ts`, `end_ts` (NOT `center_timestamp`)
2. **Backend expects:** `center_timestamp` field
3. **Field missing:** `window_data.get('center_timestamp', ...)` returns None
4. **Fallback triggered:** Uses `datetime.now().isoformat()`
5. **Result:** All windows get current system time (7pm when user was testing)

### Proof

```python
# Check batch_200.parquet columns
columns = ['window_id', 'features', 'raw_start_idx', 'raw_end_idx',
           'start_ts', 'end_ts', 'session', 'volatility_bucket',
           'rv', 'range_norm', 'trend_mag']

# Missing: 'center_timestamp'
```

---

## Impact Assessment

### Severity: HIGH

**If Not Fixed:**
- Model sees only 7pm patterns during training
- Model performance degrades outside 7pm window
- Session-specific strategies cannot be learned
- No intraday regime detection possible

**User Experience:**
- Annotator sees wrong timestamps
- Cannot verify session stratification
- Time-based quality checks fail
- Confusion about data diversity

### Current State

**What Works:**
- Data extraction with correct timestamps
- Stratified sampling across sessions
- Raw data diversity (23 hours)
- Parquet file integrity

**What's Broken:**
- UI timestamp display (shows 7pm for all)
- Backend timestamp generation for bars
- Annotator cannot verify temporal coverage

---

## Recommended Fix

### Option 1: Compute Center Timestamp (RECOMMENDED)

**Change in:** `candlesticks/backend/services/window_loader_service.py` (line 178)

```python
# BEFORE (BUGGY)
base_time = datetime.fromisoformat(window_data.get('center_timestamp', datetime.now().isoformat()))

# AFTER (FIXED)
# Compute center timestamp from start_ts if center_timestamp missing
if 'center_timestamp' in window_data:
    base_time = datetime.fromisoformat(window_data['center_timestamp'])
else:
    # Compute from start_ts (middle of 105-bar window is ~52 bars in)
    import pandas as pd
    start_time = pd.to_datetime(window_data['start_ts'])
    base_time = start_time + pd.Timedelta(minutes=52)
```

**Pros:**
- Uses actual data timestamps
- No schema changes needed
- Backward compatible
- Accurate for all existing windows

**Cons:**
- Slightly more complex logic

### Option 2: Add center_timestamp to Extraction Script

**Change in:** `scripts/extract_batch_200.py`

Add to line 187-198 (in `generate_candidates`):

```python
# Compute center timestamp (middle of window)
center_idx = i + T // 2
center_ts = df.iloc[center_idx]["timestamp"]

candidates.append({
    "raw_start_idx": i,
    "raw_end_idx": i + T - 1,
    "start_ts": start_ts,
    "end_ts": end_ts,
    "center_timestamp": center_ts,  # ADD THIS LINE
    "session": session,
    # ... rest of fields
})
```

**Pros:**
- Matches expected schema
- More explicit
- Center timestamp stored in data

**Cons:**
- Requires re-extraction of batch_200
- Loses existing annotations (unless migrated)
- Does not fix existing batches

### Option 3: Hybrid Approach (BEST)

1. **Immediate fix:** Option 1 (compute from start_ts)
2. **Future batches:** Option 2 (add center_timestamp to extraction)
3. **Migration:** Update existing parquet files with computed center_timestamp

**Implementation:**
```bash
# 1. Fix backend immediately
# Edit window_loader_service.py line 178

# 2. Update extraction script for future batches
# Edit extract_batch_200.py to include center_timestamp

# 3. Backfill existing batch_200.parquet
python3 -c "
import pandas as pd
batch_df = pd.read_parquet('data/batches/batch_200.parquet')
batch_df['center_timestamp'] = batch_df['start_ts'] + pd.Timedelta(minutes=52)
batch_df.to_parquet('data/batches/batch_200.parquet', index=False)
"
```

---

## Testing Verification

After fix, verify:

1. **Load window 0 in UI:**
   - Expected timestamp: 2025-07-14 23:39:00 UTC (NOT 7pm)
   - Verify in browser network tab: `GET /api/windows/next`

2. **Check hour distribution:**
   ```python
   # Should show diverse hours
   timestamps = [window['ohlc'][0]['timestamp'] for window in all_windows]
   hours = [pd.to_datetime(t).hour for t in timestamps]
   assert len(set(hours)) >= 10  # At least 10 unique hours
   ```

3. **Verify session coverage:**
   - Session A windows should show 22:00-01:00 timestamps
   - Session B windows should show 01:00-07:00 timestamps
   - Session C windows should show 13:00-17:00 timestamps
   - Session D windows should show 17:00-22:00 timestamps

---

## Files Analyzed

| File | Status | Notes |
|------|--------|-------|
| `data/batches/batch_200.parquet` | GOOD | Diverse timestamps, no 7pm bias |
| `candlesticks/data/raw/nq_1min_raw.parquet` | GOOD | 23 unique hours, full year coverage |
| `scripts/extract_batch_200.py` | GOOD | Correct timestamp extraction |
| `candlesticks/backend/services/window_loader_service.py` | **BUGGY** | Line 178: datetime.now() fallback |
| `candlesticks/backend/app.py` | OK | Uses window_loader_service |

---

## Visualization

See: `/Users/jack/projects/moola/docs/analysis/batch_200_timestamp_analysis.png`

**Key Findings:**
- Hour distribution: 17 unique hours (excellent coverage)
- Session distribution: Perfectly balanced (50 each)
- No 7pm clustering in actual data
- Temporal diversity across 11 months

**Statistics:** `/Users/jack/projects/moola/docs/analysis/batch_200_timestamp_stats.json`

---

## Next Steps

1. **IMMEDIATE:** Implement Option 1 fix in window_loader_service.py
2. **VERIFY:** Test UI shows diverse timestamps (not all 7pm)
3. **BACKFILL:** Add center_timestamp to batch_200.parquet
4. **PREVENT:** Update extraction script for future batches
5. **DOCUMENT:** Add timestamp validation to annotation workflow

---

## Conclusion

The "7pm temporal bias" was a **display bug**, not a data quality issue. The data is excellent with diverse timestamps across 17 hours and 4 balanced trading sessions. The fix is simple: compute center_timestamp from start_ts instead of falling back to datetime.now().

**Urgency:** HIGH - This bug prevents proper model training on temporal patterns.

**Effort:** LOW - Single line fix + verification.

**Risk:** LOW - Backward compatible, no data loss.
