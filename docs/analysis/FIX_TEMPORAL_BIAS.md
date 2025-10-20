# Fix for Temporal Bias Bug

## Quick Fix (Option 1 - Recommended)

### 1. Fix Backend Service

**File:** `candlesticks/backend/services/window_loader_service.py`
**Line:** 178

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
    start_time = pd.to_datetime(window_data.get('start_ts'))
    if start_time is not None:
        base_time = start_time + pd.Timedelta(minutes=52)
    else:
        # Last resort fallback (should never happen with batch_200)
        base_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        print(f"[WARNING] Window {window_data.get('window_id')} missing both center_timestamp and start_ts, using current time")
```

### 2. Backfill Existing Data (Optional but Recommended)

This adds the `center_timestamp` field to batch_200.parquet so future code works seamlessly:

```bash
python3 << 'EOF'
import pandas as pd

# Load batch_200.parquet
batch_df = pd.read_parquet('data/batches/batch_200.parquet')

# Compute center_timestamp (middle of 105-bar window)
batch_df['center_timestamp'] = batch_df['start_ts'] + pd.Timedelta(minutes=52)

# Save back
batch_df.to_parquet('data/batches/batch_200.parquet', index=False)

print(f"✓ Added center_timestamp to {len(batch_df)} windows")
print(f"Sample center_timestamp: {batch_df['center_timestamp'].iloc[0]}")
EOF
```

### 3. Restart Backend

```bash
cd /Users/jack/projects/moola/candlesticks/backend
python3 app.py
```

### 4. Verify Fix

Open browser to `http://localhost:8056` and check:

1. Click "Load Next Window"
2. Check timestamp in UI (should NOT be 7pm for all windows)
3. Expected: Diverse timestamps across different hours

**Expected Results:**
- Window 0: ~23:39 UTC (not 7pm)
- Window 1: ~00:33 UTC (not 7pm)
- Window 2: ~23:00 UTC (not 7pm)
- etc.

---

## Full Fix (Option 3 - For Future Batches)

### Update Extraction Script

**File:** `scripts/extract_batch_200.py`
**Line:** 187-198

```python
def generate_candidates(df: pd.DataFrame, yearly_stats: Dict[str, float]) -> pd.DataFrame:
    # ... existing code ...

    for i in range(total_possible):
        # ... existing code ...

        # Get session
        start_ts = window.iloc[0]["timestamp"]
        end_ts = window.iloc[-1]["timestamp"]

        # ADD THIS: Compute center timestamp
        center_idx = i + T // 2
        center_ts = df.iloc[center_idx]["timestamp"]

        session = get_session_from_timestamp(start_ts)

        candidates.append(
            {
                "raw_start_idx": i,
                "raw_end_idx": i + T - 1,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "center_timestamp": center_ts,  # ADD THIS LINE
                "session": session,
                "rv": rv,
                "range_norm": range_norm,
                "trend_mag": trend_mag,
                "ohlc": ohlc_array,
            }
        )
```

---

## Testing Script

```python
# test_timestamp_fix.py
import pandas as pd
import requests

# 1. Verify data has diverse timestamps
batch_df = pd.read_parquet('data/batches/batch_200.parquet')
hours = batch_df['start_ts'].dt.hour.unique()
print(f"✓ Data has {len(hours)} unique hours: {sorted(hours)}")

# 2. Test backend API
response = requests.get('http://localhost:8056/api/windows/next')
if response.status_code == 200:
    window = response.json()['window']
    first_bar_ts = window['ohlc'][0]['timestamp']
    hour = pd.to_datetime(first_bar_ts).hour
    print(f"✓ Backend returned timestamp: {first_bar_ts}")
    print(f"  Hour: {hour} (should NOT always be 19)")
else:
    print(f"✗ Backend error: {response.status_code}")

# 3. Load 10 windows and check hour diversity
hours_seen = set()
for i in range(10):
    response = requests.get('http://localhost:8056/api/windows/next')
    if response.status_code == 200:
        window = response.json()['window']
        first_bar_ts = window['ohlc'][0]['timestamp']
        hour = pd.to_datetime(first_bar_ts).hour
        hours_seen.add(hour)
    else:
        break

print(f"\n✓ First 10 windows span {len(hours_seen)} unique hours: {sorted(hours_seen)}")
if len(hours_seen) >= 3:
    print("✓ PASS: Timestamps are diverse (no 7pm bias)")
else:
    print("✗ FAIL: Timestamps still showing bias")
```

Run:
```bash
python3 test_timestamp_fix.py
```

---

## Expected Output After Fix

```
✓ Data has 17 unique hours: [0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23]
✓ Backend returned timestamp: 2025-07-14T23:39:00+00:00
  Hour: 23 (should NOT always be 19)

✓ First 10 windows span 7 unique hours: [0, 14, 16, 19, 22, 23]
✓ PASS: Timestamps are diverse (no 7pm bias)
```

---

## Rollback Plan

If fix causes issues:

1. Revert backend change:
   ```bash
   git checkout candlesticks/backend/services/window_loader_service.py
   ```

2. Restore original batch_200.parquet (if backfilled):
   ```bash
   git checkout data/batches/batch_200.parquet
   ```

3. Restart backend

---

## Summary

**Minimum fix:** Edit line 178 in window_loader_service.py
**Full fix:** Also backfill center_timestamp + update extraction script
**Testing:** Verify UI shows diverse hours (not all 7pm)
**Impact:** Enables proper temporal pattern learning in model
