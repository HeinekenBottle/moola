# Candlesticks Tracking System - Quick Reference

## TL;DR

**Rejections are ALREADY implemented as D-grade windows:**
- Location: `master_index.csv` column `window_quality="D"`
- Batch files: `"window_quality": "D", "expansions": []`
- Multiple examples exist in current data (windows 22, 24, 36, 41...)
- Frontend prevents saving on D-grade (requires skip or override)

---

## How to Identify Rejected Windows

### In Master Index:
```bash
grep "^[0-9]*,.*,D," /Users/jack/projects/candlesticks/data/corrections/candlesticks_annotations/master_index.csv
```

### In Batch Files:
```bash
jq '.[] | select(.window_quality=="D")' batch_*.json
```

### Result Pattern:
```json
{
  "window_id": 22,
  "window_quality": "D",
  "expansions": [],
  "annotation_timestamp": "2025-09-30T18:26:03.858278"
}
```

---

## Three Tracking Layers

| Layer | Purpose | Location | Index |
|-------|---------|----------|-------|
| **New Annotations** | User-created annotations | `candlesticks_annotations/` | `master_index.csv` |
| **Review Corrections** | CleanLab flagged fixes | `review_corrections/` | `review_master_index.csv` |
| **Legacy Backup** | v2.x historical data | `multi_expansion_annotations_v2_backup/` | (read-only) |

---

## Key Files (Don't Touch Without Care)

### CRITICAL:
- `app.py` line 359-387: Window ID mapping (sequential vs original)
- `annotation_persistence_service.py` line 56-57: DictReader CSV parsing
- `master_index.csv` header: Column order matters!

### Safe to Extend:
- Batch JSON fields (add new fields as nullable)
- `annotator_notes` field (use for rejection reason)
- master_index.csv: Can add new columns at end

---

## How D-Grade Windows Work

### Frontend (useAnnotationStore.ts):
```typescript
// Line 817-825: Prevent saving on D-grade
if (quality === "D") {
  set({ status: { message: "⛔ Cannot save annotation on D-grade..." } });
  return;
}

// Line 1039-1045: Skip button creates D-grade
const payload = {
  window_quality: "D",
  expansions: [],
  annotator_notes: "Skipped - Quality D"
};
```

### Backend (annotation_persistence_service.py):
```python
# Validation passes for D-grade (num_expansions can be 0)
# Saves to batch file with quality="D"
# Updates master_index: window_quality="D", num_expansions=0
```

### Result:
- Window marked as "processed" (no re-annotation)
- Flagged as rejected in master_index
- Can filter out during training data prep

---

## Filtering Out Rejections

### Python Consumer Code:
```python
import csv
from pathlib import Path

master_index = Path("/Users/jack/projects/candlesticks/data/corrections/candlesticks_annotations/master_index.csv")

with open(master_index, 'r') as f:
    reader = csv.DictReader(f)
    usable = [row for row in reader if row['window_quality'] != 'D']
    print(f"Usable windows: {len(usable)}")
    print(f"Rejected windows: {sum(1 for row in reader if row['window_quality'] == 'D')}")
```

### Shell One-Liner:
```bash
# Count usable vs rejected
awk -F',' '$3 != "D" {print}' master_index.csv | wc -l  # usable
awk -F',' '$3 == "D" {print}' master_index.csv | wc -l  # rejected
```

---

## Adding Rejection Reason (Safe Enhancement)

### Option 1: Use Existing Field
```json
{
  "window_id": 22,
  "window_quality": "D",
  "annotator_notes": "Rejected: No clear pattern, extreme noise"  // ← This field!
}
```

### Option 2: Add New Field (Backward Compatible)
```python
# In annotation_persistence_service.py:
'rejection_reason': payload.get('rejection_reason'),  # Nullable
'annotator_notes': payload.get('annotator_notes', '')
```

No master_index changes needed - both approaches work!

---

## CleanLab Review Mode Rejection Tracking

**Separate from new annotations:**
```
File: /Users/jack/projects/moola/data/corrections/cleanlab_reviewed.json

{
  "reviewed_windows": [
    {
      "window_id": 10,
      "reviewed_date": "2025-10-15T14:30:00Z",
      "action": "skipped"  // ← or "corrected"
    }
  ]
}
```

Backend marks windows with `action="skipped"` when quality="D"

---

## API Endpoints for Rejection

### Check if Window is Rejected:
```bash
GET /api/annotations/<window_id>

# Returns: {exists: true, annotation: {..., window_quality: "D"}}
# or {exists: false}
```

### Get Window Quality:
```bash
jq -r '.window_quality' batch_<window_id>.json
```

### Check Progress (Excludes D-Grade):
```bash
GET /api/progress

# Returns:
# {
#   "total_windows": 115,
#   "annotated": 95,
#   "unannotated": 20,  ← D-grade NOT counted as "annotated"
#   "quality_distribution": {...}
# }
```

---

## Audit Trail

**All operations logged:**
```
File: /Users/jack/projects/candlesticks/data/integrity/registry/audit_log.json

{
  "operations": [
    {
      "timestamp": "2025-10-18T14:30:00Z",
      "operation_type": "register_sample",
      "sample_id": "window_22",
      "dataset_type": "candlesticks_annotations",
      "result": "success"
    }
  ]
}
```

---

## Common Queries

### All Rejected Windows:
```bash
awk -F',' '$3=="D" {print $1, $5}' master_index.csv
```

### Rejection Count by Day:
```bash
awk -F',' '$3=="D" {print $5}' master_index.csv | cut -d'T' -f1 | sort | uniq -c
```

### Batch File with D-Grade:
```bash
jq '.[] | select(.window_quality=="D") | .window_id' batch_*.json
```

### All Quality Grades:
```bash
awk -F',' '{print $3}' master_index.csv | sort | uniq -c
```

---

## What NOT to Touch

- ❌ Window ID mapping logic in `app.py`
- ❌ CSV header order in `master_index.csv`
- ❌ Batch file naming scheme
- ❌ Separation of `candlesticks_annotations/` vs `review_corrections/`
- ❌ `cleanlab_reviewed.json` structure

---

## Summary

1. **D-grade windows = rejections** (already implemented)
2. **Track via:** `window_quality="D"` in master_index + batch files
3. **Filter out via:** `if row['window_quality'] != 'D'`
4. **Add reason via:** `annotator_notes` field (no code changes!)
5. **No system redesign needed** - just consume existing D-grade marker

---

**Status:** ✅ Fully Functional - Ready to Use
**Implementation:** ✅ Already Complete - No Changes Required
**Extension:** ⚠️ Possible - Use annotator_notes or add rejection_reason field

