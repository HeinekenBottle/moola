# Candlesticks Project: Annotation Tracking System Audit

**Date:** October 18, 2025  
**Scope:** Read-only audit of existing tracking/marking systems  
**Status:** Complete - NO CHANGES MADE

## Executive Summary

The Candlesticks project has a **fully functional, well-designed annotation tracking system** with three separate persistence layers and sophisticated rejection handling. Rejections are already marked as **D-grade windows with 0 expansions**, making the current system extensible for enhancement without redesign.

---

## 1. EXISTING ANNOTATION TRACKING

### 1.1 Three-Layer Architecture

#### Layer 1: New Annotations (Bespoke Mode)
- **Location:** `/Users/jack/projects/candlesticks/data/corrections/candlesticks_annotations/`
- **Files:** `batch_*.json` (JSON arrays of annotations)
- **Index:** `master_index.csv` (tracks all window IDs)
- **Schema:** v3.0.0 with full OHLC point data, expansion types, phenomenon ranges

#### Layer 2: Review Annotations (CleanLab Mode - SEPARATE)
- **Location:** `/Users/jack/projects/candlesticks/data/corrections/review_corrections/`
- **Files:** Separate batch files for corrections only
- **Index:** `review_master_index.csv` (independent tracking)
- **Purpose:** Corrections to Window105Dataset flagged samples
- **Isolation:** Complete separation from new annotations

#### Layer 3: Legacy Backup
- **Location:** `/Users/jack/projects/candlesticks/data/corrections/multi_expansion_annotations_v2_backup/`
- **Files:** 190+ batch files from v2.3.0 annotations
- **Purpose:** Historical data, not actively used

### 1.2 Window ID Reference System

**Current Scheme:**
```
window_id: integer (0-114 for Window105Dataset, variable for dataset)
Key uniqueness: (window_id) in master_index.csv

Batch naming: batch_<window_id>.json (v3.0.0)
OR batch_<YYYYMMDD>_<sequence>.json (v2.x legacy)
```

**How Windows Are Referenced:**
- Frontend passes `window_id` as string/number to API
- Backend maps to dataset index or sequential position
- master_index.csv tracks: `window_id, batch_file, annotation_date, quality_grade, expansion_count`

---

## 2. REJECTION/SKIP MARKING SYSTEM (ALREADY EXISTS!)

### 2.1 Current "Rejection" Mechanism: D-Grade Windows

**How It Works:**
```json
{
  "window_id": 22,
  "window_quality": "D",
  "expansions": [],
  "bars": 45,
  "annotation_timestamp": "2025-09-30T18:26:03.858278"
}
```

**Key Properties:**
- `window_quality: "D"` = **Rejected/Skipped/Unusable**
- `expansions: []` = No expansions marked (empty list)
- Still persisted to batch files and master_index
- Tracked in `num_expansions=0` in master_index

**Evidence from Data:**
```csv
22,2024-09-06 16:26:00,D,0,2025-09-30T18:26:03.858278,batch_017.json
24,2024-09-09 01:41:00,D,3,2025-09-30T18:28:23.427783,batch_018.json
36,2024-09-10 07:37:00,D,1,2025-09-30T20:13:37.311350,batch_030.json
41,2024-09-10 18:32:00,D,0,2025-10-01T13:06:56.841336,batch_055.json
```

Multiple D-grade windows already exist in dataset!

### 2.2 Rejection Prevention: Dual Mechanisms

#### Backend Prevention (Lines 817-825 in useAnnotationStore.ts):
```typescript
if (quality === "D") {
  set({
    status: {
      message: "⛔ Cannot save annotation on D-grade window. Use 'Skip' or quality override to proceed.",
      tone: "error",
    },
  });
  return;
}
```

#### Frontend Skip Button (skipWindow):
- Automatically sets `quality: "D"`, `expansions: []`
- Can navigate with "Next" button to skip without annotating
- Still saves to persistent storage

### 2.3 CleanLab "Reviewed" Tracking

**Independent Rejection Tracking for Review Mode:**
- File: `/Users/jack/projects/moola/data/corrections/cleanlab_reviewed.json`
- Tracks: `{ reviewed_windows: [{window_id, reviewed_date, action}] }`
- Actions: `"corrected"` or `"skipped"`
- Used to prevent re-reviewing already-processed windows

**Code Evidence:**
```python
# cleanlab_loader.py, line 283
self.reviewed_windows.add(window_id)
self._persist_reviewed(window_id, action)
```

---

## 3. STATE PERSISTENCE

### 3.1 Master Index Structure

**File:** `/Users/jack/projects/candlesticks/data/corrections/<mode>/master_index.csv`

**Schema:**
```csv
window_id,center_timestamp,window_quality,num_expansions,annotation_timestamp,batch_file
```

**Key Points:**
- Single row per window (collision detection prevents duplicates)
- `window_quality` includes: A, B, C, D
- `num_expansions`: 0 for D-grade (rejected)
- Updated in real-time after every save (annotation_persistence_service.py, line 107-112)

### 3.2 Batch File Structure

**Format:** JSON array of annotation objects
```json
[
  {
    "schema_version": "3.0.0",
    "window_id": 40,
    "window_quality": "B",
    "expansions": [...],
    "annotation_timestamp": "2025-09-30T20:18:52.521194"
  }
]
```

**Note:** Each batch file can contain ONE annotation per window_id (duplicates removed, line 622)

### 3.3 Audit Trail

**File:** `/Users/jack/projects/candlesticks/data/integrity/registry/audit_log.json`

Tracks all operations:
- registration_timestamp
- operation_type: `register_sample`, `validate_dataset`
- dataset_type
- result

**Currently Used For:**
- Data integrity validation (cross-contamination prevention)
- Not directly for rejection tracking, but extensible

---

## 4. FRONTEND/BACKEND COMMUNICATION FLOW

### 4.1 Window Loading Pipeline

**Normal Mode:**
```
Frontend: fetchNextWindow()
  ↓
Backend: GET /api/windows/next
  ↓
WindowLoaderService: get_next_unannotated()
  ↓
Check master_index.csv for existing annotations
  ↓
Returns next unmarked window
  ↓
Frontend: Checks fetchSavedAnnotation(window_id)
  ↓
If exists: loads from batch file
If not: starts fresh
```

**Review Mode (CleanLab):**
```
Frontend: toggleReviewMode()
  ↓
Backend: GET /api/cleanlab/flagged-windows
  ↓
CleanLabReviewer: get_flagged_windows()
  ↓
Load from cleanlab_studio_priority_review.csv
  ↓
Filter: exclude already-reviewed windows
  ↓
Return: sorted by severity
```

### 4.2 Save Pipeline

**New Annotation Save:**
```
Frontend: saveAndNext()
  ↓
API POST /annotations
  ↓
AnnotationPersistenceService.save_annotation()
  ↓
1. Validate annotation (ExpansionValidator)
2. Check collision (master_index.csv)
3. Build v3.0.0 JSON
4. Write to batch_<window_id>.json
5. Update master_index.csv
6. Mark window as reviewed (if CleanLab mode)
```

**Rejection (D-Grade Save):**
```
Frontend: skipWindow()
  ↓
Creates payload with quality="D", expansions=[]
  ↓
API POST /annotations (same endpoint!)
  ↓
Persists to batch file with D-grade marker
  ↓
master_index.csv: num_expansions=0
  ↓
Window marked as reviewed (in CleanLab mode)
```

### 4.3 Collision Detection

**Location:** `annotation_persistence_service.py`, line 42-68

**Logic:**
```python
def check_collision(self, window_id):
    # Search master_index.csv by window_id
    # Return: {exists, window_id, annotation_date, batch_file, quality_grade, expansion_count}
```

**Prevents:** Same window annotated twice without overwrite confirmation

---

## 5. METADATA AND FILTERING

### 5.1 Existing Config Files

**Main Config:** `/Users/jack/projects/candlesticks/backend/config.py`

```python
CANDLESTICKS_ANNOTATIONS_DIR = DATA_DIR / "corrections" / "candlesticks_annotations"
REVIEW_CORRECTIONS_DIR = DATA_DIR / "corrections" / "review_corrections"
WINDOW_SIZE = 105  # bars
TOTAL_WINDOWS = 205  # for Window105Dataset
```

### 5.2 Quality Grading System

**Already Implemented:**
- A: High quality, well-defined patterns
- B: Standard quality, clear patterns
- C: Low quality, ambiguous patterns
- D: **REJECTED** - unsuitable for training

**Usage:**
- Tracked in master_index.csv
- Stored in batch JSON files
- Searchable/filterable by downstream consumers

### 5.3 D-Grade Handling

**Current Filtering Logic (Frontend Store):**
```typescript
// Line 817-825 in useAnnotationStore.ts
if (quality === "D") {
  // Prevent saving annotation on D-grade
  // Require explicit skip or override
}
```

**Backend Logic:**
```python
# app.py, line 153
action = "skipped" if payload.get("window_quality") == "D" else "corrected"
reviewer.mark_window_reviewed(int(window_id), action)
```

---

## 6. FRAGILE AREAS TO AVOID

### 6.1 CRITICAL - Don't Break Window ID Mapping

**File:** `app.py`, line 359-387 (legacy endpoint)

```python
# CRITICAL FIX: Override window_id to use dataset index
# Frontend expects window_id to match the index for annotation loading
formatted_window['window_id'] = dataset_index
```

**Risk:** Mismatch between sequential index (0-114) and original window_id causes:
- Collision detection failures
- Annotation loading failures
- Review mode breakage

**Safe Areas:** Don't modify WindowLoaderService._format_window_response()

### 6.2 Don't Contaminate Review vs New Annotations

**Separation Achieved Via:**
1. Different output directories (candlesticks_annotations vs review_corrections)
2. Different persistence services (AnnotationPersistenceService vs ReviewAnnotationPersistenceService)
3. Different master_index files

**Risk Areas:**
- Don't merge batch files between modes
- Don't modify CleanLabReviewer to write to new annotations dir
- Don't share master_index.csv between modes

### 6.3 JSON Schema Backward Compatibility

**Current Schema:** v3.0.0 (OHLC points with index, price, ohlc type)
**Legacy Schema:** v2.3.0 (simple indices without OHLC)

**Risk:** Loading v2.3.0 files into v3.0.0 logic causes field mismatches

**Safe Areas:**
- expand.expansions must be an array
- Each expansion must have .num field
- OHLC points must be objects with {index, price, ohlc} OR simple integers

### 6.4 Master Index CSV Format

**Do NOT Change Header Order:**
```
window_id,center_timestamp,window_quality,num_expansions,annotation_timestamp,batch_file
```

**DictReader expects exact column names** - used by:
- annotation_persistence_service.py (line 56-57)
- Multiple collision checks

---

## 7. MINIMAL EXTENSION APPROACH FOR REJECTIONS

### 7.1 Current State (No Changes Needed)

Candlesticks ALREADY tracks rejections as **D-grade windows**:
- ✅ Marked in master_index.csv: `window_quality="D"`
- ✅ Marked in batch JSON: `"window_quality": "D", "expansions": []`
- ✅ Filterable by downstream consumers
- ✅ Persistent across sessions
- ✅ Separate action tracking in CleanLab mode ("skipped" vs "corrected")

### 7.2 To Add "Rejection Reason" (Optional Enhancement)

**Minimal Change Option A: Add annotator_notes**

```json
{
  "window_id": 22,
  "window_quality": "D",
  "expansions": [],
  "annotator_notes": "Rejected: No clear pattern, high noise"  // ← Use this field
}
```

**Minimal Change Option B: Add rejection_reason field**

```python
# annotation_persistence_service.py, _build_annotation_v3()
return {
    ...
    'rejection_reason': payload.get('rejection_reason'),  # ← New field (nullable)
    'annotator_notes': payload.get('annotator_notes', '')
}
```

**Safe Because:**
- Doesn't change master_index.csv
- Doesn't break existing logic
- Consumers can ignore field if not present
- Backward compatible with existing D-grade files

### 7.3 To Filter Out Rejections

**Recommended Query Pattern:**
```python
# In consuming code:
with open(master_index_path, 'r') as f:
    reader = csv.DictReader(f)
    usable_windows = [row for row in reader if row['window_quality'] != 'D']
```

**Already Implemented:**
```python
# window_loader_service.py
# get_progress() returns unannotated count (excluding D-grade)
```

---

## 8. EXACT FILE LOCATIONS (REFERENCE)

| Purpose | Path |
|---------|------|
| New Annotations | `/Users/jack/projects/candlesticks/data/corrections/candlesticks_annotations/` |
| New Annotations Index | `/Users/jack/projects/candlesticks/data/corrections/candlesticks_annotations/master_index.csv` |
| Review Corrections | `/Users/jack/projects/candlesticks/data/corrections/review_corrections/` |
| CleanLab Reviewed Tracking | `/Users/jack/projects/moola/data/corrections/cleanlab_reviewed.json` |
| Integrity Registry | `/Users/jack/projects/candlesticks/data/integrity/registry/` |
| Backend Config | `/Users/jack/projects/candlesticks/backend/config.py` |
| Persistence Service | `/Users/jack/projects/candlesticks/backend/services/annotation_persistence_service.py` |
| CleanLab Loader | `/Users/jack/projects/candlesticks/backend/services/cleanlab_loader.py` |
| Frontend Store | `/Users/jack/projects/candlesticks/frontend/src/state/useAnnotationStore.ts` |
| Flask App | `/Users/jack/projects/candlesticks/backend/app.py` |

---

## 9. KEY CODE SNIPPETS FOR REFERENCE

### Rejection Detection (Master Index)
```python
# annotation_persistence_service.py:80
if row['window_id'] == window_id:
    return {
        'quality_grade': row['quality_grade'],  # Will be 'D' for rejections
        'expansion_count': int(row['expansion_count'])  # Will be 0
    }
```

### Collision Check
```python
# app.py:129
collision = annotation_persistence.check_collision(window_id)
if collision and not overwrite:
    raise ValueError(f"Window already annotated.")
```

### D-Grade Save
```python
# useAnnotationStore.ts:1039-1045
const payload = {
    window_id: String(currentWindow.window_id),
    window_quality: "D" as const,
    expansions: [],
    annotator_notes: "Skipped - Quality D",
    // ...
};
```

---

## 10. SUMMARY: MINIMAL INTERVENTION PLAN

### What's Already Working:
- ✅ Rejections marked as D-grade
- ✅ D-grade windows tracked in master_index
- ✅ Separate persistence for review corrections
- ✅ Collision detection prevents duplicate saves
- ✅ Window ID to batch file mapping
- ✅ Timestamp audit trail
- ✅ Backward compatibility with v2.x

### What's Fragile (Don't Touch):
- ❌ Window ID mapping (sequential index vs original ID)
- ❌ Master index CSV structure/header order
- ❌ Batch file naming conventions
- ❌ Separation of new vs review annotations
- ❌ CleanLab reviewed tracking file format

### Safe Extension Points:
- ✅ Add fields to batch JSON (nullable, backward compatible)
- ✅ Add rejection_reason or use annotator_notes
- ✅ Add new columns to master_index (for future fields)
- ✅ Create new filtering logic in consumers

**Conclusion:** Candlesticks has a **robust, well-designed system**. Enhancement requires only careful, minimal additions without redesign.

