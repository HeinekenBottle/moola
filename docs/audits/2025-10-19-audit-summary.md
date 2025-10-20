# Candlesticks Project: Annotation Tracking Audit - SUMMARY

**Date:** October 18, 2025  
**Auditor:** Claude Code  
**Scope:** Read-only exploration of tracking/marking systems  
**Status:** Complete - NO MODIFICATIONS MADE

---

## Key Finding: REJECTIONS ALREADY IMPLEMENTED

**Candlesticks has a fully functional rejection system using D-grade windows.**

The project already tracks sample rejection/marking using a simple, elegant approach:
- Mark rejected samples with `window_quality="D"` 
- Track in `master_index.csv` with `num_expansions=0`
- Persist to batch JSON files
- Prevent re-annotation via collision detection
- Enable filtering in downstream consumers

**No redesign necessary - system is production-ready.**

---

## Three Documents Created

### 1. CANDLESTICKS_TRACKING_AUDIT.md (14KB)
**Complete technical audit** - read this first if you want all details.

Contains:
- Three-layer persistence architecture breakdown
- Window ID reference system (exact)
- Rejection/skip marking mechanism (step-by-step)
- State persistence (master_index, batch files, audit trail)
- Frontend/backend communication flows
- Metadata and filtering systems
- Fragile areas (DO NOT TOUCH)
- Minimal extension approach
- Exact file locations and line numbers

### 2. CANDLESTICKS_QUICK_REFERENCE.md (6KB)
**Quick lookup guide** - use this for common tasks.

Contains:
- TL;DR at top
- How to identify D-grade windows
- Key files with line numbers
- How D-grade windows work (code snippets)
- Filtering patterns (Python + shell)
- Adding rejection reason (two safe options)
- API endpoints for rejection checking
- Common queries
- What NOT to touch

### 3. CANDLESTICKS_ARCHITECTURE.txt (16KB)
**Visual system overview** - diagrams and flows.

Contains:
- ASCII diagrams of three persistence layers
- D-grade marking visualization
- Data flow architecture (frontend → backend)
- Window ID reference system
- Collision detection flow
- Rejection flow (detailed)
- Clean Lab review mode tracking
- Key file locations (tree view)
- Fragile vs safe areas
- Summary checklist

---

## Quick Answer to Your Questions

### 1. How are annotated windows marked/stored?
- **Master Index:** `/data/corrections/candlesticks_annotations/master_index.csv`
  - One row per window: `window_id, center_timestamp, window_quality, num_expansions, annotation_timestamp, batch_file`
- **Batch Files:** JSON arrays saved to `batch_<window_id>.json`
- **Collision Detection:** Prevents same window annotated twice

### 2. How does Candlesticks prevent annotating the same window twice?
- Backend checks `master_index.csv` on every save
- If window_id exists and no `overwrite=true`, returns 409 collision error
- Frontend shows collision dialog with annotation date and quality

### 3. Can samples be "rejected" or "marked as bad"?
- **YES - D-grade windows**
- Set `window_quality="D"` and `expansions=[]`
- Multiple examples in current data (windows 22, 24, 36, 41...)
- Frontend prevents saving on D-grade (requires skip or override)
- Backend persists D-grade to batch files and master_index

### 4. What ID scheme is used in batch files?
- **Schema v3.0.0:** Batch files named `batch_<window_id>.json`
- **Legacy v2.x:** Named `batch_<YYYYMMDD>_<sequence>.json` (read-only)
- Schema extensible for "rejected" status - already have D-grade marker

### 5. Where is annotation state saved? Is there an audit trail?
- **State Saved:** `/data/corrections/candlesticks_annotations/`
  - master_index.csv (tracks all changes)
  - Batch JSON files (persisted with timestamp)
- **Audit Trail:** `/data/integrity/registry/audit_log.json`
  - All operations logged (register_sample, validate_dataset, etc.)
  - Timestamps all changes
- **Batch Files:** Also track (for CleanLab review mode)
  - `/data/corrections/cleanlab_reviewed.json`
  - Tracks action: "skipped" or "corrected"

### 6. How does frontend get window list? How does it submit annotations?
- **Get Windows:**
  - Frontend: `fetchNextWindow()` or `fetchWindow(<id>)`
  - Backend: Checks `master_index.csv` for unannotated
  - Returns next window with full OHLC data
- **Submit Annotations:**
  - Frontend: `POST /api/annotations` with payload
  - Backend: AnnotationPersistenceService validates, checks collision, saves
  - Returns: `{status: "saved", window_id, batch_file, expansion_count}`

### 7. Are there config files that mark windows as "do not use"?
- **Backend Config:** `/backend/config.py`
  - `CANDLESTICKS_ANNOTATIONS_DIR` (new annotations)
  - `REVIEW_CORRECTIONS_DIR` (review mode)
- **Window Quality Grades:** A, B, C, D
  - D = "do not use" (rejected)
- **Filtering Logic:** Downstream consumers check `window_quality != 'D'`

---

## System Status

| Aspect | Status | Details |
|--------|--------|---------|
| **Annotation Tracking** | ✅ Complete | 3-layer architecture, separate modes |
| **Window ID System** | ✅ Complete | Sequential index + original ID mapping |
| **Rejection Marking** | ✅ Complete | D-grade windows (num_expansions=0) |
| **State Persistence** | ✅ Complete | master_index.csv + batch JSON files |
| **Collision Detection** | ✅ Complete | Prevents duplicate saves |
| **Audit Trail** | ✅ Complete | audit_log.json logs all operations |
| **Frontend/Backend Separation** | ✅ Complete | Clean API boundaries, state management |
| **Review Mode (CleanLab)** | ✅ Complete | Separate tracking (cleanlab_reviewed.json) |
| **Backward Compatibility** | ✅ Complete | v2.3.0 and v3.0.0 coexist |

---

## Fragile Areas (DO NOT TOUCH WITHOUT REVIEW)

1. **Window ID Mapping** (app.py:359-387)
   - Sequential index (0-114) vs original window_id
   - Mismatch causes collision detection failures

2. **Master Index CSV Format**
   - Header order MUST NOT CHANGE
   - DictReader expects exact column names

3. **Batch File Naming**
   - v3.0.0: `batch_<window_id>.json`
   - Don't change without updating collision detection

4. **Layer Separation**
   - `candlesticks_annotations/` vs `review_corrections/`
   - Use different persistence services
   - Don't merge batch files between modes

5. **JSON Schema Version**
   - v3.0.0: OHLC points with {index, price, ohlc_type}
   - v2.3.0: Simple indices (legacy)
   - Loading logic must handle both

---

## Safe Extension Points

1. **Add Fields to Batch JSON** (backward compatible)
   - New fields default to nullable
   - Existing consumers ignore unknown fields

2. **Rejection Reason** (two options, both safe)
   - Option A: Use `annotator_notes` field (no code changes)
   - Option B: Add `rejection_reason` field (new field, nullable)

3. **Add Columns to master_index.csv**
   - Add at END only (don't change header order)
   - New columns are optional

4. **Audit Trail Extensions**
   - Log rejection reasons to `audit_log.json`
   - Create rejection analytics

5. **Filtering Logic**
   - Create consumers that check `window_quality != 'D'`
   - Filter out before training data prep

---

## File Manifest

```
/Users/jack/projects/moola/
├── CANDLESTICKS_TRACKING_AUDIT.md      (14KB - Complete technical audit)
├── CANDLESTICKS_QUICK_REFERENCE.md     (6KB - Quick lookup guide)
├── CANDLESTICKS_ARCHITECTURE.txt       (16KB - Visual diagrams + flows)
└── AUDIT_SUMMARY.md                     (This file)

/Users/jack/projects/candlesticks/
├── data/corrections/
│   ├── candlesticks_annotations/        (Layer 1: New annotations)
│   │   ├── batch_*.json
│   │   └── master_index.csv
│   ├── review_corrections/              (Layer 2: CleanLab corrections)
│   │   └── (similar structure)
│   └── multi_expansion_annotations_v2_backup/  (Layer 3: Legacy)
├── backend/
│   ├── app.py                          (API endpoints)
│   ├── config.py                       (Configuration)
│   └── services/
│       ├── annotation_persistence_service.py
│       └── cleanlab_loader.py
└── frontend/
    └── src/state/useAnnotationStore.ts (State management)
```

---

## Next Steps

### If You Want to Add Rejection Reasons
1. Read: `CANDLESTICKS_QUICK_REFERENCE.md` section "Adding Rejection Reason"
2. Choose Option A (use existing `annotator_notes`) - ZERO CODE CHANGES
3. Or choose Option B (add `rejection_reason` field) - 1-2 lines in backend

### If You Want to Filter Rejections for Training
1. Read: `CANDLESTICKS_QUICK_REFERENCE.md` section "Filtering Out Rejections"
2. Use pattern: `if row['window_quality'] != 'D'`
3. Done - no code changes needed

### If You Want to Understand Architecture
1. Read: `CANDLESTICKS_ARCHITECTURE.txt` for visual overview
2. Read: `CANDLESTICKS_TRACKING_AUDIT.md` for technical details
3. Reference: `CANDLESTICKS_QUICK_REFERENCE.md` for code locations

### If You Want to Extend the System
1. Check `CANDLESTICKS_TRACKING_AUDIT.md` section 7 "Minimal Extension Approach"
2. Identify fragile areas to avoid
3. Use safe extension points
4. Test collision detection after changes

---

## Key Insights

### 1. Rejections Are Already Implemented
- D-grade windows serve as rejection marker
- No code changes needed to track rejections
- Just consume `window_quality="D"` in downstream systems

### 2. System Is Well-Designed
- Three clean layers (new annotations, review, legacy)
- Separate persistence services per mode
- Collision detection prevents data corruption
- Audit trail for compliance

### 3. Expansion Is Safe
- Most extensions are backward compatible
- Can add fields without breaking existing code
- Master index can grow (add columns at end)
- Config files are separate from data

### 4. Window ID Mapping Is Critical
- Sequential index (0-114) != Original window_id
- One mismatch breaks collision detection
- This is the most fragile area

### 5. No Redesign Needed
- Current system is production-ready
- Rejection tracking works via D-grade marker
- Can enhance without touching core logic

---

## Verification

All claims in this audit are supported by:
- ✅ Read source code (backend services, frontend store, API routes)
- ✅ Examined actual data files (master_index.csv, sample batch files)
- ✅ Traced data flows (frontend → backend → persistence)
- ✅ Identified fragile areas (with line numbers and risks)
- ✅ Verified collision detection logic
- ✅ Checked schema versions and backward compatibility

**No changes made to any files during audit.**

---

## Contact / Questions

Refer to the three detailed documents:
1. **CANDLESTICKS_TRACKING_AUDIT.md** - Full technical reference
2. **CANDLESTICKS_QUICK_REFERENCE.md** - Fast lookup
3. **CANDLESTICKS_ARCHITECTURE.txt** - Visual overview

All documents are in `/Users/jack/projects/moola/`

---

**Audit Complete - System Ready for Use**

