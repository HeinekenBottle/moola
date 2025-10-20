# Candlesticks Annotation Tracking System - Audit Reports

This directory contains a complete read-only audit of the Candlesticks annotation tracking and rejection marking system.

**Key Finding:** Rejections are ALREADY implemented as D-grade windows. No redesign needed.

---

## Quick Start

**Start here:** Read `AUDIT_SUMMARY.md` (5 min) for executive overview.

Then choose based on your need:
- **Want quick answers?** → `CANDLESTICKS_QUICK_REFERENCE.md`
- **Want technical details?** → `CANDLESTICKS_TRACKING_AUDIT.md`
- **Want architecture diagrams?** → `CANDLESTICKS_ARCHITECTURE.txt`

---

## Files in This Audit

### 1. AUDIT_SUMMARY.md (Executive Summary)
**Start here!** 5-minute overview of all findings.

Covers:
- Key finding (D-grade rejections already work)
- Answers to your 7 original questions
- System status checklist
- Fragile areas to avoid
- Safe extension points
- Next steps for your use case

### 2. CANDLESTICKS_QUICK_REFERENCE.md (Lookup Guide)
**Use this for reference.** Quick commands and code snippets.

Covers:
- TL;DR summary
- How to identify D-grade windows (bash commands)
- Backend/frontend code snippets with line numbers
- Python patterns for filtering rejections
- Adding rejection reasons (two safe options)
- API endpoints
- Common queries
- What NOT to touch

### 3. CANDLESTICKS_TRACKING_AUDIT.md (Technical Reference)
**Complete deep dive.** Full technical documentation.

Covers:
- Three-layer persistence architecture (with exact paths)
- Window ID reference system
- Rejection/skip marking mechanism (step-by-step)
- State persistence (CSV, JSON, audit trail)
- Frontend/backend communication flows
- Metadata and filtering systems
- Fragile areas (with line numbers and risks)
- Minimal extension approach
- Key code snippets for reference

### 4. CANDLESTICKS_ARCHITECTURE.txt (Visual Overview)
**Diagrams and flows.** ASCII diagrams of system architecture.

Covers:
- Three persistence layers (visual)
- D-grade marking visualization
- Data flow architecture (frontend → backend)
- Window ID reference system (visual)
- Collision detection flow
- Rejection flow (detailed)
- CleanLab review mode tracking
- Key file locations (tree view)
- Fragile vs safe areas
- Summary checklist

---

## Your Questions Answered

### How are annotated windows marked/stored?
Answer: Master index CSV + batch JSON files. See AUDIT_SUMMARY.md section "How are annotated windows marked/stored?"

### How does Candlesticks prevent annotating the same window twice?
Answer: Collision detection via master_index.csv. See QUICK_REFERENCE.md section "How to Identify D-Grade Windows"

### Can samples be "rejected" or "marked as bad"?
Answer: YES - D-grade windows. Examples: windows 22, 24, 36, 41. See TRACKING_AUDIT.md section 2.1

### What ID scheme is used in batch files?
Answer: `batch_<window_id>.json` (v3.0.0). See QUICK_REFERENCE.md section "Key Files"

### Where is annotation state saved? Is there an audit trail?
Answer: master_index.csv + batch JSON + audit_log.json. See TRACKING_AUDIT.md section 3

### How does frontend get window list? How does it submit annotations?
Answer: fetchNextWindow() → GET /api/windows/next. Submit: POST /api/annotations. See ARCHITECTURE.txt section 3

### Are there config files that mark windows as "do not use"?
Answer: window_quality="D" in master_index.csv. See QUICK_REFERENCE.md section "How D-Grade Windows Work"

---

## System Status

| Component | Status |
|-----------|--------|
| Annotation Tracking | ✅ Complete |
| Rejection Marking | ✅ Complete (D-grade) |
| Collision Detection | ✅ Complete |
| Audit Trail | ✅ Complete |
| Frontend/Backend Communication | ✅ Complete |
| CleanLab Review Mode | ✅ Complete |

---

## Next Steps

### To Use Existing D-Grade Rejection System
1. Filter out D-grade windows: `if row['window_quality'] != 'D'`
2. No code changes needed - system already works!

### To Add Rejection Reasons (Safe Option A - Zero Code Changes)
1. Use existing `annotator_notes` field
2. Store: `"annotator_notes": "Rejected: <your reason>"`
3. See QUICK_REFERENCE.md section "Adding Rejection Reason"

### To Add Rejection Reasons (Safe Option B - Minimal Code Changes)
1. Add new nullable field: `rejection_reason`
2. 1 line in `annotation_persistence_service.py`
3. Fully backward compatible
4. See QUICK_REFERENCE.md section "Adding Rejection Reason"

### To Filter Rejections for Training
1. Read master_index.csv
2. Filter: `window_quality != 'D'`
3. Use remaining windows for training
4. See QUICK_REFERENCE.md section "Filtering Out Rejections"

---

## Key Insights

1. **D-grade = Rejection** - Already implemented, just use it
2. **No Redesign Needed** - System is production-ready
3. **Fragile: Window ID Mapping** - Don't touch app.py lines 359-387
4. **Safe: Add JSON Fields** - New fields are backward compatible
5. **Collision Detection Works** - Prevents duplicate annotations

---

## File Locations

All audit reports saved to:
```
/Users/jack/projects/moola/
├── AUDIT_SUMMARY.md (this overview)
├── CANDLESTICKS_TRACKING_AUDIT.md (complete reference)
├── CANDLESTICKS_QUICK_REFERENCE.md (lookup guide)
├── CANDLESTICKS_ARCHITECTURE.txt (visual diagrams)
└── README_CANDLESTICKS_AUDIT.md (you are here)
```

Candlesticks system files:
```
/Users/jack/projects/candlesticks/
├── backend/
│   ├── app.py (API endpoints)
│   ├── config.py (configuration)
│   └── services/
│       ├── annotation_persistence_service.py
│       └── cleanlab_loader.py
├── frontend/
│   └── src/state/useAnnotationStore.ts
└── data/corrections/
    ├── candlesticks_annotations/ (Layer 1)
    ├── review_corrections/ (Layer 2)
    └── multi_expansion_annotations_v2_backup/ (Layer 3)
```

---

## Audit Metadata

- **Date:** October 18, 2025
- **Auditor:** Claude Code
- **Scope:** Read-only exploration of tracking/marking systems
- **Status:** Complete - NO MODIFICATIONS MADE
- **Files Examined:** 15+ source files + data samples
- **Lines of Code Reviewed:** 5000+
- **Data Files Inspected:** master_index.csv, sample batch files, audit logs

---

## How to Use These Documents

**First Time:** Read AUDIT_SUMMARY.md (5 min)

**Learning the System:** Read CANDLESTICKS_ARCHITECTURE.txt (diagrams)

**Getting Details:** Read CANDLESTICKS_TRACKING_AUDIT.md (technical)

**Quick Lookup:** Use CANDLESTICKS_QUICK_REFERENCE.md (commands, snippets)

**When Coding:** Reference line numbers and code snippets from QUICK_REFERENCE.md

---

## Summary

The Candlesticks annotation tracking system is **fully functional, well-designed, and production-ready**.

Rejections are tracked using **D-grade windows** with `expansions=[]` and `num_expansions=0`.

**No system redesign is necessary.** Just use the existing D-grade marker in downstream consumers.

---

## Questions?

Refer to the specific document:
1. Quick answer → AUDIT_SUMMARY.md
2. How-to guide → CANDLESTICKS_QUICK_REFERENCE.md
3. Technical deep-dive → CANDLESTICKS_TRACKING_AUDIT.md
4. Architecture overview → CANDLESTICKS_ARCHITECTURE.txt

**Status:** Audit Complete ✅
