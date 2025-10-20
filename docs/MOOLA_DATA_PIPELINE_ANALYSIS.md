# Moola Data Pipeline for Candlesticks Annotation Training

## Executive Summary

The Moola project uses a sophisticated data pipeline that integrates raw OHLC data with pre-training on unlabeled windows, followed by annotation and fine-tuning on labeled samples. The Candlesticks tool provides a keyboard-first interface for annotating 105-bar windows, with tracking mechanisms for annotation status and quality review.

---

## 1. Raw Data Location & Structure

### Primary Data Locations

```
/Users/jack/projects/moola/
├── data/
│   ├── raw/
│   │   ├── unlabeled_windows.parquet    (11,873 samples, 2.2 MB)
│   │   └── (source: processed from market_data_*.parquet)
│   │
│   └── processed/
│       ├── train_pivot_134.parquet     (105 labeled samples)
│       ├── train_clean.parquet         (backup)
│       ├── train_smote_300.parquet     (augmented version)
│       └── [other experimental versions]
│
└── candlesticks/
    └── data/
        ├── raw/
        │   └── nq_1min_raw.parquet      (118k candles, NQ futures)
        └── annotations/
            ├── fresh/                   (new annotations)
            └── multi_expansion_v3/      (archived)
```

### Data Formats

**Raw OHLC Structure:**
```python
{
    'timestamp': datetime,
    'open': float,
    'high': float,
    'low': float,
    'close': float
}
```

**Time-Series Window (105 bars):**
```python
{
    'window_id': str,              # e.g., "window_0", "0_exp_1"
    'features': List[List[float]]  # [105, 4] OHLC array
                                   # [open, high, low, close] per bar
}
```

**Labeled Window:**
```python
{
    'window_id': str,              # e.g., "0_exp_1"
    'label': str,                  # "consolidation" or "retracement"
    'expansion_start': int,        # Start index in [30, 74]
    'expansion_end': int,          # End index in [30, 74]
    'features': List[List[float]]  # [105, 4] OHLC
}
```

---

## 2. Pre-training Data & Organization

### Unlabeled Windows Dataset

**Location:** `/Users/jack/projects/moola/data/raw/unlabeled_windows.parquet`

**Statistics:**
- 11,873 samples
- 105 bars per sample (30 past + 45 prediction + 30 future)
- Pre-training window: bars [0:105]
- Derived from: Raw OHLC market data

**Usage:**
- Pre-training phase: BiLSTM masked autoencoder training
- Encoder learns rich OHLC representations without labels
- 15% masking ratio applied during pre-training
- Produces encoder weights for transfer to SimpleLSTM

### Window Structure Breakdown

```
105-bar window structure:
[0-29]     = Past context (30 bars)
[30-74]    = Prediction window (45 bars) - WHERE patterns occur
[75-104]   = Future outcome (30 bars)

Pre-training: Uses full [0-105] sequence
Fine-tuning:  Focuses on [30-74] expansion region
```

### Data Leakage Risks

**Potential Leakage Point 1: Raw Data Source**
- Both unlabeled windows and labeled data sourced from same raw OHLC file
- **Risk**: High if labeled samples' windows overlap with unlabeled training data
- **Mitigation**: Unlabeled data intentionally uses different time periods or markets

**Potential Leakage Point 2: Expansion Index Leakage**
- Labeled data includes `expansion_start/end` indices
- Pre-training doesn't see these labels (self-supervised)
- **Risk**: Very low - pre-training is self-supervised (reconstructing masked OHLC)

**Potential Leakage Point 3: Feature Engineering**
- Dual-input pipeline extracts engineered features from OHLC
- Both pre-training and labeled data go through same feature extraction
- **Risk**: Acceptable - feature extractors don't see labels

---

## 3. Candlesticks Integration

### Candlesticks Project Structure

```
/Users/jack/projects/candlesticks/ (symlink to moola/candlesticks)
├── README.md                          # Annotation interface docs
├── backend/
│   ├── app.py                         # Flask server (port 8055)
│   ├── config.py                      # Configuration + Moola integration
│   ├── models/
│   │   └── candlesticks_annotator.py # Core annotation logic (v3.0.0)
│   ├── services/                      # Business logic services
│   └── requirements.txt
│
└── frontend/
    ├── src/
    │   ├── components/                # React UI components
    │   ├── state/                     # Zustand state management
    │   └── api/                       # API client
    └── vite.config.ts                 # Build configuration
```

### Data Flow: Candlesticks Integration

```
Raw OHLC Data (nq_1min_raw.parquet)
    ↓
Candlesticks Backend (config.py)
    ├─→ RAW_DATA_PATH = /Users/jack/projects/moola/data/raw
    ├─→ CANDLESTICKS_ANNOTATIONS_DIR = moola/data/corrections/candlesticks_annotations
    └─→ BASELINE_WINDOWS_PATH = moola/data/processed/baseline_windows.parquet
    ↓
Window Selection (105 bars from center timestamp)
    ├─→ Load center timestamp from baseline_windows
    ├─→ Extract 15 bars before + 1 center + 29 after = 45 bars
    ├─→ Or 30 before + 1 center + 29 after = 60 bars (extended mode)
    ↓
Candlesticks UI (keyboard-driven annotation)
    ├─→ Window quality grading (A/B/C/D)
    ├─→ Multi-expansion support
    ├─→ Expansion type (consolidation/retracement/reversal)
    ├─→ Start/End points marking
    ↓
JSON Persistence (batch files)
    └─→ data/corrections/candlesticks_annotations/batch_*.json
        ├─→ window_id
        ├─→ annotation_version (3.0.0)
        ├─→ window_quality
        ├─→ expansions[] (array of annotated patterns)
        └─→ timestamp + annotator_id
```

### Candlesticks Data Format Expected

**Input: baseline_windows.parquet**
```python
{
    'window_id': int,              # 0-204 (205 windows total)
    'center_timestamp': datetime   # Center point of 105-bar window
}
```

**Output: batch_*.json (per annotation)**
```json
{
    "window_id": "0",
    "annotation_version": "3.0.0",
    "window_quality": "B",
    "window_size": 105,
    "expansions": [
        {
            "expansion_id": 1,
            "type": "consolidation",
            "start_bar": 40,
            "end_bar": 50,
            "quality": "A",
            "start_point": {...},
            "end_point": {...}
        }
    ],
    "annotator_notes": "...",
    "timestamp": "2025-10-15T12:14:31.286976Z",
    "annotator_id": "bespoke_user"
}
```

### Current Annotation Progress

**Master Index:** `/Users/jack/projects/moola/data/corrections/candlesticks_annotations/master_index.csv`

Sample of tracked windows:
```
window_id,batch_file,annotation_date,quality_grade,expansion_count
19,batch_19.json,2025-10-15T12:14:00.962561Z,D,0
0,batch_0.json,2025-10-15T12:14:31.289320Z,D,0
110,batch_110.json,2025-10-15T15:18:48.931758Z,D,0
...
```

---

## 4. Sample Selection & Marking System

### Current Annotation Tracking Mechanisms

#### A. Master Index CSV
**File:** `/Users/jack/projects/moola/data/corrections/candlesticks_annotations/master_index.csv`

**Tracks:**
- window_id: Unique identifier
- batch_file: Which JSON file contains the annotation
- annotation_date: When it was annotated
- quality_grade: A/B/C/D (user-assigned quality)
- expansion_count: Number of patterns marked

#### B. CleanLab Review File
**File:** `/Users/jack/projects/moola/data/corrections/cleanlab_reviewed.json`

**Purpose:** Track which samples have been reviewed and corrected

**Sample entries:**
```json
{
  "reviewed_windows": [
    {
      "window_id": 110,
      "reviewed_date": "2025-10-15T12:12:45.672146Z",
      "action": "corrected"
    },
    {
      "window_id": 19,
      "reviewed_date": "2025-10-15T12:14:00.965696Z",
      "action": "skipped"
    }
  ],
  "total_reviewed": 15,
  "last_updated": "2025-10-15T21:26:05.518632Z"
}
```

**Actions tracked:**
- `corrected`: Sample was reviewed and corrections applied
- `skipped`: Sample was reviewed but no corrections needed
- (implicitly) `not_reviewed`: Not in the file yet

#### C. Annotation Template CSV
**File:** `/Users/jack/projects/moola/data/corrections/moola_annotations_template.csv`

**Purpose:** Track sample metadata and correction status

**Key columns:**
```
window_id,original_label,original_expansion_start,original_expansion_end,expansion_length,
corrected_label,corrected_expansion_start,corrected_expansion_end,confidence,
correction_timestamp,notes,correction_type,needs_review
```

**Sample rows (showing pending corrections):**
```
29_exp_4,consolidation,57,57,0,,,,,,,pending,True
97_exp_5,consolidation,63,66,3,,,,,,,pending,False
115_exp_1,consolidation,36,42,6,,,,,,,pending,False
```

**Status values:**
- `pending`: Not yet corrected
- (implicit) `corrected`: Has corrected_label filled in
- `needs_review`: Flagged for manual review (needs_review=True)

#### D. Label Quality Issues (CleanLab Output)
**File:** `/Users/jack/projects/moola/data/corrections/cleanlab_label_issues.csv`

**Purpose:** ML-detected potentially mislabeled samples

**Sample rows:**
```
window_id,given_label,given_label_name,pred_label,pred_label_name,prob_consolidation,prob_retracement,label_quality,is_label_issue
110_exp_1,1,retracement,0,consolidation,0.5550563931465149,0.4449436366558075,0.4449436366558075,True
19_exp_1,1,retracement,0,consolidation,0.5550537705421448,0.4449462294578552,0.4449462294578552,True
```

**Interpretation:**
- `is_label_issue=True`: CleanLab flagged this sample as potentially mislabeled
- Label quality score: Confidence that the given label is correct
- Mismatch between pred_label and given_label suggests annotation error

---

## 5. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAW DATA SOURCES                           │
└─────────────────────────────────────────────────────────────────┘
           │
           ├─→ candlesticks/data/raw/nq_1min_raw.parquet (118k bars)
           └─→ moola/data/raw/unlabeled_windows.parquet (11,873 windows)
                  │
                  ├──────────────┬──────────────┐
                  │              │              │
                  ▼              ▼              ▼
        ┌─────────────────┬────────────────┬────────────────┐
        │  PRE-TRAINING   │   ANNOTATION   │   LABELED DATA │
        │   (Unlabeled)   │  (Candlesticks)│   (Training)   │
        └─────────────────┴────────────────┴────────────────┘
                  │              │              │
                  ▼              ▼              ▼
        ┌─────────────────┬────────────────┬────────────────┐
        │ BiLSTM Encoder  │  105-bar OHLC  │  With Expansion│
        │  (11,873 samp)  │  + Expansion   │    Indices     │
        │                 │   Annotation   │  (98-105 samp) │
        └─────────────────┴────────────────┴────────────────┘
                  │              │              │
                  ▼              ▼              ▼
        ┌─────────────────┬────────────────┬────────────────┐
        │  Encoder        │  Batch JSON    │  Corrections   │
        │  Weights        │  Files         │  Tracked in:   │
        │  (.pt)          │  (batch_*.json)│  - Master Idx  │
        │                 │  + Master CSV  │  - CleanLab    │
        │                 │                │  - Template    │
        └─────────────────┴────────────────┴────────────────┘
                  │              │              │
                  └──────────────┴──────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │  Fine-Tuning Pipeline              │
        │  - Transfer Learning               │
        │  - Feature Engineering (50 feats)  │
        │  - Training with 98 samples        │
        │  - Validation with forward-chain   │
        └────────────────────────────────────┘
```

---

## 6. Potential Leakage Points

### 1. **Raw Data Overlap (HIGH RISK - UNKNOWN)**

**Issue:** Are unlabeled_windows derived from same time period as labeled samples?

**Current Status:** Unknown from code inspection. Need to verify:
- Time ranges of nq_1min_raw.parquet
- Time ranges of unlabeled_windows.parquet
- Time ranges of labeled training data

**Impact:** If overlapping, model learns from labeled data twice (once in pre-training, once in fine-tuning)

**Mitigation Needed:**
```python
# Recommended: Add timestamp validation
def check_data_leakage():
    unlabeled = pd.read_parquet('data/raw/unlabeled_windows.parquet')
    labeled = pd.read_parquet('data/processed/train_pivot_134.parquet')
    
    # Extract timestamps from features[30] (center of window)
    # Check for overlap - should be ZERO
    assert len(unlabeled_times.intersection(labeled_times)) == 0
```

### 2. **Expansion Index Information Leakage (MEDIUM RISK - ACCEPTABLE)**

**Issue:** Labeled data includes expansion_start/end indices not in raw OHLC

**Status:** Acceptable because:
- Pre-training is self-supervised (masks OHLC, doesn't see labels)
- Fine-tuning uses indices for feature engineering (intentional)
- No information flows backward to pre-training

### 3. **CleanLab Feedback Loop (LOW RISK - MITIGATED)**

**Issue:** CleanLab reviewed predictions may bias selection

**Status:** Currently mitigated by:
- `corrected` vs `skipped` distinction
- Explicit `needs_review` flag for manual inspection
- Separate tracking in `cleanlab_reviewed.json`

### 4. **Feature Engineering (LOW RISK - INTENTIONAL)**

**Issue:** Same feature extraction used for both pre-training and fine-tuning

**Status:** Acceptable - feature engineers are label-blind

---

## 7. Current Annotation Workflow & Rejection Tracking

### Annotation Workflow

```
1. Candlesticks UI loads unannotated windows
   └─→ Check master_index.csv for already annotated
   └─→ Load from baseline_windows + raw OHLC

2. User annotates (keyboard-driven)
   ├─→ Quality grade: A/B/C/D
   ├─→ Multi-expansions: type, start, end points
   └─→ Space to save, ↑ to skip

3. Save to JSON batch file
   └─→ batch_0.json, batch_1.json, ..., batch_110.json

4. Update master index
   ├─→ Add row with window_id, batch_file, date, quality_grade
   └─→ This is the "source of truth" for what's been annotated

5. CleanLab Review Phase (Optional)
   ├─→ Run label quality check
   ├─→ Update cleanlab_label_issues.csv
   └─→ Update cleanlab_reviewed.json with corrections
```

### Rejection/Exclusion Mechanisms

**Current mechanisms are IMPLICIT, not explicit:**

| Mechanism | Storage | Status | Notes |
|-----------|---------|--------|-------|
| Quality Grade D | master_index.csv | Implicit exclusion | No filter built in |
| Empty Annotations | batch_*.json | Implicit exclusion | expansion_count=0 |
| CleanLab Flagged | cleanlab_label_issues.csv | Tracked but not used | is_label_issue=True |
| needs_review=True | moola_annotations_template.csv | Tracked but not used | Pending manual review |

### Missing Explicit Tracking

**There is NO built-in mechanism to:**
- Mark a sample as "REJECTED" (must exclude manually)
- Track WHY a sample was rejected
- Filter rejected samples from training
- Bulk remove batches of poor annotations

---

## 8. Recommendations for Marking Rejected Samples

### Option 1: Extend master_index.csv (Minimal Changes)

Add column to track rejection status:
```csv
window_id,batch_file,annotation_date,quality_grade,expansion_count,status,rejection_reason
19,batch_19.json,2025-10-15T12:14:00.962561Z,D,0,rejected,insufficient_quality
0,batch_0.json,2025-10-15T12:14:31.289320Z,D,0,accepted,
110,batch_110.json,2025-10-15T15:18:48.931758Z,D,0,pending_review,cleanlab_flagged
```

**Pros:**
- Single file to manage
- Easy to filter: `df[df.status != 'rejected']`
- Version-controlled

**Cons:**
- No detailed audit trail
- Hard to track multiple rejection reasons

### Option 2: Create Rejection Log JSON (Recommended)

Create `/Users/jack/projects/moola/data/corrections/candlesticks_annotations/rejections.json`:

```json
{
  "rejected_samples": [
    {
      "window_id": 19,
      "batch_file": "batch_19.json",
      "rejection_reason": "quality_d_grade",
      "rejected_date": "2025-10-16T10:30:00Z",
      "rejected_by": "data_curator",
      "notes": "Only 0 expansions marked, likely no pattern in window"
    },
    {
      "window_id": 0,
      "batch_file": "batch_0.json",
      "rejection_reason": "cleanlab_high_uncertainty",
      "rejected_date": "2025-10-16T10:31:00Z",
      "rejected_by": "cleanlab_review",
      "notes": "Predicted label contradicts given label with 45% prob"
    }
  ],
  "rejection_reasons": {
    "quality_d_grade": "Quality grade D (user said low quality)",
    "cleanlab_high_uncertainty": "CleanLab: > 45% probability wrong label",
    "empty_annotation": "No expansions marked (expansion_count=0)",
    "manual_review": "Manually marked as incorrect after human review",
    "overlapping_data": "Time overlap with unlabeled pre-training data",
    "outlier_expansion": "Expansion length outside [2-15] bar range"
  },
  "total_rejected": 2,
  "last_updated": "2025-10-16T10:31:00Z"
}
```

**Pros:**
- Detailed audit trail
- Machine-readable
- Easy to add reasons
- Can filter by reason for analysis

**Cons:**
- Additional file to maintain
- Need code to integrate with training pipeline

### Option 3: Database-backed Tracking (Future)

Would require:
- SQLite or PostgreSQL
- ORM (SQLAlchemy)
- Violates "JSON-only" constraint in project philosophy

**Not recommended for current phase**

### Implementation: Filtering Rejected Samples

```python
# In training pipeline (src/moola/data/load.py)

import json
from pathlib import Path

def load_training_data_with_rejection_filtering():
    """Load training data, excluding rejected samples."""
    
    df = pd.read_parquet('data/processed/train_pivot_134.parquet')
    
    # Load rejection log
    rejection_log_path = Path('data/corrections/candlesticks_annotations/rejections.json')
    
    if rejection_log_path.exists():
        with open(rejection_log_path) as f:
            rejection_data = json.load(f)
            rejected_ids = {
                s['window_id'] for s in rejection_data['rejected_samples']
            }
        
        n_before = len(df)
        df = df[~df['window_id'].astype(str).str.split('_').str[0].astype(int).isin(rejected_ids)]
        n_after = len(df)
        
        logger.info(f"Filtered: {n_before} → {n_after} samples (rejected {n_before - n_after})")
    
    return df
```

---

## 9. Summary Table: Data Locations & Flows

| Component | Location | Format | Purpose | Size |
|-----------|----------|--------|---------|------|
| **Raw OHLC** | candlesticks/data/raw/nq_1min_raw.parquet | Parquet | Source for window extraction | 2.2M |
| **Unlabeled Windows** | data/raw/unlabeled_windows.parquet | Parquet | Pre-training data | 2.2M |
| **Labeled Data** | data/processed/train_pivot_134.parquet | Parquet | Fine-tuning data | 94K |
| **Annotations (Batch)** | data/corrections/candlesticks_annotations/batch_*.json | JSON | Individual window annotations | ~2K each |
| **Master Index** | data/corrections/candlesticks_annotations/master_index.csv | CSV | Annotation tracking | ~5K |
| **CleanLab Results** | data/corrections/cleanlab_label_issues.csv | CSV | ML-detected label issues | ~14K |
| **Review Tracker** | data/corrections/cleanlab_reviewed.json | JSON | Review actions (corrected/skipped) | ~2K |
| **Template** | data/corrections/moola_annotations_template.csv | CSV | Correction metadata | ~6K |

---

## 10. Key Insights & Action Items

### Current State
- ✅ Annotation system is working (Candlesticks v3.0.0)
- ✅ Tracking master index exists
- ✅ CleanLab integration for quality review
- ⚠️ Implicit rejection (no explicit "rejected" status)
- ⚠️ Data leakage risk unknown (timestamps need verification)

### Action Items

1. **URGENT: Verify Data Leakage**
   ```bash
   python3 -c "
   import pandas as pd
   unlabeled = pd.read_parquet('data/raw/unlabeled_windows.parquet')
   labeled = pd.read_parquet('data/processed/train_pivot_134.parquet')
   # Compare time ranges and window IDs
   print(f'Unlabeled: {len(unlabeled)} windows')
   print(f'Labeled: {len(labeled)} windows')
   # Check if any window_ids overlap
   "
   ```

2. **Create Rejection Log** (Recommended: Option 2 above)
   - File: `data/corrections/candlesticks_annotations/rejections.json`
   - Add rejection reasons for all D-grade, empty, and CleanLab-flagged samples

3. **Update Training Pipeline**
   - Integrate rejection filtering in `load_training_data_with_rejection_filtering()`
   - Log filtered counts for audit

4. **Document Timestamp Ranges**
   - Create markdown doc with time ranges of each dataset
   - Publish to data_infrastructure.md

---

**End of Analysis**
**Generated:** 2025-10-18
**Status:** Ready for implementation
