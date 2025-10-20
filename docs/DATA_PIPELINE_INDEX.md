# Moola Data Pipeline - Documentation Index

## Quick Navigation

### For Quick Answers (5-10 min read)
Start here if you need fast reference:
- **CANDLESTICKS_DATA_REFERENCE.md** - One-liners, file paths, Python snippets
- **Section 7** of main analysis for file locations
- **Section 9** of main analysis for current state

### For Understanding the Full System (30 min read)
Start here for comprehensive understanding:
- **MOOLA_DATA_PIPELINE_ANALYSIS.md** - Full 10-section analysis
- Read in order: Sections 1-5 for context, 6-8 for risks/recommendations

### For Specific Tasks

#### "Where is the training data?"
See: CANDLESTICKS_DATA_REFERENCE.md → Data Locations table

#### "What is the current annotation progress?"
See: CANDLESTICKS_DATA_REFERENCE.md → Quick Checks → Count annotations by quality
```bash
head -20 /Users/jack/projects/moola/data/corrections/candlesticks_annotations/master_index.csv
```

#### "How do I check for data leakage?"
See: MOOLA_DATA_PIPELINE_ANALYSIS.md → Section 6
Python code provided in Section 6 for verification

#### "How should I mark rejected samples?"
See: MOOLA_DATA_PIPELINE_ANALYSIS.md → Section 8
Recommended: Create `rejections.json` with detailed audit trail

#### "What's in the Candlesticks backend?"
See: MOOLA_DATA_PIPELINE_ANALYSIS.md → Section 3
Backend structure, config, data formats explained

#### "How do I integrate rejection filtering?"
See: MOOLA_DATA_PIPELINE_ANALYSIS.md → Section 8
Python implementation code provided

#### "What are the expansion indices?"
See: CANDLESTICKS_DATA_REFERENCE.md → Window Structure
Visual diagram + example range [30-74]

#### "What's the label distribution?"
See: CANDLESTICKS_DATA_REFERENCE.md → Label Distribution
Currently: 60 consolidation, 45 retracement

---

## File Structure

```
docs/
├── MOOLA_DATA_PIPELINE_ANALYSIS.md      ← Main technical analysis
├── CANDLESTICKS_DATA_REFERENCE.md       ← Quick reference guide
├── DATA_PIPELINE_INDEX.md               ← You are here
├── data_infrastructure.md               ← Existing data framework
└── [other docs...]

data/
├── raw/
│   └── unlabeled_windows.parquet        (Pre-training: 11,873 samples)
├── processed/
│   └── train_pivot_134.parquet          (Fine-tuning: 105 labeled samples)
└── corrections/
    ├── candlesticks_annotations/
    │   ├── master_index.csv             ← Annotation tracker
    │   ├── batch_*.json                 ← Individual annotations (15 files)
    │   ├── rejections.json              ← RECOMMENDED (not yet created)
    │   └── [other tracking files]
    ├── cleanlab_reviewed.json           ← Review outcomes
    ├── cleanlab_label_issues.csv        ← ML-flagged samples
    └── [other correction files]

candlesticks/
└── backend/config.py                    ← Candlesticks configuration
```

---

## Key Findings Summary

### Data Organization
- **11,873 unlabeled** windows for pre-training (BiLSTM)
- **105 labeled** windows for fine-tuning (SimpleLSTM)
- **105-bar windows**: 30 past + 45 prediction + 30 future
- **Expansion indices**: Where patterns occur [30-74] range

### Candlesticks Integration
- Keyboard-first annotation interface
- Stores results as JSON batch files
- Tracks via master_index.csv
- Integrates with Moola via config.py

### Tracking Systems
- **Master Index**: Source of truth (15 annotated windows)
- **CleanLab Reviews**: Quality assessment outcomes
- **Template CSV**: Correction metadata (status tracking)
- **Label Issues CSV**: ML-detected suspicious samples

### Gaps & Recommendations

| Issue | Priority | Solution | File |
|-------|----------|----------|------|
| No explicit rejection tracking | HIGH | Create rejections.json | Section 8 |
| Unknown data leakage | URGENT | Verify timestamp ranges | Section 6 |
| No rejection filtering in pipeline | HIGH | Add load_training_data_with_rejection_filtering() | Section 8 |
| No timestamp documentation | MEDIUM | Create DATA_TIMESTAMP_RANGES.md | Section 8 |

---

## Common Commands

```bash
# View annotation progress
head -20 /Users/jack/projects/moola/data/corrections/candlesticks_annotations/master_index.csv

# Count annotated windows
wc -l /Users/jack/projects/moola/data/corrections/candlesticks_annotations/master_index.csv

# Check for CleanLab issues
cat /Users/jack/projects/moola/data/corrections/cleanlab_label_issues.csv | wc -l

# Inspect training data
python3 -c "import pandas as pd; df = pd.read_parquet('/Users/jack/projects/moola/data/processed/train_pivot_134.parquet'); print(f'Samples: {len(df)}, Labels: {df.label.value_counts().to_dict()}')"

# Check master index schema
head -1 /Users/jack/projects/moola/data/corrections/candlesticks_annotations/master_index.csv
```

---

## Related Documentation

### Existing Moola Docs
- **data_infrastructure.md** - Data infrastructure framework
- **ARCHITECTURE.md** - System design
- **GETTING_STARTED.md** - Quick start guide

### Candlesticks Docs
- **candlesticks/README.md** - Annotation interface guide
- **candlesticks/backend/config.py** - Configuration details

### Implementation Files
- **src/moola/data_infra/schemas.py** - Pydantic data schemas
- **src/moola/data/load.py** - Data loading utilities
- **src/moola/data/dual_input_pipeline.py** - Feature extraction pipeline

---

## Next Steps (Action Items)

### Immediate (Before next training)
1. Verify data leakage using timestamp check (URGENT)
   - Script location: See MOOLA_DATA_PIPELINE_ANALYSIS.md Section 6

### Short-term (This sprint)
2. Create rejections.json with explicit rejection tracking
   - Template: See MOOLA_DATA_PIPELINE_ANALYSIS.md Section 8

3. Integrate rejection filtering in training pipeline
   - Code: See MOOLA_DATA_PIPELINE_ANALYSIS.md Section 8

### Medium-term (Documentation)
4. Document timestamp ranges for all datasets
5. Update data_infrastructure.md with Candlesticks details

---

## Document Versions

| Document | Date | Status | Sections |
|----------|------|--------|----------|
| MOOLA_DATA_PIPELINE_ANALYSIS.md | 2025-10-18 | Final | 10 |
| CANDLESTICKS_DATA_REFERENCE.md | 2025-10-18 | Final | 8 |
| DATA_PIPELINE_INDEX.md | 2025-10-18 | Final | This file |

---

## Contact / Questions

For questions about:
- **Data organization**: See MOOLA_DATA_PIPELINE_ANALYSIS.md Sections 1-5
- **Candlesticks integration**: See MOOLA_DATA_PIPELINE_ANALYSIS.md Section 3
- **Rejection tracking**: See MOOLA_DATA_PIPELINE_ANALYSIS.md Section 8
- **Quick reference**: See CANDLESTICKS_DATA_REFERENCE.md

---

**Last Updated:** 2025-10-18  
**Status:** Complete and ready for team review
