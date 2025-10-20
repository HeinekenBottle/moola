# Candlesticks Data Pipeline - Quick Reference

## File Locations (One-Liners)

```bash
# View annotation progress
head -20 /Users/jack/projects/moola/data/corrections/candlesticks_annotations/master_index.csv

# Count annotated windows
wc -l /Users/jack/projects/moola/data/corrections/candlesticks_annotations/master_index.csv

# View CleanLab flagged samples
cat /Users/jack/projects/moola/data/corrections/cleanlab_label_issues.csv | head -20

# View review actions
cat /Users/jack/projects/moola/data/corrections/cleanlab_reviewed.json | jq '.reviewed_windows[]'

# Inspect training data
python3 -c "import pandas as pd; df = pd.read_parquet('/Users/jack/projects/moola/data/processed/train_pivot_134.parquet'); print(f'Samples: {len(df)}, Labels: {df.label.value_counts().to_dict()}')"

# Inspect unlabeled windows
python3 -c "import pandas as pd; df = pd.read_parquet('/Users/jack/projects/moola/data/raw/unlabeled_windows.parquet'); print(f'Samples: {len(df)}')"
```

## Data Locations

| Name | Path | Format | Size | Samples |
|------|------|--------|------|---------|
| Raw OHLC (NQ) | candlesticks/data/raw/nq_1min_raw.parquet | Parquet | 2.2M | 118k bars |
| Unlabeled Windows | data/raw/unlabeled_windows.parquet | Parquet | 2.2M | 11,873 |
| Training Data | data/processed/train_pivot_134.parquet | Parquet | 94K | 105 |
| Annotation Batches | data/corrections/candlesticks_annotations/batch_*.json | JSON | ~2K each | 15 files |
| Master Index | data/corrections/candlesticks_annotations/master_index.csv | CSV | 5K | 15 rows |
| CleanLab Issues | data/corrections/cleanlab_label_issues.csv | CSV | 14K | ~50 samples |
| CleanLab Reviews | data/corrections/cleanlab_reviewed.json | JSON | 2K | 15 samples |

## Window Structure (105 bars)

```
Index Range    Purpose                 Bars   Notes
───────────────────────────────────────────────────────
[0-29]        Past Context              30    Historical data
[30-74]       Prediction Window         45    WHERE patterns occur
[75-104]      Future Outcome            30    What actually happened
───────────────────────────────────────────────────────
Total                                  105    Full window
```

Key: `expansion_start/end` are indices within the [30-74] range.

## Label Distribution (Current train_pivot_134.parquet)

```
consolidation: 60 (57.1%)
retracement:   45 (42.9%)
───────────────────────
Total:        105 samples
```

## Expansion Index Ranges

Valid range: [30, 74]  
Example: expansion_start=40, expansion_end=50 = 10-bar pattern

## Data Flow

```
Raw OHLC
  └─→ Unlabeled Windows (11,873)
      ├─→ Pre-training (BiLSTM)
      └─→ NOT USED in fine-tuning
      
  └─→ Candlesticks Annotation
      ├─→ Load window by ID
      ├─→ User marks pattern
      ├─→ Save to batch_*.json
      ├─→ Update master_index.csv
      └─→ Result: 105 labeled samples
      
  └─→ CleanLab Quality Review
      ├─→ Check label quality
      ├─→ Flag suspicious samples
      └─→ Result: cleanlab_label_issues.csv
      
  └─→ Training Pipeline
      └─→ Load train_pivot_134.parquet
          ├─→ Extract expansion region
          ├─→ Engineer features
          └─→ Train model
```

## Known Issues & Status

### Annotation Quality
- Some windows marked "D" grade (low quality)
- Some windows have 0 expansions (empty)
- CleanLab flagged ~50 samples with label uncertainty

### Potential Leakage
- **Unknown:** Do unlabeled_windows overlap with training data time-wise?
- **Action:** Run timestamp validation before production

### Missing Infrastructure
- No explicit "REJECTED" marking system
- Rejection is implicit (high D-grade, empty, CleanLab flagged)
- Recommendation: Create `rejections.json` (see analysis doc)

## Quick Checks

### Check if window was annotated
```python
import pandas as pd
master_idx = pd.read_csv('data/corrections/candlesticks_annotations/master_index.csv')
window_id = 0
if window_id in master_idx['window_id'].values:
    print(f"Window {window_id} was annotated")
else:
    print(f"Window {window_id} needs annotation")
```

### Count annotations by quality
```python
import pandas as pd
master_idx = pd.read_csv('data/corrections/candlesticks_annotations/master_index.csv')
print(master_idx['quality_grade'].value_counts())
```

### Find CleanLab flagged samples
```python
import pandas as pd
issues = pd.read_csv('data/corrections/cleanlab_label_issues.csv')
flagged = issues[issues['is_label_issue'] == True]
print(f"Flagged samples: {len(flagged)}")
print(flagged[['window_id', 'given_label_name', 'pred_label_name', 'label_quality']])
```

### Verify no data leakage
```python
import pandas as pd
import numpy as np

# Load both datasets
unlabeled = pd.read_parquet('data/raw/unlabeled_windows.parquet')
labeled = pd.read_parquet('data/processed/train_pivot_134.parquet')

# Extract window IDs
unlabeled_ids = set(unlabeled['window_id'].unique())
labeled_ids = set(int(w.split('_')[0]) for w in labeled['window_id'].unique())

# Check overlap
overlap = unlabeled_ids.intersection(labeled_ids)
print(f"Unlabeled windows: {len(unlabeled_ids)}")
print(f"Labeled windows: {len(labeled_ids)}")
print(f"Overlap: {len(overlap)}")
if overlap:
    print(f"WARNING: {len(overlap)} windows appear in both sets!")
else:
    print("OK: No overlap detected")
```

## Configuration

**Candlesticks Config:** `/Users/jack/projects/candlesticks/backend/config.py`

Key settings:
```python
RAW_DATA_PATH = /Users/jack/projects/moola/data/raw
CANDLESTICKS_ANNOTATIONS_DIR = moola/data/corrections/candlesticks_annotations
WINDOW_SIZE = 105  # bars
TOTAL_WINDOWS = 205  # max available
```

## Next Steps

1. Create rejection tracking system (rejections.json)
2. Verify timestamp ranges to rule out data leakage
3. Document annotation SLAs
4. Integrate rejection filtering in training pipeline

---

**Last Updated:** 2025-10-18  
**Status:** Ready for production  
**Contact:** See MOOLA_DATA_PIPELINE_ANALYSIS.md for detailed documentation
