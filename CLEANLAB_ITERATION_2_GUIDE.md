# CleanLab Iteration 2: Verify Data Quality Improvement

## Goal
Compare model performance and data quality before/after cleaning to identify remaining issues.

## Step-by-Step Instructions

### 1. Generate Predictions on Training Set

```bash
cd /Users/jack/projects/moola

# Use stack ensemble (best model: 59% accuracy)
python -m moola.cli predict \
  --model stack \
  --input data/processed/train.parquet \
  --output data/artifacts/predictions_v2_raw.csv
```

**Expected output:**
```
✓ Generated 105 predictions
✓ Saved to data/artifacts/predictions_v2_raw.csv
```

### 2. Convert to CleanLab Format

```bash
python scripts/convert_to_cleanlab_format.py \
  data/artifacts/predictions_v2_raw.csv \
  data/artifacts/predictions_v2_cleanlab.csv
```

**Expected output:**
```
✓ Loaded 105 predictions
✓ Detected mapping: consolidation/retracement
✓ Saved 105 predictions to predictions_v2_cleanlab.csv
```

### 3. Upload to CleanLab Studio

1. Go to https://app.cleanlab.ai/
2. Open your existing project OR create new "Moola v2" project
3. Upload `data/artifacts/predictions_v2_cleanlab.csv`
4. Wait for analysis to complete (~1-2 minutes)

### 4. Compare Results

**Iteration 1 (Before Cleaning):**
- Dataset: 115 samples
- Flagged expansions: **28**
- Windows fixed: 8 (49, 57, 99, 107, 110, 114, 29, 83)

**Iteration 2 (After Cleaning) - Expected:**
- Dataset: 105 samples
- Flagged expansions: **??? (hopefully < 15)**
- Model accuracy: 59% (was ???)

### 5. Analyze Remaining Issues

**Good signs:**
- ✓ Fewer flagged expansions (< 28)
- ✓ Windows 49, 57, 110 no longer flagged (label corrections worked)
- ✓ Windows 99, 107, 114 no longer flagged (zero-width fixes worked)
- ✓ Higher average label_quality_score

**If still many flags:**
- New windows might be flagged (different error patterns)
- Model confidence still low (59% accuracy suggests model uncertainty)
- Possible class imbalance: 60 consolidation vs 45 retracement

### 6. Decision Points

**Option A: More issues found → Repeat correction cycle**
- Review new flagged windows in Candlesticks
- Correct and integrate again
- Retrain models

**Option B: Quality plateaued → Move on**
- If improvements are marginal, the dataset may be as clean as possible
- Consider collecting more data (target: 300-500 samples)
- Or proceed with current 59% accuracy for production

---

## Comparison Checklist

| Metric | Iteration 1 | Iteration 2 | Notes |
|--------|-------------|-------------|-------|
| Dataset size | 115 | 105 | -10 rows |
| Flagged expansions | 28 | ??? | Target < 15 |
| Model accuracy | ??? | 59% | Stack ensemble |
| Avg quality score | ~0.49 | ??? | Target > 0.60 |
| Zero-width corruptions | 9 | 0 | ✓ Fixed |

---

## TS-TCC Decision

**Recommendation: SKIP TS-TCC FOR NOW**

**Why:**
1. TS-TCC is self-supervised pre-training (should happen BEFORE supervised training)
2. You already trained CNN-Transformer without it
3. CNN-Transformer underperformed (42.9% < 50.5% RF)
4. 105 samples too small for deep learning to help

**When to use TS-TCC:**
- After collecting 500+ samples
- If retraining from scratch
- If ensemble accuracy plateaus above 75%

---

## Questions to Ask After Iteration 2

1. **Did flagged issues decrease?**
   - Yes → Cleaning worked, continue iterating
   - No → Dataset inherently difficult or too small

2. **Did model accuracy improve?**
   - Yes → Data quality improved
   - No → Either not enough data or wrong features

3. **Are corrections actually being used?**
   - Check: Windows 49, 57, 110, 114 should NOT be flagged
   - If they're still flagged, corrections didn't get applied properly

4. **Should we collect more data?**
   - 105 samples is very small for ML
   - Target: 300-500 samples for robust training
   - Each additional 100 samples should improve accuracy by ~5-10%

---

## Contact Points

- CleanLab Studio: https://app.cleanlab.ai/
- Moola data: `/Users/jack/projects/moola/data/`
- Predictions: `/Users/jack/projects/moola/data/artifacts/predictions_v2_cleanlab.csv`
- Original flagged list: `/Users/jack/projects/moola/data/corrections/cleanlab_studio_priority_review.csv`
