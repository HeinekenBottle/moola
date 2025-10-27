# Parallel Experiments: Threshold Tuning + Data Augmentation

Two complementary experiments designed to improve expansion span detection F1 score through inference optimization (A) and training data diversity (B).

## Overview

| Experiment | Goal | Runtime | Expected Outcome |
|------------|------|---------|------------------|
| **A: Threshold Precision Tuning** | Find optimal threshold for span detection | 5 min | F1 > 0.23, Recall >= 0.40 |
| **B: Data Augmentation Strategy** | Test if jitter improves generalization | 20-25 min | F1 >= 0.25 via diversity |

**Total runtime:** ~30 minutes on RTX 4090 GPU

## Context

**Current baseline:** F1 ~0.22 at threshold 0.5 (from 100-epoch baseline training)

**Problem:**
- Soft span predictions may not align optimally at threshold 0.5
- Small dataset (210 samples) limits diversity
- Class imbalance (7.1% in-span) requires careful tuning

**Solution:**
- **Experiment A:** Grid search thresholds 0.30-0.40 to find precision/recall sweet spot
- **Experiment B:** Augment training data 3x via Gaussian jitter (σ=0.03)

## Experiment A: Threshold Precision Tuning

### Objective
Find the threshold that maximizes F1 while maintaining acceptable recall (>= 0.40).

### Method
1. Load trained baseline model checkpoint
2. Grid search thresholds: 0.30, 0.32, 0.34, 0.36, 0.38, 0.40
3. Compute F1, precision, recall at each threshold on validation set
4. Identify optimal operating point

### Expected Behavior
- **Lower thresholds (0.30-0.35):** Higher recall, lower precision
- **Higher thresholds (0.38-0.40):** Higher precision, lower recall
- **Optimal:** Balanced F1 > 0.23 without recall collapse

### Target Metrics
- ✅ **F1 Score:** > 0.23
- ✅ **Recall:** >= 0.40 (don't miss real expansions)
- ✅ **Precision:** Maximize while meeting F1/recall targets

### Usage

```bash
# Requires pre-trained baseline checkpoint
python3 scripts/experiment_a_threshold_grid.py \
    --checkpoint artifacts/baseline_100ep/best_model.pt \
    --data data/processed/labeled/train_latest_overlaps_v2.parquet \
    --output results/threshold_grid.csv \
    --min-threshold 0.30 \
    --max-threshold 0.40 \
    --step 0.02 \
    --device cuda
```

### Output Files
- `results/threshold_grid.csv` - Full results for all thresholds
- `results/threshold_summary.txt` - Best threshold recommendation

### Interpretation

**Example output:**
```
Threshold    F1       Precision    Recall     Target Met
0.30         0.2150   0.1500       0.4200     ✓ YES
0.32         0.2380   0.1680       0.4050     ✓ YES  <- OPTIMAL
0.34         0.2290   0.1850       0.3700       no
0.36         0.2050   0.1980       0.3400       no
```

**Recommendation:** Use threshold 0.32 (F1=0.238, Recall=0.405)

## Experiment B: Data Augmentation Strategy

### Objective
Test if training data diversity via Gaussian jitter improves generalization and F1 score.

### Method
1. **Augmentation:** Apply Gaussian noise (σ=0.03) to 13D features
2. **Expansion:** 210 original samples → 630 total (3x via 2 augmented copies per sample)
3. **Training:** 20 epochs with uncertainty-weighted multi-task loss
4. **Validation:** Original samples only (no augmentation leakage)

### Rationale
- **Small dataset problem:** 210 samples insufficient for high-capacity model (52K params)
- **Regularization via diversity:** Jitter forces model to learn robust features
- **Prior work:** Synth PDF (dropout=0.65 prevents overfit) suggests diversity helps
- **Target:** F1 0.25+ if augmentation provides meaningful signal

### Configuration
- **Jitter σ = 0.03:** 3% of normalized feature range (mild, preserves structure)
- **Augmentation:** 2 copies per sample (3x total data)
- **Positive weight:** 13.1 (compensates for 7.1% in-span class imbalance)
- **Dropout:** 0.7 recurrent, 0.6 dense, 0.3 input (prevents memorization)
- **Epochs:** 20 (sufficient to see convergence trends)

### Usage

```bash
python3 scripts/experiment_b_augmentation.py \
    --data data/processed/labeled/train_latest_overlaps_v2.parquet \
    --output artifacts/augmentation_exp/ \
    --epochs 20 \
    --n-augment 2 \
    --sigma 0.03 \
    --pos-weight 13.1 \
    --batch-size 32 \
    --lr 1e-3 \
    --device cuda
```

### Output Files
- `artifacts/augmentation_exp/training_history.csv` - Epoch-by-epoch metrics
- `artifacts/augmentation_exp/training_curves.png` - Loss and F1 curves
- `artifacts/augmentation_exp/best_model.pt` - Best checkpoint by F1
- `artifacts/augmentation_exp/metadata.json` - Experiment config

### Interpretation

**Success criteria:**
- ✅ **F1 >= 0.25** at epoch 20
- ✅ **No overfitting:** Val loss decreases or stabilizes
- ✅ **Recall >= 0.40** maintained

**Failure modes:**
- ❌ **F1 < 0.22:** Augmentation degraded signal (jitter too strong)
- ❌ **F1 plateaus early:** Augmentation insufficient, need more diversity
- ❌ **Overfitting:** Val loss increases while train loss decreases

**Next steps if successful (F1 >= 0.25):**
1. Deploy augmentation to production training pipeline
2. Test higher augmentation (4x, 5x) for further gains
3. Combine with optimal threshold from Experiment A

## Running Both Experiments

Use the master script for automated execution:

```bash
bash scripts/run_parallel_experiments.sh
```

This will:
1. Verify baseline checkpoint exists
2. Run Experiment A (threshold tuning)
3. Run Experiment B (augmentation)
4. Save all results to timestamped directory

### Prerequisites

**Required:**
- Baseline checkpoint: `artifacts/baseline_100ep/best_model.pt`
- Labeled data: `data/processed/labeled/train_latest_overlaps_v2.parquet`
- GPU with 8GB+ VRAM (RTX 4090 recommended)

**If baseline checkpoint missing:**
```bash
python3 scripts/train_baseline_100ep.py \
    --data data/processed/labeled/train_latest_overlaps_v2.parquet \
    --output artifacts/baseline_100ep/ \
    --epochs 100 \
    --device cuda
```

## Expected Results Summary

### Experiment A: Threshold Grid Search

**Input:** Trained baseline model (F1 ~0.22 at threshold 0.5)

**Output:**
- CSV with F1/precision/recall for thresholds 0.30-0.40
- Recommended threshold meeting target (F1 > 0.23, recall >= 0.40)

**Expected best threshold:** 0.32-0.36 range
- Lower threshold → higher recall (detect more spans)
- May sacrifice precision slightly but worth it for recall target

### Experiment B: Data Augmentation (20 Epochs)

**Input:** 210 samples → 630 augmented (3x expansion)

**Expected F1 progression:**
- **Epoch 1-5:** Rapid improvement (0.10 → 0.18)
- **Epoch 6-12:** Gradual improvement (0.18 → 0.23)
- **Epoch 13-20:** Convergence or slight gains (0.23 → 0.25+)

**Success indicator:** F1 >= 0.25 by epoch 20

**Comparison to baseline:**
| Method | F1 | Precision | Recall | Conclusion |
|--------|----|-----------| -------|------------|
| Baseline (100 ep, no aug) | 0.22 | 0.XX | 0.XX | Baseline |
| Augmentation (20 ep, 3x data) | **0.25+** | 0.XX | 0.XX | **+14% improvement** |

## Design Rationale

### Why these specific parameters?

**pos_weight = 13.1:**
- Training data has 7.1% in-span timesteps (class imbalance)
- pos_weight = 1 / 0.071 ≈ 14.1
- Conservative 13.1 to avoid over-weighting minority class

**σ = 0.03 (jitter):**
- Mild noise (3% of normalized range)
- Strong enough to provide diversity without destroying signal
- Features are normalized to [-1, 1] → σ=0.03 preserves structure

**n_augment = 2 (3x total data):**
- Moderate expansion to test concept
- If successful, can increase to 4x, 5x
- Avoids excessive training time for initial test

**Epochs = 20 (not 100):**
- Sufficient to see convergence trends
- Augmentation should show gains early (10-15 epochs)
- Saves GPU time vs full 100-epoch run

### Why position_encoding matters

Model input is 13 features (not 12):
1. `open_norm` - Normalized open price
2. `close_norm` - Normalized close price
3. `body_pct` - Body size as % of range
4. `upper_wick_pct` - Upper wick size
5. `lower_wick_pct` - Lower wick size
6. `range_z` - Z-scored range
7. `dist_to_prev_SH` - Distance to previous swing high
8. `dist_to_prev_SL` - Distance to previous swing low
9. `bars_since_SH_norm` - Bars since swing high (normalized)
10. `bars_since_SL_norm` - Bars since swing low (normalized)
11. `expansion_proxy` - Expansion phase indicator
12. `consol_proxy` - Consolidation phase indicator
13. **`position_encoding`** - Timestep position t/(K-1) for ICT structure awareness

**Why position matters:** ICT (Inner Circle Trader) concepts rely on temporal structure within 105-bar windows (e.g., Asian session → London → New York). Position encoding helps model learn these temporal patterns.

## Troubleshooting

### "Checkpoint not found"
```bash
# Train baseline first
python3 scripts/train_baseline_100ep.py \
    --data data/processed/labeled/train_latest_overlaps_v2.parquet \
    --output artifacts/baseline_100ep/ \
    --epochs 100 \
    --device cuda
```

### "CUDA out of memory"
Reduce batch size in experiment scripts:
```bash
# For Experiment B
python3 scripts/experiment_b_augmentation.py --batch-size 16  # Default 32
```

### "F1 much worse than baseline"
Possible causes:
1. **Jitter too strong:** Reduce σ from 0.03 to 0.01
2. **Augmentation leakage:** Verify validation uses original samples only
3. **pos_weight mismatch:** Recalculate based on actual class distribution

### "Experiment B doesn't improve F1"
This is valid experimental result:
- Documents that jitter augmentation doesn't help for this dataset
- Try alternative augmentations (time warping, mixup)
- Focus on threshold optimization instead

## Integration with Production

### If Experiment A succeeds (optimal threshold found):

Update inference code:
```python
# In moola/models/jade_core.py or deployment script
OPTIMAL_THRESHOLD = 0.32  # From experiment results

def predict_spans(model, features):
    output = model(features)
    pred_probs = output["expansion_binary"]
    pred_spans = (pred_probs > OPTIMAL_THRESHOLD).float()  # Use optimal threshold
    return pred_spans
```

### If Experiment B succeeds (F1 >= 0.25):

Add augmentation to training pipeline:
```python
# In training scripts
from moola.data.augmentation import jitter_features  # Create this module

dataset = AugmentedExpansionDataset(
    data_path="data/processed/labeled/train_latest_overlaps_v2.parquet",
    n_augment=2,  # 3x total data
    sigma=0.03,   # Validated jitter strength
)
```

### Combined strategy (both succeed):

1. **Training:** Use augmentation (3x data)
2. **Inference:** Use optimal threshold from Experiment A
3. **Expected boost:** Baseline 0.22 → 0.28-0.30 F1 (combined gains)

## References

**Uncertainty weighting:**
- Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics", CVPR 2018

**Data augmentation for time series:**
- Um et al., "Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring using Convolutional Neural Networks", ACM ICMI 2017

**Class imbalance with positive weights:**
- Huang et al., "Learning Deep Representation for Imbalanced Classification", CVPR 2016

## Questions?

See project documentation:
- `CLAUDE.md` - Project context for AI assistants
- `docs/ARCHITECTURE.md` - System design details
- `WORKFLOW_SSH_SCP_GUIDE.md` - RunPod GPU training workflow
