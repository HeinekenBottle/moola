# Experiments Quick Start

## TL;DR - Copy-Paste Commands

### Prerequisites (If baseline checkpoint missing)

```bash
# Train baseline model (100 epochs, ~2 hours on GPU)
python3 scripts/train_baseline_100ep.py \
    --data data/processed/labeled/train_latest_overlaps_v2.parquet \
    --output artifacts/baseline_100ep/ \
    --epochs 100 \
    --device cuda
```

### Run Both Experiments (Master Script)

```bash
# Automated execution (~30 min total)
bash scripts/run_parallel_experiments.sh
```

### Run Individual Experiments

#### Experiment A: Threshold Grid Search (5 min)

```bash
python3 scripts/experiment_a_threshold_grid.py \
    --checkpoint artifacts/baseline_100ep/best_model.pt \
    --data data/processed/labeled/train_latest_overlaps_v2.parquet \
    --output results/threshold_grid.csv \
    --min-threshold 0.30 \
    --max-threshold 0.40 \
    --step 0.02 \
    --device cuda
```

**Expected output:**
```
Threshold    F1       Precision    Recall     Target Met
0.30         0.2150   0.1500       0.4200     ✓ YES
0.32         0.2380   0.1680       0.4050     ✓ YES  <- RECOMMENDED
0.34         0.2290   0.1850       0.3700       no
```

#### Experiment B: Data Augmentation (20 epochs, 25 min)

```bash
python3 scripts/experiment_b_augmentation.py \
    --data data/processed/labeled/train_latest_overlaps_v2.parquet \
    --output artifacts/augmentation_exp/ \
    --epochs 20 \
    --n-augment 2 \
    --sigma 0.03 \
    --pos-weight 13.1 \
    --device cuda
```

**Expected output:**
```
Epoch 20/20: train_loss=0.4523, val_loss=0.5678, F1=0.252, P=0.185, R=0.415
✅ TARGET MET: F1 >= 0.25
```

## Quick Analysis Commands

### Experiment A: Find Best Threshold

```bash
# View all results
cat results/threshold_summary.txt

# Or parse CSV for best F1
python3 << 'EOF'
import pandas as pd
df = pd.read_csv("results/threshold_grid.csv")
best = df.loc[df["f1"].idxmax()]
print(f"Best threshold: {best['threshold']:.2f}")
print(f"F1: {best['f1']:.4f}, Precision: {best['precision']:.4f}, Recall: {best['recall']:.4f}")
EOF
```

### Experiment B: Plot Training Curves

```bash
# View training curves (automatically generated)
open artifacts/augmentation_exp/training_curves.png

# Check if target met
python3 << 'EOF'
import pandas as pd
df = pd.read_csv("artifacts/augmentation_exp/training_history.csv")
best_f1 = df["span_f1"].max()
final_f1 = df["span_f1"].iloc[-1]
print(f"Best F1: {best_f1:.4f} (epoch {df['span_f1'].idxmax() + 1})")
print(f"Final F1: {final_f1:.4f} (epoch {len(df)})")
print(f"Target (F1 >= 0.25): {'✅ MET' if best_f1 >= 0.25 else '❌ NOT MET'}")
EOF
```

## Output Files Checklist

After running experiments, verify these files exist:

### Experiment A
- [ ] `results/threshold_grid.csv` - Full results table
- [ ] `results/threshold_summary.txt` - Best threshold recommendation

### Experiment B
- [ ] `artifacts/augmentation_exp/training_history.csv` - Metrics per epoch
- [ ] `artifacts/augmentation_exp/training_curves.png` - Visualization
- [ ] `artifacts/augmentation_exp/best_model.pt` - Best checkpoint
- [ ] `artifacts/augmentation_exp/metadata.json` - Experiment config

## Success Criteria

| Experiment | Target | Success Indicator |
|------------|--------|-------------------|
| **A: Threshold** | F1 > 0.23, Recall >= 0.40 | ✅ Found threshold in [0.30, 0.40] meeting criteria |
| **B: Augmentation** | F1 >= 0.25 | ✅ Best F1 >= 0.25 within 20 epochs |

## Next Steps After Experiments

### If Experiment A succeeds:
1. Note recommended threshold from `threshold_summary.txt`
2. Update inference code to use optimal threshold
3. Re-run validation to confirm improvement

### If Experiment B succeeds:
1. Deploy augmentation to production training pipeline
2. Test higher augmentation factors (4x, 5x)
3. Combine with optimal threshold for maximum gain

### If both succeed:
1. **Training:** Use 3x augmentation (σ=0.03)
2. **Inference:** Use optimal threshold from Exp A
3. **Expected combined gain:** 0.22 → 0.28-0.30 F1 (+27-36%)

## Troubleshooting Quick Fixes

### GPU out of memory
```bash
# Reduce batch size for Experiment B
python3 scripts/experiment_b_augmentation.py --batch-size 16  # Default 32
```

### Missing checkpoint
```bash
# Train baseline first (see Prerequisites section above)
```

### F1 worse than baseline
```bash
# Try lower jitter strength
python3 scripts/experiment_b_augmentation.py --sigma 0.01  # Default 0.03
```

## Customization Options

### Experiment A: Extended Threshold Range

```bash
# Test wider range with finer granularity
python3 scripts/experiment_a_threshold_grid.py \
    --min-threshold 0.25 \
    --max-threshold 0.50 \
    --step 0.01  # Finer steps
```

### Experiment B: More Aggressive Augmentation

```bash
# 5x data augmentation (4 copies per sample)
python3 scripts/experiment_b_augmentation.py \
    --n-augment 4 \
    --sigma 0.05  # Slightly stronger jitter
```

### Experiment B: Longer Training

```bash
# 50 epochs for full convergence
python3 scripts/experiment_b_augmentation.py \
    --epochs 50
```

## Time Estimates

| Task | GPU | CPU (not recommended) |
|------|-----|-----------------------|
| Baseline training (100 ep) | 2 hours | 12-16 hours |
| Experiment A (threshold) | 5 min | 15 min |
| Experiment B (20 ep, 3x aug) | 25 min | 2-3 hours |
| **Total (both experiments)** | **30 min** | **2.5-3.5 hours** |

**Hardware:** RTX 4090 GPU (24GB VRAM)

## Questions?

See full documentation:
- `PARALLEL_EXPERIMENTS_README.md` - Detailed experiment design and rationale
- `CLAUDE.md` - Project context and architecture
