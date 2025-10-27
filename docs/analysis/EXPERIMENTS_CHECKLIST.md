# Experiments Pre-Flight Checklist

## Before Running Experiments

### Environment Setup
- [ ] Python 3.10+ installed
- [ ] PyTorch with CUDA support installed
- [ ] GPU available (8GB+ VRAM recommended)
- [ ] All dependencies installed (`pip3 install -r requirements.txt`)

### Data Preparation
- [ ] Labeled data exists: `data/processed/labeled/train_latest_overlaps_v2.parquet`
- [ ] Data has 210 samples (or close to it)
- [ ] Data contains required columns: `features`, `label`, `expansion_start`, `expansion_end`

### Baseline Model (Required for Experiment A)
- [ ] Baseline training completed (100 epochs)
- [ ] Checkpoint exists: `artifacts/baseline_100ep/best_model.pt`
- [ ] Checkpoint contains `model_state_dict` key

**If baseline missing, run:**
```bash
python3 scripts/train_baseline_100ep.py \
    --data data/processed/labeled/train_latest_overlaps_v2.parquet \
    --output artifacts/baseline_100ep/ \
    --epochs 100 \
    --device cuda
```

### Script Verification
- [ ] `scripts/experiment_a_threshold_grid.py` exists
- [ ] `scripts/experiment_b_augmentation.py` exists
- [ ] `scripts/run_parallel_experiments.sh` exists and is executable
- [ ] `scripts/analyze_experiment_results.py` exists

## During Experiments

### Experiment A (5 min)
- [ ] Script starts without errors
- [ ] Model checkpoint loads successfully
- [ ] Validation data loads (should show ~42 samples)
- [ ] Thresholds tested: 0.30, 0.32, 0.34, 0.36, 0.38, 0.40
- [ ] Progress printed for each threshold
- [ ] Script completes in < 10 minutes

### Experiment B (20-25 min)
- [ ] Script starts without errors
- [ ] Dataset shows correct augmentation (e.g., "210 original → 630 total")
- [ ] Train/val split shows augmented train, original val
- [ ] Model created (should show ~52K parameters)
- [ ] Epoch progress prints every epoch
- [ ] No CUDA out of memory errors
- [ ] Script completes in < 30 minutes

## After Experiments

### Output Files Generated
- [ ] `results/threshold_grid.csv` (Experiment A)
- [ ] `results/threshold_summary.txt` (Experiment A)
- [ ] `artifacts/augmentation_exp/training_history.csv` (Experiment B)
- [ ] `artifacts/augmentation_exp/training_curves.png` (Experiment B)
- [ ] `artifacts/augmentation_exp/best_model.pt` (Experiment B)
- [ ] `artifacts/augmentation_exp/metadata.json` (Experiment B)

### Results Validation

**Experiment A:**
- [ ] CSV contains 6 rows (one per threshold)
- [ ] F1 scores are reasonable (0.15-0.25 range)
- [ ] At least one threshold meets target (F1 > 0.23, recall >= 0.40)
- [ ] Summary file contains recommended threshold

**Experiment B:**
- [ ] Training history has 20 rows (one per epoch)
- [ ] F1 scores increase over training (no immediate collapse)
- [ ] Best F1 >= 0.20 (minimum sanity check)
- [ ] Training curves show smooth progression (no wild oscillations)

### Quality Checks

- [ ] Experiment A: Best threshold is in reasonable range (0.30-0.40)
- [ ] Experiment A: Recall >= 0.35 for recommended threshold
- [ ] Experiment B: Final F1 is not worse than baseline (0.22)
- [ ] Experiment B: No overfitting (val loss stable or decreasing)
- [ ] Experiment B: Training completed all 20 epochs

## Analysis

### Run Analysis Script
```bash
python3 scripts/analyze_experiment_results.py \
    --threshold-csv results/threshold_grid.csv \
    --augmentation-dir artifacts/augmentation_exp/
```

- [ ] Analysis script runs without errors
- [ ] Experiment A section shows best threshold
- [ ] Experiment B section shows F1 progression
- [ ] Comparison section estimates combined improvement
- [ ] Recommendations are printed

### Interpret Results

**Experiment A Success Criteria:**
- [ ] ✅ Found threshold with F1 > 0.23
- [ ] ✅ Recall >= 0.40 at recommended threshold
- [ ] ✅ Improvement over baseline threshold 0.5

**Experiment B Success Criteria:**
- [ ] ✅ Best F1 >= 0.25 within 20 epochs
- [ ] ✅ F1 improved vs baseline (0.22)
- [ ] ✅ No overfitting (val loss trend reasonable)

## Next Steps

### If Experiment A Succeeds
- [ ] Note recommended threshold from analysis
- [ ] Update inference code to use optimal threshold
- [ ] Document threshold choice in deployment guide
- [ ] Re-run validation with new threshold to confirm

### If Experiment B Succeeds
- [ ] Document augmentation parameters (n_augment=2, sigma=0.03)
- [ ] Consider testing higher augmentation (4x, 5x)
- [ ] Update training pipeline to include augmentation
- [ ] Re-train production model with augmentation

### If Both Succeed
- [ ] Calculate combined improvement estimate
- [ ] Deploy augmentation to training pipeline
- [ ] Deploy optimal threshold to inference pipeline
- [ ] Run full validation to confirm combined gains
- [ ] Update production documentation with new configuration

## Troubleshooting

### Common Issues

**"Checkpoint not found"**
- Run baseline training first (see Baseline Model section)

**"CUDA out of memory"**
- Reduce batch size: Add `--batch-size 16` to Experiment B

**"F1 scores are all near zero"**
- Check data format (features should be 13D)
- Verify labels are binary (0/1)
- Check if model checkpoint loaded correctly

**"Training diverges (loss increases)"**
- Reduce learning rate: Add `--lr 5e-4` to Experiment B
- Check for data corruption in augmentation

**"Augmentation doesn't improve F1"**
- This is a valid result (jitter doesn't help)
- Try lower sigma: `--sigma 0.01`
- Document findings and focus on threshold optimization

## Emergency Recovery

### If experiments crash midway:

**Experiment A:**
- Safe to restart (no checkpointing, runs in 5 min)
- Just re-run the command

**Experiment B:**
- Check if partial training history exists
- If epoch 10+ complete, can analyze partial results
- Otherwise, restart from scratch (no incremental checkpointing)

### If results look wrong:

1. Verify data integrity:
```bash
python3 << 'EOF'
import pandas as pd
df = pd.read_parquet("data/processed/labeled/train_latest_overlaps_v2.parquet")
print(f"Samples: {len(df)}")
print(f"Features shape: {df.iloc[0]['features'][0].shape}")
print(f"Label values: {df['label'].unique()}")
EOF
```

2. Verify model checkpoint:
```bash
python3 << 'EOF'
import torch
ckpt = torch.load("artifacts/baseline_100ep/best_model.pt", map_location="cpu")
print(f"Keys: {ckpt.keys()}")
print(f"Epoch: {ckpt.get('epoch', 'unknown')}")
EOF
```

3. Re-run with debug flags:
```bash
# Experiment B with reduced epochs to test
python3 scripts/experiment_b_augmentation.py --epochs 5 --device cpu
```

## Documentation

All documentation files are in `/Users/jack/projects/moola/`:
- `EXPERIMENTS_SUMMARY.txt` - Quick reference
- `EXPERIMENTS_QUICKSTART.md` - Copy-paste commands
- `PARALLEL_EXPERIMENTS_README.md` - Full design and rationale
- `EXPERIMENTS_CHECKLIST.md` - This file

## Contact

If issues persist:
1. Check `PARALLEL_EXPERIMENTS_README.md` for detailed troubleshooting
2. Review recent git commits for any breaking changes
3. Verify PYTHONPATH includes `src/` directory
