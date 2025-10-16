# Phase 2 Iteration 2: RunPod GPU Training Results

**Date:** 2025-10-16
**Execution:** RunPod RTX 4090 (SSH: root@213.173.102.99:27424)

## Summary

Completed Steps 3-5 of the ML pipeline with GPU-accelerated training on RunPod. Step 4 (TS-TCC pre-training) was skipped due to CLI constraints, and Step 5 used original 98-sample data instead of CleanLab-cleaned 89-sample data due to fold split incompatibility.

## Execution Timeline

### ✅ Step 3: Extract Unlabeled Windows
- **Status:** Already completed
- **Data:** `data/raw/unlabeled_windows.parquet`
- **Samples:** 11,873 unlabeled windows (105 bars each)
- **Size:** 2.3 MB

### ⚠️ Step 4: TS-TCC Pre-training
- **Status:** SKIPPED
- **Reason:** `moola.cli pretrain-tcc` command:
  - Requires `configs/` directory with Hydra config files
  - Loads data from `paths.data / "processed" / "train.parquet"` (hardcoded)
  - No `--unlabeled` parameter to specify custom data path
- **Impact:** Models trained without pre-trained encoder weights
- **Workaround Required:** Modify CLI to accept unlabeled data path or temporarily replace train.parquet

### ✅ Step 5: Train with Augmentation
- **Status:** COMPLETED
- **Training Data:** Original 98 samples (`data/processed/train.parquet`)
- **Note:** CleanLab-cleaned 89-sample data NOT used due to incompatible fold splits
- **Augmentation:**
  - SMOTE oversampling (per-fold, ~78→88-90 samples)
  - Mixup α=0.4 (increased from 0.2)
  - Temporal augmentation enabled
  - Batch size: 1024 (increased from 512)
  - Early stopping patience: 20 epochs

## Results

### Model Performance (Augmented Training on 98 samples)

| Model | Accuracy | AUC | Notes |
|-------|----------|-----|-------|
| logreg | 48.98% | 0.4966 | Convergence warnings (lbfgs) |
| rf | 50.00% | 0.4736 | Random forest baseline |
| xgb | 54.08% | 0.5442 | Best performer, SMOTE balanced |
| simple_lstm | 53.06% | 0.4881 | Early stopping epoch ~23-36 |
| cnn_transformer | 48.98% | 0.5001 | Early stopping epoch ~21-41 |

### Comparison: Augmented vs Clean Baseline

| Model | Augmented | Clean (Step 1) | Change |
|-------|-----------|----------------|--------|
| logreg | 48.98% | 56.1% | -7.12% ❌ |
| rf | 50.00% | 46.9% | +3.10% ✅ |
| xgb | 54.08% | 55.1% | -1.02% ≈ |
| simple_lstm | 53.06% | 57.1% | -4.04% ❌ |
| cnn_transformer | 48.98% | 46.9% | +2.08% ✅ |

**Observation:** Augmentation did not consistently improve performance. Some models degraded significantly (logreg -7.1%, simple_lstm -4.0%), suggesting:
1. The dataset may be too small for aggressive augmentation
2. SMOTE may introduce noise in this time-series context
3. Original data distribution may be more representative

## Technical Details

### GPU Training Configuration
- **GPU:** NVIDIA GeForce RTX 4090
- **Memory Usage:** 0.02-0.03 GB (very light)
- **Mixed Precision:** FP16 enabled
- **Device:** CUDA

### Data Pipeline
1. **Original Data:** 98 samples (56 consolidation, 42 retracement)
2. **CleanLab Cleaned:** 89 samples (removed 9 noisy labels)
3. **SMOTE Augmentation:** Per-fold resampling to ~88-90 samples
4. **Unlabeled Windows:** 11,873 samples available but unused

### Files Generated
```
data/oof/
├── logreg_augmented.npy          # 1.7 KB, shape=(98, 2)
├── rf_augmented.npy              # 1.7 KB, shape=(98, 2)
├── xgb_augmented.npy             # 1.7 KB, shape=(98, 2)
├── simple_lstm_augmented.npy     # 1.7 KB, shape=(98, 2)
└── cnn_transformer_augmented.npy # 1.7 KB, shape=(98, 2)
```

## Blockers Encountered

### 1. CleanLab Cleaned Data Incompatibility
**Issue:** `IndexError: index 89 is out of bounds for axis 0 with size 89`

**Root Cause:**
- Pre-generated fold splits in `data/splits/` contain indices for 98 samples
- CleanLab removed 9 samples → 89 samples
- Split indices reference rows 0-97, but cleaned data only has 0-88

**Workaround Applied:**
- Ran Step 5 WITHOUT `--use-cleaned-data` flag
- Used original 98-sample dataset instead

**Proper Solution (requires code changes):**
1. Regenerate fold splits for 89-sample dataset
2. Store cleaned data splits separately from original splits
3. Modify `regenerate_oof_phase2.py` to auto-detect and regenerate splits when using cleaned data

### 2. TS-TCC Pre-training CLI Constraints
**Issue:** Cannot specify unlabeled data path

**Root Cause:**
- `moola.cli pretrain-tcc` command:
  - Loads from `paths.data / "processed" / "train.parquet"` (line 483 of cli.py)
  - No command-line parameter to override data source
  - Requires `configs/` directory with Hydra config files

**Workaround Options (not implemented):**
1. Temporarily replace `train.parquet` with unlabeled windows
2. Modify CLI to accept `--unlabeled` parameter
3. Use moola as Python library instead of CLI

**Proper Solution (requires code changes):**
Add `--unlabeled-data` parameter to `pretrain-tcc` command

## Constraints Followed

✅ **No modification of core training logic**
- Used existing `regenerate_oof_phase2.py` script as-is
- Did not modify model classes or training loops

✅ **Used existing scripts where possible**
- Leveraged RunPod training infrastructure
- Only created wrapper bash scripts for orchestration

✅ **Kept it simple**
- Skipped TS-TCC when CLI constraints encountered
- Used original data when cleaned data caused indexing issues
- Minimal intervention, maximum use of existing tooling

## Next Steps

### Option A: Fix CleanLab Integration (Recommended)
1. Create `scripts/regenerate_splits.py` to generate new fold indices for 89 samples
2. Store splits in `data/splits_cleaned/` to avoid conflicts
3. Modify `regenerate_oof_phase2.py` to accept `--splits-dir` parameter
4. Re-run Step 5 with `--use-cleaned-data --splits-dir data/splits_cleaned`

### Option B: Enable TS-TCC Pre-training
1. Add `--unlabeled-data` parameter to `moola.cli.pretrain_tcc`
2. Run: `moola pretrain-tcc --unlabeled data/raw/unlabeled_windows.parquet`
3. Use encoder: `--pretrained-encoder models/ts_tcc/pretrained_encoder.pt` in Step 5

### Option C: Proceed with Stacking
- Current augmented OOF predictions are available
- Can proceed to stacking phase with existing results
- Augmentation shows mixed results, so may not improve final ensemble

## Files Reference

### Key Scripts
- `/Users/jack/projects/moola/scripts/runpod_train.sh` - Original full pipeline
- `/Users/jack/projects/moola/scripts/runpod_steps_4_5.sh` - Failed attempt with TS-TCC
- `/Users/jack/projects/moola/scripts/runpod_step5_only.sh` - Failed with cleaned data
- `/Users/jack/projects/moola/scripts/runpod_step5_no_cleaned.sh` - ✅ Successful execution

### Data Files
- `data/processed/train_clean_phase2.parquet` - CleanLab cleaned (89 samples)
- `data/raw/unlabeled_windows.parquet` - Unlabeled windows (11,873 samples)
- `data/oof/*_augmented.npy` - Augmented OOF predictions (98 samples)
- `data/oof/*_clean.npy` - Clean baseline OOF predictions (98 samples)

### RunPod Connection
```bash
# SSH
ssh -p 27424 -i ~/.ssh/id_ed25519 root@213.173.102.99

# SCP download
scp -P 27424 -i ~/.ssh/id_ed25519 'root@213.173.102.99:/workspace/moola/data/oof/*.npy' ./data/oof/
```

## Conclusion

Successfully completed GPU training on RunPod with augmentation enabled. However, results indicate that aggressive augmentation (SMOTE + mixup α=0.4) may not be beneficial for this small time-series dataset. Consider:

1. **Using original clean baseline** (57.1% simple_lstm) as the stronger starting point
2. **Fixing CleanLab integration** to properly leverage the 89-sample cleaned dataset
3. **Skipping TS-TCC** unless willing to modify CLI (effort vs reward unclear)
4. **Proceeding to stacking** to see if ensemble methods can improve upon 57.1% baseline

The infrastructure is now in place for rapid iteration on RunPod GPU. Future experiments can build on this foundation with proper split regeneration and CLI modifications if needed.
