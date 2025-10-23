# Heavy Files Tracking Report
**Date:** 2025-10-21
**Project:** Moola ML Pipeline

## Summary

**Status:** ✅ All heavy artifacts properly git-ignored

No heavy files are currently tracked in git. All model checkpoints, data files, and artifacts are properly excluded via `.gitignore`.

## Verification

### Check for Tracked Heavy Files
```bash
git ls-files | grep -E "\.(pt|pth|pkl|parquet|npy)$"
# Result: (empty) ✅
```

### Largest Tracked Files
```bash
git ls-files | xargs du -h 2>/dev/null | sort -rh | head -10
```

| Size | File | Status |
|------|------|--------|
| 292K | `uv.lock` | ✅ OK (dependency lock file) |
| 92K | `src/moola/cli.py` | ✅ OK (source code) |
| 72K | `src/moola/utils/validation/pseudo_sample_generation.py` | ✅ OK (source code) |
| 72K | `src/moola/models/enhanced_simple_lstm.py` | ✅ OK (source code) |
| 44K | `src/moola/models/simple_lstm.py` | ✅ OK (source code) |
| 40K | `src/moola/utils/validation/pseudo_sample_validation.py` | ✅ OK (source code) |
| 32K | `src/moola/features/small_dataset_features.py` | ✅ OK (source code) |
| 32K | `src/moola/features/price_action_features.py` | ✅ OK (source code) |
| 32K | `src/moola/data/dual_input_pipeline.py` | ✅ OK (source code) |
| 28K | `src/moola/runpod/scp_orchestrator.py` | ✅ OK (source code) |

**All tracked files are source code or configuration files under 300K.** ✅

## Heavy Files Present (Git-Ignored)

The following heavy files exist in the repository but are properly git-ignored:

### Model Checkpoints (artifacts/)
- `*.pt` files: 9 encoder files (100K-5MB each)
- `*.pkl` files: 20+ model checkpoints (100K-10MB each)
- Location: `artifacts/encoders/`, `artifacts/models/`, `artifacts/pretrained/`
- Status: ✅ Git-ignored via `.gitignore` line 109: `artifacts/`

### Data Files (data/)
- `*.parquet` files: Training/validation data
- `*.npy` files: OOF predictions, feature arrays
- Location: `data/processed/`, `data/oof/`, `data/batches/`
- Status: ✅ Git-ignored via `.gitignore` line 10: `data/**/*`

### Log Files (logs/)
- `*.log` files: Training logs
- Location: `logs/`
- Status: ✅ Git-ignored via `.gitignore` line 25: `logs`

## .gitignore Coverage

The `.gitignore` file properly excludes:

```gitignore
# Heavy artifacts
*.pkl          # Model checkpoints
*.pt           # PyTorch models
*.pth          # PyTorch models
*.h5           # Keras models
*.ckpt         # TensorFlow checkpoints
*.joblib       # Scikit-learn models

# Data files
data/**/*      # All data files
*.parquet      # (redundant, but explicit)
*.npy          # (redundant, but explicit)

# Artifacts
artifacts/     # All artifacts directory

# Logs
logs           # All log files
*.log          # (redundant, but explicit)
```

## Actions Taken

**None required.** All heavy files are already properly git-ignored.

## Recommendations

1. ✅ **Continue using SCP for artifacts:** Transfer model checkpoints and results via SCP (per CLAUDE.md workflow)
2. ✅ **Keep data/ git-ignored:** Never commit data files to git
3. ✅ **Use experiment_results.jsonl:** Log metrics to JSON lines file (git-ignored), not database

## Conclusion

**Status:** ✅ PASS

The repository is clean and lightweight. All heavy artifacts (models, data, logs) are properly git-ignored. No action required.

**Largest tracked file:** 292K (uv.lock) - well within acceptable limits.
**Heavy files untracked:** 0 (all were already git-ignored)

