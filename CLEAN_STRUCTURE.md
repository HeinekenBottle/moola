# Clean Repository Structure
**Date:** 2025-10-21
**Project:** Moola ML Pipeline - Stones 80/20 Cleanup

## Directory Tree (Top 3 Levels)

```
moola/
├── artifacts/                    # Git-ignored: Model checkpoints, results
│   ├── encoders/
│   │   ├── pretrained/          # BiLSTM MAE encoders (*.pt)
│   │   └── supervised/          # Supervised encoders
│   ├── models/                  # Model checkpoints (*.pkl)
│   │   ├── supervised/          # Production models
│   │   ├── enhanced_simple_lstm/
│   │   ├── simple_lstm/
│   │   ├── logreg/
│   │   ├── rf/
│   │   ├── xgb/
│   │   └── stack/
│   ├── oof/                     # Out-of-fold predictions
│   │   ├── supervised/
│   │   └── pretrained/
│   ├── reports/                 # Evaluation reports
│   ├── results/                 # Experiment results
│   ├── runs/                    # Training run artifacts
│   └── splits/                  # Data splits
│       └── v1/
├── configs/                     # ✅ Stones YAML configs
│   ├── default.yaml
│   └── model/
│       ├── jade.yaml            # ✅ Jade (moola-lstm-m-v1.0)
│       ├── sapphire.yaml        # ✅ Sapphire (moola-preenc-fr-s-v1.0)
│       └── opal.yaml            # ✅ Opal (moola-preenc-ad-m-v1.0)
├── data/                        # Git-ignored: All data files
│   ├── batches/                 # Annotation batches
│   ├── corrections/             # Human annotations
│   │   ├── candlesticks_annotations/
│   │   └── review_corrections/
│   ├── logs/                    # Data processing logs
│   ├── oof/                     # OOF predictions (*.npy)
│   │   ├── supervised/
│   │   └── pretrained/
│   ├── processed/               # Processed datasets
│   │   ├── labeled/             # Training data
│   │   ├── unlabeled/           # Pre-training data
│   │   └── archived/            # Historical datasets
│   ├── raw/                     # Raw market data
│   │   ├── labeled/
│   │   └── unlabeled/
│   └── splits/                  # Train/val splits
├── logs/                        # Git-ignored: Training logs
├── scripts/                     # Utility scripts
│   ├── cleanlab/
│   │   └── run_cleanlab.py      # ✅ CleanLab entrypoint
│   ├── generate_report.py       # ✅ Report generation
│   └── runpod/
│       ├── README.md            # RunPod utilities docs
│       ├── dependency_audit.py  # ✅ Validation utility
│       └── verify_runpod_env.py # ✅ Validation utility
├── src/                         # Source code
│   ├── data/                    # Data utilities
│   │   └── window105.py         # Candlesticks integration
│   └── moola/                   # Main package
│       ├── api/                 # API interfaces
│       ├── aug/                 # Augmentation
│       ├── calibrate/           # Calibration
│       ├── cli.py               # ✅ CLI entrypoint
│       ├── config/              # Python config classes
│       ├── configs/             # YAML configs (internal)
│       │   ├── model/
│       │   └── train/
│       ├── data/                # Data loading
│       ├── data_infra/          # Data infrastructure
│       ├── encoder/             # Encoder utilities
│       ├── eval/                # Evaluation
│       ├── features/            # Feature engineering
│       ├── heads/               # Model heads
│       ├── loss/                # Loss functions
│       ├── metrics/             # Metrics
│       ├── models/              # ✅ Model architectures
│       │   ├── jade.py          # ✅ Jade (Stones)
│       │   ├── enhanced_simple_lstm.py  # ✅ Sapphire/Opal base
│       │   ├── simple_lstm.py   # ✅ Baseline
│       │   ├── logreg.py        # Logistic regression
│       │   ├── rf.py            # Random forest
│       │   ├── xgb.py           # XGBoost
│       │   └── stack.py         # Stacking ensemble
│       ├── pretraining/         # Self-supervised learning
│       ├── runpod/              # RunPod utilities
│       │   ├── README.md
│       │   └── scp_orchestrator.py
│       ├── schemas/             # Data schemas
│       ├── train/               # Training loops
│       ├── utils/               # Utilities
│       └── validation/          # Validation
├── tests/                       # Test suite
│   ├── data/                    # Data tests
│   ├── integration/             # Integration tests
│   ├── models/                  # Model tests
│   └── utils/                   # Utility tests
├── CLAUDE.md                    # ✅ AI assistant context
├── COMPLIANCE_REPORT.md         # ✅ Stones compliance (NEW)
├── HEAVY_UNTRACKED.md           # ✅ Heavy files report (NEW)
├── Makefile                     # ✅ Stones workflow automation
├── README.md                    # ✅ Project documentation
├── _final_inventory.md          # ✅ Cleanup inventory (NEW)
├── pyproject.toml               # ✅ Project metadata
├── requirements.txt             # ✅ Dependencies
└── uv.lock                      # ✅ Dependency lock
```

## File Counts by Directory

| Directory | Files | Notes |
|-----------|-------|-------|
| `src/moola/` | ~150 | Source code (Python) |
| `tests/` | ~20 | Test suite |
| `configs/` | 4 | Stones YAML configs only |
| `scripts/` | 3 | Essential utilities only |
| `artifacts/` | ~50 | Git-ignored (models, results) |
| `data/` | ~30 | Git-ignored (datasets) |
| `logs/` | ~10 | Git-ignored (training logs) |

## Stones-Compliant Structure ✅

### Configuration Files
- ✅ `configs/model/jade.yaml` - Jade configuration
- ✅ `configs/model/sapphire.yaml` - Sapphire configuration
- ✅ `configs/model/opal.yaml` - Opal configuration
- ✅ `configs/default.yaml` - Default configuration

**No extra configs** - Only Stones models present

### Model Files
- ✅ `src/moola/models/jade.py` - Jade (moola-lstm-m-v1.0)
- ✅ `src/moola/models/enhanced_simple_lstm.py` - Sapphire/Opal base
- ✅ `src/moola/models/simple_lstm.py` - Baseline reference
- ✅ `src/moola/models/{logreg,rf,xgb,stack}.py` - Ensemble components

### Scripts
- ✅ `scripts/cleanlab/run_cleanlab.py` - CleanLab entrypoint
- ✅ `scripts/generate_report.py` - Report generation (used by Makefile)
- ✅ `scripts/runpod/dependency_audit.py` - Validation utility
- ✅ `scripts/runpod/verify_runpod_env.py` - Validation utility

**No shell scripts** - Complies with SSH/SCP workflow constraint

### Documentation
- ✅ `README.md` - Project overview
- ✅ `CLAUDE.md` - AI assistant context (21K)
- ✅ `Makefile` - Workflow automation (20K)
- ✅ `COMPLIANCE_REPORT.md` - Stones compliance verification (NEW)
- ✅ `HEAVY_UNTRACKED.md` - Heavy files report (NEW)

**No stray docs** - Cleanup summaries archived

## Changes from Previous State

### Removed
- ❌ 200+ missing files untracked from git
- ❌ All `__pycache__/` directories (30+)
- ❌ `.pytest_cache`, `.ruff_cache`, `.benchmarks`
- ❌ `.dvc/` directory (DVC not used)
- ❌ `CLEANUP_SESSION_2025-10-21.md` (archived)
- ❌ `README_CLEANUP.txt` (archived)
- ❌ `scripts/demo_bootstrap_ci.py` (archived)
- ❌ `src/moola/cli_feature_aware.py` (archived)
- ❌ `src/moola/configs/model/enhanced_simple_lstm.yaml` (archived)

### Kept
- ✅ All Stones models (Jade, Sapphire, Opal)
- ✅ All source code in `src/moola/`
- ✅ All tests in `tests/`
- ✅ Essential scripts only
- ✅ Makefile Stones targets
- ✅ Core documentation (README, CLAUDE, Makefile)

## Summary

**Total directories:** ~80 (including subdirectories)
**Total tracked files:** ~200 (source code, configs, docs)
**Git-ignored files:** ~100+ (artifacts, data, logs)
**Largest tracked file:** 292K (uv.lock)

**Structure status:** ✅ CLEAN
- No duplicate directories
- No stray documentation
- No heavy tracked files
- Stones-compliant configs only
- Essential scripts only

