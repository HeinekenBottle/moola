# Network Storage Contents Breakdown

## Overview

Your RunPod network volume has **10 GB** capacity. Here's exactly what will be stored and the expected sizes.

## What Gets Uploaded Initially (via `sync-to-storage.sh all`)

### 1. Deployment Scripts (~12 KB)
```
/runpod-volume/scripts/
├── pod-startup.sh           # 126 lines - Pod initialization
└── runpod-train.sh          # 126 lines - Training pipeline
```

**Purpose**: Bootstrap new pods, run training
**Size**: ~12 KB
**Upload once**: Yes, update only when you change the scripts

### 2. Training Data (~232 KB)
```
/runpod-volume/data/processed/
├── train.parquet            # 116 KB - Original training data
└── window105_train.parquet  # 116 KB - Processed OHLC features (134 samples)
```

**Purpose**: Training dataset for all models
**Size**: ~232 KB (0.23 MB)
**Upload once**: Yes, unless you update the dataset

### 3. Configuration Files (~1 KB)
```
/runpod-volume/configs/
├── default.yaml             # 263 bytes - Base configuration
└── hardware/
    ├── cpu.yaml
    └── gpu.yaml
```

**Purpose**: Model hyperparameters, paths
**Size**: ~1 KB
**Upload once**: Yes, update if you change hyperparameters

**Total Initial Upload**: ~245 KB (0.24 MB)

## What Gets Created During Training (on Network Storage)

### 4. Python Virtual Environment (~2-3 GB)
```
/runpod-volume/venv/
├── bin/
│   ├── python3
│   ├── pip
│   └── moola
├── lib/python3.10/site-packages/
│   ├── torch/              # ~2 GB (PyTorch with CUDA)
│   ├── numpy/              # ~20 MB
│   ├── pandas/             # ~50 MB
│   ├── sklearn/            # ~100 MB
│   ├── xgboost/            # ~50 MB
│   ├── transformers/       # (if used) ~500 MB
│   └── [other deps]        # ~200 MB
└── pyvenv.cfg
```

**Created by**: `pod-startup.sh` running `uv venv && uv pip install -e .`

**Dependencies installed** (from `pyproject.toml`):
- **PyTorch 2.x** with CUDA support (~2 GB)
- **NumPy, Pandas, Scikit-learn** (~200 MB)
- **XGBoost** with GPU support (~50 MB)
- **Hydra, Rich, Loguru, Click** (~50 MB)
- **PyArrow, Pydantic, PyYAML** (~50 MB)
- **OpenAI** (~10 MB)

**Size**: ~2-3 GB
**Created once**: Yes, reused across all future pods
**Time saved**: 5-10 minutes per pod startup

### 5. Training Artifacts (~500 MB - 1 GB)

```
/runpod-volume/artifacts/
├── models/
│   ├── logreg/
│   │   ├── model.pkl        # ~1 MB - Logistic Regression
│   │   └── metrics.json
│   ├── rf/
│   │   ├── model.pkl        # ~50 MB - RandomForest (1000 trees)
│   │   └── metrics.json
│   ├── xgb/
│   │   ├── model.pkl        # ~20 MB - XGBoost
│   │   └── metrics.json
│   ├── rwkv_ts/
│   │   ├── model.pkl        # ~100 MB - RWKV state-space model
│   │   └── metrics.json
│   ├── cnn_transformer/
│   │   ├── model.pkl        # ~150 MB - CNN+Transformer
│   │   └── metrics.json
│   └── stack/
│       ├── stack.pkl        # ~50 MB - Stacking ensemble
│       └── metrics.json
│
├── oof/                     # Out-of-fold predictions
│   ├── logreg/v1/
│   │   └── seed_1337.npy    # ~3 KB (134 samples × 3 classes × 8 bytes)
│   ├── rf/v1/
│   │   └── seed_1337.npy    # ~3 KB
│   ├── xgb/v1/
│   │   └── seed_1337.npy    # ~3 KB
│   ├── rwkv_ts/v1/
│   │   └── seed_1337.npy    # ~3 KB
│   └── cnn_transformer/v1/
│       └── seed_1337.npy    # ~3 KB
│
├── splits/v1/               # CV fold definitions
│   └── seed_1337.json       # ~5 KB
│
├── predictions/
│   └── stack_test.csv       # ~10 KB
│
├── manifest.json            # ~2 KB
└── runs.csv                 # ~10 KB
```

**Created by**: Training pipeline
**Size breakdown**:
- Models: ~370 MB (5 base models + 1 stacker)
- OOF predictions: ~15 KB (tiny, just softmax outputs)
- Splits/metadata: ~20 KB

**Total artifacts**: ~400-500 MB

### 6. Training Logs (~10-50 MB)

```
/runpod-volume/logs/
├── train_logreg_1337.log
├── train_rf_1337.log
├── train_xgb_1337.log
├── train_rwkv_ts_1337.log
├── train_cnn_transformer_1337.log
├── stack_train_1337.log
└── audit.log
```

**Size**: ~10-50 MB (depends on verbosity)

## Total Network Storage Usage

| Category | Size | % of 10 GB |
|----------|------|------------|
| Scripts | 12 KB | 0.0001% |
| Data | 232 KB | 0.002% |
| Configs | 1 KB | 0.00001% |
| **Python venv** | **2-3 GB** | **20-30%** |
| **Training artifacts** | **400-500 MB** | **4-5%** |
| **Logs** | **10-50 MB** | **0.1-0.5%** |
| **TOTAL** | **~3-4 GB** | **30-40%** |

**Remaining space**: ~6-7 GB (plenty for multiple training runs)

## What Does NOT Go on Network Storage

### ❌ Source Code (~5 MB)
```
/workspace/moola/src/
```
**Why**: Changes frequently, should be fresh from git
**Location**: Pod local storage (ephemeral)

### ❌ Git Repository (~10 MB)
```
/workspace/moola/.git/
```
**Why**: Large, changes frequently
**Location**: Pod local storage

### ❌ Temporary Files
- `__pycache__/`
- `.pytest_cache/`
- Intermediate training checkpoints

## Optimization Tips

### Cache Hit Benefits
If you keep the venv on network storage:
- **First pod startup**: 5-10 minutes (install everything)
- **Subsequent pods**: ~30 seconds (just activate venv)

### What to Clear Periodically
If you run low on space:
```bash
# On pod or via S3 CLI
aws s3 rm --recursive s3://hg878tp14w/logs/         # Clear old logs
aws s3 rm --recursive s3://hg878tp14w/artifacts/oof/  # Clear old OOF
```

### Multiple Experiments
If you run multiple experiments:
```
/runpod-volume/artifacts/
├── experiment_001/
├── experiment_002/
└── experiment_003/
```

Each experiment: ~500 MB
Max experiments on 10 GB: ~12-15 (with venv)

## Upload Command Breakdown

### `./sync-to-storage.sh all` uploads:

1. **scripts/** (~12 KB)
   ```bash
   aws s3 sync .runpod/scripts/ s3://hg878tp14w/scripts/
   ```

2. **data/processed/** (~232 KB)
   ```bash
   aws s3 sync data/processed/ s3://hg878tp14w/data/processed/
   ```

3. **configs/** (~1 KB)
   ```bash
   aws s3 sync configs/ s3://hg878tp14w/configs/
   ```

**Total upload time**: ~1-2 seconds (0.25 MB total)

## Download Command Breakdown

### `./sync-from-storage.sh all` downloads:

1. **artifacts/** (~500 MB)
2. **logs/** (~10-50 MB)

**Total download time**: ~10-30 seconds (depending on your internet speed)

## Virtual Environment Contents

**Installed via** `uv pip install -e .`:

### Core ML Libraries
- `torch==2.x` (~2 GB with CUDA)
- `numpy>=1.26` (~20 MB)
- `pandas>=2.2` (~50 MB)
- `scikit-learn>=1.3` (~100 MB)
- `xgboost>=2.0` (~50 MB)

### Data & Config
- `pyarrow>=14.0` (~30 MB)
- `pydantic>=2.8` (~5 MB)
- `pyyaml>=6.0` (~1 MB)
- `hydra-core>=1.3` (~10 MB)

### CLI & Logging
- `typer>=0.12` (~2 MB)
- `click>=8.1` (~1 MB)
- `rich>=13.7` (~5 MB)
- `loguru>=0.7` (~1 MB)

### Utilities
- `python-dotenv>=1.0` (~1 MB)
- `openai>=2.3.0` (~5 MB)
- `pandera>=0.26.1` (~10 MB)

### Dependencies of dependencies
- `tqdm`, `joblib`, `threadpoolctl`, etc. (~50 MB)

**Total**: ~2.3-2.5 GB (mostly PyTorch)

## Key Takeaways

1. **Initial upload is tiny**: 0.25 MB (scripts + data + configs)
2. **Venv is the largest**: 2-3 GB, but saves 5-10 min per pod
3. **Artifacts are moderate**: 500 MB per training run
4. **Plenty of space**: 10 GB allows ~6-7 training runs + cached venv
5. **Fast iteration**: Upload changes in seconds, not minutes

## Recommended Workflow

1. **First time**: Upload everything, let pod create venv (~10 min)
2. **Update code**: Just git pull in pod (don't re-upload venv)
3. **Update scripts**: `./sync-to-storage.sh scripts` (~1 sec)
4. **New experiment**: Results saved automatically to network storage
5. **Download results**: `./sync-from-storage.sh artifacts` (~30 sec)

Your 10 GB network volume is more than enough for this project! 🎉
