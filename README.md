# Moola
Independent system build. No inheritance from prior projects. Clean pipeline.

## Current Status: Stage 1 Ready ✅

- **Dataset**: Window105 training dataset (134 samples, 420 OHLC features, 3 classes)
- **Pipeline**: RandomForest meta-learner stacking with 5 base models
- **Ready for**: GPU training on RunPod, OOF generation, ensemble evaluation

## Quick Start

### Local Development
```bash
# Install dependencies
uv install

# Ingest corrected dataset
uv run -m moola.cli ingest --input data/processed/window105_train.parquet

# Verify setup
uv run -m moola.cli doctor
uv run -m moola.cli audit --section data
```

### RunPod Setup
```bash
# Clone and build on RunPod
git clone https://github.com/HeinekenBottle/moola.git ~/moola && cd ~/moola
docker build -f docker/Dockerfile.gpu -t moola:gpu .

# Run pipeline with GPU
docker run --rm --gpus all -v ~/moola:/workspace/moola -w /workspace/moola moola:gpu \
  uv run -m moola.cli oof --model cnn_transformer --device cuda --seed 1337
```

## Training Pipeline

### 1. Classical Models (CPU)
```bash
uv run -m moola.cli oof --model rf --device cpu --seed 1337
uv run -m moola.cli oof --model xgb --device cpu --seed 1337
```

### 2. Deep Learning Models (GPU)
```bash
docker run --gpus all moola:gpu uv run -m moola.cli oof --model cnn_transformer --device cuda --seed 1337
docker run --gpus all moola:gpu uv run -m moola.cli oof --model rwkv_ts --device cuda --seed 1337
```

### 3. Stack and Evaluate
```bash
docker run --gpus all moola:gpu uv run -m moola.cli stack-train --stacker rf --seed 1337
docker run --gpus all moola:gpu uv run -m moola.cli audit
```

## Architecture

- **Base Models**: logreg, rf, xgb, rwkv_ts, cnn_transformer
- **Meta-Learner**: RandomForest (1000 trees, balanced_subsample)
- **Features**: 420 OHLC features (105 timesteps × 4 OHLC channels)
- **Validation**: 5-fold StratifiedKFold (seed=1337)
- **Acceptance**: F1 improvement ≥2pp OR ECE ≤0.03

## Dataset

```
window105_train.parquet
├── Shape: (134, 3)
├── Features: 420 (OHLC expanded properly)
├── Labels: consolidation (48.5%), retracement (37.3%), reversal (14.2%)
└── Schema: window_id | label | features
```

## Environment

- Python 3.11+
- PyTorch <2.5 (CUDA 12.1)
- XGBoost with CUDA support
- Scikit-learn, Pandas, NumPy
- Docker with GPU support
