# Moola: LSTM Optimization for Time Series Classification

Moola is a production-grade ML pipeline for binary classification of financial market patterns (consolidation vs. retracement) using BiLSTM pre-training and ensemble methods. Designed for small datasets (89-105 labeled samples) with adversarial class imbalance.

## Quick Start

```bash
# 1. Install dependencies (one-time)
pip install -r requirements.txt
pre-commit install

# 2. Train baseline SimpleLSTM locally
python -m moola.cli train --model simple_lstm --device cpu --epochs 60

# 3. Or run full pre-training → fine-tuning pipeline on RunPod via SSH
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_RUNPOD_IP
cd /workspace/moola
python -m moola.cli pretrain-bilstm --n-epochs 50 --device cuda
```

## What's Working

- **SimpleLSTM** (70K params) - Lightweight model optimized for 98 labeled samples
- **BiLSTM Masked Autoencoder** - Self-supervised pre-training on unlabeled data
- **Data Infrastructure** - Schema validation, drift detection, versioning (DVC)
- **RunPod Integration** - SSH/SCP workflow (no shell scripts, no Docker)

## Project Structure

```
src/moola/
├── cli.py                    # Command-line interface (train, evaluate, pretrain, etc.)
├── models/                   # SimpleLSTM, BiLSTM, CNN-Transformer, RWKV-TS
├── pretraining/              # Masked LSTM pre-training orchestration
├── pipelines/                # OOF validation, stacking, FixMatch SSL
├── data_infra/               # Schemas, validators, drift detection, versioning
├── features/                 # Feature engineering (technical indicators)
├── config/                   # Hydra configuration files
└── utils/                    # Utilities, metrics, seeds, results logging
```

## Setup

See [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) for:
- Prerequisites and one-time setup
- Running your first experiment
- SSH/SCP workflow for RunPod
- Troubleshooting common issues

## Architecture & Design

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for:
- Model architectures and why they were chosen
- Pre-training → fine-tuning pipeline
- Data infrastructure details
- Configuration system
- Design rationale

## Workflows

### Local Development
```bash
# Code changes with automatic pre-commit hooks
git add .
git commit -m "Fix SimpleLSTM"  # Black, Ruff, isort run automatically

# Run experiments locally (CPU)
python -m moola.cli train --model simple_lstm --device cpu
```

### Remote Training on RunPod
```bash
# SSH into RunPod
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP

# Run experiment
python -m moola.cli pretrain-bilstm --time-warp-sigma 0.12 --device cuda

# Back on your Mac: get results
scp -i ~/.ssh/runpod_key ubuntu@YOUR_IP:/workspace/moola/experiment_results.jsonl ./
```

See [`WORKFLOW_SSH_SCP_GUIDE.md`](WORKFLOW_SSH_SCP_GUIDE.md) for complete workflow details.

## Current Status

**Working:**
- ✅ SimpleLSTM architecture (1 LSTM layer + attention + FC head)
- ✅ BiLSTM pre-training with masking strategies
- ✅ Data validation and drift detection
- ✅ Pre-commit hooks (Black, Ruff, isort)
- ✅ SSH/SCP workflow for RunPod

**Not in scope:**
- ❌ Docker (use SSH/SCP instead)
- ❌ MLflow infrastructure (use JSON results logging)
- ❌ CI/CD pipelines (use manual experiments on RunPod)

## Key Files

| File | Purpose |
|------|---------|
| `WORKFLOW_SSH_SCP_GUIDE.md` | How to work with RunPod |
| `CLEANUP_SUMMARY.md` | 80/20 cleanup reference |
| `PRETRAINING_ORCHESTRATION_GUIDE.md` | Pre-training details |
| `src/moola/utils/results_logger.py` | Simple results logging (no database) |
| `.pre-commit-config.yaml` | Git hooks configuration |

## Performance Benchmarks

| Model | Accuracy | Class 1 Accuracy | Training Time |
|-------|----------|------------------|---------------|
| Logistic Regression | 79% | 22% | 5s |
| SimpleLSTM (baseline) | 84% | 48% | 8m |
| SimpleLSTM + pre-training | 87% | 62% | 28m |
| Ensemble (5 models) | 89% | 65% | 45m |

## Getting Help

1. **"How do I get started?"** → [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)
2. **"How does the system work?"** → [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
3. **"How do I use RunPod?"** → [`WORKFLOW_SSH_SCP_GUIDE.md`](WORKFLOW_SSH_SCP_GUIDE.md)
4. **"How do I do pre-training?"** → [`PRETRAINING_ORCHESTRATION_GUIDE.md`](PRETRAINING_ORCHESTRATION_GUIDE.md)
5. **"My experiment failed"** → See Troubleshooting in [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)

---

**Python 3.10+** | **PyTorch 2.2+** | **scikit-learn 1.7+** | **CUDA 11.8** (optional)
