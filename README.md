# MOOLA — Production ML Pipeline for NQ Futures Pattern Classification

Production-grade ML pipeline for binary classification of financial patterns (consolidation vs. retracement) in NQ futures using BiLSTM pre-training. Designed for small datasets (33-200 labeled samples) with strict workflow constraints.

## Stones Doctrine (Non-Negotiable)

- **Pointer**: Center+Length (Huber δ≈0.08)
- **Loss**: Uncertainty-weighted (Kendall) — NO manual λ
- **Dropout**: recurrent 0.6–0.7, dense 0.4–0.5, input 0.2–0.3
- **Augment**: jitter σ=0.03 + magnitude-warp σ=0.2 (×3, on-the-fly)
- **Uncertainty**: MC Dropout 50–100 + Temperature Scaling
- **Gates**: Hit@±3 ≥60, F1-macro ≥0.50, ECE <0.10, Joint ≥40
- **Input**: [B,105,11] single canonical schema

## Stones Collection (Production SKUs)

| SKU | Codename | Description |
|-----|----------|-------------|
| `moola-lstm-m-v1.0` | **Jade** | Production BiLSTM with multi-task learning |
| `moola-preenc-fr-s-v1.0` | **Sapphire** | Frozen encoder transfer learning |
| `moola-preenc-ad-m-v1.0` | **Opal** | Adaptive fine-tuning transfer learning |

## Quick Start

```bash
# Install dependencies (one-time)
pip install -r requirements.txt
pre-commit install

# Train Jade (production model)
make train-jade DEVICE=cuda EPOCHS=60

# Train all Stones models
make stones DEVICE=cuda

# Evaluate with gates
make eval

# Generate report
make report
```

See [MODEL_REGISTRY.md](src/moola/models/MODEL_REGISTRY.md) for full documentation.

## Documentation

**Core Documentation** (in this repository):
- `README.md` — This file (quick start and overview)
- `CLAUDE.md` — Project context for AI assistants
- `src/moola/models/MODEL_REGISTRY.md` — Stones SKU documentation

**Extended Documentation** (archived):
- Phase summaries, implementation guides, and research notes are archived in `../moola_docs_archive/`
- Legacy code and experimental models are archived in `../moola_legacy_20251021_193518/`
- Databento utilities are in `../databento/`

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
