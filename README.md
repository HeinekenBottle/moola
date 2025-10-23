# Moola — ML Pipeline for Financial Pattern Classification

A production-ready ML pipeline for binary classification of financial patterns in NQ futures data. Features automated feature engineering, model training, and deployment with strict compliance to Stones doctrine.

## Key Features

- **Automated Feature Engineering**: Relativity features with candle norms, swing relativity, ATR normalization
- **Model Training**: Jade BiLSTM with multi-task learning, uncertainty weighting
- **Data Pipeline**: Pre-computation for fast training, windowed data loading
- **Deployment**: RunPod integration for GPU training, automated sync scripts
- **Compliance**: Stones doctrine enforcement (Float32, no absolute prices, causality)

## Quick Start

```bash
# Install
pip install -r requirements.txt
pip install -e .
pre-commit install

# Pre-compute features (requires 5-year NQ data in data_raw/)
python3 scripts/precompute_nq_features.py --data data_raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet --output data/processed/nq_features

# Train Jade model
python3 -m moola.cli train --model jade --device cuda

# Evaluate
python3 -m moola.cli evaluate --model jade
```

## Project Structure

- `src/moola/` - Core package
- `configs/` - Hydra configurations
- `scripts/` - Training and utility scripts
- `tests/` - Test suite
- `data_raw/` - Raw data storage (not in repo)
- `artifacts/` - Model outputs and logs (not in repo)

## Documentation

- `AGENTS.md` - Detailed guide for models, features, and workflows
- `CLAUDE.md` - Context for AI assistants
- `.factory/` - AI agent workflows and documentation

## Data Setup

1. Download 5-year NQ data using Databento API (see .factory/docs/2025-10-22-download-missing-nq-data-months-via-databento-api.md)
2. Place in `data_raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet`
3. Run feature pre-computation

## Training

- Use `python3 -m moola.cli` for all operations
- Models: Jade (BiLSTM), with pre-training support
- Features: Relativity (candle norms, swing relativity)
- Configs in `configs/` directory

## Deployment

- Sync to RunPod: `./scripts/sync_to_runpod.sh <IP>`
- Train remotely, sync results back with `./scripts/sync_from_runpod.sh <IP>`

## Compliance

- Stones doctrine: Float32, uncertainty weighting, no absolute prices
- Causality: No future information leakage
- Scale invariance: Features normalized and bounded

---

**Python 3.10+** | **PyTorch 2.3+** | **CUDA 12.1+**
