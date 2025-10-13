# Moola

ML trading pattern recognition system using expansion indices for precise feature extraction.

## Features

- **Expansion Index Integration**: Extracts features from variable-length pattern regions (5-23 bars)
- **ICT-Aligned Features**: Market structure, liquidity zones, fair value gaps, order blocks
- **Multi-Model Pipeline**: XGBoost, Random Forest, Logistic Regression, CNN-Transformer, RWKV-TS
- **RunPod GPU Deployment**: Optimized for cloud training with minimal dependencies

## Quick Start

### Local Training
```bash
python -m moola.cli train --model xgb --device cpu --seed 1337
```

### RunPod Deployment
```bash
# Deploy from local machine
.runpod/deploy-fast.sh

# Train on RunPod
ssh runpod
cd /workspace
bash scripts/start.sh --train
```

## Dataset

**Pivot v2.7**: 134 CleanLab-cleaned samples with expansion indices
- Consolidation: 65 samples (48.5%)
- Retracement: 50 samples (37.3%)
- Reversal: 19 samples (14.2%)

## Performance

- Test accuracy: 51.9% (beats 33.3% random baseline by 55.9%)
- 56.8% of features have variance >0.1 (healthy signal from expansion regions)

## Architecture

```
data/processed/train_pivot_134.parquet (134 samples with expansion_start/end)
    ↓
src/moola/features/price_action_features.py (extract from expansion regions)
    ↓
src/moola/models/ (xgb.py, rf.py, etc.)
    ↓
data/artifacts/ (trained models + OOF predictions)
```

## Requirements

- Python 3.10+
- PyTorch (for deep learning models)
- XGBoost 2.0+
- See `requirements-runpod.txt` for minimal deployment dependencies

## License

MIT
