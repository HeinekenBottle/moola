# Jade Pre-training Guide

## Overview

This guide explains how to pre-train the Jade encoder on 5 years of NQ futures data (1.8M bars).

**What is Jade?**
- **Model**: BiLSTM (2 layers, 128 hidden, bidirectional)
- **Input**: 10 features × 105 timesteps (NOT 11 features!)
- **Purpose**: Self-supervised pre-training → transfer to supervised task
- **Expected boost**: +3-5% accuracy on 174 labeled samples

**The 10 Features** (from `src/moola/features/relativity.py`):
1. `open_norm` - Open position within candle [0,1]
2. `close_norm` - Close position within candle [0,1]
3. `body_pct` - Body size as % of range [-1,1]
4. `upper_wick_pct` - Upper wick percentage [0,1]
5. `lower_wick_pct` - Lower wick percentage [0,1]
6. `range_z` - Range normalized by EMA [0,3]
7. `dist_to_prev_SH` - Distance to swing high (ATR) [-3,3]
8. `dist_to_prev_SL` - Distance to swing low (ATR) [-3,3]
9. `bars_since_SH_norm` - Time since swing high [0,3]
10. `bars_since_SL_norm` - Time since swing low [0,3]

---

## Quick Start (RunPod)

### 1. Upload Data to RunPod

```bash
# On your Mac
scp -i ~/.ssh/runpod_key \
  data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet \
  ubuntu@YOUR_RUNPOD_IP:/workspace/moola/data/raw/
```

### 2. SSH to RunPod

```bash
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_RUNPOD_IP
cd /workspace/moola
```

### 3. Run Batch Size Sweep (RECOMMENDED)

```bash
# Test 3 batch sizes: 512, 768, 1024
bash scripts/runpod_batch_size_sweep.sh
```

This will:
- Pre-train Jade encoder 3 times (50 epochs each)
- Save checkpoints for each batch size
- Take ~3-4 hours total on RTX 4090

### 4. Retrieve Results

```bash
# On your Mac
scp -i ~/.ssh/runpod_key -r \
  ubuntu@YOUR_IP:/workspace/moola/artifacts/batch_size_sweep \
  ./artifacts/
```

### 5. Compare Results

```bash
# Check best validation loss for each batch size
cat artifacts/batch_size_sweep/batch_512/training_results.json | jq '.best_val_loss'
cat artifacts/batch_size_sweep/batch_768/training_results.json | jq '.best_val_loss'
cat artifacts/batch_size_sweep/batch_1024/training_results.json | jq '.best_val_loss'
```

---

## Single Experiment (Manual)

If you want to run a single pre-training experiment:

```bash
python3 scripts/train_jade_pretrain.py \
  --config configs/windowed.yaml \
  --data data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet \
  --epochs 50 \
  --batch-size 1024 \
  --seed 42 \
  --output-dir artifacts/jade_pretrain
```

---

## Data Statistics

**Input File**: `data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet`
- Total bars: 1,797,854 (5 years of 1-minute NQ futures)
- File size: 29.4 MB
- Date range: 2020-09-01 to 2025-09-30

**Windowed Dataset** (with stride=52, ~50% overlap):
- Windows: ~34,573
- Shape: [N, 105, 10] (windows × timesteps × features)
- Memory: ~0.14 GB (fp32)

**Train/Val/Test Split** (from `configs/windowed.yaml`):
- Train: 2020-09-01 to 2024-12-31 (~80%)
- Val: 2025-01-01 to 2025-03-31 (~10%)
- Test: 2025-04-01 to 2025-06-30 (~10%)

---

## Batch Size Recommendations

| Batch Size | VRAM Usage | Training Time (50 epochs) | Recommendation |
|------------|------------|---------------------------|----------------|
| 512 | 0.66 GB | ~2-2.5 hours | Good baseline |
| 768 | 0.74 GB | ~1.5-2 hours | Sweet spot ⭐ |
| 1024 | 0.82 GB | ~1-1.5 hours | Maximum throughput |

**Recommendation**: Start with **1024** for fastest iteration. All batch sizes fit comfortably in 24GB VRAM (<1GB used).

---

## Training Configuration

From `scripts/train_jade_pretrain.py`:

```python
# Model Architecture
input_size: 10          # 10 features (NOT 11!)
hidden_size: 128        # BiLSTM hidden units per direction
num_layers: 2           # BiLSTM layers
dropout: 0.2            # Dropout rate

# Training
epochs: 50              # Standard pre-training duration
batch_size: 256-1024    # Sweep to find optimal
learning_rate: 1e-3     # Initial LR
weight_decay: 1e-2      # AdamW regularization
warmup_epochs: 5        # Linear warmup period

# Masking
mask_ratio: 0.15        # Fraction of timesteps to mask
huber_delta: 1.0        # Huber loss delta

# Optimization
grad_clip: 1.0          # Gradient clipping
mixed_precision: true   # FP16 training (if available)
```

---

## Expected Results

**Pre-training Metrics**:
- Reconstruction loss: Target ~0.5-1.0 (Huber loss)
- Training time: ~1.5 hours @ batch 1024 on RTX 4090
- VRAM usage: <1GB peak

**Transfer Learning** (on 174 labeled samples):
- Baseline (no pre-training): 84% accuracy
- With pre-training: 87-89% accuracy (+3-5%)
- Class 1 recall: 48% → 62% (minority class improvement)

---

## Troubleshooting

### OOM Error (Out of Memory)

If you get OOM errors at batch size 1024:

```bash
# Reduce batch size
python3 scripts/train_jade_pretrain.py \
  --batch-size 768 \
  --epochs 50
```

### Data Not Found

Make sure you've uploaded the parquet file:

```bash
ls -lh data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet
# Should show: 29.4 MB file
```

### Slow Training

If training is slower than expected:
- Check GPU is being used: `nvidia-smi`
- Verify mixed precision is enabled (default: true)
- Increase batch size to maximize GPU utilization

---

## Next Steps After Pre-training

### 1. Extract Best Encoder

```bash
# Copy best encoder checkpoint
cp artifacts/batch_size_sweep/batch_1024/checkpoint_best.pt \
   artifacts/encoders/pretrained/jade_encoder_5yr_v1.pt
```

### 2. Fine-tune on Labeled Data

```bash
python3 -m moola.cli train \
  --model jade \
  --pretrained-encoder artifacts/encoders/pretrained/jade_encoder_5yr_v1.pt \
  --freeze-encoder true \
  --device cuda \
  --n-epochs 60
```

### 3. Compare vs Baseline

```bash
# Baseline (no pre-training)
python3 -m moola.cli train --model jade --device cuda --n-epochs 60

# With pre-training
python3 -m moola.cli train \
  --model jade \
  --pretrained-encoder artifacts/encoders/pretrained/jade_encoder_5yr_v1.pt \
  --device cuda \
  --n-epochs 60
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `scripts/train_jade_pretrain.py` | Main pre-training script |
| `scripts/runpod_batch_size_sweep.sh` | Batch size sweep automation |
| `configs/windowed.yaml` | Windowing configuration |
| `src/moola/models/jade_core.py` | Jade architecture (supervised) |
| `src/moola/models/jade_pretrain.py` | Jade pre-training model |
| `src/moola/features/relativity.py` | 10-feature engineering pipeline |
| `src/moola/data/windowed_loader.py` | Data loading for pre-training |

---

## Important Notes

1. **Feature Count**: Jade uses **10 features** (NOT 11). This is the current implementation in `relativity.py`.

2. **Data Source**: The 5-year NQ parquet file (`nq_ohlcv_1min_2020-09_2025-09_fixed.parquet`) contains raw OHLC data. Features are computed on-the-fly during data loading.

3. **Batch Size**: All recommended batch sizes (512-1024) fit comfortably in 24GB VRAM. Choose larger for speed.

4. **Pre-training vs Supervised**:
   - `JadePretrainer` (this guide) = self-supervised MAE on unlabeled data
   - `JadeCore` = supervised model for classification on labeled data

5. **Transfer Learning**: After pre-training, the encoder weights are transferred to the supervised Jade model and optionally frozen during fine-tuning.

---

## Questions?

See:
- `CLAUDE.md` - Project context for AI assistants
- `README.md` - Project overview
- `src/moola/models/MODEL_REGISTRY.md` - Jade model documentation
