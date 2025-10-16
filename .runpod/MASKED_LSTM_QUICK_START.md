# Masked LSTM Pre-training - Quick Start Guide

## TL;DR

**One command to rule them all**:
```bash
.runpod/full_pipeline_masked_lstm.sh 213.173.110.220 36832
```

**Time**: ~50-60 minutes
**Cost**: ~$0.31 (RTX 4090 on-demand)
**Expected improvement**: +8-12% accuracy over baseline

---

## What This Does

1. **Pre-trains** BiLSTM encoder on 5,000 unlabeled OHLC sequences using masked reconstruction
2. **Fine-tunes** SimpleLSTM with frozen pre-trained encoder (10 epochs)
3. **Unfreezes** encoder and continues training (full fine-tuning)
4. **Trains** baseline SimpleLSTM (no pre-training) for comparison
5. **Downloads** results and compares performance

---

## Quick Commands

### Full Pipeline (Recommended)
```bash
# Complete workflow: pre-training → fine-tuning → baseline → comparison
.runpod/full_pipeline_masked_lstm.sh <HOST> <PORT>

# Example
.runpod/full_pipeline_masked_lstm.sh 213.173.110.220 36832
```

### Pre-training Only
```bash
# Just pre-train and download encoder
.runpod/deploy_pretrain_masked_lstm.sh <HOST> <PORT>
```

### Monitor Progress
```bash
# Real-time monitoring (refresh every 5s)
python scripts/monitor_pretraining.py --host <HOST> --port <PORT> --watch

# One-time status check
python scripts/monitor_pretraining.py --host <HOST> --port <PORT>
```

### Compare Results
```bash
# Analyze performance improvements
python scripts/compare_masked_lstm_results.py
```

---

## File Structure

**Generated files**:
```
data/
├── processed/
│   └── unlabeled_pretrain.parquet          # 5,000 unlabeled sequences (auto-generated)
└── artifacts/
    └── pretrained/
        └── bilstm_encoder.pt               # Pre-trained encoder (download from RunPod)
    └── oof/
        └── simple_lstm/v1/
            ├── seed_1337_pretrained.npy    # Fine-tuned predictions
            └── seed_42_baseline.npy        # Baseline predictions
```

**Deployment scripts**:
```
.runpod/
├── full_pipeline_masked_lstm.sh            # Full pipeline (recommended)
├── deploy_pretrain_masked_lstm.sh          # Pre-training only
├── MASKED_LSTM_DEPLOYMENT_RUNBOOK.md       # Detailed guide
├── MASKED_LSTM_QUICK_START.md              # This file
└── RTX_4090_OPTIMIZATION_GUIDE.md          # Performance tuning
```

---

## Expected Output

### Pre-training Logs
```
[MASKED LSTM PRE-TRAINING]
======================================================================
  Dataset size: 5000 samples
  Mask strategy: patch
  Mask ratio: 0.15
  Batch size: 512
  Epochs: 50
  Device: cuda
======================================================================

Epoch [1/50]
  Train Loss: 0.0234 | Val Loss: 0.0189
  Train Recon: 0.0234 | Val Recon: 0.0189
  LR: 0.001000

...

Epoch [42/50]
  Train Loss: 0.0012 | Val Loss: 0.0011
  Train Recon: 0.0012 | Val Recon: 0.0011
  LR: 0.000234

[EARLY STOPPING] Triggered at epoch 42

======================================================================
PRE-TRAINING COMPLETE
======================================================================
  Final train loss: 0.0012
  Final val loss: 0.0011
  Best val loss: 0.0011
  Encoder saved: /workspace/artifacts/pretrained/bilstm_encoder.pt
======================================================================
```

### Fine-tuning Logs
```
[SSL PRE-TRAINING] Loading pre-trained encoder from: /workspace/artifacts/pretrained/bilstm_encoder.pt
[SSL PRE-TRAINING] Architecture verified (hidden_dim=128)
[SSL PRE-TRAINING] Loaded 12 parameter tensors:
  ✓ lstm.weight_ih_l0
  ✓ lstm.weight_hh_l0
  ✓ lstm.bias_ih_l0
  ✓ lstm.bias_hh_l0
  ...
[SSL PRE-TRAINING] Freezing LSTM encoder weights
  → Encoder frozen. Only classifier will be trained initially.

Epoch [10/60] Train Loss: 0.3245 Acc: 0.6834 | Val Loss: 0.3567 Acc: 0.6532

[SSL PRE-TRAINING] Unfreezing LSTM encoder at epoch 11
[SSL PRE-TRAINING] Reduced LR to 0.000250

Epoch [20/60] Train Loss: 0.2891 Acc: 0.7234 | Val Loss: 0.3012 Acc: 0.7045
Epoch [30/60] Train Loss: 0.2567 Acc: 0.7456 | Val Loss: 0.2834 Acc: 0.7245

Early stopping triggered at epoch 38
```

### Comparison Results
```
======================================================================
MASKED LSTM PRE-TRAINING RESULTS COMPARISON
======================================================================

OVERALL METRICS
--------------------------------------------------------------------------------
Metric                          Pre-trained        Baseline          Δ
--------------------------------------------------------------------------------
Accuracy                            0.7245          0.6532      +0.0713
Balanced Accuracy                   0.6823          0.5641      +0.1182

PER-CLASS METRICS
--------------------------------------------------------------------------------
Class 0:
  Precision                          0.7423          0.7156      +0.0267
  Recall                             0.7834          0.8923      -0.1089
  F1-score                           0.7623          0.7973      -0.0350
  Support                                56

Class 1:
  Precision                          0.5833          0.0000      +0.5833
  Recall                             0.5250          0.0000      +0.5250
  F1-score                           0.5526          0.0000      +0.5526
  Support                                20

Class 2:
  Precision                          0.7321          0.5556      +0.1765
  Recall                             0.7273          0.4545      +0.2728
  F1-score                           0.7297          0.5000      +0.2297
  Support                                22

======================================================================
SUMMARY
======================================================================
✅ IMPROVEMENT: +0.0713 accuracy (+10.9%)

Class-specific improvements:
  ✅ Class 1 recall: 0.0000 → 0.5250 (+0.5250)
  ✅ Class 2 recall: 0.4545 → 0.7273 (+0.2728)
======================================================================
```

---

## Timing Breakdown (RTX 4090)

| Stage | Time | What's Happening |
|-------|------|------------------|
| Setup | ~2 min | Clone repo, setup venv, upload data |
| Pre-training | ~30-40 min | 5,000 sequences, 50 epochs, batch_size=512 |
| Fine-tuning (pre-trained) | ~10-15 min | 98 samples, 60 epochs, frozen→unfrozen |
| Fine-tuning (baseline) | ~10-15 min | 98 samples, 60 epochs, from scratch |
| Download | ~1 min | Download encoder + predictions |
| **Total** | **~50-60 min** | **End-to-end pipeline** |

---

## Troubleshooting

### "CUDA out of memory"
```python
# Reduce batch size in src/moola/config/training_config.py
MASKED_LSTM_BATCH_SIZE = 256  # Down from 512
```

### "Pre-training too slow" (>60 min)
```bash
# Check GPU utilization (should be 90-100%)
ssh root@<HOST> -p <PORT> "nvidia-smi dmon -s u"

# If low utilization, increase batch size
MASKED_LSTM_BATCH_SIZE = 1024  # Up from 512
```

### "No improvement over baseline"
```bash
# Increase pre-training dataset
python scripts/generate_unlabeled_data.py --target-count 2000 --augment-factor 5

# Or increase pre-training epochs
MASKED_LSTM_N_EPOCHS = 100  # Up from 50
```

### "Connection refused"
```bash
# Verify pod is running and connection works
.runpod/check-connection.sh <HOST> <PORT>
```

---

## Configuration Files

All hyperparameters in `src/moola/config/training_config.py`:

```python
# Pre-training architecture
MASKED_LSTM_HIDDEN_DIM = 128
MASKED_LSTM_NUM_LAYERS = 2
MASKED_LSTM_DROPOUT = 0.2

# Masking strategy
MASKED_LSTM_MASK_RATIO = 0.15
MASKED_LSTM_MASK_STRATEGY = "patch"
MASKED_LSTM_PATCH_SIZE = 7

# Training
MASKED_LSTM_N_EPOCHS = 50
MASKED_LSTM_LEARNING_RATE = 1e-3
MASKED_LSTM_BATCH_SIZE = 512

# Transfer learning
MASKED_LSTM_FREEZE_EPOCHS = 10
MASKED_LSTM_UNFREEZE_LR_REDUCTION = 0.5
```

---

## Next Steps

After successful deployment:

1. **Analyze results**:
   ```bash
   python scripts/compare_masked_lstm_results.py
   ```

2. **Experiment with larger datasets**:
   ```bash
   python scripts/generate_unlabeled_data.py --target-count 2000 --augment-factor 8
   ```

3. **Try different masking strategies**:
   - Edit `MASKED_LSTM_MASK_STRATEGY` in `training_config.py`
   - Options: `"random"`, `"block"`, `"patch"`

4. **Deploy to production**:
   - Export pre-trained encoder
   - Integrate into inference pipeline

---

## Documentation

- **Full runbook**: `.runpod/MASKED_LSTM_DEPLOYMENT_RUNBOOK.md`
- **Performance tuning**: `.runpod/RTX_4090_OPTIMIZATION_GUIDE.md`
- **Troubleshooting**: `.runpod/TROUBLESHOOTING.md`

---

## Support

**Issues**: Open GitHub issue with:
- Error message
- Pod specs (GPU, template)
- Deployment script used
- Relevant logs

**Quick help**:
```bash
# Monitor training in real-time
python scripts/monitor_pretraining.py --host <HOST> --port <PORT> --watch

# Check GPU stats
ssh root@<HOST> -p <PORT> "nvidia-smi"

# View logs
ssh root@<HOST> -p <PORT> "tail -f /tmp/training.log"
```
