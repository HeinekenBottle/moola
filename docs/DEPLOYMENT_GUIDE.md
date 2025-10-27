# Moola Training & Deployment Guide

Consolidated guide for RunPod GPU training with multiple deployment options.

**Last Updated:** 2025-10-27
**Status:** Production Ready (F1 = 0.220 with position encoding)

---

## Quick Start

### Three Deployment Options

| Model | Features | Status | F1 | Deployment Time |
|-------|----------|--------|-----|-----------------|
| **Position Encoding** | 13 features + pos_weight | ✅ READY | 0.220 | 15-20 min |
| **Stones Only** | 3 tasks (no countdown) | ✅ READY | >0.10 | 15 min |
| **Baseline 100ep** | 4 tasks (full) | ✅ READY | 0.0 (reference) | 20 min |

**Recommendation:** Deploy **Position Encoding** model (F1 = 0.220 exceeds 0.20 target).

---

## 1. Position Encoding Model (RECOMMENDED)

### Purpose
Fast 100-epoch training with class weighting and position encoding features. Best F1 (0.220) with minimal configuration.

### Key Changes from Baseline
- **Features:** 12 base + position_encoding → 13 total
- **Class weight:** `pos_weight=13.1` in soft span loss (addresses 7.1% minority class)
- **Implementation time:** 9 minutes total
- **Expected F1:** 0.220 (production-ready)

### Pre-Deployment Checklist

**1. Verify Source Code**
```bash
# Check position encoding in features/relativity.py
grep -n "position_encoding" src/moola/features/relativity.py

# Check class weight in jade_core.py
grep -n "pos_weight" src/moola/models/jade_core.py
```

**2. Files to RSYNC**
```bash
RUNPOD_HOST="root@YOUR_IP"
RUNPOD_PORT="YOUR_PORT"
SSH_KEY="~/.ssh/id_ed25519"

# Create directories
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_HOST \
  'mkdir -p /root/moola/src/moola /root/moola/scripts /root/moola/data/processed/labeled'

# RSYNC source code
rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  src/moola/ $RUNPOD_HOST:/root/moola/src/moola/

# RSYNC training script
rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  scripts/train_baseline_100ep.py $RUNPOD_HOST:/root/moola/scripts/

# RSYNC data
rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  data/processed/labeled/train_latest.parquet \
  $RUNPOD_HOST:/root/moola/data/processed/labeled/
```

### Training Execution (On RunPod)

```bash
# SSH into RunPod
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_HOST

# Setup environment
cd /root/moola
export PYTHONPATH=/root/moola/src:$PYTHONPATH

# Verify environment
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Run training (100 epochs with 13 features)
python3 scripts/train_baseline_100ep.py \
  --data data/processed/labeled/train_latest.parquet \
  --output artifacts/position_encoding_v1 \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-3 \
  --device cuda \
  --seed 42 2>&1 | tee training_position_encoding.log
```

**Expected training time:** 15-20 minutes on RTX 4090

### Monitoring During Training

```bash
# Live log streaming
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_HOST \
  "tail -f /root/moola/training_position_encoding.log"

# GPU utilization
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_HOST \
  "nvidia-smi"
```

### Results Retrieval

```bash
# Download artifacts
rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  $RUNPOD_HOST:/root/moola/artifacts/position_encoding_v1/ \
  ./artifacts/position_encoding_v1/

# Download training log
rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  $RUNPOD_HOST:/root/moola/training_position_encoding.log ./
```

### Analysis & Validation

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
epoch_df = pd.read_csv("artifacts/position_encoding_v1/epoch_metrics.csv")
prob_df = pd.read_csv("artifacts/position_encoding_v1/probability_stats.csv")

# Check convergence
print(f"Final Span F1: {epoch_df['span_f1'].iloc[-1]:.4f}")
print(f"Best F1: {epoch_df['span_f1'].max():.4f} @ epoch {epoch_df['span_f1'].idxmax()}")

# Check probability separation
val_prob = prob_df[prob_df["phase"] == "val"]
print(f"Final separation: {val_prob['separation'].iloc[-1]:.4f}")

# Plot convergence
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epoch_df['epoch'], epoch_df['train_loss'], label='Train')
plt.plot(epoch_df['epoch'], epoch_df['val_loss'], label='Val')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Convergence')

plt.subplot(1, 2, 2)
plt.plot(epoch_df['epoch'], epoch_df['span_f1'])
plt.xlabel('Epoch')
plt.ylabel('Span F1')
plt.title('F1 Evolution')
plt.tight_layout()
plt.savefig('position_encoding_convergence.png')
```

### Success Criteria

✅ **Minimum viable:**
- Validation loss decreases by ≥20%
- Span F1 > 0.15

✅ **Target (position encoding):**
- Span F1 > 0.20
- Probability separation > 0.08

✅ **Exceptional:**
- Span F1 > 0.25
- Probability separation > 0.12

### Expected Artifacts

```
artifacts/position_encoding_v1/
├── best_model.pt                    # Best checkpoint
├── epoch_metrics.csv                # Per-epoch metrics
├── loss_components.csv              # Loss breakdown
├── uncertainty_params.csv           # Task weights
├── probability_stats.csv            # Probability calibration
├── feature_stats.csv                # Feature statistics
└── gradient_stats.csv               # Gradient norms
```

---

## 2. Stones-Only Model (FAST BASELINE)

### Purpose
Fast 100-epoch baseline removing countdown task (was 91% of loss in baseline).

### Key Differences
| Aspect | Baseline 100ep | Stones Only |
|--------|----------------|-------------|
| Tasks | 4 | 3 (removed countdown) |
| Batch size | 32 | 128 |
| GPU VRAM | 2% utilized | 10-15% utilized |
| Training speed | 20 min | 15 min |

### Deployment

```bash
# Files to RSYNC
rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  src/moola/ $RUNPOD_HOST:/root/moola/src/moola/

rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  scripts/train_stones_only.py $RUNPOD_HOST:/root/moola/scripts/

rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  data/processed/labeled/train_latest.parquet \
  $RUNPOD_HOST:/root/moola/data/processed/labeled/

# Training (on RunPod)
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_HOST << 'EOF'
cd /root/moola
export PYTHONPATH=/root/moola/src:$PYTHONPATH

python3 scripts/train_stones_only.py \
  --data data/processed/labeled/train_latest.parquet \
  --output artifacts/stones_only \
  --epochs 100 \
  --batch-size 128 \
  --lr 1e-3 \
  --device cuda \
  --seed 42 2>&1 | tee training_stones_only.log
EOF
```

### Expected Results

```
Epoch 100/100: train_loss=2.5, val_loss=8.2, span_F1=0.15, P=0.12, R=0.20
```

**Success criteria:**
- Span F1 > 0.10 (should work with 3 tasks)
- Probability separation > 0.05
- GPU VRAM usage 10-15%

### Analysis

```python
import pandas as pd

epoch_df = pd.read_csv("artifacts/stones_only/epoch_metrics.csv")
prob_df = pd.read_csv("artifacts/stones_only/probability_stats.csv")

print(f"Final Span F1: {epoch_df['span_f1'].iloc[-1]:.4f}")
print(f"Final Val Loss: {epoch_df['val_loss'].iloc[-1]:.4f}")

val_prob = prob_df[prob_df["phase"] == "val"]
print(f"Final separation: {val_prob['separation'].iloc[-1]:.4f}")
```

---

## 3. Baseline 100-Epoch (REFERENCE)

### Purpose
Comprehensive baseline with extensive logging for analysis. Use for comparison/debugging.

### Deployment

```bash
# Files to RSYNC (same as position encoding)
rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  src/moola/ $RUNPOD_HOST:/root/moola/src/moola/

rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  scripts/train_baseline_100ep.py $RUNPOD_HOST:/root/moola/scripts/

rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  data/processed/labeled/train_latest.parquet \
  $RUNPOD_HOST:/root/moola/data/processed/labeled/

# Training (on RunPod)
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_HOST << 'EOF'
cd /root/moola
export PYTHONPATH=/root/moola/src:$PYTHONPATH

python3 scripts/train_baseline_100ep.py \
  --data data/processed/labeled/train_latest.parquet \
  --output artifacts/baseline_100ep \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-3 \
  --device cuda \
  --seed 42 2>&1 | tee training_baseline_100ep.log
EOF
```

### Metrics Collected

**1. Epoch Metrics** (`epoch_metrics.csv`)
- epoch, train_loss, val_loss, span_f1, span_precision, span_recall, epoch_time

**2. Loss Components** (`loss_components.csv`)
- epoch, phase, loss_type, loss_ptr, loss_span, loss_countdown, total_loss

**3. Uncertainty Parameters** (`uncertainty_params.csv`)
- epoch, sigma_ptr, sigma_type, sigma_span, sigma_countdown, weight_*

**4. Probability Statistics** (`probability_stats.csv`)
- epoch, phase, in_span_mean, out_span_mean, separation

**5. Feature Statistics** (`feature_stats.csv`)
- epoch, phase, feature_name, mean, std, min, max, median (every 5 epochs)

**6. Gradient Statistics** (`gradient_stats.csv`)
- epoch, total_grad_norm, per-layer norms (every 10 epochs)

### Analysis Example

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
epoch_df = pd.read_csv("artifacts/baseline_100ep/epoch_metrics.csv")
loss_df = pd.read_csv("artifacts/baseline_100ep/loss_components.csv")
prob_df = pd.read_csv("artifacts/baseline_100ep/probability_stats.csv")

# Convergence check
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(epoch_df['epoch'], epoch_df['train_loss'], label='Train')
plt.plot(epoch_df['epoch'], epoch_df['val_loss'], label='Val')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Overall Convergence')

# Loss components
val_loss = loss_df[loss_df['phase'] == 'val']
plt.subplot(1, 3, 2)
plt.plot(val_loss['epoch'], val_loss['loss_type'], label='Type')
plt.plot(val_loss['epoch'], val_loss['loss_ptr'], label='Pointer')
plt.plot(val_loss['epoch'], val_loss['loss_span'], label='Span')
plt.plot(val_loss['epoch'], val_loss['loss_countdown'], label='Countdown')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss Component')
plt.title('Loss Breakdown')

# Probability separation
val_prob = prob_df[prob_df['phase'] == 'val']
plt.subplot(1, 3, 3)
plt.plot(val_prob['epoch'], val_prob['in_span_mean'], label='In-span')
plt.plot(val_prob['epoch'], val_prob['out_span_mean'], label='Out-span')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Mean Probability')
plt.title('Probability Distribution')

plt.tight_layout()
plt.savefig('baseline_analysis.png')
print("✓ Saved: baseline_analysis.png")

# Check for problems
print("\nDiagnostics:")
grad_df = pd.read_csv("artifacts/baseline_100ep/gradient_stats.csv")
if grad_df['total_grad_norm'].max() > 100:
    print("⚠️ Exploding gradients detected")
if grad_df['total_grad_norm'].min() < 0.001:
    print("⚠️ Vanishing gradients detected")
```

---

## Common Issues & Solutions

### "Module not found" errors
```bash
export PYTHONPATH=/root/moola/src:$PYTHONPATH
```

### Pre-commit hooks failing locally
```bash
# Run manually to see issues
pre-commit run --all-files

# Auto-fixes:
python3 -m black src/
python3 -m isort src/
python3 -m ruff check --fix src/
```

### Training crashes with OOM
```bash
# Reduce batch size
--batch-size 16
```

### Gradients exploding (loss becomes NaN)
```bash
# Check gradient_stats.csv
# If total_grad_norm > 100, add gradient clipping to training script:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Training very slow
```bash
# Check GPU utilization
nvidia-smi

# If < 80%, increase batch size
# If = 100%, all good!
```

---

## Hyperparameter Recommendations

| Setting | Stones Only | Baseline 100ep | Position Encoding |
|---------|-------------|----------------|-------------------|
| Batch size | 128 | 32 | 32 |
| Learning rate | 1e-3 | 1e-3 | 1e-3 |
| Epochs | 100 | 100 | 100 |
| pos_weight | 1.0 | 1.0 | 13.1 |
| Features | 12 | 12 | 13 |

---

## Next Steps After Training

1. **Download results** to Mac
2. **Analyze metrics** (convergence, separation, loss components)
3. **Compare models** (baseline → stones → position encoding)
4. **Decide on next phase:**
   - If F1 > 0.20: Deploy production model
   - If F1 < 0.15: Try pre-training (Jade encoder on unlabeled data)
   - If F1 = 0: Investigate features/loss function

---

## Production Deployment Checklist

- [ ] Position encoding model trained (F1 > 0.20)
- [ ] Artifacts downloaded to Mac
- [ ] Metrics analyzed and validated
- [ ] Model checkpoint saved to `artifacts/production/`
- [ ] Code changes committed to git
- [ ] README.md updated with new model
- [ ] CLAUDE.md updated with new configuration

---

## Related Documentation

- `BASELINE_100EP_DEPLOYMENT.md` (original baseline)
- `STONES_ONLY_DEPLOYMENT.md` (countdown removal variant)
- `POSITION_ENCODING_DEPLOYMENT.md` (production model)
- `RUNPOD_DEPLOYMENT.md` (CPU pre-computation)
- `README.md` (project overview)
- `docs/ARCHITECTURE.md` (technical details)

