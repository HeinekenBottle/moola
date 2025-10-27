# Baseline 100-Epoch Training - Deployment Guide

**Purpose**: Establish comprehensive baseline with extensive logging for surgical analysis
**Duration**: ~15-20 minutes on RTX 4090 GPU (100 epochs × 210 samples)
**Expected outcome**: F1 0.14 → 0.40-0.55 (baseline without pre-training)

---

## Pre-Deployment Checklist

### 1. **Files to RSYNC to RunPod**

```bash
# Core source code
src/moola/models/jade_core.py
src/moola/features/relativity.py
src/moola/features/zigzag.py
src/moola/data/

# Training script
scripts/train_baseline_100ep.py

# Data
data/processed/labeled/train_latest_overlaps_v2.parquet

# Configuration (if needed)
# configs/
```

### 2. **RSYNC Commands** (from Mac)

**Get SSH details from user first**, then:

```bash
# Set RunPod connection (REPLACE WITH ACTUAL VALUES)
RUNPOD_HOST="root@YOUR_IP"
RUNPOD_PORT="YOUR_PORT"
SSH_KEY="~/.ssh/id_ed25519"

# RSYNC source code
rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  src/ $RUNPOD_HOST:/root/moola/src/

# RSYNC training script
rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  scripts/train_baseline_100ep.py $RUNPOD_HOST:/root/moola/scripts/

# RSYNC data
rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  data/processed/labeled/train_latest_overlaps_v2.parquet \
  $RUNPOD_HOST:/root/moola/data/processed/labeled/

# Verify files
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_HOST << 'EOF'
cd /root/moola
echo "=== Source files ==="
ls -lh src/moola/models/jade_core.py
ls -lh src/moola/features/relativity.py
echo "=== Training script ==="
ls -lh scripts/train_baseline_100ep.py
echo "=== Data ==="
ls -lh data/processed/labeled/train_latest_overlaps_v2.parquet
EOF
```

---

## Training Execution

### **On RunPod (via SSH)**

```bash
# SSH into RunPod
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_HOST

# Navigate to moola
cd /root/moola

# Set PYTHONPATH
export PYTHONPATH=/root/moola/src:$PYTHONPATH

# Verify Python environment
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Run training (100 epochs, full logging)
python3 scripts/train_baseline_100ep.py \
  --data data/processed/labeled/train_latest_overlaps_v2.parquet \
  --output artifacts/baseline_100ep \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-3 \
  --device cuda \
  --seed 42 2>&1 | tee training_baseline_100ep.log
```

**Expected output structure**:
```
================================================================================
BASELINE TRAINING RUN - 100 EPOCHS WITH COMPREHENSIVE LOGGING
================================================================================
Data: data/processed/labeled/train_latest_overlaps_v2.parquet
Output: artifacts/baseline_100ep
Epochs: 100
Device: cuda
Seed: 42

Loading data...
Train: 168 samples, Val: 42 samples

Creating model...
Parameters: 97,547 total, 97,547 trainable
✓ Saved metadata: artifacts/baseline_100ep/metadata.json

Training for 100 epochs...
--------------------------------------------------------------------------------
Epoch   1/100: train_loss=8.5065, val_loss=17.6855, span_F1=0.000, ...
Epoch   2/100: train_loss=7.8234, val_loss=17.2341, span_F1=0.000, ...
...
```

---

## Monitoring During Training

### **Option 1: Live Monitoring (from Mac)**

While training runs, monitor from Mac:

```bash
# Stream training log
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_HOST \
  "tail -f /root/moola/training_baseline_100ep.log"

# Check GPU utilization
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_HOST \
  "watch -n 1 nvidia-smi"
```

### **Option 2: Retrieve Partial Metrics**

```bash
# Download partial metrics (while training is still running)
rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  $RUNPOD_HOST:/root/moola/artifacts/baseline_100ep/*.csv ./artifacts/baseline_100ep/

# Analyze partial results locally
python3 scripts/monitor_baseline.py
```

---

## Post-Training Retrieval

### **RSYNC Results Back to Mac**

```bash
# Download all training artifacts
rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  $RUNPOD_HOST:/root/moola/artifacts/baseline_100ep/ \
  ./artifacts/baseline_100ep/

# Download training log
rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  $RUNPOD_HOST:/root/moola/training_baseline_100ep.log ./
```

---

## Expected Artifacts

After training completes, you'll have:

```
artifacts/baseline_100ep/
├── metadata.json                    # Training configuration and dataset info
├── best_model.pt                    # Best model checkpoint (lowest val loss)
├── checkpoint_epoch_10.pt           # Checkpoint at epoch 10
├── checkpoint_epoch_20.pt           # Checkpoint at epoch 20
├── ...
├── checkpoint_epoch_100.pt          # Final checkpoint
├── epoch_metrics.csv                # Per-epoch: train_loss, val_loss, F1, precision, recall
├── loss_components.csv              # Per-epoch: loss_type, loss_ptr, loss_span, loss_countdown
├── uncertainty_params.csv           # Per-epoch: σ_ptr, σ_type, σ_span, σ_countdown, task weights
├── probability_stats.csv            # Per-epoch: in_span_mean, out_span_mean, separation
├── feature_stats.csv                # Per-feature per-epoch: mean, std, min, max (every 5 epochs)
└── gradient_stats.csv               # Per-layer gradient norms (every 10 epochs)

training_baseline_100ep.log          # Full console output
```

---

## Metrics Collected

### **1. Epoch Metrics** (`epoch_metrics.csv`)

| Column | Description |
|--------|-------------|
| epoch | Epoch number (1-100) |
| train_loss | Training loss (uncertainty-weighted) |
| val_loss | Validation loss |
| span_f1 | Span F1 score @ threshold 0.5 |
| span_precision | Span precision |
| span_recall | Span recall |
| epoch_time | Training time per epoch (seconds) |

**Use for**: Overall training trajectory, convergence patterns

---

### **2. Loss Components** (`loss_components.csv`)

| Column | Description |
|--------|-------------|
| epoch | Epoch number |
| phase | "train" or "val" |
| loss_type | Classification loss (cross-entropy) |
| loss_ptr | Pointer loss (Huber, δ=0.08) |
| loss_span | Span loss (soft span loss) |
| loss_countdown | Countdown loss (Huber, δ=1.0) |
| total_loss | Sum of all components |

**Use for**: Understanding which loss components dominate/plateau/improve

**Example analysis**:
```python
import pandas as pd
import matplotlib.pyplot as plt

loss_df = pd.read_csv("artifacts/baseline_100ep/loss_components.csv")
val_loss = loss_df[loss_df["phase"] == "val"]

plt.figure(figsize=(12, 6))
plt.plot(val_loss["epoch"], val_loss["loss_type"], label="Classification")
plt.plot(val_loss["epoch"], val_loss["loss_ptr"], label="Pointers")
plt.plot(val_loss["epoch"], val_loss["loss_span"], label="Span")
plt.plot(val_loss["epoch"], val_loss["loss_countdown"], label="Countdown")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Component Evolution (Validation)")
plt.savefig("loss_components.png")
```

---

### **3. Uncertainty Parameters** (`uncertainty_params.csv`)

| Column | Description |
|--------|-------------|
| epoch | Epoch number (every 5 epochs) |
| sigma_ptr | Uncertainty parameter for pointers |
| sigma_type | Uncertainty parameter for classification |
| sigma_span | Uncertainty parameter for span |
| sigma_countdown | Uncertainty parameter for countdown |
| weight_ptr | Effective task weight for pointers (%) |
| weight_type | Effective task weight for classification (%) |
| weight_span | Effective task weight for span (%) |
| weight_countdown | Effective task weight for countdown (%) |

**Use for**: Understanding how task importance evolves during training

**Expected pattern** (from previous runs):
- Epoch 1: Random initialization (σ ≈ 1.0 for all)
- Epoch 50: Pointers dominant (weight ≈ 46%), span secondary (weight ≈ 25%)
- Epoch 100: Further specialization (check if weights stabilize)

---

### **4. Probability Statistics** (`probability_stats.csv`)

| Column | Description |
|--------|-------------|
| epoch | Epoch number |
| phase | "train" or "val" |
| in_span_mean | Mean predicted probability for in-expansion positions |
| in_span_std | Std deviation for in-expansion predictions |
| out_span_mean | Mean predicted probability for out-of-expansion positions |
| out_span_std | Std deviation for out-of-expansion predictions |
| separation | in_span_mean - out_span_mean (KEY METRIC) |

**Use for**: Tracking probability calibration evolution

**Critical metric**: `separation` - how well model distinguishes expansions
- Epoch 1: Expected ≈ 0.000 (random)
- Epoch 50: Previous run = 0.000 (weak)
- Epoch 100: **Target > 0.10** (10% separation = viable for thresholding)

---

### **5. Feature Statistics** (`feature_stats.csv`)

| Column | Description |
|--------|-------------|
| epoch | Epoch number (every 5 epochs) |
| phase | "train" or "val" |
| feature_name | Name of feature (12 features) |
| mean | Mean value across all windows |
| std | Standard deviation |
| min | Minimum value |
| max | Maximum value |
| median | Median value |

**Use for**: Detecting feature drift, saturation, or collapse during training

**Example check**:
```python
feat_df = pd.read_csv("artifacts/baseline_100ep/feature_stats.csv")
expansion_proxy_val = feat_df[
    (feat_df["feature_name"] == "expansion_proxy") &
    (feat_df["phase"] == "val")
]

# Check if feature drifts during training
print(expansion_proxy_val[["epoch", "mean", "std"]])
```

---

### **6. Gradient Statistics** (`gradient_stats.csv`)

| Column | Description |
|--------|-------------|
| epoch | Epoch number (every 10 epochs) |
| total_grad_norm | L2 norm of all gradients |
| lstm.weight_ih_l0 | Gradient norm for LSTM input-hidden weights |
| lstm.weight_hh_l0 | Gradient norm for LSTM hidden-hidden weights |
| ... | (all other model parameters) |

**Use for**: Detecting vanishing/exploding gradients, identifying problematic layers

**Red flags**:
- `total_grad_norm > 100`: Exploding gradients
- `total_grad_norm < 0.001`: Vanishing gradients
- Specific layer norm >> others: That layer is bottleneck

---

## Analysis After Training

### **Quick Sanity Checks**

```python
import pandas as pd
import matplotlib.pyplot as plt

# 1. Check convergence
epoch_df = pd.read_csv("artifacts/baseline_100ep/epoch_metrics.csv")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epoch_df["epoch"], epoch_df["train_loss"], label="Train")
plt.plot(epoch_df["epoch"], epoch_df["val_loss"], label="Val")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Convergence")

plt.subplot(1, 2, 2)
plt.plot(epoch_df["epoch"], epoch_df["span_f1"])
plt.xlabel("Epoch")
plt.ylabel("Span F1")
plt.title("Span F1 Evolution")
plt.tight_layout()
plt.savefig("convergence_summary.png")
print("✓ Saved: convergence_summary.png")

# 2. Check probability separation
prob_df = pd.read_csv("artifacts/baseline_100ep/probability_stats.csv")
val_prob = prob_df[prob_df["phase"] == "val"]

print("\nProbability Separation Evolution:")
print(val_prob[["epoch", "in_span_mean", "out_span_mean", "separation"]].tail(10))

# 3. Check final task weights
uncert_df = pd.read_csv("artifacts/baseline_100ep/uncertainty_params.csv")
final_weights = uncert_df.iloc[-1]

print("\nFinal Task Weights:")
print(f"  Pointers:       {final_weights['weight_ptr']:.1%}")
print(f"  Classification: {final_weights['weight_type']:.1%}")
print(f"  Span:           {final_weights['weight_span']:.1%}")
print(f"  Countdown:      {final_weights['weight_countdown']:.1%}")
```

---

## Expected Training Timeline

| Time | Epoch | Event |
|------|-------|-------|
| T+0min | 1 | Training starts, random initialization |
| T+1min | 10 | First checkpoint saved |
| T+2min | 20 | Second checkpoint saved |
| T+5min | 50 | Mid-training checkpoint (compare to previous 50-epoch run) |
| T+10min | 100 | Training completes, all metrics saved |

**Typical epoch time**: ~6-10 seconds on RTX 4090 (210 samples, batch_size=32)

---

## Success Criteria

### **Minimum viable baseline**:
- ✅ Training completes without errors
- ✅ Validation loss decreases by ≥ 20% (e.g., 17.68 → 14.0)
- ✅ Span F1 improves from 0.000 (all files generated)

### **Strong baseline** (target):
- ✅ Validation loss decreases by ≥ 30% (e.g., 17.68 → 12.4)
- ✅ Span F1 > 0.30 by epoch 100
- ✅ Probability separation > 0.10 (in_span_mean - out_span_mean)

### **Exceptional baseline** (stretch goal):
- ✅ Validation loss decreases by ≥ 40%
- ✅ Span F1 > 0.45
- ✅ Probability separation > 0.15

---

## Troubleshooting

### **Training crashes with OOM (Out of Memory)**

```bash
# Reduce batch size
python3 scripts/train_baseline_100ep.py ... --batch-size 16
```

### **Gradients exploding (loss becomes NaN)**

Check `gradient_stats.csv` - if `total_grad_norm > 100`, add gradient clipping:

```python
# In train_baseline_100ep.py, after loss.backward():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### **Training very slow**

- Check GPU utilization: `nvidia-smi`
- If GPU util < 80%, increase batch size
- If GPU util = 100%, all good!

---

## Next Steps After Baseline

1. **Analyze metrics** - Use analysis scripts above
2. **Compare to 50-epoch run** - Check if extended training helps
3. **Decide on pre-training** - If F1 < 0.40, consider improved features + pre-training
4. **Surgical improvements** - Use feature_stats, gradient_stats to identify bottlenecks

---

**Ready to deploy?** Provide SSH details and I'll prep the RSYNC commands! 🚀
