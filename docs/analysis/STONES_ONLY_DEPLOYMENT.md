# Stones-Only Training - Fast Baseline (No Countdown)

**Purpose**: Fast baseline with optimized GPU utilization and comprehensive metrics
**Duration**: ~15 minutes on RTX 4090 (100 epochs × 210 samples, batch_size=128)
**Key change**: Removed countdown task (was 91% of loss in baseline_100ep)

---

## What's Different from baseline_100ep

| Aspect | Baseline 100ep | Stones Only |
|--------|----------------|-------------|
| **Tasks** | 4 (type, ptr, span, countdown) | **3 (type, ptr, span)** |
| **Batch size** | 32 | **128 (4x larger)** |
| **GPU VRAM** | ~2% utilized | **~10-15% (5x better)** |
| **Training speed** | ~20 minutes | **~15 minutes (25% faster)** |
| **Countdown loss** | 10.08 (91% of total) | **REMOVED** |
| **Expected F1** | 0.0000 (failed) | **>0.10 (should work)** |

---

## Quick Deployment to RunPod

### 1. RSYNC Files

```bash
# Set RunPod connection (REPLACE WITH YOUR VALUES)
RUNPOD_HOST="root@YOUR_IP"
RUNPOD_PORT="YOUR_PORT"
SSH_KEY="~/.ssh/id_ed25519"

# Create directories
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_HOST \
  'mkdir -p /root/moola/src/moola /root/moola/scripts /root/moola/data/processed/labeled /root/moola/artifacts'

# RSYNC source code
rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  src/moola/ $RUNPOD_HOST:/root/moola/src/moola/

# RSYNC new training script
rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  scripts/train_stones_only.py $RUNPOD_HOST:/root/moola/scripts/

# RSYNC data
rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  data/processed/labeled/train_latest_overlaps_v2.parquet \
  $RUNPOD_HOST:/root/moola/data/processed/labeled/
```

### 2. Install Dependencies (if needed)

```bash
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_HOST << 'EOF'
pip3 install matplotlib pandas numpy scikit-learn seaborn pyarrow pydantic torch
EOF
```

### 3. Run Training

```bash
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_HOST << 'EOF'
cd /root/moola
export PYTHONPATH=/root/moola/src:$PYTHONPATH

# Run with larger batch size for better GPU utilization
nohup python3 scripts/train_stones_only.py \
  --data data/processed/labeled/train_latest_overlaps_v2.parquet \
  --output artifacts/stones_only \
  --epochs 100 \
  --batch-size 128 \
  --lr 1e-3 \
  --device cuda \
  --seed 42 > training_stones_only.log 2>&1 &

echo "Training started! PID: $!"
EOF
```

---

## Monitoring During Training

### Check Progress

```bash
# Check if running
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_HOST \
  'ps aux | grep train_stones_only | grep -v grep'

# Check latest epoch
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_HOST \
  'tail -20 /root/moola/training_stones_only.log'

# Check GPU utilization (should be ~10-15% now, not 2%)
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_HOST \
  'nvidia-smi'
```

---

## Expected Results

### Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Span F1** | >0.10 | Model predicts SOME expansions (not all zeros) |
| **Probability Separation** | >0.05 | In-span > Out-span by viable margin |
| **Val Loss** | <10.0 | Better than baseline (11.05) |
| **GPU VRAM** | ~10-15% | 5x better utilization than baseline |

### What Success Looks Like

```
Epoch 100/100: train_loss=2.5, val_loss=8.2, span_F1=0.15, P=0.12, R=0.20
```

**Interpretation**:
- F1 = 0.15 → Model is predicting expansions (not all zeros like baseline)
- Separation > 0.05 → Probabilities are viable for thresholding
- Loss < 10 → Better than baseline without countdown interference

### What Failure Looks Like (Same as Baseline)

```
Epoch 100/100: train_loss=1.9, val_loss=11.0, span_F1=0.000, P=0.000, R=0.000
```

**If this happens**: Features are insufficient, need better feature engineering

---

## After Training Completes

### Download Results

```bash
# Download all artifacts
rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  $RUNPOD_HOST:/root/moola/artifacts/stones_only/ \
  ./artifacts/stones_only/

# Download log
rsync -avz --progress \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
  $RUNPOD_HOST:/root/moola/training_stones_only.log ./
```

### Quick Analysis

```bash
python3 << 'EOF'
import pandas as pd

# Load metrics
epoch_df = pd.read_csv("artifacts/stones_only/epoch_metrics.csv")
prob_df = pd.read_csv("artifacts/stones_only/probability_stats.csv")
loss_df = pd.read_csv("artifacts/stones_only/loss_components.csv")

print("=" * 80)
print("STONES-ONLY RESULTS SUMMARY")
print("=" * 80)
print()

# Overall performance
print("📊 PERFORMANCE:")
print(f"  Final Val Loss:  {epoch_df['val_loss'].iloc[-1]:.4f}")
print(f"  Final Span F1:   {epoch_df['span_f1'].iloc[-1]:.4f}")
print(f"  Final Precision: {epoch_df['span_precision'].iloc[-1]:.4f}")
print(f"  Final Recall:    {epoch_df['span_recall'].iloc[-1]:.4f}")
print()

# Probability separation
val_prob = prob_df[prob_df["phase"] == "val"]
final_sep = val_prob['separation'].iloc[-1]
print("🎯 PROBABILITY SEPARATION:")
print(f"  In-span mean:  {val_prob['in_span_mean'].iloc[-1]:.4f}")
print(f"  Out-span mean: {val_prob['out_span_mean'].iloc[-1]:.4f}")
print(f"  Separation:    {final_sep:.4f} {'✅' if final_sep > 0.05 else '❌ (need >0.05)'}")
print()

# Loss breakdown
val_loss = loss_df[loss_df["phase"] == "val"]
final_loss = val_loss[val_loss["epoch"] == 100].iloc[0]
print("🔍 LOSS BREAKDOWN (Final Epoch):")
print(f"  Classification: {final_loss['loss_type']:.4f}")
print(f"  Pointers:       {final_loss['loss_ptr']:.4f}")
print(f"  Span:           {final_loss['loss_span']:.4f}")
print(f"  Total:          {final_loss['total_loss']:.4f}")
print()

# Success check
print("✅ SUCCESS CHECK:")
if epoch_df['span_f1'].iloc[-1] > 0.10:
    print("  ✅ Span F1 > 0.10 - Model is learning!")
else:
    print("  ❌ Span F1 = 0 - Still failing (features insufficient)")

if final_sep > 0.05:
    print("  ✅ Separation > 0.05 - Probabilities viable")
else:
    print("  ❌ Separation < 0.05 - Probabilities not separated")

EOF
```

---

## Speed Optimizations Applied

### 1. Removed Countdown Task
- **Problem**: Countdown loss = 10.08 (91% of total loss)
- **Solution**: Remove countdown entirely
- **Impact**: Model can focus on span detection

### 2. Increased Batch Size
- **Before**: 32 (GPU 2% utilized)
- **After**: 128 (GPU ~10-15% utilized)
- **Impact**: 4x faster batches, 25% faster overall training

### 3. Why Not Larger Batch?
- Dataset: 168 train samples → 128 batch = 2 batches per epoch
- Larger batch (e.g., 256) would be 1 batch per epoch (too coarse for learning)
- 128 is optimal balance: GPU utilization vs. gradient updates

---

## Optional: Run Multiple Configs in Parallel

If you want to sweep hyperparameters:

```bash
# Run 3 experiments simultaneously (different learning rates)
ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_HOST << 'EOF'
cd /root/moola
export PYTHONPATH=/root/moola/src:$PYTHONPATH

# LR 1e-4
nohup python3 scripts/train_stones_only.py \
  --data data/processed/labeled/train_latest_overlaps_v2.parquet \
  --output artifacts/stones_lr1e4 \
  --epochs 100 --batch-size 128 --lr 1e-4 --device cuda \
  > training_lr1e4.log 2>&1 &

# LR 1e-3
nohup python3 scripts/train_stones_only.py \
  --data data/processed/labeled/train_latest_overlaps_v2.parquet \
  --output artifacts/stones_lr1e3 \
  --epochs 100 --batch-size 128 --lr 1e-3 --device cuda \
  > training_lr1e3.log 2>&1 &

# LR 1e-2
nohup python3 scripts/train_stones_only.py \
  --data data/processed/labeled/train_latest_overlaps_v2.parquet \
  --output artifacts/stones_lr1e2 \
  --epochs 100 --batch-size 128 --lr 1e-2 --device cuda \
  > training_lr1e2.log 2>&1 &

echo "3 parallel experiments started!"
EOF
```

**Why this works**: Small dataset + GPU has 24GB VRAM → can fit 3 models simultaneously

**Runtime**: Same ~15 minutes (parallel execution)

---

## Next Steps After Baseline

1. **If F1 > 0.10**: Try pre-training (Jade encoder on unlabeled data)
2. **If F1 = 0**: Improve features (momentum, acceleration, pattern-based)
3. **Compare to baseline_100ep**: Verify countdown removal helped

---

**Ready to deploy?** Provide SSH details and I'll prep the RSYNC commands! 🚀
