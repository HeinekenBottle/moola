# Getting Started with Moola

Complete setup and first-experiment guide for moola.

## Prerequisites

- Python 3.10+ (check: `python --version`)
- pip package manager
- SSH key for RunPod (optional, only if using RunPod)
- CUDA 11.8 GPU (optional, CPU works but training is slow)

## One-Time Setup

### 1. Install Dependencies

```bash
cd ~/projects/moola
pip install -r requirements.txt
```

**On RunPod:** Dependencies should already be installed. Skip this step.

### 2. Install Pre-commit Hooks

Pre-commit hooks automatically format and lint your code before commits (Black, Ruff, isort).

```bash
# On Mac
pip install pre-commit==4.3.0
pre-commit install

# Verify installation
pre-commit run --all-files  # Should pass all checks
```

**On RunPod:** You don't commit here, so skip this. Pre-commit is enforced on Mac only.

### 3. (Optional) Generate SSH Key for RunPod

If you'll use RunPod for training:

```bash
# Generate SSH key (one time)
ssh-keygen -t ed25519 -f ~/.ssh/runpod_key -N ""

# Copy public key to RunPod (in RunPod UI: Settings → SSH Public Keys)
cat ~/.ssh/runpod_key.pub
```

## Running Your First Experiment

### Option A: Local Training (CPU - Slow but works)

```bash
# Train SimpleLSTM baseline for 10 epochs
python -m moola.cli train \
  --model simple_lstm \
  --n-epochs 10 \
  --batch-size 32 \
  --device cpu

# Output:
# Epoch 1/10: loss=0.65, accuracy=0.58
# Epoch 2/10: loss=0.42, accuracy=0.72
# ...
# ✓ Model saved to models/simple_lstm_latest.pt
```

**Expected time:** 3-5 minutes on CPU (will be much slower than GPU)

### Option B: RunPod GPU Training (Fast)

#### Step 1: SSH into RunPod

```bash
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_RUNPOD_IP

# Once connected:
cd /workspace/moola
```

#### Step 2: Run Experiment

```bash
# Train SimpleLSTM with GPU
python -m moola.cli train \
  --model simple_lstm \
  --n-epochs 60 \
  --batch-size 1024 \
  --device cuda

# Output:
# Epoch 1/60: loss=0.68, accuracy=0.55, val_accuracy=0.52
# Epoch 2/60: loss=0.42, accuracy=0.74, val_accuracy=0.71
# ...
# ✓ Model saved
```

**Expected time:** 6-8 minutes on GPU (RTX 4090) vs 2+ hours on CPU

#### Step 3: Get Results Back to Mac

**In a new terminal on your Mac (keep SSH terminal open):**

```bash
# Copy results while training or after it completes
scp -i ~/.ssh/runpod_key ubuntu@YOUR_IP:/workspace/moola/experiment_results.jsonl ./

# View results
cat experiment_results.jsonl | tail -1 | python -m json.tool
```

### Option C: Pre-training + Fine-tuning Pipeline

This is the full workflow: pre-train on unlabeled data, then fine-tune on labeled data.

```bash
# SSH into RunPod
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP
cd /workspace/moola

# Step 1: Pre-train BiLSTM encoder (50 epochs)
python -m moola.cli pretrain-bilstm \
  --n-epochs 50 \
  --masking-strategy random \
  --device cuda

# Step 2: Fine-tune SimpleLSTM with pre-trained encoder (60 epochs)
python -m moola.cli train \
  --model simple_lstm \
  --n-epochs 60 \
  --use-pretrained-encoder \
  --device cuda
```

**Expected time:** 25-30 minutes total

## Expected File Structure

### On Your Mac

```
~/projects/moola/
├── README.md                    # Project overview
├── src/moola/                   # Source code
├── tests/                       # Tests
├── docs/                        # Documentation
├── WORKFLOW_SSH_SCP_GUIDE.md   # How to use SSH/SCP
├── .pre-commit-config.yaml     # Git hooks
├── requirements.txt            # Dependencies
└── experiment_results.jsonl    # Results (SCPed from RunPod)
```

### On RunPod (/workspace/moola)

```
/workspace/moola/
├── src/moola/                   # Source code (same as Mac)
├── data/                        # Training data
│   ├── raw/                    # Raw OHLC data
│   ├── processed/              # Processed windows
│   └── train.parquet           # Labeled dataset (98 samples)
├── models/                      # Saved checkpoints
│   └── simple_lstm_latest.pt   # Latest model
├── experiment_results.jsonl    # Results (append-only)
└── mlruns/                     # MLflow artifacts (if using)
```

## Getting Results Back to Mac

### After Each Experiment

```bash
# While training is still running (or after it completes):
scp -i ~/.ssh/runpod_key ubuntu@YOUR_IP:/workspace/moola/experiment_results.jsonl ./

# View the results
cat experiment_results.jsonl | tail -5  # Last 5 results
```

### After All Experiments

```bash
# Archive results by phase
scp -i ~/.ssh/runpod_key ubuntu@YOUR_IP:/workspace/moola/experiment_results.jsonl ./phase1_results.jsonl

# Compare all results
python -c "
import json
results = [json.loads(line) for line in open('phase1_results.jsonl')]
best = max(results, key=lambda x: x['metrics'].get('accuracy', 0))
print(f\"Best result: {best['experiment_id']}\")
print(f\"Accuracy: {best['metrics']['accuracy']:.4f}\")
"
```

### Getting Saved Models Back

```bash
# Copy trained model to Mac
scp -i ~/.ssh/runpod_key ubuntu@YOUR_IP:/workspace/moola/models/simple_lstm_latest.pt ./
```

## Troubleshooting

### Pre-commit Hook Failures

**Error:** `Black would reformat your files` or `Ruff found issues`

**Fix:**
```bash
# Let pre-commit auto-fix the issues
pre-commit run --all-files

# Or manually fix and retry commit
git add .
git commit -m "Your message"
```

### SSH Connection Refused

**Error:** `ssh: connect to host ... port 22: Connection refused`

**Fix:**
- Check RunPod is running: Go to RunPod UI and verify instance is "Running"
- Check SSH key path: `ssh -i ~/.ssh/runpod_key ubuntu@IP`
- Test connection: `ssh -i ~/.ssh/runpod_key ubuntu@IP echo hello`

### GPU Out of Memory (OOM)

**Error:** `CUDA out of memory. Tried to allocate X.XX GiB`

**Fix:**
```bash
# Reduce batch size
python -m moola.cli train \
  --model simple_lstm \
  --batch-size 256 \  # Reduced from 1024
  --device cuda
```

### SCP "No such file or directory"

**Error:** `experiment_results.jsonl: No such file or directory`

**Fix:**
- Wait for training to complete (results file is created after first epoch)
- Check file exists on RunPod: `ssh ubuntu@IP ls /workspace/moola/experiment_results.jsonl`

### Training Very Slow on CPU

**Issue:** CPU training takes 30+ minutes for 60 epochs

**Expected behavior:** CPU training is 10-100x slower than GPU. This is normal.

**Solution:** Use RunPod GPU (much faster) or reduce epochs for testing.

### "Command not found: moola.cli"

**Error:** `python -m moola.cli: No module named moola.cli`

**Fix:**
```bash
# Make sure you're in the right directory
cd ~/projects/moola  # or /workspace/moola on RunPod

# Make sure moola is installed
pip install -e .  # Installs in editable mode
```

## Next Steps

1. **Understand the architecture:** See [`ARCHITECTURE.md`](ARCHITECTURE.md)
2. **Learn the workflow:** See [`WORKFLOW_SSH_SCP_GUIDE.md`](../WORKFLOW_SSH_SCP_GUIDE.md)
3. **Explore pre-training:** See [`PRETRAINING_ORCHESTRATION_GUIDE.md`](../PRETRAINING_ORCHESTRATION_GUIDE.md)
4. **Check the code:** Start with `src/moola/cli.py` to see available commands

---

**Questions?** Check the Troubleshooting section above, or read the detailed guides linked.
