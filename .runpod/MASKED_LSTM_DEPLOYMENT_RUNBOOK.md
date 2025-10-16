# Masked LSTM Pre-training Deployment Runbook

## Quick Start

**One-command full pipeline** (recommended):
```bash
# Run complete pipeline: pre-training → fine-tuning → comparison
.runpod/full_pipeline_masked_lstm.sh 213.173.110.220 36832
```

**Time**: ~50-60 minutes on RTX 4090
**Cost**: ~$0.31 (on-demand) or ~$0.18 (spot)

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step-by-Step Deployment](#step-by-step-deployment)
3. [Monitoring](#monitoring)
4. [Results Analysis](#results-analysis)
5. [Troubleshooting](#troubleshooting)
6. [Performance Benchmarks](#performance-benchmarks)

---

## Prerequisites

### Local Environment
- SSH key: `~/.ssh/id_ed25519` (configured for RunPod)
- Git repo cloned: `moola/`
- Training data: `data/processed/train_pivot_134.parquet`

### RunPod Pod
- **GPU**: RTX 4090 (24GB VRAM)
- **Template**: PyTorch 2.4 (recommended, 97% faster setup)
- **Storage**: 50GB+ recommended
- **Network**: Secure Cloud or Public IP

### Verify Connection
```bash
.runpod/check-connection.sh 213.173.110.220 36832
```

---

## Step-by-Step Deployment

### Option 1: Full Pipeline (Recommended)

**Single command** for complete workflow:
```bash
.runpod/full_pipeline_masked_lstm.sh <HOST> <PORT>
```

**What it does**:
1. Pre-train BiLSTM encoder on 5,000 unlabeled sequences (~30-40 min)
2. Fine-tune SimpleLSTM with pre-trained encoder (~10-15 min)
3. Train baseline SimpleLSTM (no pre-training) for comparison (~10-15 min)
4. Download all results for analysis

**Expected output**:
```
data/artifacts/pretrained/bilstm_encoder.pt           # Pre-trained encoder
data/artifacts/oof/simple_lstm/v1/seed_1337_pretrained.npy  # Fine-tuned predictions
data/artifacts/oof/simple_lstm/v1/seed_42_baseline.npy      # Baseline predictions
```

### Option 2: Pre-training Only

**If you only want pre-trained encoder**:
```bash
.runpod/deploy_pretrain_masked_lstm.sh <HOST> <PORT>
```

**What it does**:
1. Generate unlabeled data (1,000 base + 4x augmentation = 5,000)
2. Upload to RunPod
3. Run masked LSTM pre-training (~30-40 min)
4. Download pre-trained encoder

**Expected output**:
```
data/artifacts/pretrained/bilstm_encoder.pt
```

### Option 3: Manual Step-by-Step

**For debugging or custom workflows**:

#### Step 1: Generate Unlabeled Data
```bash
python scripts/generate_unlabeled_data.py \
    --input data/processed/train_pivot_134.parquet \
    --output data/processed/unlabeled_pretrain.parquet \
    --target-count 1000 \
    --augment-factor 4 \
    --seed 1337
```

**Output**: `data/processed/unlabeled_pretrain.parquet` (~5,000 sequences)

#### Step 2: Upload Data
```bash
POD_HOST="213.173.110.220"
POD_PORT="36832"
SSH_KEY="~/.ssh/id_ed25519"

scp -P $POD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no \
    data/processed/unlabeled_pretrain.parquet \
    root@$POD_HOST:/workspace/data/processed/unlabeled_pretrain.parquet

scp -P $POD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no \
    data/processed/train_pivot_134.parquet \
    root@$POD_HOST:/workspace/data/processed/train.parquet
```

#### Step 3: Setup Environment
```bash
ssh root@$POD_HOST -p $POD_PORT -i $SSH_KEY << 'EOF'
cd /workspace
git clone https://github.com/HeinekenBottle/moola.git
cd moola

python3 -m venv /tmp/moola-venv --system-site-packages
source /tmp/moola-venv/bin/activate
pip install -e .
EOF
```

#### Step 4: Run Pre-training
```bash
ssh root@$POD_HOST -p $POD_PORT -i $SSH_KEY << 'EOF'
cd /workspace/moola
source /tmp/moola-venv/bin/activate

python3 -c '
from pathlib import Path
import numpy as np
import pandas as pd
from moola.pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer

# Load data
df = pd.read_parquet("/workspace/data/processed/unlabeled_pretrain.parquet")
X_unlabeled = np.stack(df["ohlc_sequence"].values)

# Pre-train
pretrainer = MaskedLSTMPretrainer(device="cuda", seed=1337)
history = pretrainer.pretrain(
    X_unlabeled=X_unlabeled,
    n_epochs=50,
    save_path=Path("/workspace/artifacts/pretrained/bilstm_encoder.pt")
)
'
EOF
```

#### Step 5: Download Results
```bash
scp -P $POD_PORT -i $SSH_KEY \
    root@$POD_HOST:/workspace/artifacts/pretrained/bilstm_encoder.pt \
    data/artifacts/pretrained/bilstm_encoder.pt
```

---

## Monitoring

### Real-time Monitoring (Recommended)
```bash
# Continuously monitor training progress
python scripts/monitor_pretraining.py \
    --host 213.173.110.220 \
    --port 36832 \
    --watch
```

**Displays**:
- Training status (active/idle)
- GPU utilization (target: 90-100%)
- VRAM usage (target: 8-12GB / 24GB)
- Temperature
- Recent training logs

### One-time Status Check
```bash
python scripts/monitor_pretraining.py --host 213.173.110.220 --port 36832
```

### Manual GPU Monitoring
```bash
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519 "nvidia-smi"
```

### Check Training Logs
```bash
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519 "tail -f /tmp/training.log"
```

---

## Results Analysis

### Compare Pre-trained vs Baseline
```bash
python scripts/compare_masked_lstm_results.py
```

**Expected improvements**:
- Overall accuracy: +8-12%
- Class 1 recall: 0% → 45-55%
- Balanced accuracy: +10-15%

**Example output**:
```
OVERALL METRICS
--------------------------------------------------------------------------------
Metric                          Pre-trained        Baseline          Δ
--------------------------------------------------------------------------------
Accuracy                            0.7245          0.6532      +0.0713
Balanced Accuracy                   0.6823          0.5641      +0.1182

PER-CLASS METRICS
--------------------------------------------------------------------------------
Class 1:
  Precision                          0.5833          0.0000      +0.5833
  Recall                             0.5250          0.0000      +0.5250
  F1-score                           0.5526          0.0000      +0.5526
```

### Load Pre-trained Encoder in Python
```python
from pathlib import Path
from moola.models import SimpleLSTMModel

# Train with pre-trained encoder
model = SimpleLSTMModel(device="cuda")
model.fit(X_train, y_train)
model.load_pretrained_encoder(
    encoder_path=Path("data/artifacts/pretrained/bilstm_encoder.pt"),
    freeze_encoder=True
)

# Fine-tune
model.fit(X_train, y_train, unfreeze_encoder_after=10)
```

---

## Troubleshooting

### Pre-training Fails (OOM Error)
**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size in `src/moola/config/training_config.py`:
   ```python
   MASKED_LSTM_BATCH_SIZE = 256  # Down from 512
   ```

2. Reduce unlabeled dataset size:
   ```bash
   python scripts/generate_unlabeled_data.py --target-count 500 --augment-factor 3
   ```

3. Reduce sequence length (advanced):
   - Edit `WINDOW_SIZE` in `training_config.py`: `105 → 75`

### Pre-training Too Slow
**Symptom**: >60 minutes for 5,000 sequences

**Check**:
1. GPU utilization:
   ```bash
   ssh root@<HOST> -p <PORT> "nvidia-smi dmon -s u"
   ```
   Expected: 90-100% GPU utilization

2. Increase batch size if VRAM allows:
   ```python
   MASKED_LSTM_BATCH_SIZE = 1024  # Up from 512
   ```

3. Check CPU bottleneck:
   ```bash
   ssh root@<HOST> -p <PORT> "htop"
   ```

### Poor Pre-training Quality
**Symptom**: Validation loss not decreasing

**Solutions**:
1. Increase unlabeled dataset:
   ```bash
   python scripts/generate_unlabeled_data.py --target-count 2000 --augment-factor 5
   ```

2. Increase epochs:
   ```python
   MASKED_LSTM_N_EPOCHS = 100  # Up from 50
   ```

3. Reduce learning rate:
   ```python
   MASKED_LSTM_LEARNING_RATE = 5e-4  # Down from 1e-3
   ```

### Fine-tuning Doesn't Improve Over Baseline
**Symptom**: Pre-trained model ≈ baseline accuracy

**Check**:
1. Verify encoder was loaded:
   - Look for `[SSL PRE-TRAINING] Loading pre-trained encoder` in logs
   - Check encoder file exists and is >1MB

2. Verify encoder was frozen initially:
   - Look for `[SSL PRE-TRAINING] Freezing LSTM encoder weights` in logs

3. Verify unfreezing occurred:
   - Look for `[SSL PRE-TRAINING] Unfreezing LSTM encoder at epoch 11` in logs

4. Try longer pre-training:
   ```python
   MASKED_LSTM_N_EPOCHS = 100  # More pre-training
   ```

### SSH Connection Issues
**Symptom**: `Connection refused` or `Permission denied`

**Solutions**:
1. Verify pod is running:
   - Check RunPod dashboard
   - Pod status should be "Running"

2. Verify connection details:
   ```bash
   .runpod/check-connection.sh <HOST> <PORT>
   ```

3. Check SSH key permissions:
   ```bash
   chmod 600 ~/.ssh/id_ed25519
   ```

---

## Performance Benchmarks

### Pre-training (RTX 4090, 5,000 sequences)
| Metric | Target | Good | Concerning |
|--------|--------|------|------------|
| Time | 30-40 min | <45 min | >60 min |
| GPU Util | 90-100% | >80% | <70% |
| VRAM | 8-12GB | 6-14GB | >18GB or <4GB |
| Temp | 70-80°C | <85°C | >90°C |
| Final Val Loss | 0.001-0.005 | <0.01 | >0.02 |

### Fine-tuning (98 samples, 60 epochs)
| Metric | Target | Good | Concerning |
|--------|--------|------|------------|
| Time | 10-15 min | <20 min | >30 min |
| Accuracy | 70-75% | >68% | <65% |
| Class 1 Recall | 45-55% | >40% | <30% |
| Improvement vs Baseline | +8-12% | >+5% | <+3% |

### Cost Analysis
| Configuration | Time | Cost (on-demand) | Cost (spot) |
|---------------|------|------------------|-------------|
| Full pipeline | ~55 min | ~$0.31 | ~$0.18 |
| Pre-training only | ~35 min | ~$0.20 | ~$0.12 |
| Fine-tuning only | ~15 min | ~$0.09 | ~$0.05 |

---

## Advanced: Hyperparameter Tuning

### Pre-training Hyperparameters
```python
# src/moola/config/training_config.py

# Architecture
MASKED_LSTM_HIDDEN_DIM = 128      # ↑ for more capacity, ↓ for speed
MASKED_LSTM_NUM_LAYERS = 2        # ↑ for depth, ↓ for speed
MASKED_LSTM_DROPOUT = 0.2         # ↑ for regularization

# Masking strategy
MASKED_LSTM_MASK_RATIO = 0.15     # ↑ for harder task (0.15-0.30)
MASKED_LSTM_MASK_STRATEGY = "patch"  # "random", "block", or "patch"
MASKED_LSTM_PATCH_SIZE = 7        # Patch size (if using "patch")

# Training
MASKED_LSTM_N_EPOCHS = 50         # ↑ for better pre-training
MASKED_LSTM_LEARNING_RATE = 1e-3  # ↓ if training unstable
MASKED_LSTM_BATCH_SIZE = 512      # ↑ if VRAM allows, ↓ if OOM
```

### Fine-tuning Hyperparameters
```python
# Transfer learning
MASKED_LSTM_FREEZE_EPOCHS = 10              # Epochs to keep encoder frozen
MASKED_LSTM_UNFREEZE_LR_REDUCTION = 0.5     # LR multiplier after unfreezing
```

---

## Next Steps

After successful deployment:

1. **Experiment with larger datasets**:
   ```bash
   python scripts/generate_unlabeled_data.py --target-count 2000 --augment-factor 8
   ```

2. **Try different masking strategies**:
   - Edit `MASKED_LSTM_MASK_STRATEGY` in `training_config.py`
   - Options: `"random"`, `"block"`, `"patch"`

3. **Fine-tune other models**:
   - Adapt pre-trained encoder for CNN-Transformer
   - Try different downstream architectures

4. **Deploy to production**:
   - Export pre-trained encoder
   - Integrate into inference pipeline
   - Monitor performance on live data

---

## Support

**Issues**: Create issue in GitHub repo with:
- Full error message
- Pod specs (GPU, VRAM, template)
- Deployment script used
- Relevant logs

**Quick help**:
- Check `TROUBLESHOOTING.md` in `.runpod/`
- Review `RTX_4090_OPTIMIZATION_GUIDE.md`
- Monitor with `scripts/monitor_pretraining.py --watch`
