# Moola RunPod Deployment Guide

Complete guide for setting up and running Moola training on RunPod cloud GPUs.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Initial Setup (One-Time)](#initial-setup-one-time)
3. [Running Training](#running-training)
4. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
5. [Best Practices](#best-practices)

---

## Prerequisites

### 1. RunPod Account & API Keys

1. Create account at https://www.runpod.io
2. Go to https://www.runpod.io/console/user/settings
3. Navigate to "S3 Access Keys" section
4. Create new S3 API key pair
5. Save credentials (these are sensitive - never commit to git):
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

### 2. Local Environment Setup

Add RunPod credentials to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
# Add these lines to ~/.zshrc or ~/.bashrc
export AWS_ACCESS_KEY_ID="your-runpod-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-runpod-secret-access-key"
export AWS_DEFAULT_REGION="eu-ro-1"

# Reload configuration
source ~/.zshrc  # or source ~/.bashrc
```

### 3. AWS CLI Installation

```bash
# macOS
brew install awscli

# Or use pip (any platform)
pip install awscli
```

---

## Initial Setup (One-Time)

### Step 1: Configure Network Storage Credentials

Edit `/Users/jack/projects/moola/.runpod/network-storage.env`:

```bash
cd /Users/jack/projects/moola/.runpod

# Edit network-storage.env with your credentials
# Set:
#   AWS_ACCESS_KEY_ID=<your-key>
#   AWS_SECRET_ACCESS_KEY=<your-secret>
#   S3_BUCKET=hg878tp14w
#   S3_ENDPOINT=https://s3api-eu-ro-1.runpod.io
#   AWS_REGION=eu-ro-1
```

### Step 2: Upload Initial Data to Network Storage

```bash
cd /Users/jack/projects/moola/.runpod

# Load credentials
source network-storage.env

# Make scripts executable
chmod +x sync-to-storage.sh sync-from-storage.sh

# Upload all files (data, configs, scripts)
./sync-to-storage.sh all

# Expected output:
# 📤 Syncing deployment scripts...
# ✅ deployment scripts synced
# 📤 Syncing processed datasets...
# ✅ processed datasets synced
# ... (more sections)
```

---

## Running Training

### Step 1: Start a RunPod Pod

1. Go to https://www.runpod.io/console/pods
2. Click "Deploy" or "Rent GPU"
3. Configure pod:
   - **Template**: PyTorch 2.x (CUDA 12.x)
   - **GPU**: RTX 4090 or A100 (budget-dependent)
   - **Network Volume**: Select "moola" (ID: hg878tp14w)
   - **Container Disk**: Minimum 50 GB
   - **Data Center**: eu-ro-1 (Romania) for lowest latency
4. Wait for pod status to show "Running"
5. Click pod name to view SSH connection details

### Step 2: Initialize Pod Environment

SSH into your pod:

```bash
# SSH connection (replace POD_ID with your actual pod ID)
ssh -i ~/.ssh/id_rsa root@<your-runpod-ip>
```

On the pod, initialize the environment:

```bash
# Download startup script from network storage
aws s3 cp \
    --region eu-ro-1 \
    --endpoint-url https://s3api-eu-ro-1.runpod.io \
    s3://hg878tp14w/scripts/pod-startup.sh \
    /tmp/pod-startup.sh

# Run startup script
bash /tmp/pod-startup.sh

# This will:
# - Clone moola repository
# - Install dependencies
# - Download training data from network storage
# - Set up workspace directories
```

### Step 3: Run Full Training Pipeline

```bash
# From pod SSH session
cd /workspace/moola

# Run full training pipeline
bash scripts/runpod-train.sh

# This will:
# - Ingest and validate training data
# - Train base models (logreg, rf, xgb, rwkv_ts, cnn_transformer)
# - Generate out-of-fold predictions
# - Train stacking meta-learner
# - Save models and metrics
# - Upload results to network storage
```

### Step 4: Download Results to Local Machine

```bash
# Back on your local machine
cd /Users/jack/projects/moola/.runpod

# Download training results
./sync-from-storage.sh all

# This will download:
# - Trained models
# - OOF predictions
# - Metrics and evaluation reports
# - Training logs
```

---

## Monitoring & Troubleshooting

### During Training

To monitor training in real-time:

```bash
# SSH into pod (in separate terminal)
ssh -i ~/.ssh/id_rsa root@<your-runpod-ip>

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check training logs
tail -f /workspace/logs/moola.log

# Check disk usage
df -h /workspace
```

### Common Issues & Solutions

#### Issue: "Pod keeps stopping/crashing"
**Solutions:**
1. **Increase container disk**: Minimum 50 GB required
2. **Check GPU memory**: Some models need 20+ GB VRAM
3. **Monitor logs**: `tail -f /workspace/logs/moola.log` for errors

#### Issue: "Network storage upload fails"
**Solutions:**
1. **Verify credentials**: Check `network-storage.env` has correct keys
2. **Check AWS CLI**: Run `aws s3 ls --endpoint-url https://s3api-eu-ro-1.runpod.io s3://hg878tp14w`
3. **Network timeout**: Try smaller files first with `sync-to-storage.sh models`

#### Issue: "OOF predictions all zeros"
**Solutions:**
1. **Check data loading**: Verify `train.parquet` format
2. **Check class balance**: `moola audit --section base`
3. **Review fold results**: Each fold should have non-zero predictions

#### Issue: "CNN-Transformer accuracy very low"
**Solutions:**
1. **Verify encoder loading**: Check logs for `[SSL]` messages
2. **Check data shape**: Must be [N, 105, 4] for OHLC
3. **Verify random seed**: Use fixed seed for reproducibility

---

## Best Practices

### 1. Resource Management
- **Monitor GPU memory**: Stop other tasks before training
- **Use appropriate batch size**: Adjust based on available VRAM
- **Clean up artifacts**: Remove old models before new runs

```bash
# Check available VRAM
nvidia-smi

# Kill hanging processes
pkill -f python
pkill -f moola
```

### 2. Data Management
- **Always validate data first**: Run `moola ingest` before training
- **Keep backups**: Download results after each run
- **Version control**: Tag successful model versions

```bash
# Validate data
moola ingest --cfg-dir configs

# Check pipeline status
moola audit --section all
```

### 3. Reproducibility
- **Use fixed seeds**: Set `seed=1337` in configs
- **Document hyperparameters**: Save config in artifacts
- **Version models**: Tag with timestamp and git SHA

```bash
# Get git commit
git rev-parse --short HEAD

# Tag model version
mkdir -p artifacts/models/cnn_transformer/v$(date +%Y%m%d)
```

### 4. Cost Optimization
- **Use spot instances**: 70% cheaper than on-demand
- **Stop pods when idle**: Avoid unnecessary charges
- **Monitor costs**: Check RunPod dashboard regularly

```bash
# After training, stop pod (from RunPod web UI)
# This stops billing until you restart

# For next session: Start existing pod (faster, keeps network storage data)
```

---

## Architecture Overview

### Network Storage Layout
```
Network Volume: moola (eu-ro-1)
├── scripts/                 # Deployment scripts
│   ├── pod-startup.sh      # Pod initialization
│   └── runpod-train.sh     # Full training pipeline
├── data/
│   ├── raw/                # Input data
│   └── processed/          # Cleaned training data
├── configs/                # Training configurations
└── artifacts/              # Results, models, logs
    ├── models/
    ├── oof/
    ├── metrics/
    └── logs/
```

### Training Pipeline
```
1. Data Ingest (validate & clean)
   ↓
2. Base Model Training (5-fold CV)
   - logreg, rf, xgb, rwkv_ts, cnn_transformer
   ↓
3. OOF Generation (ensemble features)
   ↓
4. Stacking (meta-learner training)
   ↓
5. Evaluation & Results Upload
```

---

## Quick Reference

### Essential Commands

```bash
# Pod Setup
aws s3 cp s3://hg878tp14w/scripts/pod-startup.sh /tmp/pod-startup.sh && bash /tmp/pod-startup.sh

# Training
cd /workspace/moola && bash scripts/runpod-train.sh

# Monitoring
nvidia-smi          # GPU status
df -h /workspace    # Disk usage
tail -f /workspace/logs/moola.log  # Training logs

# Results Download (from local machine)
cd ~/.projects/moola/.runpod && ./sync-from-storage.sh all
```

### File Locations (Pod)
- **Code**: `/workspace/moola`
- **Data**: `/workspace/data`
- **Logs**: `/workspace/logs/moola.log`
- **Results**: `/workspace/artifacts`
- **Network Storage**: `/mnt/network-storage` (if mounted)

---

## Support & Debugging

### Enable Verbose Logging

```bash
# Set log level to DEBUG in configs/default.yaml
log_level: DEBUG

# Or via CLI override
moola train --over logging.level=DEBUG
```

### Check System Resources

```bash
# GPU status and memory
nvidia-smi

# CPU and RAM usage
htop

# Disk space
df -h /workspace

# Network connectivity
curl -v https://s3api-eu-ro-1.runpod.io
```

### Validate Setup

```bash
# Check moola installation
python -c "import moola; print(moola.__version__)"

# Verify data loading
python -c "from moola.data.load import validate_expansions; print('OK')"

# Test GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Next Steps

1. Follow [Initial Setup](#initial-setup-one-time) to prepare infrastructure
2. Review [Running Training](#running-training) for pod setup
3. Monitor first training run with GPU stats
4. Download results and validate model quality
5. Set up recurring training schedule (optional: scheduled pod restarts)

---

*Last Updated: 2025-10-16*
*Tested on: PyTorch 2.4.1+cu124, CUDA 12.4, RTX 4090*
