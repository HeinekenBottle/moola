# RunPod Network Storage Setup & Workflow

This directory contains scripts and configuration for managing RunPod Network Storage for the Moola project.

## Network Volume Details

- **Volume Name**: `moola`
- **Volume ID**: `hg878tp14w`
- **Region**: `eu-ro-1` (Romania)
- **Size**: 10 GB
- **S3 Endpoint**: `https://s3api-eu-ro-1.runpod.io`

## Directory Structure

```
.runpod/
├── README.md                    # This file
├── network-storage.env          # Storage credentials (DO NOT COMMIT AWS KEYS!)
├── sync-to-storage.sh           # Upload files from local → network storage
├── sync-from-storage.sh         # Download results from network storage → local
└── scripts/
    ├── pod-startup.sh           # Run when you first SSH into a new pod
    └── runpod-train.sh          # Full training pipeline on RunPod
```

## Prerequisites

### 1. Get RunPod API Keys

1. Go to https://www.runpod.io/console/user/settings
2. Create an API key (S3 Access Keys section)
3. Save your `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

### 2. Set Up AWS CLI (Local Machine)

```bash
# Install AWS CLI if not already installed
brew install awscli  # macOS
# or
pip install awscli

# Configure credentials (don't commit these!)
export AWS_ACCESS_KEY_ID="your-runpod-access-key"
export AWS_SECRET_ACCESS_KEY="your-runpod-secret-key"
```

**IMPORTANT**: Add these to your `~/.zshrc` or `~/.bashrc`, NOT to git!

```bash
# Add to ~/.zshrc (macOS) or ~/.bashrc (Linux)
export AWS_ACCESS_KEY_ID="your-runpod-access-key"
export AWS_SECRET_ACCESS_KEY="your-runpod-secret-key"
```

## Workflow

### Step 1: Initial Setup (One-Time)

Upload scripts and data to network storage:

```bash
cd /Users/jack/projects/moola/.runpod

# Make scripts executable
chmod +x sync-to-storage.sh sync-from-storage.sh
chmod +x scripts/*.sh

# Load environment variables
source network-storage.env

# Upload everything to network storage
./sync-to-storage.sh all
```

### Step 2: Start a RunPod Pod

1. Go to https://www.runpod.io/console/pods
2. Deploy a new pod:
   - **Template**: PyTorch 2.x (or Moola custom template if created)
   - **GPU**: RTX 4090 / A100 (depending on budget)
   - **Network Volume**: Select `moola` (hg878tp14w)
   - **Container Disk**: 50 GB minimum
3. Wait for pod to start
4. Click "Connect" → SSH or Web Terminal

### Step 3: Initialize Pod Environment

First time SSH into a new pod:

```bash
# Download and run startup script from network storage
aws s3 cp \
    --region eu-ro-1 \
    --endpoint-url https://s3api-eu-ro-1.runpod.io \
    s3://hg878tp14w/scripts/pod-startup.sh \
    /tmp/pod-startup.sh

bash /tmp/pod-startup.sh
```

This will:
- Clone the Moola repository
- Setup Python virtual environment (cached on network storage)
- Create symlinks for artifacts/data
- Verify GPU availability
- Configure aliases

### Step 4: Run Training

```bash
# Option 1: Use alias
moola-train

# Option 2: Run script directly
bash /runpod-volume/scripts/runpod-train.sh

# Option 3: Manual commands
cd /workspace/moola
source /runpod-volume/venv/bin/activate
python -m moola.cli oof --model rwkv_ts --device cuda --seed 1337
python -m moola.cli oof --model cnn_transformer --device cuda --seed 1337
# ... etc
```

### Step 5: Monitor Training

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Check artifacts
moola-status
# or
ls -lh /runpod-volume/artifacts/

# Watch logs
moola-logs
# or
tail -f /runpod-volume/logs/*.log
```

### Step 6: Download Results (From Local Machine)

After training completes on RunPod:

```bash
cd /Users/jack/projects/moola/.runpod

# Download all artifacts and logs
./sync-from-storage.sh all

# Or download specific items
./sync-from-storage.sh artifacts
./sync-from-storage.sh logs
./sync-from-storage.sh models
```

Results will be in:
- `data/artifacts/` - Trained models, OOF predictions, metrics
- `data/logs/` - Training logs

### Step 7: Terminate Pod

**IMPORTANT**: Your data is safe on network storage after pod termination!

1. Go to RunPod console
2. Stop/Terminate the pod
3. Network storage persists with all your artifacts

## Network Storage Layout

```
s3://hg878tp14w/
├── scripts/
│   ├── pod-startup.sh          # Pod initialization
│   └── runpod-train.sh         # Training pipeline
├── data/
│   └── processed/
│       └── window105_train.parquet
├── artifacts/
│   ├── models/
│   │   ├── logreg/
│   │   ├── rf/
│   │   ├── xgb/
│   │   ├── rwkv_ts/
│   │   ├── cnn_transformer/
│   │   └── stack/
│   ├── oof/                    # Out-of-fold predictions
│   ├── splits/                 # CV fold definitions
│   └── predictions/
├── logs/
│   └── *.log
├── venv/                       # Cached Python environment
└── configs/
```

## What Goes Where

### ✅ Network Storage (Persistent)
- Training artifacts (models, OOF predictions)
- Datasets (processed parquet files)
- Python virtual environment (cached dependencies)
- Training logs
- Deployment scripts

### ❌ Pod Local Storage (Ephemeral)
- Source code (git clone fresh each time)
- Temporary files
- Current training session logs (archived to network storage later)

## Cost Optimization Tips

1. **Pre-cache dependencies**: Virtual environment on network storage saves 5-10 min per pod startup
2. **Reuse artifacts**: Don't re-train base models if you're just tweaking the stacker
3. **Use spot instances**: 50-70% cheaper for interruptible workloads
4. **Stop pods when idle**: You're charged by the hour, not by the artifact
5. **Monitor GPU utilization**: If below 80%, you're wasting money

## Quick Commands Reference

### Local Machine

```bash
# Upload to network storage
cd /Users/jack/projects/moola/.runpod
./sync-to-storage.sh all              # Upload everything
./sync-to-storage.sh scripts          # Upload scripts only
./sync-to-storage.sh data             # Upload data only

# Download from network storage
./sync-from-storage.sh all            # Download everything
./sync-from-storage.sh artifacts      # Download artifacts only
./sync-from-storage.sh models         # Download models only

# Check network storage contents
aws s3 ls \
    --region eu-ro-1 \
    --endpoint-url https://s3api-eu-ro-1.runpod.io \
    s3://hg878tp14w/ --recursive --human-readable
```

### On RunPod Pod

```bash
# Setup pod (first time)
bash /tmp/pod-startup.sh

# Start training
moola-train

# Monitor
watch -n 1 nvidia-smi
moola-status
moola-logs

# Check results
ls -lh /runpod-volume/artifacts/models/
cat /runpod-volume/artifacts/manifest.json
```

## Troubleshooting

### "Permission denied" when syncing
Make sure AWS credentials are set:
```bash
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY
```

### "Network storage not found" on pod
Check mount point:
```bash
ls -la /runpod-volume/
ls -la /workspace/storage/
df -h | grep runpod
```

### "CUDA not available" on pod
Verify pod has GPU:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Artifacts not persisting
Make sure environment variables point to network storage:
```bash
echo $MOOLA_ARTIFACTS_DIR  # Should be /runpod-volume/artifacts
```

## Security Notes

1. **DO NOT commit AWS credentials to git**
2. Add to `.gitignore`:
   ```
   .runpod/network-storage.env
   **/aws-credentials*
   ```
3. Use RunPod's S3-compatible API, not your personal AWS account
4. Rotate API keys periodically

## Next Steps

1. ✅ Configure AWS credentials locally
2. ✅ Upload scripts to network storage
3. ✅ Start a RunPod pod with network volume attached
4. ✅ Run pod startup script
5. ✅ Train models with GPU
6. ✅ Download results
7. ✅ Terminate pod

Your data is safe on network storage and will be available for the next pod! 🎉
