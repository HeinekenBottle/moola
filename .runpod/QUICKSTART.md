# RunPod Network Storage - Quick Start

## 1. Get Your RunPod API Keys (One-Time)

1. Go to https://www.runpod.io/console/user/settings
2. Scroll to "S3 Access Keys" section
3. Click "Create S3 API Key"
4. Save the credentials:
   - Access Key ID
   - Secret Access Key

## 2. Configure Local Environment

```bash
# Add to your ~/.zshrc (macOS) or ~/.bashrc (Linux)
export AWS_ACCESS_KEY_ID="your-access-key-here"
export AWS_SECRET_ACCESS_KEY="your-secret-key-here"

# Reload shell config
source ~/.zshrc  # or source ~/.bashrc
```

**IMPORTANT**: Never commit these credentials to git!

## 3. Upload Scripts to Network Storage

```bash
cd /Users/jack/projects/moola/.runpod

# Load network storage configuration
source network-storage.env

# Upload everything (scripts, data, configs)
./sync-to-storage.sh all
```

Expected output:
```
📤 Syncing deployment scripts...
✅ deployment scripts synced
📤 Syncing processed datasets...
✅ processed datasets synced
📤 Syncing configuration files...
✅ configuration files synced
🎉 All files synced to network storage

📊 Network storage contents:
[list of uploaded files]
```

## 4. Start a RunPod Pod

1. Go to https://www.runpod.io/console/pods
2. Click "Deploy"
3. Select template:
   - **PyTorch 2.4+** (or create custom template)
4. Configure:
   - **GPU**: RTX 4090 or A100 (depending on budget)
   - **Container Disk**: 50 GB
   - **Volume**: Select "moola" (hg878tp14w) ✅ IMPORTANT!
5. Click "Deploy"
6. Wait ~1-2 minutes for pod to start

## 5. SSH Into Pod and Initialize

```bash
# Click "Connect" → "SSH Terminal" in RunPod web UI
# Or use SSH command shown in the pod details

# Once connected, run:
cd /runpod-volume
ls scripts/  # Verify scripts are there

# Run startup script
bash scripts/pod-startup.sh
```

This will:
- Clone repository
- Setup Python environment (cached for future pods)
- Create symlinks
- Verify GPU

Expected output:
```
🚀 Moola RunPod Pod Startup
✅ Network storage detected at: /runpod-volume
📥 Cloning Moola repository...
📦 Creating virtual environment...
✅ Virtual environment created and cached
🔗 Creating symlinks...
✅ GPU Available: NVIDIA RTX 4090
✅ Pod setup complete!
```

## 6. Start Training

```bash
# Option 1: Use convenient alias
moola-train

# Option 2: Run script directly
bash /runpod-volume/scripts/runpod-train.sh
```

The script will:
1. Generate OOF predictions for all 5 base models
2. Train the RandomForest stacker
3. Run full pipeline audit
4. Test predictions

Expected time:
- **With GPU**: 10-20 minutes total
- **Without GPU fixes**: 60-120 minutes

## 7. Monitor Training

Open a second SSH terminal and run:

```bash
# Watch GPU utilization (should be 80-95%)
watch -n 1 nvidia-smi

# Or check artifacts
ls -lh /runpod-volume/artifacts/models/

# Or watch logs
tail -f /runpod-volume/logs/*.log
```

## 8. Download Results (Back to Local Machine)

After training completes:

```bash
# On your local machine
cd /Users/jack/projects/moola/.runpod

# Download everything
./sync-from-storage.sh all

# Or download specific items
./sync-from-storage.sh artifacts
./sync-from-storage.sh models
```

Results will be in:
```
/Users/jack/projects/moola/
├── data/
│   ├── artifacts/
│   │   ├── models/        # Trained model .pkl files
│   │   ├── oof/           # Out-of-fold predictions
│   │   └── manifest.json  # Training metadata
│   └── logs/              # Training logs
```

## 9. Terminate Pod

**Your data is safe!** It's on network storage, not the pod.

1. Go to RunPod console
2. Click "Stop" or "Terminate" on your pod
3. Network storage persists with all artifacts

Next time you start a new pod:
- Your Python environment is already cached
- Your training artifacts are ready
- No need to re-upload data

## Common Issues

### "Permission denied" when syncing
```bash
# Check credentials are set
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY

# If empty, add to ~/.zshrc and reload
source ~/.zshrc
```

### "Network storage not found" on pod
Make sure you attached the volume when creating the pod!
- Volume name: `moola`
- Volume ID: `hg878tp14w`

### "CUDA not available"
```bash
# Check GPU
nvidia-smi

# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# If False, select a different pod template with CUDA support
```

## Success Checklist

- [ ] API keys configured locally
- [ ] Scripts uploaded to network storage (`./sync-to-storage.sh all`)
- [ ] Pod started with network volume attached
- [ ] Pod startup script ran successfully
- [ ] GPU detected and working
- [ ] Training completed without errors
- [ ] Artifacts downloaded to local machine
- [ ] Pod terminated (not running idle)

## Cost Estimate

| GPU | Hourly Cost | Training Time | Total Cost |
|-----|-------------|---------------|------------|
| RTX 4090 | $0.50/hr | ~15 min | ~$0.13 |
| RTX 4090 | $0.50/hr | ~1 hour | ~$0.50 |
| A100 | $1.50/hr | ~10 min | ~$0.25 |
| A100 | $1.50/hr | ~1 hour | ~$1.50 |

**Without GPU fixes**: Would take 60-120 minutes on CPU, wasting $0.50-2.00+ on unused GPU! 💸

**With GPU fixes**: 10-20 minutes, proper GPU utilization 80-95% ✅

---

That's it! You now have a complete RunPod workflow with persistent network storage. 🎉
