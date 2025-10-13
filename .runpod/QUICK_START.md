# RunPod Quick Start - 2 Commands to Train

## Prerequisites (One-Time Setup)

1. **Upload to network storage** (from local machine):
   ```bash
   cd /Users/jack/projects/moola/.runpod
   bash deploy-fast.sh deploy
   ```

2. **Start RunPod Pod**:
   - Template: PyTorch 2.1
   - GPU: RTX 4090
   - Network Volume: moola (22uv11rdjk)

## Training (Every Pod)

### Step 1: Setup (2-3 minutes)
```bash
cd /workspace
bash scripts/optimized-setup.sh
```

### Step 2: Train (25-30 minutes)
```bash
moola-train
```

That's it! Two commands.

## Download Results (From Local Machine)

```bash
cd /Users/jack/projects/moola/.runpod
bash sync-from-storage.sh artifacts
```

Results in: `data/artifacts/models/stack/`

## What Gets Fixed

✅ Automatic data symlinks (train.parquet)
✅ PYTHONPATH configured automatically
✅ Venv activates on login
✅ All packages installed correctly
✅ GPU verified
✅ Data verified

## If Something Goes Wrong

### Data not found
```bash
ls -la /workspace/moola/data/processed/train.parquet
# Should point to train_2class.parquet
```

### Import errors
```bash
echo $PYTHONPATH
# Should include /workspace/moola/src
```

### GPU not detected
```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Training Aliases

After setup, these work anywhere:
- `moola-train` - Start training
- `moola-status` - Check training progress

## Expected Results

- **Stack F1**: 57-65%
- **Stack Accuracy**: 60-70%
- **Stack ECE**: 0.10-0.11
- **Training time**: 25-30 minutes on RTX 4090

## Files

- Setup script: `/workspace/scripts/optimized-setup.sh`
- Training script: `/workspace/scripts/fast-train.sh`
- Results: `/workspace/moola/data/artifacts/`
