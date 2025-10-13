# RunPod Quick Reference

## One-Command Deployment

### Setup (First Time Only)
```bash
# Set AWS credentials (once)
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
```

### Daily Workflow
```bash
# Local: Deploy everything to RunPod
bash .runpod/deploy.sh deploy

# SSH to RunPod
ssh runpod

# RunPod: Auto-setup and start training
bash .runpod/deploy.sh train
```

### Commands
```bash
bash .runpod/deploy.sh deploy    # Deploy project to RunPod
bash .runpod/deploy.sh train     # Start training (run on RunPod)
bash .runpod/deploy.sh status    # Check deployment status
bash .runpod/deploy.sh logs      # View training logs
bash .runpod/deploy.sh cleanup   # Clean up all files
```

## What Gets Deployed

✅ **Included:**
- Configuration files (`configs/`)
- Training data (`data/processed/`)
- Source code (`src/`)
- Dependencies (`pyproject.toml`)

❌ **Excluded:**
- `.git/` directory
- `__pycache__/` files
- `.venv/` directories
- Temporary files

## Training Pipeline

When you run `deploy.sh train`, it automatically:

1. **Environment Setup**
   - Creates virtual environment (cached)
   - Installs PyTorch 2.1.2 with CUDA 11.8
   - Installs project dependencies

2. **Verification**
   - Checks GPU availability
   - Tests all imports
   - Verifies data loading

3. **Training**
   - Classical models (CPU): LogReg, RF, XGBoost
   - Deep learning models (GPU): CNN-Transformer, RWKV-TS
   - Meta-learner: RandomForest stacker

4. **Results**
   - Saves all artifacts to `/workspace/artifacts/`
   - Logs saved to `/workspace/logs/`
   - Everything persists on network storage

## File Locations

On RunPod, everything is stored at:
```
/workspace/  # or /runpod-volume/
├── data/processed/          # Training data
├── configs/                 # Configuration files
├── src/                     # Source code
├── scripts/start.sh         # Auto-setup script
├── venv/                    # Cached Python environment
├── artifacts/               # Training results
│   ├── models/              # Trained models
│   ├── oof/                 # OOF predictions
│   └── predictions/         # Test predictions
└── logs/                    # Training logs
```

## Common Issues

### "AWS credentials not set"
```bash
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
```

### "Network storage not found"
```bash
# On RunPod, check mount points
ls -la /workspace
ls -la /runpod-volume
```

### "Deployment failed"
```bash
# Safe to re-run (idempotent)
bash .runpod/deploy.sh deploy
```

### "Training failed"
```bash
# Check status and logs
bash .runpod/deploy.sh status
bash .runpod/deploy.sh logs

# Restart (safe to re-run)
bash .runpod/deploy.sh train
```

## Performance Tips

1. **Use GPU**: Ensure CUDA models are used when available
2. **Cached Environment**: Subsequent runs are much faster
3. **Monitor GPU**: Check GPU utilization during training
4. **Persistent Storage**: All results survive pod restarts

## Training Commands (Manual)

If you want to run specific models manually:
```bash
# Activate environment
source /workspace/venv/bin/activate

# Classical models (CPU)
python -m moola.cli oof --model logreg --device cpu --seed 1337
python -m moola.cli oof --model rf --device cpu --seed 1337
python -m moola.cli oof --model xgb --device cpu --seed 1337

# Deep learning models (GPU)
python -m moola.cli oof --model cnn_transformer --device cuda --seed 1337
python -m moola.cli oof --model rwkv_ts --device cuda --seed 1337

# Meta-learner
python -m moola.cli stack-train --stacker rf --seed 1337

# Test predictions
python -m moola.cli predict --model stack \
    --input /workspace/data/processed/train.parquet \
    --output /workspace/artifacts/predictions/test.csv
```

## Getting Help

1. **Status Check**: `bash .runpod/deploy.sh status`
2. **View Logs**: `bash .runpod/deploy.sh logs`
3. **Fresh Start**: `bash .runpod/deploy.sh cleanup && bash .runpod/deploy.sh deploy`
4. **Full Reset**: Delete and recreate RunPod pod

## Migration from Old Workflow

This new system replaces ALL old scripts:
- No more `sync-scripts-robust.sh`
- No more `network-storage-cleanup.sh`
- No more `robust-setup.sh`
- No more manual environment setup
- No more pagination errors

Just one script handles everything: `deploy.sh`