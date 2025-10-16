# RunPod Training Quick Start Guide

**For Moola ML Training Pipeline**

## Template Selection

**Recommended:**
```
runpod/pytorch:2.4-py3.11-cuda12.4-ubuntu22.04
```

**Conservative (if issues):**
```
runpod/pytorch:2.2-py3.10-cuda12.1-ubuntu22.04
```

## Pod Launch Checklist

1. **Create Pod:**
   - Template: PyTorch 2.4 + CUDA 12.4
   - GPU: RTX 4090 or A6000 (24GB+ VRAM)
   - Disk: 50GB minimum
   - Expose ports: 8888 (JupyterLab), 5000 (MLflow)

2. **Connect to Pod:**
   ```bash
   ssh root@<pod-ip> -p <pod-port> -i ~/.ssh/id_ed25519
   ```

## Setup Commands (Copy-Paste)

```bash
# Navigate to workspace
cd /workspace

# Clone repository
git clone https://github.com/yourusername/moola.git
cd moola

# Install dependencies
pip install --no-cache-dir -r requirements-runpod.txt

# Verify environment
python -c "
import torch
import numpy as np
import sklearn
import xgboost
import imblearn

print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ NumPy: {np.__version__}')
print(f'✅ CUDA Available: {torch.cuda.is_available()}')
print(f'✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'✅ scikit-learn: {sklearn.__version__}')
print(f'✅ XGBoost: {xgboost.__version__}')
print(f'✅ imbalanced-learn: {imblearn.__version__}')
"

# Upload data (if not in git)
# Option 1: SCP from local
# scp -P <pod-port> -i ~/.ssh/id_ed25519 \
#   -r data/processed/train.parquet \
#   root@<pod-ip>:/workspace/moola/data/processed/

# Option 2: Download from cloud storage
# aws s3 cp s3://your-bucket/train.parquet data/processed/

# Run training pipeline
python scripts/train_full_pipeline.py \
  --device cuda \
  --mlflow-experiment runpod-production \
  --seed 1337

# Or run individual steps
# python -m moola.cli pretrain-tcc --device cuda --epochs 100
# python -m moola.cli oof --model cnn_transformer --device cuda
# python -m moola.cli stack-train --seed 1337
```

## Download Results

```bash
# From your local machine:
scp -P <pod-port> -i ~/.ssh/id_ed25519 \
  -r root@<pod-ip>:/workspace/moola/data/artifacts \
  ./runpod-results/

# Or sync with rsync:
rsync -avz -e "ssh -p <pod-port> -i ~/.ssh/id_ed25519" \
  root@<pod-ip>:/workspace/moola/data/artifacts/ \
  ./runpod-results/
```

## Monitoring

```bash
# Inside pod - watch GPU usage
watch -n 1 nvidia-smi

# Check training logs
tail -f logs/moola_*.log

# MLflow UI (if exposed on port 5000)
mlflow ui --host 0.0.0.0 --port 5000
# Then visit http://<pod-ip>:5000 in browser
```

## Troubleshooting

### NumPy Error
```bash
pip uninstall numpy -y
pip install "numpy>=1.26.4,<2.0"
```

### Out of Memory
Edit `src/moola/models/ts_tcc.py`:
```python
# Line 311-312: Reduce batch size and workers
batch_size: int = 256,      # Was 512
num_workers: int = 8,       # Was 16
```

### Missing Data
```bash
# Check data directory
ls -lh data/processed/
# Should see: train.parquet or train_clean.parquet
```

## Expected Training Time

| Component | Duration (RTX 4090) | VRAM Usage |
|-----------|---------------------|------------|
| TS-TCC Pre-training | 20-30 min | 18-22 GB |
| OOF Generation (5 models) | 30-45 min | 8-16 GB |
| Stack Training | 2-5 min | 4-8 GB |
| **Total Pipeline** | **~1-1.5 hours** | **22 GB peak** |

## Cost Optimization

```bash
# After training completes, IMMEDIATELY:
# 1. Download artifacts
# 2. Stop pod (don't leave running)
# 3. RunPod charges by the second

# Estimated cost (RTX 4090):
# $0.30-0.50 per hour = $0.45-0.75 per full run
```

## Pre-Flight Checklist

Before launching pod:
- [ ] Code is committed and pushed to git
- [ ] Data is available (local file or cloud storage)
- [ ] requirements-runpod.txt is updated
- [ ] SSH key is configured in RunPod
- [ ] Sufficient RunPod credits ($5+ recommended)

After training:
- [ ] Download artifacts to local
- [ ] Check MLflow metrics
- [ ] Verify model files exist
- [ ] Stop pod to prevent charges
- [ ] Document results

## Emergency Stop

If training is stuck or costs are running up:
```bash
# Cancel training
Ctrl+C

# Check processes
ps aux | grep python

# Kill if needed
pkill -9 python

# Exit pod
exit

# In RunPod dashboard: Click "Stop" on pod
```
