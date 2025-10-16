# RunPod Infrastructure Quick Reference

Essential specifications and architecture for Moola RunPod deployments.

---

## Network Storage Architecture

### Volume Specifications

```
Volume Name:    moola
Volume ID:      hg878tp14w
Region:         eu-ro-1 (Romania)
Capacity:       10 GB
S3 Endpoint:    https://s3api-eu-ro-1.runpod.io
```

### Directory Structure

```
moola/ (10 GB network volume)
├── scripts/                    # Deployment automation (~12 KB)
│   ├── pod-startup.sh         # Pod initialization script
│   └── runpod-train.sh        # Full training pipeline
│
├── data/processed/            # Training datasets (~232 KB)
│   ├── train.parquet          # Original training data
│   └── window105_train.parquet # Processed OHLC features
│
├── configs/                    # Configuration files (~1 KB)
│   ├── default.yaml
│   └── hardware/
│       ├── cpu.yaml
│       └── gpu.yaml
│
├── venv/                       # Cached Python environment (~2-3 GB)
│   ├── bin/python3, pip, moola
│   └── lib/python3.10/site-packages/
│       ├── torch/             # PyTorch with CUDA (~2 GB)
│       ├── numpy/             # (~20 MB)
│       ├── pandas/            # (~50 MB)
│       ├── sklearn/           # (~100 MB)
│       ├── xgboost/           # (~50 MB)
│       └── [other deps]       # (~200 MB)
│
└── artifacts/                 # Training results (~5-6 GB available)
    ├── models/                # Trained model weights
    ├── oof/                   # Out-of-fold predictions
    ├── metrics/               # Evaluation metrics
    ├── splits/                # CV fold manifests
    └── logs/                  # Training logs
```

### Storage Capacity Planning

**Initial Upload:** ~245 KB (scripts + data + configs)
**Environment (cached):** ~2-3 GB (one-time per pod, reused)
**Training Artifacts:** ~500 MB - 1 GB per full training run
**Available for Results:** ~5-6 GB

---

## Template Specifications

### Recommended RunPod Template

```
runpod/pytorch:2.4-py3.11-cuda12.4-ubuntu22.04
```

**Pre-installed packages:**
```
torch==2.4.1+cu124       (includes CUDA 12.4)
torchvision==0.19.1+cu124
torchaudio==2.4.1+cu124
numpy==1.26.4
triton==3.0.0
```

### What This Template Includes/Excludes

✅ **Included (no pip install needed):**
- PyTorch 2.4 with CUDA 12.4 support (~2 GB pre-installed)
- NumPy, Triton, TorchVision, TorchAudio
- CUDA runtime and compilation tools
- Ubuntu 22.04 base image

❌ **NOT Included (must install via pip):**
- Pandas, SciPy, Scikit-learn (not pre-compiled)
- XGBoost, LightGBM
- Hydra, Pydantic, PyYAML
- Click, Loguru, Rich

### Why Template Selection Matters

Using templates that DON'T have scipy/sklearn pre-installed causes 45-minute pip compilation:
- `scipy` compiles from source (~15-20 minutes)
- `scikit-learn` compiles from source (~20-30 minutes)
- Total setup: 45+ minutes instead of 90 seconds

**Solution:** Accept that pip will compile pandas/scipy/sklearn (~5-10 minutes) instead of searching for a template that has everything.

### Verifying Template Packages

On RunPod pod, check what's pre-installed:

```bash
python3 << 'EOF'
packages = ['torch', 'numpy', 'pandas', 'scipy', 'sklearn', 'xgboost']
for pkg in packages:
    try:
        mod = __import__(pkg)
        print(f"✓ {pkg}: {mod.__version__}")
    except ImportError:
        print(f"✗ {pkg}: NOT FOUND")
EOF
```

---

## Pod Configuration

### Recommended Pod Settings

| Setting | Value | Reason |
|---------|-------|--------|
| **GPU** | RTX 4090 or A100 | 24 GB VRAM for large models |
| **vCPU** | 8-12 cores | Parallel data loading |
| **RAM** | 32-64 GB | Caching and large datasets |
| **Container Disk** | 50-100 GB | Room for venv + artifacts |
| **Network Volume** | moola (hg878tp14w) | Persistent storage |
| **Data Center** | eu-ro-1 | Lowest latency |
| **Template** | pytorch:2.4-py3.11-cuda12.4 | See above |

---

## Quick Commands

### Local Setup (One-Time)

```bash
# 1. Set AWS credentials
export AWS_ACCESS_KEY_ID="your-runpod-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-runpod-secret-access-key"

# 2. Upload initial data to network storage
cd /Users/jack/projects/moola/.runpod
source network-storage.env
./sync-to-storage.sh all
```

### Pod Initialization

```bash
# SSH to pod (replace with your pod IP)
ssh -i ~/.ssh/id_rsa root@<pod-ip>

# Download and run startup script
aws s3 cp \
    --region eu-ro-1 \
    --endpoint-url https://s3api-eu-ro-1.runpod.io \
    s3://hg878tp14w/scripts/pod-startup.sh \
    /tmp/pod-startup.sh

bash /tmp/pod-startup.sh
```

### Training Commands (on Pod)

```bash
cd /workspace/moola

# Full training pipeline
bash scripts/runpod-train.sh

# Individual models (manual)
source venv/bin/activate

moola oof --model logreg --device cpu --seed 1337
moola oof --model cnn_transformer --device cuda --seed 1337
moola stack-train --stacker rf --seed 1337

# Monitor
nvidia-smi           # GPU status
df -h /workspace     # Disk usage
tail -f /workspace/logs/moola.log  # Training logs
```

### Results Download (Local)

```bash
# Back on local machine
cd /Users/jack/projects/moola/.runpod

# Download results
./sync-from-storage.sh all
```

---

## Performance Characteristics

### Training Time per Fold (RTX 4090)

| Model | Single Fold | 5-Fold CV |
|-------|-------------|-----------|
| LogReg | ~2s | ~10s |
| RandomForest | ~5s | ~25s |
| XGBoost | ~10s | ~50s |
| CNN-Transformer | ~30s | ~150s |
| RWKV-TS | ~20s | ~100s |
| Stacking | ~5s | N/A |

**Total full pipeline:** ~5 minutes (with 5-fold CV)

### Memory Usage

| Model | VRAM Required |
|-------|---------------|
| CNN-Transformer | ~12-15 GB (batch_size=512) |
| RWKV-TS | ~8-10 GB (batch_size=512) |
| XGBoost | ~500 MB |
| RandomForest | ~300 MB |
| LogReg | ~100 MB |

---

## Environment Variables (on Pod)

```bash
# AWS S3 Access
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="eu-ro-1"

# Paths
export WORKSPACE="/workspace"
export PROJECT_DIR="/workspace/moola"
export VENV_DIR="/workspace/venv"

# PyTorch
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

# Reproducibility
export PYTHONHASHSEED="1337"
```

---

## Cost Optimization Tips

### Pod Costs

- **On-Demand**: ~$0.44/hour (RTX 4090)
- **Spot**: ~$0.13/hour (RTX 4090, 70% cheaper)
- **Network Storage**: $10/month (10 GB)

### Reduce Costs

1. **Use Spot Instances**: Save 70% on compute
2. **Stop Pods When Idle**: Network storage persists, only charge pod time
3. **Reuse Environment**: Network storage caches venv (~3 GB)
4. **Batch Runs**: Run multiple experiments per pod session

### Cost Example (5 training runs)

```
Spot RTX 4090: $0.13/hour × 5 runs × 10 min = $0.11
Network Storage: (shared) $10/month
Total: ~$10.11/month for 5 full training runs
```

---

## Important Notes

### Data Persistence
- **Network Storage**: Persists across pod restarts (safe)
- **Container Disk**: Lost when pod stops (temporary)
- **Pod-local files**: Lost when pod terminates

**Best Practice:** All important files (models, logs, artifacts) saved to network storage

### Reproducibility
- **Seed Management**: Always use fixed seed (default: 1337)
- **Deterministic CUDA**: Enabled in config (`SEED_REPRODUCIBLE=True`)
- **Environment Pinning**: Specify exact versions in pyproject.toml

### Security
- **Never commit credentials**: AWS keys go in environment variables only
- **Use `.env` files**: Add `.runpod/network-storage.env` to `.gitignore`
- **Rotate keys periodically**: RunPod console → Settings → S3 Access Keys

---

## Troubleshooting Quick Links

- **Pod won't start**: See TROUBLESHOOTING.md → Pod Issues
- **Training fails**: See TROUBLESHOOTING.md → Training Issues
- **Storage problems**: See TROUBLESHOOTING.md → Deployment & Sync Issues
- **GPU memory errors**: See TROUBLESHOOTING.md → Performance Issues

---

## See Also

- [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) - Full setup instructions
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Common issues & solutions

---

*Last Updated: 2025-10-16*
*PyTorch 2.4.1+cu124, CUDA 12.4, RTX 4090*
