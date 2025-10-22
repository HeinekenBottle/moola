# Clean Stones Doctrine RunPod Commands

## Connection Status
❌ **RunPod instance 213.181.111.2:31187 currently unavailable**

## Manual Execution Steps

When the RunPod instance becomes accessible, execute these commands sequentially:

### 0) Clean Sync
```bash
# Create comprehensive rsync ignore file
cat > .rsyncignore <<'EOF'
.git/
__pycache__/
.artifacts/
artifacts/
data/raw/
data/processed/
logs/
wandb/
.env
.venv
.ipynb_checkpoints/
.DS_Store
*.tmp
*.pth
*.pt
*.ckpt
*.pyc
*.pyo
.pyd
.coverage
.pytest_cache
.mypy_cache
htmlcov/
.tox/
.nyc_output
coverage.xml
*.cover
*.log
.env.*
.DS_Store?
.vscode/
.idea/
*.swp
*.swo
*~
EOF

# Sync to RunPod
rsync -av --delete --exclude-from='.rsyncignore' \
  ./ root@213.181.111.2:/workspace/Moola \
  -e "ssh -p 31187 -i ~/.ssh/id_ed25519"
```

### 1) Remote Environment Setup
```bash
ssh -p 31187 -i ~/.ssh/id_ed25519 root@213.181.111.2 <<'SH'
set -e
cd /workspace/Moola

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install CUDA 12.1 PyTorch
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1+cu121 torchvision==0.18.1+cu121

# Install core dependencies
pip install numpy==1.26.4 scipy==1.11.4 pyarrow==16.1.0 pandas==2.2.2 \
  opencv-python-headless==4.10.0.84 tqdm pyyaml matplotlib wandb lightning==2.1.3

# Verify installation
python - <<'PY'
import sys, torch
print("Python:", sys.executable)
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory // 1024**3, "GB")
PY
SH
```

### 2) Doctrine Guards
```bash
ssh -p 31187 -i ~/.ssh/id_ed25519 root@213.181.111.2 <<'SH'
cd /workspace/Moola
source .venv/bin/activate

python - <<'PY'
import yaml, sys
from pathlib import Path

# Check model configs
models = ['jade', 'sapphire', 'opal']
for model in models:
    config_path = f"configs/model/{model}.yaml"
    if Path(config_path).exists():
        cfg = yaml.safe_load(open(config_path))
        assert cfg["model"]["name"] in {"jade","sapphire","opal"}, f"Invalid model name: {cfg['model']['name']}"
        assert cfg["model"]["pointer_head"]["encoding"]=="center_length", f"Invalid encoding: {cfg['model']['pointer_head']['encoding']}"
        assert cfg["train"]["batch_size"]==29, f"Invalid batch size: {cfg['train']['batch_size']}"
        print(f"✅ {model} config valid")
    else:
        print(f"⚠️  {model} config not found")

print("✅ All invariants validated")
PY
SH
```

### 3) MAE Pretraining
```bash
ssh -p 31187 -i ~/.ssh/id_ed25519 root@213.181.111.2 <<'SH'
cd /workspace/Moola
source .venv/bin/activate

# Prepare OHLC data
python - <<'PY'
import pandas as pd
import numpy as np
from pathlib import Path

# Load unlabeled data and convert to OHLC format
df = pd.read_parquet('data/raw/unlabeled_windows.parquet')
features = df['features'].values
ohlc_data = np.stack([np.stack(f) for f in features])

print(f"OHLC data shape: {ohlc_data.shape}")
print(f"Data type: {ohlc_data.dtype}")
print(f"Has NaN: {np.isnan(ohlc_data).any()}")

# Save processed OHLC data
Path('data/processed').mkdir(parents=True, exist_ok=True)
np.save('data/processed/ohlc_pretrain.npy', ohlc_data)
print("✅ OHLC data prepared and saved")
PY

# Run MAE pretraining
make pretrain-encoder DEVICE=cuda EPOCHS=100 BATCH=64 MASK_RATIO=0.4 DROPOUT=0.15
SH
```

### 4) Health Check
```bash
ssh -p 31187 -i ~/.ssh/id_ed25519 root@213.181.111.2 <<'SH'
cd /workspace/Moola
source .venv/bin/activate

python - <<'PY'
import os, json, torch
p="artifacts/encoders/pretrained/stones_encoder_mae.pt"
if os.path.exists(p):
    encoder_data = torch.load(p, map_location='cpu')
    stats = encoder_data.get('training_stats', {})
    print(json.dumps({
        "encoder_exists": True,
        "size_bytes": os.path.getsize(p),
        "final_train_mae": stats.get('final_train_mae', 'N/A'),
        "final_val_mae": stats.get('final_val_mae', 'N/A'),
        "mae_gap_pct": stats.get('mae_gap_pct', 'N/A')
    }, indent=2))
else:
    print(json.dumps({"encoder_exists": False, "error": "Encoder not found"}))
PY
SH
```

### 5) Supervised Jade Training
```bash
ssh -p 31187 -i ~/.ssh/id_ed25519 root@213.181.111.2 <<'SH'
cd /workspace/Moola
source .venv/bin/activate

# Set pointer warmup epochs
export WARMUP_PTR_EPOCHS=2

# Run supervised Jade training
make train-jade-clean DEVICE=cuda EPOCHS=60 BATCH=29
SH
```

### 6) Optional: Sapphire & Opal
```bash
ssh -p 31187 -i ~/.ssh/id_ed25519 root@213.181.111.2 <<'SH'
cd /workspace/Moola
source .venv/bin/activate

ENC="artifacts/encoders/pretrained/stones_encoder_mae.pt"

# Train Sapphire with frozen encoder
if [ -f "$ENC" ]; then
    make train-sapphire-clean DEVICE=cuda EPOCHS=40 BATCH=29 PREENC=$ENC
    echo "✅ Sapphire training completed"
else
    echo "⚠️  Skipping Sapphire - encoder not found"
fi

# Train Opal with adaptive fine-tuning
if [ -f "$ENC" ]; then
    make train-opal-clean DEVICE=cuda EPOCHS=40 BATCH=29 PREENC=$ENC
    echo "✅ Opal training completed"
else
    echo "⚠️  Skipping Opal - encoder not found"
fi
SH
```

### 7) Final Report
```bash
ssh -p 31187 -i ~/.ssh/id_ed25519 root@213.181.111.2 <<'SH'
cd /workspace/Moola
source .venv/bin/activate

echo "=== SEARCHING FOR KEY METRICS ==="
rg -n "Hit@±3|hit_at_pm3|F1_macro|ECE|log_sigma_(ptr|cls)|Early stopping|ReduceLROnPlateau" -S logs artifacts || echo "No metrics found in logs"

echo ""
echo "=== ARTIFACTS GENERATED ==="
find artifacts -name "*.pt" -o -name "*.pth" -o -name "*.json" | head -10

echo ""
echo "=== DISK USAGE ==="
du -sh artifacts/ logs/ 2>/dev/null || echo "No artifacts/logs directories found"
SH
```

## Retrieve Results
```bash
# Get final report
scp -P 31187 -i ~/.ssh/id_ed25519 root@213.181.111.2:/workspace/Moola/artifacts/clean_run_report.json ./

# Get trained models
scp -P 31187 -i ~/.ssh/id_ed25519 root@213.181.111.2:/workspace/Moola/artifacts/models/jade_compact.pt ./
scp -P 31187 -i ~/.ssh/id_ed25519 root@213.181.111.2:/workspace/Moola/artifacts/encoders/pretrained/stones_encoder_mae.pt ./
```

## Expected Results

### Pretraining Targets
- Train MAE: ~0.048
- Val MAE: ~0.052
- MAE Gap: 5-15%

### Supervised Targets (Jade)
- Hit@±3: ≥60%
- F1_macro: ≥0.50
- ECE: <0.10
- Joint: ≥0.40
- log_sigma_ptr: ≈0.74
- log_sigma_cls: 1.00 (fixed)

### Doctrine Compliance
- Model: jade ∈ {jade,sapphire,opal}
- Encoding: center_length
- Batch size: 29
- Float32: end-to-end
- No prohibited techniques

## Quick Deploy Script
When the instance is available, run:
```bash
./scripts/runpod_clean_deploy.sh 213.181.111.2 31187 root ~/.ssh/id_ed25519
```

This will execute the complete clean Stones doctrine run automatically.