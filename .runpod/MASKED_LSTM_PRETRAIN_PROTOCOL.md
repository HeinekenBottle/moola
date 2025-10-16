# 🚀 Masked LSTM Pre-training Protocol - RTX 4090

**Pod**: 213.173.110.220:36832
**Target**: Pre-train bidirectional masked LSTM encoder
**Time estimate**: ~40-50 minutes total

---

## Phase 1: Verify Pod State

### Check what's running
```bash
ssh root@213.173.110.220 -p 36832 -i ~/.ssh/id_ed25519 "ps aux | grep -E 'python|moola' | head -5"
```

### Check directory structure
```bash
ssh root@213.173.110.220 -p 36832 -i ~/.ssh/id_ed25519 "ls -la /workspace/"
```

### Check data
```bash
ssh root@213.173.110.220 -p 36832 -i ~/.ssh/id_ed25519 "ls -lh /workspace/data/processed/ 2>/dev/null || echo 'No data yet'"
```

---

## Phase 2: Pull Latest Code (Commit 7c9b1bb)

```bash
ssh root@213.173.110.220 -p 36832 -i ~/.ssh/id_ed25519 'bash -s' << 'EOF'
cd /workspace/moola
echo "📥 Pulling latest code..."
git pull origin main
echo "✅ Code updated"
echo ""
git log -1 --oneline
EOF
```

**Should see**: `7c9b1bb feat: complete bidirectional Masked LSTM pre-training implementation`

---

## Phase 3: Setup Environment

```bash
ssh root@213.173.110.220 -p 36832 -i ~/.ssh/id_ed25519 'bash -s' << 'EOF'
set -e

cd /workspace/moola

echo "🔧 Verifying environment..."
echo ""

# Check Python
python3 --version

# Check venv
if [ ! -d /tmp/moola-venv ]; then
    echo "📦 Creating venv..."
    python3 -m venv /tmp/moola-venv --system-site-packages
fi

source /tmp/moola-venv/bin/activate

# Install moola package
echo "📦 Installing moola..."
pip install -q -e . --no-deps

# Verify imports
python3 -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA: {torch.cuda.is_available()}')
print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
from moola.pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer
print('✅ MaskedLSTMPretrainer imported')
"

echo ""
echo "✅ Environment ready"
EOF
```

---

## Phase 4: Upload Training Data (if needed)

```bash
# Check if data exists locally
ls -lh data/processed/train_pivot_134.parquet

# Upload
scp -P 36832 -i ~/.ssh/id_ed25519 \
    data/processed/train_pivot_134.parquet \
    root@213.173.110.220:/workspace/data/processed/train_pivot_134.parquet

# Verify on pod
ssh root@213.173.110.220 -p 36832 -i ~/.ssh/id_ed25519 \
    "ls -lh /workspace/data/processed/train_pivot_134.parquet"
```

---

## Phase 5: Generate Unlabeled Data

**Key augmentation parameters** (optimized for financial time series on RTX 4090):
- **Time warping**: σ=0.12 (±12%) ← Conservative to preserve pivot locations
- **Jittering**: σ=0.05 (±5%)
- **Volatility scaling**: (0.85, 1.15) ± 15%
- **Augmentation factor**: 4x (1000 base → 5000 total)

```bash
ssh root@213.173.110.220 -p 36832 -i ~/.ssh/id_ed25519 'bash -s' << 'EOF'
source /tmp/moola-venv/bin/activate
cd /workspace/moola
export PYTHONPATH="/workspace/moola/src:$PYTHONPATH"

echo "📊 Generating unlabeled pre-training data..."
echo "=============================================="
echo ""
echo "Augmentation parameters:"
echo "  Time warping σ: 0.12 (±12%)"
echo "  Jittering σ: 0.05 (±5%)"
echo "  Volatility scale: (0.85, 1.15)"
echo "  Augmentation factor: 4x"
echo ""

python3 scripts/generate_unlabeled_data.py \
    --input /workspace/data/processed/train_pivot_134.parquet \
    --output /workspace/data/processed/unlabeled_pretrain.parquet \
    --target-count 1000 \
    --augment-factor 4 \
    --seed 1337

echo ""
echo "✅ Generated unlabeled data"
ls -lh /workspace/data/processed/unlabeled_pretrain.parquet
EOF
```

**Expected output**: ~2.5 MB parquet file with 5,000 sequences

---

## Phase 6: Pre-train Masked LSTM Encoder

### Command with monitoring:
```bash
ssh root@213.173.110.220 -p 36832 -i ~/.ssh/id_ed25519 'bash -s' << 'EOF'
source /tmp/moola-venv/bin/activate
cd /workspace/moola
export PYTHONPATH="/workspace/moola/src:$PYTHONPATH"
export MOOLA_DATA_DIR="/workspace/data"
export MOOLA_ARTIFACTS_DIR="/workspace/artifacts"

echo "🚀 MASKED LSTM PRE-TRAINING"
echo "==========================="
echo ""
echo "Pod: RTX 4090 (24GB VRAM)"
echo "Data: 5,000 unlabeled sequences (105 timesteps, 4 features)"
echo "Masking: Patch strategy (15% ratio, 7-bar patches)"
echo "Augmentation: Time warp ±12%, jitter ±5%, volatility ±15%"
echo ""
echo "Expected time: 30-40 minutes"
echo "Expected final loss: Train 0.012-0.015, Val 0.015-0.020"
echo ""

python3 -m moola.cli pretrain-bilstm \
    --input /workspace/data/processed/unlabeled_pretrain.parquet \
    --output /workspace/artifacts/pretrained/bilstm_encoder.pt \
    --device cuda \
    --epochs 50 \
    --batch-size 512 \
    --mask-strategy patch \
    --mask-ratio 0.15 \
    --augment \
    --seed 1337

echo ""
echo "✅ PRE-TRAINING COMPLETE"
echo ""
ls -lh /workspace/artifacts/pretrained/bilstm_encoder.pt
EOF
```

**Monitor in separate terminal**:
```bash
ssh root@213.173.110.220 -p 36832 -i ~/.ssh/id_ed25519 "watch -n 2 nvidia-smi"
```

**Expected GPU usage**: 90-100% utilization, 8-12 GB VRAM

---

## Phase 7: Verify Encoder

```bash
ssh root@213.173.110.220 -p 36832 -i ~/.ssh/id_ed25519 'bash -s' << 'EOF'
echo "🔍 Verifying pre-trained encoder..."
echo ""

# Check file exists and size
ls -lh /workspace/artifacts/pretrained/bilstm_encoder.pt

# Check contents
python3 << 'PYTHON'
import torch

encoder_path = "/workspace/artifacts/pretrained/bilstm_encoder.pt"
checkpoint = torch.load(encoder_path, map_location="cpu")

print("✅ Checkpoint structure:")
for key in checkpoint.keys():
    print(f"  - {key}")

if 'encoder_state_dict' in checkpoint:
    print("\n✅ Encoder state dict (keys):")
    for key in list(checkpoint['encoder_state_dict'].keys())[:5]:
        print(f"  - {key}")

if 'hyperparams' in checkpoint:
    print("\n✅ Hyperparameters:")
    for k, v in checkpoint['hyperparams'].items():
        print(f"  - {k}: {v}")

print("\n✅ Encoder ready for fine-tuning!")
PYTHON

EOF
```

---

## Phase 8: Download Encoder

```bash
mkdir -p data/artifacts/pretrained

scp -P 36832 -i ~/.ssh/id_ed25519 \
    root@213.173.110.220:/workspace/artifacts/pretrained/bilstm_encoder.pt \
    data/artifacts/pretrained/bilstm_encoder.pt

ls -lh data/artifacts/pretrained/bilstm_encoder.pt
```

---

## Error Handling

### Out of Memory
```bash
# Reduce batch size and restart
ssh root@213.173.110.220 -p 36832 -i ~/.ssh/id_ed25519 'bash -s' << 'EOF'
source /tmp/moola-venv/bin/activate
cd /workspace/moola
python3 -m moola.cli pretrain-bilstm \
    --input /workspace/data/processed/unlabeled_pretrain.parquet \
    --output /workspace/artifacts/pretrained/bilstm_encoder.pt \
    --device cuda \
    --epochs 50 \
    --batch-size 256 \
    --mask-strategy patch \
    --augment \
    --seed 1337
EOF
```

### Module not found
```bash
ssh root@213.173.110.220 -p 36832 -i ~/.ssh/id_ed25519 'bash -s' << 'EOF'
cd /workspace/moola
git pull origin main
source /tmp/moola-venv/bin/activate
pip install -q -e . --no-deps
python3 -c "from moola.pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer; print('✅ OK')"
EOF
```

### Slow pre-training (>50 min)
```bash
# Check GPU utilization
ssh root@213.173.110.220 -p 36832 -i ~/.ssh/id_ed25519 "nvidia-smi dmon -s u"
# Should see 90-100% in column 'u'

# If low, increase batch size
--batch-size 1024
```

---

## Next Steps After Pre-training

1. **Download encoder** (Phase 8)
2. **Fine-tune SimpleLSTM** with pre-trained encoder
3. **Compare results** vs baseline (57.14% accuracy)
4. **Expected improvement**: +8-12% overall, Class 1 from 0% → 45-55%

---

## Checklist

- [ ] Phase 1: Verify pod state
- [ ] Phase 2: Pull latest code (commit 7c9b1bb)
- [ ] Phase 3: Setup environment
- [ ] Phase 4: Upload training data (if needed)
- [ ] Phase 5: Generate unlabeled data (5,000 sequences)
- [ ] Phase 6: Pre-train encoder (~30-40 min)
- [ ] Phase 7: Verify encoder checkpoint
- [ ] Phase 8: Download encoder
- [ ] Next: Fine-tune SimpleLSTM

---

## Reference

**Augmentation Strategy**: `.runpod/AUGMENTATION_STRATEGY_ANALYSIS.md`
- Time warping: 12% (conservative for masked AE)
- Why not 10-20%? See section "Scientific Justification"

**Implementation Roadmap**: `MASKED_LSTM_IMPLEMENTATION_ROADMAP.md`
- Full technical details of masking strategies
- Architecture decisions
- Loss computation logic

**Quick Reference**: `QUICK_START_RUNPOD_MASKED_LSTM.md`
- 8-step deployment guide

---

**Commit**: 7c9b1bb
**Status**: Ready for pre-training
