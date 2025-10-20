# RunPod Quick Reference Card
**11D Training Pipeline - Command Cheatsheet**

---

## 🚀 Initial Setup

```bash
# Set RunPod IP (replace with actual)
export RUNPOD_IP="YOUR_RUNPOD_IP"

# Transfer files from Mac
cd /Users/jack/projects/moola
rsync -avz --progress --exclude='.git' --exclude='__pycache__' \
  --exclude='*.pyc' --exclude='.pytest_cache' \
  --exclude='data/raw/unlabeled' --exclude='artifacts/models' \
  -e "ssh -i ~/.ssh/runpod_key" \
  . ubuntu@$RUNPOD_IP:/workspace/moola/

# SSH into RunPod
ssh -i ~/.ssh/runpod_key ubuntu@$RUNPOD_IP
```

---

## ✅ Verification (Run on RunPod)

```bash
cd /workspace/moola

# Run verification script
python3 verify_runpod_env.py

# If all checks pass, proceed to training
```

---

## 🎯 Training Commands (Run on RunPod)

```bash
# Setup environment
cd /workspace/moola
export PYTHONPATH=/workspace/moola/src:$PYTHONPATH

# Step 1: Pretrain BiLSTM Encoder (20-30 min)
python3 -m moola.cli pretrain-bilstm \
  --input data/processed/labeled/train_latest_relative.parquet \
  --input-dim 11 --hidden-dim 128 --num-layers 2 \
  --mask-ratio 0.15 --mask-strategy patch --epochs 50 \
  --batch-size 256 --device cuda --seed 17 \
  --output artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt

# Step 2: Fine-tune Frozen (5-10 min)
python3 -m moola.cli train \
  --model enhanced_simple_lstm \
  --data data/processed/labeled/train_latest_relative.parquet \
  --split data/splits/fwd_chain_v3.json \
  --pretrained-encoder artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt \
  --freeze-encoder --input-dim 11 --epochs 5 \
  --augment-data false --device cuda --seed 17 --save-run true

# Step 3: Fine-tune Unfrozen (15-20 min)
python3 -m moola.cli train \
  --model enhanced_simple_lstm \
  --data data/processed/labeled/train_latest_relative.parquet \
  --split data/splits/fwd_chain_v3.json \
  --pretrained-encoder artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt \
  --no-freeze-encoder --input-dim 11 --epochs 25 \
  --augment-data false --device cuda --seed 17 --save-run true
```

---

## 📊 Retrieve Results (Run on Mac)

```bash
# Get encoder
scp -i ~/.ssh/runpod_key \
  ubuntu@$RUNPOD_IP:/workspace/moola/artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt \
  artifacts/encoders/pretrained/

# Get all run artifacts
scp -i ~/.ssh/runpod_key -r \
  ubuntu@$RUNPOD_IP:/workspace/moola/artifacts/runs/ \
  artifacts/

# Get logs
scp -i ~/.ssh/runpod_key -r \
  ubuntu@$RUNPOD_IP:/workspace/moola/data/logs/ \
  data/
```

---

## 🔍 Monitoring Commands (Run on RunPod)

```bash
# Check GPU usage
nvidia-smi

# Monitor training (if logs enabled)
tail -f data/logs/*.log

# Check encoder size after Step 1
ls -lh artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt
# Expected: ~500 KB

# List runs after Steps 2 & 3
ls -lht artifacts/runs/ | head -5
```

---

## 🐛 Emergency Commands

```bash
# If OOM error - reduce batch size
# In Step 1: --batch-size 128 (or 64)
# In Steps 2/3: batch size is auto (32), try 16

# Check CUDA memory
python3 -c "import torch; print(f'Free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB')"

# Kill training if needed
# Ctrl+C

# Re-verify environment
python3 verify_runpod_env.py
```

---

## ⏱️ Expected Timeline

| Step | Time | Command |
|------|------|---------|
| Setup + Transfer | 10-15 min | rsync, ssh |
| Verification | 1 min | verify_runpod_env.py |
| Step 1: Pretrain | 20-30 min | pretrain-bilstm |
| Step 2: Freeze | 5-10 min | train --freeze-encoder |
| Step 3: Unfreeze | 15-20 min | train --no-freeze-encoder |
| Retrieve | 5 min | scp |
| **TOTAL** | **55-80 min** | |

---

## 📋 Critical Parameters

**MUST use in ALL commands:**
- `--input-dim 11` (not 4!)
- `--seed 17` (reproducibility)
- `--augment-data false` (in train commands)
- `--device cuda` (GPU)

**Dataset:**
- Input: `train_latest_relative.parquet` (11D features)
- Split: `fwd_chain_v3.json` (purge_window=104)

**Output:**
- Encoder: `bilstm_mae_11d_v1.pt`
- Models: `artifacts/runs/<RUN_ID>/`

---

## ✅ Success Indicators

**Step 1 (Pretrain):**
- ✅ Loss decreases steadily
- ✅ Encoder file created (~500 KB)
- ✅ No CUDA OOM errors

**Step 2 (Freeze):**
- ✅ Quick convergence (5 epochs)
- ✅ Accuracy ~80-85%
- ✅ RUN_ID created

**Step 3 (Unfreeze):**
- ✅ Improvement over Step 2
- ✅ Accuracy ~85-90%
- ✅ Final RUN_ID noted

**Overall:**
- ✅ No input_dim mismatch errors
- ✅ Seed=17 used consistently
- ✅ All artifacts saved
