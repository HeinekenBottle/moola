# RunPod Training Cheatsheet

**One-page reference for GPU training on RunPod**

---

## Template

```
runpod/pytorch:2.4-py3.11-cuda12.4-ubuntu22.04
```

---

## Setup (Copy-Paste)

```bash
cd /workspace
git clone https://github.com/yourusername/moola.git
cd moola
pip install --no-cache-dir -r requirements-runpod.txt
python scripts/verify_runpod_env.py
```

---

## Train

```bash
# Full pipeline
python scripts/train_full_pipeline.py --device cuda --mlflow-experiment runpod

# Individual steps
python -m moola.cli pretrain-tcc --device cuda --epochs 100
python -m moola.cli oof --model cnn_transformer --device cuda
python -m moola.cli stack-train --seed 1337
```

---

## Monitor

```bash
# GPU usage
watch -n 1 nvidia-smi

# Training logs
tail -f logs/moola_*.log

# MLflow (if exposed on port 5000)
mlflow ui --host 0.0.0.0 --port 5000
```

---

## Download Results

```bash
# From local machine
scp -P <pod-port> -i ~/.ssh/id_ed25519 \
  -r root@<pod-ip>:/workspace/moola/data/artifacts \
  ./runpod-results/
```

---

## Critical Versions

```
PyTorch:  2.4.0
NumPy:    1.26.4 (MUST be <2.0)
Python:   3.11
CUDA:     12.4
```

---

## NumPy Error Fix

```bash
pip uninstall numpy -y
pip install "numpy>=1.26.4,<2.0"
```

---

## OOM Fix

Edit `src/moola/models/ts_tcc.py`:
```python
batch_size: int = 256,      # Was 512
num_workers: int = 8,       # Was 16
```

---

## Training Time (RTX 4090)

- TS-TCC: 20-30 min
- OOF: 30-45 min
- Stack: 2-5 min
- **Total: ~1-1.5 hours**

---

## Cost

**$0.45-0.75 per run** (RTX 4090 @ $0.30-0.50/hour)

---

## Emergency Stop

```bash
Ctrl+C                    # Cancel training
pkill -9 python           # Force kill if stuck
exit                      # Leave pod
# Then stop pod in RunPod dashboard
```

---

## Verification

```bash
python scripts/verify_runpod_env.py
# Should see: ✅ ALL CHECKS PASSED
```

---

## Pre-Flight Checklist

- [ ] Code pushed to git
- [ ] Data available
- [ ] requirements-runpod.txt updated
- [ ] SSH key configured
- [ ] RunPod credits ($5+)

---

## Post-Training Checklist

- [ ] Download artifacts
- [ ] Check MLflow metrics
- [ ] Verify model files
- [ ] **STOP POD** (prevents charges)
- [ ] Document results

---

## Common Commands

```bash
# Check environment
python -c "import torch; print(torch.__version__)"
python -c "import numpy; print(numpy.__version__)"
python -c "import torch; print(torch.cuda.is_available())"

# Disk space
df -h

# GPU info
nvidia-smi

# Processes
ps aux | grep python

# Kill process
kill -9 <PID>
```

---

## Links

- Analysis: docs/RUNPOD_DEPENDENCY_ANALYSIS.md
- Quick Start: docs/RUNPOD_QUICKSTART.md
- Matrix: docs/DEPENDENCY_MATRIX.md
- Requirements: requirements-runpod.txt
