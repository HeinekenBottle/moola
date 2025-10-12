# Simple RunPod Workflow

## One-Time Local Setup

```bash
cd /Users/jack/projects/moola/.runpod

# 1. Clean storage
chmod +x clean-storage.sh
./clean-storage.sh

# 2. Upload fresh files
./sync-to-storage.sh all
```

## On RunPod Pod

**Pick GPU: RTX 4090 or RTX 3090 (NOT 5090!)**
**Network Volume: moola (hg878tp14w)**

### Setup (3 minutes)
```bash
bash /workspace/scripts/setup.sh
```

### Train (10-20 minutes)
```bash
bash /workspace/scripts/train.sh
```

### Monitor GPU
```bash
watch -n 1 nvidia-smi
```

## Download Results (Local)

```bash
cd /Users/jack/projects/moola/.runpod
./sync-from-storage.sh all
```

Results in: `data/artifacts/`

---

## That's It!

3 commands total:
1. Setup pod
2. Train
3. Download results
