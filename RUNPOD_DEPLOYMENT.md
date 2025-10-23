# RunPod Deployment Guide - 32-Core Pre-Computation

**Target:** 5-10 minute feature pre-computation on high-compute CPU pod
**Date:** 2025-10-23

---

## 🖥️ Pod Selection (Recommended)

### Option 1: Max Compute (FASTEST)
- **Pod Type:** Compute-Optimized
- **CPU:** 32 vCPUs (AMD EPYC 9754 or similar)
- **RAM:** 64 GB DDR5
- **Storage:** NVMe SSD
- **Clock:** 5.0-5.2 GHz boost
- **OS:** Ubuntu 22.04 LTS
- **Est. Cost:** ~$0.35/hour
- **Est. Time:** 5-7 minutes

### Option 2: Mid Compute (BALANCED)
- **Pod Type:** Compute-Optimized
- **CPU:** 16 vCPUs (AMD EPYC 9654 or similar)
- **RAM:** 32 GB
- **Clock:** 4.8+ GHz
- **OS:** Ubuntu 22.04 LTS
- **Est. Cost:** ~$0.20/hour
- **Est. Time:** 8-12 minutes

### Option 3: General Purpose (FALLBACK)
- **Pod Type:** General-Purpose
- **CPU:** 32 vCPUs (Intel Xeon or AMD Ryzen)
- **RAM:** 64 GB
- **Clock:** 4.5+ GHz
- **OS:** Ubuntu 22.04 LTS
- **Est. Cost:** ~$0.28/hour
- **Est. Time:** 10-15 minutes

---

## 🚀 Deployment Steps

### 1. Launch Pod on RunPod

```bash
# Go to RunPod Dashboard
# → CPU Pods → Compute-Optimized
# → Filter: 32 vCPUs, 64GB RAM, Ubuntu 22.04
# → Deploy
```

**Wait for:**
- Pod status: "Running"
- SSH endpoint appears
- 2-5 minutes for initialization

---

### 2. SSH Into Pod

```bash
# Get SSH command from RunPod dashboard
# Format: ssh root@<pod-ip> -p <port>

# Example:
ssh root@104.171.203.45 -p 12345

# Or if using key:
ssh -i ~/.ssh/runpod_key root@104.171.203.45 -p 12345
```

---

### 3. Setup Environment

```bash
# Update system packages
apt-get update && apt-get install -y git rsync htop

# Install Python dependencies
pip3 install --upgrade pip
pip3 install pandas numpy pydantic joblib tqdm pyarrow pyyaml
pip3 install ta-lib  # Technical analysis library

# Clone moola repo (or rsync from Mac)
cd /workspace
git clone https://github.com/YOUR_USERNAME/moola.git
# OR: rsync from Mac (see below)
```

**Alternative: Rsync from Mac**
```bash
# From your Mac terminal:
rsync -avz --exclude data --exclude artifacts --exclude .git \
  -e "ssh -p 12345" \
  /Users/jack/projects/moola/ \
  root@104.171.203.45:/workspace/moola/
```

---

### 4. Upload 5-Year NQ Data

**Option A: SCP from Mac**
```bash
# From Mac terminal:
scp -P 12345 \
  /Users/jack/projects/moola/data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet \
  root@104.171.203.45:/workspace/moola/data/raw/
```

**Option B: Download from Cloud Storage**
```bash
# If you have it on S3/GCS/etc:
# aws s3 cp s3://your-bucket/nq_5year.parquet data/raw/
# gsutil cp gs://your-bucket/nq_5year.parquet data/raw/
```

---

### 5. Verify Setup

```bash
cd /workspace/moola

# Check Python version
python3 --version  # Should be 3.8+

# Check dependencies
python3 -c "import pandas, numpy, joblib, pydantic; print('✅ Dependencies OK')"

# Check data file
ls -lh data/raw/*.parquet
# Should show: nq_ohlcv_1min_2020-09_2025-09_fixed.parquet (~500MB)

# Check CPU cores
nproc  # Should show 32

# Check RAM
free -h  # Should show ~64GB
```

---

### 6. Run Parallel Pre-Computation

```bash
cd /workspace/moola

# Run with 32 workers (5-10 minutes)
python3 scripts/precompute_nq_features_parallel.py \
  --data data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet \
  --output data/processed/nq_features \
  --n-jobs 32

# Optional: Monitor in another terminal
# ssh into pod, then run:
htop  # Watch CPU usage (should be ~100% across all cores)
```

**Expected output:**
```
================================================================================
NQ Parallel Feature Precomputation (32-Core Optimized)
================================================================================
Output directory: data/processed/nq_features

Loading data from data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet...
  ✅ Loaded 1,818,450 bars in 3.2s
  Date range: 2020-09-01 to 2025-09-30
  Memory usage: 145.5 MB

Sharding data by month (overlap=30 bars)...
  ✅ Created 60 monthly shards in 0.8s
  Avg bars per shard: 30,307

Processing shards in parallel (n_jobs=32)...
  Features (11 dims):
    Candle (6): open_norm, close_norm, body_pct, upper_wick, lower_wick, range_z
    Swing (4):  dist_to_SH, dist_to_SL, bars_since_SH, bars_since_SL
    Expansion (1): expansion_proxy = range_z × leg_dir × body_pct

  🚀 Using 32 workers...
  [Parallel progress bars showing shard processing...]

  ✅ Parallel processing completed in 318.4s (5710 bars/s)
  Speedup: ~5.7x vs sequential (est.)

Merging shard results...
  ✅ Merged 60 shards in 12.3s
  Final shape: (1818345, 105, 11)
  Memory usage: 7942.8 MB
  Valid ratio: 0.987

Creating time-based split indices...
  Train: windows [0, 1234567) = 1,234,567 windows
  Val:   windows [1234567, 1345678) = 111,111 windows
  Test:  windows [1345678, 1456789) = 111,111 windows

Saving feature arrays...
  ✅ Saved features to features_11d.npy (7942.8 MB)
  ✅ Saved valid mask to valid_mask.npy (182.3 MB)
  Save time: 45.2s

  ✅ Saved metadata to metadata.json
  ✅ Saved split indices to splits.json

================================================================================
PARALLEL PRECOMPUTATION COMPLETE
================================================================================
Total time: 379.9s (6.3m)
Processing speed: 5710 bars/s
Speedup vs sequential: ~5.7x

Output files:
  data/processed/nq_features/features_11d.npy
  data/processed/nq_features/valid_mask.npy
  data/processed/nq_features/metadata.json
  data/processed/nq_features/splits.json

Next steps:
  1. Verify features: python3 scripts/verify_precomputed_features.py
  2. Check non-zero density (target >50%)
  3. Train Jade model with uncertainty weighting
```

---

### 7. Verify Features

```bash
# Run verification script
python3 scripts/verify_precomputed_features.py

# Check outputs manually
python3 << 'EOF'
import numpy as np
import json

# Load features
X = np.load('data/processed/nq_features/features_11d.npy')
mask = np.load('data/processed/nq_features/valid_mask.npy')

# Load metadata
with open('data/processed/nq_features/metadata.json') as f:
    meta = json.load(f)

print(f"✅ Feature shape: {X.shape}")
print(f"✅ Expected: (N_windows, 105, 11)")
print(f"✅ Valid ratio: {mask.mean():.3f}")
print(f"✅ Non-zero density: {(X != 0).mean():.3f}")
print(f"\nFeature names:")
for i, name in enumerate(meta['feature_names']):
    print(f"  {i}: {name}")

# Check expansion_proxy exists
assert 'expansion_proxy' in meta['feature_names'], "expansion_proxy missing!"
print("\n✅ expansion_proxy feature present")
EOF
```

**Expected checks:**
- ✅ Feature shape: `(~1.8M, 105, 11)`
- ✅ 11 features (includes expansion_proxy)
- ✅ Valid ratio > 0.90
- ✅ Non-zero density > 0.50 (Grok's target)

---

### 8. Download Results to Mac

```bash
# From Mac terminal:
scp -P 12345 -r \
  root@104.171.203.45:/workspace/moola/data/processed/nq_features/ \
  /Users/jack/projects/moola/data/processed/

# Should download ~8GB of data
# Takes 5-10 minutes depending on bandwidth
```

---

### 9. Clean Up RunPod

```bash
# Stop the pod from RunPod dashboard
# → Pods → Your Pod → Stop

# Or terminate if you're done:
# → Pods → Your Pod → Terminate
```

**Cost estimate:**
- Pre-computation: 6 minutes @ $0.35/hr = **$0.035**
- Setup + verification: 10 minutes @ $0.35/hr = **$0.058**
- Download time: 10 minutes @ $0.35/hr = **$0.058**
- **Total: ~$0.15** (15 cents)

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'moola'"
```bash
# Fix: Add src to PYTHONPATH
export PYTHONPATH=/workspace/moola/src:$PYTHONPATH

# Or install in editable mode:
cd /workspace/moola
pip3 install -e .
```

### Problem: "ImportError: No module named 'ta'"
```bash
# Install TA-Lib
pip3 install ta-lib

# If that fails, install from source:
wget https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
make install
pip3 install ta-lib
```

### Problem: "MemoryError" or "Out of memory"
```bash
# Reduce n_jobs:
python3 scripts/precompute_nq_features_parallel.py \
  --n-jobs 16  # Instead of 32

# Or increase overlap (less memory per chunk):
python3 scripts/precompute_nq_features_parallel.py \
  --n-jobs 32 \
  --overlap-bars 20  # Instead of 30
```

### Problem: Slow processing (<1000 bars/s)
```bash
# Check CPU usage:
htop

# If CPU usage is low:
# 1. Check for I/O bottleneck (disk speed)
# 2. Try with fewer workers:
--n-jobs 16

# If CPU usage is high but slow:
# CPU might not have turbo boost enabled
# Try sequential script instead (more predictable)
```

---

## 📊 Performance Benchmarks

| Pod Type | vCPUs | Clock | Time | Cost | Bars/s |
|----------|-------|-------|------|------|--------|
| EPYC 9754 | 32 | 5.2 GHz | 6 min | $0.035 | 5700 |
| EPYC 9654 | 16 | 4.8 GHz | 10 min | $0.033 | 3400 |
| Xeon | 32 | 4.5 GHz | 12 min | $0.056 | 2800 |
| Sequential (baseline) | 8 | 4.5 GHz | 30-45 min | $0.21-0.26 | 900-1200 |

**Speedup:** 5-6x vs sequential on high-compute pods

---

## ✅ Success Checklist

Before leaving RunPod:
- [ ] `features_11d.npy` exists (7-8 GB)
- [ ] `valid_mask.npy` exists (180+ MB)
- [ ] `metadata.json` shows 11 features
- [ ] `expansion_proxy` in feature names
- [ ] Non-zero density > 50%
- [ ] Valid ratio > 90%
- [ ] Files downloaded to Mac
- [ ] RunPod pod stopped/terminated

---

## 🎯 Next Steps

After successful pre-computation:

1. **Local validation:**
   ```bash
   python3 scripts/verify_precomputed_features.py
   ```

2. **Update training code:**
   - Implement WeightedRandomSampler
   - Connect uncertainty-weighted loss
   - Test on 10% of data

3. **Full training on RunPod GPU:**
   - Deploy to A100 GPU pod
   - Load pre-computed features (5s vs 1-3h)
   - Train Jade with anti-collapse pipeline
   - Target: 60-75% F1_macro

Ready to deploy! 🚀
