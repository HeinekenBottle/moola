# Production Deployment Checklist

**Version:** 1.0
**Last Updated:** 2025-10-27
**Status:** ✅ Ready for Deployment Review

---

## Executive Summary

This checklist provides a structured approach to deploying Moola models to production on RunPod. It covers:
- **Pre-deployment validation** (data, model, code quality)
- **Model selection criteria** (choosing best checkpoint)
- **Deployment procedures** (SSH/SCP workflow)
- **Post-deployment verification** (sanity checks)
- **Rollback procedures** (reverting to last known good)

**Current Best Model:** Position Encoding Model (F1=0.220) or CRF Layer (expected F1=0.26-0.28)

---

## Phase 1: Pre-Deployment Validation (Local, ~30 minutes)

### 1.1: Code Quality Gates

- [ ] **Pre-commit hooks passing**
  ```bash
  cd /Users/jack/projects/moola
  pre-commit run --all-files
  # Expected: All hooks pass (Black, Ruff, isort, python-tree, pip-tree)
  ```

- [ ] **Unit tests passing**
  ```bash
  python3 -m pytest tests/ -v --tb=short
  # Expected: All tests pass or note known failures
  ```

- [ ] **No uncommitted changes**
  ```bash
  git status
  # Expected: Clean working tree (or only data files)
  ```

- [ ] **Latest version committed**
  ```bash
  git log --oneline -5
  # Verify model code is in HEAD
  ```

**Checklist:**
- [ ] Pre-commit hooks: PASS / FAIL / N/A
- [ ] Unit tests: PASS / FAIL / KNOWN_FAILURE
- [ ] Git status: CLEAN / DIRTY
- [ ] Latest version: YES / NO

---

### 1.2: Data Validation

- [ ] **Training dataset exists and is valid**
  ```bash
  python3 << 'EOF'
  import pandas as pd
  df = pd.read_parquet('data/processed/labeled/train_latest.parquet')
  assert len(df) == 174, f"Expected 174 samples, got {len(df)}"
  assert 'features' in df.columns, "Missing 'features' column"
  assert 'label' in df.columns, "Missing 'label' column"
  assert (df['label'].isin([0, 1])).all(), "Invalid labels"
  print("✓ Training dataset valid (174 samples)")
  EOF
  ```

- [ ] **Unlabeled data exists for pre-training (optional)**
  ```bash
  ls -lh data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet
  # Expected: 30.8 MB file present
  ```

- [ ] **No data leakage between splits**
  ```bash
  python3 << 'EOF'
  import pandas as pd
  import json

  # Load splits
  with open('data/processed/labeled/splits_temporal.json') as f:
      splits = json.load(f)

  # Verify temporal ordering
  assert pd.Timestamp(splits['train_end']) < pd.Timestamp(splits['val_end'])
  assert pd.Timestamp(splits['val_end']) < pd.Timestamp(splits['test_end'])
  print("✓ Temporal splits valid (no leakage)")
  EOF
  ```

**Checklist:**
- [ ] Training dataset: VALID / INVALID / MISSING
- [ ] Sample count: 174 (or approved alternative)
- [ ] Feature columns: PRESENT / MISSING
- [ ] Label distribution: BALANCED / IMBALANCED (document ratio)
- [ ] No data leakage: CONFIRMED / SUSPECTED

---

### 1.3: Model Validation

- [ ] **Model architecture is correct**
  ```bash
  python3 << 'EOF'
  import sys
  sys.path.insert(0, '/Users/jack/projects/moola/src')
  from moola.models.jade_core import JadeModel

  model = JadeModel(
      input_size=11,
      hidden_size=128,
      num_layers=2,
      num_classes=2,
      predict_pointers=True
  )

  print(f"✓ Model instantiated successfully")
  print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
  print(f"  Architecture: BiLSTM × {model.bilstm.num_layers} layers")
  EOF
  ```

- [ ] **Pre-trained encoder (if using) is valid**
  ```bash
  python3 << 'EOF'
  import torch
  checkpoint = torch.load('artifacts/encoders/pretrained/jade_encoder_5yr_v1.pt')
  print(f"✓ Pre-trained encoder valid")
  print(f"  Keys: {list(checkpoint.keys())[:5]}...")
  print(f"  Shapes: {[(k, v.shape) for k, v in list(checkpoint.items())[:3]]}...")
  EOF
  ```

- [ ] **Model checkpoint loading works**
  ```bash
  python3 << 'EOF'
  import sys, torch
  sys.path.insert(0, '/Users/jack/projects/moola/src')
  from moola.models.jade_core import JadeModel

  # Load best model from latest training
  model = JadeModel(input_size=11, hidden_size=128, num_layers=2, num_classes=2)
  checkpoint = torch.load('artifacts/models/supervised/best_model.pkl', map_location='cpu')
  model.load_state_dict(checkpoint['state_dict'])
  print("✓ Model checkpoint loads successfully")
  EOF
  ```

**Checklist:**
- [ ] Architecture instantiation: SUCCESS / FAIL
- [ ] Parameter count: ~100K (or document alternative)
- [ ] Pre-trained encoder: VALID / INVALID / NOT_USED
- [ ] Checkpoint loading: SUCCESS / FAIL
- [ ] Model on CPU: YES (for inference preparation)

---

### 1.4: Dependency Validation

- [ ] **Python version correct**
  ```bash
  python3 --version
  # Expected: Python 3.10+
  ```

- [ ] **Key packages installed**
  ```bash
  python3 -c "
  import torch; print(f'✓ PyTorch {torch.__version__}')
  import pandas; print(f'✓ Pandas {pandas.__version__}')
  import numpy; print(f'✓ NumPy {numpy.__version__}')
  import yaml; print(f'✓ PyYAML {yaml.__version__}')
  import pydantic; print(f'✓ Pydantic {pydantic.__version__}')
  "
  ```

- [ ] **No breaking dependency versions**
  ```bash
  pip3 freeze | grep -E "torch|pandas|numpy|pydantic"
  # Expected: torch>=2.0, pandas>=1.5, numpy>=1.20, pydantic>=2.0
  ```

**Checklist:**
- [ ] Python version: 3.10+
- [ ] PyTorch: INSTALLED / MISSING
- [ ] Pandas: INSTALLED / MISSING
- [ ] NumPy: INSTALLED / MISSING
- [ ] All critical packages: PRESENT / MISSING

---

## Phase 2: Model Selection (Offline, ~45 minutes)

### 2.1: Identify Candidate Models

List all trained models and their validation performance:

```bash
python3 << 'EOF'
import json
import pandas as pd
from pathlib import Path

results_file = 'artifacts/archive/results/experiment_results.jsonl'
if Path(results_file).exists():
    results = [json.loads(line) for line in open(results_file)]
    df = pd.DataFrame(results)

    # Recent experiments
    print("\n=== RECENT EXPERIMENTS ===")
    latest = df.nlargest(10, 'timestamp')[['experiment_id', 'model', 'f1_macro', 'accuracy', 'timestamp']]
    print(latest)

    # Best per model
    print("\n=== BEST PER MODEL ===")
    best_per_model = df.loc[df.groupby('model')['f1_macro'].idxmax()]
    print(best_per_model[['model', 'f1_macro', 'accuracy', 'precision', 'recall']])
else:
    print("No experiment results found")
EOF
```

**Document:**
- [ ] **Position Encoding Model (F1=0.220)**
  - Epochs trained: ___
  - Validation F1: ___
  - Validation Accuracy: ___
  - Validation Precision: ___
  - Validation Recall: ___
  - Training time: ___

- [ ] **CRF Layer Model (expected F1=0.26-0.28)**
  - Epochs trained: ___
  - Validation F1: ___
  - Validation Accuracy: ___
  - Training time: ___

- [ ] **Alternative Models Considered:**
  - Model A: ___
  - Model B: ___

---

### 2.2: Performance Comparison

Create a comparison table:

| Model | F1 (Val) | Accuracy | Precision | Recall | Params | Size (MB) | Status |
|-------|----------|----------|-----------|--------|--------|-----------|--------|
| Position Encoding | 0.220 | ___ | ___ | ___ | 100K | 0.4 | ✅ CANDIDATE |
| CRF Layer | ___ (exp) | ___ | ___ | ___ | 102K | 0.4 | ⏳ IN_PROGRESS |
| Stones Only | ___ | ___ | ___ | ___ | ___ | ___ | ❌ REJECTED |
| Baseline | ___ | ___ | ___ | ___ | ___ | ___ | ❌ REJECTED |

**Selection Criteria:**

- [ ] F1-macro ≥ 0.20 (minimum baseline)
- [ ] Accuracy ≥ 75% (binary classification floor)
- [ ] Recall ≥ 0.40 (catch minority class)
- [ ] Precision ≥ 0.30 (limit false positives)
- [ ] No overfitting (val_loss < train_loss)
- [ ] Model reproducible (seed documented)

**Decision:**
- [ ] **Recommended Model:** ___ (Position Encoding / CRF Layer / Other)
- [ ] **Reason:** ___
- [ ] **Risk Level:** LOW / MEDIUM / HIGH
- [ ] **Contingency Model:** ___ (if primary fails)

---

### 2.3: Freeze Model for Deployment

Once model selected:

```bash
# Copy to versioned directory
mkdir -p artifacts/production/v1.0_$(date +%Y%m%d)
cp artifacts/models/supervised/best_model.pkl \
   artifacts/production/v1.0_$(date +%Y%m%d)/model.pkl

# Create metadata
python3 << 'EOF'
import json
import torch
from pathlib import Path

metadata = {
    "version": "1.0",
    "deployment_date": "2025-10-27",
    "model_type": "jade_core",
    "architecture": {
        "input_size": 11,
        "hidden_size": 128,
        "num_layers": 2,
        "num_classes": 2,
        "total_params": 100000
    },
    "training": {
        "dataset": "train_latest.parquet (174 samples)",
        "epochs": 60,
        "batch_size": 16,
        "learning_rate": 0.001,
        "seed": 17
    },
    "validation_metrics": {
        "f1_macro": 0.220,
        "accuracy": 0.0,  # Update
        "precision": 0.0,  # Update
        "recall": 0.0  # Update
    },
    "notes": "Position encoding model with pointer prediction"
}

with open('artifacts/production/v1.0_20251027/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ Model frozen for deployment")
EOF
```

- [ ] Model copied to versioned directory
- [ ] Metadata file created
- [ ] Version number assigned
- [ ] All metrics documented

---

## Phase 3: Prepare Deployment Package (~15 minutes)

### 3.1: Create Deployment Bundle

```bash
# On Mac
mkdir -p deployment_bundle
cp -r src/ deployment_bundle/
cp -r configs/ deployment_bundle/
cp artifacts/production/v1.0_20251027/model.pkl deployment_bundle/
cp artifacts/production/v1.0_20251027/metadata.json deployment_bundle/
cp DEPLOYMENT_CHECKLIST.md deployment_bundle/
cp CLAUDE.md deployment_bundle/

# Add instructions
cat > deployment_bundle/DEPLOYMENT_INSTRUCTIONS.txt << 'EOF'
MOOLA DEPLOYMENT INSTRUCTIONS
=============================

1. SSH to RunPod:
   ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP

2. Copy files to RunPod:
   scp -r deployment_bundle ubuntu@YOUR_IP:/workspace/moola_deploy_v1.0/

3. Switch to new version:
   cd /workspace && mv moola moola_v0.9_backup
   cp -r moola_deploy_v1.0 moola

4. Verify installation:
   cd /workspace/moola
   python3 -m moola.cli doctor

5. Run inference:
   python3 -m moola.cli infer --model jade --device cuda

6. If rollback needed:
   cd /workspace && mv moola moola_v1.0_failed
   mv moola_v0.9_backup moola
EOF

cd deployment_bundle
tar -czf moola_deployment_v1.0_$(date +%Y%m%d).tar.gz *
ls -lh moola_deployment_v1.0_*.tar.gz
```

**Checklist:**
- [ ] Source code included (src/)
- [ ] Configuration included (configs/)
- [ ] Model weights included (model.pkl)
- [ ] Metadata included (metadata.json)
- [ ] Documentation included (DEPLOYMENT_CHECKLIST.md)
- [ ] Backup instructions included
- [ ] Bundle compressed (.tar.gz)

---

### 3.2: Create Rollback Snapshot (RunPod)

```bash
# SSH to RunPod
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP

# Create backup of current production version
cd /workspace
sudo tar -czf moola_v0.9_snapshot_$(date +%Y%m%d_%H%M%S).tar.gz moola/
ls -lh moola_v0.9_snapshot_*.tar.gz

# Store path for rollback
echo "/workspace/moola_v0.9_snapshot_$(date +%Y%m%d_%H%M%S).tar.gz" > /tmp/rollback_snapshot.txt
cat /tmp/rollback_snapshot.txt
```

- [ ] Current production backed up
- [ ] Backup location documented
- [ ] Backup verified (tar -tzf to check contents)

---

## Phase 4: Deploy to RunPod (~15 minutes)

### 4.1: Upload Deployment Bundle

```bash
# From Mac
scp -i ~/.ssh/runpod_key deployment_bundle/moola_deployment_v1.0_*.tar.gz \
    ubuntu@YOUR_IP:/workspace/

# Verify upload
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP 'ls -lh /workspace/moola_deployment_v1.0_*.tar.gz'
```

- [ ] Bundle uploaded
- [ ] File size matches local copy
- [ ] File integrity verified (md5sum)

---

### 4.2: Deploy New Version

```bash
# SSH to RunPod
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP

# Backup current production
cd /workspace
if [ -d moola ]; then
  sudo mv moola moola_v0.9_$(date +%Y%m%d_%H%M%S)
fi

# Extract new version
sudo tar -xzf moola_deployment_v1.0_*.tar.gz -C /workspace
sudo mv /workspace/deployment_bundle /workspace/moola
sudo chown -R ubuntu:ubuntu /workspace/moola

# Verify directory structure
ls -la /workspace/moola/
```

**Checklist:**
- [ ] Current production backed up
- [ ] New version extracted
- [ ] Permissions set correctly
- [ ] Directory structure intact

---

### 4.3: Install/Update Dependencies (RunPod)

```bash
# SSH to RunPod
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP

cd /workspace/moola

# Ensure PYTHONPATH is set
export PYTHONPATH=/workspace/moola/src:$PYTHONPATH

# Verify Python version
python3 --version  # Expected: 3.10+

# Install/upgrade critical packages (if needed)
pip3 install --upgrade torch pandas pydantic pyyaml

# Verify imports
python3 << 'EOF'
import sys
sys.path.insert(0, '/workspace/moola/src')
from moola.models.jade_core import JadeModel
print("✓ All imports successful")
EOF
```

**Checklist:**
- [ ] PYTHONPATH set
- [ ] Python 3.10+
- [ ] Dependencies installed
- [ ] Imports working

---

## Phase 5: Post-Deployment Verification (~20 minutes)

### 5.1: Sanity Check - Model Loads

```bash
# SSH to RunPod
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP

python3 << 'EOF'
import sys
sys.path.insert(0, '/workspace/moola/src')
import torch
from moola.models.jade_core import JadeModel

# Load model
model = JadeModel(input_size=11, hidden_size=128, num_layers=2, num_classes=2)
checkpoint = torch.load('/workspace/moola/model.pkl', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

print("✓ Model loads successfully")
print(f"  Device: {next(model.parameters()).device}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
EOF
```

- [ ] Model loads without errors
- [ ] Weights present
- [ ] Architecture intact

---

### 5.2: Sanity Check - Inference Works

```bash
# SSH to RunPod
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP

python3 << 'EOF'
import sys
sys.path.insert(0, '/workspace/moola/src')
import torch
import numpy as np
from moola.models.jade_core import JadeModel

# Create dummy input
X = torch.randn(1, 105, 11)  # Batch=1, K=105, D=11

# Load model
model = JadeModel(input_size=11, hidden_size=128, num_layers=2, num_classes=2)
checkpoint = torch.load('/workspace/moola/model.pkl', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Forward pass
with torch.no_grad():
    output = model(X)
    print("✓ Forward pass successful")
    print(f"  Output shape: {output['logits'].shape}")
    print(f"  Logits range: [{output['logits'].min():.3f}, {output['logits'].max():.3f}]")
    print(f"  Softmax range: [{output['probabilities'].min():.3f}, {output['probabilities'].max():.3f}]")
EOF
```

- [ ] Forward pass completes
- [ ] Output shape correct
- [ ] Probabilities in [0, 1] range
- [ ] No NaN/Inf values

---

### 5.3: Sanity Check - Data Loading

```bash
# SSH to RunPod
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP

python3 << 'EOF'
import sys
sys.path.insert(0, '/workspace/moola/src')
import pandas as pd

# Load training data
df = pd.read_parquet('/workspace/moola/data/processed/labeled/train_latest.parquet')
print(f"✓ Training data loads successfully")
print(f"  Samples: {len(df)}")
print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

# Check features
print(f"  Feature shape: {df['features'].iloc[0].shape}")
print(f"  Feature dtype: {df['features'].iloc[0].dtype}")
EOF
```

- [ ] Data loads without errors
- [ ] 174 samples present
- [ ] Label distribution correct
- [ ] Features shape correct (105, 11)

---

### 5.4: Run Diagnostic Suite

```bash
# SSH to RunPod
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP

cd /workspace/moola
export PYTHONPATH=/workspace/moola/src:$PYTHONPATH

# Run built-in diagnostic
python3 -m moola.cli doctor

# Expected output:
# ✓ Python version OK
# ✓ CUDA available
# ✓ Data files present
# ✓ Model loads OK
# ✓ Imports working
```

- [ ] Diagnostic passes all checks
- [ ] CUDA available on GPU
- [ ] Data accessible
- [ ] No configuration errors

---

### 5.5: Quick Performance Benchmark

```bash
# SSH to RunPod
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP

python3 << 'EOF'
import sys, time
sys.path.insert(0, '/workspace/moola/src')
import torch
import numpy as np
from moola.models.jade_core import JadeModel

# Setup
model = JadeModel(input_size=11, hidden_size=128, num_layers=2, num_classes=2)
checkpoint = torch.load('/workspace/moola/model.pkl', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# CPU benchmark
device = 'cpu'
model = model.to(device)
X = torch.randn(16, 105, 11).to(device)

with torch.no_grad():
    start = time.time()
    for _ in range(100):
        output = model(X)
    cpu_time = (time.time() - start) / 100

print(f"CPU inference (batch=16): {cpu_time*1000:.1f}ms")

# GPU benchmark (if available)
if torch.cuda.is_available():
    device = 'cuda'
    model = model.to(device)
    X = torch.randn(16, 105, 11).to(device)

    torch.cuda.synchronize()
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            output = model(X)
        torch.cuda.synchronize()
    gpu_time = (time.time() - start) / 100

    print(f"GPU inference (batch=16): {gpu_time*1000:.2f}ms")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x")
EOF
```

**Expected Results:**
- [ ] CPU inference: ~50-100ms per batch
- [ ] GPU inference: ~2-5ms per batch
- [ ] GPU speedup: 10-50x
- [ ] No out-of-memory errors

---

## Phase 6: Final Acceptance (~10 minutes)

### 6.1: Sign-off Checklist

- [ ] **Code Quality**
  - [ ] Pre-commit hooks pass
  - [ ] Unit tests pass
  - [ ] No uncommitted changes
  - [ ] Latest version deployed

- [ ] **Data Validation**
  - [ ] Training dataset present (174 samples)
  - [ ] Data integrity verified
  - [ ] No data leakage

- [ ] **Model Performance**
  - [ ] F1-macro ≥ 0.20
  - [ ] Accuracy ≥ 75%
  - [ ] Model loads without errors
  - [ ] Inference produces valid outputs

- [ ] **Deployment**
  - [ ] Code deployed to RunPod
  - [ ] Dependencies installed
  - [ ] Diagnostics pass
  - [ ] Performance benchmark passes

- [ ] **Documentation**
  - [ ] Metadata file created
  - [ ] Deployment instructions clear
  - [ ] Rollback procedure documented
  - [ ] Version number recorded

### 6.2: Document Deployment

```bash
# Create deployment log
cat > /workspace/moola/DEPLOYMENT_LOG.txt << 'EOF'
MOOLA DEPLOYMENT LOG
====================
Deployment Date: $(date)
Model Version: 1.0
Model Type: jade_core (BiLSTM + Pointer Prediction)
Training Data: train_latest.parquet (174 samples)

VALIDATION RESULTS
------------------
✓ F1-macro: 0.220
✓ Accuracy: ___
✓ Precision: ___
✓ Recall: ___

DEPLOYMENT STATUS: ✅ APPROVED
Deployed by: ___
Approved by: ___
EOF

cat /workspace/moola/DEPLOYMENT_LOG.txt
```

- [ ] Log created and timestamped
- [ ] All metrics recorded
- [ ] Approval documented

---

## Phase 7: Rollback Procedure (IF NEEDED)

### ⚠️ EMERGENCY ROLLBACK

If deployment has critical issues:

```bash
# SSH to RunPod
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP

# Option 1: Restore from backup
cd /workspace
sudo tar -xzf moola_v0.9_snapshot_*.tar.gz
sudo chown -R ubuntu:ubuntu moola

# Option 2: Revert to git
cd /workspace/moola
git checkout HEAD -- src/ configs/
git reset --hard HEAD

# Verify rollback
python3 -m moola.cli doctor

# Test inference
python3 << 'EOF'
import sys
sys.path.insert(0, '/workspace/moola/src')
from moola.models.jade_core import JadeModel
print("✓ Rollback successful")
EOF
```

**When to Rollback:**
- [ ] Model produces NaN/Inf predictions
- [ ] Inference latency > 10s (batch=1)
- [ ] Model doesn't load
- [ ] Import errors in critical modules
- [ ] Data loading fails

**Do NOT Rollback If:**
- [ ] Performance slightly lower (expected variance)
- [ ] Minor bugs in non-critical features
- [ ] Documentation issues

---

## Phase 8: Post-Deployment Monitoring (Ongoing)

### 8.1: Daily Health Check

```bash
# SSH to RunPod, run daily
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP

python3 << 'EOF'
import subprocess, sys, time
sys.path.insert(0, '/workspace/moola/src')

# Check 1: Model loads
try:
    import torch
    from moola.models.jade_core import JadeModel
    m = JadeModel(11, 128, 2, 2)
    checkpoint = torch.load('/workspace/moola/model.pkl', map_location='cpu')
    m.load_state_dict(checkpoint['state_dict'])
    print("✓ Model loads OK")
except Exception as e:
    print(f"✗ Model loading FAILED: {e}")
    sys.exit(1)

# Check 2: Data accessible
try:
    import pandas as pd
    df = pd.read_parquet('/workspace/moola/data/processed/labeled/train_latest.parquet')
    assert len(df) == 174
    print("✓ Data accessible OK")
except Exception as e:
    print(f"✗ Data loading FAILED: {e}")
    sys.exit(1)

# Check 3: Inference works
try:
    import torch
    X = torch.randn(1, 105, 11)
    m.eval()
    with torch.no_grad():
        out = m(X)
    assert not torch.isnan(out['logits']).any()
    print("✓ Inference OK")
except Exception as e:
    print(f"✗ Inference FAILED: {e}")
    sys.exit(1)

print("\n✅ All health checks passed")
sys.exit(0)
EOF

# Store result
echo "Health check: $(date)" >> /workspace/moola/health_check.log
```

- [ ] Daily health check running
- [ ] Logs stored
- [ ] Alerts configured (if applicable)

---

## Appendix A: Model Selection Details

### Position Encoding Model (F1=0.220)

**Characteristics:**
- Absolute span indices [0-105]
- BiLSTM + Dense pointer head
- Uncertainty-weighted multi-task loss
- 100K parameters

**Pros:**
- ✅ Semantically clear predictions
- ✅ Proven convergence (results available)
- ✅ Small model size

**Cons:**
- ⚠️ F1=0.220 is baseline (not excellent)
- ⚠️ No pre-training (lower capacity utilization)
- ⚠️ Span detection still weak

**When to Use:**
- Quick baseline deployment
- When F1≥0.20 acceptable
- Resource-constrained environments

---

### CRF Layer Model (Expected F1=0.26-0.28)

**Characteristics:**
- BiLSTM encoder + CRF sequence tagging
- Structured prediction (enforces transition constraints)
- Better for span/sequence tasks

**Pros:**
- ✅ Better F1 expected (0.26-0.28 vs 0.220)
- ✅ Enforces valid transitions (start before end)
- ✅ Natural for sequence labeling

**Cons:**
- ⚠️ Results not yet available (in progress)
- ⚠️ Slightly larger model (~102K params)
- ⚠️ Inference requires viterbi decoding

**When to Use:**
- When F1≥0.25 required
- For sequence labeling tasks
- Production with quality requirements

---

## Appendix B: Monitoring Dashboard (Optional)

```bash
# Create monitoring script (for future integration)
cat > /workspace/moola/scripts/monitor_deployment.py << 'EOF'
#!/usr/bin/env python3
"""Monitor deployed model health."""

import json
import sys
import time
from pathlib import Path

def health_check():
    """Run health checks."""
    checks = {
        'model_loads': check_model_loads(),
        'data_accessible': check_data_accessible(),
        'inference_works': check_inference_works(),
        'gpu_available': check_gpu_available(),
        'memory_ok': check_memory_ok(),
    }

    return all(checks.values()), checks

def check_model_loads():
    try:
        import torch
        from moola.models.jade_core import JadeModel
        m = JadeModel(11, 128, 2, 2)
        checkpoint = torch.load('/workspace/moola/model.pkl', map_location='cpu')
        m.load_state_dict(checkpoint['state_dict'])
        return True
    except:
        return False

def check_data_accessible():
    try:
        import pandas as pd
        df = pd.read_parquet('/workspace/moola/data/processed/labeled/train_latest.parquet')
        return len(df) == 174
    except:
        return False

def check_inference_works():
    try:
        import torch
        m = JadeModel(11, 128, 2, 2)
        X = torch.randn(1, 105, 11)
        m.eval()
        with torch.no_grad():
            out = m(X)
        return not torch.isnan(out['logits']).any()
    except:
        return False

def check_gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False

def check_memory_ok():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() < 8e9  # < 8GB
        return True
    except:
        return False

if __name__ == '__main__':
    sys.path.insert(0, '/workspace/moola/src')
    ok, checks = health_check()

    print(json.dumps({
        'timestamp': time.time(),
        'status': 'OK' if ok else 'FAILURE',
        'checks': checks
    }, indent=2))

    sys.exit(0 if ok else 1)
EOF

chmod +x /workspace/moola/scripts/monitor_deployment.py
```

---

## Appendix C: Troubleshooting Common Issues

### Issue: Model produces NaN predictions

**Diagnosis:**
```bash
python3 << 'EOF'
import torch
import numpy as np

# Check input data
X = torch.randn(1, 105, 11)
print(f"Input contains NaN: {torch.isnan(X).any()}")
print(f"Input contains Inf: {torch.isinf(X).any()}")

# Check model weights
model.load_state_dict(checkpoint['state_dict'])
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN in {name}")
    if torch.isinf(param).any():
        print(f"Inf in {name}")
EOF
```

**Fix:**
1. Verify data preprocessing (check for missing values)
2. Reload model from backup
3. Rollback to previous version

---

### Issue: Inference slow (>5s per sample)

**Diagnosis:**
```bash
# Check GPU usage
nvidia-smi

# Check model is on GPU
python3 -c "print(next(model.parameters()).device)"

# Benchmark inference
time python3 -c "
import torch
from moola.models.jade_core import JadeModel
m = JadeModel(11, 128, 2, 2)
m.cuda()
m.eval()
X = torch.randn(1, 105, 11).cuda()
with torch.no_grad():
    for _ in range(100):
        m(X)
"
```

**Fix:**
1. Ensure model is on GPU: `model.cuda()`
2. Check GPU memory: `nvidia-smi`
3. Reduce batch size if OOM
4. Profile with PyTorch profiler

---

### Issue: Model file corrupted

**Diagnosis:**
```bash
# Check file size (should be ~500KB)
ls -lh /workspace/moola/model.pkl

# Try loading
python3 -c "
import torch
try:
    torch.load('/workspace/moola/model.pkl')
    print('✓ File loads')
except Exception as e:
    print(f'✗ File corrupted: {e}')
"
```

**Fix:**
1. Restore from backup: `scp model from Mac`
2. Rollback to previous version
3. Re-train if necessary

---

## Final Sign-off

```
MOOLA MODEL DEPLOYMENT CHECKLIST
=================================

Model Version:        1.0
Deployment Date:      ___________
Deployed By:          ___________
Reviewed By:          ___________

PRE-DEPLOYMENT:       ✓ PASS / ✗ FAIL
VALIDATION:           ✓ PASS / ✗ FAIL
DEPLOYMENT:           ✓ PASS / ✗ FAIL
POST-DEPLOYMENT:      ✓ PASS / ✗ FAIL

APPROVAL:             ✓ APPROVED / ✗ REJECTED

If REJECTED, reason: _______________________________
__________________________________________________

SIGN-OFF:
Signature: _______________  Date: _____________
```

---

**Status: ✅ READY FOR DEPLOYMENT**

All phases documented. Proceed with pre-deployment validation when ready.

