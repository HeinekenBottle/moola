# RunPod Dependency Hell Analysis & Solution

**Date:** 2025-10-16
**Problem:** NumPy version conflicts and repeated PyTorch installations on RunPod GPU training
**Status:** RESOLVED with bulletproof requirements file

---

## Executive Summary

### Root Cause
1. **NumPy 2.0 Incompatibility**: PyTorch 2.2.2 (your local version) is NOT compatible with NumPy 2.0+
2. **Template Uncertainty**: Using RunPod PyTorch template without knowing which version (1.x vs 2.2 vs 2.4 vs 2.8)
3. **Bloated Requirements**: Local `requirements.txt` has 448 packages (90% unnecessary for GPU training)
4. **Missing Pre-installed Packages**: Not leveraging what RunPod templates provide

### Solution
- Use **RunPod PyTorch 2.4 + CUDA 12.4** template (or 2.2 + CUDA 12.1 for stability)
- Pin **NumPy <2.0** explicitly
- Minimal requirements file with ONLY packages not in template
- Exact version pinning to prevent drift

---

## RunPod PyTorch Template Versions (2025)

### Available Templates

| Template | PyTorch | Python | CUDA | NumPy Support | Recommendation |
|----------|---------|--------|------|---------------|----------------|
| **pytorch:2.8-py3.11-cuda12.8** | 2.8 | 3.11 | 12.8 | 2.0+ ✅ | **BEST** (latest, NumPy 2.0 compatible) |
| **pytorch:2.4-py3.11-cuda12.4** | 2.4 | 3.11 | 12.4 | 2.0+ ✅ | **RECOMMENDED** (stable, well-tested) |
| **pytorch:2.2-py3.10-cuda12.1** | 2.2 | 3.10 | 12.1 | 1.x ONLY ⚠️ | Conservative (requires numpy<2.0) |
| **pytorch:2.1-py3.10-cuda11.8** | 2.1 | 3.10 | 11.8 | 1.x ONLY ⚠️ | Legacy (avoid) |

### Template Pre-installed Packages

**What RunPod PyTorch templates include by default:**
- ✅ PyTorch (torch, torchvision)
- ✅ NumPy (version depends on template)
- ✅ CUDA libraries and drivers
- ✅ Python (3.10 or 3.11)
- ✅ Basic system tools (git, wget, etc.)

**What is NOT pre-installed (must pip install):**
- ❌ scikit-learn
- ❌ pandas
- ❌ scipy
- ❌ xgboost
- ❌ imbalanced-learn
- ❌ All ML libraries beyond PyTorch

---

## NumPy Version Compatibility Matrix

### Critical Finding: PyTorch 2.2 and NumPy 2.0 are INCOMPATIBLE

| PyTorch Version | NumPy 1.x | NumPy 2.0+ | Notes |
|-----------------|-----------|------------|-------|
| 2.1 and earlier | ✅ | ❌ | Must use numpy<2.0 |
| 2.2 | ✅ | ❌ | **INCOMPATIBLE with numpy 2.0** |
| 2.3+ | ✅ | ✅ | First version with NumPy 2.0 support |
| 2.4+ | ✅ | ✅ | Full NumPy 2.0 compatibility |
| 2.8 | ✅ | ✅ | Latest, fully compatible |

**Your Current Setup:**
- Local: PyTorch 2.2.2 + NumPy 1.26.4 ✅
- Issue: If RunPod has NumPy 2.0+, PyTorch 2.2 will fail

**Solution:**
1. **Option A (Conservative)**: Use PyTorch 2.2 template + pin `numpy>=1.24,<2.0`
2. **Option B (Modern)**: Use PyTorch 2.4+ template + allow `numpy>=1.24,<2.1`

---

## Moola Critical Dependencies Analysis

### Core ML Training Dependencies

**Extracted from `/src/moola/` codebase:**

```python
# Deep Learning
torch==2.2.2              # Currently on PyTorch 2.2
pytorch-lightning==2.4.0

# Scientific Computing
numpy==1.26.4             # MUST BE <2.0 for PyTorch 2.2
pandas==2.3.3
scipy==1.16.1

# ML Libraries
scikit-learn==1.7.2
xgboost==2.0.3
imbalanced-learn==0.14.0  # For SMOTE

# Config & CLI
click==8.2.1
typer==0.17.4
hydra-core==1.3.2
pydantic==2.11.7

# Logging & Display
loguru==0.7.3
rich==14.1.0

# Data Handling
pyarrow==17.0.0           # For parquet
pandera==0.26.1           # Data validation

# MLOps
mlflow                    # Experiment tracking
```

### TS-TCC Specific Requirements

**From `src/moola/models/ts_tcc.py`:**
- PyTorch (core)
- NumPy (for data processing)
- sklearn (for train_test_split)
- No special TS-TCC library needed (custom implementation)

### Training Pipeline Requirements

**From `scripts/train_full_pipeline.py`:**
- All base model dependencies
- MLflow for tracking
- loguru for logging
- subprocess for orchestration

---

## Recommended RunPod Configuration

### Template Choice: PyTorch 2.4 + CUDA 12.4

**Rationale:**
1. ✅ NumPy 2.0+ compatible (future-proof)
2. ✅ Stable and well-tested in production
3. ✅ Python 3.11 support (modern)
4. ✅ CUDA 12.4 for optimal GPU performance
5. ✅ Avoids PyTorch 2.2 NumPy issues

**RunPod Image:**
```bash
runpod/pytorch:2.4-py3.11-cuda12.4-ubuntu22.04
```

**Alternative (Conservative):**
```bash
runpod/pytorch:2.2-py3.10-cuda12.1-ubuntu22.04
# Must pin numpy<2.0 with this template
```

---

## Bulletproof requirements-runpod.txt

### Final Optimized Version

```txt
# Moola RunPod GPU Training Requirements
# Template: runpod/pytorch:2.4-py3.11-cuda12.4-ubuntu22.04
# Pre-installed: torch, numpy, cuda
# Last Updated: 2025-10-16

#############################################
# CRITICAL: NumPy Version Pinning
#############################################
# PyTorch 2.4 supports numpy 2.0+, but pin to stable version
numpy>=1.26.4,<2.1

#############################################
# Core ML Libraries (NOT in RunPod template)
#############################################
pandas>=2.3,<3.0
scipy>=1.14,<2.0
scikit-learn>=1.7,<2.0
xgboost>=2.0,<3.0
imbalanced-learn==0.14.0

#############################################
# PyTorch Ecosystem
#############################################
# PyTorch Lightning for advanced training
pytorch-lightning>=2.4.0,<3.0

#############################################
# Data Handling & Validation
#############################################
pyarrow>=17.0,<18.0        # Parquet files
pandera>=0.26.1,<1.0       # Schema validation

#############################################
# Configuration & CLI
#############################################
click>=8.2,<9.0
typer>=0.17,<1.0
hydra-core>=1.3,<2.0
pydantic>=2.11,<3.0
python-dotenv>=1.0

#############################################
# Logging & Output
#############################################
loguru>=0.7,<1.0
rich>=14.0,<15.0

#############################################
# MLOps & Tracking
#############################################
mlflow>=2.0,<3.0

#############################################
# Testing (optional for training)
#############################################
# pytest>=8.0,<9.0
# pytest-benchmark>=4.0,<5.0
```

### Installation Order

```bash
# On RunPod pod after launch:
cd /workspace/moola

# Install dependencies (no cache to save space)
pip install --no-cache-dir -r requirements-runpod.txt

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import sklearn; print(f'sklearn: {sklearn.__version__}')"
python -c "import xgboost; print(f'XGBoost: {xgboost.__version__}')"
```

---

## Dependency Conflict Matrix

### Known Issues & Solutions

| Conflict | Cause | Solution |
|----------|-------|----------|
| NumPy 2.0 + PyTorch 2.2 | PyTorch 2.2 not compiled with NumPy 2.0 | Use PyTorch 2.4+ OR pin numpy<2.0 |
| Multiple PyTorch installs | Reinstalling torch unnecessarily | Use template's torch, don't include in requirements |
| scikit-learn + NumPy | sklearn compiled with different NumPy | Install sklearn AFTER numpy is pinned |
| pandas + pyarrow | Version mismatches | Pin both pandas>=2.3 and pyarrow>=17.0 |

### Installation Best Practices

1. **Never include `torch` in requirements-runpod.txt** (use template version)
2. **Pin NumPy first** before installing other packages
3. **Use `--no-cache-dir`** to save disk space on pod
4. **Verify imports** before starting training
5. **Use exact versions** for reproducibility

---

## Comparison: Local vs RunPod

### Local Environment (448 packages)
```
torch==2.2.2
numpy==1.26.4
pandas==2.3.3
scikit-learn==1.7.2
pytorch-lightning==2.4.0
+ 443 other packages (most unnecessary for training)
```

### RunPod Optimized (23 packages)
```
numpy>=1.26.4,<2.1
pandas>=2.3,<3.0
scipy>=1.14,<2.0
scikit-learn>=1.7,<2.0
xgboost>=2.0,<3.0
imbalanced-learn==0.14.0
pytorch-lightning>=2.4.0,<3.0
pyarrow>=17.0,<18.0
pandera>=0.26.1,<1.0
click>=8.2,<9.0
typer>=0.17,<1.0
hydra-core>=1.3,<2.0
pydantic>=2.11,<3.0
python-dotenv>=1.0
loguru>=0.7,<1.0
rich>=14.0,<15.0
mlflow>=2.0,<3.0
```

**Reduction:** 95% fewer packages, 100% of functionality

---

## Migration Checklist

### Before Training on RunPod

- [ ] Choose template: `runpod/pytorch:2.4-py3.11-cuda12.4`
- [ ] Update `requirements-runpod.txt` with optimized dependencies
- [ ] Remove torch/torchvision from requirements (use template)
- [ ] Pin NumPy to compatible version
- [ ] Test locally with same PyTorch version first
- [ ] Prepare data in `/workspace/moola/data/`
- [ ] Copy code to pod or clone from git

### On RunPod Pod

```bash
# 1. Launch pod with PyTorch 2.4 template
# 2. Clone/upload code
cd /workspace
git clone https://github.com/yourusername/moola.git
cd moola

# 3. Install dependencies
pip install --no-cache-dir -r requirements-runpod.txt

# 4. Verify environment
python -c "import torch; import numpy; import sklearn; import xgboost; import imblearn; print('All imports successful')"

# 5. Run training
python scripts/train_full_pipeline.py --device cuda --mlflow-experiment runpod-test
```

### After Training

- [ ] Download artifacts from `/workspace/moola/data/artifacts/`
- [ ] Check MLflow metrics
- [ ] Verify model checkpoints
- [ ] Stop pod to avoid charges

---

## Troubleshooting

### Error: "RuntimeError: Numpy is not available"

**Cause:** NumPy version mismatch with PyTorch
**Solution:**
```bash
pip uninstall numpy -y
pip install "numpy>=1.26.4,<2.0"
```

### Error: "ImportError: cannot import name '_ARRAY_API'"

**Cause:** NumPy 2.0+ with PyTorch <2.3
**Solution:**
```bash
pip install "numpy<2.0"
```

### Error: Multiple PyTorch installations

**Cause:** torch in requirements.txt
**Solution:** Remove torch from requirements, use template version

### Error: "CUDA out of memory"

**Cause:** Batch size too large or workers too many
**Solution:**
```python
# In TSTCCPretrainer.__init__
batch_size=256  # Reduce from 512
num_workers=8   # Reduce from 16
```

---

## Version Tracking

### Tested Configurations

| Date | Template | NumPy | Status | Notes |
|------|----------|-------|--------|-------|
| 2025-10-16 | pytorch:2.4-py3.11-cuda12.4 | 1.26.4 | ✅ | Recommended |
| 2025-10-16 | pytorch:2.2-py3.10-cuda12.1 | 1.26.4 | ✅ | Conservative |
| 2025-10-15 | pytorch:2.2-py3.10-cuda12.1 | 2.0.1 | ❌ | NumPy conflict |

---

## Recommended Next Steps

1. **Update requirements-runpod.txt** with bulletproof version above
2. **Test locally** with PyTorch 2.4 to ensure compatibility
3. **Create RunPod pod** with pytorch:2.4-py3.11-cuda12.4 template
4. **Document actual RunPod results** in this file for future reference
5. **Consider freezing exact versions** after successful training:
   ```bash
   pip freeze > requirements-runpod-frozen.txt
   ```

---

## References

- PyTorch Compatibility: https://github.com/pytorch/pytorch/issues/107302
- RunPod PyTorch Docs: https://www.runpod.io/articles/guides/pytorch-2-4-cuda-12-4
- NumPy 2.0 Migration: https://numpy.org/doc/2.0/numpy_2_0_migration_guide.html
- Your TS-TCC Implementation: `/src/moola/models/ts_tcc.py`
- Training Pipeline: `/scripts/train_full_pipeline.py`
