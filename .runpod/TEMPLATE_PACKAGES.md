# RunPod Template Package Inventory

## Purpose

This document tracks what packages are pre-installed in RunPod templates to prevent 45-minute pip compilation disasters.

## Critical Lesson Learned

**NEVER add packages to requirements.txt that are already in the template!**

When you use `--system-site-packages` and then pip install a package with a version that doesn't match the template:
- Pip will download and compile from source (15-20 minutes per package)
- Result: 45+ minute setup time instead of 90 seconds

## Recommended Template

### Primary Template (Recommended)

```
runpod/pytorch:2.4-py3.11-cuda12.4-ubuntu22.04
```

**Pre-installed packages:**
```
torch==2.4.1+cu124
torchvision==0.19.1+cu124
torchaudio==2.4.1+cu124
numpy==1.26.4
triton==3.0.0
```

**NOTE:** This template does NOT include pandas, scipy, or scikit-learn by default.

**What this means:**
- ✅ You CAN use template's torch and numpy
- ❌ You MUST install pandas, scipy, scikit-learn (or use a different template)
- ⚠️ Installing pandas/scipy from pip will take 30-45 minutes

### Alternative Template (If Available)

```
runpod/pytorch:2.4-py3.11-cuda12.4-devel-ubuntu22.04
```

**Pre-installed packages (hypothetical - VERIFY BEFORE USING):**
```
torch==2.4.1
numpy==1.26.4
pandas==2.2.3
scipy==1.13.1
scikit-learn==1.5.2
```

**How to verify:**
```bash
# On RunPod pod, BEFORE setup:
python3 << 'EOF'
packages = ['torch', 'numpy', 'pandas', 'scipy', 'sklearn']
for pkg in packages:
    try:
        mod = __import__(pkg)
        print(f"{pkg}: {mod.__version__}")
    except:
        print(f"{pkg}: NOT FOUND")
EOF
```

## Template Selection Guide

### If Template Has: torch, numpy ONLY

**Use this strategy:**
```bash
# verify-template.sh will FAIL
# You need to install pandas, scipy, scikit-learn

# Option 1: Fast but risky (download pre-built wheels)
pip install pandas scipy scikit-learn  # ~5-10 minutes

# Option 2: Use Conda (faster for compiled packages)
conda install pandas scipy scikit-learn  # ~3-5 minutes
```

### If Template Has: torch, numpy, pandas, scipy, scikit-learn

**Use this strategy:**
```bash
# verify-template.sh will PASS
# Install ONLY moola-specific extras
pip install --no-cache-dir -r requirements-runpod-extras.txt  # ~60 seconds
```

## Version Compatibility Matrix

### NumPy Version Requirements

| Package | NumPy Version | Notes |
|---------|--------------|-------|
| torch 2.4 | 1.26.x or 2.0+ | Compatible with both |
| torch 2.2 | 1.26.x only | **FAILS with numpy 2.0** |
| pandas 2.2+ | 1.26.x or 2.0+ | Compatible with both |
| scipy 1.13+ | 1.26.x or 2.0+ | Compatible with both |
| scikit-learn 1.5+ | 1.26.x or 2.0+ | Compatible with both |

**Recommendation:** Pin numpy to 1.26.4 for maximum compatibility.

### PyTorch + CUDA Compatibility

| Template | CUDA | PyTorch | NumPy |
|----------|------|---------|-------|
| pytorch:2.4 | 12.4 | 2.4.1 | 1.26.4 |
| pytorch:2.2 | 12.1 | 2.2.x | 1.26.x |
| pytorch:2.1 | 11.8 | 2.1.x | 1.24.x |

## Template Verification Process

### Before EVERY Deployment

1. **Start RunPod pod**
2. **Run verification script FIRST:**
   ```bash
   bash /workspace/scripts/verify-template.sh
   ```
3. **If verification fails:**
   - STOP and terminate pod
   - Select different template
   - Repeat verification
4. **Only proceed if verification passes**

### Verification Script Output

**PASS (proceed with setup):**
```
✅ PyTorch        : 2.4.1
✅ NumPy          : 1.26.4
✅ Pandas         : 2.2.3
✅ SciPy          : 1.13.1
✅ scikit-learn   : 1.5.2

✅ SUCCESS: Template is correct!
Expected setup time: 60-90 seconds
```

**FAIL (terminate pod immediately):**
```
✅ PyTorch        : 2.4.1
✅ NumPy          : 1.26.4
❌ Pandas         : NOT FOUND
❌ SciPy          : NOT FOUND
❌ scikit-learn   : NOT FOUND

❌ CRITICAL ERROR: Wrong RunPod Template!
⚠️  WITHOUT THESE PACKAGES:
   - Setup will take 45-60 minutes (pip compilation)
   - You will waste GPU time and money

🛑 TERMINATE THIS POD IMMEDIATELY
```

## Package Installation Time Reference

### With Correct Template (pandas, scipy in template)

| Operation | Duration | Size |
|-----------|----------|------|
| Create venv | 5 sec | 10 MB |
| Install extras | 60 sec | 40 MB |
| Install moola | 5 sec | 5 MB |
| **Total** | **70 sec** | **~55 MB** |

### With Wrong Template (pandas, scipy NOT in template)

| Operation | Duration | Size | Notes |
|-----------|----------|------|-------|
| Create venv | 5 sec | 10 MB | Same |
| **Download pandas** | **5 min** | **50 MB** | Source tarball |
| **Compile pandas** | **15 min** | **+100 MB** | C extensions |
| **Download scipy** | **10 min** | **80 MB** | Large source |
| **Compile scipy** | **20 min** | **+150 MB** | Fortran/C |
| Download scikit-learn | 3 min | 30 MB | - |
| Compile scikit-learn | 10 min | +80 MB | - |
| Install extras | 60 sec | 40 MB | - |
| **Total** | **~64 min** | **~540 MB** | ❌ DISASTER |

## Requirements File Strategy

### DO NOT USE (Old Approach)

```plaintext
# requirements-runpod.txt
numpy>=1.26,<2.0        # ❌ In template, will shadow/reinstall
pandas>=2.3,<3.0        # ❌ Template has 2.2.3, will download 2.3+
scipy>=1.14,<2.0        # ❌ Template has 1.13.x, will download 1.14+
scikit-learn>=1.7,<2.0  # ❌ Template has 1.5.x, will download 1.7+
```

**Result:** 45+ minute compilation disaster.

### DO USE (New Approach)

**File 1: requirements-runpod-template.txt** (verification only)
```plaintext
# Expected in template (DO NOT INSTALL)
torch>=2.4
numpy>=1.26,<2.0
pandas>=2.2
scipy>=1.13
scikit-learn>=1.5
```

**File 2: requirements-runpod-extras.txt** (actual installation)
```plaintext
# Only packages NOT in template
imbalanced-learn==0.14.0
xgboost>=2.0
pytorch-lightning>=2.4.0
loguru>=0.7
click>=8.2
# ... other moola-specific packages
```

## Troubleshooting

### Setup Takes > 5 Minutes

**STOP IMMEDIATELY (Ctrl+C)**

Check what's happening:
```bash
# What is pip doing?
ps aux | grep pip

# Is it building from source?
ls -la /tmp/pip-*
```

If you see:
- "Building wheel for pandas"
- "Compiling Cython extensions"
- "Running setup.py"

**You have the wrong template or wrong requirements!**

### Venv Size > 500MB

```bash
du -sh /tmp/moola-venv
```

If venv is large, packages were installed instead of using template versions.

**Fix:**
1. Delete venv: `rm -rf /tmp/moola-venv`
2. Run verify-template.sh
3. Check requirements-runpod-extras.txt doesn't include template packages
4. Re-run setup

### "ImportError: cannot import name 'packaging'"

```bash
# Install packaging separately
pip install packaging
```

This is a build dependency, safe to install (small, fast).

## Template Update Policy

### When to Update This Document

1. **After successful pod deployment** - Document what worked
2. **When template versions change** - Update compatibility matrix
3. **When encountering new issues** - Add to troubleshooting

### How to Document New Template

```bash
# On fresh RunPod pod:
python3 << 'EOF'
import pkg_resources
packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

core = ['torch', 'numpy', 'pandas', 'scipy', 'scikit-learn',
        'matplotlib', 'seaborn', 'jupyter', 'jupyterlab']

print("Template inventory:")
for pkg in core:
    version = packages.get(pkg, 'NOT FOUND')
    print(f"{pkg}: {version}")
EOF

# Save output to this document
```

## Quick Reference

### Template Verification Command (One-Liner)

```bash
python3 -c "import sys; pkgs=['torch','numpy','pandas','scipy','sklearn']; missing=[p for p in pkgs if __import__(p) is None]; sys.exit(0 if not missing else 1)" && echo "✅ Template OK" || echo "❌ Wrong template"
```

### Installation Time Expectations

| Time Range | Status | Action |
|------------|--------|--------|
| < 2 min | ✅ Perfect | Continue with training |
| 2-5 min | ⚠️ Slow | Check what's installing |
| > 5 min | ❌ Wrong | Stop, fix template/requirements |

---

**Last Updated:** 2025-10-16
**Template Tested:** runpod/pytorch:2.4-py3.11-cuda12.4-ubuntu22.04
**Verification Required:** YES (before every deployment)
