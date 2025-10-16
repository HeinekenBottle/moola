# Quick Fix Checklist - Stop the 45-Minute Disaster

**Status:** 🔴 URGENT - Current deployment is BROKEN
**Time to fix:** ~30 minutes
**Testing time:** ~40 minutes

---

## The Problem in 30 Seconds

Your `requirements-runpod.txt` includes packages that are already in the RunPod template, but with version specs that don't match. Result: pip downloads and compiles pandas/scipy from source for 45+ minutes instead of using the template versions.

---

## Quick Fix Steps

### Step 1: Update optimized-setup.sh (5 minutes)

**File:** `.runpod/scripts/optimized-setup.sh`

**Add BEFORE line 49 (before "Step 3/6: Creating virtual environment"):**

```bash
# 3. Verify template packages (NEW)
echo "🔍 Step 3/6: Verifying template packages..."
python3 << 'VERIFY_EOF'
import sys

required = {
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'scipy': 'SciPy',
    'sklearn': 'scikit-learn'
}

missing = []
for module, name in required.items():
    try:
        mod = __import__(module)
        print(f'✅ {name}: {mod.__version__}')
    except ImportError:
        missing.append(name)
        print(f'❌ {name}: NOT FOUND')

if missing:
    print(f'\n❌ CRITICAL: Missing {missing}')
    print('⚠️  Setup will take 45+ minutes (pip compilation)')
    print('🛑 TERMINATE POD and select correct template')
    sys.exit(1)

print('✅ All template packages found')
VERIFY_EOF

echo ""
```

**Replace lines 57-73 (pip install section) with:**

```bash
    echo "   Installing moola-specific packages only..."
    # Install ONLY packages NOT in template
    pip install --no-cache-dir \
        imbalanced-learn==0.14.0 \
        xgboost \
        pytorch-lightning \
        pyarrow \
        pandera \
        click \
        typer \
        rich \
        loguru \
        hydra-core \
        pydantic \
        pydantic-settings \
        pyyaml \
        python-dotenv \
        mlflow \
        joblib \
        packaging \
        hatchling
```

**Add timing check at end (after line 170):**

```bash
echo "⏱️  Setup completed in $((SECONDS / 60))m $((SECONDS % 60))s"
echo ""
if [ $SECONDS -gt 180 ]; then
    echo "⚠️  WARNING: Setup took longer than expected (>3 min)"
    echo "    This suggests packages were compiled from source"
    echo "    Check if template packages were detected correctly"
fi
```

**Add at start of script (after set -e):**

```bash
SECONDS=0  # Track execution time
```

### Step 2: Update deploy-fast.sh (5 minutes)

**File:** `.runpod/deploy-fast.sh`

**Replace the embedded script (lines 84-188) with updated version that includes template verification.**

Alternatively, modify the embedded script to call verify-template.sh:

**Add after line 103 (before venv creation):**

```bash
# Verify template packages (CRITICAL)
echo "🔍 Verifying template packages..."
python3 << 'VERIFY_EOF'
import sys
required = ['torch', 'numpy', 'pandas', 'scipy', 'sklearn']
missing = []
for pkg in required:
    try:
        mod = __import__(pkg)
        print(f'✅ {pkg}: {mod.__version__}')
    except ImportError:
        missing.append(pkg)
        print(f'❌ {pkg}: NOT FOUND')

if missing:
    print(f'\n❌ WRONG TEMPLATE! Missing: {missing}')
    print('⚠️  Installation will take 45+ minutes')
    print('🛑 Select template with full scientific stack')
    sys.exit(1)
VERIFY_EOF
```

**Replace lines 122-123 with:**

```bash
    # Install ONLY moola-specific packages
    echo "📦 Installing moola extras (~60 seconds)..."
    pip install --no-cache-dir \
        imbalanced-learn==0.14.0 \
        xgboost pytorch-lightning \
        pyarrow pandera \
        click typer rich loguru \
        hydra-core pydantic pydantic-settings \
        pyyaml python-dotenv \
        mlflow joblib \
        packaging hatchling
```

### Step 3: Archive Old requirements-runpod.txt (1 minute)

```bash
cd /Users/jack/projects/moola

# Rename old file
mv requirements-runpod.txt requirements-runpod.OLD.txt

# Add warning to old file
cat > requirements-runpod.OLD.txt.WARNING << 'EOF'
⚠️  THIS FILE IS DEPRECATED AND CAUSES 45-MINUTE SETUP TIMES ⚠️

DO NOT USE THIS FILE!

Use requirements-runpod-extras.txt instead.

This file includes pandas, scipy, scikit-learn which are already
in the RunPod template. When pip sees version mismatches, it
downloads and compiles from source (45+ minutes).

See: .runpod/CRITICAL_INFRASTRUCTURE_AUDIT.md for details.
EOF

# Use new file
ln -sf requirements-runpod-extras.txt requirements-runpod.txt
```

### Step 4: Update Documentation (5 minutes)

**File:** `docs/RUNPOD_QUICKSTART.md`

**Add BEFORE "Setup Commands" section:**

```markdown
## ⚠️  CRITICAL: Template Verification

**ALWAYS run template verification BEFORE setup!**

This prevents 45-minute pip compilation disasters.

```bash
# SSH into pod
cd /workspace

# Download verification script
git clone https://github.com/yourusername/moola.git
cd moola

# Run verification
bash .runpod/verify-template.sh
```

**Expected output:**
```
✅ PyTorch        : 2.4.1
✅ NumPy          : 1.26.4
✅ Pandas         : 2.2.3
✅ SciPy          : 1.13.1
✅ scikit-learn   : 1.5.2

✅ SUCCESS: Template is correct!
```

**If any package shows "NOT FOUND":**
- STOP immediately
- Terminate the pod
- Select a different template with full scientific stack
- Re-run verification

Only proceed with setup if ALL packages are found.
```

**Replace line 40-41 with:**

```markdown
# Install dependencies (moola-specific packages only)
pip install --no-cache-dir -r requirements-runpod-extras.txt
```

### Step 5: Deploy Updated Scripts (2 minutes)

```bash
cd /Users/jack/projects/moola/.runpod

# Verify syntax
bash -n scripts/optimized-setup.sh
bash -n verify-template.sh
bash -n deploy-fast.sh

# Deploy to network storage
bash deploy-fast.sh deploy
```

---

## Testing Steps (40 minutes)

### Test 1: Fresh Pod Deployment (30 minutes)

1. **Start RunPod pod:**
   - Template: `runpod/pytorch:2.4-py3.11-cuda12.4-ubuntu22.04`
   - GPU: RTX 4090
   - Volume: 22uv11rdjk
   - Disk: 50GB

2. **SSH into pod:**
   ```bash
   ssh root@<pod-ip> -p <port>
   ```

3. **Run verification FIRST:**
   ```bash
   cd /workspace
   bash scripts/verify-template.sh
   ```

   **Expected:** All packages found, exits with 0

4. **Run setup with timing:**
   ```bash
   time bash scripts/optimized-setup.sh
   ```

   **Watch for:**
   - ✅ No "Building wheel" messages
   - ✅ No "Compiling" messages
   - ✅ Completes in <2 minutes
   - ❌ If takes >5 minutes, STOP and investigate

5. **Check venv size:**
   ```bash
   du -sh /tmp/moola-venv
   ```

   **Expected:** <200MB (target: ~50-100MB)

6. **Verify imports:**
   ```bash
   python3 -c "
   import torch, numpy, pandas, scipy, sklearn
   import xgboost, imblearn, loguru
   print('✅ All imports successful')
   "
   ```

7. **Run training:**
   ```bash
   bash /workspace/scripts/fast-train.sh
   ```

### Test 2: Verify No Template Package Shadowing (5 minutes)

```bash
# Check which packages are in venv vs system
python3 << 'EOF'
import torch, numpy, pandas
import sys

print("Package locations:")
print(f"torch: {torch.__file__}")
print(f"numpy: {numpy.__file__}")
print(f"pandas: {pandas.__file__}")

# Should show /usr/local/lib/python3.11/dist-packages (system)
# NOT /tmp/moola-venv/lib/python3.11/site-packages (venv)

venv_count = sum(1 for pkg in [torch, numpy, pandas]
                 if 'moola-venv' in pkg.__file__)

if venv_count > 0:
    print(f"\n⚠️  WARNING: {venv_count} core packages in venv (should be 0)")
    print("    Packages were reinstalled instead of using template")
else:
    print(f"\n✅ All core packages using template versions")
EOF
```

### Test 3: Document Results (5 minutes)

Update `TEMPLATE_PACKAGES.md` with:
- Actual template used
- Verification output
- Setup time
- Venv size
- Any issues

---

## Success Criteria

- [ ] verify-template.sh passes in <10 seconds
- [ ] optimized-setup.sh completes in <2 minutes
- [ ] No "Building wheel" or "Compiling" messages
- [ ] Venv size <200MB
- [ ] All imports work
- [ ] Training pipeline runs successfully
- [ ] Core packages (torch, numpy, pandas) from system, not venv

---

## Rollback Plan

If fixed version fails:

```bash
# Local machine
cd /Users/jack/projects/moola

# Restore old files
mv requirements-runpod.OLD.txt requirements-runpod.txt
git checkout .runpod/scripts/optimized-setup.sh
git checkout .runpod/deploy-fast.sh

# But DON'T use them - they're broken!
# Instead, investigate why fix failed
```

---

## Common Issues During Fix

### Issue: "verify-template.sh fails"

**Cause:** Wrong template, missing pandas/scipy/scikit-learn

**Fix:**
1. Terminate pod
2. Select template: `runpod/pytorch:2.4-py3.11-cuda12.4-devel-ubuntu22.04`
3. Or install missing packages (but takes 30-45 min)

### Issue: "Setup still takes >5 minutes"

**Cause:** Requirements file still has template packages

**Debug:**
```bash
# What is pip installing?
pip list | grep pandas
pip list | grep scipy

# Check if template packages in requirements
grep -E 'pandas|scipy|sklearn' requirements-runpod-extras.txt
```

### Issue: "Import errors after setup"

**Cause:** Packages not installed or wrong Python path

**Fix:**
```bash
# Verify venv activated
which python3
# Should show: /tmp/moola-venv/bin/python3

# Reinstall package
pip install --no-cache-dir <package-name>
```

---

## Files Modified Summary

```
Modified:
  .runpod/scripts/optimized-setup.sh   # Add verification, remove template packages
  .runpod/deploy-fast.sh               # Add verification, update pip install
  docs/RUNPOD_QUICKSTART.md            # Add verification step

Created:
  .runpod/verify-template.sh           # Template verification script
  .runpod/CRITICAL_INFRASTRUCTURE_AUDIT.md  # Full audit report
  .runpod/TEMPLATE_PACKAGES.md         # Template documentation
  .runpod/AUDIT_SUMMARY.md             # Quick summary
  .runpod/QUICK_FIX_CHECKLIST.md       # This file
  requirements-runpod-extras.txt       # Only extras, not template packages

Archived:
  requirements-runpod.txt → requirements-runpod.OLD.txt  # Old broken file
```

---

## After Successful Test

1. **Commit changes:**
   ```bash
   git add .
   git commit -m "fix: prevent 45-min pip compilation disaster with template verification"
   git push
   ```

2. **Update previous audit:**
   Add note to `DEPLOYMENT_AUDIT_REPORT.md`:
   ```markdown
   ## CRITICAL UPDATE (2025-10-16)

   This audit was INCOMPLETE. See CRITICAL_INFRASTRUCTURE_AUDIT.md for
   actual issues found and fixes applied.

   The deployment was NOT production ready as claimed.
   ```

3. **Document in project README:**
   Add warning about template verification requirement

---

## Estimated Time Investment

- **Implementing fixes:** 30 minutes
- **Testing on pod:** 40 minutes
- **Documentation:** 10 minutes
- **Total:** ~80 minutes

**Return on investment:**
- **Time saved per deployment:** 43 minutes
- **Cost saved per deployment:** $0.37
- **Break-even:** 2 deployments
- **Savings after 10 deployments:** 7+ hours, $3.70

---

## Final Checklist

Before marking as complete:

- [ ] optimized-setup.sh updated with verification
- [ ] deploy-fast.sh updated with verification
- [ ] requirements-runpod-extras.txt created
- [ ] verify-template.sh created and executable
- [ ] Documentation updated
- [ ] Deployed to network storage
- [ ] Tested on fresh pod
- [ ] Setup time <2 minutes
- [ ] No compilation messages
- [ ] Training works
- [ ] Results documented

**Only mark as "PRODUCTION READY" after ALL items checked.**

---

**Created:** 2025-10-16
**Urgency:** 🔴 HIGH - Current system is broken
**Next Action:** Implement Step 1 (Update optimized-setup.sh)
