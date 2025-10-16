# RunPod Orchestrator Deployment Checklist

## Pre-Deployment Verification

### 1. Files Created ✅

Core orchestrator:
- [x] `src/moola/runpod/__init__.py`
- [x] `src/moola/runpod/scp_orchestrator.py` (684 lines)
- [x] `src/moola/runpod/README.md`

Monitoring and validation:
- [x] `src/moola/validation/__init__.py` (updated)
- [x] `src/moola/validation/training_monitor.py` (381 lines)

Deployment scripts:
- [x] `src/moola/scripts/test_orchestrator.py` (287 lines)
- [x] `src/moola/scripts/deploy_fixes.py` (235 lines)
- [x] `src/moola/scripts/runpod_retrain_pipeline.py` (412 lines)

Documentation:
- [x] `docs/runpod_orchestrator_runbook.md` (800+ lines)
- [x] `examples/runpod_quickstart.py` (320+ lines)
- [x] `RUNPOD_ORCHESTRATOR_SUMMARY.md`
- [x] `DEPLOYMENT_CHECKLIST.md` (this file)

**Total Production Code**: ~2,000 lines
**Total Documentation**: ~2,000 lines

### 2. Integration Points ✅

Model integration:
- [x] `CnnTransformerModel.freeze_encoder()` method exists
- [x] `CnnTransformerModel.unfreeze_encoder_gradual()` method exists
- [x] `CnnTransformerModel.load_pretrained_encoder()` wired to fit()

Config integration:
- [x] `CNNTR_FREEZE_EPOCHS` defined in training_config.py
- [x] `CNNTR_GRADUAL_UNFREEZE` flag added
- [x] `CNNTR_UNFREEZE_SCHEDULE` defined

Validation integration:
- [x] `training_monitor.py` integrated into validation module
- [x] `detect_class_collapse()` function available

### 3. Dependencies ✅

Required packages (should be installed):
- [x] `paramiko` or `ssh` (for SSH/SCP operations)
- [x] `numpy` (for result validation)
- [x] `torch` (already present)
- [x] Standard library: `subprocess`, `pathlib`, `dataclasses`

No new dependencies required - uses built-in SSH tools.

## Deployment Steps

### Step 1: Verify Local Environment

```bash
# Check Python version
python --version  # Should be 3.9+

# Check SSH key exists
ls -la ~/.ssh/id_ed25519

# Check project structure
ls src/moola/runpod/
ls src/moola/scripts/
ls src/moola/validation/
```

**Expected Output**: All files present

### Step 2: Test Orchestrator Connection

```bash
# Run test suite
python src/moola/scripts/test_orchestrator.py \
    --host 213.173.98.6 \
    --port 14385 \
    --key ~/.ssh/id_ed25519
```

**Expected Output**:
```
✓ PASS: Connection
✓ PASS: File Upload/Download
✓ PASS: Directory Operations
✓ PASS: Environment Verification
✓ PASS: Python Imports

✅ ALL TESTS PASSED - Orchestrator ready for use
```

**If tests fail**:
- Check SSH key: `ssh-add ~/.ssh/id_ed25519`
- Verify pod is running on RunPod dashboard
- Check network connectivity: `ping 213.173.98.6`

### Step 3: Verify RunPod Environment

```python
from moola.runpod import RunPodOrchestrator

orch = RunPodOrchestrator(
    host="213.173.98.6",
    port=14385,
    key_path="~/.ssh/id_ed25519"
)

# Run pre-flight checks
env_status = orch.verify_environment()
```

**Expected Output**:
```
[VERIFY] Running pre-flight checks...
  ✓ SSH Connection
  ✓ PyTorch
  ✓ CUDA
  ✓ GPU Info
  ✓ Workspace
  ✓ Data Files
  ✓ Source Code
  ✓ Artifacts Dir
  ✓ Pre-trained Encoder (or ⚠ if not found - that's OK)

[VERIFY] ✓ All checks passed - environment ready
```

**If checks fail**:
- SSH Connection: Check pod status
- Data Files: Run data preprocessing first
- Pre-trained Encoder: Run SSL pre-training or train from scratch

### Step 4: Deploy Encoder Fixes

```bash
# Deploy encoder freezing fixes
python src/moola/scripts/deploy_fixes.py \
    --preset encoder_fixes \
    --test-imports
```

**Expected Output**:
```
[DEPLOY] Deploying 2 fixed files...
  ✓ Deployed: cnn_transformer.py
  ✓ Deployed: training_config.py
[DEPLOY] Complete: 2/2 files deployed

[VERIFY] Checking deployed files...
  ✓ cnn_transformer.py
  ✓ training_config.py

[TEST] Verifying imports...
  ✓ CnnTransformerModel
  ✓ training_config

[SUCCESS] Deployment complete
```

**If deployment fails**:
- Check file paths are correct
- Verify workspace exists: `/workspace/moola`
- Check permissions on remote

### Step 5: Test Training (Short Run)

```python
from moola.runpod import RunPodOrchestrator

orch = RunPodOrchestrator(
    host="213.173.98.6",
    port=14385,
    key_path="~/.ssh/id_ed25519"
)

# Short test run (1 epoch) to verify everything works
exit_code = orch.run_training(
    model="cnn_transformer",
    device="cuda",
    encoder_path="/workspace/artifacts/pretrained/encoder_weights.pt",
    extra_args="--n-epochs 1",
    timeout=600
)

print(f"Exit code: {exit_code}")
```

**Expected Output**:
```
[TRAINING] Starting cnn_transformer on cuda...
[SSL] Loading pre-trained encoder from /workspace/artifacts/pretrained/encoder_weights.pt
[FREEZE] Encoder frozen: 1243 frozen params, 128 trainable params
Epoch [1/1] Train Loss: 0.6234 Acc: 0.6500 | Val Loss: 0.5823 Acc: 0.7000
[TRAINING] ✓ Training complete
```

**If training fails**:
- Download logs: `orch.download_logs("/tmp/debug/")`
- Check error messages in logs
- Verify GPU is available: `orch.execute_command("nvidia-smi")`

### Step 6: Full Training Run

```bash
# Run complete retraining pipeline
python src/moola/scripts/runpod_retrain_pipeline.py \
    --models cnn_transformer \
    --host 213.173.98.6 \
    --port 14385
```

**Expected Duration**: 30-60 minutes for CNN-Transformer

**Expected Output**:
```
============================================================
TRAINING: CNN-TRANSFORMER (Pre-trained + FIXES)
============================================================
[SSL] Loading pre-trained encoder...
[FREEZE] Encoder frozen...
[UNFREEZE] Stage 1: Last transformer layer unfrozen
[UNFREEZE] Stage 2: All transformer layers unfrozen
[UNFREEZE] Stage 3: Full model unfrozen

✓ cnn_transformer: 82.1% accuracy
✓ Class collapse FIXED!

[SUCCESS] All models trained successfully
```

### Step 7: Validate Results

```python
import numpy as np
from pathlib import Path

# Load predictions
preds = np.load("/tmp/results/seed_1337.npy")

# Check for class collapse
unique_classes = len(np.unique(preds))
print(f"Unique classes predicted: {unique_classes}")

# Per-class distribution
for cls in np.unique(preds):
    count = (preds == cls).sum()
    pct = count / len(preds) * 100
    print(f"  Class {cls}: {count} samples ({pct:.1f}%)")
```

**Expected Output**:
```
Unique classes predicted: 2 (or more)
  Class 0: 45 samples (46.4%)
  Class 1: 52 samples (53.6%)
```

**If class collapse detected** (only 1 unique class):
- Check logs for encoder freezing messages
- Verify gradual unfreezing triggered
- Inspect model parameters: `orch.execute_command("...")`

## Post-Deployment Verification

### Functionality Checklist

- [ ] SSH connection works
- [ ] File upload/download works
- [ ] Command execution works
- [ ] Environment verification passes
- [ ] Encoder fixes deployed successfully
- [ ] Training runs without errors
- [ ] Results can be downloaded
- [ ] No class collapse in predictions
- [ ] Real-time monitoring detects errors
- [ ] Logs can be downloaded for debugging

### Performance Checklist

- [ ] File upload speed: ~1 MB/s
- [ ] Command latency: ~100ms
- [ ] Training output streams in real-time
- [ ] No buffering delays
- [ ] GPU utilization > 90% during training
- [ ] No memory leaks or crashes

### Documentation Checklist

- [ ] README.md explains the system
- [ ] Runbook covers all workflows
- [ ] Quick start example runs successfully
- [ ] API documentation is complete
- [ ] Error messages are clear
- [ ] Troubleshooting guide is helpful

## Known Issues and Workarounds

### Issue 1: SSH Connection Timeout

**Symptoms**: Connection refused or timeout errors

**Workaround**:
```bash
# Check pod is running
# Visit RunPod dashboard

# Test connection manually
ssh root@213.173.98.6 -p 14385 -i ~/.ssh/id_ed25519

# Add key to agent
ssh-add ~/.ssh/id_ed25519
```

### Issue 2: Import Errors on RunPod

**Symptoms**: `ModuleNotFoundError: No module named 'moola'`

**Workaround**:
```bash
# SSH into pod and reinstall
ssh root@213.173.98.6 -p 14385 -i ~/.ssh/id_ed25519
cd /workspace/moola
pip install -e .
```

### Issue 3: Pre-trained Encoder Not Found

**Symptoms**: `FileNotFoundError: encoder_weights.pt`

**Workaround**:
```bash
# Run SSL pre-training first
python -m moola.cli ssl-pretrain --device cuda

# Or train from scratch (no encoder path)
orch.run_training("cnn_transformer", device="cuda")  # No encoder_path
```

### Issue 4: Class Collapse Persists

**Symptoms**: Model predicts only one class

**Investigation**:
```python
# Check encoder frozen status
orch.execute_command("""
    cd /workspace/moola &&
    python -c 'from moola.models import CnnTransformerModel;
    m = CnnTransformerModel();
    m._build_model(4, 2);
    m.freeze_encoder();
    print(f\"Frozen: {sum(1 for p in m.model.parameters() if not p.requires_grad)}\");
    print(f\"Trainable: {sum(1 for p in m.model.parameters() if p.requires_grad)}\");'
""")

# Check logs for unfreezing messages
orch.download_logs("/tmp/debug/")
grep "UNFREEZE" /tmp/debug/*.log
```

## Success Criteria

✅ **All tests pass** in `test_orchestrator.py`
✅ **Environment verification** shows all checks passed
✅ **Training completes** without errors
✅ **Results download** successfully
✅ **No class collapse** (>= 2 unique classes predicted)
✅ **Monitoring detects errors** (if any occur)
✅ **Logs are accessible** for debugging

## Rollback Plan

If deployment fails:

1. **Revert code changes**:
   ```bash
   git checkout HEAD~1 src/moola/models/cnn_transformer.py
   git checkout HEAD~1 src/moola/config/training_config.py
   ```

2. **Redeploy previous version**:
   ```bash
   python src/moola/scripts/deploy_fixes.py \
       --files src/moola/models/cnn_transformer.py \
               src/moola/config/training_config.py
   ```

3. **Verify rollback**:
   ```bash
   python src/moola/scripts/test_orchestrator.py
   ```

## Next Steps After Successful Deployment

1. **Train all models**:
   ```bash
   python src/moola/scripts/runpod_retrain_pipeline.py \
       --models logreg,rf,xgb,simple_lstm,cnn_transformer
   ```

2. **Train stack ensemble**:
   ```bash
   python -m moola.cli stack-train --stacker rf
   ```

3. **Generate submission**:
   ```bash
   python -m moola.cli predict --model stack_rf --output submission.csv
   ```

4. **Submit to Kaggle**:
   ```bash
   kaggle competitions submit -f submission.csv
   ```

## Support

If you encounter issues:

1. Check `docs/runpod_orchestrator_runbook.md` for detailed troubleshooting
2. Run `test_orchestrator.py` to diagnose connection issues
3. Download logs: `orch.download_logs("/tmp/debug/")`
4. Check RunPod dashboard for pod status
5. Verify SSH key: `ssh-add -l`

## Maintenance

### Weekly Tasks

- [ ] Clean up old artifacts: `orch.cleanup_artifacts(keep_latest=3)`
- [ ] Download training logs for analysis
- [ ] Monitor disk usage on pod
- [ ] Update documentation if workflows change

### Monthly Tasks

- [ ] Review error detection patterns
- [ ] Update fix presets based on common issues
- [ ] Optimize deployment scripts
- [ ] Update documentation with new best practices

---

**Deployment Status**: ✅ Ready
**Last Updated**: 2025-10-16
**Version**: 1.0.0
