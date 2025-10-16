# RunPod SCP Orchestrator Runbook

**AI-Friendly Training Orchestration for Iterative Debugging**

## Overview

The RunPod SCP Orchestrator enables precise, file-by-file interaction with RunPod GPU pods for ML training. Unlike black-box shell scripts, this system allows AI models to:

- Inspect files before/after deployment
- Upload specific fixes incrementally
- Monitor training in real-time
- Download results immediately for validation
- Debug issues interactively

## Why SCP Over Shell Scripts?

| Shell Scripts | SCP Orchestrator |
|--------------|------------------|
| Black box execution | Transparent file operations |
| All-or-nothing deployment | Incremental fixes |
| Post-mortem debugging only | Real-time monitoring |
| Hard to iterate | Fast debug loop |

## Architecture

```
Local Machine (AI Agent)
    ↓ SCP upload
RunPod Pod (GPU)
    ↓ SSH execute
Training Process
    ↓ SCP download
Local Machine (Validation)
```

## Quick Start

### 1. Test Orchestrator

```bash
python src/moola/scripts/test_orchestrator.py \
    --host 213.173.98.6 \
    --port 14385 \
    --key ~/.ssh/id_ed25519
```

This runs 5 tests:
- SSH connectivity
- File upload/download
- Directory operations
- Environment verification
- Python imports

All tests must pass before proceeding.

### 2. Verify Environment

```python
from moola.runpod import RunPodOrchestrator

orch = RunPodOrchestrator(
    host="213.173.98.6",
    port=14385,
    key_path="~/.ssh/id_ed25519",
    workspace="/workspace/moola"
)

# Run pre-flight checks
env_status = orch.verify_environment()

if all(env_status.values()):
    print("✓ Environment ready for training")
else:
    print("✗ Fix environment issues first")
```

### 3. Deploy Fixes

```bash
# Deploy specific files
python src/moola/scripts/deploy_fixes.py \
    --files src/moola/models/cnn_transformer.py \
           src/moola/config/training_config.py

# Or use a preset
python src/moola/scripts/deploy_fixes.py --preset encoder_fixes
```

Available presets:
- `encoder_fixes`: CNN-Transformer encoder freezing fixes
- `augmentation_fixes`: Data augmentation modules
- `loss_fixes`: Loss function updates
- `oof_pipeline`: OOF pipeline changes
- `all_models`: All model files
- `all_pipelines`: All pipeline files

### 4. Run Training

```python
# Option A: Single model
exit_code = orch.run_training(
    model="cnn_transformer",
    device="cuda",
    encoder_path="/workspace/artifacts/pretrained/encoder_weights.pt",
    timeout=7200  # 2 hours
)

# Option B: With monitoring
from moola.validation import monitor_training_with_error_detection

exit_code, errors, metrics = monitor_training_with_error_detection(
    orch, "cnn_transformer", device="cuda"
)

if errors:
    print("Detected issues:", [e.error_type for e in errors])
```

### 5. Download and Validate Results

```python
# Download OOF predictions
orch.download_results("cnn_transformer", "/tmp/cnn_results/")

# Validate for class collapse
import numpy as np

preds = np.load("/tmp/cnn_results/seed_1337.npy")
unique_classes = len(np.unique(preds))

if unique_classes < 2:
    print("⚠️ Class collapse detected")
else:
    print(f"✓ {unique_classes} classes predicted")
```

## Common Workflows

### Workflow 1: Fix Encoder Freezing Issue

**Problem**: CNN-Transformer not learning from pre-trained encoder (class collapse)

**Solution**:

```python
# 1. Deploy encoder freezing fix
orch.deploy_fixes([
    "src/moola/models/cnn_transformer.py",  # Added freeze_encoder()
    "src/moola/config/training_config.py",  # SSL hyperparams
])

# 2. Verify fix deployed
orch.execute_command(
    "cd /workspace/moola && "
    "grep -n 'freeze_encoder' src/moola/models/cnn_transformer.py | head -5"
)

# 3. Test import
orch.execute_command(
    "cd /workspace/moola && "
    "source /tmp/moola-venv/bin/activate && "
    "python -c 'from moola.models import CnnTransformerModel; "
    "m = CnnTransformerModel(); print(hasattr(m, \"freeze_encoder\"))'"
)

# 4. Retrain with fixes
exit_code = orch.run_training(
    "cnn_transformer",
    device="cuda",
    encoder_path="/workspace/artifacts/pretrained/encoder_weights.pt"
)

# 5. Download and check results
orch.download_results("cnn_transformer", "/tmp/results/")

# 6. Validate fix worked
preds = np.load("/tmp/results/seed_1337.npy")
print(f"Unique classes: {len(np.unique(preds))}")  # Should be >= 2
```

### Workflow 2: Update Hyperparameters

**Problem**: Need to adjust learning rate, batch size, or epochs

**Solution**:

```python
# 1. Edit training_config.py locally
# (AI edits the file with new hyperparameters)

# 2. Deploy just the config file
orch.upload_file(
    "src/moola/config/training_config.py",
    "/workspace/moola/src/moola/config/training_config.py"
)

# 3. Verify config updated
orch.execute_command(
    "cd /workspace/moola && "
    "python -c 'from moola.config.training_config import CNNTR_LEARNING_RATE; "
    "print(f\"LR: {CNNTR_LEARNING_RATE}\")'"
)

# 4. Retrain with new hyperparameters
orch.run_training("cnn_transformer", device="cuda")
```

### Workflow 3: Debug Training Failure

**Problem**: Training fails with cryptic error

**Solution**:

```python
# 1. Download logs immediately
orch.download_logs("/tmp/debug_logs/")

# 2. Read logs locally (AI can analyze)
with open("/tmp/debug_logs/training.log") as f:
    logs = f.read()
    print(logs)

# 3. Inspect error pattern
if "CUDA out of memory" in logs:
    print("Issue: Batch size too large")
    print("Fix: Reduce CNNTR_BATCH_SIZE in training_config.py")
elif "shape mismatch" in logs:
    print("Issue: Data preprocessing error")
    print("Fix: Check windowing logic")

# 4. Deploy fix and retry
# (AI deploys specific fix based on error analysis)
```

### Workflow 4: Complete Pipeline Retraining

**Problem**: Need to retrain all models with fixes

**Solution**:

```bash
# Use the automated pipeline
python src/moola/scripts/runpod_retrain_pipeline.py \
    --host 213.173.98.6 \
    --port 14385 \
    --models logreg,rf,xgb,simple_lstm,cnn_transformer
```

This runs:
1. Classical models (logreg, rf, xgb) on CPU
2. SimpleLSTM baseline on GPU
3. CNN-Transformer with pre-trained encoder + fixes
4. Stack ensemble on OOF predictions

Results are validated automatically, with per-class accuracy checks.

## Error Detection

The training monitor automatically detects:

| Error Type | Pattern | Suggestion |
|-----------|---------|------------|
| CUDA OOM | `OutOfMemoryError` | Reduce batch size |
| NaN Loss | `loss: nan` | Reduce learning rate |
| Class Collapse | `Class 1.*0.0000` | Check encoder freezing |
| Gradient Explosion | `loss: inf` | Use gradient clipping |
| Import Error | `ModuleNotFoundError` | Check venv activation |
| Shape Mismatch | `RuntimeError.*shape` | Check data preprocessing |

Example:

```python
from moola.validation import TrainingMonitor

monitor = TrainingMonitor(verbose=True)

# Monitor training logs
for line in training_output:
    monitor.process_line(line)

# Check for issues
if monitor.has_errors():
    errors = monitor.get_errors(severity="critical")
    for error in errors:
        print(f"Error: {error.error_type}")
        print(f"Fix: {error.suggestion}")
```

## File Organization

```
src/moola/
├── runpod/
│   ├── __init__.py
│   └── scp_orchestrator.py          # Core orchestrator
├── scripts/
│   ├── test_orchestrator.py         # Test suite
│   ├── deploy_fixes.py              # Incremental deployment
│   └── runpod_retrain_pipeline.py   # Full retraining pipeline
└── validation/
    ├── training_monitor.py          # Real-time error detection
    └── training_validator.py        # Post-training validation
```

## API Reference

### RunPodOrchestrator

```python
class RunPodOrchestrator:
    def __init__(
        self,
        host: str,              # RunPod IP (e.g., "213.173.98.6")
        port: int,              # SSH port (e.g., 14385)
        key_path: str,          # SSH key path
        workspace: str = "/workspace/moola",
        timeout: int = 600,
        verbose: bool = True,
    )

    # Core Operations
    def upload_file(local_path, remote_path) -> bool
    def download_file(remote_path, local_path) -> bool
    def upload_directory(local_dir, remote_dir) -> bool
    def download_directory(remote_dir, local_dir) -> bool
    def execute_command(command, timeout=None, stream_output=True) -> int

    # High-Level Operations
    def verify_environment() -> Dict[str, bool]
    def deploy_fixes(fix_files: List[Path]) -> bool
    def run_training(model, device="cuda", encoder_path=None, timeout=3600) -> int
    def download_results(model, output_dir) -> bool
    def download_logs(output_dir) -> bool
    def check_training_status(model) -> Dict[str, Any]
```

### TrainingMonitor

```python
class TrainingMonitor:
    def __init__(verbose: bool = True)

    # Processing
    def process_line(line: str) -> None
    def has_errors() -> bool
    def get_errors(severity: Optional[str] = None) -> List[DetectedError]

    # Metrics
    def get_metrics() -> List[TrainingMetrics]
    def get_latest_metrics() -> Optional[TrainingMetrics]
    def check_convergence(window: int = 5) -> Dict[str, bool]

    # Reporting
    def generate_report() -> str
```

## Troubleshooting

### Issue: SSH connection refused

**Symptoms**: `Connection refused` or timeout errors

**Solutions**:
1. Verify pod is running: Check RunPod dashboard
2. Verify SSH port: Check pod connection info
3. Verify SSH key: `ssh-add ~/.ssh/id_ed25519`
4. Test connection: `ssh root@213.173.98.6 -p 14385`

### Issue: File not found after upload

**Symptoms**: Upload succeeds but file not found

**Solutions**:
1. Check remote path: Should be absolute (start with `/`)
2. Verify workspace exists: `orch.execute_command("ls -la /workspace/moola")`
3. Check permissions: `orch.execute_command("ls -la /workspace/moola/src/")`

### Issue: Import errors on RunPod

**Symptoms**: `ModuleNotFoundError` when running training

**Solutions**:
1. Activate venv: `source /tmp/moola-venv/bin/activate`
2. Check PYTHONPATH: `export PYTHONPATH="/workspace/moola/src:$PYTHONPATH"`
3. Verify installation: `pip list | grep moola`
4. Reinstall if needed: `cd /workspace/moola && pip install -e .`

### Issue: Training hangs

**Symptoms**: No output for extended period

**Solutions**:
1. Check GPU status: `orch.execute_command("nvidia-smi")`
2. Check process: `orch.execute_command("ps aux | grep python")`
3. Check logs: `orch.download_logs("/tmp/logs/")`
4. Kill stuck process: `orch.execute_command("pkill -9 python")`

### Issue: Class collapse persists

**Symptoms**: Model predicts only one class despite fixes

**Solutions**:
1. Verify encoder loaded: Check logs for `[SSL] Loading pre-trained encoder`
2. Verify encoder frozen: Check logs for `[FREEZE] Encoder frozen`
3. Check gradual unfreezing: Should unfreeze at epochs 10, 20, 30
4. Inspect model:
   ```python
   orch.execute_command(
       "python -c 'from moola.models import CnnTransformerModel; "
       "m = CnnTransformerModel(); m._build_model(4, 2); "
       "m.freeze_encoder(); "
       "print(sum(1 for p in m.model.parameters() if not p.requires_grad))'"
   )
   ```

## Best Practices

### 1. Always Test First

Run `test_orchestrator.py` before any deployment to catch SSH/network issues early.

### 2. Deploy Incrementally

Upload only changed files, not the entire codebase. Faster and easier to debug.

### 3. Verify After Upload

Always check that deployed files exist and are correct:

```python
orch.execute_command("cat /workspace/moola/src/moola/models/cnn_transformer.py | grep freeze_encoder")
```

### 4. Monitor Training

Use `monitor_training_with_error_detection()` for automatic issue detection:

```python
exit_code, errors, metrics = monitor_training_with_error_detection(
    orch, "cnn_transformer", device="cuda"
)
```

### 5. Download Logs Immediately

If training fails, download logs right away for debugging:

```python
orch.download_logs("/tmp/debug/")
```

### 6. Validate Results

Always check OOF predictions for class collapse after training:

```python
preds = np.load("/tmp/results/seed_1337.npy")
print(f"Unique classes: {len(np.unique(preds))}")
```

### 7. Use Presets for Common Fixes

Save time with predefined fix bundles:

```bash
python src/moola/scripts/deploy_fixes.py --preset encoder_fixes
```

## Advanced Usage

### Custom Training Command

```python
# Pass custom arguments to training
exit_code = orch.run_training(
    model="cnn_transformer",
    device="cuda",
    encoder_path="/workspace/artifacts/pretrained/encoder_weights.pt",
    extra_args="--n-epochs 100 --batch-size 256"
)
```

### Parallel Training

```python
# Train multiple models simultaneously
import concurrent.futures

models = ["logreg", "rf", "xgb"]

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        executor.submit(orch.run_training, model, "cpu"): model
        for model in models
    }

    for future in concurrent.futures.as_completed(futures):
        model = futures[future]
        exit_code = future.result()
        print(f"{model}: {'✓' if exit_code == 0 else '✗'}")
```

### Custom Monitoring

```python
from moola.validation import TrainingMonitor

monitor = TrainingMonitor(verbose=False)

# Custom error pattern
monitor.ERROR_PATTERNS["custom_error"] = {
    "pattern": r"my custom error pattern",
    "severity": "error",
    "suggestion": "Try this fix",
}

# Monitor logs
for line in training_logs:
    monitor.process_line(line)

# Extract metrics
metrics = monitor.get_metrics()
for m in metrics:
    print(f"Epoch {m.epoch}: Loss={m.train_loss:.4f} Acc={m.train_accuracy:.4f}")
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: RunPod Training

on:
  push:
    branches: [main]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup SSH
        run: |
          mkdir -p ~/.ssh
          echo "${{ secrets.RUNPOD_SSH_KEY }}" > ~/.ssh/id_ed25519
          chmod 600 ~/.ssh/id_ed25519

      - name: Test Orchestrator
        run: python src/moola/scripts/test_orchestrator.py

      - name: Deploy Fixes
        run: python src/moola/scripts/deploy_fixes.py --preset all_models

      - name: Run Training
        run: python src/moola/scripts/runpod_retrain_pipeline.py

      - name: Validate Results
        run: python src/moola/scripts/validate_results.py
```

## Summary

The RunPod SCP Orchestrator provides:

- ✅ **Precise Control**: Upload specific files, not entire codebases
- ✅ **Real-Time Feedback**: Monitor training as it happens
- ✅ **Iterative Debugging**: Upload fix → test → upload fix → test
- ✅ **AI-Friendly**: AI can inspect files, logs, and results directly
- ✅ **Production-Grade**: Error detection, validation, and recovery

Perfect for iterative ML development where rapid debugging is critical.
