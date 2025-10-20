# RunPod SCP Orchestrator

**AI-Driven ML Training Infrastructure for Iterative Debugging**

## Purpose

Enable AI models (like Claude Code) to interact directly with RunPod GPU pods for:

- **Precise deployment**: Upload specific fixed files, not entire codebases
- **Real-time monitoring**: Stream training logs and detect errors immediately
- **Iterative debugging**: Fix → Deploy → Test → Repeat cycle
- **Result validation**: Download and inspect artifacts immediately

## Why This Exists

Traditional shell-script-based deployment is a black box:
- Can't inspect intermediate results
- All-or-nothing deployments
- Post-mortem debugging only
- Hard for AI to iterate

The SCP orchestrator fixes this by making every step transparent and controllable.

## Quick Start

```python
from moola.runpod import RunPodOrchestrator

# 1. Initialize
orch = RunPodOrchestrator(
    host="213.173.98.6",
    port=14385,
    key_path="~/.ssh/id_ed25519"
)

# 2. Verify environment
orch.verify_environment()

# 3. Deploy fixes
orch.deploy_fixes([
    "src/moola/models/cnn_transformer.py",
    "src/moola/config/training_config.py"
])

# 4. Train with monitoring
from moola.validation import monitor_training_with_error_detection

exit_code, errors, metrics = monitor_training_with_error_detection(
    orch, "cnn_transformer", device="cuda"
)

# 5. Download results
orch.download_results("cnn_transformer", "/tmp/results/")
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Local Machine (AI Agent)                                │
│                                                          │
│  ┌─────────────────────────────────────────────┐        │
│  │ RunPodOrchestrator                          │        │
│  │  - upload_file()                            │        │
│  │  - download_file()                          │        │
│  │  - execute_command()                        │        │
│  │  - run_training()                           │        │
│  └─────────────────────────────────────────────┘        │
│           │                                              │
│           │ SCP/SSH                                      │
└───────────┼──────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│ RunPod GPU Pod                                          │
│                                                          │
│  /workspace/moola/                                      │
│    ├── src/moola/                                       │
│    │   ├── models/                                      │
│    │   │   └── cnn_transformer.py    ← SCP upload      │
│    │   ├── config/                                      │
│    │   │   └── training_config.py    ← SCP upload      │
│    │   └── pipelines/                                   │
│    │       └── oof.py                                   │
│    ├── data/                                            │
│    │   └── processed/                                   │
│    ├── artifacts/                                       │
│    │   ├── oof/                      ← SCP download     │
│    │   └── pretrained/                                  │
│    └── logs/                         ← SCP download     │
│                                                          │
│  Training Process:                                      │
│    python -m moola.cli oof --model cnn_transformer     │
│                             ↓                           │
│                      Streaming logs                     │
└─────────────────────────────────────────────────────────┘
            │
            │ Real-time output
            ▼
┌─────────────────────────────────────────────────────────┐
│ TrainingMonitor                                         │
│  - Detect errors (OOM, NaN, collapse)                  │
│  - Extract metrics (loss, accuracy)                    │
│  - Suggest fixes                                        │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. RunPodOrchestrator

Main class for RunPod interaction.

**Key Methods**:

```python
# File operations
orch.upload_file(local, remote)           # Upload single file
orch.download_file(remote, local)         # Download single file
orch.upload_directory(local, remote)      # Upload directory recursively
orch.download_directory(remote, local)    # Download directory recursively

# Command execution
orch.execute_command(cmd, timeout=600, stream_output=True)  # Run SSH command

# High-level operations
orch.verify_environment()                 # Pre-flight checks
orch.deploy_fixes(file_list)             # Deploy specific fixes
orch.run_training(model, device, encoder_path)  # Execute training
orch.download_results(model, output_dir)  # Download OOF predictions
orch.download_logs(output_dir)           # Download training logs
```

### 2. TrainingMonitor

Real-time log monitoring with automatic error detection.

**Detected Errors**:
- CUDA OOM
- NaN/Inf losses
- Class collapse
- Shape mismatches
- Import errors
- Gradient explosions

**Usage**:

```python
from moola.validation import TrainingMonitor

monitor = TrainingMonitor(verbose=True)

for line in training_output:
    monitor.process_line(line)

if monitor.has_errors():
    errors = monitor.get_errors(severity="critical")
    for e in errors:
        print(f"Fix: {e.suggestion}")
```

### 3. Deployment Scripts

#### test_orchestrator.py

Test suite to verify orchestrator functionality:

```bash
python src/moola/scripts/test_orchestrator.py
```

Tests:
- SSH connectivity
- File upload/download
- Directory operations
- Environment verification
- Python imports

#### deploy_fixes.py

Incremental fix deployment:

```bash
# Deploy specific files
python src/moola/scripts/deploy_fixes.py \
    --files src/moola/models/cnn_transformer.py

# Deploy preset bundle
python src/moola/scripts/deploy_fixes.py --preset encoder_fixes
```

Presets:
- `encoder_fixes`: Encoder freezing fixes
- `augmentation_fixes`: Data augmentation
- `loss_fixes`: Loss functions
- `all_models`: All model files
- `all_pipelines`: All pipeline files

#### runpod_retrain_pipeline.py

Complete retraining pipeline:

```bash
python src/moola/scripts/runpod_retrain_pipeline.py \
    --models logreg,rf,xgb,simple_lstm,cnn_transformer
```

Trains:
1. Classical models (CPU)
2. SimpleLSTM baseline (GPU)
3. CNN-Transformer with encoder (GPU)
4. Stack ensemble (CPU)

Auto-validates results and detects class collapse.

## Common Workflows

### Workflow 1: Deploy Encoder Fix

```python
# 1. Initialize
orch = RunPodOrchestrator(host="213.173.98.6", port=14385, key_path="~/.ssh/id_ed25519")

# 2. Deploy fix
orch.upload_file(
    "src/moola/models/cnn_transformer.py",
    "/workspace/moola/src/moola/models/cnn_transformer.py"
)

# 3. Verify deployment
orch.execute_command("grep -n 'freeze_encoder' /workspace/moola/src/moola/models/cnn_transformer.py")

# 4. Retrain
exit_code = orch.run_training("cnn_transformer", device="cuda")

# 5. Validate
orch.download_results("cnn_transformer", "/tmp/results/")
```

### Workflow 2: Debug Training Failure

```python
# 1. Train with monitoring
exit_code, errors, metrics = monitor_training_with_error_detection(
    orch, "cnn_transformer", device="cuda"
)

# 2. If failed, download logs
if exit_code != 0:
    orch.download_logs("/tmp/debug/")

    # 3. Analyze errors
    for error in errors:
        print(f"Issue: {error.error_type}")
        print(f"Fix: {error.suggestion}")

    # 4. Deploy fix
    # (AI edits file based on error)

    # 5. Retry
    exit_code = orch.run_training("cnn_transformer", device="cuda")
```

### Workflow 3: Update Hyperparameters

```python
# 1. Edit config locally
# (AI modifies training_config.py)

# 2. Deploy just the config
orch.upload_file(
    "src/moola/config/training_config.py",
    "/workspace/moola/src/moola/config/training_config.py"
)

# 3. Retrain with new hyperparameters
orch.run_training("cnn_transformer", device="cuda")
```

## Error Detection Reference

| Error Type | Pattern | Auto-Detected | Suggestion |
|-----------|---------|---------------|------------|
| CUDA OOM | `OutOfMemoryError` | ✓ | Reduce batch size |
| NaN Loss | `loss: nan` | ✓ | Reduce learning rate |
| Class Collapse | `Class 1.*0.0000` | ✓ | Check encoder freezing |
| Gradient Explosion | `loss: inf` | ✓ | Use gradient clipping |
| Shape Mismatch | `RuntimeError.*shape` | ✓ | Check preprocessing |
| Import Error | `ModuleNotFoundError` | ✓ | Check venv |
| Encoder Not Found | `FileNotFoundError.*encoder` | ✓ | Run SSL pre-training |

## Best Practices

### 1. Always Test First

```bash
python src/moola/scripts/test_orchestrator.py
```

Verifies SSH, file operations, and environment before training.

### 2. Deploy Incrementally

Upload only changed files:

```python
orch.deploy_fixes([
    "src/moola/models/cnn_transformer.py"  # Only upload what changed
])
```

Not the entire codebase.

### 3. Monitor Training

Use `monitor_training_with_error_detection()` for automatic issue detection.

### 4. Validate Results

Check for class collapse after training:

```python
preds = np.load("/tmp/results/seed_1337.npy")
assert len(np.unique(preds)) >= 2, "Class collapse detected"
```

### 5. Download Logs on Failure

```python
if exit_code != 0:
    orch.download_logs("/tmp/debug/")
```

## Troubleshooting

### SSH Connection Issues

```bash
# Test connection manually
ssh root@213.173.98.6 -p 14385 -i ~/.ssh/id_ed25519

# Check pod status on RunPod dashboard
# Verify SSH key is added: ssh-add ~/.ssh/id_ed25519
```

### Import Errors

```python
# Check venv activation
orch.execute_command("which python")

# Check PYTHONPATH
orch.execute_command("echo $PYTHONPATH")

# Reinstall if needed
orch.execute_command("cd /workspace/moola && pip install -e .")
```

### Training Hangs

```python
# Check GPU status
orch.execute_command("nvidia-smi")

# Check running processes
orch.execute_command("ps aux | grep python")

# Kill if stuck
orch.execute_command("pkill -9 python")
```

## Documentation

- **Full Runbook**: `docs/runpod_orchestrator_runbook.md`
- **Quick Start Example**: `examples/runpod_quickstart.py`
- **Test Suite**: `src/moola/scripts/test_orchestrator.py`

## Integration

### With Task Master AI

```python
# Task: Deploy encoder fixes and retrain
task_master.next()  # Get next task

# Use orchestrator for deployment
orch.deploy_fixes(["src/moola/models/cnn_transformer.py"])

# Train
exit_code, errors, metrics = monitor_training_with_error_detection(
    orch, "cnn_transformer", device="cuda"
)

# Update task
task_master.update_subtask(
    id="3.2",
    prompt=f"Training complete. Exit code: {exit_code}. Errors: {len(errors)}"
)
```

### With Claude Code

```python
# Claude can use this directly in conversations:
# "Deploy the encoder fix and retrain CNN-Transformer on RunPod"

from moola.runpod import RunPodOrchestrator

orch = RunPodOrchestrator(...)
orch.deploy_fixes([...])
exit_code = orch.run_training("cnn_transformer", device="cuda")
```

## Performance

- **File upload**: ~1 MB/s (depends on network)
- **Command execution**: ~100ms latency
- **Training**: Real-time streaming (no buffering delay)
- **Result download**: ~5 MB/s

## Security

- Uses SSH key authentication (no passwords)
- StrictHostKeyChecking=no for automation (RunPod pods are ephemeral)
- Files uploaded with 644 permissions
- No sudo required (running as root on pod)

## Future Enhancements

- [ ] Parallel training across multiple pods
- [ ] Automatic checkpoint resumption
- [ ] Cost tracking and optimization
- [ ] Multi-region deployment
- [ ] Automatic pod provisioning
- [ ] Model artifact caching
- [ ] Distributed training support

## Support

For issues or questions:
1. Check `docs/runpod_orchestrator_runbook.md`
2. Run `test_orchestrator.py` to verify setup
3. Inspect logs with `orch.download_logs()`
4. Check RunPod dashboard for pod status
