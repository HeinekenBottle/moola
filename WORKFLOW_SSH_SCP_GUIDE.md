# SSH/SCP Workflow Guide - RunPod + Mac

**No shell scripts. No Docker. No MLflow. Just SSH, SCP, and this guide.**

---

## 1. Setup (One-time)

### On Your Mac

```bash
# Install pre-commit hooks (one time)
cd ~/projects/moola
pip install pre-commit==4.3.0
pre-commit install

# This enables:
# - Black formatting (automatic)
# - Ruff linting (fixes issues)
# - isort import sorting (automatic)
# All run automatically before commit
```

### On RunPod

Nothing special needed. Just have Python 3.10+ and your models.

---

## 2. Development Workflow

### Phase: Code Changes on Mac

```bash
# Make changes locally
nano src/moola/models/simple_lstm.py

# Pre-commit runs automatically on commit
git add .
git commit -m "Fix SimpleLSTM encoder freezing"
# ✓ Black formats code
# ✓ Ruff fixes issues
# ✓ isort sorts imports
# If pre-commit fails: fix issues, retry commit
```

### Phase: Training on RunPod via SSH

```bash
# SSH into RunPod
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_RUNPOD_IP

# Once connected:
cd /workspace/moola

# Run your training
python -m moola.cli pretrain-bilstm --n-epochs 50 --device cuda
# OR
python -m moola.cli train --model simple_lstm --device cuda

# Training logs to stdout + results_logger file locally
```

### Phase: Retrieve Results via SCP

**After training completes (or while it's running):**

```bash
# From your Mac (different terminal):
scp -i ~/.ssh/runpod_key ubuntu@YOUR_RUNPOD_IP:/workspace/moola/experiment_results.jsonl ./

# View results
cat experiment_results.jsonl | tail -5

# Expected output:
# {"timestamp": "2025-10-17T14:30:22", "phase": 1, "experiment_id": "phase1_sigma_0.12", ...}
# {"timestamp": "2025-10-17T14:45:15", "phase": 1, "experiment_id": "phase1_sigma_0.15", ...}
```

### Phase: Compare and Decide

```bash
# On your Mac, use Python to find best result
python -c "
import json

results = [json.loads(line) for line in open('experiment_results.jsonl')]
phase1 = [r for r in results if r['phase'] == 1]
best = max(phase1, key=lambda x: x['metrics'].get('accuracy', 0))
print(f'Best Phase 1: {best[\"experiment_id\"]} ({best[\"metrics\"][\"accuracy\"]:.4f})')
"

# Use that config for next phase on RunPod
```

---

## 3. Results Logging Integration

### In Your Training Script

```python
from src.moola.utils.results_logger import ResultsLogger

logger = ResultsLogger("experiment_results.jsonl")

# After training
logger.log(
    phase=1,
    experiment_id="phase1_time_warp_0.12",
    metrics={
        "accuracy": 0.87,
        "class_1_accuracy": 0.62,
        "val_loss": 0.31,
    },
    config={
        "time_warp_sigma": 0.12,
        "batch_size": 1024,
        "epochs": 50,
    }
)
```

Results are **locally stored** as JSON lines (one result per line). No database. No infrastructure.

---

## 4. Multi-Experiment Workflow (Phase 1-3)

### Example: Phase 1 (4 parallel runs on RunPod)

```bash
# SSH session 1
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_RUNPOD_IP
cd /workspace/moola
python -m moola.cli pretrain-bilstm --time-warp-sigma 0.10 --device cuda

# SSH session 2 (new terminal, same command different param)
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_RUNPOD_IP
cd /workspace/moola
python -m moola.cli pretrain-bilstm --time-warp-sigma 0.12 --device cuda

# SSH session 3 & 4: repeat with 0.15, 0.20
```

All 4 write to same `experiment_results.jsonl` (append mode - safe for concurrent writes).

### After All Phase 1 Runs Complete

```bash
# SCP all results
scp -i ~/.ssh/runpod_key ubuntu@YOUR_RUNPOD_IP:/workspace/moola/experiment_results.jsonl ./phase1_results.jsonl

# Find winner
python -c "
import json
results = [json.loads(line) for line in open('phase1_results.jsonl')]
winner = max(results, key=lambda x: x['metrics'].get('accuracy', 0))
print(f'Phase 1 Winner: {winner[\"config\"][\"time_warp_sigma\"]}')
"

# Use that config for Phase 2 on RunPod
```

---

## 5. Handling Failures

### Training Fails? Check Immediately

```bash
# While training is running on RunPod:
ssh ubuntu@YOUR_RUNPOD_IP tail -50 moola.log

# See the error instantly
# Fix locally, re-SCP code, retry
```

### Need to Stop a Run?

```bash
# SSH into RunPod
ssh ubuntu@YOUR_RUNPOD_IP
pkill -f "moola.cli"
# Done. No waiting for cleanup scripts.
```

### Results Didn't Save?

```bash
# Check if results_logger file exists
scp ubuntu@YOUR_RUNPOD_IP:/workspace/moola/experiment_results.jsonl .
wc -l experiment_results.jsonl  # See how many results logged
```

---

## 6. File Structure

```
On RunPod:
/workspace/moola/
├── src/moola/        # Your code (synced from Mac via git)
├── data/             # Training data
├── experiment_results.jsonl  # Results (append-only, safe concurrent writes)
└── logs/             # stdout captured (optional)

On Mac:
~/projects/moola/
├── ... (same as RunPod)
├── experiment_results.jsonl  # SCPed from RunPod for review
└── phase1_results.jsonl      # Archived results by phase
```

---

## 7. Pre-commit Hooks (Auto-Enforced)

**These run automatically before every commit on your Mac:**

```bash
# 1. Black - Auto-formats code
#    Ensures consistent style, no arguments

# 2. Ruff - Auto-fixes linting issues
#    Removes unused imports, fixes simple errors

# 3. isort - Auto-sorts imports
#    Organizes imports in standard order

# If any fail:
# 1. Fix the issues it reports
# 2. Run git add again
# 3. git commit again
```

Example:
```bash
$ git commit -m "Fix SimpleLSTM"
Black reformatted src/moola/models/simple_lstm.py
isort reformatted src/moola/models/simple_lstm.py
Ruff fixed 2 issues in src/moola/models/simple_lstm.py

# Pre-commit made changes, retry commit
$ git add .
$ git commit -m "Fix SimpleLSTM"
# ✓ All checks pass
```

---

## 8. Quick Reference

| Task | Command |
|------|---------|
| SSH to RunPod | `ssh -i ~/.ssh/runpod_key ubuntu@IP` |
| SCP results back | `scp -i ~/.ssh/runpod_key ubuntu@IP:/workspace/moola/experiment_results.jsonl ./` |
| View last 5 results | `tail -5 experiment_results.jsonl` |
| Find best result | `python -c "import json; r=[json.loads(l) for l in open('experiment_results.jsonl')]; b=max(r, key=lambda x: x['metrics'].get('accuracy')); print(b)"` |
| Stop training on RunPod | `pkill -f moola.cli` |
| Setup pre-commit (Mac) | `pre-commit install` |
| Check hook status | `pre-commit run --all-files` |

---

## 9. Tips

1. **Keep terminal windows organized:**
   - Terminal 1: SSH to RunPod (training)
   - Terminal 2: Local Mac (SCP, editing)
   - Terminal 3: Monitoring (tail logs)

2. **Check results while training:**
   - Results append to JSON immediately
   - SCP while training is running
   - No need to wait for completion

3. **Name experiments clearly:**
   - `phase1_time_warp_0.12` (better)
   - `exp_1` (avoid)
   - Makes winner selection obvious

4. **Archive results by phase:**
   ```bash
   cp experiment_results.jsonl phase1_results.jsonl
   # Clear for next phase
   rm experiment_results.jsonl
   ```

5. **Never modify experiment_results.jsonl manually:**
   - JSON lines format (one JSON per line)
   - Append-only is intentional
   - Manual edits break parsing

---

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| "Connection refused" | Check RunPod is running: `ssh -i ~/.ssh/runpod_key ubuntu@IP echo hello` |
| "Permission denied (publickey)" | SSH key path wrong or not set: `ssh -i ~/.ssh/runpod_key ubuntu@IP` |
| "No such file" on SCP | Results file doesn't exist yet. Wait for training to complete first. |
| Pre-commit blocks commit | Run `pre-commit run --all-files` to see issues. Fix them, retry commit. |
| Out of GPU memory | Reduce batch_size in training config. Retry. |

---

## That's It

- **No shell scripts** ✓
- **No Docker** ✓
- **No MLflow infrastructure** ✓
- **No multi-hour configuration** ✓
- **No surprises from RunPod** ✓

Just SSH, SCP, code, and results. Clean and simple.
