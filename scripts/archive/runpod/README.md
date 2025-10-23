# RunPod Utility Scripts

**Purpose:** Validation and diagnostic utilities for RunPod GPU training environment.

## Philosophy

This project uses **manual SSH/SCP workflow** for RunPod training (see `CLAUDE.md` and `.claude/skills/runpod-workflow.md`). These scripts are **validation utilities only**, not deployment automation.

## Available Scripts

### 1. verify_runpod_env.py

**Purpose:** Verify RunPod environment has all required dependencies with compatible versions.

**Usage:**
```bash
# On RunPod (via SSH):
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP
cd /workspace/moola
python3 scripts/runpod/verify_runpod_env.py
```

**Checks:**
- Core ML packages (torch, numpy, sklearn, xgboost, etc.)
- NumPy version compatibility with PyTorch
- CUDA availability and GPU detection
- Version constraints from requirements.txt

**Exit codes:**
- 0: All checks passed
- 1: One or more checks failed

### 2. dependency_audit.py

**Purpose:** Comprehensive dependency validation and compatibility checking.

**Usage:**
```bash
# On RunPod (via SSH):
python3 scripts/runpod/dependency_audit.py [--verbose]
python3 scripts/runpod/dependency_audit.py --check-imports-only
python3 scripts/runpod/dependency_audit.py --save-report audit_report.json
```

**Features:**
- Module import validation
- Version compatibility checking
- Environment info collection
- JSON report generation

**Options:**
- `--verbose`: Detailed output
- `--check-imports-only`: Only validate imports
- `--version-compatibility`: Only check version compatibility
- `--save-report FILE`: Save audit report to JSON file

## Workflow Integration

These scripts are meant to be run **manually on RunPod** after SSH connection:

```bash
# 1. SSH into RunPod
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP

# 2. Navigate to workspace
cd /workspace/moola

# 3. Run validation (optional, for debugging)
python3 scripts/runpod/verify_runpod_env.py

# 4. Run training (main workflow)
python3 -m moola.cli train --model enhanced_simple_lstm --device cuda
```

## What This Folder Does NOT Contain

- ❌ Shell scripts (violate project constraints)
- ❌ Automated deployment scripts (use manual SSH/SCP instead)
- ❌ Orchestration scripts (use manual commands instead)

See `.claude/skills/runpod-workflow.md` for the current RunPod workflow.

## Archived Scripts

The following scripts were archived to `~/moola_archive/` on 2025-10-21:

**Shell scripts** (archived to `~/moola_archive/scripts_runpod_shell_scripts/`):
- `build_runpod_bundle.sh`
- `runpod_baseline_workflow.sh`
- `runpod_deploy_bundle.sh`
- `runpod_quick_train.sh`

**Python deployment scripts** (archived to `~/moola_archive/scripts_runpod_python_deployment/`):
- `bulletproof_deployment.py`
- `deploy_to_fresh_pod.py`
- `pre_deployment_check.py`

**Reason:** These scripts implemented automated deployment workflows that conflict with the project's manual SSH/SCP approach (per CLAUDE.md).

## Related Documentation

- `CLAUDE.md` - Project constraints and RunPod workflow
- `.claude/skills/runpod-workflow.md` - Step-by-step RunPod workflow
- `src/moola/runpod/README.md` - RunPod SCP orchestrator (Python API)

