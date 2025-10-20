# RunPod Gated Training Workflow

Complete execution workflow for strict gated training pipeline on RunPod GPU.

## Overview

This workflow implements **7 sequential gates** with strict validation and automatic failure handling. Each gate must pass before proceeding to the next.

**Philosophy**: Fail fast, validate early, prevent progression on degradation.

## RunPod Connection

```bash
# SSH to RunPod
ssh root@213.173.110.215 -p 26324 -i ~/.ssh/id_ed25519

# Verify environment
cd /workspace/moola
nvidia-smi
```

**Hardware**: RTX 4090 (24GB VRAM)
**Image**: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

## Data Locations (from Phase 0)

```
/workspace/moola/data/processed/train_clean.parquet        # 98 labeled samples
/workspace/moola/data/raw/unlabeled_windows.parquet        # 11,873 unlabeled samples
/workspace/moola/data/artifacts/splits/v1/fold_0.json     # Temporal split
```

## Gates

### Gate 0: Environment Verification

**Purpose**: Validate CUDA, data files, and temporal splits.

**Checks**:
- CUDA availability (RTX 4090)
- All required data files exist
- Temporal splits are valid (no overlap, correct ordering)
- Labeled data schema validation

**Exit**: `0` = passed, `1` = failed

```bash
python3 scripts/runpod_gated_workflow/0_verify_env.py
```

---

### Gate 1: Smoke Test - EnhancedSimpleLSTM

**Purpose**: Quick validation run (3 epochs) to establish baseline metrics.

**Configuration**:
- Model: EnhancedSimpleLSTM
- Epochs: 3 (smoke test only)
- Pretrained encoder: Load if exists (≥80% match required)
- Frozen encoder: Yes
- Augmentation: No

**Gates**:
- Must load pretrained encoder with ≥80% tensor match (if encoder exists)
- Record baseline metrics for comparison

**Output**: Baseline metrics in `gated_workflow_results.jsonl`

```bash
python3 scripts/runpod_gated_workflow/1_smoke_enhanced.py
```

---

### Gate 2: Control Test - MiniRocket

**Purpose**: Validate that deep learning with pretraining outperforms classical methods.

**Configuration**:
- Model: MiniRocket + RidgeClassifierCV
- Same split as Gate 1

**Gates**:
- If MiniRocket F1 ≥ Enhanced F1 → **ABORT WORKFLOW**
- Deep learning should outperform classical baselines

**Rationale**: If MiniRocket beats Enhanced, pretraining is ineffective or data quality is poor.

```bash
python3 scripts/runpod_gated_workflow/2_control_minirocket.py
```

---

### Gate 3: Pretrain BiLSTM Encoder

**Purpose**: Pretrain bidirectional LSTM encoder on unlabeled data.

**Configuration**:
- Model: BiLSTM Masked Autoencoder
- Data: 11,873 unlabeled samples
- Strategy: Patch masking (15%, patch_size=7)
- Epochs: 50
- Batch size: 512
- Device: cuda

**Gates**:
- Linear probe validation accuracy ≥ 55% → Encoder quality gate
- If probe < 55% → **ABORT** (encoder too weak for transfer learning)

**Output**: Encoder saved to `/workspace/moola/artifacts/pretrained/encoder_v1.pt`

```bash
python3 scripts/runpod_gated_workflow/3_pretrain_bilstm.py
```

**Expected duration**: ~20 minutes on RTX 4090

---

### Gate 4: Finetune EnhancedSimpleLSTM

**Purpose**: Finetune with pretrained encoder using advanced strategies.

**Configuration**:
- Model: EnhancedSimpleLSTM
- Pretrained encoder: Load from Gate 3
- Epochs: 60 (full training)
- Strategy: Two-phase finetuning
  - Phase 1: Freeze encoder (3 epochs)
  - Phase 2: Progressive unfreeze with discriminative LRs
- Augmentation: Temporal (jitter, scaling, time_warp)
- Techniques: L2-SP regularization, gradient clipping, EMA, SWA

**Gates**:
- Must improve over Gate 1 baseline smoke run
- If no improvement → **ABORT** (pretraining not helping)

**Output**: Finetuned model saved to `/workspace/moola/artifacts/models/enhanced_finetuned_v1.pt`

```bash
python3 scripts/runpod_gated_workflow/4_finetune_enhanced.py
```

**Expected duration**: ~45 minutes on RTX 4090

---

### Gate 5: Train with Pseudo-Sample Augmentation

**Purpose**: Enable controlled augmentation on training set only.

**Configuration**:
- Augmentation ratio: 2.0:1 (synthetic:real)
- Max synthetic samples: 210
- Quality threshold: 0.85
- Strategies: Temporal + pattern-based (safe only)
- Val/test sets: 100% real (no augmentation)

**Gates**:
- KS p-value ≥ 0.1 (distribution similarity)
- Quality threshold ≥ 0.85
- Deduplication enforced
- No degradation on real validation set (tolerance: 2pp)

**Output**: Augmented model saved to `/workspace/moola/artifacts/models/enhanced_augmented_v1.pt`

```bash
python3 scripts/runpod_gated_workflow/5_augment_train.py
```

**Note**: Current implementation uses temporal augmentation only. Full pseudo-sample augmentation pipeline to be integrated.

---

### Gate 6: Baseline SimpleLSTM

**Purpose**: Validate pretraining benefit by comparing against unidirectional baseline.

**Configuration**:
- Model: SimpleLSTM (unidirectional)
- Hidden size: 32 (smaller than Enhanced: 128)
- Training: From scratch (no pretraining)
- Epochs: 60

**Gates**:
- Should underperform EnhancedSimpleLSTM with pretraining
- If SimpleLSTM ≥ Enhanced → **WARNING** (re-inspect pretraining benefit)

**Output**: Baseline model saved to `/workspace/moola/artifacts/models/simple_lstm_baseline_v1.pt`

```bash
python3 scripts/runpod_gated_workflow/6_baseline_simplelstm.py
```

---

### Gate 7: Ensemble

**Purpose**: Combine approved models with calibrated averaging.

**Configuration**:
- Models: EnhancedSimpleLSTM (finetuned)
- Calibration: Isotonic regression
- Strategy: Calibrated probability averaging

**Gates**:
- Ensemble should match or improve best individual model
- If degradation > 1pp → **WARNING**

**Output**: Ensemble metrics in `gated_workflow_results.jsonl`

```bash
python3 scripts/runpod_gated_workflow/7_ensemble.py
```

---

## Master Orchestrator

Run all gates sequentially with automatic failure handling:

```bash
# Run all gates (0-7)
python3 scripts/runpod_gated_workflow/run_all.py

# Resume from specific gate
python3 scripts/runpod_gated_workflow/run_all.py --start-gate 3

# Run subset of gates
python3 scripts/runpod_gated_workflow/run_all.py --start-gate 3 --end-gate 5
```

**Behavior**:
- Executes gates sequentially
- On failure: STOP immediately, log reason
- Exit code: `0` = all passed, `1` = at least one failed
- Resume capability via `--start-gate` flag

---

## SSH/SCP Workflow

### Initial Setup (from Mac)

```bash
# 1. SCP code to RunPod
scp -i ~/.ssh/id_ed25519 -r -P 26324 \
    /Users/jack/projects/moola/scripts/runpod_gated_workflow \
    root@213.173.110.215:/workspace/moola/scripts/

scp -i ~/.ssh/id_ed25519 -r -P 26324 \
    /Users/jack/projects/moola/src \
    root@213.173.110.215:/workspace/moola/

# 2. SSH to RunPod
ssh root@213.173.110.215 -p 26324 -i ~/.ssh/id_ed25519

# 3. Verify environment
cd /workspace/moola
python3 scripts/runpod_gated_workflow/0_verify_env.py
```

### Execute Workflow (on RunPod)

```bash
# Run all gates
python3 scripts/runpod_gated_workflow/run_all.py

# Or run individually for debugging
python3 scripts/runpod_gated_workflow/1_smoke_enhanced.py
python3 scripts/runpod_gated_workflow/2_control_minirocket.py
# ... etc
```

### Retrieve Results (from Mac)

```bash
# Get results file
scp -i ~/.ssh/id_ed25519 -P 26324 \
    root@213.173.110.215:/workspace/moola/gated_workflow_results.jsonl \
    /Users/jack/projects/moola/

# Get trained models
scp -i ~/.ssh/id_ed25519 -r -P 26324 \
    root@213.173.110.215:/workspace/moola/artifacts/models \
    /Users/jack/projects/moola/artifacts/

# Get pretrained encoder
scp -i ~/.ssh/id_ed25519 -r -P 26324 \
    root@213.173.110.215:/workspace/moola/artifacts/pretrained \
    /Users/jack/projects/moola/artifacts/
```

---

## Results Tracking

All gates append results to `/workspace/moola/gated_workflow_results.jsonl` in JSON Lines format.

**Format**:
```json
{
  "gate": "1_smoke_enhanced",
  "timestamp": "2025-10-18T12:34:56Z",
  "model": "enhanced_simple_lstm",
  "config": {...},
  "metrics": {
    "train_acc": 0.850,
    "val_acc": 0.720,
    "val_f1": 0.715
  },
  "status": "passed"
}
```

**Analysis** (on Mac):

```bash
# Show all results
cat gated_workflow_results.jsonl | jq .

# Show only passed gates
cat gated_workflow_results.jsonl | jq 'select(.status == "passed")'

# Compare F1 scores across gates
cat gated_workflow_results.jsonl | jq '{gate: .gate, f1: .metrics.val_f1}'

# Find best model
cat gated_workflow_results.jsonl | jq -s 'max_by(.metrics.val_f1 // 0)'
```

---

## Troubleshooting

### Gate 0 fails: CUDA not available
```bash
# Check GPU
nvidia-smi

# Verify PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Gate 1 fails: Pretrained encoder not found
```bash
# Check if encoder exists
ls -lh /workspace/moola/artifacts/pretrained/encoder_v1.pt

# Run Gate 3 first if missing
python3 scripts/runpod_gated_workflow/3_pretrain_bilstm.py
```

### Gate 2 fails: MiniRocket >= Enhanced
- Indicates pretraining is ineffective
- Check pretrained encoder quality (Gate 3 linear probe)
- Review data quality and splits

### Gate 3 fails: Linear probe < 55%
- Encoder quality too low
- Increase pretraining epochs (default: 50)
- Try different masking strategy (random, block, patch)
- Check unlabeled data quality

### Out of memory errors
```bash
# Reduce batch size in gate scripts
# Edit gate script and change batch_size from 512 to 256
vim scripts/runpod_gated_workflow/3_pretrain_bilstm.py
```

### Resume after failure
```bash
# Resume from failed gate
python3 scripts/runpod_gated_workflow/run_all.py --start-gate 4
```

---

## Expected Timeline

| Gate | Task | Duration | Total |
|------|------|----------|-------|
| 0 | Environment verification | <1 min | ~1 min |
| 1 | Smoke test (3 epochs) | ~2 min | ~3 min |
| 2 | MiniRocket control | ~3 min | ~6 min |
| 3 | Pretrain BiLSTM (50 epochs) | ~20 min | ~26 min |
| 4 | Finetune Enhanced (60 epochs) | ~45 min | ~71 min |
| 5 | Train with augmentation | ~45 min | ~116 min |
| 6 | SimpleLSTM baseline | ~40 min | ~156 min |
| 7 | Ensemble | <1 min | ~157 min |

**Total**: ~2.6 hours on RTX 4090

---

## Integration with Existing CLI

These scripts use the existing `moola.cli` commands where available:

- `moola.models.enhanced_simple_lstm.EnhancedSimpleLSTMModel`
- `moola.models.simple_lstm.SimpleLSTMModel`
- `moola.pretraining.masked_lstm_pretrain.MaskedLSTMPretrainer`
- `moola.models.pretrained_utils.load_pretrained_strict`

For missing functionality, scripts implement inline with proper guards.

All use **seed=17** for reproducibility.

---

## Key Constraints

- **Workflow**: SSH/SCP only (no Docker, no MLflow)
- **Execution**: RunPod GPU via SSH
- **Results**: JSON Lines file (append mode)
- **Splits**: Temporal only (no random/stratified)
- **Code quality**: Pre-commit hooks enforced on Mac
- **Standalone**: Each script independent, no cross-dependencies

---

## Next Steps After Workflow

1. **Retrieve results**: SCP `gated_workflow_results.jsonl` to Mac
2. **Analyze metrics**: Compare F1 scores across gates
3. **Select best model**: Based on validation F1 and training time
4. **Production deployment**: Use approved model from ensemble
5. **Iteration**: Re-run workflow with improved configurations

---

## Files

```
scripts/runpod_gated_workflow/
├── README.md                      # This file
├── run_all.py                     # Master orchestrator
├── 0_verify_env.py               # Gate 0: Environment verification
├── 1_smoke_enhanced.py           # Gate 1: Smoke test
├── 2_control_minirocket.py       # Gate 2: Control test
├── 3_pretrain_bilstm.py          # Gate 3: Pretrain encoder
├── 4_finetune_enhanced.py        # Gate 4: Finetune with pretraining
├── 5_augment_train.py            # Gate 5: Train with augmentation
├── 6_baseline_simplelstm.py      # Gate 6: Baseline comparison
└── 7_ensemble.py                 # Gate 7: Ensemble
```

---

## Author Notes

This workflow implements **strict gated training** with the following principles:

1. **Fail fast**: Each gate validates assumptions before proceeding
2. **Control tests**: MiniRocket ensures deep learning adds value
3. **Quality gates**: Linear probe validates encoder quality (≥55%)
4. **Comparison gates**: Baseline SimpleLSTM validates pretraining benefit
5. **No regression**: Augmentation must not degrade validation performance
6. **Reproducibility**: Fixed seed (17), deterministic splits, append-only logging

All gates are **standalone** and can be run individually for debugging.
