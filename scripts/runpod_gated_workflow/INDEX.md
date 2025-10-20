# RunPod Gated Workflow - Complete Index

## Quick Start

**First time setup**:
```bash
# 1. Upload code to RunPod
scp -i ~/.ssh/id_ed25519 -r -P 26324 scripts/runpod_gated_workflow root@213.173.110.215:/workspace/moola/scripts/
scp -i ~/.ssh/id_ed25519 -r -P 26324 src root@213.173.110.215:/workspace/moola/

# 2. SSH to RunPod
ssh root@213.173.110.215 -p 26324 -i ~/.ssh/id_ed25519

# 3. Run workflow
cd /workspace/moola
python3 scripts/runpod_gated_workflow/run_all.py

# 4. Download results (from Mac after completion)
scp -i ~/.ssh/id_ed25519 -P 26324 root@213.173.110.215:/workspace/moola/gated_workflow_results.jsonl ./
```

---

## Documentation Files

### Core Documentation

| File | Purpose | Read if... |
|------|---------|-----------|
| **README.md** | Complete workflow guide | You want full details on gates, configuration, and troubleshooting |
| **GATE_SUMMARY.md** | Gate decision tree and metrics | You need to understand gate logic and failure scenarios |
| **SSH_QUICK_REFERENCE.md** | SSH/SCP commands | You need copy-paste commands for upload/download |
| **INDEX.md** | This file - navigation guide | You're new and want to find the right doc |

### Script Files

| File | Type | Purpose |
|------|------|---------|
| `run_all.py` | Orchestrator | Master script - runs all gates sequentially |
| `0_verify_env.py` | Gate | Environment verification |
| `1_smoke_enhanced.py` | Gate | Smoke test baseline |
| `2_control_minirocket.py` | Gate | Classical baseline control |
| `3_pretrain_bilstm.py` | Gate | Pretrain encoder on unlabeled data |
| `4_finetune_enhanced.py` | Gate | Finetune with pretrained encoder |
| `5_augment_train.py` | Gate | Train with augmentation |
| `6_baseline_simplelstm.py` | Gate | Unidirectional baseline |
| `7_ensemble.py` | Gate | Calibrated ensemble |

---

## Reading Guide by Role

### I'm a Data Scientist (want to run experiments)
1. Read: **README.md** (sections: Gates, SSH/SCP Workflow)
2. Skim: **GATE_SUMMARY.md** (understand gate logic)
3. Bookmark: **SSH_QUICK_REFERENCE.md** (for quick commands)
4. Run: `run_all.py`

### I'm a Machine Learning Engineer (debugging failures)
1. Read: **GATE_SUMMARY.md** (complete decision tree)
2. Read: **README.md** (section: Troubleshooting)
3. Run individual gate scripts with modifications
4. Check: `gated_workflow_results.jsonl` for failure logs

### I'm a DevOps Engineer (setting up infrastructure)
1. Read: **SSH_QUICK_REFERENCE.md** (connection details)
2. Read: **README.md** (section: RunPod Connection)
3. Verify: All data files exist at expected paths
4. Test: `0_verify_env.py` first

### I'm New to This Project (onboarding)
1. Read: This file (**INDEX.md**) - you're here!
2. Read: **GATE_SUMMARY.md** (understand workflow philosophy)
3. Read: **README.md** (full context)
4. Try: Run individual gates manually to understand flow

---

## Workflow Phases

### Phase 1: Setup (5 minutes)
- **Docs to read**: SSH_QUICK_REFERENCE.md
- **Scripts to run**: None (upload code only)
- **Expected output**: Code on RunPod at `/workspace/moola`

### Phase 2: Verification (1 minute)
- **Docs to read**: README.md (Gate 0 section)
- **Scripts to run**: `0_verify_env.py`
- **Expected output**: All environment checks pass

### Phase 3: Baseline Establishment (5 minutes)
- **Docs to read**: README.md (Gates 1-2 sections)
- **Scripts to run**: `1_smoke_enhanced.py`, `2_control_minirocket.py`
- **Expected output**: Baseline metrics recorded, MiniRocket < Enhanced

### Phase 4: Pretraining (20 minutes)
- **Docs to read**: README.md (Gate 3 section)
- **Scripts to run**: `3_pretrain_bilstm.py`
- **Expected output**: Encoder saved, linear probe ≥ 55%

### Phase 5: Transfer Learning (45 minutes)
- **Docs to read**: README.md (Gate 4 section)
- **Scripts to run**: `4_finetune_enhanced.py`
- **Expected output**: Finetuned model, F1 > baseline

### Phase 6: Advanced Training (90 minutes)
- **Docs to read**: README.md (Gates 5-6 sections)
- **Scripts to run**: `5_augment_train.py`, `6_baseline_simplelstm.py`
- **Expected output**: Augmented model, baseline comparison

### Phase 7: Ensemble (1 minute)
- **Docs to read**: README.md (Gate 7 section)
- **Scripts to run**: `7_ensemble.py`
- **Expected output**: Ensemble metrics, final results

### Phase 8: Analysis (10 minutes)
- **Docs to read**: README.md (Results Tracking section)
- **Scripts to run**: None (use `jq` for analysis)
- **Expected output**: Best model selection, performance summary

---

## Key Concepts

### Gates
Sequential checkpoints with validation logic. Each gate must pass before the next can run.

### Temporal Splits
Time-based train/val/test splits that prevent look-ahead bias in financial data.

### Pretrained Encoder
BiLSTM encoder trained on unlabeled data using masked reconstruction.

### Linear Probe
Simple validation: freeze encoder, train linear classifier. Tests if encoder learned useful features.

### Transfer Learning
Load pretrained encoder weights into downstream model and finetune.

### Two-Phase Finetuning
1. Freeze encoder, train head only
2. Unfreeze encoder, train end-to-end with lower LR

### Pseudo-Sample Augmentation
Generate synthetic training samples with distribution matching and quality checks.

### Calibrated Ensemble
Combine models with isotonic regression calibration for better probability estimates.

---

## Common Questions

### Q: Can I skip gates?
**A**: Yes, use `--start-gate` and `--end-gate` flags. But be aware of dependencies:
- Gate 4 requires Gate 3 (pretrained encoder)
- Gate 5 requires Gate 4 (finetuned model for comparison)
- Gate 7 requires Gates 4-6 (models to ensemble)

### Q: How do I debug a failed gate?
**A**:
1. Check `gated_workflow_results.jsonl` for failure reason
2. Read GATE_SUMMARY.md decision tree for that gate
3. Run gate individually with verbose logging
4. Modify gate script parameters and re-run

### Q: Can I modify gate configurations?
**A**: Yes! Each gate script is standalone. Edit hyperparameters directly:
- Epochs: `n_epochs=60`
- Batch size: `batch_size=512`
- Learning rate: `learning_rate=5e-4`
- Mask strategy: `mask_strategy="patch"`

### Q: What if I disconnect during training?
**A**: Use `nohup` for background execution:
```bash
nohup python3 scripts/runpod_gated_workflow/run_all.py > workflow.log 2>&1 &
```
Workflow continues even if SSH disconnects.

### Q: How do I know which model is best?
**A**: Check `gated_workflow_results.jsonl`:
```bash
cat gated_workflow_results.jsonl | jq -s 'max_by(.metrics.val_f1 // 0) | {gate: .gate, f1: .metrics.val_f1, model: .model}'
```

### Q: Can I run this on my local GPU?
**A**: Yes! Change paths in gate scripts:
- `/workspace/moola` → `/path/to/your/moola`
- Update data paths to match your local structure
- Use `device="cuda"` if GPU available, `"cpu"` otherwise

### Q: What's the minimum hardware requirement?
**A**:
- GPU: 24GB VRAM recommended (RTX 4090, A100, etc.)
- CPU: 8+ cores
- RAM: 32GB+
- Disk: 50GB free space

---

## File Dependencies

```
run_all.py
├── 0_verify_env.py
├── 1_smoke_enhanced.py
├── 2_control_minirocket.py (reads Gate 1 results)
├── 3_pretrain_bilstm.py
├── 4_finetune_enhanced.py (reads Gate 1 results, loads Gate 3 encoder)
├── 5_augment_train.py (reads Gate 4 results, loads Gate 3 encoder)
├── 6_baseline_simplelstm.py (reads Gate 4 results)
└── 7_ensemble.py (reads all previous results, loads Gate 4 model)
```

**Key**: Each gate appends to `gated_workflow_results.jsonl` and later gates read previous results for comparison.

---

## Output Artifacts

### Training Artifacts
```
/workspace/moola/artifacts/
├── pretrained/
│   └── encoder_v1.pt                    # Gate 3 output
├── models/
│   ├── enhanced_finetuned_v1.pt        # Gate 4 output
│   ├── enhanced_augmented_v1.pt        # Gate 5 output
│   └── simple_lstm_baseline_v1.pt      # Gate 6 output
```

### Results Artifacts
```
/workspace/moola/
├── gated_workflow_results.jsonl        # All gates append here
└── workflow.log                         # nohup output (if used)
```

### Data Artifacts (already exist)
```
/workspace/moola/data/
├── processed/
│   └── train_clean.parquet             # 98 labeled samples
├── raw/
│   └── unlabeled_windows.parquet       # 11,873 unlabeled samples
└── artifacts/splits/v1/
    └── fold_0.json                      # Temporal split
```

---

## Success Checklist

Before running workflow:
- [ ] Code uploaded to RunPod
- [ ] SSH connection works
- [ ] Data files exist (run Gate 0)
- [ ] GPU available (`nvidia-smi`)

After running workflow:
- [ ] All gates passed (or acceptable failures)
- [ ] Results file downloaded
- [ ] Models downloaded
- [ ] Best model identified
- [ ] Validation metrics recorded

For production:
- [ ] Best model tested on held-out test set
- [ ] Model performance documented
- [ ] Deployment plan created
- [ ] Monitoring set up

---

## Support Resources

### Internal Documentation
- Phase 0 Analysis: `PHASE0_EXECUTIVE_SUMMARY.md`
- Architecture Docs: `docs/ARCHITECTURE.md`
- Getting Started: `docs/GETTING_STARTED.md`
- Pretraining Guide: `PRETRAINING_ORCHESTRATION_GUIDE.md`
- SSH Workflow: `WORKFLOW_SSH_SCP_GUIDE.md`

### Project Files
- Config: `src/moola/config/training_config.py`
- Models: `src/moola/models/`
- Pretraining: `src/moola/pretraining/`
- CLI: `src/moola/cli.py`

### Constraint Document
- `.claude/SYSTEM.md` - System constraints and workflow rules

---

## Version History

- **v1.0** (2025-10-18): Initial gated workflow implementation
  - 8 gates (0-7)
  - Strict validation with automatic abort
  - RunPod SSH/SCP integration
  - Comprehensive documentation

---

## Contact & Feedback

If gates consistently fail or documentation is unclear:
1. Check GATE_SUMMARY.md decision tree
2. Review PHASE0 analysis for data issues
3. Verify temporal splits are correct
4. Consider simplifying workflow (skip augmentation gates)

**Remember**: The workflow is designed to fail fast and surface issues early. A gate failure is validation working correctly, not a bug.
