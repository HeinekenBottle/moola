# LSTM Optimization Phase IV - Quick Start Guide

**5-Minute Setup to Experiments**

---

## Step 1: Validate Environment (1 minute)

```bash
cd /Users/jack/projects/moola

# Check dependencies and setup
python3 scripts/setup_mlops_experiments.py --fix
```

**Expected Output**:
```
✓ PyTorch installed
✓ CUDA available: RTX 4090
✓ All data files present
✓ Experiment configs valid
✅ VALIDATION PASSED - Ready to run experiments!
```

**If you see errors**, follow the instructions in the output.

---

## Step 2: Preview Experiments (30 seconds)

```bash
python3 scripts/preview_experiments.py
```

**What you'll see**:
- 13 experiments across 3 phases
- Expected accuracy ranges
- Estimated total time (~5.5 hours)

---

## Step 3: Run Experiments (5-7 hours)

### Option A: Full Pipeline (Recommended)

```bash
# Terminal 1: Run experiments
python3 scripts/orchestrate_phases.py --mode sequential

# Terminal 2: Monitor with MLflow UI
mlflow ui
# Then open: http://localhost:5000
```

**This will**:
1. Run Phase 1 (4 experiments) → Select winner
2. Run Phase 2 (3 experiments) → Select winner
3. Run Phase 3 (3 experiments) → Select final winner
4. Generate final report

### Option B: Test with Phase 1 First

```bash
# Run only Phase 1 (2.3 hours)
python3 scripts/orchestrate_phases.py --mode sequential --phase 1
```

Then review results before continuing.

---

## Step 4: Analyze Results (2 minutes)

```bash
python3 scripts/aggregate_results.py
```

**Generates**:
- `data/artifacts/phase_iv_analysis.txt` - Human-readable report
- `data/artifacts/phase_iv_final_report.json` - Machine-readable results

**Report includes**:
- Winner for each phase
- Performance comparison tables
- Best overall configuration
- Improvement vs baseline

---

## Step 5: Deploy Best Config (5 minutes)

1. **Find winning config**:
   ```bash
   cat data/artifacts/phase_iv_final_report.json | grep experiment_id
   ```

2. **Update production config** (`src/moola/config/training_config.py`):
   ```python
   # Example: If Phase 1 winner is sigma=0.12
   TEMPORAL_AUG_TIME_WARP_SIGMA = 0.12  # Was 0.20

   # Example: If Phase 2 winner is 128 hidden, 8 heads
   # Update SimpleLSTM defaults:
   # (in simple_lstm.py or create new config constants)

   # Example: If Phase 3 winner is 75 epochs
   MASKED_LSTM_N_EPOCHS = 75  # Was 50
   ```

3. **Use best pre-trained encoder** in production:
   ```python
   from moola.models.simple_lstm import SimpleLSTMModel

   model = SimpleLSTMModel(
       hidden_size=128,  # From winner
       num_heads=8,      # From winner
       # ... other params
   )

   # Load best encoder (example ID)
   encoder_path = "data/artifacts/pretrained/exp_phase3_depth_75_encoder.pt"
   model.load_pretrained_encoder(encoder_path)

   model.fit(X_train, y_train, unfreeze_encoder_after=10)
   ```

---

## Troubleshooting

### Data Not Found
```bash
# Error: train.npz, test.npz, unlabeled_augmented.npz not found

# Solution: Run data pipeline first
python3 scripts/generate_augmented_data.py  # Or equivalent data prep script
```

### CUDA Not Available
```bash
# Warning: CUDA not available - experiments will run on CPU (very slow)

# Solution: Use cloud GPU (Google Colab, Paperspace, etc.)
# Or continue on CPU (expect ~10x longer execution time)
```

### MLflow Not Installed
```bash
# Error: MLflow not found

# Solution:
pip install mlflow

# Or experiments still work without MLflow (just no tracking UI)
```

### Out of Memory (OOM)
```bash
# Error: CUDA out of memory during pre-training

# Solution: Reduce batch size
# Edit scripts/experiment_configs.py:
batch_size: int = 256  # Was 512
```

---

## What to Expect

### Time Estimates (RTX 4090)

- **Phase 1** (4 experiments): ~2.3 hours
- **Phase 2** (3 experiments): ~1.7 hours
- **Phase 3** (3 experiments): ~2.0 hours
- **Total**: ~5.5-7 hours

### Performance Expectations

**Baseline** (current):
- Accuracy: 57-63%
- Class 1 Accuracy: 0-25% (class collapse issue)

**After Phase IV** (expected):
- Accuracy: 65-72%
- Class 1 Accuracy: 45-55% (class collapse fixed!)

**Improvement**: +8-15% accuracy, +45-55% Class 1 accuracy

---

## Monitoring Progress

### MLflow UI
```bash
mlflow ui
# Open: http://localhost:5000
```

**What you'll see**:
- All experiment runs
- Real-time metrics
- Parameter comparison
- Artifact storage

### Terminal Output
Each experiment logs:
- Pre-training progress (epoch-by-epoch)
- Fine-tuning progress
- Final test metrics
- Total time

### Files Created
```
data/artifacts/
├── pretrained/
│   ├── exp_phase1_timewarp_0.10_encoder.pt
│   ├── exp_phase1_timewarp_0.12_encoder.pt
│   └── ...
├── exp_phase1_timewarp_0.10/
│   └── results.json
├── exp_phase1_timewarp_0.12/
│   └── results.json
└── ...
```

---

## Advanced Usage

### Run Specific Experiment
```bash
python3 scripts/run_lstm_experiment.py --experiment_id exp_phase1_timewarp_0.12
```

### Run Specific Phase
```bash
# Phase 1 only
python3 scripts/orchestrate_phases.py --mode sequential --phase 1

# Phase 2 only (uses default Phase 1 winner)
python3 scripts/orchestrate_phases.py --mode sequential --phase 2

# Phase 3 only (uses default Phase 1+2 winners)
python3 scripts/orchestrate_phases.py --mode sequential --phase 3
```

### Parallel Execution (NOT recommended for single RTX 4090)
```bash
# Only use if you have 4+ GPUs
python3 scripts/orchestrate_phases.py --mode parallel --num_workers 4
```

---

## Files You'll Use

### Before Experiments
- `scripts/setup_mlops_experiments.py` - Validate environment
- `scripts/preview_experiments.py` - Preview experiment matrix

### During Experiments
- `scripts/orchestrate_phases.py` - Main orchestrator
- MLflow UI (`mlflow ui`) - Monitor progress

### After Experiments
- `scripts/aggregate_results.py` - Analyze results
- `data/artifacts/phase_iv_analysis.txt` - Results report
- `data/artifacts/phase_iv_final_report.json` - Winner config

---

## Support

For detailed information, see:
- **Full Guide**: `scripts/MLOPS_ORCHESTRATION_README.md` (500+ lines)
- **Implementation Summary**: `MLOPS_IMPLEMENTATION_SUMMARY.md`
- **Experiment Configs**: `scripts/experiment_configs.py`

---

**Ready to Start?**

```bash
# 1. Validate
python3 scripts/setup_mlops_experiments.py --fix

# 2. Preview
python3 scripts/preview_experiments.py

# 3. Run
python3 scripts/orchestrate_phases.py --mode sequential

# 4. Analyze
python3 scripts/aggregate_results.py
```

**Good luck! 🚀**
