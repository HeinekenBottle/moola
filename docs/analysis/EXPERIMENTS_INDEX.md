# Experiments Documentation Index

Quick navigation guide for all experiment-related documentation.

## Quick Start (Pick One)

| If you want to... | Read this |
|-------------------|-----------|
| **Run experiments RIGHT NOW** | [`EXPERIMENTS_QUICKSTART.md`](EXPERIMENTS_QUICKSTART.md) |
| **Understand the design** | [`PARALLEL_EXPERIMENTS_README.md`](PARALLEL_EXPERIMENTS_README.md) |
| **Quick reference commands** | [`EXPERIMENTS_SUMMARY.txt`](EXPERIMENTS_SUMMARY.txt) |
| **Pre-flight checklist** | [`EXPERIMENTS_CHECKLIST.md`](EXPERIMENTS_CHECKLIST.md) |

## Documentation Files

### Primary Guides

1. **[EXPERIMENTS_QUICKSTART.md](EXPERIMENTS_QUICKSTART.md)** (⭐ START HERE)
   - Copy-paste commands to run experiments
   - Quick analysis commands
   - Expected outputs with examples
   - Troubleshooting quick fixes

2. **[PARALLEL_EXPERIMENTS_README.md](PARALLEL_EXPERIMENTS_README.md)** (DETAILED)
   - Complete experiment design and rationale
   - Technical details (architecture, features, loss functions)
   - Interpretation guidelines
   - Integration with production

3. **[EXPERIMENTS_SUMMARY.txt](EXPERIMENTS_SUMMARY.txt)** (REFERENCE)
   - Plain text quick reference
   - All commands in one place
   - Expected outputs summarized
   - Troubleshooting section

4. **[EXPERIMENTS_CHECKLIST.md](EXPERIMENTS_CHECKLIST.md)** (VALIDATION)
   - Pre-flight checklist
   - During-experiment monitoring
   - Post-experiment validation
   - Quality checks and next steps

### Script Files

Located in `/Users/jack/projects/moola/scripts/`:

1. **`experiment_a_threshold_grid.py`** - Experiment A implementation
   - Grid search thresholds 0.30-0.40
   - Computes F1, precision, recall at each threshold
   - Outputs CSV and summary text

2. **`experiment_b_augmentation.py`** - Experiment B implementation
   - Jitter augmentation (σ=0.03, 3x data expansion)
   - 20-epoch training with multi-task loss
   - Generates training curves and history CSV

3. **`run_parallel_experiments.sh`** - Master automation script
   - Runs both experiments in sequence
   - Verifies prerequisites
   - Creates timestamped output directory

4. **`analyze_experiment_results.py`** - Post-experiment analysis
   - Parses results from both experiments
   - Generates summary report
   - Compares experiments and estimates combined gains

## Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. READ: EXPERIMENTS_QUICKSTART.md                          │
│    - Understand what experiments do                         │
│    - Note prerequisites                                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. VERIFY: EXPERIMENTS_CHECKLIST.md                         │
│    - Check baseline checkpoint exists                       │
│    - Verify data and environment                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. RUN: bash scripts/run_parallel_experiments.sh            │
│    - Automated execution (~30 min)                          │
│    - Outputs saved to timestamped directory                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. ANALYZE: python3 scripts/analyze_experiment_results.py   │
│    - Comprehensive summary report                           │
│    - Recommendations for next steps                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. DEPLOY: Follow recommendations from analysis             │
│    - Update inference threshold                             │
│    - Add augmentation to training                           │
│    - Document configuration changes                         │
└─────────────────────────────────────────────────────────────┘
```

## Single-Command Execution

For fully automated execution:

```bash
# 1. Run experiments
bash scripts/run_parallel_experiments.sh

# 2. Analyze results
python3 scripts/analyze_experiment_results.py
```

That's it! Results and recommendations will be printed to console.

## Individual Experiment Execution

If you want to run experiments separately:

### Experiment A Only (5 min)

```bash
python3 scripts/experiment_a_threshold_grid.py \
    --checkpoint artifacts/baseline_100ep/best_model.pt \
    --data data/processed/labeled/train_latest_overlaps_v2.parquet \
    --output results/threshold_grid.csv \
    --device cuda
```

### Experiment B Only (25 min)

```bash
python3 scripts/experiment_b_augmentation.py \
    --data data/processed/labeled/train_latest_overlaps_v2.parquet \
    --output artifacts/augmentation_exp/ \
    --epochs 20 \
    --n-augment 2 \
    --sigma 0.03 \
    --device cuda
```

## Output Directory Structure

After running experiments, expect this structure:

```
results/
├── threshold_grid.csv           # Experiment A: Full results table
└── threshold_summary.txt        # Experiment A: Recommendation

artifacts/
└── augmentation_exp/
    ├── training_history.csv     # Experiment B: Epoch-by-epoch metrics
    ├── training_curves.png      # Experiment B: Visualization
    ├── best_model.pt            # Experiment B: Best checkpoint
    └── metadata.json            # Experiment B: Configuration
```

## Reading Order by Role

### For Data Scientists (want to understand design)

1. `PARALLEL_EXPERIMENTS_README.md` - Full design rationale
2. `EXPERIMENTS_QUICKSTART.md` - Commands to execute
3. Run experiments
4. `EXPERIMENTS_CHECKLIST.md` - Validate results

### For ML Engineers (want to deploy)

1. `EXPERIMENTS_QUICKSTART.md` - Quick execution
2. Run experiments
3. `analyze_experiment_results.py` - Get recommendations
4. `PARALLEL_EXPERIMENTS_README.md` - Integration section

### For Researchers (want all details)

1. `PARALLEL_EXPERIMENTS_README.md` - Complete technical details
2. Review script source code (`scripts/experiment_*.py`)
3. `EXPERIMENTS_CHECKLIST.md` - Quality validation
4. Run experiments with variations

### For Project Managers (want summary)

1. `EXPERIMENTS_SUMMARY.txt` - High-level overview
2. Look at expected outputs section
3. Check success criteria section
4. Review next steps recommendations

## Key Concepts

### Experiment A: Threshold Tuning

**What it does:** Finds the optimal threshold for converting soft span predictions (probabilities 0-1) into hard binary predictions (0 or 1).

**Why it matters:** Default threshold 0.5 may not be optimal. Different thresholds trade off precision vs recall. Goal is to maximize F1 while maintaining acceptable recall.

**Expected outcome:** Threshold around 0.32-0.36 that achieves F1 > 0.23 with recall >= 0.40.

### Experiment B: Data Augmentation

**What it does:** Trains model with 3x more data by adding Gaussian jitter (noise) to original samples.

**Why it matters:** Small dataset (210 samples) limits model capacity. Augmentation provides diversity without collecting more labels.

**Expected outcome:** F1 >= 0.25 if augmentation successfully improves generalization.

### Combined Strategy

If both experiments succeed:
- **Training:** Use augmentation (3x data, σ=0.03)
- **Inference:** Use optimal threshold from Experiment A
- **Expected gain:** Baseline 0.22 → 0.28-0.30 F1 (+27-36%)

## Technical Context

### Model: JadeCompact
- Architecture: 1-layer BiLSTM (96 hidden × 2 directions)
- Parameters: ~52K (appropriate for 210-sample dataset)
- Input: 13 features (12 relativity + position_encoding)
- Multi-task: Classification + pointers + span detection + countdown

### Data
- Training: 210 samples (168 train, 42 val)
- Window size: 105 timesteps (1-minute bars)
- Class imbalance: 7.1% in-span (handled via pos_weight=13.1)

### Loss Function
- Uncertainty-weighted multi-task loss (Kendall et al., CVPR 2018)
- Soft span loss with positive class weighting
- Huber loss for pointer and countdown regression

## Success Metrics

| Metric | Baseline | Target | Best Case |
|--------|----------|--------|-----------|
| **F1 Score** | 0.22 | 0.25 | 0.30+ |
| **Precision** | 0.15-0.20 | 0.18+ | 0.20+ |
| **Recall** | 0.35-0.40 | 0.40+ | 0.45+ |

## Common Questions

**Q: Why 20 epochs for Experiment B (vs 100 for baseline)?**
A: Augmentation should show gains early (10-15 epochs). 20 is sufficient to test concept and saves GPU time.

**Q: Why threshold range 0.30-0.40?**
A: Preliminary analysis showed default 0.5 is too high. Lower thresholds increase recall (critical for span detection).

**Q: Can I run both experiments in parallel?**
A: Yes, but master script runs sequentially to avoid GPU memory contention. If you have 2 GPUs, you can run manually in parallel.

**Q: What if experiments fail to meet targets?**
A: This is valid experimental result. Documents that these specific interventions don't help. Try alternative strategies (different augmentation, architecture changes, etc.).

**Q: How do I deploy the results?**
A: See "Integration with Production" section in `PARALLEL_EXPERIMENTS_README.md`.

## Troubleshooting

For detailed troubleshooting, see:
- `EXPERIMENTS_QUICKSTART.md` - Quick fixes
- `EXPERIMENTS_CHECKLIST.md` - Validation steps
- `PARALLEL_EXPERIMENTS_README.md` - Common failure modes

Quick fixes:
- **"Checkpoint not found"** → Train baseline first
- **"CUDA OOM"** → Reduce batch size (`--batch-size 16`)
- **"F1 near zero"** → Check data format and model checkpoint

## Version History

- **v1.0 (2025-10-27):** Initial experiment design and implementation
  - Experiment A: Threshold grid search
  - Experiment B: Jitter augmentation
  - Master automation script
  - Comprehensive documentation

## Future Enhancements

Potential follow-up experiments:
1. **Augmentation variants:** Time warping, mixup, cutout
2. **Threshold optimization:** Bayesian optimization instead of grid search
3. **Ensemble strategies:** Combine multiple augmentation levels
4. **Architecture search:** Test different LSTM hidden sizes

## Contact

For questions or issues:
1. Check documentation (you're here!)
2. Review script source code
3. Check project `CLAUDE.md` for broader context
