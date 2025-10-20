# MLOps Training Orchestration for LSTM Optimization Phase IV

**Comprehensive MLOps pipeline for parallel LSTM optimization experiments on RTX 4090.**

## Overview

Executes 13 experiments across 4 phases with full tracking and reproducibility:
- **Phase 1**: Time warping ablation (4 experiments)
- **Phase 2**: Architecture search (3 experiments)
- **Phase 3**: Pre-training depth search (3 experiments)
- **Phase 4**: Results aggregation and best config selection

**Total estimated time**: ~7-8 hours sequential on RTX 4090 (24GB VRAM)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install mlflow torch torchvision scikit-learn numpy pandas
```

### 2. Prepare Data

Ensure data is ready:
```bash
ls data/processed/
# Should contain:
# - train.npz (labeled training data)
# - test.npz (labeled test data)
# - unlabeled_augmented.npz (augmented unlabeled data for pre-training)
```

### 3. Run Full Pipeline (Recommended)

```bash
# Sequential execution (safe for single RTX 4090)
python scripts/orchestrate_phases.py --mode sequential
```

### 4. Monitor Progress

```bash
# In another terminal
mlflow ui
# Open http://localhost:5000 to view experiments
```

---

## Experiment Matrix

### Phase 1: Time Warping Ablation (4 experiments)

| Experiment ID | Time Warp Sigma | Expected Accuracy | Expected Class 1 | Notes |
|--------------|-----------------|-------------------|------------------|-------|
| `exp_phase1_timewarp_0.10` | 0.10 | 63-65% | 35-40% | Conservative baseline |
| `exp_phase1_timewarp_0.12` | 0.12 | **65-69%** | **45-55%** | **RECOMMENDED** |
| `exp_phase1_timewarp_0.15` | 0.15 | 64-68% | 40-50% | Moderate |
| `exp_phase1_timewarp_0.20` | 0.20 | 60-63% | 15-25% | Current (baseline) |

**Winner Selection**: `max(accuracy)` where `class_1_accuracy >= 30%`

### Phase 2: Architecture Search (3 experiments)

Uses Phase 1 winner's `time_warp_sigma`.

| Experiment ID | Hidden Size | Num Heads | Per-Head Dim | Expected Accuracy |
|--------------|-------------|-----------|--------------|-------------------|
| `exp_phase2_arch_64_4` | 64 | 4 | 16 | 62-66% |
| `exp_phase2_arch_128_8` | 128 | 8 | 16 | **66-70%** ✓ |
| `exp_phase2_arch_128_4` | 128 | 4 | 32 | 64-68% |

**Winner Selection**: `max(accuracy)` where `class_1_accuracy >= 30%`

### Phase 3: Depth Search (3 experiments)

Uses Phase 1+2 winners' configs.

| Experiment ID | Pretrain Epochs | Expected Accuracy | Notes |
|--------------|-----------------|-------------------|-------|
| `exp_phase3_depth_50` | 50 | 65-69% | Current baseline |
| `exp_phase3_depth_75` | 75 | **67-72%** ✓ | **RECOMMENDED** |
| `exp_phase3_depth_100` | 100 | 66-71% | Diminishing returns |

**Winner Selection**: `max(accuracy)` where `class_1_accuracy >= 30%`

---

## Usage Patterns

### Run All Phases (Full Pipeline)

```bash
# Sequential (recommended for single GPU)
python scripts/orchestrate_phases.py --mode sequential

# Parallel (requires 4 GPUs, NOT recommended for RTX 4090)
python scripts/orchestrate_phases.py --mode parallel --num_workers 4
```

### Run Specific Phase

```bash
# Phase 1 only
python scripts/orchestrate_phases.py --mode sequential --phase 1

# Phase 2 only (uses default Phase 1 winner: sigma=0.12)
python scripts/orchestrate_phases.py --mode sequential --phase 2

# Phase 3 only (uses default Phase 1+2 winners)
python scripts/orchestrate_phases.py --mode sequential --phase 3
```

### Run Single Experiment

```bash
python scripts/run_lstm_experiment.py --experiment_id exp_phase1_timewarp_0.12
```

### Aggregate Results

```bash
# Generate analysis report
python scripts/aggregate_results.py

# Custom paths
python scripts/aggregate_results.py \
    --results_dir data/artifacts \
    --output results_analysis.txt
```

---

## Architecture

### Training Pipeline (Per Experiment)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PRE-TRAINING (30-35 min on RTX 4090)                    │
│    - Load unlabeled augmented data (~60K samples)          │
│    - Initialize BiLSTM Masked Autoencoder                  │
│      * hidden_dim=128, bidirectional=True                  │
│      * mask_ratio=0.15, mask_strategy="patch"              │
│    - Train for N epochs (50/75/100)                        │
│    - Save encoder: data/artifacts/pretrained/{exp_id}.pt   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. FINE-TUNING (2.5 min)                                   │
│    - Load pre-trained encoder                              │
│    - Initialize SimpleLSTM                                 │
│      * hidden_size={64,128}, num_heads={4,8}               │
│      * time_warp_sigma={0.10,0.12,0.15,0.20}              │
│    - Freeze encoder for 10 epochs                          │
│    - Unfreeze with 0.5× LR reduction                       │
│    - Train for 50 epochs total                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. EVALUATION                                               │
│    - Predict on test set                                    │
│    - Compute metrics:                                       │
│      * Overall accuracy                                     │
│      * Per-class accuracy (Class 0, Class 1)                │
│      * Precision, Recall, F1                                │
│    - Save results.json                                      │
└─────────────────────────────────────────────────────────────┘
```

### Phase Orchestration

```
PHASE 1 (4 experiments)
  ├─ exp_phase1_timewarp_0.10
  ├─ exp_phase1_timewarp_0.12
  ├─ exp_phase1_timewarp_0.15
  └─ exp_phase1_timewarp_0.20
          ↓ (select winner)
PHASE 2 (3 experiments)
  ├─ exp_phase2_arch_64_4
  ├─ exp_phase2_arch_128_8
  └─ exp_phase2_arch_128_4
          ↓ (select winner)
PHASE 3 (3 experiments)
  ├─ exp_phase3_depth_50
  ├─ exp_phase3_depth_75
  └─ exp_phase3_depth_100
          ↓ (select final winner)
FINAL REPORT
```

---

## MLflow Tracking

### Logged Parameters (per experiment)

```python
{
    "experiment_id": "exp_phase1_timewarp_0.12",
    "phase": 1,
    "time_warp_sigma": 0.12,
    "hidden_size": 128,
    "num_heads": 8,
    "pretrain_epochs": 50,
    "finetune_epochs": 50,
    "device": "cuda",
    # ... all config parameters
}
```

### Logged Metrics

```python
{
    # Pre-training
    "pretrain_time_min": 32.5,
    "pretrain_final_train_loss": 0.0095,
    "pretrain_final_val_loss": 0.0110,
    "pretrain_best_val_loss": 0.0105,

    # Fine-tuning
    "finetune_time_min": 2.3,

    # Evaluation
    "accuracy": 0.6742,
    "class_0_accuracy": 0.8100,
    "class_1_accuracy": 0.5384,
    "class_0_precision": 0.75,
    "class_1_precision": 0.80,
    # ... all metrics

    # Total
    "total_time_min": 34.8,
}
```

### Logged Artifacts

- `results.json`: Full experiment results
- `{exp_id}_encoder.pt`: Pre-trained encoder checkpoint

---

## File Structure

```
moola/
├── scripts/
│   ├── experiment_configs.py          # Experiment matrix definitions
│   ├── run_lstm_experiment.py         # Single experiment runner
│   ├── orchestrate_phases.py          # Phase-based orchestrator
│   ├── aggregate_results.py           # Results analysis
│   └── MLOPS_ORCHESTRATION_README.md  # This file
│
├── MLproject                          # MLflow project definition
│
├── data/
│   ├── processed/                     # Input data
│   │   ├── train.npz
│   │   ├── test.npz
│   │   └── unlabeled_augmented.npz
│   │
│   └── artifacts/                     # Output artifacts
│       ├── pretrained/                # Pre-trained encoders
│       │   ├── exp_phase1_timewarp_0.12_encoder.pt
│       │   └── ...
│       ├── exp_phase1_timewarp_0.10/  # Per-experiment results
│       │   └── results.json
│       ├── exp_phase1_timewarp_0.12/
│       ├── ...
│       ├── phase_iv_final_report.json # Final winner report
│       └── phase_iv_analysis.txt      # Human-readable analysis
│
└── mlruns/                            # MLflow tracking data
    └── 0/
        ├── meta.yaml
        └── {run_id}/
```

---

## Hardware Requirements

### RTX 4090 (24GB VRAM) - Recommended

- **Mode**: Sequential only (parallel will OOM)
- **Pre-training memory**: ~18-20 GB
- **Fine-tuning memory**: ~8-10 GB
- **Expected time per experiment**: ~35-40 minutes
- **Total time (13 experiments)**: ~7-8 hours

### Multi-GPU Setup (4× RTX 4090)

- **Mode**: Parallel for Phase 1 (4 jobs)
- **Phase 1 time**: ~35-40 minutes (same as 1 experiment)
- **Total time**: ~3-4 hours (Phase 1 parallel, Phase 2+3 sequential)

### Resource Monitoring

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor process
ps aux | grep python
```

---

## Winner Selection Algorithm

```python
def select_winner(results, min_class1_accuracy=0.30):
    """
    Selection criteria (in order):
    1. Class 1 accuracy >= 30% (prevent class collapse)
    2. Highest overall accuracy among valid candidates
    """
    valid_results = [
        r for r in results
        if r['class_1_accuracy'] >= min_class1_accuracy
    ]

    if not valid_results:
        return None  # No valid candidates

    return max(valid_results, key=lambda r: r['accuracy'])
```

**Why Class 1 Accuracy Matters**:
- Class collapse (predicting only Class 0) is failure mode
- 30% threshold ensures model learned retracement patterns
- Overall accuracy alone can be misleading (e.g., 63% with 0% Class 1)

---

## Expected Performance

### Conservative Estimate
- **Baseline** (exp_phase1_timewarp_0.20): 60-63% accuracy, 15-25% Class 1
- **Phase IV Winner**: 64-67% accuracy, 40-50% Class 1
- **Improvement**: +4-7% accuracy, +25-35% Class 1 (major improvement)

### Optimistic Estimate
- **Baseline**: 57% accuracy, 0% Class 1 (class collapse)
- **Phase IV Winner**: 68-72% accuracy, 48-58% Class 1
- **Improvement**: +11-15% accuracy, +48-58% Class 1 (full recovery)

---

## Troubleshooting

### OOM (Out of Memory)

```bash
# Reduce batch size in experiment_configs.py
batch_size: int = 256  # Was 512

# Or run experiments one at a time
python scripts/run_lstm_experiment.py --experiment_id exp_phase1_timewarp_0.12
```

### MLflow Not Found

```bash
pip install mlflow

# Or disable MLflow (experiments still work, just no tracking)
# Edit run_lstm_experiment.py: MLFLOW_AVAILABLE = False
```

### Data Not Found

```bash
# Check data directory
ls data/processed/

# If missing, run data pipeline first
python scripts/generate_augmented_data.py  # Or equivalent
```

### CUDA Not Available

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, experiments will run on CPU (very slow, ~10x slower)
# Consider using Google Colab or cloud GPU
```

---

## Advanced Usage

### Custom Experiment

```python
from experiment_configs import ExperimentConfig

custom_exp = ExperimentConfig(
    experiment_id="exp_custom_test",
    phase=1,
    description="Custom configuration test",
    time_warp_sigma=0.13,
    hidden_size=96,
    num_heads=6,
    pretrain_epochs=60,
)

from run_lstm_experiment import ExperimentRunner

runner = ExperimentRunner(custom_exp, data_dir=Path("data/processed"))
results = runner.run()
```

### Export Best Config

```python
import json

# Load final report
with open("data/artifacts/phase_iv_final_report.json", 'r') as f:
    report = json.load(f)

best_config = report['final_recommendation']
print(f"Best experiment: {best_config['experiment_id']}")
print(f"Accuracy: {best_config['accuracy']:.4f}")

# Extract hyperparameters for production
# Update src/moola/config/training_config.py with best config
```

---

## Integration with Existing Pipeline

### Update Training Config

After finding best config, update `src/moola/config/training_config.py`:

```python
# From Phase IV results
TEMPORAL_AUG_TIME_WARP_SIGMA = 0.12  # Was 0.20

# SimpleLSTM architecture
SIMPLE_LSTM_HIDDEN_SIZE = 128  # Was 64
SIMPLE_LSTM_NUM_HEADS = 8      # Was 4

# Pre-training
MASKED_LSTM_N_EPOCHS = 75      # Was 50
```

### Production Training

```python
from moola.models.simple_lstm import SimpleLSTMModel

model = SimpleLSTMModel(
    hidden_size=128,  # From Phase IV
    num_heads=8,      # From Phase IV
    time_warp_sigma=0.12,  # From Phase IV
    # ... other params
)

# Load best pre-trained encoder
model.load_pretrained_encoder(
    Path("data/artifacts/pretrained/exp_phase3_depth_75_encoder.pt")
)

model.fit(X_train, y_train, unfreeze_encoder_after=10)
```

---

## Contact & Support

For issues or questions:
1. Check MLflow UI for detailed logs
2. Review individual experiment results in `data/artifacts/{exp_id}/`
3. Check GPU memory: `nvidia-smi`

---

**Document Version**: 1.0
**Last Updated**: 2025-10-16
**Hardware Target**: RTX 4090 (24GB VRAM)
**Expected Total Time**: 7-8 hours (sequential)
