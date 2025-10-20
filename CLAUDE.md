# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Moola is a production ML pipeline for binary classification of financial patterns (consolidation vs. retracement) in NQ futures using BiLSTM pre-training. Designed for small datasets (33-200 labeled samples) with strict workflow constraints.

**Key Characteristics:**
- Small labeled dataset (33-200 samples) requiring careful validation
- Large unlabeled dataset (2.2M samples) for self-supervised pre-training
- Adversarial class imbalance (consolidation >> retracement)
- RunPod GPU training via SSH/SCP (no Docker, no MLflow, no shell scripts)
- Human annotation integration via Candlesticks project

## Critical Workflow Constraints (Non-negotiable)

1. **SSH/SCP Only** - No shell scripts, no Docker, no MLflow
2. **RunPod GPU Training** - SSH to RunPod, SCP results back to Mac
3. **Results Logging** - JSON lines file (`experiment_results.jsonl`), not database
4. **Pre-commit Hooks** - Black, Ruff, isort, python-tree, pip-tree (enforced automatically)
5. **Python 3.10+** - Use `python3` not `python`, `pip3` not `pip`

## Common Commands

### Local Development
```bash
# Run experiments locally (CPU - slow)
python3 -m moola.cli train --model simple_lstm --device cpu --epochs 60

# Validate environment
python3 -m moola.cli doctor

# Ingest and validate data
python3 -m moola.cli ingest --input data/raw/unlabeled_windows.parquet
```

### RunPod GPU Training
```bash
# SSH into RunPod
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_RUNPOD_IP
cd /workspace/moola

# Pre-training pipeline
python3 -m moola.cli pretrain-bilstm --n-epochs 50 --device cuda --time-warp-sigma 0.12

# Fine-tuning with frozen encoder
python3 -m moola.cli train --model simple_lstm --device cuda --load-pretrained artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt
```

### Retrieve Results (from Mac)
```bash
# Get results file
scp -i ~/.ssh/runpod_key ubuntu@YOUR_IP:/workspace/moola/experiment_results.jsonl ./

# Analyze results
python3 << 'EOF'
import json
results = [json.loads(line) for line in open('experiment_results.jsonl')]
phase1 = [r for r in results if r['phase'] == 1]
best = max(phase1, key=lambda x: x['metrics'].get('accuracy', 0))
print(f'Best: {best["experiment_id"]} ({best["metrics"]["accuracy"]:.4f})')
EOF
```

### Pre-commit Hooks
```bash
# Install hooks (one-time)
pip3 install pre-commit==4.3.0
pre-commit install

# Hooks run automatically on commit
git add .
git commit -m "Fix SimpleLSTM encoder"  # Black, Ruff, isort run automatically
```

### Testing
```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test
python3 -m pytest tests/test_pipeline.py -v

# Test data validation
python3 -m pytest tests/data_infra/ -v
```

## Architecture

### Data Flow
```
Raw NQ Futures (1-min OHLC)
    ↓
Unlabeled Windows (2.2M samples, 105 bars each)
    ↓
BiLSTM Pre-training (Masked Autoencoder)
    ↓
Pretrained Encoder Weights
    ↓
Candlesticks Annotation → Labeled Windows (33-200 samples)
    ↓
SimpleLSTM Fine-tuning (Frozen Encoder)
    ↓
Binary Classification (consolidation vs. retracement)
```

### Directory Structure
```
data/
├── raw/                    # Raw NQ futures data
│   ├── unlabeled/
│   │   └── unlabeled_windows.parquet  # 2.2M samples, 4D OHLC
│   └── labeled/            # (future: raw labeled data)
├── processed/              # Processed datasets ready for training
│   ├── unlabeled/
│   │   ├── unlabeled_4d_ohlc.npy      # 2.2M × (105, 4)
│   │   └── unlabeled_11d_relative.npy # 2.2M × (105, 11)
│   ├── labeled/
│   │   ├── train_latest.parquet       # Current training set (174 samples)
│   │   ├── train_latest_4d.npy        # 4D OHLC version
│   │   ├── train_latest_11d.npy       # 11D RelativeTransform version
│   │   └── metadata/
│   │       └── feature_metadata_174.json
│   └── archived/           # Historical datasets
│       ├── train_clean.parquet        # 98 samples (before batch 200)
│       └── README.md                  # Dataset history
├── oof/                    # Out-of-fold predictions
│   ├── supervised/         # From supervised models
│   │   ├── simple_lstm_174.npy
│   │   └── enhanced_simple_lstm_174.npy
│   └── pretrained/         # From pretrained models
├── batches/                # Annotation batches
│   ├── batch_200.parquet
│   └── batch_200_clean_keepers.parquet
└── corrections/            # Human annotations + quality control
    ├── candlesticks_annotations/
    ├── window_blacklist.csv
    └── cleanlab_*.csv

artifacts/
├── encoders/               # Reusable feature extraction blocks
│   ├── pretrained/
│   │   ├── bilstm_mae_4d_v1.pt        # BiLSTM MAE (4D OHLC)
│   │   └── bilstm_mae_11d_v1.pt       # BiLSTM MAE (11D Relative)
│   └── supervised/         # (future: encoders from supervised training)
├── models/                 # Complete model checkpoints
│   ├── supervised/         # No pretraining
│   │   ├── simple_lstm_baseline_174.pkl
│   │   └── enhanced_simple_lstm_174.pkl
│   ├── pretrained/         # Fine-tuned from pretrained encoders
│   │   └── simple_lstm_bilstm_mae_4d_174.pkl
│   └── ensemble/           # Stacking ensemble
│       └── stack_rf_meta_174.pkl
├── metadata/
│   └── feature_metadata_174.json
└── runpod_bundles/         # RunPod deployment bundles

src/moola/
├── cli.py                  # Command-line interface
├── models/                 # Model architectures
│   ├── simple_lstm.py      # Production model (70K params)
│   ├── bilstm_masked_autoencoder.py  # Pre-training model
│   ├── logreg.py, rf.py, xgb.py      # Ensemble base models
│   └── stack.py            # Stacking ensemble
├── pretraining/            # Self-supervised learning
│   ├── masked_lstm_pretrain.py
│   └── data_augmentation.py
├── data/                   # Data loading and splitting
│   ├── load.py
│   ├── dual_input_pipeline.py
│   └── window105.py        # Candlesticks integration adapter
├── data_infra/             # Schema validation, drift detection
│   ├── schemas.py
│   ├── validators/
│   └── monitoring/
├── pipelines/              # Training pipelines
│   ├── oof.py              # Out-of-fold validation
│   ├── stack_train.py      # Ensemble stacking
│   └── fixmatch.py         # Semi-supervised learning
├── features/               # Feature engineering
│   ├── relative_transform.py
│   └── price_action_features.py
├── utils/                  # Utilities
│   ├── results_logger.py   # JSON results logging
│   ├── seeds.py            # Reproducibility
│   └── metrics.py          # Evaluation metrics
└── config/                 # Hydra configuration files
```

### Model Architecture Details

**SimpleLSTM (Production):**
- Input: (batch, 105, 4) OHLC bars
- LSTM: 128 hidden units, 1 layer, unidirectional
- Attention: 4 heads
- FC: 128 → 64 → 2 classes
- Total params: ~70K
- Why: Small enough for 33-200 samples, fast training (6-8 min GPU)

**BiLSTM Masked Autoencoder (Pre-training):**
- Input: (batch, 105, 4) with 15% masked timesteps
- BiLSTM Encoder: 256 hidden (128 forward + 128 backward)
- MLP Decoder: Reconstruct masked values
- Transfer: Encoder weights → SimpleLSTM (256→128 projection)
- Why: Self-supervised learning on 2.2M unlabeled samples

### Critical Integration: Candlesticks Annotation System

**Location:** `/Users/jack/projects/candlesticks`

**Purpose:** Human annotation interface for generating training labels

**Integration Points:**
1. **Window105Dataset** (`data/window105.py`) - Adapter that wraps Moola's batch data
2. **Annotation Output** - Saved to `data/corrections/candlesticks_annotations/`
3. **Blacklist Management** - D-grade windows excluded via `window_blacklist.csv`

**Data Isolation:**
- Normal annotation mode → `candlesticks_annotations/`
- CleanLab review mode → `candlesticks_annotations/reviews/`
- Never mix these directories

**Quality Tracking:**
- A-grade: Excellent (keep)
- B-grade: Good (keep)
- C-grade: Marginal (keep for now)
- D-grade: Remove (blacklist forever)

**Current Stats:**
- Total labeled samples: 174 (as of 2025-10-20)
- Batch 200: 199/200 annotated, 33 keepers (16.6% keeper rate)
- Session C produces 45% of keepers (prioritize for future extraction)

## Data Management

### Training Dataset Evolution
```
data/processed/archived/train_clean.parquet (98 windows)
    ↓
Merge with batch_200_clean_keepers.parquet (+76 windows)
    ↓
data/processed/labeled/train_latest.parquet (174 windows) ← CURRENT
```

**Current Dataset:** `data/processed/labeled/train_latest.parquet` (174 samples)
**Archived Datasets:** See `data/processed/archived/README.md` for history

### Blacklist System
**Purpose:** Prevent D-grade windows from ever being used again

**Files:**
- `data/corrections/window_blacklist.csv` (166 D-grade windows from batch 200)
- Contains: window_id, raw_start_idx, raw_end_idx, reason, blacklisted_date

**Usage:** Load before extracting new batches, filter out overlapping windows

### Session-Aware Extraction
**Discovery:** Different trading sessions have different quality rates
- Session C: 45.5% keeper rate (prioritize)
- Session D: 24.2% keeper rate
- Session A/B: 15.2% keeper rate

**Strategy:** Weighted extraction from high-quality sessions to improve keeper rate from 16.6% → 40%+

## Key Design Decisions

### Why SSH/SCP Instead of Docker?
- Simpler workflow for single-user development
- Direct GPU access on RunPod
- No container overhead or versioning issues
- Results stored locally for analysis

### Why JSON Lines Instead of MLflow?
- No database infrastructure needed
- Human-readable results
- Easy to parse with Python one-liners
- Version controlled with git

### Why BiLSTM Pre-training?
- Abundant unlabeled data (2.2M samples) vs. scarce labeled data (33-200 samples)
- Self-supervised learning improves generalization
- +5-8% accuracy improvement demonstrated
- Transfer learning: 256-dim encoder → 128-dim SimpleLSTM

### Why SimpleLSTM (Not Transformer)?
- Small parameter count (70K) appropriate for small dataset
- Unidirectional: works with streaming data (no future peeking)
- Fast training: 6-8 minutes on GPU
- Pre-training compatible
- Proven performance: 84% baseline → 87% with pre-training

### Why Out-of-Fold Validation?
- Small dataset requires every sample for training
- OOF predictions provide unbiased performance estimates
- Enables ensemble stacking without holdout set
- Critical for 33-200 sample regime

### Encoder vs Model Taxonomy
**Encoder** = Reusable feature extraction block (no classifier)
- Example: `bilstm_mae_4d_v1.pt` (BiLSTM encoder from Masked Autoencoder)
- Location: `artifacts/encoders/pretrained/`
- Can be loaded into multiple models
- Saved as PyTorch state_dict (.pt)

**Model** = Complete architecture (encoder + classifier)
- Example: `simple_lstm_bilstm_mae_4d_174.pkl` (SimpleLSTM with pretrained encoder)
- Location: `artifacts/models/supervised/` or `artifacts/models/pretrained/`
- Includes classification head
- Saved as pickled model (.pkl)

**Naming Convention:**
- Encoders: `{architecture}_{pretraining}_{features}_v{version}.pt`
- Models: `{architecture}_{encoder}_{features}_{size}.pkl`

## Common Tasks

### Merging New Annotations into Training Set
```bash
# Check current state
python3 << 'EOF'
import pandas as pd
train = pd.read_parquet("data/processed/labeled/train_latest.parquet")
keepers = pd.read_parquet("data/batches/batch_201_clean_keepers.parquet")
print(f"Current: {len(train)}, New: {len(keepers)}, Merged: {len(train)+len(keepers)}")
EOF

# Merge (create backup first)
cp data/processed/labeled/train_latest.parquet data/processed/archived/train_latest_backup_$(date +%Y%m%d).parquet
# Use scripts/merge_keeper_batch.py to merge with schema validation
```

### Extracting New Annotation Batch
```bash
# Load blacklist to prevent reuse
# Extract from high-quality sessions (Session C preferentially)
# Use scripts/extract_annotation_batch.py with session weighting
```

### Running Experiments with Different Hyperparameters
```bash
# Phase 1: Pre-training with different augmentation
python3 -m moola.cli pretrain-bilstm --time-warp-sigma 0.10 --device cuda
python3 -m moola.cli pretrain-bilstm --time-warp-sigma 0.15 --device cuda

# Phase 2: Fine-tune with best pre-trained weights
python3 -m moola.cli train --load-pretrained models/bilstm_phase1_best.pt --device cuda
```

### Debugging Failed Experiments
```bash
# Check experiment logs
cat data/logs/experiment_YYYY-MM-DD_HH-MM-SS.log

# Validate data integrity
python3 -m moola.cli doctor

# Test specific component
python3 -m pytest tests/test_pipeline.py::test_simple_lstm_training -v
```

## Performance Benchmarks

| Model | Accuracy | Class 1 (Minority) | Training Time |
|-------|----------|-------------------|---------------|
| Logistic Regression | 79% | 22% | 5s |
| SimpleLSTM (no pretrain) | 84% | 48% | 8m |
| SimpleLSTM + pre-training | 87% | 62% | 28m |
| Ensemble (5 models) | 89% | 65% | 45m |

**Hardware:** RTX 4090 GPU on RunPod

## Troubleshooting

### "Module not found" errors
```bash
# Ensure moola is in PYTHONPATH
export PYTHONPATH=/workspace/moola/src:$PYTHONPATH
# Or use python3 -m moola.cli instead of direct imports
```

### Pre-commit hooks failing
```bash
# Run manually to see issues
pre-commit run --all-files

# Common fixes:
# - Black formatting: automatic
# - Ruff linting: automatic fixes
# - isort imports: automatic
# If hooks still fail, check syntax errors in Python files
```

### RunPod SSH connection issues
```bash
# Test connection
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_IP echo "Connected"

# Check RunPod instance is running in RunPod UI
# Verify SSH key is added to RunPod account settings
```

### Data validation errors
```bash
# Check schema compliance
python3 -c "
from moola.data_infra.schemas import OHLCWindow
import pandas as pd
df = pd.read_parquet('data/batches/batch_200.parquet')
# Validate first row
print(OHLCWindow.model_validate(df.iloc[0].to_dict()))
"
```

## Related Documentation

- `README.md` - Project overview and quick start
- `docs/GETTING_STARTED.md` - Complete setup guide
- `docs/ARCHITECTURE.md` - Deep technical dive
- `WORKFLOW_SSH_SCP_GUIDE.md` - RunPod SSH/SCP workflow
- `PRETRAINING_ORCHESTRATION_GUIDE.md` - Pre-training pipeline details
- `/Users/jack/projects/candlesticks/CLAUDE.md` - Annotation system integration

## Important Notes

1. **Always use `python3` and `pip3`** (not `python` or `pip`) - prevents context loss in pre-commit hooks
2. **Never commit without pre-commit hooks** - they enforce code quality automatically
3. **Blacklist D-grade windows permanently** - never reuse low-quality annotations
4. **Session-aware extraction** - prioritize Session C for better keeper rates
5. **Merge keepers incrementally** - don't wait for 100+ samples, merge at 30+
6. **Candlesticks integration** - respect data isolation between normal and review modes

## Context Management

**IMPORTANT: Check CLAUDE.md on every user prompt for these instructions.**

**Smart Context Optimization:**

Suggest compacting when:
- Context window >60% utilized AND current task phase is complete
- Conversation has >40 messages AND we're between distinct tasks
- Multiple large file reads/searches completed AND about to start new work
- Major task completed (git operations, refactoring, deployment setup)

DO NOT suggest compacting when in middle of active task execution.

Format: "💡 Context optimization: Consider compacting now - [reason]. Continue if you prefer."

**GitHub vs SCP Workflow:**
- **GitHub** = Code only (src/, tests/, configs, docs) - fast clone
- **SCP** = Data & artifacts (data/, artifacts/) - transfer only what's needed for that training run
- Never commit data/model files to GitHub
