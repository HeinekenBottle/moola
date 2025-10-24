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

**⚠️ CRITICAL: Uncertainty-weighted loss REQUIRED but CLI flag not implemented yet!**

**Current workaround:** Change default in code (see below) until CLI flag is added.

```bash
# SSH into RunPod
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_RUNPOD_IP
cd /workspace/moola

# ✅ STEP 1: Enable uncertainty weighting in code (REQUIRED)
# Edit src/moola/models/enhanced_simple_lstm.py line 206:
# FROM: use_uncertainty_weighting: bool = False,
# TO:   use_uncertainty_weighting: bool = True,

# ✅ STEP 2: Train with multi-task pointers
python3 -m moola.cli train \
  --model enhanced_simple_lstm \
  --predict-pointers \
  --device cuda \
  --n-epochs 60

# ✅ OPTION: With pre-training boost (+3-5% accuracy)
python3 -m moola.cli train \
  --model enhanced_simple_lstm \
  --predict-pointers \
  --pretrained-encoder artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt \
  --freeze-encoder true \
  --device cuda \
  --n-epochs 60

# Optional: Pre-train new encoder (if you have new unlabeled data)
python3 -m moola.cli pretrain-bilstm \
  --n-epochs 50 \
  --device cuda \
  --mask-strategy patch \
  --batch-size 256
```

**TODO: Add `--use-uncertainty-weighting` CLI flag to `cli.py:train` function**

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
│   │   └── (features built on-the-fly from parquet during pre-training)
│   ├── labeled/
│   │   ├── train_latest.parquet       # Current training set (174 samples)
│   │   ├── train_latest_11d.parquet   # Legacy 11-feature format (deprecated)
│   │   ├── train_latest_relative.parquet # Current 10-feature format
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
│   │   └── jade_encoder_5yr_v1.pt     # Jade encoder (10 features, 5-year NQ data)
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
│   ├── jade_core.py        # Jade production model (85K params)
│   ├── jade_pretrain.py    # Jade pre-training model (MAE)
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
│   ├── relativity.py       # 10-feature pipeline (current)
│   └── zigzag.py          # Causal swing detection
├── utils/                  # Utilities
│   ├── results_logger.py   # JSON results logging
│   ├── seeds.py            # Reproducibility
│   └── metrics.py          # Evaluation metrics
└── config/                 # Hydra configuration files
```

### Model Architecture Details

**Jade (Production Model):**
- **Model ID**: `moola-lstm-m-v1.0` (codename: Jade)
- **Input**: (batch, 105, 11) - **11 features** from relativity.py
- **BiLSTM Encoder**: 128 hidden units × 2 directions = 256 total
- **Architecture**: 2-layer BiLSTM with global average pooling
- **Multi-task Learning**: Classification + pointer prediction (center, length)
- **Total params**: ~85K
- **Why**: Optimal for small labeled dataset (174 samples)
- **Location**: `src/moola/models/jade_core.py`

**CRITICAL CONFIGURATION (Must Enable):**
```python
# ✅ CORRECT: Use uncertainty-weighted loss (research-backed optimal strategy)
model = EnhancedSimpleLSTMModel(
    predict_pointers=True,
    use_uncertainty_weighting=True,  # ← REQUIRED (not default!)
)

# ❌ WRONG: Manual lambda weights (current default, suboptimal)
model = EnhancedSimpleLSTMModel(
    predict_pointers=True,
    use_uncertainty_weighting=False,  # Default is False
    loss_alpha=1.0,  # Manual weights don't adapt
    loss_beta=0.7
)
```

**Why Uncertainty Weighting?**
- Learns optimal task balance automatically (σ² parameters)
- Prevents manual tuning of λ_type and λ_pointer
- Research-validated: Kendall et al., CVPR 2018
- Formula: `(1/2σ²)L_pointer + log(σ) + (1/σ²)L_type + log(σ)`

**Architecture Validation (✅ Verified Correct):**
| Component | Implementation | Status |
|-----------|----------------|--------|
| Pointer encoding | center + length (not start/end) | ✅ Correct |
| Huber loss delta | δ = 0.08 (8 timesteps transition) | ✅ Correct |
| Loss function | Uncertainty-weighted available | ⚠️ Must enable flag |
| Center weight | 1.0 (higher than length 0.8) | ✅ Correct |

**Jade Pre-training (Optional):**
- **Model**: `JadePretrainer` - masked autoencoder for self-supervised learning
- **Input**: (batch, 105, 11) - **11 features** from relativity.py
- **BiLSTM Encoder**: 128 hidden × 2 directions (same as Jade core)
- **Decoder**: Linear(256 → 11) for masked reconstruction
- **Loss**: Huber (δ=1.0) on masked positions only
- **Mask ratio**: 15% of timesteps
- **Data**: 5-year NQ parquet file (1.8M bars → ~34K windows)
- **Transfer**: Encoder weights → Jade core model
- **Why**: Self-supervised learning on abundant unlabeled data
- **Expected boost**: +3-5% accuracy (84% → 87-89%)
- **Location**: `src/moola/models/jade_pretrain.py`
- **Guide**: See `JADE_PRETRAINING_GUIDE.md` for details

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

### Why Jade Pre-training?
- Abundant unlabeled data (1.8M bars) vs. scarce labeled data (174 samples)
- Self-supervised learning improves generalization
- +3-5% accuracy improvement demonstrated (84% → 87-89%)
- Transfer learning: Jade encoder weights → Jade core model

### Why Jade (BiLSTM, Not Transformer)?
- Small parameter count (85K) appropriate for small dataset (174 samples)
- Bidirectional: better context understanding for pattern classification
- Fast training: 10-15 minutes on GPU
- Pre-training compatible with masked autoencoding
- Proven performance: 84% baseline → 87-89% with pre-training

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

### Running Pre-training Experiments
```bash
# Pre-train Jade encoder on 5-year NQ data (RunPod)
bash scripts/runpod_batch_size_sweep.sh

# Or single experiment:
python3 scripts/train_jade_pretrain.py \
  --config configs/windowed.yaml \
  --data data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet \
  --epochs 50 \
  --batch-size 1024 \
  --device cuda

# Fine-tune with pre-trained encoder
python3 -m moola.cli train \
  --model jade \
  --pretrained-encoder artifacts/encoders/pretrained/jade_encoder_5yr_v1.pt \
  --device cuda
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

| Model | Configuration | Accuracy | Class 1 (Minority) | Training Time |
|-------|--------------|----------|-------------------|---------------|
| Logistic Regression | Baseline | 79% | 22% | 5s |
| SimpleLSTM | Single-task | 84% | 48% | 8m |
| EnhancedSimpleLSTM | Multi-task (manual λ) | ~58% | ~35% | 12m |
| EnhancedSimpleLSTM | **Multi-task (uncertainty-weighted)** | **Expected: 65-70%** | **Expected: 45-55%** | **12m** |
| EnhancedSimpleLSTM | + Pre-training (optional) | **Expected: 70-75%** | **Expected: 50-60%** | **28m** |
| Ensemble (5 models) | Stacking | 89% | 65% | 45m |

**Hardware:** RTX 4090 GPU on RunPod

**⚠️ WARNING:** Manual λ weights (loss_alpha=1.0, loss_beta=0.7) are suboptimal for multi-task learning. Always use `--use-uncertainty-weighting` flag for production training.

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

### Multi-task model performing poorly (accuracy < 60%)

**Symptom:** EnhancedSimpleLSTM with pointer prediction shows ~58% accuracy or class collapse

**Root Cause:** Uncertainty-weighted loss NOT enabled (using manual λ weights instead)

**Solution (CLI flag not yet implemented):**
```bash
# ✅ Change code default in src/moola/models/enhanced_simple_lstm.py:206
# FROM: use_uncertainty_weighting: bool = False,
# TO:   use_uncertainty_weighting: bool = True,

# Then train normally:
python3 -m moola.cli train \
  --model enhanced_simple_lstm \
  --predict-pointers \
  --device cuda
```

**Alternative (Python API):**
```python
from moola.models import EnhancedSimpleLSTMModel

model = EnhancedSimpleLSTMModel(
    predict_pointers=True,
    use_uncertainty_weighting=True,  # ← Enable this
    device="cuda"
)
model.fit(X, y, expansion_start=starts, expansion_end=ends)
```

**Verification:** Check training logs for uncertainty parameters (σ_ptr, σ_type)
```python
# During training, you should see:
# "Pointer σ: 0.XXX, Type σ: 0.YYY"
# These are the learned uncertainty weights
```

**Why Manual Weights Fail:**
- Multi-task learning requires dynamic task balancing
- Manual λ_type=1.0, λ_pointer=0.7 don't adapt during training
- One task can dominate, causing the other to collapse
- Uncertainty weighting learns optimal balance automatically

**TODO:** Add `--use-uncertainty-weighting` flag to CLI for easier access

## Related Documentation

- `README.md` - Project overview and quick start
- `CLAUDE.md` - This file (project context for AI assistants)
- `JADE_PRETRAINING_GUIDE.md` - **Complete guide for pre-training Jade encoder on 5-year NQ data**
- `docs/GETTING_STARTED.md` - Complete setup guide
- `docs/ARCHITECTURE.md` - Deep technical dive
- `WORKFLOW_SSH_SCP_GUIDE.md` - RunPod SSH/SCP workflow
- `src/moola/models/MODEL_REGISTRY.md` - Stones model documentation
- `/Users/jack/projects/candlesticks/CLAUDE.md` - Annotation system integration

## Important Notes

1. **✅ VERIFIED: Jade uses 11 features** - See `src/moola/features/relativity.py` for the current feature pipeline (6 candle + 4 swing + 1 expansion)
2. **✅ VERIFIED: Model names** - "Jade" is the production model (`jade_core.py`), "JadePretrainer" is for pre-training (`jade_pretrain.py`)
3. **Pre-training guide** - See `JADE_PRETRAINING_GUIDE.md` for complete instructions on pre-training Jade encoder on 5-year NQ data
4. **Batch size recommendation** - Use 1024 for fastest training (all batch sizes 512-1024 fit comfortably in 24GB VRAM)
5. **Data source** - 5-year NQ parquet file: `data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet` (1.8M bars)
4. **Always use `python3` and `pip3`** (not `python` or `pip`) - prevents context loss in pre-commit hooks
5. **Never commit without pre-commit hooks** - they enforce code quality automatically
6. **Blacklist D-grade windows permanently** - never reuse low-quality annotations
7. **Session-aware extraction** - prioritize Session C for better keeper rates
8. **Merge keepers incrementally** - don't wait for 100+ samples, merge at 30+
9. **Candlesticks integration** - respect data isolation between normal and review modes

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
