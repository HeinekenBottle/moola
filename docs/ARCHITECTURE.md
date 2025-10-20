# Architecture & Design of Moola

Technical deep-dive into moola's design, components, and rationale.

## System Overview

```
Unlabeled Data (OHLC) → Pre-training (BiLSTM Encoder)
                              ↓
                    Frozen Encoder + Fine-tuning
                              ↓
                     Labeled Data (98 samples) → Training
                              ↓
                        SimpleLSTM Model
                              ↓
                   Ensemble Stack (5 base models)
                              ↓
                      Binary Classification
```

**Design principle:** Pre-train on abundant unlabeled data to bootstrap learning with limited labeled samples (98).

## Core Components

### 1. Models

#### SimpleLSTM (Production Model)

**Architecture:**
- Input (batch, 105, 4) [sequence of OHLC bars]
- LSTM Layer (128 hidden, unidirectional)
- Multi-Head Attention (4 heads)
- Fully Connected (128 → 64)
- Output (batch, 2) [consolidation vs retracement]

**Why SimpleLSTM?**
- 70K parameters - Small enough for 98 labeled samples (avoid overfitting)
- Unidirectional - Works with online streaming data (no future context needed)
- Pre-training compatible - Can load weights from BiLSTM encoder
- Fast training - 6-8 minutes with GPU, reasonable on CPU

**Key settings:**
- hidden_size = 128
- num_layers = 1
- attention_heads = 4
- dropout = 0.3
- batch_size = 1024
- learning_rate = 1e-3

**File:** `src/moola/models/simple_lstm.py`

#### BiLSTM Masked Autoencoder (Pre-training Model)

**Architecture:**
- Input (batch, 105, 4) → Mask 15% of timesteps
- BiLSTM Encoder (256 hidden: 128 forward + 128 backward)
- MLP Decoder
- Reconstruct Masked Values

**Why BiLSTM for pre-training?**
- Bidirectional context - Better reconstruction (can look backward/forward)
- Self-supervised learning - No labels needed, works on unlabeled data
- Transfer learning - Encoder weights → SimpleLSTM (unidirectional fine-tuning)

**File:** `src/moola/models/bilstm_masked_autoencoder.py`

### 2. Pre-training Pipeline

**Goal:** Learn rich representations from 10,000+ unlabeled OHLC sequences

**Process:**
```python
# Step 1: Generate unlabeled windows
unlabeled_windows = generate_ohlc_windows(raw_data, window_size=105)

# Step 2: Pre-train BiLSTM encoder
encoder = BiLSTMPretrainer(hidden_size=256, masking_ratio=0.15)
encoder.fit(unlabeled_windows, epochs=50)

# Step 3: Transfer to SimpleLSTM
model = SimpleLSTM(hidden_size=128)
model.load_pretrained_encoder(pretrained_weights)  # 256→128

# Step 4: Fine-tune on labeled data
model.fit(labeled_data_X, labeled_data_y, epochs=60, freeze_encoder=True)
```

**Result:** +5-8% accuracy improvement from pre-training

**File:** `src/moola/pretraining/masked_lstm_pretrain.py`

### 3. Data Infrastructure

**Format:** Pydantic schema with validation
- OHLCBar: open, high, low, close
- TimeSeriesWindow: (105, 4) OHLC sequences
- LabeledWindow: OHLC + binary label + confidence

**Validation:**
- OHLC logic (high >= low, etc.)
- Price jump detection (>200% = anomaly)
- Stale data detection
- Missing values check

**File:** `src/moola/data_infra/schemas.py`

### 4. Data Versioning (DVC)

**Purpose:** Reproducible datasets, track data changes

**Benefits:**
- Reproducible data processing
- Track data versions (v1.0, v1.1, etc.)
- Validate data integrity
- Rollback to previous datasets

**File:** `dvc.yaml`, `src/moola/data_infra/lineage/tracker.py`

### 5. Drift Detection

**Methods:**
- Kolmogorov-Smirnov test - Detects distribution shifts
- Population Stability Index (PSI) - Measures population changes
- Wasserstein distance - Optimal transport metric

**File:** `src/moola/data_infra/monitoring/drift_detector.py`

### 6. Configuration System (Hydra)

**Structure:**
```
configs/
├── default.yaml           # Global defaults
├── simple_lstm.yaml       # SimpleLSTM-specific params
├── hardware/
│   ├── cpu.yaml          # CPU settings
│   └── gpu.yaml          # GPU settings
└── ssl.yaml              # SSL pre-training
```

**Override from CLI:**
```bash
python -m moola.cli train \
  --config simple_lstm \
  --model.hidden_size 64 \
  --training.batch_size 512
```

**Files:** `src/moola/config/`, `configs/`

### 7. Results Logging

**Purpose:** Track experiments without database infrastructure

**Format:** JSON lines (one result per line)
```json
{"timestamp": "2025-10-17T14:30:22", "phase": 1, "experiment_id": "exp_1", "metrics": {"accuracy": 0.87}}
```

**Why not MLflow?**
- No database needed
- No Docker required
- Works on RunPod via SCP
- Simple, transparent, queryable

**File:** `src/moola/utils/results_logger.py`

## File Organization

```
src/moola/
├── cli.py                      # Command-line interface
├── models/                     # SimpleLSTM, BiLSTM, alternatives
├── pretraining/                # Pre-training orchestration
├── pipelines/                  # OOF validation, stacking, SSL
├── data_infra/                 # Schemas, validators, drift detection
├── features/                   # Feature engineering
├── config/                     # Hydra configuration
└── utils/                      # Utilities, metrics, logging
```

## Design Decisions

### 1. Why SimpleLSTM for Production?

- **70K params** - Small for 98 samples
- **Unidirectional** - Works with streaming data
- **Pre-training compatible** - Can load BiLSTM weights
- **Proven performance** - 87% accuracy with pre-training

### 2. Why 15% Masking Ratio?

Testing showed 15% is optimal (Goldilocks zone):
- 10% masking → reconstruction too easy
- 15% masking → optimal representation learning
- 25% masking → too hard, unstable training

### 3. Why Focal Loss?

Standard cross-entropy would predict "consolidation" always (61% class prior).

Focal loss forces the model to:
- Learn minority class (retracement, only 38/98 samples)
- Distinguish patterns (not just memorize majority)

### 4. Why SSH/SCP Workflow (Not Docker)?

**Docker:** Complex builds, 15+ min overhead, error-prone
**SSH/SCP:** Instant feedback, no build overhead, seconds to transfer

### 5. Why JSON Results (Not MLflow)?

**MLflow:** Database, server, installation complexity
**JSON:** Single file, append-only, queryable, SCP-friendly

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Training time (SimpleLSTM, 60 epochs, GPU) | 6-8 minutes |
| Pre-training time (BiLSTM, 50 epochs, GPU) | 12-15 minutes |
| Memory usage (batch_size=1024) | ~18 GB (RTX 4090) |
| Accuracy (SimpleLSTM baseline) | 84% |
| Accuracy (SimpleLSTM + pre-training) | 87% |
| Accuracy (Ensemble, 5 models) | 89% |
| Class 1 recall | 62-65% |

## Next Steps

1. **Run experiments:** See [`GETTING_STARTED.md`](GETTING_STARTED.md)
2. **Understand workflows:** See [`WORKFLOW_SSH_SCP_GUIDE.md`](../WORKFLOW_SSH_SCP_GUIDE.md)
3. **Explore code:** Start with `src/moola/models/simple_lstm.py`
4. **Read pre-training guide:** See [`PRETRAINING_ORCHESTRATION_GUIDE.md`](../PRETRAINING_ORCHESTRATION_GUIDE.md)
