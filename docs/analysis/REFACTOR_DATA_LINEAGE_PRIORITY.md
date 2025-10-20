# Moola Refactor: Data Lineage & Model Clarity (Priority Plan)

**Created:** 2025-10-18
**Focus:** Fix data tracking, model roles, and RunPod optimization
**Timeline:** 15-20 hours (prioritizes correctness over LOC reduction)
**Owner:** Jack (with Claude Code assistance)

---

## The Real Problem

You lost last night because:

1. **Data confusion** - Don't know which data is where or what's mixed into what
   - Raw unlabeled scattered across locations
   - Labeled windows unclear (105-sample set vs. what exactly?)
   - Synthetic data applied unpredictably (val/test contamination risk)
   - Feature paths inconsistent

2. **Model confusion** - Not clear which model does what or which to use
   - SimpleLSTM vs. Enhanced SimpleLSTM indistinguishable in codebase
   - Enhanced BiLSTM pretrained path exists but not actually used
   - TS2Vec pretrain weights not tracked/versioned
   - Legacy models cluttering the registry

3. **Training proof gaps** - Can't verify what trained on what
   - No manifest showing split purity
   - Pretrained load success/failure not logged
   - Synthetic contamination undetected
   - Augmentation state unknown per run

4. **RunPod inefficiency** - Complex initialization wastes GPU time
   - Flags for local vs. RunPod scattered
   - CPU/GPU device handling brittle
   - Pre-training setup requires manual tuning
   - Checkpoints don't travel cleanly

---

## Solution: Three-Pillar Refactor

### Pillar 1: Data Registry (Immutable, Versioned)
### Pillar 2: Model Roles (Clear, Explicit)
### Pillar 3: Validation & Logging (Prove Correctness)

---

## PILLAR 1: DATA REGISTRY

### Goal
Every run reads exact versioned paths. No ambiguity. No mixing. Provable split purity.

### Structure

```
data/
├── raw_unlabeled/              # Large corpus for TS2Vec pretrain
│   └── parquet files          # OHLC sequences, no labels
│
├── labeled_windows/            # Canonical labeled sets (versioned)
│   ├── v1/                     # Windows-105 current set
│   │   ├── X_ohlc.npy         # Shape: (105, 256, 4) [samples, timesteps, OHLC]
│   │   ├── X_feats20.npy      # Shape: (105, 20) [samples, engineered features]
│   │   ├── y.npy              # Shape: (105,) [labels 0-2 or binary]
│   │   ├── meta.json          # Metadata: T=256, D_ohlc=4, D_feats=20, class_distribution
│   │   └── manifest.json      # Full audit trail (sources, transformations, creator, timestamp)
│   │
│   └── v2/                     # Future iterations (unchanged v1 stays as is)
│
├── unlabeled_windows/          # For TS2Vec self-supervised pretraining
│   ├── v1/                     # Unlabeled version 1
│   │   ├── X_ohlc.npy         # Shape: (?, 256, 4)
│   │   ├── meta.json
│   │   └── manifest.json
│   │
│   └── v1_stratified/          # Variant with stratification control
│
├── synthetic_cache/            # Train-only augmented data (versioned)
│   ├── v1_temporal_q0.9_r0.5/  # Temporal augmentation, 90% quality, 50% ratio
│   │   ├── X_ohlc.npy
│   │   ├── X_feats20.npy
│   │   ├── y.npy
│   │   ├── meta.json
│   │   └── augmentation_config.json  # Exact aug params used
│   │
│   └── v1_mixed_pattern/        # Mixed augmentation variant
│
└── splits/                      # Train/val/test indices (never random)
    ├── fwd_chain_v3.json      # Forward-chaining with purge
    │   ```json
    │   {
    │     "name": "fwd_chain_v3",
    │     "labeled_version": "v1",
    │     "strategy": "forward_chaining",
    │     "purge_days": 20,
    │     "train_indices": [0, 1, 2, ...],
    │     "val_indices": [53, 54, ..., 79],
    │     "test_indices": [80, 81, ..., 104],
    │     "counts": {"train": 53, "val": 27, "test": 25},
    │     "created_by": "jack",
    │     "timestamp": "2025-10-17T14:32:00Z"
    │   }
    │   ```
    │
    └── stratified_v1.json      # Alternative: stratified split
```

### Data Loader Interface

```python
# src/moola/data/registry.py

class DataRegistry:
    """Load versioned, validated datasets."""

    def __init__(self, data_root: Path = Path("data")):
        self.data_root = data_root
        self._validate_structure()

    def load_labeled(
        self,
        version: str = "v1",
        split_name: str = "fwd_chain_v3",
        include_synthetic: bool = False,
        synthetic_version: str = None,
    ):
        """Load labeled windows with optional synthetic augmentation.

        Args:
            version: labeled_windows/v1, v2, etc.
            split_name: splits/fwd_chain_v3.json, etc.
            include_synthetic: if True, append synthetic to train only
            synthetic_version: e.g., "v1_temporal_q0.9_r0.5"

        Returns:
            DataContainer with:
                - X_train, y_train (may include synthetic)
                - X_val, y_val (never synthetic)
                - X_test, y_test (never synthetic)
                - metadata (shapes, split info, synthetic ratio)
                - audit_trail (proof of data lineage)

        Guarantees:
            - val/test contain ZERO synthetic samples
            - shapes match meta.json
            - split is non-random (forward-chaining with purge)
        """
        labeled = self._load_labeled_version(version)
        split = self._load_split(split_name)

        # Validate split indices are valid
        self._validate_split(split, len(labeled["y"]))

        # Extract train/val/test
        X_train_real = labeled["X_ohlc"][split["train_indices"]]
        X_val = labeled["X_ohlc"][split["val_indices"]]
        X_test = labeled["X_ohlc"][split["test_indices"]]

        y_train_real = labeled["y"][split["train_indices"]]
        y_val = labeled["y"][split["val_indices"]]
        y_test = labeled["y"][split["test_indices"]]

        # Optionally append synthetic to train ONLY
        synthetic_ratio = 0.0
        if include_synthetic and synthetic_version:
            synthetic = self._load_synthetic(synthetic_version, labeled["D_ohlc"])
            X_train_synthetic = synthetic["X_ohlc"]
            y_train_synthetic = synthetic["y"]

            # Append
            X_train = np.vstack([X_train_real, X_train_synthetic])
            y_train = np.hstack([y_train_real, y_train_synthetic])

            synthetic_ratio = len(y_train_synthetic) / len(y_train)

            audit = {
                "labeled_version": version,
                "split": split_name,
                "synthetic_applied": synthetic_version,
                "synthetic_ratio": synthetic_ratio,
                "n_real_train": len(y_train_real),
                "n_synthetic_train": len(y_train_synthetic),
                "val_synthetic_count": 0,
                "test_synthetic_count": 0,
            }
        else:
            X_train = X_train_real
            y_train = y_train_real
            audit = {
                "labeled_version": version,
                "split": split_name,
                "synthetic_applied": False,
                "synthetic_ratio": 0.0,
                "n_real_train": len(y_train_real),
                "val_synthetic_count": 0,
                "test_synthetic_count": 0,
            }

        return DataContainer(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            metadata={
                "T": labeled["T"],
                "D_ohlc": labeled["D_ohlc"],
                "D_feats": labeled.get("D_feats", 0),
                "split_name": split_name,
                "split_strategy": split.get("strategy", "unknown"),
            },
            audit_trail=audit,
        )

    def load_unlabeled(self, version: str = "v1"):
        """Load raw unlabeled windows for TS2Vec pretraining."""
        unlabeled = self._load_unlabeled_version(version)
        return {
            "X_ohlc": unlabeled["X_ohlc"],
            "metadata": unlabeled["meta.json"],
            "manifest": unlabeled.get("manifest.json"),
        }

    def _validate_structure(self):
        """Ensure data/ and splits/ exist with required files."""
        required = [
            self.data_root / "labeled_windows" / "v1",
            self.data_root / "splits",
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(
                    f"Data registry incomplete: missing {path}. "
                    f"Run setup: mkdir -p data/{{labeled_windows/v1,unlabeled_windows/v1,synthetic_cache,splits}}"
                )

    def _validate_split(self, split, n_samples):
        """Ensure split indices are non-overlapping and valid."""
        train = set(split["train_indices"])
        val = set(split["val_indices"])
        test = set(split["test_indices"])

        if train & val or train & test or val & test:
            raise ValueError("Split indices overlap (data leak!).")

        if not (train | val | test <= set(range(n_samples))):
            raise ValueError("Split indices out of range.")

        if len(train) + len(val) + len(test) != n_samples:
            raise ValueError("Split does not cover all samples.")

        print(f"✓ Split '{split['name']}' validated: {len(train)} train, {len(val)} val, {len(test)} test")
```

### CLI Usage

```bash
# Train Enhanced SimpleLSTM on labeled_v1 with fwd_chain_v3 split (no synthetic)
moola train \
  --model enhanced_simple_lstm \
  --dataset labeled_windows/v1 \
  --split splits/fwd_chain_v3.json \
  --augment false

# Train with synthetic augmentation (appended to train only)
moola train \
  --model enhanced_simple_lstm \
  --dataset labeled_windows/v1 \
  --split splits/fwd_chain_v3.json \
  --augment true \
  --synthetic synthetic_cache/v1_temporal_q0.9_r0.5

# Train TS2Vec on unlabeled corpus
moola pretrain-ts2vec \
  --unlabeled unlabeled_windows/v1 \
  --output artifacts/ts2vec/encoder_v1.pt
```

---

## PILLAR 2: MODEL ROLES (Clear, Explicit)

### Goal
Developers instantly know: which model to use, what it does, and which are legacy.

### Active Models (In-Repo)

```
models/
├── simple_lstm.py
│   """
│   SimpleLSTM: Baseline unidirectional model.
│
│   - Input: OHLC only (shape: B×T×4)
│   - Encoder: single-layer unidirectional LSTM
│   - Params: ~5k–10k (tiny)
│   - Pretrain: none
│   - Use case: sanity checks, quick tests, baseline
│
│   Config:
│     hidden_size: 32 (default)
│     num_layers: 1
│     dropout: 0.1
│     bidirectional: False
│   """
│
├── enhanced_simple_lstm.py
│   """
│   Enhanced SimpleLSTM: Main production model.
│
│   - Input dual-path:
│     a) OHLC → BiLSTM encoder (can be pretrained)
│     b) 20 engineered features → MLP
│   - Output: fusion layer combines OHLC + features
│   - Params: ~16k–17k
│   - Pretrain: optionally load external BiLSTM weights
│   - Use case: main line, transfer learning, feature fusion
│
│   Config:
│     hidden_size: 128 (OHLC path)
│     num_layers: 2
│     bidirectional: True
│     pretrained_encoder_path: (optional) "artifacts/ts2vec/encoder_v1.pt"
│     freeze_encoder: True (initially), then unfreeze
│     feature_mlp_hidden: [64, 32]
│   """
│
├── adapter_lstm.py
│   """
│   TS2Vec → Adapter LSTM: Transfer learning variant.
│
│   - Pretrain phase: self-supervised TS2Vec on unlabeled OHLC
│   - Finetune phase: freeze encoder, train adapters + small head
│     then gradually unfreeze encoder
│   - Params: encoder (frozen) + adapters (~10k) + head (~5k)
│   - Use case: leverage large unlabeled corpus for transfer
│
│   Config:
│     pretrained_encoder_path: "artifacts/ts2vec/encoder_v1.pt" (required)
│     adapter_hidden: 32
│     freeze_phases: [30, 10, 5] (epochs frozen, then unfreeze gradually)
│   """
│
└── minirocket.py
    """
    MiniRocket: Fixed feature baseline (control model).

    - Input: OHLC only
    - Encoder: fixed random convolutions (no training)
    - Classifier: linear layer
    - Params: ~2k (fixed features) + linear weights
    - Use case: detect data/label issues, sanity check

    Config: none (fixed by design)
    """
```

### Legacy Models (Move to models_extras/)

These are experimental or deprecated. Keep pointers in docs.

```
models_extras/
├── README.md
│   """
│   # Legacy & Experimental Models
│
│   These models are retained for reference or experimental work.
│   They are NOT part of the main pipeline and require manual
│   setup if you want to use them.
│
│   ## Available
│
│   | Model | Status | When to Use | Notes |
│   |-------|--------|------------|-------|
│   | tcn.py | Experimental | Time-series classification comparison | Requires manual feature engineering |
│   | transformer_lstm_hybrid.py | Experimental | Attention-based variant | High memory, slow on RunPod |
│   | rwkv_ts.py | Prototype | RNN-free experiment | Not optimized for OHLC data |
│   | cnn_transformer.py | Deprecated | Replaced by Enhanced SimpleLSTM | Archive only |
│
│   ## Re-enable a Legacy Model
│
│   To use, e.g., TCN:
│
│   1. Copy src/moola/models_extras/tcn.py → src/moola/models/tcn.py
│   2. Add to ModelRegistry (models/registry.py)
│   3. Update CLI: moola train --model tcn
│   4. Run minimal tests to ensure device/data handling works
│   """
│
├── tcn.py
├── transformer_lstm_hybrid.py
├── rwkv_ts.py
└── cnn_transformer.py
```

### Model Registry (Clear Loading)

```python
# src/moola/models/registry.py

class ModelRegistry:
    """Load models with validation."""

    ACTIVE = {
        "simple_lstm": ("SimpleLSTM", "Unidirectional LSTM baseline"),
        "enhanced_simple_lstm": ("EnhancedSimpleLSTM", "Bidirectional LSTM with feature fusion (main)"),
        "adapter_lstm": ("AdapterLSTM", "TS2Vec encoder + adapters (transfer)"),
        "minirocket": ("MiniRocket", "Fixed feature baseline (control)"),
    }

    LEGACY = {
        "tcn": "models_extras/tcn.py (experimental)",
        "transformer_lstm": "models_extras/transformer_lstm_hybrid.py (deprecated)",
        "rwkv_ts": "models_extras/rwkv_ts.py (prototype)",
    }

    @classmethod
    def create(cls, model_name: str, config: dict = None):
        """Create a model by name with validation."""
        if model_name not in cls.ACTIVE:
            if model_name in cls.LEGACY:
                raise ValueError(
                    f"Model '{model_name}' is legacy. "
                    f"See models_extras/README.md to re-enable."
                )
            else:
                raise ValueError(
                    f"Model '{model_name}' not found. "
                    f"Available: {list(cls.ACTIVE.keys())}"
                )

        # Import and instantiate
        if model_name == "simple_lstm":
            from moola.models.simple_lstm import SimpleLSTM
            return SimpleLSTM(config or {})

        elif model_name == "enhanced_simple_lstm":
            from moola.models.enhanced_simple_lstm import EnhancedSimpleLSTM
            return EnhancedSimpleLSTM(config or {})

        # ... etc
```

---

## PILLAR 3: VALIDATION & LOGGING (Prove Correctness)

### Goal
Every run produces a manifest proving: data purity, split correctness, pretrained load success.

### Run Manifest (run.json)

```python
# src/moola/cli.py

def train(
    cfg_dir: str,
    model: str = "enhanced_simple_lstm",
    dataset: str = "labeled_windows/v1",
    split: str = "splits/fwd_chain_v3.json",
    pretrained_encoder: Optional[str] = None,
    augment: bool = False,
    synthetic_version: Optional[str] = None,
    seed: int = 42,
    log_dir: str = "artifacts/runs",
):
    """Train with full validation and manifest logging."""
    import json
    from datetime import datetime
    import subprocess

    # Get git SHA
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()[:8]

    # Create run ID
    run_id = f"{datetime.now().isoformat()}_{git_sha}_{model}"
    run_path = Path(log_dir) / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    # Load data with validation
    registry = DataRegistry()
    data = registry.load_labeled(
        version=dataset.split("/")[-1],
        split_name=split.split("/")[-1],
        include_synthetic=augment,
        synthetic_version=synthetic_version,
    )

    # Load model
    model_instance = ModelRegistry.create(model, config={})

    # If pretrained specified, load and validate
    pretrained_stats = {}
    if pretrained_encoder and model in ["enhanced_simple_lstm", "adapter_lstm"]:
        pretrained_stats = report_state_dict_load(
            model_instance,
            pretrained_encoder,
            freeze_encoder=True,
        )

    # Train
    trainer = Trainer(model_instance, data, run_path)
    results = trainer.fit(epochs=60, early_stopping_patience=20)

    # Compute metrics
    metrics = compute_metrics_pack(results, data)

    # Write manifest
    manifest = {
        "run_id": run_id,
        "git_sha": git_sha,
        "model": model,
        "seed": seed,
        "config_dir": cfg_dir,
        "timestamp": datetime.now().isoformat(),

        # Data lineage
        "data": {
            "labeled_version": dataset,
            "split": split,
            "split_strategy": data.metadata.get("split_strategy"),
            "split_counts": {
                "train": len(data.y_train),
                "val": len(data.y_val),
                "test": len(data.y_test),
            },
            "augmentation": {
                "synthetic_applied": augment,
                "synthetic_version": synthetic_version,
                "synthetic_ratio": data.audit_trail.get("synthetic_ratio", 0.0),
                "n_real_train": data.audit_trail.get("n_real_train"),
                "n_synthetic_train": data.audit_trail.get("n_synthetic_train", 0),
            },
            "audit_trail": data.audit_trail,
        },

        # Pretrained loading proof
        "pretrained": {
            "encoder_path": pretrained_encoder or None,
            "loaded": bool(pretrained_encoder),
            **pretrained_stats,
        },

        # Metrics
        "metrics": {
            "accuracy": metrics["accuracy"],
            "pr_auc": metrics["pr_auc"],
            "class_f1": metrics["class_f1"],
            "brier": metrics["brier"],
            "ece": metrics["ece"],
        },

        # Training
        "training": {
            "epochs": results["epochs"],
            "early_stopping_at": results.get("early_stopped_at"),
            "best_val_loss": results["best_val_loss"],
        },
    }

    with open(run_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"✓ Run manifest written: {run_path / 'manifest.json'}")

    return manifest


def report_state_dict_load(model, checkpoint_path, freeze_encoder=False):
    """Load pretrained weights and report matched/missing/mismatched tensors."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model_state = model.state_dict()

    matched = []
    missing = []
    mismatched = []

    # Check what's in the checkpoint
    for name, param in state_dict.items():
        if name in model_state:
            if param.shape == model_state[name].shape:
                matched.append(name)
            else:
                mismatched.append((name, param.shape, model_state[name].shape))
        else:
            # In checkpoint but not in model
            pass

    # Check what's missing from checkpoint
    for name in model_state:
        if name not in state_dict:
            missing.append(name)

    # Load matched tensors
    model.load_state_dict(state_dict, strict=False)

    # Freeze encoder if requested
    if freeze_encoder:
        n_frozen = 0
        for name, param in model.named_parameters():
            if "encoder" in name or "ohlc" in name:
                param.requires_grad = False
                n_frozen += 1

    stats = {
        "n_matched_tensors": len(matched),
        "n_missing_tensors": len(missing),
        "n_mismatched_shapes": len(mismatched),
        "matched_tensor_names": matched[:5],  # First 5
        "missing_tensor_names": missing[:5],
        "mismatched_details": [
            {"name": n, "checkpoint_shape": s1, "model_shape": s2}
            for n, s1, s2 in mismatched[:3]
        ],
        "frozen_params": n_frozen if freeze_encoder else 0,
    }

    print(f"✓ Pretrained load report:")
    print(f"  Matched: {len(matched)} tensors")
    print(f"  Missing: {len(missing)} tensors (will be trained from scratch)")
    print(f"  Mismatched shapes: {len(mismatched)}")
    if freeze_encoder:
        print(f"  Frozen params: {n_frozen}")

    if len(mismatched) > 0:
        print(f"  ⚠ Mismatched shapes (first 3):")
        for n, s1, s2 in mismatched[:3]:
            print(f"    {n}: checkpoint {s1} vs model {s2}")

    return stats
```

### Guards & Validation

```python
# src/moola/cli.py - enforce rules

def train(...):
    # Guard 1: Forbid random splits
    if split == "random":
        raise ValueError(
            "Random splits are forbidden (data leak risk). "
            "Use a versioned split: splits/fwd_chain_v3.json, splits/stratified_v1.json, etc."
        )

    # Guard 2: If synthetic + low KS p-value, abort
    if augment and synthetic_version:
        synth_meta = load_json(f"data/synthetic_cache/{synthetic_version}/meta.json")
        ks_p = synth_meta.get("ks_pval", 0.0)
        if ks_p < 0.1:
            raise ValueError(
                f"Synthetic data KS p-value too low ({ks_p:.3f} < 0.1). "
                f"Data quality is poor; augmentation will mislead training. "
                f"Use a different synthetic version or augment=false."
            )

    # Guard 3: If pretrained specified but <80% tensors matched, abort
    if pretrained_encoder:
        stats = report_state_dict_load(model_instance, pretrained_encoder, freeze_encoder=True)
        total_model_tensors = len(model_instance.state_dict())
        match_ratio = stats["n_matched_tensors"] / total_model_tensors

        if match_ratio < 0.8:
            raise ValueError(
                f"Pretrained load: only {match_ratio:.1%} ({stats['n_matched_tensors']}/{total_model_tensors}) "
                f"tensors matched. This model may be incompatible with the encoder. "
                f"Mismatch details:\n{json.dumps(stats['mismatched_details'], indent=2)}"
            )
```

### CLI Status Report

```bash
# Every run prints:

$ moola train --model enhanced_simple_lstm \
    --dataset labeled_windows/v1 \
    --split splits/fwd_chain_v3.json \
    --pretrained-encoder artifacts/ts2vec/encoder_v1.pt

════════════════════════════════════════════════════════════════════════════════
  MOOLA TRAINING RUN
════════════════════════════════════════════════════════════════════════════════

Run ID:           2025-10-18T16:45:30_abc1234_enhanced_simple_lstm
Git SHA:          abc1234 (main branch)
Seed:             42

MODEL
  Name:           enhanced_simple_lstm
  Params:         16,946
  Bidirectional:  True
  Feature Fusion: True
  Pretrained:     artifacts/ts2vec/encoder_v1.pt (✓ loaded)

DATA & SPLIT
  Labeled:        labeled_windows/v1
  Split:          splits/fwd_chain_v3.json (forward-chaining, purge=20d)
  Counts:
    Train:        53 real samples + 0 synthetic (ratio 0.0)
    Val:          27 samples (✓ zero synthetic)
    Test:         25 samples (✓ zero synthetic)

PRETRAINED LOAD
  Encoder:        artifacts/ts2vec/encoder_v1.pt
  Tensors matched: 42/42 (100%)
  Frozen params:  ~8,000 (first 3 epochs)

AUGMENTATION
  Temporal:       disabled
  Mixup:          disabled
  Synthetic:      none

HYPERPARAMS
  Epochs:         60
  Batch size:     32
  Learning rate:  5e-4
  Early stopping:  patience=20

════════════════════════════════════════════════════════════════════════════════
[Epoch 1/60] Train loss=0.95 | Val loss=0.88 | Grad norm (encoder/head)=0.12/0.34
[Epoch 2/60] Train loss=0.82 | Val loss=0.79 | Grad norm (encoder/head)=0.10/0.31
...
[Epoch 45/60] Early stop at epoch 45, best val loss=0.34
════════════════════════════════════════════════════════════════════════════════

RESULTS
  Accuracy:       0.84
  PR-AUC:         0.91
  Class-wise F1:  [0.78, 0.85, 0.88]
  Brier:          0.18
  ECE:            0.08

Manifest:        artifacts/runs/2025-10-18T16:45:30_abc1234_enhanced_simple_lstm/manifest.json
Reliability PNG: artifacts/runs/.../reliability.png (ECE diagram)
Confusion PNG:   artifacts/runs/.../confusion.png
```

---

## PILLAR 3b: Metrics & Visualization

### Compute Metrics Pack (Always)

```python
# src/moola/metrics.py

def compute_metrics_pack(results, data):
    """Compute: Accuracy, PR-AUC, class-wise F1, Brier, ECE."""
    from sklearn.metrics import (
        accuracy_score, precision_recall_curve, auc,
        f1_score, brier_score_loss
    )

    y_true = data.y_test
    y_pred_proba = results["y_pred_proba_test"]  # Shape: (n, n_classes)
    y_pred = results["y_pred_test"]  # Shape: (n,)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # PR-AUC (macro)
    precision, recall, _ = precision_recall_curve(
        (y_true == 1).astype(int),
        y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0],
    )
    pr_auc = auc(recall, precision)

    # Class-wise F1
    class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Brier
    brier = brier_score_loss(y_true, y_pred_proba.max(axis=1))

    # ECE (Expected Calibration Error)
    ece = compute_ece(y_true, y_pred_proba)

    return {
        "accuracy": float(acc),
        "pr_auc": float(pr_auc),
        "class_f1": [float(f) for f in class_f1],
        "brier": float(brier),
        "ece": float(ece),
    }


def compute_ece(y_true, y_pred_proba, n_bins=10):
    """Expected Calibration Error."""
    import numpy as np

    y_pred = y_pred_proba.argmax(axis=1)
    confidence = y_pred_proba.max(axis=1)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0

    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            acc_in_bin = (y_pred[in_bin] == y_true[in_bin]).mean()
            avg_conf_in_bin = confidence[in_bin].mean()
            ece += np.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin

    return ece


def save_reliability_diagram(y_true, y_pred_proba, output_path):
    """Save calibration/reliability diagram."""
    import matplotlib.pyplot as plt
    import numpy as np

    y_pred = y_pred_proba.argmax(axis=1)
    confidence = y_pred_proba.max(axis=1)

    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
        if in_bin.sum() > 0:
            bin_acc = (y_pred[in_bin] == y_true[in_bin]).mean()
            bin_conf = confidence[in_bin].mean()
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_counts.append(in_bin.sum())

    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.3)

    # Reliability diagram
    ax.scatter(bin_confs, bin_accs, s=[c * 5 for c in bin_counts], alpha=0.6, label="Bins")
    ax.plot(bin_confs, bin_accs, "o-", alpha=0.6)

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Calibration Diagram")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"✓ Reliability diagram saved: {output_path}")
```

---

## EXECUTION PLAN (15-20 hours)

### Phase 0: Baseline & Planning (1-2 hours)

- [ ] Review current data locations (data/, raw/, experiments/, etc.)
- [ ] Identify 105-sample labeled set location
- [ ] Locate unlabeled OHLC corpus for TS2Vec
- [ ] Identify existing synthetic augmentation outputs
- [ ] List all active vs. experimental models
- [ ] Document how splits are currently chosen
- [ ] Check if manifests/logs exist (unlikely)

### Phase 1: Build Data Registry (4-5 hours)

- [ ] Create `data/` structure as specified
- [ ] Copy labeled windows → `data/labeled_windows/v1/`
  - [ ] X_ohlc.npy, X_feats20.npy, y.npy
  - [ ] Create meta.json with shapes, class counts
  - [ ] Create manifest.json with audit trail
- [ ] Create unlabeled → `data/unlabeled_windows/v1/`
  - [ ] X_ohlc.npy
  - [ ] meta.json
- [ ] Create split → `data/splits/fwd_chain_v3.json`
  - [ ] Document strategy (forward-chaining, purge=20d)
  - [ ] Define train/val/test indices
- [ ] Implement `DataRegistry` class (registry.py)
- [ ] Write unit tests for DataRegistry
  - [ ] Test split validation (no overlap, non-random)
  - [ ] Test synthetic contamination guards (val/test zero synthetic)
  - [ ] Test audit trail generation

### Phase 2: Clarify Models (3-4 hours)

- [ ] Document SimpleLSTM vs. Enhanced
  - [ ] Confirm directions (uni vs. bi)
  - [ ] Confirm inputs (OHLC only vs. dual-path)
  - [ ] Confirm params (~5k vs. ~17k)
- [ ] Add clear docstrings to each model
- [ ] Create ModelRegistry with clarity:
  - [ ] ACTIVE = 4 models (simple, enhanced, adapter, minirocket)
  - [ ] LEGACY = experimental models
- [ ] Move experimental models to `models_extras/`
  - [ ] TCN, Transformer variants, RWKV, CNN-Transformer
  - [ ] Create `models_extras/README.md` with re-enable instructions
- [ ] Add backward-compatibility imports for any code using old paths
- [ ] Verify registry loading works for each model

### Phase 3: Build Validation & Logging (4-5 hours)

- [ ] Implement `report_state_dict_load()` function
  - [ ] Test with mock checkpoint
  - [ ] Verify matched/missing/mismatched reporting
  - [ ] Test freeze_encoder logic
- [ ] Implement guards:
  - [ ] Guard: forbid random splits
  - [ ] Guard: abort if pretrained match <80%
  - [ ] Guard: abort if synthetic KS p < 0.1
- [ ] Implement run manifest generation
  - [ ] Create artifacts/runs/{ run_id}/ directory
  - [ ] Write manifest.json with full lineage
  - [ ] Include data audit trail, pretrained stats, metrics
- [ ] Implement metrics pack:
  - [ ] Accuracy, PR-AUC, class-wise F1, Brier, ECE
  - [ ] Save reliability diagram (calibration)
  - [ ] Save confusion matrix
- [ ] Update CLI to emit status report (pretty print)

### Phase 4: RunPod Optimization (2-3 hours)

- [ ] Ensure Device handling is clean (no CPU fallback flags needed)
  - [ ] Detect GPU: `torch.cuda.is_available()`
  - [ ] Raise error if RunPod but GPU unavailable (fail fast)
  - [ ] Default device selection in CLI
- [ ] Simplify model checkpoint saving for SCP
  - [ ] Save only: model state_dict + config (not full checkpoint)
  - [ ] Include normalizer stats (scaler) in checkpoint
- [ ] Add "no unnecessary flags" to training
  - [ ] If not RunPod, warn about certain flags being ignored
  - [ ] Default config should "just work" on RunPod

### Phase 5: Testing & Integration (2-3 hours)

- [ ] Write integration test for full pipeline:
  ```python
  def test_full_pipeline():
      # Load data
      registry = DataRegistry()
      data = registry.load_labeled()

      # Create model
      model = ModelRegistry.create("enhanced_simple_lstm")

      # Train
      trainer = Trainer(model, data, "./test_run")
      results = trainer.fit(epochs=2)

      # Verify manifest exists
      assert (Path("./test_run") / "manifest.json").exists()

      # Verify metrics computed
      manifest = json.load(open("./test_run/manifest.json"))
      assert "metrics" in manifest
      assert "accuracy" in manifest["metrics"]
  ```
- [ ] Test with pretrained encoder load
  - [ ] Mock a checkpoint
  - [ ] Test matching logic
  - [ ] Test abort on <80% match
- [ ] Test guards:
  - [ ] Forbid random split
  - [ ] Abort on poor synthetic KS
  - [ ] Abort on bad pretrained load
- [ ] End-to-end CLI test:
  ```bash
  moola train --model enhanced_simple_lstm \
    --dataset labeled_windows/v1 \
    --split splits/fwd_chain_v3.json \
    --pretrained-encoder artifacts/ts2vec/encoder_v1.pt

  # Assert:
  # - Run manifest written
  # - Status report printed
  # - Metrics computed
  # - No random split error
  ```

### Phase 6: Documentation (1-2 hours)

- [ ] Update README.md:
  - [ ] Data lineage section
  - [ ] Model roles (simple vs. enhanced vs. adapter vs. minirocket)
  - [ ] Example commands with data registry
- [ ] Create `docs/DATA_LINEAGE.md`
  - [ ] Folder structure
  - [ ] How to add new labeled/unlabeled sets
  - [ ] How to create new splits
  - [ ] Audit trail requirements
- [ ] Create `docs/MODEL_CLARITY.md`
  - [ ] What each model does
  - [ ] When to use each
  - [ ] How to enable pretrained loading
- [ ] Update CLAUDE.md with:
  - [ ] Data registry usage
  - [ ] Model selection rules
  - [ ] Manifest expectations
  - [ ] Guard enforcement

---

## SUCCESS CRITERIA

### Data Lineage
- [ ] Every run tracks: which data, which split, synthetic ratio, split purity
- [ ] Manifest.json written with audit trail
- [ ] Val/test contain zero synthetic samples (guard enforced)
- [ ] Split is non-random (forward-chaining, versioned)

### Model Clarity
- [ ] SimpleLSTM vs. Enhanced clearly documented
- [ ] ModelRegistry lists active models only
- [ ] Legacy models in models_extras/ with re-enable docs
- [ ] Pretrained loading validated and reported

### Validation & Logging
- [ ] Guard: forbid random splits
- [ ] Guard: abort if pretrained <80% match
- [ ] Guard: abort if synthetic KS p < 0.1
- [ ] Manifest includes: data audit trail, pretrained stats, metrics
- [ ] Status report printed every run

### RunPod Efficiency
- [ ] No unnecessary CPU fallback flags
- [ ] Device detection clear
- [ ] GPU utilization maximized
- [ ] Training setup "just works" on RunPod

### Testing
- [ ] Integration test passes (full pipeline)
- [ ] Guards tested
- [ ] Manifest generation tested
- [ ] No broken imports

---

## SUMMARY

This refactor **centers on correctness and clarity**, not LOC reduction.

**The three pillars:**

1. **Data Registry** (4-5h) → Immutable, versioned, validated data paths
   - No more confusion about which data is where
   - Audit trail proves split purity
   - Guards prevent contamination

2. **Model Roles** (3-4h) → Clear which model does what, active vs. legacy
   - SimpleLSTM = tiny baseline
   - Enhanced SimpleLSTM = main production model
   - Adapter LSTM = transfer learning
   - Legacy models moved out with re-enable docs

3. **Validation & Logging** (4-5h) → Prove every run is correct
   - Manifest.json proves data lineage
   - Report shows pretrained load success/failure
   - Guards prevent silent failures
   - Metrics pack always computed

**RunPod optimization** (2-3h) + **Testing & Docs** (3-4h) completes the refactor.

**Total: 15-20 hours**
**Result:** Experiments are reproducible, debuggable, and trustworthy.

---

## Next Step: Approval

Questions to clarify before starting:

1. **105-sample set location?** Where is the canonical labeled data currently stored?
2. **Unlabeled corpus?** Where is the raw OHLC for TS2Vec pretraining?
3. **Existing splits?** How are train/val/test currently chosen?
4. **Pretrained BiLSTM?** Does artifacts/ts2vec/encoder_v1.pt exist, or do we need to create it?
5. **Synthetic cache?** Where are augmented samples currently stored?
6. **Models to move?** Which models in `models/` are experimental vs. active?
7. **Ready to start Phase 1?** Or clarifications needed?

Let me know and I'll begin Phase 0 (baseline) immediately.
