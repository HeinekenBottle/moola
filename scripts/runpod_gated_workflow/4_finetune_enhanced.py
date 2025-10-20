#!/usr/bin/env python3
"""Gate 4: Finetune EnhancedSimpleLSTM with Pretrained Encoder.

Advanced finetuning strategy:
- Freeze encoder for 3 epochs
- Progressive unfreezing with discriminative LRs
- L2-SP regularization toward pretrained weights
- Gradient clipping, EMA, SWA

Gates:
- Must improve over Gate 1 baseline smoke run
- Use full training (60 epochs, augmentation enabled)

Exit codes:
- 0: Finetuning passed
- 1: Finetuning failed (no improvement over baseline)
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add moola to path
sys.path.insert(0, "/workspace/moola/src")

import numpy as np
import pandas as pd
import torch
from moola.models.enhanced_simple_lstm import EnhancedSimpleLSTMModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


def log_result(message: str, status: str = "INFO"):
    """Log with timestamp and status."""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] [{status}] {message}")


def log_to_jsonl(results: dict, filepath: Path):
    """Append results to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(results) + "\n")


def calculate_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE).

    Args:
        y_true: True binary labels [N]
        y_proba: Predicted probabilities for positive class [N]
        n_bins: Number of bins for calibration curve (default: 10)

    Returns:
        Expected Calibration Error (ECE) score
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')

    # Calculate bin counts
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bin_edges[1:-1])
    bin_counts = np.bincount(bin_indices, minlength=n_bins)

    # Calculate ECE as weighted average of absolute calibration error per bin
    ece = 0.0
    n_samples = len(y_true)
    for i in range(n_bins):
        if bin_counts[i] > 0:
            # Weight by bin proportion
            weight = bin_counts[i] / n_samples
            # Absolute difference between confidence and accuracy in bin
            calibration_error = abs(prob_pred[i] - prob_true[i]) if i < len(prob_pred) else 0.0
            ece += weight * calibration_error

    return ece


def load_baseline_metrics(results_path: Path) -> dict:
    """Load baseline metrics from Gate 1."""
    if not results_path.exists():
        log_result("✗ Baseline results not found - run Gate 1 first", "ERROR")
        sys.exit(1)

    with open(results_path, "r") as f:
        for line in f:
            result = json.loads(line)
            if result.get("gate") == "1_smoke_enhanced":
                return result["metrics"]

    log_result("✗ Gate 1 baseline not found in results", "ERROR")
    sys.exit(1)


def main():
    """Finetune EnhancedSimpleLSTM with pretrained encoder."""
    log_result("=" * 70)
    log_result("GATE 4: FINETUNE EnhancedSimpleLSTM with Pretrained Encoder")
    log_result("=" * 70)

    # Paths
    data_path = Path("/workspace/moola/data/processed/train_clean.parquet")
    split_path = Path("/workspace/moola/data/artifacts/splits/v1/fold_0.json")
    encoder_path = Path("/workspace/moola/artifacts/pretrained/encoder_v1.pt")
    model_output = Path("/workspace/moola/artifacts/models/enhanced_finetuned_v1.pt")
    results_path = Path("/workspace/moola/gated_workflow_results.jsonl")

    model_output.parent.mkdir(parents=True, exist_ok=True)

    # Load baseline from Gate 1
    baseline_metrics = load_baseline_metrics(results_path)
    baseline_val_f1 = baseline_metrics["val_f1"]
    log_result(f"Baseline (Smoke) Val F1: {baseline_val_f1:.3f}")

    # Verify encoder exists
    if not encoder_path.exists():
        log_result(
            f"✗ GATE FAILED: Pretrained encoder not found at {encoder_path}", "ERROR"
        )
        log_result("Run Gate 3 first to pretrain encoder.", "ERROR")
        sys.exit(1)

    log_result(f"✓ Pretrained encoder found: {encoder_path}")

    # Load data
    log_result("Loading data...")
    df = pd.read_parquet(data_path)
    X = np.stack([np.stack(f) for f in df["features"]])
    y = df["label"].values

    # Load split
    with open(split_path, "r") as f:
        split_data = json.load(f)

    train_idx = np.array(split_data.get("train_indices", split_data.get("train_idx", [])))
    val_idx = np.array(split_data.get("val_indices", split_data.get("val_idx", [])))

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    log_result(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Initialize model with full training config
    # CRITICAL: num_layers=2 to match pretrained encoder (2-layer BiLSTM)
    log_result("Initializing EnhancedSimpleLSTM (LAYER-MATCHED for pretrained encoder)...")
    model = EnhancedSimpleLSTMModel(
        seed=17,
        hidden_size=128,
        num_layers=2,  # CHANGED: Match pretrained encoder (was 1, now 2)
        num_heads=2,
        dropout=0.1,
        n_epochs=60,  # Full training
        batch_size=512,
        learning_rate=5e-4,
        device="cuda",
        use_amp=True,
        early_stopping_patience=20,
        val_split=0.0,  # Use manual split
        use_temporal_aug=True,  # Enable augmentation
        mixup_alpha=0.4,
        cutmix_prob=0.5,
    )

    # Train with two-phase finetuning
    log_result("=" * 70)
    log_result("FINETUNING STRATEGY")
    log_result("=" * 70)
    log_result("Phase 1: Freeze encoder (3 epochs)")
    log_result("Phase 2: Progressive unfreeze with discriminative LRs")
    log_result("Techniques: L2-SP regularization, gradient clipping, EMA, SWA")
    log_result("=" * 70)

    start_time = datetime.now()

    # Fit with pretrained encoder
    # unfreeze_encoder_after=3 triggers two-phase training
    model.fit(
        X_train,
        y_train,
        pretrained_encoder_path=encoder_path,
        freeze_encoder=True,
        unfreeze_encoder_after=3,  # Unfreeze after 3 epochs
    )

    train_time = (datetime.now() - start_time).total_seconds()

    # Evaluate
    log_result("Evaluating finetuned model...")
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Get probability predictions for calibration metrics
    y_val_proba = model.predict_proba(X_val)[:, 1]  # Probability of positive class

    # Basic metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="weighted")

    # Advanced metrics: PR-AUC, Brier, ECE
    val_pr_auc = average_precision_score(y_val, y_val_proba)
    val_brier = brier_score_loss(y_val, y_val_proba)
    val_ece = calculate_ece(y_val, y_val_proba, n_bins=10)

    log_result(f"Train Acc: {train_acc:.3f}")
    log_result(f"Val Acc: {val_acc:.3f}")
    log_result(f"Val F1: {val_f1:.3f}")
    log_result(f"Val PR-AUC: {val_pr_auc:.3f} (↑ better)")
    log_result(f"Val Brier: {val_brier:.3f} (↓ better)")
    log_result(f"Val ECE: {val_ece:.3f} (↓ better)")

    # GATE: Must improve over baseline
    improvement = val_f1 - baseline_val_f1

    if improvement <= 0:
        log_result("=" * 70)
        log_result(
            f"✗ GATE FAILED: No improvement over baseline", "ERROR"
        )
        log_result(f"  Baseline F1: {baseline_val_f1:.3f}", "ERROR")
        log_result(f"  Finetuned F1: {val_f1:.3f}", "ERROR")
        log_result(f"  Delta: {improvement:+.3f}", "ERROR")
        log_result("Pretrained encoder may be poor quality or finetuning strategy inadequate.", "ERROR")
        log_result("=" * 70)

        # Still log results
        results = {
            "gate": "4_finetune_enhanced",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "enhanced_simple_lstm_finetuned",
            "config": {
                "epochs": 60,
                "pretrained": True,
                "freeze_phase": 3,
                "progressive_unfreeze": True,
                "augmentation": True,
            },
            "metrics": {
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "val_f1": float(val_f1),
                "val_pr_auc": float(val_pr_auc),
                "val_brier": float(val_brier),
                "val_ece": float(val_ece),
            },
            "comparison": {
                "baseline_f1": baseline_val_f1,
                "finetuned_f1": float(val_f1),
                "delta": float(improvement),
            },
            "train_time_sec": train_time,
            "status": "failed",
        }
        log_to_jsonl(results, results_path)

        sys.exit(1)

    # Save model
    log_result(f"Saving finetuned model to {model_output}...")
    model.save(model_output)

    # Record results
    results = {
        "gate": "4_finetune_enhanced",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "enhanced_simple_lstm_finetuned",
        "config": {
            "epochs": 60,
            "pretrained": True,
            "freeze_phase": 3,
            "progressive_unfreeze": True,
            "augmentation": True,
        },
        "metrics": {
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
            "val_pr_auc": float(val_pr_auc),
            "val_brier": float(val_brier),
            "val_ece": float(val_ece),
        },
        "comparison": {
            "baseline_f1": baseline_val_f1,
            "finetuned_f1": float(val_f1),
            "delta": float(improvement),
        },
        "train_time_sec": train_time,
        "model_path": str(model_output),
        "status": "passed",
    }

    log_to_jsonl(results, results_path)

    log_result("=" * 70)
    log_result(
        f"GATE 4: PASSED - Improvement {improvement:+.3f} over baseline", "SUCCESS"
    )
    log_result(f"Model saved to: {model_output}")
    log_result("=" * 70)

    sys.exit(0)


if __name__ == "__main__":
    main()
