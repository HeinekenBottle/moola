#!/usr/bin/env python3
"""Gate 3: Pretrain BiLSTM Encoder on Unlabeled Data.

Gates:
- Linear probe validation accuracy >= 55% (encoder quality gate)
- Save encoder to /workspace/moola/artifacts/pretrained/encoder_v1.pt

Exit codes:
- 0: Pretraining passed
- 1: Pretraining failed (linear probe < 55%)
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
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from moola.pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer


def log_result(message: str, status: str = "INFO"):
    """Log with timestamp and status."""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] [{status}] {message}")


def log_to_jsonl(results: dict, filepath: Path):
    """Append results to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(results) + "\n")


def linear_probe_validation(
    encoder: nn.Module, X_labeled: np.ndarray, y_labeled: np.ndarray, device: str
) -> float:
    """Validate encoder quality with linear probe.

    Args:
        encoder: Pretrained BiLSTM encoder
        X_labeled: Labeled data [N, 105, 4]
        y_labeled: Labels [N]
        device: Device for inference

    Returns:
        Validation accuracy from 3-fold CV
    """
    log_result("Running linear probe validation...")

    # Extract features using encoder
    encoder.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_labeled).to(device)
        encoded, _ = encoder(X_tensor)  # [N, 105, 256]
        # Use last timestep
        features = encoded[:, -1, :].cpu().numpy()  # [N, 256]

    # Train logistic regression with cross-validation
    clf = LogisticRegression(max_iter=1000, random_state=17)

    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(clf, features, y_labeled, cv=3, scoring="accuracy")
    mean_acc = scores.mean()

    log_result(f"Linear probe CV accuracy: {mean_acc:.3f} (±{scores.std():.3f})")

    return mean_acc


def main():
    """Pretrain BiLSTM encoder."""
    log_result("=" * 70)
    log_result("GATE 3: PRETRAIN BiLSTM ENCODER")
    log_result("=" * 70)

    # Paths
    unlabeled_path = Path("/workspace/moola/data/raw/unlabeled_windows.parquet")
    labeled_path = Path("/workspace/moola/data/processed/train_clean.parquet")
    split_path = Path("/workspace/moola/data/artifacts/splits/v1/fold_0.json")
    encoder_output = Path("/workspace/moola/artifacts/pretrained/encoder_v1.pt")
    results_path = Path("/workspace/moola/gated_workflow_results.jsonl")

    encoder_output.parent.mkdir(parents=True, exist_ok=True)

    # Load unlabeled data
    log_result("Loading unlabeled data...")
    df_unlabeled = pd.read_parquet(unlabeled_path)
    X_unlabeled = np.stack([np.stack(f) for f in df_unlabeled["features"]])
    log_result(f"Unlabeled samples: {len(X_unlabeled)}")

    # Load labeled data for validation
    log_result("Loading labeled data for validation...")
    df_labeled = pd.read_parquet(labeled_path)
    X_labeled = np.stack([np.stack(f) for f in df_labeled["features"]])
    y_labeled = df_labeled["label"].values

    # Load split for validation set
    with open(split_path, "r") as f:
        split_data = json.load(f)

    val_idx = np.array(split_data.get("val_indices", split_data.get("val_idx", [])))
    X_val = X_labeled[val_idx]
    y_val = y_labeled[val_idx]

    log_result(f"Validation samples: {len(X_val)}")

    # Initialize pretrainer
    log_result("Initializing BiLSTM Masked Autoencoder...")
    pretrainer = MaskedLSTMPretrainer(
        input_dim=4,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        mask_ratio=0.15,
        mask_strategy="patch",
        patch_size=7,
        learning_rate=1e-3,
        batch_size=512,
        device="cuda",
        seed=17,
    )

    # Pretrain
    log_result("Starting pretraining...")
    start_time = datetime.now()

    history = pretrainer.pretrain(
        X_unlabeled=X_unlabeled,
        n_epochs=50,
        val_split=0.1,
        patience=10,
        save_path=encoder_output,
        verbose=True,
    )

    pretrain_time = (datetime.now() - start_time).total_seconds()

    log_result(f"Pretraining complete in {pretrain_time:.1f}s")
    log_result(f"Final train loss: {history['train_loss'][-1]:.4f}")
    log_result(f"Final val loss: {history['val_loss'][-1]:.4f}")
    log_result(f"Best val loss: {min(history['val_loss']):.4f}")

    # GATE: Linear probe validation
    log_result("=" * 70)
    log_result("LINEAR PROBE VALIDATION")
    log_result("=" * 70)

    probe_acc = linear_probe_validation(
        encoder=pretrainer.model.encoder_lstm, X_labeled=X_val, y_labeled=y_val, device="cuda"
    )

    if probe_acc < 0.55:
        log_result("=" * 70)
        log_result(
            f"✗ GATE FAILED: Linear probe accuracy {probe_acc:.3f} < 55%", "ERROR"
        )
        log_result("Encoder quality insufficient for transfer learning.", "ERROR")
        log_result("Possible issues: insufficient unlabeled data, poor masking strategy.", "ERROR")
        log_result("=" * 70)

        # Still log results
        results = {
            "gate": "3_pretrain_bilstm",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "bilstm_masked_autoencoder",
            "config": {
                "hidden_dim": 128,
                "mask_strategy": "patch",
                "mask_ratio": 0.15,
                "epochs": 50,
            },
            "metrics": {
                "final_train_loss": float(history["train_loss"][-1]),
                "final_val_loss": float(history["val_loss"][-1]),
                "best_val_loss": float(min(history["val_loss"])),
                "linear_probe_acc": float(probe_acc),
            },
            "pretrain_time_sec": pretrain_time,
            "encoder_path": str(encoder_output),
            "status": "failed",
        }
        log_to_jsonl(results, results_path)

        sys.exit(1)

    # Record results
    results = {
        "gate": "3_pretrain_bilstm",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "bilstm_masked_autoencoder",
        "config": {
            "hidden_dim": 128,
            "mask_strategy": "patch",
            "mask_ratio": 0.15,
            "epochs": 50,
        },
        "metrics": {
            "final_train_loss": float(history["train_loss"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "best_val_loss": float(min(history["val_loss"])),
            "linear_probe_acc": float(probe_acc),
        },
        "pretrain_time_sec": pretrain_time,
        "encoder_path": str(encoder_output),
        "status": "passed",
    }

    log_to_jsonl(results, results_path)

    log_result("=" * 70)
    log_result(
        f"GATE 3: PASSED - Linear probe accuracy {probe_acc:.3f} >= 55%", "SUCCESS"
    )
    log_result(f"Encoder saved to: {encoder_output}")
    log_result("=" * 70)

    sys.exit(0)


if __name__ == "__main__":
    main()
