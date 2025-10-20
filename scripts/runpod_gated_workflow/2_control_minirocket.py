#!/usr/bin/env python3
"""Gate 2: Control Test - MiniRocket Baseline.

Train MiniRocket on same split as control comparison.

Gates:
- If MiniRocket >= EnhancedSimpleLSTM, ABORT with warning
- MiniRocket should be weaker than deep learning with pretraining

Exit codes:
- 0: Control test passed (MiniRocket < Enhanced)
- 1: Control test failed (MiniRocket >= Enhanced, abort workflow)
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add moola to path
sys.path.insert(0, "/workspace/moola/src")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model import RidgeClassifierCV


def log_result(message: str, status: str = "INFO"):
    """Log with timestamp and status."""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] [{status}] {message}")


def log_to_jsonl(results: dict, filepath: Path):
    """Append results to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(results) + "\n")


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
    """Run MiniRocket control test."""
    log_result("=" * 70)
    log_result("GATE 2: CONTROL TEST - MiniRocket Baseline")
    log_result("=" * 70)

    # Paths
    data_path = Path("/workspace/moola/data/processed/train_clean.parquet")
    split_path = Path("/workspace/moola/data/artifacts/splits/v1/fold_0.json")
    results_path = Path("/workspace/moola/gated_workflow_results.jsonl")

    # Load baseline from Gate 1
    baseline_metrics = load_baseline_metrics(results_path)
    baseline_val_f1 = baseline_metrics["val_f1"]
    log_result(f"Baseline (Enhanced) Val F1: {baseline_val_f1:.3f}")

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
    log_result(f"Input shape: {X_train.shape}")  # [N, 105, 4]

    # Reshape for MiniRocket: [N, n_timepoints, n_features] -> [N, n_features, n_timepoints]
    # MiniRocket expects [N, n_channels, n_timepoints] format
    log_result("Reshaping data for MiniRocket (transpose time and features)...")
    X_train_mr = X_train.transpose(0, 2, 1)  # [N, 105, 4] -> [N, 4, 105]
    X_val_mr = X_val.transpose(0, 2, 1)      # [N, 105, 4] -> [N, 4, 105]
    log_result(f"MiniRocket input shape: {X_train_mr.shape}")  # [N, 4, 105]

    # Train MiniRocket
    log_result("Training MiniRocket...")
    start_time = datetime.now()

    # Initialize MiniRocket transformer
    minirocket = MiniRocket(random_state=17)
    X_train_transform = minirocket.fit_transform(X_train_mr)
    X_val_transform = minirocket.transform(X_val_mr)

    log_result(f"MiniRocket features: {X_train_transform.shape[1]}")

    # Train Ridge classifier
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=3)
    classifier.fit(X_train_transform, y_train)

    train_time = (datetime.now() - start_time).total_seconds()

    # Evaluate
    log_result("Evaluating MiniRocket...")
    y_train_pred = classifier.predict(X_train_transform)
    y_val_pred = classifier.predict(X_val_transform)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="weighted")

    log_result(f"Train Acc: {train_acc:.3f}")
    log_result(f"Val Acc: {val_acc:.3f}")
    log_result(f"Val F1: {val_f1:.3f}")

    # GATE: MiniRocket should not outperform Enhanced
    if val_f1 >= baseline_val_f1:
        log_result("=" * 70)
        log_result(
            f"✗ GATE FAILED: MiniRocket ({val_f1:.3f}) >= Enhanced ({baseline_val_f1:.3f})",
            "ERROR"
        )
        log_result(
            "Deep learning with pretraining should outperform classical methods.",
            "ERROR"
        )
        log_result(
            "Possible issues: bad pretrained encoder, insufficient training, or data quality.",
            "ERROR"
        )
        log_result("=" * 70)

        # Still log results
        results = {
            "gate": "2_control_minirocket",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "minirocket",
            "config": {"random_state": 17},
            "metrics": {
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "val_f1": float(val_f1),
            },
            "comparison": {
                "baseline_f1": baseline_val_f1,
                "minirocket_f1": float(val_f1),
                "delta": float(val_f1 - baseline_val_f1),
            },
            "train_time_sec": train_time,
            "status": "failed",
        }
        log_to_jsonl(results, results_path)

        sys.exit(1)

    # Record results
    results = {
        "gate": "2_control_minirocket",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "minirocket",
        "config": {"random_state": 17},
        "metrics": {
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
        },
        "comparison": {
            "baseline_f1": baseline_val_f1,
            "minirocket_f1": float(val_f1),
            "delta": float(val_f1 - baseline_val_f1),
        },
        "train_time_sec": train_time,
        "status": "passed",
    }

    log_to_jsonl(results, results_path)

    log_result("=" * 70)
    log_result(
        f"GATE 2: PASSED - MiniRocket ({val_f1:.3f}) < Enhanced ({baseline_val_f1:.3f})",
        "SUCCESS"
    )
    log_result("=" * 70)

    sys.exit(0)


if __name__ == "__main__":
    main()
