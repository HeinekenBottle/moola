#!/usr/bin/env python3
"""Compare masked LSTM pre-training results to baseline.

Analyzes OOF predictions from:
1. SimpleLSTM with pre-trained encoder (transfer learning)
2. SimpleLSTM baseline (no pre-training)

Reports overall accuracy, per-class metrics, and improvements.

Usage:
    python scripts/compare_masked_lstm_results.py
    python scripts/compare_masked_lstm_results.py --pretrained seed_1337_pretrained.npy --baseline seed_42_baseline.npy
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)


def load_predictions(path: Path) -> np.ndarray:
    """Load OOF predictions from .npy file.

    Args:
        path: Path to .npy file

    Returns:
        Predicted labels
    """
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path}")

    preds = np.load(path)
    return preds


def load_ground_truth(data_path: Path = Path("data/processed/train_pivot_134.parquet")) -> np.ndarray:
    """Load ground truth labels from training data.

    Args:
        data_path: Path to training data parquet

    Returns:
        True labels
    """
    df = pd.read_parquet(data_path)
    return df["label"].values


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with accuracy, balanced accuracy, and per-class metrics
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'confusion_matrix': cm,
        'per_class': report,
    }


def print_comparison(
    pretrained_metrics: dict,
    baseline_metrics: dict,
    pretrained_name: str = "Pre-trained",
    baseline_name: str = "Baseline",
) -> None:
    """Print side-by-side comparison of metrics.

    Args:
        pretrained_metrics: Metrics from pre-trained model
        baseline_metrics: Metrics from baseline model
        pretrained_name: Display name for pre-trained model
        baseline_name: Display name for baseline model
    """
    print("="*80)
    print("MASKED LSTM PRE-TRAINING RESULTS COMPARISON")
    print("="*80)
    print()

    # Overall metrics
    print("OVERALL METRICS")
    print("-"*80)
    print(f"{'Metric':<30} {pretrained_name:>15} {baseline_name:>15} {'Δ':>10}")
    print("-"*80)

    acc_pre = pretrained_metrics['accuracy']
    acc_base = baseline_metrics['accuracy']
    acc_delta = acc_pre - acc_base

    bal_acc_pre = pretrained_metrics['balanced_accuracy']
    bal_acc_base = baseline_metrics['balanced_accuracy']
    bal_acc_delta = bal_acc_pre - bal_acc_base

    print(f"{'Accuracy':<30} {acc_pre:>15.4f} {acc_base:>15.4f} {acc_delta:>+10.4f}")
    print(f"{'Balanced Accuracy':<30} {bal_acc_pre:>15.4f} {bal_acc_base:>15.4f} {bal_acc_delta:>+10.4f}")
    print()

    # Per-class metrics
    print("PER-CLASS METRICS")
    print("-"*80)

    # Get class labels (excluding avg metrics)
    classes = [k for k in pretrained_metrics['per_class'].keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]

    for cls in sorted(classes):
        print(f"\nClass {cls}:")
        print(f"  {'Metric':<28} {pretrained_name:>15} {baseline_name:>15} {'Δ':>10}")
        print(f"  {'-'*78}")

        for metric in ['precision', 'recall', 'f1-score']:
            pre_val = pretrained_metrics['per_class'][cls][metric]
            base_val = baseline_metrics['per_class'][cls][metric]
            delta = pre_val - base_val

            print(f"  {metric.capitalize():<28} {pre_val:>15.4f} {base_val:>15.4f} {delta:>+10.4f}")

        # Support (sample count)
        support = pretrained_metrics['per_class'][cls]['support']
        print(f"  {'Support':<28} {support:>15.0f}")

    print()

    # Confusion matrices
    print("CONFUSION MATRICES")
    print("-"*80)

    print(f"\n{pretrained_name}:")
    print(pretrained_metrics['confusion_matrix'])

    print(f"\n{baseline_name}:")
    print(baseline_metrics['confusion_matrix'])

    print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)

    if acc_delta > 0:
        improvement = "✅ IMPROVEMENT"
        pct_change = (acc_delta / acc_base) * 100
        print(f"{improvement}: +{acc_delta:.4f} accuracy ({pct_change:+.1f}%)")
    elif acc_delta < 0:
        print(f"⚠️  REGRESSION: {acc_delta:.4f} accuracy")
    else:
        print(f"➖ NO CHANGE")

    print()

    # Highlight class imbalance improvements
    print("Class-specific improvements:")
    for cls in sorted(classes):
        pre_recall = pretrained_metrics['per_class'][cls]['recall']
        base_recall = baseline_metrics['per_class'][cls]['recall']
        delta = pre_recall - base_recall

        if delta > 0.05:  # Significant improvement
            print(f"  ✅ Class {cls} recall: {base_recall:.4f} → {pre_recall:.4f} ({delta:+.4f})")
        elif delta < -0.05:  # Significant regression
            print(f"  ⚠️  Class {cls} recall: {base_recall:.4f} → {pre_recall:.4f} ({delta:+.4f})")

    print()
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare masked LSTM pre-training results to baseline"
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path("data/artifacts/oof/simple_lstm/v1/seed_1337_pretrained.npy"),
        help="Path to pre-trained model predictions",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("data/artifacts/oof/simple_lstm/v1/seed_42_baseline.npy"),
        help="Path to baseline model predictions",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/processed/train_pivot_134.parquet"),
        help="Path to ground truth labels",
    )

    args = parser.parse_args()

    # Load predictions
    print("Loading predictions...")
    y_pretrained = load_predictions(args.pretrained)
    y_baseline = load_predictions(args.baseline)
    y_true = load_ground_truth(args.ground_truth)

    print(f"  Pre-trained: {args.pretrained}")
    print(f"  Baseline: {args.baseline}")
    print(f"  Ground truth: {args.ground_truth}")
    print(f"  Samples: {len(y_true)}")
    print()

    # Compute metrics
    print("Computing metrics...")
    pretrained_metrics = compute_metrics(y_true, y_pretrained)
    baseline_metrics = compute_metrics(y_true, y_baseline)
    print()

    # Print comparison
    print_comparison(
        pretrained_metrics,
        baseline_metrics,
        pretrained_name="Pre-trained",
        baseline_name="Baseline",
    )


if __name__ == "__main__":
    main()
