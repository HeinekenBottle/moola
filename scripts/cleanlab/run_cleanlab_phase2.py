#!/usr/bin/env python3
"""CleanLab noise detection for Phase 2 - identify mislabeled samples.

This script uses confident learning to detect potentially noisy labels in the training data.
It loads OOF (out-of-fold) predictions from Phase 1 models and uses CleanLab to:
1. Identify samples with low label quality scores
2. Rank samples by confidence of being mislabeled
3. Generate a clean training dataset by removing the noisiest samples

Expected impact: +3-5% accuracy by removing top 10-15% noisy samples.

Usage:
    python scripts/run_cleanlab_phase2.py --threshold 0.3 --output data/processed/train_clean_phase2.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores


def load_oof_predictions(oof_dir: Path, model_names: list[str]) -> dict:
    """Load OOF predictions from multiple models.

    Args:
        oof_dir: Directory containing OOF .npy files
        model_names: List of model names to load

    Returns:
        Dictionary mapping model names to OOF predictions [N, n_classes]
    """
    oof_preds = {}

    for model_name in model_names:
        oof_path = oof_dir / f"{model_name}_clean.npy"

        if not oof_path.exists():
            print(f"[WARNING] OOF file not found: {oof_path}")
            continue

        preds = np.load(oof_path)
        print(f"[LOADED] {model_name}: {preds.shape}")

        oof_preds[model_name] = preds

    return oof_preds


def ensemble_predictions(oof_preds: dict, method: str = "mean") -> np.ndarray:
    """Ensemble multiple OOF predictions.

    Args:
        oof_preds: Dictionary of OOF predictions [N, n_classes]
        method: Ensembling method ("mean" or "geometric_mean")

    Returns:
        Ensembled predictions [N, n_classes]
    """
    if len(oof_preds) == 0:
        raise ValueError("No OOF predictions provided")

    # Stack predictions: [n_models, N, n_classes]
    stacked = np.stack(list(oof_preds.values()), axis=0)

    if method == "mean":
        # Arithmetic mean (standard ensembling)
        ensemble = np.mean(stacked, axis=0)
    elif method == "geometric_mean":
        # Geometric mean (more conservative)
        ensemble = np.exp(np.mean(np.log(stacked + 1e-10), axis=0))
        # Renormalize
        ensemble = ensemble / ensemble.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    return ensemble


def detect_noisy_labels(
    oof_predictions: np.ndarray,
    true_labels: np.ndarray,
    threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect potentially mislabeled samples using CleanLab.

    Uses confident learning to identify samples with low label quality.

    Args:
        oof_predictions: OOF probability predictions [N, n_classes]
        true_labels: True labels [N] (integer class indices)
        threshold: Quality threshold for flagging (lower = stricter)
                  Default 0.3 means flag bottom 30% of samples

    Returns:
        Tuple of:
        - noisy_idx: Indices of potentially mislabeled samples
        - label_quality: Quality score for each sample [0-1]
    """
    print(f"\n[CLEANLAB] Detecting noisy labels...")
    print(f"[CLEANLAB] Input shape: {oof_predictions.shape}")
    print(f"[CLEANLAB] Unique labels: {np.unique(true_labels)}")

    # Calculate label quality scores
    label_quality = get_label_quality_scores(
        labels=true_labels,
        pred_probs=oof_predictions,
    )

    # Find label issues (ranked by severity)
    label_issues_idx = find_label_issues(
        labels=true_labels,
        pred_probs=oof_predictions,
        return_indices_ranked_by='self_confidence',  # Rank by how wrong they are
    )

    # Flag samples below quality threshold
    noisy_idx = np.where(label_quality < threshold)[0]

    print(f"[CLEANLAB] Label quality scores: min={label_quality.min():.3f}, "
          f"mean={label_quality.mean():.3f}, max={label_quality.max():.3f}")
    print(f"[CLEANLAB] Found {len(label_issues_idx)} potential label issues")
    print(f"[CLEANLAB] Flagged {len(noisy_idx)} samples below quality threshold {threshold}")

    return noisy_idx, label_quality


def generate_clean_dataset(
    train_df: pd.DataFrame,
    noisy_idx: np.ndarray,
    label_quality: np.ndarray,
    output_path: Path,
    remove_percentage: float = 0.10,
) -> pd.DataFrame:
    """Generate cleaned training dataset.

    Args:
        train_df: Original training DataFrame
        noisy_idx: Indices of potentially noisy samples
        label_quality: Quality scores [N]
        output_path: Where to save cleaned dataset
        remove_percentage: Percentage of lowest-quality samples to remove (default: 10%)

    Returns:
        Cleaned DataFrame
    """
    n_samples = len(train_df)
    n_to_remove = int(n_samples * remove_percentage)

    # Sort by quality and take bottom N
    sorted_indices = np.argsort(label_quality)
    remove_indices = sorted_indices[:n_to_remove]

    print(f"\n[CLEAN] Removing lowest {remove_percentage*100:.0f}% of samples ({n_to_remove}/{n_samples})")
    print(f"[CLEAN] Quality range of removed samples: "
          f"{label_quality[remove_indices].min():.3f} - {label_quality[remove_indices].max():.3f}")

    # Create clean dataset
    clean_mask = np.ones(n_samples, dtype=bool)
    clean_mask[remove_indices] = False

    train_clean = train_df[clean_mask].copy()

    # Add quality scores as metadata column
    train_clean['label_quality'] = label_quality[clean_mask]

    # Save cleaned dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_clean.to_parquet(output_path, index=False)

    print(f"[CLEAN] Saved cleaned dataset: {output_path}")
    print(f"[CLEAN] Samples: {len(train_df)} → {len(train_clean)} (-{n_to_remove})")

    # Class distribution comparison
    print(f"\n[CLEAN] Class distribution before/after:")
    original_dist = train_df['label'].value_counts().sort_index()
    clean_dist = train_clean['label'].value_counts().sort_index()

    for label in original_dist.index:
        orig_count = original_dist[label]
        clean_count = clean_dist.get(label, 0)
        removed = orig_count - clean_count
        print(f"  {label}: {orig_count} → {clean_count} (-{removed}, -{removed/orig_count*100:.1f}%)")

    return train_clean


def main():
    parser = argparse.ArgumentParser(description="CleanLab noise detection for Phase 2")
    parser.add_argument(
        "--oof-dir",
        type=Path,
        default=Path("data/oof"),
        help="Directory containing OOF predictions (.npy files)"
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/processed/train_clean.parquet"),
        help="Training data file (must match OOF generation)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/train_clean_phase2.parquet"),
        help="Output path for cleaned dataset"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Quality threshold for flagging noisy samples (default: 0.3)"
    )
    parser.add_argument(
        "--remove-percentage",
        type=float,
        default=0.10,
        help="Percentage of lowest-quality samples to remove (default: 0.10 = 10%%)"
    )
    parser.add_argument(
        "--ensemble-method",
        type=str,
        default="mean",
        choices=["mean", "geometric_mean"],
        help="Ensembling method for OOF predictions"
    )

    args = parser.parse_args()

    # Model names to load (Phase 1 models)
    model_names = [
        "logreg",
        "rf",
        "xgb",
        "simple_lstm",
        "cnn_transformer",
    ]

    print("=" * 80)
    print("CleanLab Noise Detection - Phase 2")
    print("=" * 80)

    # Load OOF predictions
    print(f"\n[LOAD] Loading OOF predictions from: {args.oof_dir}")
    oof_preds = load_oof_predictions(args.oof_dir, model_names)

    if len(oof_preds) == 0:
        print("[ERROR] No OOF predictions found!")
        print(f"[ERROR] Expected files in {args.oof_dir}:")
        for name in model_names:
            print(f"  - {name}_clean.npy")
        return 1

    # Ensemble predictions
    print(f"\n[ENSEMBLE] Combining {len(oof_preds)} models using {args.ensemble_method}")
    ensemble_pred_probs = ensemble_predictions(oof_preds, method=args.ensemble_method)

    # Load training data
    print(f"\n[LOAD] Loading training data: {args.train_file}")
    train_df = pd.read_parquet(args.train_file)

    print(f"[LOAD] Samples: {len(train_df)}")
    print(f"[LOAD] Classes: {train_df['label'].value_counts().to_dict()}")

    # Map string labels to indices
    label_mapping = {label: idx for idx, label in enumerate(sorted(train_df['label'].unique()))}
    true_labels = train_df['label'].map(label_mapping).values

    # Detect noisy labels
    noisy_idx, label_quality = detect_noisy_labels(
        oof_predictions=ensemble_pred_probs,
        true_labels=true_labels,
        threshold=args.threshold,
    )

    # Generate report of worst samples
    print(f"\n[REPORT] Top 10 lowest quality samples:")
    worst_indices = np.argsort(label_quality)[:10]
    for rank, idx in enumerate(worst_indices, 1):
        pred_class = np.argmax(ensemble_pred_probs[idx])
        pred_prob = ensemble_pred_probs[idx, pred_class]
        true_class = true_labels[idx]
        true_label = train_df.iloc[idx]['label']
        quality = label_quality[idx]

        pred_label = [k for k, v in label_mapping.items() if v == pred_class][0]

        print(f"  {rank:2d}. Sample {idx:3d} | Quality: {quality:.3f} | "
              f"True: {true_label} | Predicted: {pred_label} ({pred_prob:.2f})")

    # Generate clean dataset
    train_clean = generate_clean_dataset(
        train_df=train_df,
        noisy_idx=noisy_idx,
        label_quality=label_quality,
        output_path=args.output,
        remove_percentage=args.remove_percentage,
    )

    print(f"\n[SUCCESS] CleanLab analysis complete!")
    print(f"[SUCCESS] Cleaned dataset saved to: {args.output}")
    print(f"[SUCCESS] Next steps:")
    print(f"  1. Review the removed samples for false positives")
    print(f"  2. Retrain models on cleaned dataset")
    print(f"  3. Compare performance: clean vs original")

    return 0


if __name__ == "__main__":
    exit(main())
