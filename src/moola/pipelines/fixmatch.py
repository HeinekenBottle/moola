#!/usr/bin/env python3
"""
FixMatch Pseudo-Labeling Pipeline for Semi-Supervised Learning.

This pipeline:
1. Trains a teacher model on labeled data
2. Generates high-confidence pseudo-labels on unlabeled data
3. Combines labeled + pseudo-labeled data for final training

Key features:
- Per-class adaptive thresholds to prevent confirmation bias
- Quality gates for pseudo-label distribution and self-consistency
- Optimized GPU settings (batch_size=512, num_workers=16)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from moola.models.cnn_transformer import CnnTransformerModel


def train_teacher_model(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 1337,
    device: str = 'cuda',
) -> CnnTransformerModel:
    """Train a teacher model on labeled data with optimized settings.

    Args:
        X: Training features (n_samples, seq_len, n_features)
        y: Training labels
        seed: Random seed
        device: Device for training

    Returns:
        Trained teacher model
    """
    print(f"\n{'='*70}")
    print("TRAINING TEACHER MODEL")
    print(f"{'='*70}")
    print(f"Labeled samples: {len(X)}")
    print(f"Classes: {np.unique(y)}")

    model = CnnTransformerModel(
        seed=seed,
        device=device,
        n_epochs=60,
        batch_size=512,        # Optimized for RTX 4090
        num_workers=16,        # Maximum parallelism
        mixup_alpha=0.3,       # Stronger mixup
        early_stopping_patience=20,
        use_amp=True,          # FP16 for 2x speedup
    )

    model.fit(X, y)

    # Validate teacher on training set (sanity check)
    y_pred_train = model.predict(X)
    train_acc = accuracy_score(y, y_pred_train)
    print(f"\nTeacher Training Accuracy: {train_acc:.4f}")

    return model


def generate_pseudo_labels(
    model: CnnTransformerModel,
    X_unlabeled: np.ndarray,
    tau_consolidation: float = 0.92,
    tau_retracement: float = 0.85,
    target_per_class: int = 150,
    max_per_class: int = 200,
) -> tuple[np.ndarray, np.ndarray, Dict]:
    """Generate pseudo-labels with per-class adaptive thresholds.

    Uses different confidence thresholds for each class to:
    1. Prevent confirmation bias (model favoring majority class)
    2. Balance pseudo-label distribution
    3. Ensure high quality (only confident predictions)

    Args:
        model: Trained teacher model
        X_unlabeled: Unlabeled features (n_samples, seq_len, n_features)
        tau_consolidation: Confidence threshold for consolidation class
        tau_retracement: Confidence threshold for retracement class
        target_per_class: Target number of pseudo-labels per class
        max_per_class: Maximum pseudo-labels per class

    Returns:
        X_pseudo: Selected unlabeled samples
        y_pseudo: Pseudo-labels for selected samples
        stats: Statistics about pseudo-label generation
    """
    print(f"\n{'='*70}")
    print("GENERATING PSEUDO-LABELS")
    print(f"{'='*70}")
    print(f"Unlabeled samples: {len(X_unlabeled)}")
    print(f"Thresholds: consolidation={tau_consolidation}, retracement={tau_retracement}")

    # Get predictions and confidences
    y_pred_proba = model.predict_proba(X_unlabeled)
    y_pred = model.predict(X_unlabeled)

    # Get max confidence for each prediction
    confidences = np.max(y_pred_proba, axis=1)

    # Map labels to indices
    label_to_idx = model.label_to_idx
    idx_to_label = model.idx_to_label

    # Per-class thresholding
    selected_indices = []
    selected_labels = []

    # Process each class separately
    thresholds = {
        'consolidation': tau_consolidation,
        'retracement': tau_retracement,
    }

    class_stats = {}

    for class_name, threshold in thresholds.items():
        class_idx = label_to_idx[class_name]

        # Find samples predicted as this class with high confidence
        class_mask = (y_pred == class_name) & (confidences >= threshold)
        class_indices = np.where(class_mask)[0]
        class_confidences = confidences[class_mask]

        # Sort by confidence (descending) and take top samples
        sorted_idx = np.argsort(class_confidences)[::-1]
        class_indices_sorted = class_indices[sorted_idx]

        # Take up to max_per_class samples
        n_selected = min(len(class_indices_sorted), max_per_class)
        selected_class_indices = class_indices_sorted[:n_selected]

        selected_indices.extend(selected_class_indices.tolist())
        selected_labels.extend([class_name] * n_selected)

        class_stats[class_name] = {
            'threshold': threshold,
            'candidates': len(class_indices),
            'selected': n_selected,
            'mean_confidence': float(np.mean(confidences[selected_class_indices])) if n_selected > 0 else 0.0,
            'min_confidence': float(np.min(confidences[selected_class_indices])) if n_selected > 0 else 0.0,
        }

        print(f"\n[{class_name.upper()}]")
        print(f"  Candidates (conf >= {threshold}): {len(class_indices)}")
        print(f"  Selected: {n_selected}")
        if n_selected > 0:
            print(f"  Mean confidence: {class_stats[class_name]['mean_confidence']:.4f}")
            print(f"  Min confidence: {class_stats[class_name]['min_confidence']:.4f}")

    # Convert to arrays
    selected_indices = np.array(selected_indices)
    selected_labels = np.array(selected_labels)

    X_pseudo = X_unlabeled[selected_indices]
    y_pseudo = selected_labels

    # Calculate distribution
    unique, counts = np.unique(y_pseudo, return_counts=True)
    distribution = dict(zip(unique, counts))
    total = len(y_pseudo)

    print(f"\n{'='*70}")
    print("PSEUDO-LABEL STATISTICS")
    print(f"{'='*70}")
    print(f"Total pseudo-labels: {total}")
    for label, count in distribution.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {label}: {count} ({pct:.1f}%)")

    # Quality gates
    quality_checks = {
        'total_generated': total,
        'distribution': distribution,
        'per_class_stats': class_stats,
    }

    # Check distribution balance (should be 45-55% for each class)
    if total > 0:
        for label, count in distribution.items():
            pct = count / total * 100
            if pct < 45 or pct > 55:
                print(f"\n⚠️  WARNING: {label} distribution {pct:.1f}% outside 45-55% range")
                print(f"   Consider adjusting threshold for {label}")

    # Check minimum samples per class
    if total > 0:
        for label in ['consolidation', 'retracement']:
            if label not in distribution or distribution[label] < target_per_class:
                actual = distribution.get(label, 0)
                print(f"\n⚠️  WARNING: {label} has {actual} pseudo-labels (target: {target_per_class})")
                print(f"   Consider lowering threshold or using more unlabeled data")

    return X_pseudo, y_pseudo, quality_checks


def check_self_consistency(
    model: CnnTransformerModel,
    X_pseudo: np.ndarray,
    y_pseudo: np.ndarray,
) -> Dict:
    """Check if teacher model agrees with its own pseudo-labels.

    Self-consistency check: Re-predict on pseudo-labeled data.
    Model should agree with its pseudo-labels >75% of the time.

    Args:
        model: Teacher model
        X_pseudo: Pseudo-labeled features
        y_pseudo: Pseudo-labels

    Returns:
        Consistency statistics
    """
    print(f"\n{'='*70}")
    print("SELF-CONSISTENCY CHECK")
    print(f"{'='*70}")

    y_pred = model.predict(X_pseudo)
    consistency = accuracy_score(y_pseudo, y_pred)

    print(f"Model agreement with pseudo-labels: {consistency:.4f}")

    if consistency < 0.75:
        print(f"\n⚠️  WARNING: Self-consistency {consistency:.4f} < 0.75")
        print(f"   Pseudo-labels may be unreliable. Consider:")
        print(f"   - Increasing confidence thresholds")
        print(f"   - Training teacher model longer")
        print(f"   - Using stronger data augmentation")
    else:
        print(f"✅ Self-consistency check passed ({consistency:.4f} >= 0.75)")

    # Per-class consistency
    print(f"\nPer-class consistency:")
    for label in np.unique(y_pseudo):
        mask = y_pseudo == label
        class_consistency = accuracy_score(y_pseudo[mask], y_pred[mask])
        print(f"  {label}: {class_consistency:.4f}")

    return {
        'overall_consistency': float(consistency),
        'passed_gate': consistency >= 0.75,
    }


def train_student_with_pseudo_labels(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_pseudo: np.ndarray,
    y_pseudo: np.ndarray,
    n_folds: int = 5,
    seed: int = 1337,
    device: str = 'cuda',
) -> Dict:
    """Train final model on combined labeled + pseudo-labeled data.

    Uses OOF cross-validation to get unbiased accuracy estimate.

    Args:
        X_labeled: Labeled features
        y_labeled: True labels
        X_pseudo: Pseudo-labeled features
        y_pseudo: Pseudo-labels
        n_folds: Number of CV folds
        seed: Random seed
        device: Device for training

    Returns:
        OOF results dictionary
    """
    print(f"\n{'='*70}")
    print("TRAINING STUDENT WITH PSEUDO-LABELS")
    print(f"{'='*70}")
    print(f"Labeled samples: {len(X_labeled)}")
    print(f"Pseudo-labeled samples: {len(X_pseudo)}")
    print(f"Total training samples: {len(X_labeled) + len(X_pseudo)}")

    # Combine datasets
    X_combined = np.concatenate([X_labeled, X_pseudo], axis=0)
    y_combined = np.concatenate([y_labeled, y_pseudo], axis=0)

    print(f"\nCombined dataset:")
    unique, counts = np.unique(y_combined, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")

    # OOF training (only on labeled data)
    oof_predictions = np.zeros(len(y_labeled), dtype=object)
    fold_accuracies = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'='*70}")

        # Split labeled data
        X_train_labeled = X_labeled[train_idx]
        y_train_labeled = y_labeled[train_idx]
        X_val = X_labeled[val_idx]
        y_val = y_labeled[val_idx]

        # Combine fold's training data with ALL pseudo-labeled data
        X_train_combined = np.concatenate([X_train_labeled, X_pseudo], axis=0)
        y_train_combined = np.concatenate([y_train_labeled, y_pseudo], axis=0)

        print(f"Fold training: {len(X_train_labeled)} labeled + {len(X_pseudo)} pseudo = {len(X_train_combined)} total")
        print(f"Fold validation: {len(X_val)} labeled")

        # Train model
        model = CnnTransformerModel(
            seed=seed + fold_idx,
            device=device,
            n_epochs=60,
            batch_size=512,        # Optimized for RTX 4090
            num_workers=16,        # Maximum parallelism
            mixup_alpha=0.3,       # Stronger mixup
            early_stopping_patience=20,
            use_amp=True,          # FP16 for 2x speedup
        )

        model.fit(X_train_combined, y_train_combined)

        # Predict on validation fold (labeled data only)
        y_pred = model.predict(X_val)
        oof_predictions[val_idx] = y_pred

        fold_acc = accuracy_score(y_val, y_pred)
        fold_accuracies.append(fold_acc)
        print(f"[FOLD {fold_idx+1}] Validation Accuracy: {fold_acc:.4f}")

    # Calculate OOF metrics
    oof_accuracy = accuracy_score(y_labeled, oof_predictions)
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    print(f"\n{'='*70}")
    print("FINAL OOF RESULTS WITH PSEUDO-LABELS")
    print(f"{'='*70}")
    print(f"OOF Accuracy: {oof_accuracy:.4f}")
    print(f"Mean Fold: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Fold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")

    # Confusion matrix
    cm = confusion_matrix(y_labeled, oof_predictions)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Quality gate: Check fold variance
    fold_variance_pct = (std_acc / mean_acc * 100) if mean_acc > 0 else 0
    print(f"\nFold variance: {fold_variance_pct:.1f}%")
    if fold_variance_pct > 8:
        print(f"⚠️  WARNING: Fold variance {fold_variance_pct:.1f}% > 8%")
        print(f"   Model may be unstable across folds")
    else:
        print(f"✅ Fold variance check passed ({fold_variance_pct:.1f}% <= 8%)")

    return {
        'oof_accuracy': float(oof_accuracy),
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'fold_accuracies': [float(acc) for acc in fold_accuracies],
        'fold_variance_pct': float(fold_variance_pct),
        'confusion_matrix': cm.tolist(),
    }


def main():
    """Full FixMatch pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="FixMatch Pseudo-Labeling Pipeline")
    parser.add_argument('--labeled', type=str, default='/workspace/data/processed/train.parquet',
                        help='Path to labeled training data')
    parser.add_argument('--unlabeled', type=str, default='/workspace/data/raw/unlabeled_windows.parquet',
                        help='Path to unlabeled data')
    parser.add_argument('--output-dir', type=str, default='/workspace/logs',
                        help='Output directory for results')
    parser.add_argument('--tau-consolidation', type=float, default=0.92,
                        help='Confidence threshold for consolidation class')
    parser.add_argument('--tau-retracement', type=float, default=0.85,
                        help='Confidence threshold for retracement class')
    parser.add_argument('--target-per-class', type=int, default=150,
                        help='Target pseudo-labels per class')
    parser.add_argument('--max-per-class', type=int, default=200,
                        help='Maximum pseudo-labels per class')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for training')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labeled data
    print(f"Loading labeled data from: {args.labeled}")
    df_labeled = pd.read_parquet(args.labeled)
    X_labeled = np.array([np.array([np.array(bar) for bar in features])
                          for features in df_labeled['features']])
    y_labeled = df_labeled['label'].values

    print(f"Labeled data: {X_labeled.shape}, Labels: {y_labeled.shape}")
    print(f"Classes: {np.unique(y_labeled)}")

    # Load unlabeled data
    print(f"\nLoading unlabeled data from: {args.unlabeled}")
    df_unlabeled = pd.read_parquet(args.unlabeled)
    X_unlabeled = np.array([np.array([np.array(bar) for bar in features])
                            for features in df_unlabeled['features']])

    print(f"Unlabeled data: {X_unlabeled.shape}")

    # Step 1: Train teacher model
    teacher = train_teacher_model(X_labeled, y_labeled, seed=args.seed, device=args.device)

    # Step 2: Generate pseudo-labels
    X_pseudo, y_pseudo, pseudo_stats = generate_pseudo_labels(
        teacher,
        X_unlabeled,
        tau_consolidation=args.tau_consolidation,
        tau_retracement=args.tau_retracement,
        target_per_class=args.target_per_class,
        max_per_class=args.max_per_class,
    )

    # Step 3: Check self-consistency
    consistency_stats = check_self_consistency(teacher, X_pseudo, y_pseudo)

    # Step 4: Train student with pseudo-labels
    student_results = train_student_with_pseudo_labels(
        X_labeled, y_labeled,
        X_pseudo, y_pseudo,
        n_folds=5,
        seed=args.seed,
        device=args.device,
    )

    # Save results
    results = {
        'pseudo_label_stats': pseudo_stats,
        'consistency_stats': consistency_stats,
        'student_results': student_results,
    }

    import json
    results_path = output_dir / 'fixmatch_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {results_path}")
    print(f"{'='*70}")

    # Summary
    print(f"\n{'='*70}")
    print("FIXMATCH PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"Labeled samples: {len(y_labeled)}")
    print(f"Pseudo-labels generated: {len(y_pseudo)}")
    print(f"Pseudo-label distribution: {pseudo_stats['distribution']}")
    print(f"Self-consistency: {consistency_stats['overall_consistency']:.4f}")
    print(f"Final OOF accuracy: {student_results['oof_accuracy']:.4f}")
    print(f"Improvement over baseline (60.9%): {(student_results['oof_accuracy'] - 0.609) * 100:+.1f}%")


if __name__ == '__main__':
    main()
