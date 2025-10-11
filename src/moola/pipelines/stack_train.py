"""Stacking meta-learner training pipeline.

Loads OOF predictions from base models (logreg, rf, xgb), concatenates them,
and trains a LogisticRegression meta-learner for ensemble stacking.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

from ..models import get_model
from ..utils.manifest import create_manifest, write_manifest
from ..utils.metrics import calculate_ece
from ..utils.splits import load_splits


def train_stack(
    y: np.ndarray,
    seed: int,
    k: int,
    oof_dir: Path,
    splits_dir: Path,
    output_path: Path,
    metrics_path: Path,
    manifest_path: Path = None,
) -> dict:
    """Train stacking meta-learner on OOF predictions.

    Args:
        y: Target labels of shape [N]
        seed: Random seed for reproducibility
        k: Number of CV folds
        oof_dir: Directory containing OOF predictions (artifacts/oof/)
        splits_dir: Directory containing split manifests
        output_path: Path to save trained meta-learner (artifacts/models/stack/stack.pkl)
        metrics_path: Path to save metrics (artifacts/models/stack/metrics.json)
        manifest_path: Path to save manifest (artifacts/manifest.json, optional)

    Returns:
        Dictionary containing metrics (accuracy, f1, ece, etc.)

    Side Effects:
        - Loads OOF predictions from {logreg,rf,xgb}/v1/seed_{seed}.npy
        - Concatenates to [N, 3*C] matrix
        - Trains meta-learner with stratified K-fold CV
        - Saves model to output_path
        - Saves metrics to metrics_path
    """
    logger.info(f"Stack training start | seed={seed} k={k}")

    # Define base models
    base_models = ["logreg", "rf", "xgb"]

    # Load OOF predictions for each base model
    oof_predictions = []
    for model_name in base_models:
        oof_path = oof_dir / model_name / "v1" / f"seed_{seed}.npy"
        if not oof_path.exists():
            raise FileNotFoundError(f"OOF predictions not found: {oof_path}")

        oof = np.load(oof_path)
        logger.info(f"Loaded OOF {model_name} | shape={oof.shape}")
        oof_predictions.append(oof)

    # Concatenate OOF predictions [N, C] + [N, C] + [N, C] -> [N, 3*C]
    X_stack = np.concatenate(oof_predictions, axis=1)
    logger.info(f"Concatenated OOF matrix | shape={X_stack.shape}")

    # Get number of samples and classes
    n_samples = len(y)
    n_classes = len(np.unique(y))
    assert X_stack.shape == (n_samples, 3 * n_classes), (
        f"Expected shape ({n_samples}, {3 * n_classes}), got {X_stack.shape}"
    )

    # Load splits for cross-validation
    splits = load_splits(splits_dir, k=k)
    logger.info(f"Loaded {len(splits)} splits from {splits_dir}")

    # Perform stratified K-fold cross-validation on meta-learner
    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"Fold {fold_idx + 1}/{k} | train={len(train_idx)} val={len(val_idx)}")

        X_train_fold, X_val_fold = X_stack[train_idx], X_stack[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Create and train meta-learner for this fold
        meta_model = get_model("stack", seed=seed)
        meta_model.fit(X_train_fold, y_train_fold)

        # Predict on validation fold
        y_pred_fold = meta_model.predict(X_val_fold)
        y_proba_fold = meta_model.predict_proba(X_val_fold)

        # Calculate fold metrics
        acc = accuracy_score(y_val_fold, y_pred_fold)
        f1 = f1_score(y_val_fold, y_pred_fold, average="macro", zero_division=0)

        fold_metrics.append({
            "fold": fold_idx,
            "accuracy": acc,
            "f1": f1,
        })

        all_y_true.extend(y_val_fold)
        all_y_pred.extend(y_pred_fold)
        all_y_proba.extend(y_proba_fold)

        logger.info(f"Fold {fold_idx + 1} | acc={acc:.3f} f1={f1:.3f}")

    # Aggregate metrics across all folds
    mean_accuracy = np.mean([m["accuracy"] for m in fold_metrics])
    mean_f1 = np.mean([m["f1"] for m in fold_metrics])

    # Calculate ECE (Expected Calibration Error)
    all_y_proba = np.array(all_y_proba)
    all_y_true = np.array(all_y_true)
    ece = calculate_ece(all_y_true, all_y_proba, n_bins=10)

    # Calculate log loss for calibration
    logloss = log_loss(all_y_true, all_y_proba)

    logger.info(f"Meta CV metrics | acc={mean_accuracy:.3f} f1={mean_f1:.3f} ece={ece:.3f} logloss={logloss:.3f}")

    # Train final meta-learner on all data
    final_meta_model = get_model("stack", seed=seed)
    final_meta_model.fit(X_stack, y)
    logger.info("Trained final meta-learner on full dataset")

    # Save final model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_meta_model.save(output_path)
    logger.info(f"Saved meta-learner to {output_path}")

    # Save metrics
    metrics = {
        "model": "stack",
        "seed": seed,
        "cv_folds": k,
        "accuracy": mean_accuracy,
        "f1": mean_f1,
        "ece": ece,
        "logloss": logloss,
        "fold_details": fold_metrics,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Create and write manifest if path provided
    if manifest_path is not None:
        artifacts_dir = output_path.parent.parent.parent  # Navigate from models/stack/ to artifacts/
        manifest = create_manifest(
            artifacts_dir=artifacts_dir,
            models=["logreg", "rf", "xgb", "stack"],
            additional_metadata={
                "seed": seed,
                "cv_folds": k,
                "stack_f1": mean_f1,
                "stack_ece": ece,
            },
        )
        write_manifest(manifest_path, manifest)
        logger.info(f"Wrote manifest to {manifest_path}")

    return metrics
