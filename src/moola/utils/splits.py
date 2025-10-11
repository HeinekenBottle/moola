"""Deterministic K-fold split generation and persistence.

Shared splits ensure consistency across all base models for proper OOF stacking.
"""

import json
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    k: int = 5,
    output_dir: Path = None,
) -> list[Tuple[np.ndarray, np.ndarray]]:
    """Generate and persist deterministic stratified K-fold splits.

    Args:
        X: Feature matrix of shape [N, D]
        y: Target labels of shape [N]
        seed: Random seed for reproducibility
        k: Number of folds
        output_dir: Directory to save split manifests (e.g., artifacts/splits/v1/)

    Returns:
        List of (train_idx, val_idx) tuples for each fold

    Side Effects:
        If output_dir is provided, saves fold_{i}.json for each fold containing:
        - train_idx: List of training indices
        - val_idx: List of validation indices
        - seed: Random seed used
        - fold: Fold number
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    splits = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        splits.append((train_idx, val_idx))

        # Persist split manifest if output directory provided
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            fold_file = output_dir / f"fold_{fold_idx}.json"
            manifest = {
                "fold": fold_idx,
                "seed": seed,
                "k": k,
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            }
            with open(fold_file, "w") as f:
                json.dump(manifest, f, indent=2)

    return splits


def load_splits(splits_dir: Path, k: int = 5) -> list[Tuple[np.ndarray, np.ndarray]]:
    """Load persisted K-fold splits from disk.

    Args:
        splits_dir: Directory containing fold_{i}.json files
        k: Number of folds to load

    Returns:
        List of (train_idx, val_idx) tuples for each fold

    Raises:
        FileNotFoundError: If split files are missing
        ValueError: If split files are corrupted or inconsistent
    """
    splits = []

    for fold_idx in range(k):
        fold_file = splits_dir / f"fold_{fold_idx}.json"
        if not fold_file.exists():
            raise FileNotFoundError(f"Split manifest not found: {fold_file}")

        with open(fold_file, "r") as f:
            manifest = json.load(f)

        train_idx = np.array(manifest["train_idx"])
        val_idx = np.array(manifest["val_idx"])
        splits.append((train_idx, val_idx))

    return splits


def get_or_create_splits(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    k: int,
    splits_dir: Path,
) -> list[Tuple[np.ndarray, np.ndarray]]:
    """Get existing splits or create new ones if they don't exist.

    Args:
        X: Feature matrix of shape [N, D]
        y: Target labels of shape [N]
        seed: Random seed for reproducibility
        k: Number of folds
        splits_dir: Directory containing split manifests

    Returns:
        List of (train_idx, val_idx) tuples for each fold
    """
    # Check if splits already exist
    fold_0 = splits_dir / "fold_0.json"
    if fold_0.exists():
        # Load existing splits
        return load_splits(splits_dir, k=k)
    else:
        # Create new splits
        return make_splits(X, y, seed=seed, k=k, output_dir=splits_dir)
