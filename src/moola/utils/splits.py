"""Deterministic K-fold split generation and persistence.

⚠️  DEPRECATION WARNING ⚠️
This module uses StratifiedKFold which creates look-ahead bias in time series!
For financial data, use temporal splits from moola.data.splits instead.

Legacy module maintained for compatibility with existing split files only.
DO NOT use make_splits() for new work - use forward-chaining splits instead.
"""

import json
import warnings
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

# Issue deprecation warning when module is imported
warnings.warn(
    "moola.utils.splits is DEPRECATED for time series data!\n"
    "StratifiedKFold creates look-ahead bias by shuffling temporal data.\n"
    "Use moola.data.splits with forward-chaining splits instead:\n"
    "  from moola.data.splits import load_split, assert_temporal\n"
    "  split = load_split('data/artifacts/splits/v1/fold_0.json')",
    DeprecationWarning,
    stacklevel=2,
)


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    k: int = 5,
    output_dir: Path = None,
) -> list[Tuple[np.ndarray, np.ndarray]]:
    """Generate and persist deterministic stratified K-fold splits.

    ⚠️  DEPRECATED: DO NOT USE FOR TIME SERIES DATA! ⚠️

    This function uses StratifiedKFold with shuffle=True, which creates look-ahead
    bias in financial time series by shuffling temporal data. Use forward-chaining
    splits from moola.data.splits instead.

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

    Deprecated:
        Use moola.data.splits.load_split() with forward-chaining splits instead
    """
    warnings.warn(
        "make_splits() is DEPRECATED for time series!\n"
        "Use forward-chaining splits to prevent look-ahead bias:\n"
        "  from moola.data.splits import load_split\n"
        "  split = load_split('data/artifacts/splits/v1/fold_0.json')",
        DeprecationWarning,
        stacklevel=2,
    )

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
    n_samples = len(y)

    # Check if splits already exist
    fold_0 = splits_dir / "fold_0.json"
    if fold_0.exists():
        # Load existing splits
        splits = load_splits(splits_dir, k=k)

        # Validate that splits match current dataset size
        max_index = max(max(train_idx.max(), val_idx.max()) for train_idx, val_idx in splits)

        if max_index >= n_samples:
            from loguru import logger

            logger.warning(
                f"Existing splits contain out-of-bounds indices (max_index={max_index}, "
                f"n_samples={n_samples}). Regenerating splits."
            )
            # Delete stale splits and regenerate
            for fold_idx in range(k):
                fold_file = splits_dir / f"fold_{fold_idx}.json"
                if fold_file.exists():
                    fold_file.unlink()
            return make_splits(X, y, seed=seed, k=k, output_dir=splits_dir)

        return splits
    else:
        # Create new splits
        return make_splits(X, y, seed=seed, k=k, output_dir=splits_dir)
