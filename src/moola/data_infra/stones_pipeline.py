"""Stones-only data pipeline for Jade/Opal/Sapphire models.

Expects parquet files with columns:
- features: List[List[float]] - shape (T, 11) per sample
- label: int - classification label
- ptr_start: int - pointer start index (optional)
- ptr_end: int - pointer end index (optional)

Usage:
    >>> from moola.data_infra.stones_pipeline import load_parquet, make_dataloaders
    >>> 
    >>> # Load data
    >>> pack = load_parquet("data/processed/train_latest.parquet")
    >>> print(f"Loaded {pack['X'].shape[0]} samples")
    >>> 
    >>> # Create DataLoaders
    >>> train_dl, val_dl = make_dataloaders(pack, bs=29, shuffle=True)
    >>> 
    >>> # Train model
    >>> for xb, yb, ptr in train_dl:
    >>>     # xb: [batch, seq_len, 11]
    >>>     # yb: [batch]
    >>>     # ptr: [batch, 2] (start, end)
    >>>     pass
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset


def load_parquet(path: str | Path) -> dict[str, Any]:
    """Load parquet file and return dict with X, y, ptr_start, ptr_end.

    Args:
        path: Path to parquet file

    Returns:
        dict with keys:
            - X: np.ndarray of shape [N, T, 11] (features)
            - y: np.ndarray of shape [N] (labels)
            - ptr_start: np.ndarray of shape [N] (pointer start indices)
            - ptr_end: np.ndarray of shape [N] (pointer end indices)

    Raises:
        FileNotFoundError: If parquet file doesn't exist
        ValueError: If required columns are missing
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    logger.info(f"Loading parquet from {path}")
    df = pd.read_parquet(path)

    # Validate required columns
    required_cols = ["features", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Extract features (list of lists → numpy array)
    features_list = df["features"].tolist()
    X = np.stack([np.asarray(x, dtype=np.float32) for x in features_list], axis=0)

    # Extract labels
    y = np.asarray(df["label"].values, dtype=np.int64)

    # Extract pointers (optional)
    if "ptr_start" in df.columns and "ptr_end" in df.columns:
        ptr_start = np.asarray(df["ptr_start"].values, dtype=np.int64)
        ptr_end = np.asarray(df["ptr_end"].values, dtype=np.int64)
    else:
        logger.warning("Pointer columns not found, using zeros")
        ptr_start = np.zeros(len(df), dtype=np.int64)
        ptr_end = np.zeros(len(df), dtype=np.int64)

    logger.info(f"Loaded {X.shape[0]} samples with shape {X.shape}")
    logger.info(f"Label distribution: {np.bincount(y)}")

    return {
        "X": X,
        "y": y,
        "ptr_start": ptr_start,
        "ptr_end": ptr_end,
    }


def feature_stats(batch: dict[str, Any]) -> dict[str, Any]:
    """Compute feature statistics for logging.

    Args:
        batch: dict with 'X' key containing features [N, T, D]

    Returns:
        dict with mean and std statistics
    """
    X = batch["X"]  # (N, T, 11)
    mu = X.mean(axis=(0, 1))
    sd = X.std(axis=(0, 1)) + 1e-8

    return {
        "mean": mu.tolist(),
        "std": sd.tolist(),
        "min": X.min(axis=(0, 1)).tolist(),
        "max": X.max(axis=(0, 1)).tolist(),
    }


class StonesDS(Dataset):
    """PyTorch Dataset for Stones models.

    Args:
        pack: dict with X, y, ptr_start, ptr_end arrays
    """

    def __init__(self, pack: dict[str, Any]):
        self.pack = pack
        self.X = pack["X"]
        self.y = pack["y"]
        self.ptr_start = pack["ptr_start"]
        self.ptr_end = pack["ptr_end"]

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get single sample.

        Returns:
            Tuple of (features, label, pointers)
            - features: [T, 11] tensor
            - label: scalar tensor
            - pointers: [2] tensor with [start, end]
        """
        return (
            torch.from_numpy(self.X[i]),
            torch.tensor(self.y[i], dtype=torch.long),
            torch.tensor([self.ptr_start[i], self.ptr_end[i]], dtype=torch.long),
        )


def make_dataloaders(
    pack: dict[str, Any],
    bs: int = 29,
    shuffle: bool = True,
    val_split: float = 0.15,
    seed: int = 17,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders with temporal split.

    Args:
        pack: dict with X, y, ptr_start, ptr_end arrays
        bs: Batch size (default: 29, Stones requirement)
        shuffle: Whether to shuffle training data (default: True)
        val_split: Validation split ratio (default: 0.15)
        seed: Random seed for reproducibility (default: 17)

    Returns:
        Tuple of (train_dl, val_dl)
    """
    N = pack["X"].shape[0]
    cut = int((1 - val_split) * N)

    logger.info(f"Creating DataLoaders: {N} samples → {cut} train, {N-cut} val")
    logger.info(f"Batch size: {bs} (Stones requirement)")

    # Temporal split (no shuffling of indices)
    idx = np.arange(N)

    # Split into train/val
    tr_idx = idx[:cut]
    va_idx = idx[cut:]

    # Create train/val packs
    tr = {k: v[tr_idx] for k, v in pack.items()}
    va = {k: v[va_idx] for k, v in pack.items()}

    # Create DataLoaders
    train_dl = DataLoader(
        StonesDS(tr),
        batch_size=bs,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )

    val_dl = DataLoader(
        StonesDS(va),
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_dl, val_dl


def augmentation_meta() -> dict[str, Any]:
    """Return augmentation metadata stub for CLI logging compatibility.

    Returns:
        dict with augmentation metadata
    """
    return {
        "augments": [],
        "applied": False,
        "note": "Stones models use built-in dropout for regularization",
    }


def normalize_ohlc(X: np.ndarray) -> np.ndarray:
    """Apply price relevance scaling to OHLC columns.

    Normalizes OHLC (first 4 channels) to [0, 1] within each window
    to improve cross-period generalization.

    Args:
        X: Input array of shape (N, T, D) where D >= 4
           First 4 channels are OHLC

    Returns:
        Normalized array with OHLC scaled to [0, 1] per window
    """
    if X.shape[-1] < 4:
        logger.warning(f"Expected at least 4 features for OHLC, got {X.shape[-1]}")
        return X

    ohlc = X[..., :4]
    rng = ohlc.max(axis=1, keepdims=True) - ohlc.min(axis=1, keepdims=True) + 1e-8
    base = (ohlc - ohlc.min(axis=1, keepdims=True)) / rng  # 0..1 within-window

    Xn = X.copy()
    Xn[..., :4] = base

    logger.info("Applied price relevance scaling to OHLC columns")
    return Xn


def load_and_prepare(
    path: str | Path,
    *,
    normalize: bool = False,
    bs: int = 29,
    val_split: float = 0.15,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """Load parquet and prepare DataLoaders in one call.

    Args:
        path: Path to parquet file
        normalize: Apply price relevance scaling (default: False)
        bs: Batch size (default: 29)
        val_split: Validation split ratio (default: 0.15)
        shuffle: Shuffle training data (default: True)

    Returns:
        Tuple of (train_dl, val_dl, metadata)
    """
    # Load data
    pack = load_parquet(path)

    # Optional normalization
    if normalize:
        pack["X"] = normalize_ohlc(pack["X"])

    # Compute stats
    stats = feature_stats(pack)

    # Create DataLoaders
    train_dl, val_dl = make_dataloaders(pack, bs=bs, val_split=val_split, shuffle=shuffle)

    # Metadata for logging
    metadata = {
        "num_samples": pack["X"].shape[0],
        "num_features": pack["X"].shape[-1],
        "seq_len": pack["X"].shape[1],
        "num_classes": len(np.unique(pack["y"])),
        "feature_stats": stats,
        "augmentation": augmentation_meta(),
        "normalized": normalize,
    }

    return train_dl, val_dl, metadata
