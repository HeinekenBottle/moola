"""Optimized data pipeline for small dataset regime.

Provides efficient data loading with proper feature engineering
and memory optimization for 174-sample training regime.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any
from loguru import logger


class OptimizedDataset(Dataset):
    """Memory-efficient dataset for small regime training.
    
    Features:
    - Pre-computed feature tensors to avoid object dtype overhead
    - Optional 11D RelativeTransform feature loading
    - Efficient memory mapping for large datasets
    - Proper data type handling (float32 instead of object)
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        expansion_start: Optional[np.ndarray] = None,
        expansion_end: Optional[np.ndarray] = None,
        augment: bool = False,
        augment_prob: float = 0.5,
    ):
        """Initialize optimized dataset.
        
        Args:
            features: Pre-computed feature array (n_samples, seq_len, n_features)
            labels: Label array (n_samples,)
            expansion_start: Optional expansion start indices
            expansion_end: Optional expansion end indices
            augment: Whether to apply on-the-fly augmentation
            augment_prob: Probability of applying augmentation
        """
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        
        if expansion_start is not None:
            self.expansion_start = torch.from_numpy(np.array(expansion_start, dtype=np.float32))
            self.expansion_end = torch.from_numpy(np.array(expansion_end, dtype=np.float32))
        else:
            self.expansion_start = None
            self.expansion_end = None
            
        self.augment = augment
        self.augment_prob = augment_prob
        
        logger.info(f"Dataset initialized: {len(self.features)} samples, "
                   f"feature shape: {self.features.shape}, "
                   f"memory: {self.features.element_size() * self.features.nelement() / 1024**2:.1f} MB")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "features": self.features[idx],
            "labels": self.labels[idx],
        }
        
        if self.expansion_start is not None:
            item["expansion_start"] = self.expansion_start[idx]
            item["expansion_end"] = self.expansion_end[idx]
            
        return item


def load_optimized_features(
    data_path: str,
    feature_type: str = "ohlc",  # "ohlc" or "relative_transform"
    seq_len: int = 105,
):
    """Load and optimize features from parquet data.
    
    Converts object dtype arrays to proper float32 arrays for efficiency.
    
    Args:
        data_path: Path to parquet file
        feature_type: Type of features to extract
        seq_len: Sequence length for time series
        
    Returns:
        Tuple of (features, labels, expansion_start, expansion_end)
    """
    logger.info(f"Loading optimized features from {data_path}")
    
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} samples with columns: {list(df.columns)}")
    
    # Convert object dtype features to proper numpy arrays
    n_samples = len(df)
    if feature_type == "ohlc":
        features = np.zeros((n_samples, seq_len, 4), dtype=np.float32)
        for i, feature_array in enumerate(df["features"]):
            # Convert from object array of (4,) arrays to (105, 4) float32
            features[i] = np.array([timestep.flatten() for timestep in feature_array], dtype=np.float32)
    elif feature_type == "relative_transform":
        # TODO: Implement 11D RelativeTransform loading
        raise NotImplementedError("11D RelativeTransform not yet implemented")
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    
    # Extract labels
    label_map = {"consolidation": 0, "retracement": 1}
    labels = np.array([label_map[label] for label in df["label"]], dtype=np.int64)
    
    # Extract expansion indices if available
    expansion_start = None
    expansion_end = None
    if "expansion_start" in df.columns:
        expansion_start = np.asarray(df["expansion_start"])
    if "expansion_end" in df.columns:
        expansion_end = np.asarray(df["expansion_end"])
    
    logger.info(f"Features shape: {features.shape}, dtype: {features.dtype}")
    logger.info(f"Labels shape: {labels.shape}, distribution: {np.bincount(labels)}")
    
    if expansion_start is not None:
        start_arr = np.asarray(expansion_start)
        end_arr = np.asarray(expansion_end)
        logger.info(f"Expansion indices loaded: start range [{np.min(start_arr):.1f}, {np.max(start_arr):.1f}], "
                   f"end range [{np.min(end_arr):.1f}, {np.max(end_arr):.1f}]")
        # Convert to numpy arrays for consistency
        expansion_start = start_arr
        expansion_end = end_arr
    
    return features, labels, expansion_start, expansion_end


def create_optimized_dataloaders(
    data_path: str,
    batch_size: int = 16,  # Smaller batch for small dataset
    val_split: float = 0.2,
    feature_type: str = "ohlc",
    augment_train: bool = True,
    num_workers: int = 0,  # Set to 0 for small dataset to avoid overhead
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create optimized data loaders for small dataset regime.
    
    Args:
        data_path: Path to training data
        batch_size: Batch size (smaller for small dataset)
        val_split: Validation split ratio
        feature_type: Type of features to use
        augment_train: Whether to augment training data
        num_workers: Number of worker processes
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load features
    features, labels, expansion_start, expansion_end = load_optimized_features(
        data_path, feature_type=feature_type
    )
    
    # Create temporal split (train before val)
    split_idx = int(len(features) * (1 - val_split))
    
    train_features = features[:split_idx]
    train_labels = labels[:split_idx]
    train_exp_start = expansion_start[:split_idx] if expansion_start is not None else None
    train_exp_end = expansion_end[:split_idx] if expansion_end is not None else None
    
    val_features = features[split_idx:]
    val_labels = labels[split_idx:]
    val_exp_start = expansion_start[split_idx:] if expansion_start is not None else None
    val_exp_end = expansion_end[split_idx:] if expansion_end is not None else None
    
    # Create datasets
    train_dataset = OptimizedDataset(
        train_features,
        train_labels,
        train_exp_start,
        train_exp_end,
        augment=augment_train,
        augment_prob=0.7,  # High augmentation for small dataset
    )
    
    val_dataset = OptimizedDataset(
        val_features,
        val_labels,
        val_exp_start,
        val_exp_end,
        augment=False,  # No augmentation for validation
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    
    logger.info(f"Created dataloaders: train={len(train_loader)} batches, "
               f"val={len(val_loader)} batches")
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def validate_small_dataset_regime(
    n_samples: int,
    n_params: int,
    max_ratio: float = 100.0,  # Maximum params-to-samples ratio
) -> bool:
    """Validate if model is appropriate for small dataset regime.
    
    Args:
        n_samples: Number of training samples
        n_params: Number of model parameters
        max_ratio: Maximum allowed params-to-samples ratio
        
    Returns:
        True if appropriate, False otherwise
    """
    ratio = n_params / n_samples
    is_appropriate = ratio <= max_ratio
    
    if is_appropriate:
        logger.info(f"✅ Model size appropriate: {n_params:,} params for {n_samples} samples "
                   f"(ratio: {ratio:.1f}:1)")
    else:
        logger.warning(f"⚠️ Model overparameterized: {n_params:,} params for {n_samples} samples "
                      f"(ratio: {ratio:.1f}:1, recommended < {max_ratio}:1)")
    
    return is_appropriate


if __name__ == "__main__":
    # Test the optimized pipeline
    data_path = "data/processed/labeled/train_latest.parquet"
    
    train_loader, val_loader = create_optimized_dataloaders(
        data_path,
        batch_size=16,
        feature_type="ohlc",
        augment_train=True,
    )
    
    # Test a batch
    for batch in train_loader:
        logger.info(f"Batch features shape: {batch['features'].shape}")
        logger.info(f"Batch labels shape: {batch['labels'].shape}")
        if "expansion_start" in batch:
            logger.info(f"Batch expansion_start shape: {batch['expansion_start'].shape}")
        break