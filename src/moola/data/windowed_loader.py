"""Windowed data loader for Jade pretraining.

Implements sliding window approach with stride=52 (~50% overlap) for K=105 windows.
Builds relativity features on-the-fly to ensure causality and proper masking.
"""

from typing import Dict, List, Optional, Tuple, Iterator, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pydantic import BaseModel, Field, validator
import yaml
from pathlib import Path

from ..features.relativity import build_relativity_features, RelativityConfig


class WindowedConfig(BaseModel):
    """Configuration for windowed data loader."""
    window_length: int = Field(105, description="Window length K")
    stride: int = Field(52, description="Window stride (~50% overlap)")
    warmup_bars: int = Field(20, description="Warmup bars to mask")
    mask_ratio: float = Field(0.15, description="Fraction of timesteps to mask")
    padding_bars: int = Field(20, description="Number of bars to pad at start of each window")
    feature_config: Optional[Dict[str, Any]] = Field(None, description="Relativity feature config")
    splits: Optional[Dict[str, str]] = Field(None, description="Date-based splits (train_end, val_end, test_end)")
    gates: Optional[Dict[str, Any]] = Field(None, description="Quality gates")

    @validator('window_length')
    def validate_window_length(cls, v):
        if v < 10 or v > 500:
            raise ValueError('window_length must be between 10 and 500')
        return v

    @validator('stride')
    def validate_stride(cls, v, values):
        if 'window_length' in values and v <= 0:
            raise ValueError('stride must be positive')
        return v


class WindowedDataset(Dataset):
    """Dataset for sliding windows with masked reconstruction.
    
    Each sample returns:
    - X: Feature tensor [K, D] with D=10 features
    - mask: Boolean mask [K] indicating which timesteps are masked for reconstruction
    - valid_mask: Boolean mask [K] indicating valid timesteps (False for warmup)
    """
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 config: WindowedConfig,
                 feature_cache: Optional[np.ndarray] = None):
        """Initialize windowed dataset.
        
        Args:
            df: OHLC DataFrame with datetime index
            config: Windowed configuration
            feature_cache: Pre-computed features to avoid recomputation
        """
        self.df = df[['open', 'high', 'low', 'close']].copy()
        self.config = config
        self.window_length = config.window_length
        self.stride = config.stride
        
        # Build features if not cached
        if feature_cache is None:
            print("Building relativity features...")
            if config.feature_config:
                relativity_cfg = RelativityConfig(**config.feature_config)
            else:
                relativity_cfg = RelativityConfig()
            self.X_full, self.valid_mask_full, _ = build_relativity_features(self.df, relativity_cfg.dict())
        else:
            self.X_full = feature_cache
            # Create valid mask (False for first warmup_bars)
            n_windows = len(self.X_full)
            self.valid_mask_full = np.ones((n_windows, self.window_length), dtype=bool)
            for i in range(min(config.warmup_bars, self.window_length)):
                self.valid_mask_full[:, i] = False
        
        # Calculate window indices with stride
        self.n_bars = len(self.df)
        self.window_indices = list(range(0, self.n_bars - self.window_length + 1, self.stride))
        
        print(f"Dataset: {len(self.window_indices)} windows from {self.n_bars} bars")
        print(f"Feature shape: {self.X_full.shape}")
        print(f"Valid ratio: {self.valid_mask_full.mean():.3f}")
    
    def __len__(self) -> int:
        return len(self.window_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a windowed sample.

        Returns:
            X: Features [K, D]
            mask: Reconstruction mask [K] (True for masked positions)
            valid_mask: Valid mask [K] (False for warmup and padding)
        """
        window_start = self.window_indices[idx]

        # Get features for this window
        X = self.X_full[window_start]  # [K, D]
        valid_mask = self.valid_mask_full[window_start]  # [K]

        # Apply padding if configured
        if self.config.padding_bars > 0:
            X, valid_mask = self._apply_padding(X, valid_mask, window_start)

        # Create reconstruction mask (15% random masking)
        mask = torch.zeros(len(X), dtype=torch.bool)

        # Only mask valid positions (not warmup or padding)
        valid_indices = torch.where(torch.from_numpy(valid_mask))[0]
        if len(valid_indices) > 0:
            n_mask = int(self.config.mask_ratio * len(valid_indices))
            if n_mask > 0:
                mask_indices = valid_indices[torch.randperm(len(valid_indices))[:n_mask]]
                mask[mask_indices] = True

        return (
            torch.from_numpy(X).float(),  # [K + padding, D]
            mask,                          # [K + padding]
            torch.from_numpy(valid_mask)   # [K + padding]
        )

    def _apply_padding(self, X: np.ndarray, valid_mask: np.ndarray, window_start: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply padding to the window by prepending bars from global data.

        Args:
            X: Window features [K, D]
            valid_mask: Window valid mask [K]
            window_start: Starting index of window in full data

        Returns:
            Padded X and valid_mask
        """
        padding_bars = self.config.padding_bars

        if window_start == 0:
            # No padding available, repeat first bar
            padding_X = np.tile(X[0:1], (padding_bars, 1))
            padding_valid = np.zeros(padding_bars, dtype=bool)
        else:
            # Use previous bars for padding
            padding_end = window_start
            padding_start = max(0, window_start - padding_bars)

            if padding_start < padding_end:
                padding_X = self.X_full[padding_start:padding_end]
                padding_valid = self.valid_mask_full[padding_start:padding_end]
            else:
                # Not enough bars, repeat first available
                padding_X = np.tile(self.X_full[0:1], (padding_bars, 1))
                padding_valid = np.zeros(padding_bars, dtype=bool)

            # Ensure exactly padding_bars
            if len(padding_X) < padding_bars:
                repeat_X = np.tile(padding_X[0:1], (padding_bars - len(padding_X), 1))
                repeat_valid = np.zeros(padding_bars - len(padding_X), dtype=bool)
                padding_X = np.concatenate([repeat_X, padding_X], axis=0)
                padding_valid = np.concatenate([repeat_valid, padding_valid], axis=0)
            elif len(padding_X) > padding_bars:
                padding_X = padding_X[-padding_bars:]
                padding_valid = padding_valid[-padding_bars:]

        # Concatenate padding + window
        X_padded = np.concatenate([padding_X, X], axis=0)
        valid_mask_padded = np.concatenate([padding_valid, valid_mask], axis=0)

        return X_padded, valid_mask_padded


def create_time_splits(df: pd.DataFrame,
                      train_end: str = "2024-12-31",
                      val_end: str = "2025-03-31",
                      test_end: str = "2025-06-30") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create time-based splits for pretraining.

    Args:
        df: OHLC DataFrame with datetime index
        train_end: Training period end date (inclusive)
        val_end: Validation period end date (inclusive)
        test_end: Test period end date (inclusive)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    """Create time-based splits for pretraining.
    
    Args:
        df: OHLC DataFrame with datetime index
        train_end: Training period end date (inclusive)
        val_end: Validation period end date (inclusive)  
        test_end: Test period end date (inclusive)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_df = df[:train_end].copy()
    val_df = df[train_end:val_end][1:].copy()  # Start after train_end
    test_df = df[val_end:test_end][1:].copy()   # Start after val_end
    
    print(f"Train: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} bars)")
    print(f"Val:   {val_df.index.min()} to {val_df.index.max()} ({len(val_df)} bars)")
    print(f"Test:  {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} bars)")
    
    return train_df, val_df, test_df


def create_dataloaders(df: pd.DataFrame,
                      config: WindowedConfig,
                      batch_size: int = 256,
                      num_workers: int = 4,
                      pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders.

    Args:
        df: OHLC DataFrame with datetime index
        config: Windowed configuration
        batch_size: Batch size for training
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create time splits using config if available, otherwise use defaults
    if config.splits:
        train_end = config.splits.get("train_end", "2024-12-31")
        val_end = config.splits.get("val_end", "2025-03-31")
        test_end = config.splits.get("test_end", "2025-06-30")
        train_df, val_df, test_df = create_time_splits(df, train_end, val_end, test_end)
    else:
        train_df, val_df, test_df = create_time_splits(df)
    
    # Build features once for efficiency
    print("Building features for all splits...")
    if config.feature_config:
        relativity_cfg = RelativityConfig(**config.feature_config)
    else:
        relativity_cfg = RelativityConfig()
    
    # Cache features for each split
    train_X, train_valid_mask, _ = build_relativity_features(train_df, relativity_cfg.dict())
    val_X, val_valid_mask, _ = build_relativity_features(val_df, relativity_cfg.dict()) 
    test_X, test_valid_mask, _ = build_relativity_features(test_df, relativity_cfg.dict())
    
    # Create datasets
    train_dataset = WindowedDataset(train_df, config, train_X)
    val_dataset = WindowedDataset(val_df, config, val_X)
    test_dataset = WindowedDataset(test_df, config, test_X)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def save_split_manifest(train_df: pd.DataFrame, 
                       val_df: pd.DataFrame, 
                       test_df: pd.DataFrame,
                       config: WindowedConfig,
                       save_path: str = "artifacts/jade_pretrain/splits.json") -> Dict:
    """Save split manifest with metadata.
    
    Args:
        train_df, val_df, test_df: Split DataFrames
        config: Windowed configuration
        save_path: Path to save manifest
        
    Returns:
        Manifest dictionary
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        "splits": {
            "train": {
                "start_date": str(train_df.index.min()),
                "end_date": str(train_df.index.max()),
                "n_bars": len(train_df)
            },
            "val": {
                "start_date": str(val_df.index.min()),
                "end_date": str(val_df.index.max()),
                "n_bars": len(val_df)
            },
            "test": {
                "start_date": str(test_df.index.min()),
                "end_date": str(test_df.index.max()),
                "n_bars": len(test_df)
            }
        },
        "window_config": config.dict(),
        "total_bars": len(train_df) + len(val_df) + len(test_df),
        "created_at": str(pd.Timestamp.now())
    }
    
    import json
    with open(save_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Saved split manifest to {save_path}")
    return manifest


# CLI integration
def main():
    """CLI entry point for windowed data preparation.
    
    Usage: python -m moola.data.windowed_loader --config configs/windowed.yaml --data data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet
    """
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Create windowed dataset for Jade pretraining")
    parser.add_argument("--config", required=True, help="Path to windowed config")
    parser.add_argument("--data", required=True, help="Path to OHLC parquet file")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--output-dir", default="artifacts/jade_pretrain", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        # Load data
        print(f"Loading data from {args.data}")
        df = pd.read_parquet(args.data)
        print(f"Data shape: {df.shape}")
        
        # Load config
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = WindowedConfig(**config_dict)
        
        # Create splits
        train_df, val_df, test_df = create_time_splits(df)
        
        # Save manifest
        save_split_manifest(train_df, val_df, test_df, config, 
                          f"{args.output_dir}/splits.json")
        
        # Create dataloaders (test)
        train_loader, val_loader, test_loader = create_dataloaders(
            df, config, args.batch_size, args.num_workers
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test one batch
        for X, mask, valid_mask in train_loader:
            print(f"Batch shapes: X={X.shape}, mask={mask.shape}, valid={valid_mask.shape}")
            print(f"Mask ratio: {mask.float().mean():.3f}")
            print(f"Valid ratio: {valid_mask.float().mean():.3f}")
            break
            
        print("Windowed dataset creation successful!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()