"""Pretraining data pipeline for MAE on OHLC data only.

Specialized pipeline for 11-month OHLC pretraining with proper validation.
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from loguru import logger
import torch
from torch.utils.data import Dataset, DataLoader

from moola.utils.training.training_utils import enforce_float32_precision, validate_batch_schema


class PretrainDataset(Dataset):
    """Dataset for MAE pretraining on OHLC data only."""
    
    def __init__(self, ohlc_data: np.ndarray, mask_ratio: float = 0.4):
        """Initialize pretraining dataset.
        
        Args:
            ohlc_data: OHLC data of shape [N, window_size, 4]
            mask_ratio: Ratio of timesteps to mask
        """
        self.ohlc_data = torch.from_numpy(ohlc_data).float()
        self.mask_ratio = mask_ratio
        self.window_size = ohlc_data.shape[1]
        
        # Validate data
        assert self.ohlc_data.dtype == torch.float32, "OHLC data must be float32"
        assert not torch.isnan(self.ohlc_data).any(), "OHLC data contains NaN values"
        assert self.ohlc_data.shape[2] == 4, f"Expected 4 OHLC features, got {self.ohlc_data.shape[2]}"
        
        logger.info(f"PretrainDataset initialized: {len(self)} samples, window_size={self.window_size}")
    
    def __len__(self) -> int:
        return len(self.ohlc_data)
    
    def __getitem__(self, idx: int) -> dict:
        """Get pretraining sample with mask.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'X' (masked data) and 'mask' tensors
        """
        x = self.ohlc_data[idx]  # [window_size, 4]
        
        # Create random mask
        mask = torch.rand(self.window_size) < self.mask_ratio
        
        # Apply mask (set masked timesteps to zeros)
        x_masked = x.clone()
        x_masked[mask] = 0.0
        
        return {
            'X': x_masked,      # [window_size, 4] - masked input
            'target': x,        # [window_size, 4] - original target
            'mask': mask        # [window_size] - boolean mask
        }


class PretrainPipeline:
    """Pipeline for loading and preparing OHLC data for MAE pretraining."""
    
    def __init__(self, data_path: Path, months: int = 11):
        """Initialize pretraining pipeline.
        
        Args:
            data_path: Path to parquet file with OHLC data
            months: Number of months of data to use (default: 11)
        """
        self.data_path = data_path
        self.months = months
        
        # Enforce float32 precision
        enforce_float32_precision()
        
    def load_ohlc_data(self) -> np.ndarray:
        """Load OHLC data for specified number of months.
        
        Returns:
            OHLC data of shape [N, window_size, 4]
        """
        logger.info(f"Loading {self.months} months of OHLC data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Load parquet data
        df = pd.read_parquet(self.data_path)
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required OHLC columns: {missing_cols}")
        
        # Extract OHLC data - assuming data is already windowed
        # If data is not windowed, we need to create windows
        if 'window_id' in df.columns:
            # Data is already windowed
            ohlc_data = df[required_columns].values
            # Reshape to [N, window_size, 4] if needed
            if len(ohlc_data.shape) == 2:
                # Assume each row is a flattened window
                window_size = len(df) // df['window_id'].nunique()
                ohlc_data = ohlc_data.reshape(-1, window_size, 4)
        else:
            # Create sliding windows from time series
            window_size = 105  # Default window size
            ohlc_values = df[required_columns].values
            
            windows = []
            for i in range(len(ohlc_values) - window_size + 1):
                window = ohlc_values[i:i + window_size]
                windows.append(window)
            
            ohlc_data = np.array(windows)
        
        # Limit to specified months (approximate)
        samples_per_month = len(ohlc_data) // 12  # Assume roughly 12 months total
        max_samples = min(len(ohlc_data), self.months * samples_per_month)
        ohlc_data = ohlc_data[:max_samples]
        
        # Validate data
        assert len(ohlc_data) > 0, "No data loaded"
        assert ohlc_data.shape[2] == 4, f"Expected 4 OHLC features, got {ohlc_data.shape[2]}"
        assert not np.isnan(ohlc_data).any(), "OHLC data contains NaN values"
        
        logger.info(f"Loaded OHLC data: shape={ohlc_data.shape}, months≈{self.months}")
        return ohlc_data.astype(np.float32)
    
    def create_datasets(self, ohlc_data: np.ndarray, val_split: float = 0.1, 
                       mask_ratio: float = 0.4) -> Tuple[PretrainDataset, Optional[PretrainDataset]]:
        """Create training and validation datasets.
        
        Args:
            ohlc_data: OHLC data of shape [N, window_size, 4]
            val_split: Validation split ratio
            mask_ratio: Mask ratio for MAE
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Split data
        n_samples = len(ohlc_data)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val
        
        train_data = ohlc_data[:n_train]
        val_data = ohlc_data[n_train:] if n_val > 0 else None
        
        # Create datasets
        train_dataset = PretrainDataset(train_data, mask_ratio=mask_ratio)
        val_dataset = PretrainDataset(val_data, mask_ratio=mask_ratio) if val_data is not None else None
        
        logger.info(f"Created datasets: train={len(train_dataset)}, val={len(val_dataset) if val_dataset else 0}")
        return train_dataset, val_dataset
    
    def create_dataloaders(self, train_dataset: PretrainDataset, 
                          val_dataset: Optional[PretrainDataset],
                          batch_size: int = 64, num_workers: int = 4) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create optimized dataloaders.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            batch_size: Batch size
            num_workers: Number of workers
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        from moola.utils.training.training_utils import setup_optimized_dataloader
        
        train_loader = setup_optimized_dataloader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        
        val_loader = None
        if val_dataset:
            val_loader = setup_optimized_dataloader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers
            )
        
        logger.info(f"Created dataloaders: train_batches={len(train_loader)}, val_batches={len(val_loader) if val_loader else 0}")
        return train_loader, val_loader


def validate_pretrain_batch(batch: dict) -> None:
    """Validate pretraining batch schema.
    
    Args:
        batch: Batch dictionary from pretrain dataloader
        
    Raises:
        AssertionError: If validation fails
    """
    required_keys = {'X', 'target', 'mask'}
    missing_keys = required_keys - set(batch.keys())
    assert not missing_keys, f"Missing required keys in pretrain batch: {missing_keys}"
    
    # Check tensor shapes and types
    X = batch['X']
    target = batch['target']
    mask = batch['mask']
    
    assert X.dtype == torch.float32, f"X must be float32, got {X.dtype}"
    assert target.dtype == torch.float32, f"target must be float32, got {target.dtype}"
    assert mask.dtype == torch.bool, f"mask must be bool, got {mask.dtype}"
    
    # Check for NaNs
    assert not torch.isnan(X).any(), "X contains NaN values"
    assert not torch.isnan(target).any(), "target contains NaN values"
    
    # Check shapes consistency
    assert X.shape == target.shape, f"X and target shapes must match: {X.shape} vs {target.shape}"
    assert mask.shape[0] == X.shape[0], f"Mask length must match X timesteps: {mask.shape[0]} vs {X.shape[0]}"