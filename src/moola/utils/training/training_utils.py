"""Training utilities for model training pipelines.

Provides reusable components for:
- DataLoader creation with optimal configurations
- Mixed precision training setup
- Device-aware optimizations (CPU vs CUDA)
- Float32 precision enforcement
- Batch validation and conversion
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, TensorDataset


class TrainingSetup:
    """Handles common training setup tasks."""

    @staticmethod
    def create_dataloader(
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        device: torch.device,
        prefetch_factor: int = 2,
    ) -> DataLoader:
        """Create optimized DataLoader with device-aware settings.

        Args:
            X: Feature tensor
            y: Label tensor
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes (set to 0 for CPU)
            device: Target device (CPU or CUDA)
            prefetch_factor: Number of batches to prefetch per worker

        Returns:
            Configured DataLoader
        """
        dataset = TensorDataset(X, y)

        # Device-aware worker configuration
        is_cuda = device.type == "cuda"
        actual_workers = num_workers if is_cuda else 0

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=actual_workers,
            pin_memory=is_cuda,
            persistent_workers=actual_workers > 0,
            prefetch_factor=prefetch_factor if actual_workers > 0 else None,
        )

        return loader

    @staticmethod
    def setup_mixed_precision(
        use_amp: bool, device: torch.device
    ) -> Optional[torch.cuda.amp.GradScaler]:
        """Setup automatic mixed precision (FP16) training.

        Args:
            use_amp: Whether to enable AMP
            device: Target device

        Returns:
            GradScaler if AMP enabled and CUDA available, else None
        """
        if use_amp and device.type == "cuda":
            return torch.cuda.amp.GradScaler()
        return None

    @staticmethod
    def split_data(
        X: torch.Tensor,
        y: torch.Tensor,
        val_split: float,
        seed: int,
        stratify: bool = True,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, None, torch.Tensor, None],
    ]:
        """Split data into train/validation sets.

        Args:
            X: Feature tensor
            y: Label tensor
            val_split: Validation split ratio (0.0 = no validation)
            seed: Random seed for reproducibility
            stratify: Whether to preserve class distribution

        Returns:
            (X_train, X_val, y_train, y_val) if val_split > 0
            (X, None, y, None) if val_split == 0
        """
        if val_split <= 0:
            return X, None, y, None

        from sklearn.model_selection import train_test_split

        stratify_labels = y.numpy() if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=val_split,
            random_state=seed,
            stratify=stratify_labels,
        )

        return X_train, X_val, y_train, y_val


def enforce_float32_precision():
    """Enforce float32 precision throughout the pipeline."""
    torch.set_float32_matmul_precision('high')


def convert_batch_to_float32(batch: Union[Dict[str, Any], List[Any], torch.Tensor]) -> Union[Dict[str, Any], List[Any], torch.Tensor]:
    """Convert all tensors in batch to float32.
    
    Args:
        batch: Batch data (dict, list, or tensor)
        
    Returns:
        Batch with all tensors converted to float32
    """
    if isinstance(batch, dict):
        return {k: v.float() if torch.is_tensor(v) else v for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [v.float() if torch.is_tensor(v) else v for v in batch]
    elif torch.is_tensor(batch):
        return batch.float()
    else:
        return batch


def validate_batch_schema(batch: Dict[str, torch.Tensor]) -> None:
    """Validate batch schema and check for NaNs.
    
    Args:
        batch: Batch dictionary with tensors
        
    Raises:
        AssertionError: If schema validation fails or NaNs found
    """
    # Check required keys
    required_keys = {"X", "y_ptr", "y_cls"}
    missing_keys = required_keys - set(batch.keys())
    assert not missing_keys, f"Missing required keys in batch: {missing_keys}"
    
    # Check for NaNs
    for key, tensor in batch.items():
        if torch.is_tensor(tensor):
            assert not torch.isnan(tensor).any(), f"NaN values found in tensor '{key}'"
            assert tensor.dtype == torch.float32, f"Tensor '{key}' is not float32: {tensor.dtype}"


def initialize_model_biases(model: torch.nn.Module) -> None:
    """Initialize model biases for pointer-only warmup.
    
    Args:
        model: PyTorch model to initialize
    """
    if hasattr(model, 'log_sigma_ptr'):
        model.log_sigma_ptr.data.fill_(-0.30)
        print("Initialized log_sigma_ptr to -0.30")
    
    if hasattr(model, 'log_sigma_cls'):
        model.log_sigma_cls.data.fill_(0.00)
        print("Initialized log_sigma_cls to 0.00")


def setup_optimized_dataloader(dataset, batch_size: int = 64, shuffle: bool = True, 
                             num_workers: int = 4, pin_memory: bool = True, 
                             persistent_workers: bool = True, drop_last: bool = True) -> DataLoader:
    """Setup optimized DataLoader with recommended settings.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size (default: 64 for pretrain, 29 for supervised)
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        persistent_workers: Whether to keep workers alive
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=drop_last
    )
