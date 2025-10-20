"""Training utilities for model training pipelines.

Provides reusable components for:
- DataLoader creation with optimal configurations
- Mixed precision training setup
- Device-aware optimizations (CPU vs CUDA)
"""

from typing import Optional, Tuple, Union

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
    def setup_mixed_precision(use_amp: bool, device: torch.device) -> Optional[torch.cuda.amp.GradScaler]:
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
        Tuple[torch.Tensor, None, torch.Tensor, None]
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
