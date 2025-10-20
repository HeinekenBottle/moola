"""Data validation utilities for training pipelines.

Provides input validation, label mapping, and class balance analysis.
"""

import numpy as np
import torch
from loguru import logger


class DataValidator:
    """Validates and prepares data for model training."""

    @staticmethod
    def reshape_input(X: np.ndarray, expected_features: int = 4) -> np.ndarray:
        """Reshape input array to 3D format [N, T, F].

        Args:
            X: Input array of shape [N, D] or [N, T, F]
            expected_features: Expected feature dimension (default: 4 for OHLC)

        Returns:
            Reshaped array of shape [N, T, F]

        Raises:
            ValueError: If input shape is invalid
        """
        if X.ndim == 2:
            N, D = X.shape
            if D % expected_features == 0:
                T = D // expected_features
                X = X.reshape(N, T, expected_features)
                logger.debug(f"Reshaped input from [{N}, {D}] to [{N}, {T}, {expected_features}]")
            else:
                X = X.reshape(N, 1, D)
                logger.debug(f"Reshaped input from [{N}, {D}] to [{N}, 1, {D}]")
        elif X.ndim == 3:
            logger.debug(f"Input already 3D: {X.shape}")
        else:
            raise ValueError(f"Invalid input shape: {X.shape}. Expected 2D or 3D array.")

        return X

    @staticmethod
    def create_label_mapping(y: np.ndarray) -> tuple[dict, dict, int]:
        """Create label-to-index mapping from target array.

        Args:
            y: Target labels of shape [N]

        Returns:
            (label_to_idx, idx_to_label, n_classes)
        """
        unique_labels = np.unique(y)
        n_classes = len(unique_labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        logger.debug(f"Created label mapping with {n_classes} classes: {label_to_idx}")

        return label_to_idx, idx_to_label, n_classes

    @staticmethod
    def convert_labels_to_indices(y: np.ndarray, label_to_idx: dict) -> np.ndarray:
        """Convert string/int labels to continuous indices.

        Args:
            y: Original labels
            label_to_idx: Label-to-index mapping

        Returns:
            Indices as numpy array
        """
        y_indices = np.array([label_to_idx[label] for label in y])
        return y_indices

    @staticmethod
    def log_class_distribution(y: np.ndarray) -> dict[int, int]:
        """Log class distribution for imbalance analysis.

        Args:
            y: Label indices (integer array)

        Returns:
            Dictionary mapping class index to count
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique_classes.tolist(), class_counts.tolist()))

        logger.info(f"Class distribution: {class_dist}")

        return class_dist

    @staticmethod
    def prepare_training_data(
        X: np.ndarray,
        y: np.ndarray,
        expected_features: int = 4,
    ) -> tuple[np.ndarray, np.ndarray, dict, dict, int]:
        """Complete data preparation pipeline.

        Args:
            X: Feature array
            y: Label array
            expected_features: Expected feature dimension

        Returns:
            (X_reshaped, y_indices, label_to_idx, idx_to_label, n_classes)
        """
        # Reshape input
        X_reshaped = DataValidator.reshape_input(X, expected_features)

        # Create label mapping
        label_to_idx, idx_to_label, n_classes = DataValidator.create_label_mapping(y)

        # Convert labels to indices
        y_indices = DataValidator.convert_labels_to_indices(y, label_to_idx)

        # Log class distribution
        DataValidator.log_class_distribution(y_indices)

        return X_reshaped, y_indices, label_to_idx, idx_to_label, n_classes
