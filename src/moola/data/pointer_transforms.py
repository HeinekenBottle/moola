"""Pointer representation transformations for dual-task learning.

Converts between start-end and center-length pointer representations.
Center-length encoding enables decoupled gradient flow for faster convergence.

Reference:
    - "Learning to Localize and Predict" - Center-length encoding shows 20-30%
      faster convergence compared to start-end encoding in multi-task regression.

Key Concepts:
    - Start-End: [start_idx, end_idx] in range [0, seq_len]
      * Coupled parameters: changing start affects end implicitly
      * Asymmetric gradients: start and end compete

    - Center-Length: [center_position, span_length] in range [0, 1]
      * Decoupled parameters: independent optimization
      * Symmetric gradients: center and length are orthogonal
      * Better for neural networks: normalized to [0, 1]

Usage:
    >>> from moola.data.pointer_transforms import start_end_to_center_length
    >>> # Training: Convert ground truth to center-length
    >>> center, length = start_end_to_center_length(
    ...     torch.tensor([10.0]), torch.tensor([50.0]), seq_len=104
    ... )
    >>> # Inference: Convert predictions back to start-end
    >>> start, end = center_length_to_start_end(center, length, seq_len=104)
"""

import numpy as np
import torch


def start_end_to_center_length(start: torch.Tensor, end: torch.Tensor, seq_len: int = 104) -> tuple:
    """Convert start-end pointer representation to center-length.

    Maps from absolute indices to normalized center-length representation:
        center = (start + end) / 2 / seq_len  # Center position in [0, 1]
        length = (end - start) / seq_len      # Span length in [0, 1]

    Args:
        start: Start indices [B] or [B, 1]
        end: End indices [B] or [B, 1]
        seq_len: Sequence length for normalization (default: 104 for 105-bar windows)

    Returns:
        (center, length): Both normalized to [0, 1]
            - center: Center position normalized by seq_len
            - length: Span length normalized by seq_len

    Example:
        >>> start = torch.tensor([10.0, 20.0])
        >>> end = torch.tensor([50.0, 80.0])
        >>> center, length = start_end_to_center_length(start, end, seq_len=104)
        >>> assert center.shape == (2,)
        >>> assert 0 <= center.min() and center.max() <= 1
        >>> assert 0 <= length.min() and length.max() <= 1
    """
    center = (start + end) / 2.0 / seq_len
    length = (end - start) / seq_len
    return center, length


def center_length_to_start_end(
    center: torch.Tensor, length: torch.Tensor, seq_len: int = 104
) -> tuple:
    """Convert center-length representation back to start-end indices.

    Maps from normalized center-length to absolute start-end indices:
        start = (center - length / 2) * seq_len
        end = (center + length / 2) * seq_len

    Clamps to valid range [0, seq_len] to handle boundary cases.

    Args:
        center: Center position in [0, 1]
        length: Span length in [0, 1]
        seq_len: Sequence length for denormalization (default: 104)

    Returns:
        (start, end): Indices in [0, seq_len], clamped to valid range

    Example:
        >>> center = torch.tensor([0.5, 0.7])
        >>> length = torch.tensor([0.3, 0.4])
        >>> start, end = center_length_to_start_end(center, length, seq_len=104)
        >>> assert start.shape == (2,)
        >>> assert (start >= 0).all() and (start <= 104).all()
        >>> assert (end >= 0).all() and (end <= 104).all()
    """
    start = (center - length / 2) * seq_len
    end = (center + length / 2) * seq_len

    # Clamp to valid range
    start = torch.clamp(start, 0, seq_len)
    end = torch.clamp(end, 0, seq_len)

    return start, end


def numpy_start_end_to_center_length(
    start: np.ndarray, end: np.ndarray, seq_len: int = 104
) -> tuple:
    """NumPy version of start_end_to_center_length for data preprocessing.

    Args:
        start: Start indices [N]
        end: End indices [N]
        seq_len: Sequence length for normalization (default: 104)

    Returns:
        (center, length): Both normalized to [0, 1]

    Example:
        >>> start = np.array([10.0, 20.0])
        >>> end = np.array([50.0, 80.0])
        >>> center, length = numpy_start_end_to_center_length(start, end, seq_len=104)
        >>> assert center.shape == (2,)
    """
    center = (start + end) / 2.0 / seq_len
    length = (end - start) / seq_len
    return center, length


def numpy_center_length_to_start_end(
    center: np.ndarray, length: np.ndarray, seq_len: int = 104
) -> tuple:
    """NumPy version of center_length_to_start_end for evaluation.

    Args:
        center: Center position in [0, 1]
        length: Span length in [0, 1]
        seq_len: Sequence length for denormalization (default: 104)

    Returns:
        (start, end): Indices in [0, seq_len], clamped to valid range

    Example:
        >>> center = np.array([0.5, 0.7])
        >>> length = np.array([0.3, 0.4])
        >>> start, end = numpy_center_length_to_start_end(center, length, seq_len=104)
        >>> assert start.shape == (2,)
    """
    start = (center - length / 2) * seq_len
    end = (center + length / 2) * seq_len

    # Clamp to valid range
    start = np.clip(start, 0, seq_len)
    end = np.clip(end, 0, seq_len)

    return start, end
