"""Deterministic seed management for reproducible ML experiments.

This module ensures reproducibility across all random number generators:
- Python's random module
- NumPy
- PyTorch (CPU and CUDA)

Usage:
    from moola.utils.seeds import set_seed

    set_seed(1337)  # All RNGs now deterministic
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 1337) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed to use across all RNG sources

    Side Effects:
        - Sets Python random.seed()
        - Sets NumPy random seed
        - Sets PyTorch manual_seed() for CPU and CUDA
        - Enables deterministic CUDA operations
        - Disables CUDA benchmark mode

    Note:
        Deterministic CUDA ops may reduce performance but ensure
        reproducibility for debugging and scientific experiments.
    """
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str = "cpu") -> torch.device:
    """Get PyTorch device, with fallback to CPU if CUDA unavailable.

    Args:
        device: Requested device ("cpu" or "cuda")

    Returns:
        torch.device instance

    Raises:
        ValueError: If device string is invalid
    """
    if device not in ["cpu", "cuda"]:
        raise ValueError(f"Invalid device '{device}'. Must be 'cpu' or 'cuda'")

    if device == "cuda" and not torch.cuda.is_available():
        import warnings
        warnings.warn("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    return torch.device(device)
