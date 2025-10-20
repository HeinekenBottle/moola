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
        - Sets PYTHONHASHSEED environment variable

    Note:
        Deterministic CUDA ops may reduce performance but ensure
        reproducibility for debugging and scientific experiments.
    """
    import os

    # Python hash seed (for dictionary ordering)
    os.environ['PYTHONHASHSEED'] = str(seed)

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


def verify_gpu_available() -> dict:
    """Verify GPU availability and return diagnostic information.

    Returns:
        Dictionary with GPU status and diagnostic info
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "device_name": None,
        "memory_total": 0,
        "memory_allocated": 0,
        "memory_cached": 0,
        "cuda_version": None,
    }

    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
        info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info["memory_allocated"] = torch.cuda.memory_allocated(0) / 1024**3
        info["memory_cached"] = torch.cuda.memory_reserved(0) / 1024**3
        info["cuda_version"] = torch.version.cuda

    return info


def print_gpu_info() -> None:
    """Print GPU diagnostic information to console."""
    info = verify_gpu_available()

    print("\n" + "=" * 60)
    print("GPU DIAGNOSTIC INFORMATION")
    print("=" * 60)

    if info["cuda_available"]:
        print(f"✓ CUDA Available: YES")
        print(f"  Device Count: {info['device_count']}")
        print(f"  Device Name: {info['device_name']}")
        print(f"  CUDA Version: {info['cuda_version']}")
        print(f"  Total Memory: {info['memory_total']:.2f} GB")
        print(f"  Allocated: {info['memory_allocated']:.2f} GB")
        print(f"  Cached: {info['memory_cached']:.2f} GB")
        print(f"\n✓ GPU training will be ENABLED")
    else:
        print(f"✗ CUDA Available: NO")
        print(f"\n⚠ WARNING: Training will run on CPU only!")
        print(f"  This will be significantly slower than GPU training.")

    print("=" * 60 + "\n")


def log_environment() -> dict:
    """Log environment information for reproducibility.

    Returns:
        Dictionary with environment info
    """
    import logging
    import os
    import platform
    import subprocess

    logger = logging.getLogger(__name__)

    env_info = {
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'platform': platform.platform(),
        'python_hash_seed': os.environ.get('PYTHONHASHSEED', 'not set'),
    }

    # Get git SHA
    try:
        git_sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=os.path.dirname(__file__)
        ).strip()[:8]
        env_info['git_sha'] = git_sha
    except Exception:
        env_info['git_sha'] = 'unknown'

    logger.info("Environment Information:")
    for key, value in env_info.items():
        logger.info(f"  {key}: {value}")

    return env_info
