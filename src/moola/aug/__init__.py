"""Data augmentation module for MOOLA.

Provides jitter and magnitude warping augmentations following Stones specifications.
"""

from .jitter import Jitter
from .magnitude_warp import MagnitudeWarp, OnTheFlyAugmentation

__all__ = ["Jitter", "MagnitudeWarp", "OnTheFlyAugmentation"]


def create_augmentation_pipeline(
    jitter_sigma: float = 0.03,
    warp_sigma: float = 0.2,
    multiplier: int = 3,
    jitter_prob: float = 0.8,
    warp_prob: float = 0.5,
) -> OnTheFlyAugmentation:
    """Create standardized augmentation pipeline.

    Args:
        jitter_sigma: Standard deviation for Gaussian jitter (default: 0.03)
        warp_sigma: Standard deviation for magnitude warping (default: 0.2)
        multiplier: Number of augmented versions to create (default: 3)
        jitter_prob: Probability of applying jitter (default: 0.8)
        warp_prob: Probability of applying warping (default: 0.5)

    Returns:
        Configured augmentation pipeline
    """
    return OnTheFlyAugmentation(
        jitter_sigma=jitter_sigma,
        warp_sigma=warp_sigma,
        jitter_prob=jitter_prob,
        warp_prob=warp_prob,
        multiplier=multiplier,
    )
