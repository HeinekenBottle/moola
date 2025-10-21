"""Temporal data augmentation for financial time series.

Phase 2 augmentation strategy to achieve 3x effective dataset size while preserving
pattern integrity (correlation > 0.95 with originals).

Augmentation techniques:
1. Temporal Jittering: Add Gaussian noise to preserve local patterns
2. Magnitude Warping: Smooth multiplicative scaling via cubic spline

Reference: Um et al. (2017) "Data Augmentation of Wearable Sensor Data for
Parkinson's Disease Monitoring using Convolutional Neural Networks"
"""

import numpy as np
import torch
from scipy.interpolate import CubicSpline


def add_jitter(x: torch.Tensor, sigma: float = 0.03, prob: float = 0.8) -> torch.Tensor:
    """Add Gaussian noise to time series while preserving patterns.

    Args:
        x: Input tensor (batch, seq_len, features) or (seq_len, features)
        sigma: Standard deviation of Gaussian noise
        prob: Probability of applying jittering

    Returns:
        Jittered tensor with same shape as input

    Example:
        >>> x = torch.randn(32, 105, 11)  # batch of 11D features
        >>> x_aug = add_jitter(x, sigma=0.03, prob=0.8)
        >>> x_aug.shape
        torch.Size([32, 105, 11])
    """
    if np.random.rand() > prob:
        return x

    noise = torch.randn_like(x) * sigma
    return x + noise


def magnitude_warp(
    x: torch.Tensor, sigma: float = 0.2, n_knots: int = 4, prob: float = 0.5
) -> torch.Tensor:
    """Apply smooth magnitude warping using cubic spline.

    Creates a smooth multiplicative warp curve by sampling random knot points
    and interpolating with cubic spline. Preserves temporal structure while
    scaling magnitudes.

    Args:
        x: Input tensor (batch, seq_len, features) or (seq_len, features)
        sigma: Standard deviation for knot magnitudes (around mean=1.0)
        n_knots: Number of knots for cubic spline
        prob: Probability of applying warping

    Returns:
        Warped tensor with same shape as input

    Example:
        >>> x = torch.randn(32, 105, 11)
        >>> x_warped = magnitude_warp(x, sigma=0.2, n_knots=4, prob=0.5)
        >>> x_warped.shape
        torch.Size([32, 105, 11])
    """
    if np.random.rand() > prob:
        return x

    # Handle both (batch, seq, feat) and (seq, feat) inputs
    original_shape = x.shape
    if len(x.shape) == 2:
        x = x.unsqueeze(0)  # Add batch dimension

    batch_size, seq_len, n_features = x.shape

    # Sample knot positions uniformly across sequence
    knot_positions = np.linspace(0, seq_len - 1, n_knots)

    # Sample knot magnitudes from N(1.0, sigma)
    knot_magnitudes = np.random.normal(1.0, sigma, size=n_knots)

    # Create cubic spline interpolation
    cs = CubicSpline(knot_positions, knot_magnitudes)

    # Evaluate warp curve at all time steps
    time_steps = np.arange(seq_len)
    warp_curve = cs(time_steps)

    # Apply warp curve to all features (broadcast across batch and features)
    warp_tensor = torch.tensor(warp_curve, dtype=x.dtype, device=x.device).view(1, seq_len, 1)
    warped = x * warp_tensor

    # Remove batch dimension if it wasn't in original
    if len(original_shape) == 2:
        warped = warped.squeeze(0)

    return warped


def augment_temporal(
    x: torch.Tensor,
    jitter_sigma: float = 0.03,
    jitter_prob: float = 0.8,
    warp_sigma: float = 0.2,
    warp_knots: int = 4,
    warp_prob: float = 0.5,
) -> torch.Tensor:
    """Apply combined temporal augmentation pipeline.

    Sequentially applies jittering and magnitude warping to create augmented samples
    that preserve pattern integrity while increasing training diversity.

    Args:
        x: Input tensor (batch, seq_len, features) or (seq_len, features)
        jitter_sigma: Std dev for jittering
        jitter_prob: Probability of jittering
        warp_sigma: Std dev for warp knot magnitudes
        warp_knots: Number of spline knots
        warp_prob: Probability of warping

    Returns:
        Augmented tensor

    Example:
        >>> x = torch.randn(32, 105, 11)
        >>> x_aug = augment_temporal(x, jitter_prob=0.8, warp_prob=0.5)
        >>> # Expected: ~80% jittered, ~50% warped, ~40% both
    """
    # Apply augmentations sequentially
    x = add_jitter(x, sigma=jitter_sigma, prob=jitter_prob)
    x = magnitude_warp(x, sigma=warp_sigma, n_knots=warp_knots, prob=warp_prob)
    return x


def validate_augmentation(
    original: torch.Tensor, augmented: torch.Tensor, min_correlation: float = 0.95
) -> dict:
    """Validate that augmentation preserves pattern integrity.

    Computes correlation and other metrics to ensure augmented samples maintain
    high similarity with originals, critical for financial pattern recognition.

    Args:
        original: Original tensor (seq_len, features)
        augmented: Augmented tensor (seq_len, features)
        min_correlation: Minimum required correlation

    Returns:
        Dictionary with validation metrics:
            - correlation: Pearson correlation coefficient
            - passes: Boolean indicating if correlation >= min_correlation
            - mean_diff: Absolute difference in means
            - std_ratio: Ratio of standard deviations

    Example:
        >>> x = torch.randn(105, 11)
        >>> x_aug = augment_temporal(x)
        >>> metrics = validate_augmentation(x, x_aug, min_correlation=0.95)
        >>> assert metrics['passes'], f"Correlation too low: {metrics['correlation']}"
    """
    # Flatten to compute correlation
    orig_flat = original.flatten().cpu().numpy()
    aug_flat = augmented.flatten().cpu().numpy()

    correlation = np.corrcoef(orig_flat, aug_flat)[0, 1]

    return {
        "correlation": correlation,
        "passes": correlation >= min_correlation,
        "mean_diff": np.abs(orig_flat.mean() - aug_flat.mean()),
        "std_ratio": aug_flat.std() / orig_flat.std(),
    }
