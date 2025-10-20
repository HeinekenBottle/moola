"""Temporal augmentations for time series contrastive learning.

Implements augmentation techniques from TS-TCC (Time-Series representation learning via
Temporal and Contextual Contrasting) for OHLC financial data.

Reference:
    - https://arxiv.org/abs/2106.14112 (TS-TCC)
"""

import numpy as np
import torch
from typing import Tuple


def jitter(x: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
    """Add Gaussian noise to time series.

    Args:
        x: Input tensor [batch, seq_len, features]
        sigma: Standard deviation of Gaussian noise (default: 0.03 for 3% of scale)

    Returns:
        Jittered tensor [batch, seq_len, features]
    """
    noise = torch.randn_like(x) * sigma
    return x + noise


def scaling(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Scale time series by random factor.

    Args:
        x: Input tensor [batch, seq_len, features]
        sigma: Standard deviation for scaling factor (default: 0.1)

    Returns:
        Scaled tensor [batch, seq_len, features]
    """
    # Sample scaling factors from N(1, sigma^2)
    # Shape: [batch, 1, 1] to scale each sample independently
    batch_size = x.shape[0]
    scale_factors = torch.normal(
        mean=1.0,
        std=sigma,
        size=(batch_size, 1, 1),
        device=x.device
    )

    return x * scale_factors


def permutation(x: torch.Tensor, max_segments: int = 5, seg_mode: str = "equal") -> torch.Tensor:
    """Randomly permute segments of time series.

    Args:
        x: Input tensor [batch, seq_len, features]
        max_segments: Maximum number of segments to split into
        seg_mode: Segmentation mode ("equal" or "random")

    Returns:
        Permuted tensor [batch, seq_len, features]
    """
    batch_size, seq_len, n_features = x.shape

    # Random number of segments per sample
    n_segments = torch.randint(2, max_segments + 1, (batch_size,))

    x_permuted = x.clone()

    for i in range(batch_size):
        n_seg = n_segments[i].item()

        if seg_mode == "equal":
            # Equal-sized segments
            seg_len = seq_len // n_seg
            indices = torch.arange(n_seg)
            perm = torch.randperm(n_seg)

            for j in range(n_seg):
                start = j * seg_len
                end = (j + 1) * seg_len if j < n_seg - 1 else seq_len

                perm_start = perm[j] * seg_len
                perm_end = (perm[j] + 1) * seg_len if perm[j] < n_seg - 1 else seq_len

                x_permuted[i, start:end] = x[i, perm_start:perm_end]
        else:
            # Random-sized segments (not implemented for now)
            pass

    return x_permuted


def time_warp(x: torch.Tensor, sigma: float = 0.2, knot: int = 4) -> torch.Tensor:
    """Apply smooth time warping to time series.

    Randomly warps time dimension using smooth spline interpolation.

    Args:
        x: Input tensor [batch, seq_len, features]
        sigma: Standard deviation for warping magnitude (default: 0.2)
        knot: Number of warping knots (control points)

    Returns:
        Time-warped tensor [batch, seq_len, features]
    """
    batch_size, seq_len, n_features = x.shape
    device = x.device

    # Create original time indices
    orig_steps = torch.arange(seq_len, dtype=torch.float32, device=device)

    # Create random warping at knot points
    random_warps = torch.normal(
        mean=1.0,
        std=sigma,
        size=(batch_size, knot + 2),
        device=device
    )

    # Linear interpolation for each sample
    x_warped = torch.zeros_like(x)

    for i in range(batch_size):
        # Create knot points at equal intervals
        knot_positions = torch.linspace(0, seq_len - 1, knot + 2, device=device)

        # Apply warping factors cumulatively
        warped_knots = knot_positions.clone()
        for k in range(1, knot + 2):
            warped_knots[k] = warped_knots[k-1] + (knot_positions[k] - knot_positions[k-1]) * random_warps[i, k]

        # Normalize to original time range
        warped_knots = warped_knots * (seq_len - 1) / warped_knots[-1]

        # Interpolate warped time indices for all timesteps
        warped_time_np = np.interp(
            orig_steps.cpu().numpy(),
            knot_positions.cpu().numpy(),
            warped_knots.cpu().numpy()
        )
        warped_time = torch.from_numpy(np.asarray(warped_time_np, dtype=np.float32)).to(device)

        # Now interpolate features from original to warped time
        for j in range(n_features):
            feature_interp_np = np.interp(
                warped_time.cpu().numpy(),
                orig_steps.cpu().numpy(),
                x[i, :, j].cpu().numpy()
            )
            x_warped[i, :, j] = torch.from_numpy(np.asarray(feature_interp_np, dtype=np.float32)).to(device)

    return x_warped


def rotation(x: torch.Tensor) -> torch.Tensor:
    """Rotate feature dimensions.

    Applies random rotation matrix to feature space while preserving time structure.
    Useful for OHLC data to maintain relative relationships.

    Args:
        x: Input tensor [batch, seq_len, features]

    Returns:
        Rotated tensor [batch, seq_len, features]
    """
    batch_size, seq_len, n_features = x.shape
    device = x.device

    # Generate random rotation matrices for each sample
    x_rotated = torch.zeros_like(x)

    for i in range(batch_size):
        # Generate random orthogonal matrix via QR decomposition
        random_matrix = torch.randn((n_features, n_features), device=device)
        rotation_matrix, _ = torch.linalg.qr(random_matrix)

        # Apply rotation to each timestep
        for t in range(seq_len):
            x_rotated[i, t] = x[i, t] @ rotation_matrix

    return x_rotated


class TemporalAugmentation:
    """Temporal augmentation pipeline for time series contrastive learning.

    Creates two augmented views of the same time series for contrastive learning.
    Augmentations are applied with certain probabilities to create diverse views.

    Example:
        >>> aug = TemporalAugmentation(
        ...     jitter_prob=0.8,
        ...     scaling_prob=0.5,
        ...     permutation_prob=0.3,
        ...     time_warp_prob=0.5
        ... )
        >>> x = torch.randn(32, 105, 4)  # [batch, seq_len, features]
        >>> x_aug1, x_aug2 = aug(x)  # Two different augmented views
    """

    def __init__(
        self,
        jitter_prob: float = 0.8,
        jitter_sigma: float = 0.03,
        scaling_prob: float = 0.5,
        scaling_sigma: float = 0.1,
        permutation_prob: float = 0.0,  # Disabled by default (breaks temporal structure)
        time_warp_prob: float = 0.5,
        time_warp_sigma: float = 0.2,
        rotation_prob: float = 0.0,  # Disabled by default (OHLC order matters)
    ):
        """Initialize temporal augmentation pipeline.

        Args:
            jitter_prob: Probability of applying jitter
            jitter_sigma: Standard deviation for jitter noise
            scaling_prob: Probability of applying scaling
            scaling_sigma: Standard deviation for scaling factor
            permutation_prob: Probability of applying permutation
            time_warp_prob: Probability of applying time warping
            time_warp_sigma: Standard deviation for time warping
            rotation_prob: Probability of applying rotation
        """
        self.jitter_prob = jitter_prob
        self.jitter_sigma = jitter_sigma
        self.scaling_prob = scaling_prob
        self.scaling_sigma = scaling_sigma
        self.permutation_prob = permutation_prob
        self.time_warp_prob = time_warp_prob
        self.time_warp_sigma = time_warp_sigma
        self.rotation_prob = rotation_prob

    def apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to input.

        Args:
            x: Input tensor [batch, seq_len, features]

        Returns:
            Augmented tensor [batch, seq_len, features]
        """
        x_aug = x.clone()

        # Apply augmentations with their probabilities
        if torch.rand(1).item() < self.jitter_prob:
            x_aug = jitter(x_aug, sigma=self.jitter_sigma)

        if torch.rand(1).item() < self.scaling_prob:
            x_aug = scaling(x_aug, sigma=self.scaling_sigma)

        if torch.rand(1).item() < self.permutation_prob:
            x_aug = permutation(x_aug)

        if torch.rand(1).item() < self.time_warp_prob:
            x_aug = time_warp(x_aug, sigma=self.time_warp_sigma)

        if torch.rand(1).item() < self.rotation_prob:
            x_aug = rotation(x_aug)

        return x_aug

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate two augmented views of input.

        Args:
            x: Input tensor [batch, seq_len, features]

        Returns:
            Tuple of (augmented_view1, augmented_view2)
        """
        # Generate two independent augmented views
        x_aug1 = self.apply_augmentation(x)
        x_aug2 = self.apply_augmentation(x)

        return x_aug1, x_aug2
