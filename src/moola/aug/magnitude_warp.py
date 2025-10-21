"""Magnitude warping augmentation for time series data.

Applies smooth warping to time series magnitude using cubic splines.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline


class MagnitudeWarp(nn.Module):
    """Magnitude warping augmentation.

    Applies smooth warping to time series magnitude using cubic splines.
    """

    def __init__(self, sigma: float = 0.2, knots: int = 4, prob: float = 0.5):
        """Initialize magnitude warping.

        Args:
            sigma: Standard deviation for warp curve
            knots: Number of control points for spline
            prob: Probability of applying warping
        """
        super().__init__()
        self.sigma = sigma
        self.knots = knots
        self.prob = prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply magnitude warping.

        Args:
            x: Input tensor [batch, seq_len, features]

        Returns:
            Warped tensor or original if no augmentation
        """
        if self.training and torch.rand(1) < self.prob:
            batch_size, seq_len, n_features = x.shape
            x_warped = x.clone()

            # Apply warping to each sample and feature
            for b in range(batch_size):
                for f in range(n_features):
                    # Generate random warp curve
                    knot_positions = np.linspace(0, seq_len - 1, self.knots)
                    knot_values = np.random.normal(1.0, self.sigma, self.knots)

                    # Create cubic spline
                    cs = CubicSpline(knot_positions, knot_values)

                    # Apply warping to sequence
                    warp_curve = torch.tensor(
                        cs(np.arange(seq_len)), dtype=x.dtype, device=x.device
                    )
                    x_warped[b, :, f] = x[b, :, f] * warp_curve

            return x_warped

        return x


class OnTheFlyAugmentation(nn.Module):
    """On-the-fly augmentation pipeline.

    Applies jitter and magnitude warping with specified multipliers.
    """

    def __init__(
        self,
        jitter_sigma: float = 0.03,
        warp_sigma: float = 0.2,
        jitter_prob: float = 0.8,
        warp_prob: float = 0.5,
        multiplier: int = 3,
    ):
        """Initialize on-the-fly augmentation.

        Args:
            jitter_sigma: Standard deviation for jitter
            warp_sigma: Standard deviation for magnitude warping
            jitter_prob: Probability of applying jitter
            warp_prob: Probability of applying warping
            multiplier: Number of augmented versions to create
        """
        super().__init__()
        from .jitter import Jitter

        self.jitter = Jitter(sigma=jitter_sigma, prob=jitter_prob)
        self.warp = MagnitudeWarp(sigma=warp_sigma, prob=warp_prob)
        self.multiplier = multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply on-the-fly augmentation.

        Args:
            x: Input tensor [batch, seq_len, features]

        Returns:
            Augmented tensor [batch * multiplier, seq_len, features]
        """
        if not self.training:
            return x

        batch_size, seq_len, n_features = x.shape
        augmented_samples = []

        # Include original sample
        augmented_samples.append(x)

        # Generate augmented versions
        for _ in range(self.multiplier - 1):
            aug_x = x.clone()
            aug_x = self.jitter(aug_x)
            aug_x = self.warp(aug_x)
            augmented_samples.append(aug_x)

        # Stack all samples
        return torch.cat(augmented_samples, dim=0)
