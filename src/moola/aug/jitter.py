"""Jitter augmentation for time series data.

Adds Gaussian noise to input sequences to improve robustness.
"""

import torch
import torch.nn as nn


class Jitter(nn.Module):
    """Gaussian jitter augmentation.

    Adds random noise to input sequences with specified probability.
    """

    def __init__(self, sigma: float = 0.03, prob: float = 0.8):
        """Initialize jitter augmentation.

        Args:
            sigma: Standard deviation of Gaussian noise
            prob: Probability of applying jitter
        """
        super().__init__()
        self.sigma = sigma
        self.prob = prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply jitter augmentation.

        Args:
            x: Input tensor [batch, seq_len, features]

        Returns:
            Augmented tensor or original if no augmentation
        """
        if self.training and torch.rand(1) < self.prob:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x


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

            # Generate warp curve for each feature
            warp_curves = torch.randn(batch_size, n_features, self.knots) * self.sigma

            # Interpolate to sequence length
            from torch.nn.functional import interpolate

            warp_curves = interpolate(
                warp_curves.unsqueeze(1), size=seq_len, mode="linear", align_corners=False
            ).squeeze(1)

            # Apply warping
            warp_curves = warp_curves.transpose(1, 2)  # [batch, seq_len, features]
            return x * (1 + warp_curves)

        return x
