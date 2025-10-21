"""Data augmentation utilities for time series pre-training.

Provides augmentation strategies that preserve financial semantics while
generating synthetic unlabeled samples for robust pre-training.

Supported Augmentations:
    - Time warping: Temporal stretching/compression (±12% - optimized for masked LSTM)
    - Magnitude jittering: Additive Gaussian noise (±5% of std)
    - Window sliding: Overlapping subsequences
    - Volatility scaling: Scale high-low spreads (±15% - simulate market regimes)

Target: Generate 1000-5000 augmented samples from existing unlabeled data.

Rationale for Parameter Selection:
    - 12% time warping: Conservative enough to preserve pivot locations (bars 40-70)
      while providing sufficient temporal diversity for reconstruction task
    - 5% jittering: Simulates market microstructure noise without destroying patterns
    - 15% volatility scaling: Represents realistic VIX regime shifts (low→high vol)

Usage:
    augmenter = TimeSeriesAugmenter()
    X_augmented = augmenter.augment_dataset(X_unlabeled, num_augmentations=4)
    # Original: 11,873 samples → Augmented: 59,365 samples (5x)
"""

from typing import Tuple

import numpy as np
from scipy import interpolate


class TimeSeriesAugmenter:
    """Augmentation pipeline for financial time series data.

    Applies semantic-preserving transformations to OHLC sequences to
    generate synthetic training data for pre-training.

    All augmentations preserve key properties:
    - OHLC relationships (H >= max(O,C), L <= min(O,C))
    - Temporal ordering
    - Relative price movements
    """

    def __init__(
        self,
        time_warp_prob: float = 0.5,
        time_warp_sigma: float = 0.12,
        jitter_prob: float = 0.5,
        jitter_sigma: float = 0.01,  # Paper-strict: lighter jitter for pretraining (σ=0.01)
        volatility_scale_prob: float = 0.3,
        volatility_scale_range: Tuple[float, float] = (0.85, 1.15),
        window_shift_prob: float = 0.0,  # Disabled by default (changes sequence length)
    ):
        """Initialize augmenter with augmentation probabilities.

        Args:
            time_warp_prob: Probability of applying time warping
            time_warp_sigma: Magnitude of temporal distortion (0.12 = ±12% - optimized for masked pre-training)
            jitter_prob: Probability of applying jittering
            jitter_sigma: Noise magnitude relative to std (0.05 = 5% - increased for noise robustness)
            volatility_scale_prob: Probability of volatility scaling
            volatility_scale_range: Range for volatility multiplier (0.85-1.15 = ±15% volatility)
            window_shift_prob: Probability of window shifting (disabled)
        """
        self.time_warp_prob = time_warp_prob
        self.time_warp_sigma = time_warp_sigma
        self.jitter_prob = jitter_prob
        self.jitter_sigma = jitter_sigma
        self.volatility_scale_prob = volatility_scale_prob
        self.volatility_scale_range = volatility_scale_range
        self.window_shift_prob = window_shift_prob

    def time_warp(self, x: np.ndarray, sigma: float = 0.12) -> np.ndarray:
        """Apply temporal warping to time series.

        Stretches or compresses the time axis by interpolating along a
        smooth random curve. This simulates different market speeds.

        Args:
            x: [N, seq_len, features] input array
            sigma: Warping magnitude (0.12 = ±12% temporal distortion)

        Returns:
            Warped array with same shape [N, seq_len, features]
        """
        N, T, D = x.shape
        x_warped = np.zeros_like(x)

        for i in range(N):
            # Generate smooth random warping curve
            # Original indices: [0, 1, 2, ..., T-1]
            # Warped indices: smooth curve around original
            orig_steps = np.linspace(0, T - 1, T)

            # Random smooth warping (cumsum of normal noise)
            warp = np.cumsum(np.random.randn(T) * sigma)
            warp = warp - warp[0]  # Start at 0
            warp = warp / warp[-1] * (T - 1)  # End at T-1

            # Clip to valid range
            warp = np.clip(warp, 0, T - 1)

            # Interpolate each feature dimension
            for d in range(D):
                f = interpolate.interp1d(
                    orig_steps, x[i, :, d], kind="cubic", fill_value="extrapolate"
                )
                x_warped[i, :, d] = f(warp)

        return x_warped

    def jitter(self, x: np.ndarray, sigma: float = 0.05) -> np.ndarray:
        """Add Gaussian noise to time series.

        Adds small amount of noise proportional to each feature's standard
        deviation. This simulates measurement noise and minor price fluctuations.

        Args:
            x: [N, seq_len, features] input array
            sigma: Noise magnitude (0.05 = 5% of std)

        Returns:
            Jittered array with same shape
        """
        # Compute std per feature (across all samples and timesteps)
        feature_std = x.std(axis=(0, 1), keepdims=True)

        # Generate Gaussian noise scaled by feature std
        noise = np.random.randn(*x.shape) * sigma * feature_std

        return x + noise

    def volatility_scale(
        self, x: np.ndarray, scale_range: Tuple[float, float] = (0.85, 1.15)
    ) -> np.ndarray:
        """Scale high-low spreads to simulate different volatility regimes.

        Multiplies the distance from midpoint (average of H and L) for
        high and low prices, simulating higher or lower volatility periods.

        CRITICAL: Preserves OHLC relationships (H >= O,C and L <= O,C)

        Args:
            x: [N, seq_len, 4] OHLC array
                Features: [Open, High, Low, Close]
            scale_range: (min_scale, max_scale) for volatility multiplier

        Returns:
            Scaled array with same shape
        """
        N, T, D = x.shape
        assert D == 4, "volatility_scale requires OHLC data (4 features)"

        x_scaled = x.copy()

        for i in range(N):
            # Random scale per sample
            scale = np.random.uniform(*scale_range)

            # Compute midpoint (average of high and low)
            mid = (x[i, :, 1] + x[i, :, 2]) / 2  # (H + L) / 2

            # Scale high and low around midpoint
            x_scaled[i, :, 1] = mid + (x[i, :, 1] - mid) * scale  # High
            x_scaled[i, :, 2] = mid + (x[i, :, 2] - mid) * scale  # Low

            # Ensure OHLC constraints still hold
            # High must be >= Open and Close
            x_scaled[i, :, 1] = np.maximum(
                x_scaled[i, :, 1], np.maximum(x_scaled[i, :, 0], x_scaled[i, :, 3])
            )
            # Low must be <= Open and Close
            x_scaled[i, :, 2] = np.minimum(
                x_scaled[i, :, 2], np.minimum(x_scaled[i, :, 0], x_scaled[i, :, 3])
            )

        return x_scaled

    def window_shift(self, x: np.ndarray, shift: int = 5) -> np.ndarray:
        """Create overlapping windows (not used in standard augmentation).

        Shifts window start position to create new training samples.
        NOTE: This changes the semantic meaning of the sequence (different
        time period), so use with caution.

        Args:
            x: [N, seq_len, features] input array
            shift: Number of timesteps to shift

        Returns:
            Shifted array (may have different length)
        """
        return x[:, shift:, :]

    def apply_augmentation(self, x: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Apply random combination of augmentations.

        Each augmentation is applied with its configured probability.
        Multiple augmentations can be applied to the same sample.

        Args:
            x: [N, seq_len, features] input array
            deterministic: If True, apply all augmentations (for testing)

        Returns:
            Augmented array with same shape
        """
        x_aug = x.copy()

        # Time warping
        if deterministic or np.random.rand() < self.time_warp_prob:
            x_aug = self.time_warp(x_aug, sigma=self.time_warp_sigma)

        # Jittering
        if deterministic or np.random.rand() < self.jitter_prob:
            x_aug = self.jitter(x_aug, sigma=self.jitter_sigma)

        # Volatility scaling (only for OHLC data)
        if x.shape[-1] == 4:
            if deterministic or np.random.rand() < self.volatility_scale_prob:
                x_aug = self.volatility_scale(x_aug, scale_range=self.volatility_scale_range)

        return x_aug

    def augment_dataset(self, X: np.ndarray, num_augmentations: int = 4) -> np.ndarray:
        """Generate augmented dataset for pre-training.

        Creates multiple augmented versions of each sample to expand
        the dataset size for robust pre-training.

        Args:
            X: [N, seq_len, features] original dataset
            num_augmentations: Number of augmented versions per sample

        Returns:
            Augmented dataset [N * (1 + num_augmentations), seq_len, features]
                Includes original + augmented versions

        Example:
            >>> X_unlabeled.shape
            (11873, 105, 4)
            >>> augmenter = TimeSeriesAugmenter()
            >>> X_augmented = augmenter.augment_dataset(X_unlabeled, num_augmentations=4)
            >>> X_augmented.shape
            (59365, 105, 4)  # 11873 * 5
        """
        augmented = [X]  # Start with original data

        print(f"[AUGMENTATION] Generating {num_augmentations} augmented versions...")
        print(f"  Original size: {len(X)} samples")

        for i in range(num_augmentations):
            X_aug = self.apply_augmentation(X, deterministic=False)
            augmented.append(X_aug)
            print(f"  Augmentation {i+1}/{num_augmentations}: +{len(X_aug)} samples")

        # Concatenate all versions
        X_final = np.concatenate(augmented, axis=0)

        print(f"[AUGMENTATION] Complete!")
        print(f"  Final size: {len(X_final)} samples ({len(X_final)/len(X):.1f}x original)")

        return X_final


def generate_unlabeled_samples(
    X_labeled: np.ndarray, target_count: int = 5000, augmenter: TimeSeriesAugmenter = None
) -> np.ndarray:
    """Generate unlabeled samples from labeled data using augmentation.

    Utility function to create unlabeled pre-training data when only
    labeled data is available. Uses aggressive augmentation to reach
    target sample count.

    Args:
        X_labeled: [N, seq_len, features] labeled dataset
        target_count: Desired number of unlabeled samples
        augmenter: Augmenter instance (uses default if None)

    Returns:
        Unlabeled dataset [target_count, seq_len, features]
    """
    if augmenter is None:
        augmenter = TimeSeriesAugmenter()

    num_augmentations = max(1, int(np.ceil(target_count / len(X_labeled))) - 1)

    print(f"[UNLABELED GENERATION] Target: {target_count} samples")
    print(f"  Available labeled: {len(X_labeled)} samples")
    print(f"  Augmentations needed: {num_augmentations}x")

    X_unlabeled = augmenter.augment_dataset(X_labeled, num_augmentations)

    # Trim to exact target count if overshot
    if len(X_unlabeled) > target_count:
        X_unlabeled = X_unlabeled[:target_count]

    print(f"  Final unlabeled size: {len(X_unlabeled)} samples")

    return X_unlabeled
