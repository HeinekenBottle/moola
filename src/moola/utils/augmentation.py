"""Data augmentation utilities for time series classification.

Implements mixup and cutmix augmentation strategies for improved
model generalization on small datasets.

References:
    - mixup: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
    - CutMix: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers" (ICCV 2019)
"""

import numpy as np
import torch


def mixup(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup augmentation to a batch.

    Mixup creates virtual training examples by interpolating between pairs
    of samples and their labels.

    Args:
        x: Input batch of shape [B, T, F] or [B, F]
        y: Target labels of shape [B] (class indices)
        alpha: Beta distribution parameter (higher = more mixing)

    Returns:
        Tuple of (mixed_x, y_a, y_b, lam) where:
            - mixed_x: Mixed input batch
            - y_a: First set of labels
            - y_b: Second set of labels
            - lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    # Mix inputs: x_mixed = lam * x_i + (1 - lam) * x_j
    mixed_x = lam * x + (1 - lam) * x[index]

    # Return both label sets for loss computation
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply CutMix augmentation to a batch.

    CutMix replaces a region of one sample with a region from another sample,
    mixing labels proportionally to the area replaced.

    Args:
        x: Input batch of shape [B, T, F] or [B, F]
        y: Target labels of shape [B] (class indices)
        alpha: Beta distribution parameter (higher = more mixing)

    Returns:
        Tuple of (mixed_x, y_a, y_b, lam) where:
            - mixed_x: Mixed input batch with cut-and-pasted regions
            - y_a: First set of labels
            - y_b: Second set of labels
            - lam: Mixing coefficient (ratio of original sample)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    # For time series: cut along the temporal dimension
    if x.ndim == 3:  # [B, T, F]
        seq_len = x.size(1)

        # Compute cut size and position
        cut_len = int(seq_len * (1 - lam))
        cut_start = np.random.randint(0, seq_len - cut_len + 1) if cut_len > 0 else 0

        # Create mixed input by replacing temporal region
        mixed_x = x.clone()
        mixed_x[:, cut_start:cut_start + cut_len, :] = x[index, cut_start:cut_start + cut_len, :]

        # Adjust lambda to actual proportion
        lam = 1 - (cut_len / seq_len)

    else:  # [B, F] - treat as spatial
        feature_dim = x.size(1)

        # Compute cut size and position
        cut_len = int(feature_dim * (1 - lam))
        cut_start = np.random.randint(0, feature_dim - cut_len + 1) if cut_len > 0 else 0

        # Create mixed input by replacing feature region
        mixed_x = x.clone()
        mixed_x[:, cut_start:cut_start + cut_len] = x[index, cut_start:cut_start + cut_len]

        # Adjust lambda to actual proportion
        lam = 1 - (cut_len / feature_dim)

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 0.3,
    cutmix_prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply either mixup or cutmix with specified probability.

    Args:
        x: Input batch of shape [B, T, F] or [B, F]
        y: Target labels of shape [B] (class indices)
        mixup_alpha: Beta parameter for mixup (default: 0.2 for gentler mixing)
        cutmix_alpha: Beta parameter for cutmix (default: 0.3 for gentler mixing)
        cutmix_prob: Probability of applying cutmix vs mixup

    Returns:
        Tuple of (mixed_x, y_a, y_b, lam)
    """
    if np.random.rand() < cutmix_prob:
        return cutmix(x, y, alpha=cutmix_alpha)
    else:
        return mixup(x, y, alpha=mixup_alpha)


def mixup_criterion(
    criterion: callable,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute loss for mixup/cutmix augmented samples.

    Args:
        criterion: Loss function (e.g., nn.CrossEntropyLoss())
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixing coefficient

    Returns:
        Mixed loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
