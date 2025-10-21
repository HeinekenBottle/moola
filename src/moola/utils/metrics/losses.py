"""Multi-task loss functions for training models with multiple objectives.

This module provides loss functions for joint training of classification and
pointer prediction tasks in market pattern recognition.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_multitask_loss(
    outputs: dict, targets: dict, alpha: float = 0.5, beta: float = 0.25, device: str = "cpu"
) -> Tuple[torch.Tensor, dict]:
    """Compute balanced multi-task loss for classification + pointer prediction.

    This loss function enables joint training of three related tasks:
    1. Classification: Predict market pattern type (consolidation/retracement/reversal)
    2. Start pointer: Identify when market expansion begins within inner window
    3. End pointer: Identify when market expansion ends within inner window

    The loss balances all three tasks using weighted combination:
        total_loss = alpha * L_class + beta * (L_start + L_end)

    Args:
        outputs: Dictionary containing model predictions:
            - 'classification': [B, 3] class logits (raw model output, no softmax)
            - 'start': [B, 45] per-timestep start logits (raw output, no sigmoid)
            - 'end': [B, 45] per-timestep end logits (raw output, no sigmoid)
        targets: Dictionary containing ground truth labels:
            - 'class': [B] class indices as LongTensor in range [0, 2]
            - 'start_idx': [B] start indices as LongTensor in range [0, 44]
            - 'end_idx': [B] end indices as LongTensor in range [0, 44]
        alpha: Weight for classification loss (default 0.5)
            Higher alpha prioritizes classification accuracy
        beta: Weight for each pointer loss (default 0.25)
            Higher beta prioritizes pointer localization accuracy
            Note: Total weight = alpha + 2*beta should ideally equal 1.0
        device: Device for tensor operations ('cpu' or 'cuda')

    Returns:
        Tuple containing:
        - total_loss: Weighted sum of all task losses (scalar tensor)
        - loss_dict: Dictionary with individual loss values for logging:
            - 'class': Classification loss (float)
            - 'start': Start pointer loss (float)
            - 'end': End pointer loss (float)
            - 'total': Total weighted loss (float)

    Loss Formulation Details:
        L_class = CrossEntropyLoss(classification_logits, class_targets)
            Standard multi-class classification loss

        L_start = BCEWithLogitsLoss(start_logits, start_one_hot)
            Binary cross-entropy for each timestep (is this the start?)
            Treats pointer prediction as multi-label classification

        L_end = BCEWithLogitsLoss(end_logits, end_one_hot)
            Binary cross-entropy for each timestep (is this the end?)

    Example:
        >>> outputs = {
        ...     'classification': torch.randn(8, 3),  # Batch of 8
        ...     'start': torch.randn(8, 45),
        ...     'end': torch.randn(8, 45)
        ... }
        >>> targets = {
        ...     'class': torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
        ...     'start_idx': torch.tensor([5, 12, 8, 15, 3, 20, 10, 7]),
        ...     'end_idx': torch.tensor([20, 30, 25, 35, 18, 40, 28, 22])
        ... }
        >>> loss, loss_dict = compute_multitask_loss(outputs, targets)
        >>> print(f"Total loss: {loss_dict['total']:.4f}")
        >>> print(f"Class: {loss_dict['class']:.4f}, Start: {loss_dict['start']:.4f}, End: {loss_dict['end']:.4f}")

    Notes:
        - All logits should be raw outputs (before softmax/sigmoid)
        - Pointer indices are relative to inner window [0, 45)
        - One-hot encoding creates dense targets for BCEWithLogitsLoss
        - Loss weighting allows tuning task importance during training
        - For single-task training, set alpha=1.0 and beta=0.0
    """
    # Extract outputs
    classification_logits = outputs["classification"]  # [B, 3]
    start_logits = outputs["start"]  # [B, 45]
    end_logits = outputs["end"]  # [B, 45]

    # Extract targets
    class_targets = targets["class"]  # [B]
    start_idx_targets = targets["start_idx"]  # [B]
    end_idx_targets = targets["end_idx"]  # [B]

    batch_size = classification_logits.shape[0]
    inner_window_size = 45

    # 1. Classification Loss (standard cross-entropy)
    criterion_class = nn.CrossEntropyLoss()
    loss_class = criterion_class(classification_logits, class_targets)

    # 2. Pointer Losses (binary cross-entropy with one-hot targets)
    # Convert pointer indices to one-hot vectors [B, 45]
    start_one_hot = torch.zeros(batch_size, inner_window_size, device=device)
    end_one_hot = torch.zeros(batch_size, inner_window_size, device=device)

    # Scatter ones at target indices
    start_one_hot.scatter_(1, start_idx_targets.unsqueeze(1), 1.0)
    end_one_hot.scatter_(1, end_idx_targets.unsqueeze(1), 1.0)

    # Binary cross-entropy with logits (includes sigmoid)
    criterion_pointer = nn.BCEWithLogitsLoss()
    loss_start = criterion_pointer(start_logits, start_one_hot)
    loss_end = criterion_pointer(end_logits, end_one_hot)

    # 3. Combine losses with weighting
    total_loss = alpha * loss_class + beta * (loss_start + loss_end)

    # Prepare loss dictionary for logging
    loss_dict = {
        "class": loss_class.item(),
        "start": loss_start.item(),
        "end": loss_end.item(),
        "total": total_loss.item(),
    }

    return total_loss, loss_dict
