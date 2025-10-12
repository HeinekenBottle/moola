"""Focal Loss for handling class imbalance in deep learning models.

Focal Loss addresses class imbalance by down-weighting easy examples and focusing
on hard negatives. This is particularly useful when standard class weighting is
insufficient.

Reference:
    Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.

    Focal Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
    - p_t is the model's estimated probability for the true class
    - alpha_t is the class weight for the true class
    - gamma controls the focusing parameter (higher = more focus on hard examples)

    Args:
        gamma: Focusing parameter (default: 2.0). Higher values increase focus on hard examples.
        alpha: Class weights as tensor [C] or None for uniform weighting
        reduction: Specifies the reduction to apply: 'none', 'mean', or 'sum'
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Model predictions [batch_size, num_classes] (logits, not probabilities)
            targets: Ground truth labels [batch_size] (class indices)

        Returns:
            Focal loss (scalar if reduction='mean'/'sum', tensor if reduction='none')
        """
        # Compute cross entropy loss (no reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get probabilities for the true class
        p = torch.exp(-ce_loss)  # p_t = exp(-ce_loss)

        # Compute focal term: (1 - p_t)^gamma
        focal_weight = (1 - p) ** self.gamma

        # Compute focal loss
        focal_loss = focal_weight * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
