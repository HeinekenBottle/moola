"""Uncertainty-weighted loss for multi-task learning.

Implements learnable task balancing using homoscedastic uncertainty
as described in Kendall et al., CVPR 2018.

This is the DEFAULT loss function for all multi-task learning in Moola.
Manual λ weighting (loss_alpha, loss_beta) is deprecated and should
be replaced with this implementation.

Reference:
    Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses 
    for Scene Geometry and Semantics", CVPR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class UncertaintyWeightedLoss(nn.Module):
    """Production-ready uncertainty-weighted loss for multi-task learning.

    Learns optimal task weights using homoscedastic uncertainty according to:
    - Regression: (1/2σ²)L + log(σ)
    - Classification: (1/σ²)L + log(σ)

    The log(σ) regularization prevents σ → ∞ which would otherwise minimize
    the loss by making tasks infinitely uncertain.

    This is the DEFAULT loss function for EnhancedSimpleLSTM multi-task training.
    Manual loss weighting (loss_alpha, loss_beta) is deprecated.

    Args:
        init_log_var_ptr: Initial log variance for pointer regression (default: 0.0)
        init_log_var_type: Initial log variance for type classification (default: 0.0)
        min_sigma: Minimum σ value to prevent numerical instability (default: 1e-6)
        max_sigma: Maximum σ value to prevent numerical instability (default: 1e6)

    Example:
        >>> loss_fn = UncertaintyWeightedLoss()
        >>> total_loss, metrics = loss_fn(type_loss=1.2, pointer_loss=0.5)
        >>> print(f"Total: {total_loss:.4f}, σ_ptr: {metrics['pointer_sigma']:.3f}")
    """

    def __init__(
        self,
        init_log_var_ptr: float = -0.60,  # Kendall bias: favor pointer (σ≈0.74)
        init_log_var_type: float = 0.00,   # Neutral for classification (σ=1.0)
        min_sigma: float = 1e-6,
        max_sigma: float = 1e6,
    ):
        super().__init__()

        # Learnable log variance parameters
        self.log_var_ptr = nn.Parameter(torch.tensor(init_log_var_ptr, dtype=torch.float32))
        self.log_var_type = nn.Parameter(torch.tensor(init_log_var_type, dtype=torch.float32))

        # Numerical stability bounds
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.min_log_var = 2.0 * torch.log(torch.tensor(min_sigma))
        self.max_log_var = 2.0 * torch.log(torch.tensor(max_sigma))

        # Track initialization for debugging
        self._initialized = False

    def _clamp_log_vars(self) -> None:
        """Clamp log variances to prevent numerical instability."""
        with torch.no_grad():
            self.log_var_ptr.clamp_(min=self.min_log_var.item(), max=self.max_log_var.item())
            self.log_var_type.clamp_(min=self.min_log_var.item(), max=self.max_log_var.item())

    def forward(
        self, type_loss: torch.Tensor, pointer_loss: torch.Tensor, return_components: bool = False
    ) -> (
        tuple[torch.Tensor, dict[str, float]]
        | tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]
    ):
        """Compute uncertainty-weighted multi-task loss.

        Args:
            type_loss: Classification loss tensor (scalar or batch)
            pointer_loss: Pointer regression loss tensor (scalar or batch)
            return_components: If True, return individual loss components

        Returns:
            Tuple of (total_loss, metrics_dict)

        Note:
            Both losses should be reduced (mean) before passing to this function.
            The uncertainty weighting handles the task balancing automatically.
        """
        # Ensure numerical stability
        self._clamp_log_vars()

        # Compute precision (1/σ²) for each task
        precision_ptr = torch.exp(-self.log_var_ptr)  # 1/σ² for regression
        precision_type = torch.exp(-self.log_var_type)  # 1/σ² for classification

        # Apply Kendall's formulas:
        # Regression: (1/2σ²)L + log(σ)
        # Classification: (1/σ²)L + log(σ)
        weighted_ptr = 0.5 * precision_ptr * pointer_loss + self.log_var_ptr
        weighted_type = precision_type * type_loss + self.log_var_type

        # Total loss - reduce to scalar
        total_loss = (weighted_ptr + weighted_type).mean()

        # Compute metrics for monitoring
        sigma_ptr = torch.exp(0.5 * self.log_var_ptr)
        sigma_type = torch.exp(0.5 * self.log_var_type)

        # Reduce input losses to scalars for metrics
        type_loss_scalar = type_loss.mean() if type_loss.numel() > 1 else type_loss
        pointer_loss_scalar = pointer_loss.mean() if pointer_loss.numel() > 1 else pointer_loss

        metrics = {
            "total_loss": total_loss.item(),
            "type_loss": type_loss_scalar.item(),
            "pointer_loss": pointer_loss_scalar.item(),
            "weighted_type": (
                weighted_type.mean().item() if weighted_type.numel() > 1 else weighted_type.item()
            ),
            "weighted_pointer": (
                weighted_ptr.mean().item() if weighted_ptr.numel() > 1 else weighted_ptr.item()
            ),
            "log_var_type": self.log_var_type.item(),
            "log_var_pointer": self.log_var_ptr.item(),
            "type_sigma": sigma_type.item(),
            "pointer_sigma": sigma_ptr.item(),
            "type_precision": precision_type.item(),
            "pointer_precision": precision_ptr.item(),
        }

        # Log initialization on first forward pass
        if not self._initialized:
            logger.info(
                f"[UNCERTAINTY LOSS] Initialized with σ_ptr={sigma_ptr.item():.3f}, "
                f"σ_type={sigma_type.item():.3f}"
            )
            self._initialized = True

        if return_components:
            components = {
                "weighted_ptr": weighted_ptr,
                "weighted_type": weighted_type,
                "precision_ptr": precision_ptr,
                "precision_type": precision_type,
            }
            return total_loss, metrics, components

        return total_loss, metrics

    def get_uncertainties(self) -> dict[str, float]:
        """Get current uncertainty values for monitoring.

        Returns:
            Dictionary with sigma values and their interpretations:
            - sigma_ptr: Uncertainty for pointer regression (higher = more uncertain)
            - sigma_type: Uncertainty for type classification (higher = more uncertain)
            - ptr_weight: Effective weight for pointer task (1/σ²)
            - type_weight: Effective weight for type task (1/σ²)
        """
        sigma_ptr = torch.exp(0.5 * self.log_var_ptr).item()
        sigma_type = torch.exp(0.5 * self.log_var_type).item()

        return {
            "sigma_ptr": sigma_ptr,
            "sigma_type": sigma_type,
            "ptr_weight": 1.0 / (sigma_ptr**2),
            "type_weight": 1.0 / (sigma_type**2),
        }

    def get_task_balance_ratio(self) -> float:
        """Get the ratio of task weights for monitoring convergence.

        Returns:
            Ratio of pointer weight to type weight.
            Values > 1 mean pointer task is weighted more heavily.
        """
        uncertainties = self.get_uncertainties()
        return uncertainties["ptr_weight"] / uncertainties["type_weight"]


class HuberLoss(nn.Module):
    """Huber loss for robust pointer regression.

    Less sensitive to outliers than MSE with smooth quadratic-to-linear transition.
    Default δ=0.08 provides smooth transition over ~8 timesteps for 105-timestep windows.

    Args:
        delta: Transition point between quadratic and linear loss (default: 0.08)
        reduction: Reduction method ('mean', 'sum', 'none') (default: 'mean')

    Example:
        >>> huber = HuberLoss(delta=0.08)
        >>> loss = huber(pred, target)
    """

    def __init__(self, delta: float = 0.08, reduction: str = "mean"):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss.

        Args:
            pred: Predictions [batch, ...]
            target: Targets [batch, ...]

        Returns:
            Huber loss
        """
        error = pred - target
        abs_error = torch.abs(error)

        # Quadratic for small errors, linear for large errors
        quadratic = torch.minimum(abs_error, torch.tensor(self.delta, device=error.device))
        linear = abs_error - quadratic

        loss = 0.5 * quadratic**2 + self.delta * linear

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class WeightedBCELoss(nn.Module):
    """Weighted binary cross-entropy for imbalanced classification.

    Handles adversarial class imbalance (consolidation >> retracement) by
    applying higher weight to the minority positive class.

    Args:
        pos_weight: Weight for positive class (default: 2.0 for retracement)
        reduction: Reduction method ('mean', 'sum', 'none') (default: 'mean')

    Example:
        >>> bce = WeightedBCELoss(pos_weight=2.0)
        >>> loss = bce(logits, targets)
    """

    def __init__(self, pos_weight: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE loss.

        Args:
            pred: Logits [batch, n_classes]
            target: Targets [batch, n_classes]

        Returns:
            Weighted BCE loss
        """
        pos_weight = torch.tensor(self.pos_weight, device=pred.device)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pos_weight, reduction=self.reduction
        )
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.

    Focuses on hard examples by down-weighting easy examples.
    Combines well with class weighting for severe imbalance.

    Args:
        alpha: Class weights (default: [1.0, 1.17] for consolidation/retracement)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', 'none') (default: 'mean')

    Example:
        >>> focal = FocalLoss(alpha=[1.0, 1.17], gamma=2.0)
        >>> loss = focal(logits, targets)
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if self.alpha is not None:
            self.alpha = torch.tensor(self.alpha, dtype=torch.float32)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            pred: Logits [batch, n_classes]
            target: Targets [batch, n_classes]

        Returns:
            Focal loss
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(pred)

        # Compute focal weight
        pt = torch.where(target == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # Apply focal weighting
        loss = focal_weight * bce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(pred.device)
            alpha_t = torch.where(target == 1, alpha[1], alpha[0])
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


# Factory function for easy instantiation
def create_uncertainty_loss(loss_type: str = "uncertainty_weighted", **kwargs) -> nn.Module:
    """Create loss function instance.

    Args:
        loss_type: Type of loss ('uncertainty_weighted', 'huber', 'weighted_bce', 'focal')
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function instance

    Example:
        >>> loss_fn = create_uncertainty_loss('uncertainty_weighted')
        >>> huber = create_uncertainty_loss('huber', delta=0.1)
    """
    if loss_type == "uncertainty_weighted":
        return UncertaintyWeightedLoss(**kwargs)
    elif loss_type == "huber":
        return HuberLoss(**kwargs)
    elif loss_type == "weighted_bce":
        return WeightedBCELoss(**kwargs)
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Utility function for monitoring uncertainty convergence
def log_uncertainty_metrics(loss_fn: UncertaintyWeightedLoss, epoch: int) -> None:
    """Log uncertainty metrics for monitoring convergence.

    Args:
        loss_fn: UncertaintyWeightedLoss instance
        epoch: Current epoch number
    """
    uncertainties = loss_fn.get_uncertainties()
    balance_ratio = loss_fn.get_task_balance_ratio()

    logger.info(
        f"[EPOCH {epoch}] Uncertainty weights: "
        f"σ_ptr={uncertainties['sigma_ptr']:.3f} (w={uncertainties['ptr_weight']:.3f}), "
        f"σ_type={uncertainties['sigma_type']:.3f} (w={uncertainties['type_weight']:.3f}), "
        f"ratio={balance_ratio:.3f}"
    )
