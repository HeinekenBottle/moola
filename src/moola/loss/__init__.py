"""Loss functions for Moola models.

This module provides production-ready loss functions with a focus on
uncertainty-weighted multi-task learning as the default approach.

Main Components:
- UncertaintyWeightedLoss: Default for multi-task learning (Kendall et al.)
- HuberLoss: Robust regression for pointer prediction
- WeightedBCELoss: Handles class imbalance in classification
- FocalLoss: Focuses on hard examples

Usage:
    >>> from moola.loss import UncertaintyWeightedLoss, HuberLoss
    >>> loss_fn = UncertaintyWeightedLoss()
    >>> huber = HuberLoss(delta=0.08)
"""

from .uncertainty_weighted import (
    FocalLoss,
    HuberLoss,
    UncertaintyWeightedLoss,
    WeightedBCELoss,
    create_uncertainty_loss,
    log_uncertainty_metrics,
)

__all__ = [
    "UncertaintyWeightedLoss",
    "HuberLoss",
    "WeightedBCELoss",
    "FocalLoss",
    "create_uncertainty_loss",
    "log_uncertainty_metrics",
]
