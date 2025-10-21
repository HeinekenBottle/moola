"""Metrics module for MOOLA evaluation.

Provides comprehensive metrics following Stones specifications:
- Hit@±3 ≥60% threshold
- F1-macro ≥0.50 threshold
- ECE <0.10 threshold
- Joint success ≥40% threshold
- Bootstrap 1,000 for CIs
"""

import torch

from .calibration import (
    adaptive_ece,
    brier_score,
    compute_calibration_metrics,
    expected_calibration_error,
    reliability_diagram,
)
from .hit_metrics import (
    compute_joint_success_metrics,
    compute_pointer_metrics,
    hit_at_k,
)
from .joint_metrics import (
    bootstrap_confidence_intervals,
    compute_comprehensive_joint_metrics,
    joint_success_rate,
    stones_evaluation_summary,
)

__all__ = [
    # Hit metrics
    "hit_at_k",
    "compute_pointer_metrics",
    "compute_joint_success_metrics",
    # Calibration metrics
    "expected_calibration_error",
    "adaptive_ece",
    "brier_score",
    "reliability_diagram",
    "compute_calibration_metrics",
    # Joint metrics
    "joint_success_rate",
    "compute_comprehensive_joint_metrics",
    "bootstrap_confidence_intervals",
    "stones_evaluation_summary",
]


def evaluate_stones_metrics(
    type_probs: torch.Tensor,
    type_targets: torch.Tensor,
    pred_center: torch.Tensor,
    pred_length: torch.Tensor,
    true_center: torch.Tensor,
    true_length: torch.Tensor,
    bootstrap_samples: int = 1000,
) -> dict:
    """Evaluate all Stones metrics in one call.

    Args:
        type_probs: Type probabilities [batch, n_classes]
        type_targets: True type labels [batch]
        pred_center: Predicted center values [batch]
        pred_length: Predicted length values [batch]
        true_center: True center values [batch]
        true_length: True length values [batch]
        bootstrap_samples: Number of bootstrap samples for CIs

    Returns:
        Comprehensive metrics dictionary with Stones thresholds
    """
    return compute_comprehensive_joint_metrics(
        type_probs=type_probs,
        type_targets=type_targets,
        pred_center=pred_center,
        pred_length=pred_length,
        true_center=true_center,
        true_length=true_length,
        bootstrap_samples=bootstrap_samples,
    )


def check_stones_thresholds(metrics: dict) -> dict:
    """Check if metrics meet Stones thresholds.

    Args:
        metrics: Metrics dictionary from evaluate_stones_metrics

    Returns:
        Dictionary with pass/fail status for each threshold
    """
    return stones_evaluation_summary(metrics)
