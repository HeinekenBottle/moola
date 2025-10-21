"""Joint metrics for multi-task evaluation.

Computes combined metrics for classification and pointer tasks.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from .calibration import expected_calibration_error
from .hit_metrics import compute_pointer_metrics, hit_at_k


def joint_success_rate(
    type_correct: torch.Tensor,
    center_correct: torch.Tensor,
    length_correct: torch.Tensor,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute joint success rate with different weighting schemes.

    Args:
        type_correct: Boolean tensor for type classification correctness
        center_correct: Boolean tensor for center prediction correctness
        length_correct: Boolean tensor for length prediction correctness
        weights: Optional weights for different components

    Returns:
        Dictionary of joint success metrics
    """
    if weights is None:
        weights = {"type": 1.0, "center": 1.0, "length": 0.8}

    # Strict joint success (all must be correct)
    joint_strict = type_correct & center_correct & length_correct

    # Weighted joint success
    weighted_score = (
        weights["type"] * type_correct.float()
        + weights["center"] * center_correct.float()
        + weights["length"] * length_correct.float()
    ) / sum(weights.values())

    # Individual success rates
    type_success = type_correct.float().mean()
    center_success = center_correct.float().mean()
    length_success = length_correct.float().mean()

    return {
        "joint_success_strict": joint_strict.float().mean(),
        "joint_success_weighted": weighted_score.mean(),
        "type_success_rate": type_success,
        "center_success_rate": center_success,
        "length_success_rate": length_success,
        "meets_stones_threshold": joint_strict.float().mean() >= 0.4,  # Stones threshold
    }


def compute_comprehensive_joint_metrics(
    type_probs: torch.Tensor,
    type_targets: torch.Tensor,
    pred_center: torch.Tensor,
    pred_length: torch.Tensor,
    true_center: torch.Tensor,
    true_length: torch.Tensor,
    center_tolerance: int = 3,
    length_tolerance: float = 0.1,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """Compute comprehensive joint metrics with confidence intervals.

    Args:
        type_probs: Type probabilities [batch, n_classes]
        type_targets: True type labels [batch]
        pred_center: Predicted center values [batch]
        pred_length: Predicted length values [batch]
        true_center: True center values [batch]
        true_length: True length values [batch]
        center_tolerance: Tolerance for center predictions (± timesteps)
        length_tolerance: Tolerance for length predictions (relative)
        bootstrap_samples: Number of bootstrap samples for CIs
        confidence_level: Confidence level for intervals

    Returns:
        Dictionary of comprehensive metrics
    """
    batch_size = len(type_targets)

    # Type classification metrics
    type_pred = type_probs.argmax(dim=1)
    type_correct = type_pred == type_targets
    type_accuracy = type_correct.float().mean()

    # F1-macro for type classification
    type_f1_macro = f1_score(type_targets.cpu(), type_pred.cpu(), average="macro", zero_division=0)

    # Pointer metrics
    center_error = torch.abs(pred_center - true_center)
    length_error = torch.abs(pred_length - true_length)

    # Convert normalized center to timestep index
    center_timesteps = (pred_center * 104).round().long()
    true_center_timesteps = (true_center * 104).round().long()

    center_correct = (center_timesteps - true_center_timesteps).abs() <= center_tolerance
    length_correct = length_error <= length_tolerance

    # Hit@±3 for center
    hit_at_3 = hit_at_k(pred_center, true_center, k=center_tolerance)

    # Joint success metrics
    joint_metrics = joint_success_rate(type_correct, center_correct, length_correct)

    # Calibration metrics
    ece, _, _ = expected_calibration_error(type_probs, type_targets)

    # Bootstrap confidence intervals
    ci_metrics = bootstrap_confidence_intervals(
        type_correct,
        center_correct,
        length_correct,
        hit_at_k(pred_center, true_center, k=center_tolerance),
        type_f1_macro,
        ece,
        n_samples=bootstrap_samples,
        confidence_level=confidence_level,
    )

    # Combine all metrics
    all_metrics = {
        # Type classification
        "type_accuracy": type_accuracy,
        "type_f1_macro": type_f1_macro,
        "type_f1_macro_meets_threshold": type_f1_macro >= 0.5,  # Stones threshold
        # Pointer metrics
        "center_hit_at_3": hit_at_3,
        "center_hit_at_3_meets_threshold": hit_at_3 >= 0.6,  # Stones threshold
        "center_mae": center_error.mean(),
        "length_mae": length_error.mean(),
        # Joint metrics
        "joint_success_strict": joint_metrics["joint_success_strict"],
        "joint_success_meets_threshold": joint_metrics["meets_stones_threshold"],
        # Calibration
        "ece": ece,
        "ece_meets_threshold": ece < 0.1,  # Stones threshold
        # Overall success (all Stones thresholds met)
        "overall_stones_success": (
            type_f1_macro >= 0.5
            and hit_at_3 >= 0.6
            and joint_metrics["joint_success_strict"] >= 0.4
            and ece < 0.1
        ),
    }

    # Add confidence intervals
    all_metrics.update(ci_metrics)

    return all_metrics


def bootstrap_confidence_intervals(
    type_correct: torch.Tensor,
    center_correct: torch.Tensor,
    length_correct: torch.Tensor,
    hit_at_3_score: float,
    type_f1_macro: float,
    ece: float,
    n_samples: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    """Compute bootstrap confidence intervals for key metrics.

    Args:
        type_correct: Boolean tensor for type correctness
        center_correct: Boolean tensor for center correctness
        length_correct: Boolean tensor for length correctness
        hit_at_3_score: Hit@±3 score
        type_f1_macro: F1-macro score
        ece: Expected calibration error
        n_samples: Number of bootstrap samples
        confidence_level: Confidence level

    Returns:
        Dictionary with confidence intervals
    """
    batch_size = len(type_correct)
    alpha = 1 - confidence_level

    # Convert to numpy for easier bootstrapping
    type_correct_np = type_correct.cpu().numpy()
    center_correct_np = center_correct.cpu().numpy()
    length_correct_np = length_correct.cpu().numpy()

    # Bootstrap samples
    bootstrap_metrics = {
        "type_accuracy": [],
        "center_hit_rate": [],
        "length_hit_rate": [],
        "joint_success": [],
    }

    for _ in range(n_samples):
        # Sample with replacement
        indices = np.random.choice(batch_size, batch_size, replace=True)

        # Compute metrics on bootstrap sample
        type_acc = type_correct_np[indices].mean()
        center_hit = center_correct_np[indices].mean()
        length_hit = length_correct_np[indices].mean()
        joint_success = np.logical_and.reduce(
            [type_correct_np[indices], center_correct_np[indices], length_correct_np[indices]]
        ).mean()

        bootstrap_metrics["type_accuracy"].append(type_acc)
        bootstrap_metrics["center_hit_rate"].append(center_hit)
        bootstrap_metrics["length_hit_rate"].append(length_hit)
        bootstrap_metrics["joint_success"].append(joint_success)

    # Compute confidence intervals
    ci_results = {}
    for metric_name, values in bootstrap_metrics.items():
        values = np.array(values)
        lower = np.percentile(values, 100 * alpha / 2)
        upper = np.percentile(values, 100 * (1 - alpha / 2))
        ci_results[f"{metric_name}_ci"] = (lower, upper)

    # Add CI for fixed metrics (assuming normal approximation)
    # This is approximate since we can't easily bootstrap these without more data
    ci_results["type_f1_macro_ci"] = (
        max(0, type_f1_macro - 1.96 * np.sqrt(type_f1_macro * (1 - type_f1_macro) / batch_size)),
        min(1, type_f1_macro + 1.96 * np.sqrt(type_f1_macro * (1 - type_f1_macro) / batch_size)),
    )

    ci_results["hit_at_3_ci"] = (
        max(0, hit_at_3_score - 1.96 * np.sqrt(hit_at_3_score * (1 - hit_at_3_score) / batch_size)),
        min(1, hit_at_3_score + 1.96 * np.sqrt(hit_at_3_score * (1 - hit_at_3_score) / batch_size)),
    )

    ci_results["ece_ci"] = (
        max(0, ece - 1.96 * 0.1 / np.sqrt(batch_size)),  # Rough approximation
        min(1, ece + 1.96 * 0.1 / np.sqrt(batch_size)),
    )

    return ci_results


def stones_evaluation_summary(metrics: Dict[str, float]) -> Dict[str, str]:
    """Generate Stones evaluation summary.

    Args:
        metrics: Comprehensive metrics dictionary

    Returns:
        Summary with pass/fail status for each Stones threshold
    """
    summary = {}

    # Check each Stones threshold
    summary["f1_macro"] = "PASS" if metrics.get("type_f1_macro", 0) >= 0.5 else "FAIL"
    summary["hit_at_3"] = "PASS" if metrics.get("center_hit_at_3", 0) >= 0.6 else "FAIL"
    summary["joint_success"] = "PASS" if metrics.get("joint_success_strict", 0) >= 0.4 else "FAIL"
    summary["ece"] = "PASS" if metrics.get("ece", 1) < 0.1 else "FAIL"

    # Overall assessment
    all_pass = all(status == "PASS" for status in summary.values())
    summary["overall"] = "PASS" if all_pass else "FAIL"

    return summary
