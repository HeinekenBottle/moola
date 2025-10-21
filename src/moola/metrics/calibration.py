"""Calibration metrics for model confidence evaluation.

Computes Expected Calibration Error (ECE) and reliability diagrams.
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss


def expected_calibration_error(
    probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10
) -> Tuple[float, List[float], List[float]]:
    """Compute Expected Calibration Error (ECE).

    Args:
        probs: Predicted probabilities [batch, n_classes]
        targets: True targets [batch]
        n_bins: Number of confidence bins

    Returns:
        Tuple of (ece, bin_confidences, bin_accuracies)
    """
    confidence, pred_class = probs.max(dim=1)
    accuracy = (pred_class == targets).float()

    # Create bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_confidences = []
    bin_accuracies = []
    ece = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            # Compute accuracy and confidence in this bin
            accuracy_in_bin = accuracy[in_bin].mean()
            avg_confidence_in_bin = confidence[in_bin].mean()

            bin_confidences.append(float(avg_confidence_in_bin))
            bin_accuracies.append(float(accuracy_in_bin))

            # Add to ECE
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        else:
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)

    return float(ece), bin_confidences, bin_accuracies


def adaptive_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    """Compute adaptive ECE with equal-sized bins.

    Args:
        probs: Predicted probabilities [batch, n_classes]
        targets: True targets [batch]
        n_bins: Number of bins

    Returns:
        Adaptive ECE score
    """
    confidence, pred_class = probs.max(dim=1)
    accuracy = (pred_class == targets).float()

    # Sort by confidence
    sorted_conf, sorted_indices = torch.sort(confidence)
    sorted_acc = accuracy[sorted_indices]

    # Create equal-sized bins
    bin_size = len(confidence) // n_bins
    ece = 0.0

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(confidence)

        if end_idx > start_idx:
            bin_conf = sorted_conf[start_idx:end_idx].mean()
            bin_acc = sorted_acc[start_idx:end_idx].mean()
            bin_weight = (end_idx - start_idx) / len(confidence)

            ece += torch.abs(bin_conf - bin_acc) * bin_weight

    return float(ece)


def brier_score(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Brier score for calibration.

    Args:
        probs: Predicted probabilities [batch, n_classes]
        targets: True targets [batch]

    Returns:
        Brier score
    """
    # Convert to one-hot encoding
    n_classes = probs.shape[1]
    targets_onehot = F.one_hot(targets, n_classes).float()

    # Compute Brier score
    brier = ((probs - targets_onehot) ** 2).mean()
    return float(brier)


def reliability_diagram(
    probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10, save_path: str = None
) -> Tuple[plt.Figure, np.ndarray, np.ndarray]:
    """Create reliability diagram for calibration visualization.

    Args:
        probs: Predicted probabilities [batch, n_classes]
        targets: True targets [batch]
        n_bins: Number of bins
        save_path: Path to save the plot

    Returns:
        Tuple of (figure, bin_confidences, bin_accuracies)
    """
    ece, bin_confidences, bin_accuracies = expected_calibration_error(probs, targets, n_bins)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    # Plot reliability diagram
    bin_centers = np.linspace(0.05, 0.95, n_bins)
    ax.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, label=f"Model (ECE={ece:.3f})")

    # Plot confidence points
    ax.scatter(bin_centers, bin_confidences, color="red", s=50, label="Avg Confidence", zorder=5)

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, np.array(bin_confidences), np.array(bin_accuracies)


def compute_calibration_metrics(
    probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10
) -> dict:
    """Compute comprehensive calibration metrics.

    Args:
        probs: Predicted probabilities [batch, n_classes]
        targets: True targets [batch]
        n_bins: Number of bins

    Returns:
        Dictionary of calibration metrics
    """
    # Standard ECE
    ece, _, _ = expected_calibration_error(probs, targets, n_bins)

    # Adaptive ECE
    adaptive_ece_score = adaptive_ece(probs, targets, n_bins)

    # Brier score
    brier = brier_score(probs, targets)

    # Maximum Calibration Error
    confidence, pred_class = probs.max(dim=1)
    accuracy = (pred_class == targets).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    max_ce = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
        if in_bin.sum() > 0:
            accuracy_in_bin = accuracy[in_bin].mean()
            avg_confidence_in_bin = confidence[in_bin].mean()
            max_ce = max(max_ce, float(torch.abs(avg_confidence_in_bin - accuracy_in_bin)))

    return {
        "ece": ece,
        "adaptive_ece": adaptive_ece_score,
        "brier_score": brier,
        "max_calibration_error": max_ce,
        "n_bins": n_bins,
        "over_confident": ece > 0.1,  # Stones threshold
        "well_calibrated": ece < 0.05,
        "acceptable": ece < 0.1,  # Stones threshold
    }
