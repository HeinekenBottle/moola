"""Calibration metrics for probability predictions.

Implements smooth Expected Calibration Error (ECE), Brier score, and reliability diagrams
for assessing and visualizing model calibration quality.
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def compute_smooth_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    smoothing: str = "gaussian",
    bandwidth: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Compute smooth Expected Calibration Error.

    Uses kernel smoothing instead of hard binning for more robust calibration metrics.

    Args:
        probs: Predicted probabilities for positive class (n_samples,)
        labels: True binary labels (n_samples,)
        n_bins: Number of bins for calibration curve
        smoothing: Smoothing method ('gaussian' or 'uniform')
        bandwidth: Kernel bandwidth for smoothing

    Returns:
        Dictionary with ECE, MCE (max calibration error), and bin statistics
    """
    # Sort by predicted probability
    sorted_indices = np.argsort(probs)
    sorted_probs = probs[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Create bin centers
    bin_centers = np.linspace(0, 1, n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i, center in enumerate(bin_centers):
        # Compute kernel weights
        if smoothing == "gaussian":
            weights = np.exp(-((sorted_probs - center) ** 2) / (2 * bandwidth**2))
        elif smoothing == "uniform":
            weights = (np.abs(sorted_probs - center) <= bandwidth).astype(float)
        else:
            raise ValueError(f"Unknown smoothing method: {smoothing}")

        weights = weights / (weights.sum() + 1e-10)  # Normalize

        # Weighted statistics
        bin_accuracies[i] = (weights * sorted_labels).sum()
        bin_confidences[i] = (weights * sorted_probs).sum()
        bin_counts[i] = weights.sum()

    # Compute ECE: weighted average of |accuracy - confidence|
    calibration_errors = np.abs(bin_accuracies - bin_confidences)
    weights = bin_counts / bin_counts.sum()
    ece = (weights * calibration_errors).sum()

    # Max calibration error
    mce = calibration_errors.max()

    return {
        "ece": ece,
        "mce": mce,
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute Brier score (mean squared error of probabilities).

    Args:
        probs: Predicted probabilities for positive class (n_samples,)
        labels: True binary labels (n_samples,)

    Returns:
        Brier score (lower is better, range [0, 1])
    """
    return np.mean((probs - labels) ** 2)


def compute_calibration_metrics(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> Dict[str, float]:
    """Compute comprehensive calibration metrics.

    Args:
        probs: Predicted probabilities (n_samples,) or (n_samples, n_classes)
        labels: True labels (n_samples,)
        n_bins: Number of bins for ECE

    Returns:
        Dictionary with ECE, MCE, Brier score, accuracy
    """
    # Handle multi-class: extract probability of predicted class
    if len(probs.shape) == 2:
        pred_classes = probs.argmax(axis=1)
        probs_1d = probs[np.arange(len(probs)), pred_classes]
    else:
        probs_1d = probs
        pred_classes = (probs > 0.5).astype(int)

    # Binary labels for calibration
    if len(np.unique(labels)) > 2:
        # Multi-class: convert to binary (predicted class vs others)
        if len(probs.shape) == 2:
            pred_classes = probs.argmax(axis=1)
        binary_labels = (labels == pred_classes).astype(int)
    else:
        binary_labels = labels

    ece_results = compute_smooth_ece(probs_1d, binary_labels, n_bins=n_bins)
    brier = compute_brier_score(probs_1d, binary_labels)
    accuracy = (binary_labels == 1).mean()  # Fraction correct

    return {
        "ece": float(ece_results["ece"]),
        "mce": float(ece_results["mce"]),
        "brier_score": float(brier),
        "accuracy": float(accuracy),
        "ece_details": ece_results,
    }


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot reliability diagram (calibration curve).

    Args:
        probs: Predicted probabilities for positive class (n_samples,)
        labels: True binary labels (n_samples,)
        n_bins: Number of bins
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    ece_results = compute_smooth_ece(probs, labels, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=2)

    # Plot actual calibration curve
    bin_centers = ece_results["bin_centers"]
    bin_accuracies = ece_results["bin_accuracies"]
    bin_confidences = ece_results["bin_confidences"]
    bin_counts = ece_results["bin_counts"]

    # Filter bins with sufficient samples
    valid = bin_counts > 0.01

    ax.plot(
        bin_confidences[valid],
        bin_accuracies[valid],
        "o-",
        label=f'Model (ECE={ece_results["ece"]:.4f})',
        linewidth=2,
        markersize=8,
    )

    # Add confidence bars based on bin counts
    bar_widths = 0.05
    ax.bar(
        bin_centers[valid],
        bin_counts[valid] / bin_counts.max() * 0.1,
        width=bar_widths,
        alpha=0.3,
        label="Sample density",
    )

    ax.set_xlabel("Confidence", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_title("Reliability Diagram", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
