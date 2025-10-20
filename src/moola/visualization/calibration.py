"""Calibration and reliability visualization."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def save_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: str,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    figsize: tuple = (8, 8),
) -> None:
    """Generate and save reliability (calibration) diagram.

    Args:
        y_true: True labels (shape: N,)
        y_proba: Predicted probabilities (shape: N, n_classes)
        output_path: Path to save the plot
        n_bins: Number of confidence bins
        title: Plot title
        figsize: Figure size
    """
    # Get predicted class and confidence
    y_pred = y_proba.argmax(axis=1)
    confidence = y_proba.max(axis=1)

    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        in_bin = (confidence > bin_lower) & (confidence <= bin_upper)

        if in_bin.sum() > 0:
            acc_in_bin = (y_pred[in_bin] == y_true[in_bin]).mean()
            conf_in_bin = confidence[in_bin].mean()
            count_in_bin = in_bin.sum()

            bin_accs.append(acc_in_bin)
            bin_confs.append(conf_in_bin)
            bin_counts.append(count_in_bin)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.3, linewidth=2)

    # Reliability diagram (bar plot)
    if bin_confs:
        bars = ax.bar(
            bin_confs,
            bin_accs,
            width=1.0 / n_bins,
            alpha=0.6,
            edgecolor='black',
            label='Model calibration',
        )

        # Color bars by count
        max_count = max(bin_counts) if bin_counts else 1
        for bar, count in zip(bars, bin_counts):
            normalized_count = count / max_count
            bar.set_color(plt.cm.Blues(0.3 + 0.6 * normalized_count))

    # Plot calibration curve (line)
    if bin_confs:
        ax.plot(bin_confs, bin_accs, 'ro-', markersize=8, linewidth=2, alpha=0.8)

    # Calculate ECE
    ece = 0.0
    for acc, conf, count in zip(bin_accs, bin_confs, bin_counts):
        prop = count / len(y_true)
        ece += np.abs(conf - acc) * prop

    # Labels and formatting
    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'{title}\nECE = {ece:.4f}', fontsize=16)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')

    # Add sample count annotation
    total_samples = len(y_true)
    ax.text(
        0.98, 0.02,
        f'n = {total_samples}',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Reliability diagram saved: {output_path}")
