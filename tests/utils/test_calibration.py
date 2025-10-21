"""Unit tests for calibration metrics.

Tests smooth ECE, Brier score, and reliability diagram generation.
"""

import numpy as np
import pytest

from moola.utils.metrics.calibration import (
    compute_brier_score,
    compute_calibration_metrics,
    compute_smooth_ece,
    plot_reliability_diagram,
)


def test_perfect_calibration():
    """Test that perfectly calibrated predictions have low ECE."""
    np.random.seed(42)
    n_samples = 1000

    # Generate perfectly calibrated predictions
    probs = np.random.uniform(0, 1, n_samples)
    labels = (np.random.rand(n_samples) < probs).astype(int)

    # With large sample, ECE should be very small
    result = compute_smooth_ece(probs, labels, n_bins=10)
    assert result["ece"] < 0.1, f"ECE {result['ece']:.4f} too high for calibrated data"
    assert "mce" in result
    assert "bin_centers" in result
    assert "bin_accuracies" in result
    assert "bin_confidences" in result
    assert "bin_counts" in result


def test_overconfident_predictions():
    """Test that overconfident predictions have high ECE."""
    n_samples = 1000

    # Overconfident: always predict 0.9, but only 50% correct
    probs = np.full(n_samples, 0.9)
    labels = np.random.binomial(1, 0.5, n_samples)

    result = compute_smooth_ece(probs, labels, n_bins=10)
    assert result["ece"] > 0.2, f"ECE {result['ece']:.4f} too low for overconfident predictions"


def test_underconfident_predictions():
    """Test that underconfident predictions have high ECE."""
    n_samples = 1000

    # Underconfident: always predict 0.5, but 90% are correct
    probs = np.full(n_samples, 0.5)
    labels = np.random.binomial(1, 0.9, n_samples)

    result = compute_smooth_ece(probs, labels, n_bins=10)
    # Should have high ECE due to underconfidence
    assert result["ece"] > 0.15, f"ECE {result['ece']:.4f} too low for underconfident predictions"


def test_brier_score_perfect():
    """Test Brier score computation for perfect predictions."""
    # Perfect predictions
    probs_perfect = np.array([0.0, 1.0, 0.0, 1.0])
    labels = np.array([0, 1, 0, 1])
    brier_perfect = compute_brier_score(probs_perfect, labels)
    assert brier_perfect == 0.0, "Brier score should be 0 for perfect predictions"


def test_brier_score_random():
    """Test Brier score for random predictions."""
    # Random predictions (0.5 for all)
    probs_random = np.full(4, 0.5)
    labels = np.array([0, 1, 0, 1])
    brier_random = compute_brier_score(probs_random, labels)
    assert brier_random == 0.25, "Brier score should be 0.25 for 0.5 predictions"


def test_brier_score_worst():
    """Test Brier score for worst-case predictions."""
    # Worst predictions (completely wrong)
    probs_worst = np.array([1.0, 0.0, 1.0, 0.0])
    labels = np.array([0, 1, 0, 1])
    brier_worst = compute_brier_score(probs_worst, labels)
    assert brier_worst == 1.0, "Brier score should be 1.0 for completely wrong predictions"


def test_calibration_metrics_comprehensive():
    """Test comprehensive calibration metrics."""
    np.random.seed(42)
    n_samples = 500

    probs = np.random.uniform(0, 1, n_samples)
    labels = (np.random.rand(n_samples) < probs).astype(int)

    metrics = compute_calibration_metrics(probs, labels, n_bins=10)

    assert "ece" in metrics
    assert "mce" in metrics
    assert "brier_score" in metrics
    assert "accuracy" in metrics
    assert "ece_details" in metrics
    assert 0 <= metrics["ece"] <= 1
    assert 0 <= metrics["mce"] <= 1
    assert 0 <= metrics["brier_score"] <= 1
    assert 0 <= metrics["accuracy"] <= 1


def test_calibration_metrics_multiclass():
    """Test calibration metrics with multi-class probabilities."""
    np.random.seed(42)
    n_samples = 200
    n_classes = 3

    # Generate random multi-class probabilities
    probs = np.random.dirichlet(np.ones(n_classes), n_samples)
    # Generate labels based on probabilities
    labels = np.array([np.random.choice(n_classes, p=p) for p in probs])

    metrics = compute_calibration_metrics(probs, labels, n_bins=10)

    assert "ece" in metrics
    assert "mce" in metrics
    assert "brier_score" in metrics
    assert 0 <= metrics["ece"] <= 1
    assert 0 <= metrics["brier_score"] <= 1


def test_smooth_ece_gaussian_smoothing():
    """Test smooth ECE with gaussian smoothing."""
    np.random.seed(42)
    n_samples = 300

    probs = np.random.uniform(0, 1, n_samples)
    labels = (np.random.rand(n_samples) < probs).astype(int)

    result = compute_smooth_ece(probs, labels, n_bins=10, smoothing="gaussian", bandwidth=0.1)
    assert "ece" in result
    assert result["ece"] >= 0


def test_smooth_ece_uniform_smoothing():
    """Test smooth ECE with uniform smoothing."""
    np.random.seed(42)
    n_samples = 300

    probs = np.random.uniform(0, 1, n_samples)
    labels = (np.random.rand(n_samples) < probs).astype(int)

    result = compute_smooth_ece(probs, labels, n_bins=10, smoothing="uniform", bandwidth=0.1)
    assert "ece" in result
    assert result["ece"] >= 0


def test_smooth_ece_invalid_smoothing():
    """Test that invalid smoothing method raises ValueError."""
    probs = np.array([0.3, 0.7, 0.5])
    labels = np.array([0, 1, 1])

    with pytest.raises(ValueError, match="Unknown smoothing method"):
        compute_smooth_ece(probs, labels, smoothing="invalid")


def test_reliability_diagram_creation():
    """Test that reliability diagram can be created."""
    np.random.seed(42)
    n_samples = 200

    probs = np.random.uniform(0, 1, n_samples)
    labels = (np.random.rand(n_samples) < probs).astype(int)

    fig = plot_reliability_diagram(probs, labels, n_bins=10)
    assert fig is not None
    assert len(fig.axes) > 0
    # Check that axes were configured
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Confidence"
    assert ax.get_ylabel() == "Accuracy"
    assert ax.get_title() == "Reliability Diagram"


def test_reliability_diagram_save(tmp_path):
    """Test saving reliability diagram to file."""
    np.random.seed(42)
    n_samples = 200

    probs = np.random.uniform(0, 1, n_samples)
    labels = (np.random.rand(n_samples) < probs).astype(int)

    save_path = tmp_path / "reliability.png"
    fig = plot_reliability_diagram(probs, labels, n_bins=10, save_path=str(save_path))

    assert fig is not None
    assert save_path.exists()


def test_ece_edge_case_all_zeros():
    """Test ECE with all predictions at 0."""
    probs = np.zeros(100)
    labels = np.zeros(100, dtype=int)

    result = compute_smooth_ece(probs, labels, n_bins=10)
    # Should be perfectly calibrated (always predicts 0, always correct)
    assert result["ece"] < 0.1


def test_ece_edge_case_all_ones():
    """Test ECE with all predictions at 1."""
    probs = np.ones(100)
    labels = np.ones(100, dtype=int)

    result = compute_smooth_ece(probs, labels, n_bins=10)
    # Should be perfectly calibrated (always predicts 1, always correct)
    assert result["ece"] < 0.1


def test_ece_edge_case_small_sample():
    """Test ECE with very small sample size."""
    probs = np.array([0.3, 0.7, 0.5])
    labels = np.array([0, 1, 1])

    result = compute_smooth_ece(probs, labels, n_bins=5)
    assert "ece" in result
    assert 0 <= result["ece"] <= 1


def test_calibration_metrics_binary():
    """Test calibration metrics explicitly for binary classification."""
    np.random.seed(42)
    n_samples = 300

    # Binary classification probabilities
    probs = np.random.uniform(0, 1, n_samples)
    labels = (np.random.rand(n_samples) < 0.6).astype(int)  # 60% class 1

    metrics = compute_calibration_metrics(probs, labels, n_bins=15)

    assert "ece" in metrics
    assert "mce" in metrics
    assert "brier_score" in metrics
    assert "accuracy" in metrics
    assert 0 <= metrics["ece"] <= 1
    assert 0 <= metrics["brier_score"] <= 1
