"""Unit tests for bootstrap confidence interval computation.

Tests bootstrap resampling and CI computation for classification, pointer regression,
and calibration metrics on small datasets.
"""

import pytest
import numpy as np
from moola.utils.metrics.bootstrap import (
    bootstrap_resample,
    bootstrap_metric,
    bootstrap_accuracy,
    bootstrap_pointer_metrics,
    bootstrap_calibration_metrics,
    format_bootstrap_result,
)


def test_bootstrap_resample():
    """Test bootstrap resampling generates correct shape and samples with replacement."""
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])

    resampled_true, resampled_pred = bootstrap_resample(
        y_true, y_pred, n_resamples=100, random_seed=42
    )

    # Check shape
    assert resampled_true.shape == (100, 5)
    assert resampled_pred.shape == (100, 5)

    # Check that resampling with replacement works (some indices should repeat)
    # At least one resample should be different from original
    different_count = 0
    for i in range(100):
        if not np.array_equal(resampled_true[i], y_true):
            different_count += 1

    assert different_count > 0, "Bootstrap should create different samples"


def test_bootstrap_resample_reproducibility():
    """Test bootstrap resampling is reproducible with same seed."""
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])

    resampled1_true, resampled1_pred = bootstrap_resample(
        y_true, y_pred, n_resamples=10, random_seed=42
    )
    resampled2_true, resampled2_pred = bootstrap_resample(
        y_true, y_pred, n_resamples=10, random_seed=42
    )

    assert np.array_equal(resampled1_true, resampled2_true)
    assert np.array_equal(resampled1_pred, resampled2_pred)


def test_bootstrap_metric():
    """Test generic bootstrap_metric wrapper."""

    def accuracy(y_t, y_p):
        return (y_t == y_p).mean()

    np.random.seed(42)
    y_true = np.random.binomial(1, 0.7, size=100)
    y_pred = np.random.binomial(1, 0.7, size=100)

    result = bootstrap_metric(y_true, y_pred, accuracy, n_resamples=100, confidence_level=0.95)

    # Check all keys present
    assert "mean" in result
    assert "std" in result
    assert "median" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "confidence_level" in result

    # Check CI bounds are valid
    assert result["ci_lower"] < result["mean"] < result["ci_upper"]
    assert 0 <= result["mean"] <= 1  # Accuracy is in [0, 1]


def test_bootstrap_accuracy():
    """Test bootstrap accuracy CI computation."""
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.7, size=100)
    y_pred = np.random.binomial(1, 0.7, size=100)

    result = bootstrap_accuracy(y_true, y_pred, n_resamples=100, confidence_level=0.95)

    assert "mean" in result
    assert "std" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert result["ci_lower"] < result["mean"] < result["ci_upper"]
    assert 0 <= result["mean"] <= 1


def test_bootstrap_accuracy_perfect():
    """Test bootstrap accuracy with perfect predictions."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])

    result = bootstrap_accuracy(y_true, y_pred, n_resamples=50, confidence_level=0.95)

    # Perfect accuracy should have mean = 1.0, tight CI
    assert result["mean"] == 1.0
    assert result["std"] == 0.0
    assert result["ci_lower"] == 1.0
    assert result["ci_upper"] == 1.0


def test_bootstrap_accuracy_small_sample():
    """Test bootstrap accuracy on small sample (simulates 34 validation samples)."""
    np.random.seed(42)
    # Simulate 34 samples with ~70% accuracy
    y_true = np.array([1, 0, 1, 1, 0] * 6 + [1, 0, 1, 0])  # 34 samples
    y_pred = np.array([1, 0, 1, 1, 1] * 6 + [1, 1, 1, 0])  # ~70% correct

    result = bootstrap_accuracy(y_true, y_pred, n_resamples=1000, confidence_level=0.95)

    # Check CI is reasonable width for small sample
    ci_width = result["ci_upper"] - result["ci_lower"]
    assert 0.1 < ci_width < 0.4  # Small sample should have wider CI


def test_bootstrap_pointer_metrics():
    """Test bootstrap CI for pointer regression metrics."""
    # Create synthetic pointer predictions
    np.random.seed(42)
    n_samples = 50
    true_start = np.random.randint(10, 40, size=n_samples)
    true_end = true_start + np.random.randint(20, 50, size=n_samples)

    # Add noise to predictions
    pred_start = true_start + np.random.randint(-5, 5, size=n_samples)
    pred_end = true_end + np.random.randint(-5, 5, size=n_samples)

    result = bootstrap_pointer_metrics(
        pred_start=pred_start,
        pred_end=pred_end,
        true_start=true_start,
        true_end=true_end,
        tolerance=3,
        n_resamples=100,
        confidence_level=0.95,
    )

    # Check all metrics present
    expected_metrics = [
        "start_mae",
        "end_mae",
        "center_mae",
        "length_mae",
        "hit_at_pm3",
        "hit_at_pm5",
        "exact_match",
    ]
    for metric in expected_metrics:
        assert metric in result
        assert "mean" in result[metric]
        assert "std" in result[metric]
        assert "ci_lower" in result[metric]
        assert "ci_upper" in result[metric]
        assert result[metric]["ci_lower"] <= result[metric]["mean"] <= result[metric]["ci_upper"]


def test_bootstrap_calibration_metrics():
    """Test bootstrap CI for calibration metrics."""
    # Create synthetic calibration data
    np.random.seed(42)
    n_samples = 100
    probs = np.random.uniform(0, 1, size=n_samples)
    labels = (probs > 0.5).astype(int)  # Somewhat calibrated

    result = bootstrap_calibration_metrics(
        probs=probs, labels=labels, n_resamples=100, confidence_level=0.95
    )

    # Check all metrics present
    expected_metrics = ["ece", "mce", "brier"]
    for metric in expected_metrics:
        assert metric in result
        assert "mean" in result[metric]
        assert "std" in result[metric]
        assert "ci_lower" in result[metric]
        assert "ci_upper" in result[metric]
        assert result[metric]["ci_lower"] <= result[metric]["mean"] <= result[metric]["ci_upper"]

    # Check ECE is in valid range
    assert 0 <= result["ece"]["mean"] <= 1


def test_format_bootstrap_result():
    """Test formatting of bootstrap results."""
    result = {
        "mean": 0.8523,
        "std": 0.0342,
        "ci_lower": 0.7845,
        "ci_upper": 0.9201,
        "confidence_level": 0.95,
    }

    formatted = format_bootstrap_result("Accuracy", result)

    assert "Accuracy" in formatted
    assert "0.8523" in formatted
    assert "0.7845" in formatted
    assert "0.9201" in formatted
    assert "95%" in formatted


def test_format_bootstrap_result_99_percent():
    """Test formatting with 99% confidence level."""
    result = {
        "mean": 0.7234,
        "std": 0.0521,
        "ci_lower": 0.6012,
        "ci_upper": 0.8456,
        "confidence_level": 0.99,
    }

    formatted = format_bootstrap_result("Hit@±3", result)

    assert "Hit@±3" in formatted
    assert "0.7234" in formatted
    assert "99%" in formatted


def test_bootstrap_metric_with_invalid_resamples():
    """Test bootstrap handles invalid resamples gracefully (e.g., all same class)."""

    def precision_positive_class(y_t, y_p):
        """Precision for positive class - may fail if no positives predicted."""
        tp = np.sum((y_t == 1) & (y_p == 1))
        fp = np.sum((y_t == 0) & (y_p == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Heavily imbalanced dataset - some resamples may have all same class
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    result = bootstrap_metric(
        y_true, y_pred, precision_positive_class, n_resamples=100, confidence_level=0.95
    )

    # Should still return valid result even if some resamples fail
    assert "mean" in result
    assert "ci_lower" in result
    assert "ci_upper" in result


def test_bootstrap_confidence_levels():
    """Test different confidence levels produce different CI widths."""
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.7, size=50)
    y_pred = np.random.binomial(1, 0.7, size=50)

    result_95 = bootstrap_accuracy(y_true, y_pred, n_resamples=500, confidence_level=0.95)
    result_99 = bootstrap_accuracy(y_true, y_pred, n_resamples=500, confidence_level=0.99)

    ci_width_95 = result_95["ci_upper"] - result_95["ci_lower"]
    ci_width_99 = result_99["ci_upper"] - result_99["ci_lower"]

    # 99% CI should be wider than 95% CI
    assert ci_width_99 > ci_width_95
