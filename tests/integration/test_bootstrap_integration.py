"""Integration test for bootstrap CI computation in CLI.

Tests that bootstrap flags work correctly in the train command
and produce expected output format.
"""

import numpy as np
import pytest


def test_bootstrap_functions_integration():
    """Test bootstrap functions work together in realistic workflow."""
    from moola.utils.metrics.bootstrap import (
        bootstrap_accuracy,
        bootstrap_calibration_metrics,
        bootstrap_pointer_metrics,
        format_bootstrap_result,
    )

    # Simulate small validation set (34 samples)
    np.random.seed(42)
    n_samples = 34

    # Classification data
    y_true = np.random.binomial(1, 0.5, size=n_samples)
    y_pred = y_true.copy()
    flip_indices = np.random.choice(n_samples, size=8, replace=False)
    y_pred[flip_indices] = 1 - y_pred[flip_indices]

    # Bootstrap accuracy
    acc_ci = bootstrap_accuracy(y_true, y_pred, n_resamples=500, confidence_level=0.95)

    assert "mean" in acc_ci
    assert "std" in acc_ci
    assert "ci_lower" in acc_ci
    assert "ci_upper" in acc_ci
    assert 0 <= acc_ci["mean"] <= 1
    assert acc_ci["ci_lower"] < acc_ci["mean"] < acc_ci["ci_upper"]

    # Format result
    formatted = format_bootstrap_result("Accuracy", acc_ci)
    assert "Accuracy" in formatted
    assert "95%" in formatted
    assert "CI:" in formatted


def test_bootstrap_pointer_integration():
    """Test bootstrap pointer metrics with realistic data."""
    from moola.utils.metrics.bootstrap import bootstrap_pointer_metrics

    np.random.seed(42)
    n_samples = 34

    # Realistic pointer predictions
    true_start = np.random.randint(10, 40, size=n_samples)
    true_end = true_start + np.random.randint(20, 50, size=n_samples)
    pred_start = true_start + np.random.randint(-5, 6, size=n_samples)
    pred_end = true_end + np.random.randint(-5, 6, size=n_samples)

    ptr_ci = bootstrap_pointer_metrics(
        pred_start=pred_start,
        pred_end=pred_end,
        true_start=true_start,
        true_end=true_end,
        tolerance=3,
        n_resamples=500,
        confidence_level=0.95,
    )

    # Check all expected metrics present
    expected_metrics = [
        "start_mae",
        "end_mae",
        "center_mae",
        "length_mae",
        "hit_at_pm3",
        "hit_at_pm5",
        "exact_match",
    ]

    for metric_name in expected_metrics:
        assert metric_name in ptr_ci
        assert "mean" in ptr_ci[metric_name]
        assert "ci_lower" in ptr_ci[metric_name]
        assert "ci_upper" in ptr_ci[metric_name]


def test_bootstrap_calibration_integration():
    """Test bootstrap calibration metrics with realistic data."""
    from moola.utils.metrics.bootstrap import bootstrap_calibration_metrics

    np.random.seed(42)
    n_samples = 34

    # Realistic calibration data
    probs = np.random.uniform(0.2, 0.8, size=n_samples)
    labels = (probs + np.random.normal(0, 0.15, size=n_samples) > 0.5).astype(int)

    cal_ci = bootstrap_calibration_metrics(
        probs=probs, labels=labels, n_resamples=500, confidence_level=0.95
    )

    # Check ECE and Brier score
    assert "ece" in cal_ci
    assert "brier" in cal_ci

    for metric_name in ["ece", "brier"]:
        assert "mean" in cal_ci[metric_name]
        assert "ci_lower" in cal_ci[metric_name]
        assert "ci_upper" in cal_ci[metric_name]
        assert 0 <= cal_ci[metric_name]["mean"] <= 1


def test_bootstrap_ci_width_small_sample():
    """Test that CI width is appropriately wide for small samples."""
    from moola.utils.metrics.bootstrap import bootstrap_accuracy

    np.random.seed(42)

    # Very small sample (10 samples)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0, 1])

    result = bootstrap_accuracy(y_true, y_pred, n_resamples=500, confidence_level=0.95)

    ci_width = result["ci_upper"] - result["ci_lower"]

    # Very small sample should have wide CI (>20%)
    assert ci_width > 0.20


def test_bootstrap_confidence_levels_comparison():
    """Test that 99% CI is wider than 95% CI."""
    from moola.utils.metrics.bootstrap import bootstrap_accuracy

    np.random.seed(42)
    y_true = np.random.binomial(1, 0.7, size=50)
    y_pred = np.random.binomial(1, 0.7, size=50)

    result_95 = bootstrap_accuracy(y_true, y_pred, n_resamples=500, confidence_level=0.95)
    result_99 = bootstrap_accuracy(y_true, y_pred, n_resamples=500, confidence_level=0.99)

    ci_width_95 = result_95["ci_upper"] - result_95["ci_lower"]
    ci_width_99 = result_99["ci_upper"] - result_99["ci_lower"]

    # 99% CI should be wider
    assert ci_width_99 > ci_width_95


def test_bootstrap_reproducibility():
    """Test that bootstrap results are reproducible with same seed."""
    from moola.utils.metrics.bootstrap import bootstrap_accuracy

    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 1])

    result1 = bootstrap_accuracy(y_true, y_pred, n_resamples=100, confidence_level=0.95)
    result2 = bootstrap_accuracy(y_true, y_pred, n_resamples=100, confidence_level=0.95)

    # Results should be identical (same seed internally)
    assert result1["mean"] == result2["mean"]
    assert result1["ci_lower"] == result2["ci_lower"]
    assert result1["ci_upper"] == result2["ci_upper"]
