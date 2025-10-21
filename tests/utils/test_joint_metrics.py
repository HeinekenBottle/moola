"""Unit tests for joint metrics (pointer + type classification)."""

import numpy as np
import pytest

from moola.utils.metrics.joint_metrics import (
    compute_joint_hit_accuracy,
    compute_joint_hit_f1,
    compute_task_contribution_analysis,
    select_best_model_by_joint_metric,
)


def test_joint_hit_accuracy_perfect():
    """Test joint accuracy with perfect predictions."""
    n = 50
    pred_start = np.arange(n)
    pred_end = np.arange(n) + 10
    true_start = np.arange(n)
    true_end = np.arange(n) + 10
    pred_type = np.zeros(n, dtype=int)
    true_type = np.zeros(n, dtype=int)

    result = compute_joint_hit_accuracy(
        pred_start, pred_end, true_start, true_end, pred_type, true_type, tolerance=3
    )

    assert result["joint_accuracy"] == 1.0
    assert result["pointer_hit_rate"] == 1.0
    assert result["type_accuracy"] == 1.0
    assert result["joint_correct_count"] == n
    assert result["total_samples"] == n


def test_joint_hit_accuracy_partial():
    """Test joint accuracy with partial correctness."""
    # Half correct pointers, half correct types, quarter both correct
    pred_start = np.array([0, 0, 10, 10])
    pred_end = np.array([10, 10, 20, 20])
    true_start = np.array([0, 0, 0, 0])
    true_end = np.array([10, 10, 10, 10])
    pred_type = np.array([0, 1, 0, 1])
    true_type = np.array([0, 0, 0, 0])

    result = compute_joint_hit_accuracy(
        pred_start, pred_end, true_start, true_end, pred_type, true_type, tolerance=3
    )

    assert result["pointer_hit_rate"] == 0.5  # First 2 correct
    assert result["type_accuracy"] == 0.5  # First and third correct
    assert result["joint_accuracy"] == 0.25  # Only first correct
    assert result["joint_correct_count"] == 1
    assert result["total_samples"] == 4


def test_joint_hit_accuracy_tolerance():
    """Test joint accuracy respects tolerance parameter."""
    # First sample: both start/end within tolerance (±3)
    # Second sample: start within tolerance, end outside tolerance
    pred_start = np.array([2, 2])
    pred_end = np.array([12, 17])  # Second end is +5 bars off (outside ±3)
    true_start = np.array([0, 0])
    true_end = np.array([10, 12])  # Second sample end difference is +5
    pred_type = np.array([0, 0])
    true_type = np.array([0, 0])

    result = compute_joint_hit_accuracy(
        pred_start, pred_end, true_start, true_end, pred_type, true_type, tolerance=3
    )

    # Only first sample should be a pointer hit (both start and end within ±3)
    assert result["pointer_hit_rate"] == 0.5
    assert result["type_accuracy"] == 1.0
    assert result["joint_accuracy"] == 0.5


def test_joint_hit_f1_perfect():
    """Test joint F1 with perfect predictions."""
    n = 50
    pred_start = np.arange(n)
    pred_end = np.arange(n) + 10
    true_start = np.arange(n)
    true_end = np.arange(n) + 10
    pred_type = np.zeros(n, dtype=int)
    true_type = np.zeros(n, dtype=int)

    result = compute_joint_hit_f1(
        pred_start,
        pred_end,
        true_start,
        true_end,
        pred_type,
        true_type,
        tolerance=3,
        average="weighted",
    )

    assert result["joint_f1"] == 1.0
    assert result["joint_precision"] == 1.0
    assert result["joint_recall"] == 1.0
    assert result["average"] == "weighted"


def test_joint_hit_f1_partial():
    """Test joint F1 with partial predictions."""
    # Mix of correct and incorrect predictions
    pred_start = np.array([0, 0, 10, 10, 0])
    pred_end = np.array([10, 10, 20, 20, 10])
    true_start = np.array([0, 0, 0, 0, 0])
    true_end = np.array([10, 10, 10, 10, 10])
    pred_type = np.array([0, 1, 0, 1, 0])
    true_type = np.array([0, 0, 1, 1, 0])

    result = compute_joint_hit_f1(
        pred_start,
        pred_end,
        true_start,
        true_end,
        pred_type,
        true_type,
        tolerance=3,
        average="weighted",
    )

    # Should have non-zero but imperfect F1
    assert 0 < result["joint_f1"] < 1.0
    assert 0 < result["joint_precision"] < 1.0
    assert 0 < result["joint_recall"] < 1.0


def test_task_contribution_analysis():
    """Test task contribution breakdown."""
    pred_start = np.array([0, 0, 10, 10])
    pred_end = np.array([10, 10, 20, 20])
    true_start = np.array([0, 0, 0, 0])
    true_end = np.array([10, 10, 10, 10])
    pred_type = np.array([0, 1, 0, 1])
    true_type = np.array([0, 0, 0, 0])

    result = compute_task_contribution_analysis(
        pred_start, pred_end, true_start, true_end, pred_type, true_type, tolerance=3
    )

    assert result["both_correct"]["fraction"] == 0.25  # Sample 0
    assert result["both_correct"]["count"] == 1
    assert result["pointer_only_correct"]["fraction"] == 0.25  # Sample 1
    assert result["pointer_only_correct"]["count"] == 1
    assert result["type_only_correct"]["fraction"] == 0.25  # Sample 2
    assert result["type_only_correct"]["count"] == 1
    assert result["both_wrong"]["fraction"] == 0.25  # Sample 3
    assert result["both_wrong"]["count"] == 1
    assert result["total_samples"] == 4


def test_task_contribution_all_correct():
    """Test task contribution when everything is correct."""
    n = 20
    pred_start = np.arange(n)
    pred_end = np.arange(n) + 10
    true_start = np.arange(n)
    true_end = np.arange(n) + 10
    pred_type = np.zeros(n, dtype=int)
    true_type = np.zeros(n, dtype=int)

    result = compute_task_contribution_analysis(
        pred_start, pred_end, true_start, true_end, pred_type, true_type, tolerance=3
    )

    assert result["both_correct"]["fraction"] == 1.0
    assert result["both_correct"]["count"] == n
    assert result["pointer_only_correct"]["fraction"] == 0.0
    assert result["type_only_correct"]["fraction"] == 0.0
    assert result["both_wrong"]["fraction"] == 0.0


def test_select_best_model_by_joint_metric():
    """Test model selection based on joint metrics."""
    results = [
        {
            "joint_f1": 0.5,
            "joint_accuracy": 0.45,
            "pointer_hit_rate": 0.6,
            "type_accuracy": 0.7,
        },
        {
            "joint_f1": 0.7,
            "joint_accuracy": 0.65,
            "pointer_hit_rate": 0.75,
            "type_accuracy": 0.8,
        },
        {
            "joint_f1": 0.6,
            "joint_accuracy": 0.55,
            "pointer_hit_rate": 0.65,
            "type_accuracy": 0.75,
        },
    ]

    best_result, best_idx = select_best_model_by_joint_metric(
        results, metric_name="joint_f1", min_pointer_hit=0.4, min_type_acc=0.5
    )

    assert best_idx == 1  # Second model has highest joint_f1
    assert best_result["joint_f1"] == 0.7


def test_select_best_model_with_thresholds():
    """Test model selection with minimum threshold filtering."""
    results = [
        {
            "joint_f1": 0.9,
            "joint_accuracy": 0.85,
            "pointer_hit_rate": 0.3,  # Too low!
            "type_accuracy": 0.95,
        },
        {
            "joint_f1": 0.7,
            "joint_accuracy": 0.65,
            "pointer_hit_rate": 0.75,
            "type_accuracy": 0.8,
        },
    ]

    # First model should be filtered out due to low pointer_hit_rate
    best_result, best_idx = select_best_model_by_joint_metric(
        results, metric_name="joint_f1", min_pointer_hit=0.4, min_type_acc=0.5
    )

    assert best_idx == 1  # Only second model meets thresholds


def test_select_best_model_no_valid_results():
    """Test model selection when no results meet thresholds."""
    results = [
        {
            "joint_f1": 0.5,
            "joint_accuracy": 0.45,
            "pointer_hit_rate": 0.2,  # Too low
            "type_accuracy": 0.3,  # Too low
        }
    ]

    with pytest.raises(ValueError, match="No results meet minimum thresholds"):
        select_best_model_by_joint_metric(
            results, metric_name="joint_f1", min_pointer_hit=0.4, min_type_acc=0.5
        )


def test_joint_hit_accuracy_different_types():
    """Test joint accuracy with binary type classification."""
    pred_start = np.array([0, 10, 20, 30])
    pred_end = np.array([10, 20, 30, 40])
    true_start = np.array([1, 11, 21, 31])  # All within tolerance
    true_end = np.array([11, 21, 31, 41])
    pred_type = np.array([0, 0, 1, 1])
    true_type = np.array([0, 1, 0, 1])

    result = compute_joint_hit_accuracy(
        pred_start, pred_end, true_start, true_end, pred_type, true_type, tolerance=3
    )

    # All pointers correct, half types correct
    assert result["pointer_hit_rate"] == 1.0
    assert result["type_accuracy"] == 0.5
    assert result["joint_accuracy"] == 0.5  # Samples 0 and 3


def test_joint_hit_f1_macro_average():
    """Test joint F1 with macro averaging."""
    pred_start = np.array([0, 0, 10, 10])
    pred_end = np.array([10, 10, 20, 20])
    true_start = np.array([0, 0, 0, 0])
    true_end = np.array([10, 10, 10, 10])
    pred_type = np.array([0, 1, 0, 1])
    true_type = np.array([0, 0, 1, 1])

    result = compute_joint_hit_f1(
        pred_start,
        pred_end,
        true_start,
        true_end,
        pred_type,
        true_type,
        tolerance=3,
        average="macro",
    )

    assert result["average"] == "macro"
    assert 0 <= result["joint_f1"] <= 1.0
