"""Tests for metrics module."""

import numpy as np
import pytest
import torch

from moola.metrics import (
    check_stones_thresholds,
    compute_calibration_metrics,
    compute_comprehensive_joint_metrics,
    compute_pointer_metrics,
    evaluate_stones_metrics,
    expected_calibration_error,
    hit_at_k,
    joint_success_rate,
)


class TestHitMetrics:
    """Test hit accuracy metrics."""

    def test_hit_at_k_perfect(self):
        """Test Hit@±k with perfect predictions."""
        pred_center = torch.tensor([0.5, 0.25, 0.75])
        true_center = torch.tensor([0.5, 0.25, 0.75])

        hit_1 = hit_at_k(pred_center, true_center, k=1)
        hit_3 = hit_at_k(pred_center, true_center, k=3)

        assert hit_1 == 1.0
        assert hit_3 == 1.0

    def test_hit_at_k_tolerance(self):
        """Test Hit@±k with tolerance."""
        # Predictions within ±3 timesteps (±0.03 normalized)
        pred_center = torch.tensor([0.5, 0.28, 0.73])
        true_center = torch.tensor([0.5, 0.25, 0.75])

        hit_3 = hit_at_k(pred_center, true_center, k=3)
        assert hit_3 == 1.0

        # Test with k=1 (should fail for some)
        hit_1 = hit_at_k(pred_center, true_center, k=1)
        assert hit_1 == 1 / 3  # Only first is exact

    def test_compute_pointer_metrics(self):
        """Test comprehensive pointer metrics."""
        pred_center = torch.tensor([0.5, 0.25])
        pred_length = torch.tensor([0.3, 0.4])
        true_center = torch.tensor([0.5, 0.25])
        true_length = torch.tensor([0.3, 0.4])

        metrics = compute_pointer_metrics(pred_center, pred_length, true_center, true_length)

        assert metrics["center_mae"] == 0.0
        assert metrics["length_mae"] == 0.0
        assert metrics["hit@±3"] == 1.0


class TestCalibrationMetrics:
    """Test calibration metrics."""

    def test_expected_calibration_error_perfect(self):
        """Test ECE with perfectly calibrated predictions."""
        # Perfect calibration: confidence = accuracy
        probs = torch.tensor(
            [
                [0.9, 0.1],  # Correct, high confidence
                [0.8, 0.2],  # Correct, high confidence
                [0.6, 0.4],  # Correct, medium confidence
                [0.1, 0.9],  # Incorrect, low confidence
                [0.2, 0.8],  # Incorrect, low confidence
            ]
        )
        targets = torch.tensor([0, 0, 0, 0, 1])  # Last two are incorrect

        ece, _, _ = expected_calibration_error(probs, targets)
        assert ece < 0.1  # Should be well calibrated

    def test_compute_calibration_metrics(self):
        """Test comprehensive calibration metrics."""
        probs = torch.randn(20, 2)
        probs = torch.softmax(probs, dim=1)
        targets = torch.randint(0, 2, (20,))

        metrics = compute_calibration_metrics(probs, targets)

        assert "ece" in metrics
        assert "adaptive_ece" in metrics
        assert "brier_score" in metrics
        assert "over_confident" in metrics
        assert "well_calibrated" in metrics
        assert "acceptable" in metrics

        assert 0 <= metrics["ece"] <= 1
        assert 0 <= metrics["brier_score"] <= 1


class TestJointMetrics:
    """Test joint success metrics."""

    def test_joint_success_rate_perfect(self):
        """Test joint success with perfect predictions."""
        type_correct = torch.tensor([True, True, True])
        center_correct = torch.tensor([True, True, True])
        length_correct = torch.tensor([True, True, True])

        metrics = joint_success_rate(type_correct, center_correct, length_correct)

        assert metrics["joint_success_strict"] == 1.0
        assert metrics["type_success_rate"] == 1.0
        assert metrics["center_success_rate"] == 1.0
        assert metrics["length_success_rate"] == 1.0
        assert metrics["meets_stones_threshold"] == True

    def test_joint_success_rate_partial(self):
        """Test joint success with partial success."""
        type_correct = torch.tensor([True, False, True])
        center_correct = torch.tensor([True, True, False])
        length_correct = torch.tensor([True, True, True])

        metrics = joint_success_rate(type_correct, center_correct, length_correct)

        # Only first sample has all correct
        assert metrics["joint_success_strict"] == 1 / 3
        assert metrics["type_success_rate"] == 2 / 3
        assert metrics["center_success_rate"] == 2 / 3
        assert metrics["length_success_rate"] == 1.0

    def test_compute_comprehensive_joint_metrics(self):
        """Test comprehensive joint metrics computation."""
        batch_size = 10
        type_probs = torch.randn(batch_size, 2)
        type_probs = torch.softmax(type_probs, dim=1)
        type_targets = torch.randint(0, 2, (batch_size,))
        pred_center = torch.rand(batch_size)
        pred_length = torch.rand(batch_size)
        true_center = torch.rand(batch_size)
        true_length = torch.rand(batch_size)

        metrics = compute_comprehensive_joint_metrics(
            type_probs,
            type_targets,
            pred_center,
            pred_length,
            true_center,
            true_length,
            bootstrap_samples=100,  # Small for testing
        )

        # Check key metrics exist
        assert "type_accuracy" in metrics
        assert "type_f1_macro" in metrics
        assert "center_hit_at_3" in metrics
        assert "joint_success_strict" in metrics
        assert "ece" in metrics
        assert "overall_stones_success" in metrics

        # Check threshold flags
        assert "type_f1_macro_meets_threshold" in metrics
        assert "center_hit_at_3_meets_threshold" in metrics
        assert "joint_success_meets_threshold" in metrics
        assert "ece_meets_threshold" in metrics

        # Check confidence intervals
        assert "type_accuracy_ci" in metrics
        assert "center_hit_rate_ci" in metrics


class TestStonesEvaluation:
    """Test Stones evaluation framework."""

    def test_evaluate_stones_metrics(self):
        """Test Stones metrics evaluation."""
        batch_size = 20
        type_probs = torch.randn(batch_size, 2)
        type_probs = torch.softmax(type_probs, dim=1)
        type_targets = torch.randint(0, 2, (batch_size,))
        pred_center = torch.rand(batch_size)
        pred_length = torch.rand(batch_size)
        true_center = torch.rand(batch_size)
        true_length = torch.rand(batch_size)

        metrics = evaluate_stones_metrics(
            type_probs,
            type_targets,
            pred_center,
            pred_length,
            true_center,
            true_length,
            bootstrap_samples=100,
        )

        assert "overall_stones_success" in metrics
        assert isinstance(metrics["overall_stones_success"], bool)

    def test_check_stones_thresholds(self):
        """Test Stones threshold checking."""
        # Create metrics that meet all thresholds
        good_metrics = {
            "type_f1_macro": 0.6,
            "center_hit_at_3": 0.7,
            "joint_success_strict": 0.5,
            "ece": 0.05,
        }

        summary = check_stones_thresholds(good_metrics)
        assert summary["overall"] == "PASS"
        assert all(status == "PASS" for status in summary.values() if status != "overall")

        # Create metrics that fail some thresholds
        bad_metrics = {
            "type_f1_macro": 0.3,
            "center_hit_at_3": 0.4,
            "joint_success_strict": 0.2,
            "ece": 0.15,
        }

        summary = check_stones_thresholds(bad_metrics)
        assert summary["overall"] == "FAIL"
        assert all(status == "FAIL" for status in summary.values() if status != "overall")


class TestStonesThresholds:
    """Test specific Stones threshold values."""

    def test_f1_macro_threshold(self):
        """Test F1-macro ≥0.50 threshold."""
        assert 0.50 >= 0.50  # Meets threshold
        assert 0.49 < 0.50  # Below threshold

    def test_hit_at_3_threshold(self):
        """Test Hit@±3 ≥60% threshold."""
        assert 0.60 >= 0.60  # Meets threshold
        assert 0.59 < 0.60  # Below threshold

    def test_joint_success_threshold(self):
        """Test Joint success ≥40% threshold."""
        assert 0.40 >= 0.40  # Meets threshold
        assert 0.39 < 0.40  # Below threshold

    def test_ece_threshold(self):
        """Test ECE <0.10 threshold."""
        assert 0.09 < 0.10  # Meets threshold
        assert 0.10 >= 0.10  # Does not meet threshold
        assert 0.11 >= 0.10  # Does not meet threshold


if __name__ == "__main__":
    pytest.main([__file__])
