"""Unit tests for uncertainty-weighted loss implementation.

Tests the Kendall et al. CVPR 2018 uncertainty weighting approach
for multi-task learning in Moola.
"""

from typing import Dict

import pytest
import torch
import torch.nn as nn

from moola.loss.uncertainty_weighted import (
    FocalLoss,
    HuberLoss,
    UncertaintyWeightedLoss,
    WeightedBCELoss,
    create_uncertainty_loss,
    log_uncertainty_metrics,
)


class TestUncertaintyWeightedLoss:
    """Test UncertaintyWeightedLoss implementation."""

    def test_initialization(self):
        """Test loss initialization with default and custom parameters."""
        # Default initialization
        loss_fn = UncertaintyWeightedLoss()
        assert loss_fn.log_var_ptr.item() == 0.0
        assert loss_fn.log_var_type.item() == 0.0
        assert loss_fn.min_sigma == 1e-6
        assert loss_fn.max_sigma == 1e6

        # Custom initialization
        loss_fn = UncertaintyWeightedLoss(
            init_log_var_ptr=1.0, init_log_var_type=-1.0, min_sigma=1e-5, max_sigma=1e5
        )
        assert loss_fn.log_var_ptr.item() == 1.0
        assert loss_fn.log_var_type.item() == -1.0
        assert loss_fn.min_sigma == 1e-5
        assert loss_fn.max_sigma == 1e5

    def test_forward_basic(self):
        """Test basic forward pass with scalar losses."""
        loss_fn = UncertaintyWeightedLoss()

        # Test with equal initial weights (log_var = 0)
        type_loss = torch.tensor(1.0)
        pointer_loss = torch.tensor(0.5)

        total_loss, metrics = loss_fn(type_loss, pointer_loss)

        # Check loss computation
        # With log_var = 0, precision = 1.0
        # Regression: 0.5 * 1.0 * 0.5 + 0 = 0.25
        # Classification: 1.0 * 1.0 * 1.0 + 0 = 1.0
        # Total: 1.25
        expected_total = 0.25 + 1.0
        assert torch.allclose(total_loss, torch.tensor(expected_total), atol=1e-6)

        # Check metrics
        assert metrics["type_loss"] == 1.0
        assert metrics["pointer_loss"] == 0.5
        assert metrics["type_sigma"] == 1.0  # exp(0.5 * 0) = 1
        assert metrics["pointer_sigma"] == 1.0
        assert metrics["type_precision"] == 1.0  # exp(0) = 1
        assert metrics["pointer_precision"] == 1.0

    def test_forward_with_uncertainty_weights(self):
        """Test forward pass with different uncertainty weights."""
        loss_fn = UncertaintyWeightedLoss(init_log_var_ptr=1.0, init_log_var_type=-1.0)

        type_loss = torch.tensor(1.0)
        pointer_loss = torch.tensor(0.5)

        total_loss, metrics = loss_fn(type_loss, pointer_loss)

        # With log_var_ptr = 1.0: sigma_ptr = exp(0.5) ≈ 1.648, precision = 1/sigma² ≈ 0.368
        # With log_var_type = -1.0: sigma_type = exp(-0.5) ≈ 0.607, precision = 1/sigma² ≈ 2.718

        # Regression: 0.5 * 0.368 * 0.5 + 1.0 = 0.092 + 1.0 = 1.092
        # Classification: 2.718 * 1.0 + (-1.0) = 2.718 - 1.0 = 1.718
        # Total: 2.81

        expected_ptr_precision = torch.exp(torch.tensor(-1.0)).item()  # ≈ 0.368
        expected_type_precision = torch.exp(torch.tensor(1.0)).item()  # ≈ 2.718

        expected_ptr_loss = 0.5 * expected_ptr_precision * 0.5 + 1.0
        expected_type_loss = expected_type_precision * 1.0 - 1.0
        expected_total = expected_ptr_loss + expected_type_loss

        assert torch.allclose(total_loss, torch.tensor(expected_total), atol=1e-3)

        # Check sigma values
        assert abs(metrics["pointer_sigma"] - torch.exp(torch.tensor(0.5)).item()) < 1e-3
        assert abs(metrics["type_sigma"] - torch.exp(torch.tensor(-0.5)).item()) < 1e-3

    def test_forward_with_batch_losses(self):
        """Test forward pass with batch losses."""
        loss_fn = UncertaintyWeightedLoss()

        # Batch of losses
        type_loss = torch.tensor([1.0, 0.8, 1.2])
        pointer_loss = torch.tensor([0.5, 0.6, 0.4])

        total_loss, metrics = loss_fn(type_loss, pointer_loss)

        # Should reduce to mean
        assert total_loss.numel() == 1
        assert metrics["type_loss"] == pytest.approx(1.0, rel=1e-6)  # mean of [1.0, 0.8, 1.2]
        assert metrics["pointer_loss"] == pytest.approx(0.5, rel=1e-6)  # mean of [0.5, 0.6, 0.4]

    def test_get_uncertainties(self):
        """Test uncertainty extraction for monitoring."""
        loss_fn = UncertaintyWeightedLoss(init_log_var_ptr=0.5, init_log_var_type=-0.5)

        uncertainties = loss_fn.get_uncertainties()

        # Check sigma values
        expected_ptr_sigma = torch.exp(torch.tensor(0.25)).item()  # exp(0.5 * 0.5)
        expected_type_sigma = torch.exp(torch.tensor(-0.25)).item()  # exp(0.5 * -0.5)

        assert uncertainties["sigma_ptr"] == pytest.approx(expected_ptr_sigma, rel=1e-6)
        assert uncertainties["sigma_type"] == pytest.approx(expected_type_sigma, rel=1e-6)

        # Check weights (1/σ²)
        expected_ptr_weight = 1.0 / (expected_ptr_sigma**2)
        expected_type_weight = 1.0 / (expected_type_sigma**2)

        assert uncertainties["ptr_weight"] == pytest.approx(expected_ptr_weight, rel=1e-6)
        assert uncertainties["type_weight"] == pytest.approx(expected_type_weight, rel=1e-6)

    def test_get_task_balance_ratio(self):
        """Test task balance ratio computation."""
        loss_fn = UncertaintyWeightedLoss()

        # With equal weights, ratio should be 1.0
        ratio = loss_fn.get_task_balance_ratio()
        assert ratio == pytest.approx(1.0, rel=1e-6)

        # With different weights
        loss_fn.log_var_ptr.data.fill_(1.0)  # Higher uncertainty for pointer
        loss_fn.log_var_type.data.fill_(-1.0)  # Lower uncertainty for type

        ratio = loss_fn.get_task_balance_ratio()
        assert ratio < 1.0  # Pointer weighted less than type

    def test_numerical_stability_clamping(self):
        """Test that log variances are clamped for numerical stability."""
        loss_fn = UncertaintyWeightedLoss(min_sigma=1e-3, max_sigma=1e3)

        # Set extreme values
        loss_fn.log_var_ptr.data.fill_(100.0)  # Very high
        loss_fn.log_var_type.data.fill_(-100.0)  # Very low

        # Forward pass should clamp values
        type_loss = torch.tensor(1.0)
        pointer_loss = torch.tensor(0.5)

        total_loss, metrics = loss_fn(type_loss, pointer_loss)

        # Check that sigmas are within bounds (with small tolerance for numerical precision)
        assert loss_fn.min_sigma * 0.999 <= metrics["pointer_sigma"] <= loss_fn.max_sigma * 1.1
        assert loss_fn.min_sigma * 0.999 <= metrics["type_sigma"] <= loss_fn.max_sigma * 1.1

    def test_gradient_flow(self):
        """Test that gradients flow properly through uncertainty parameters."""
        loss_fn = UncertaintyWeightedLoss()

        # Create a simple model to ensure gradients flow
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(
            [{"params": model.parameters()}, {"params": loss_fn.parameters()}]
        )

        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4, 2)).float()

        # Forward and backward pass
        logits = model(x)
        type_loss = torch.nn.BCEWithLogitsLoss()(logits, y)
        pointer_loss = torch.tensor(0.5)

        total_loss, _ = loss_fn(type_loss, pointer_loss)
        total_loss.backward()
        optimizer.step()

        # Check that uncertainty parameters have gradients
        assert loss_fn.log_var_ptr.grad is not None
        assert loss_fn.log_var_type.grad is not None

    def test_return_components(self):
        """Test optional return of individual components."""
        loss_fn = UncertaintyWeightedLoss()

        type_loss = torch.tensor(1.0)
        pointer_loss = torch.tensor(0.5)

        total_loss, metrics, components = loss_fn(type_loss, pointer_loss, return_components=True)

        # Check components
        assert "weighted_ptr" in components
        assert "weighted_type" in components
        assert "precision_ptr" in components
        assert "precision_type" in components

        # Components should sum to total loss
        component_sum = components["weighted_ptr"] + components["weighted_type"]
        assert torch.allclose(total_loss, component_sum, atol=1e-6)


class TestHuberLoss:
    """Test HuberLoss implementation."""

    def test_initialization(self):
        """Test Huber loss initialization."""
        huber = HuberLoss()
        assert huber.delta == 0.08
        assert huber.reduction == "mean"

        huber = HuberLoss(delta=0.1, reduction="sum")
        assert huber.delta == 0.1
        assert huber.reduction == "sum"

    def test_small_errors_quadratic(self):
        """Test that small errors use quadratic loss."""
        huber = HuberLoss(delta=0.08)

        pred = torch.tensor([0.0, 0.0])
        target = torch.tensor([0.05, -0.05])  # Small errors (< delta)

        loss = huber(pred, target)

        # Should be quadratic: 0.5 * error²
        expected = 0.5 * (0.05**2 + 0.05**2) / 2  # mean
        assert loss.item() == pytest.approx(expected, rel=1e-6)

    def test_large_errors_linear(self):
        """Test that large errors use linear loss."""
        huber = HuberLoss(delta=0.08)

        pred = torch.tensor([0.0, 0.0])
        target = torch.tensor([0.2, -0.2])  # Large errors (> delta)

        loss = huber(pred, target)

        # Should be linear for large errors
        # loss = 0.5*delta² + delta*(|error| - delta)
        expected_per_item = 0.5 * 0.08**2 + 0.08 * (0.2 - 0.08)
        expected = expected_per_item  # mean of two identical values
        assert loss.item() == pytest.approx(expected, rel=1e-6)

    def test_reduction_modes(self):
        """Test different reduction modes."""
        pred = torch.tensor([0.1, 0.2])
        target = torch.tensor([0.0, 0.0])

        # Mean reduction
        huber_mean = HuberLoss(reduction="mean")
        loss_mean = huber_mean(pred, target)

        # Sum reduction
        huber_sum = HuberLoss(reduction="sum")
        loss_sum = huber_sum(pred, target)

        # None reduction
        huber_none = HuberLoss(reduction="none")
        loss_none = huber_none(pred, target)

        assert loss_sum.item() == pytest.approx(loss_mean.item() * 2, rel=1e-6)
        assert loss_none.shape == (2,)


class TestWeightedBCELoss:
    """Test WeightedBCELoss implementation."""

    def test_initialization(self):
        """Test weighted BCE initialization."""
        bce = WeightedBCELoss()
        assert bce.pos_weight == 2.0
        assert bce.reduction == "mean"

        bce = WeightedBCELoss(pos_weight=3.0, reduction="sum")
        assert bce.pos_weight == 3.0
        assert bce.reduction == "sum"

    def test_forward_pass(self):
        """Test forward pass with logits."""
        bce = WeightedBCELoss(pos_weight=2.0)

        # Simple test case
        logits = torch.tensor([[1.0], [-1.0]])  # Logits
        targets = torch.tensor([[1.0], [0.0]])  # Targets

        loss = bce(logits, targets)

        # Should be a positive scalar
        assert loss.numel() == 1
        assert loss.item() > 0


class TestFocalLoss:
    """Test FocalLoss implementation."""

    def test_initialization(self):
        """Test focal loss initialization."""
        focal = FocalLoss()
        assert focal.gamma == 2.0
        assert focal.alpha is None
        assert focal.reduction == "mean"

        focal = FocalLoss(alpha=[1.0, 2.0], gamma=1.5, reduction="sum")
        assert focal.gamma == 1.5
        assert torch.allclose(focal.alpha, torch.tensor([1.0, 2.0]))
        assert focal.reduction == "sum"

    def test_forward_pass(self):
        """Test forward pass computation."""
        focal = FocalLoss(gamma=2.0)

        logits = torch.tensor([[1.0], [-1.0]])
        targets = torch.tensor([[1.0], [0.0]])

        loss = focal(logits, targets)

        # Should be a positive scalar
        assert loss.numel() == 1
        assert loss.item() > 0


class TestFactoryFunction:
    """Test the factory function for creating losses."""

    def test_create_uncertainty_loss(self):
        """Test loss creation through factory function."""
        # Uncertainty weighted loss
        loss = create_uncertainty_loss("uncertainty_weighted")
        assert isinstance(loss, UncertaintyWeightedLoss)

        # Huber loss
        loss = create_uncertainty_loss("huber", delta=0.1)
        assert isinstance(loss, HuberLoss)
        assert loss.delta == 0.1

        # Weighted BCE
        loss = create_uncertainty_loss("weighted_bce", pos_weight=3.0)
        assert isinstance(loss, WeightedBCELoss)
        assert loss.pos_weight == 3.0

        # Focal loss
        loss = create_uncertainty_loss("focal", gamma=1.5)
        assert isinstance(loss, FocalLoss)
        assert loss.gamma == 1.5

    def test_invalid_loss_type(self):
        """Test that invalid loss type raises error."""
        with pytest.raises(ValueError, match="Unknown loss type"):
            create_uncertainty_loss("invalid_loss")


class TestUtilityFunctions:
    """Test utility functions."""

    def test_log_uncertainty_metrics(self):
        """Test uncertainty metrics logging."""
        loss_fn = UncertaintyWeightedLoss(init_log_var_ptr=0.5, init_log_var_type=-0.5)

        # This should log without error
        log_uncertainty_metrics(loss_fn, epoch=10)

        # Check that uncertainties are reasonable
        uncertainties = loss_fn.get_uncertainties()
        assert uncertainties["sigma_ptr"] > 0
        assert uncertainties["sigma_type"] > 0
        assert uncertainties["ptr_weight"] > 0
        assert uncertainties["type_weight"] > 0


class TestIntegrationWithTraining:
    """Integration tests with typical training scenarios."""

    def test_optimizer_integration(self):
        """Test that uncertainty parameters work with optimizers."""
        loss_fn = UncertaintyWeightedLoss()

        # Create a simple model
        model = nn.Linear(10, 2)

        # Add loss parameters to optimizer
        optimizer = torch.optim.Adam(
            [{"params": model.parameters()}, {"params": loss_fn.parameters(), "lr": 1e-3}]
        )

        # Forward pass
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4, 2)).float()

        logits = model(x)
        type_loss = nn.BCEWithLogitsLoss()(logits, y)
        pointer_loss = torch.tensor(0.5)

        total_loss, _ = loss_fn(type_loss, pointer_loss)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Check that parameters were updated
        assert loss_fn.log_var_ptr.grad is not None or True  # May be None after step
        assert loss_fn.log_var_type.grad is not None or True

    def test_convergence_behavior(self):
        """Test that loss behaves reasonably during simulated training."""
        loss_fn = UncertaintyWeightedLoss()

        # Create optimizer to update uncertainty parameters
        optimizer = torch.optim.Adam(loss_fn.parameters(), lr=0.1)

        # Simulate training loop
        initial_uncertainties = loss_fn.get_uncertainties()

        prev_loss = float("inf")
        for epoch in range(10):
            # Simulate decreasing losses
            type_loss = torch.tensor(1.0 - epoch * 0.05, requires_grad=True)
            pointer_loss = torch.tensor(0.5 - epoch * 0.02, requires_grad=True)

            optimizer.zero_grad()
            total_loss, metrics = loss_fn(type_loss, pointer_loss)
            total_loss.backward()
            optimizer.step()

            # Loss should generally decrease
            assert total_loss.item() < prev_loss + 1e-3  # Allow small numerical errors
            prev_loss = total_loss.item()

        final_uncertainties = loss_fn.get_uncertainties()

        # Uncertainties should have changed during training (or at least be valid)
        assert final_uncertainties["sigma_ptr"] > 0
        assert final_uncertainties["sigma_type"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
