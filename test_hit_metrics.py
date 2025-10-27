#!/usr/bin/env python3
"""Unit test for hit_at_k metrics with synthetic data."""

import torch

from src.moola.data.pointer_transforms import (
    center_length_to_start_end,
    start_end_to_center_length,
)
from src.moola.metrics.hit_metrics import hit_at_k


def test_hit_metrics():
    """Test hit_at_k with known cases."""
    print("=== Testing Hit@K Metrics ===")

    # Test case 1: Perfect predictions (should hit 100%)
    pred_center = torch.tensor([0.5])  # Center at position 52
    true_center = torch.tensor([0.5])  # Same position

    hit_rate = hit_at_k(pred_center, true_center, k=3)
    print(f"Perfect prediction: Hit@3 = {hit_rate:.4f} (expected: 1.000)")
    assert abs(hit_rate - 1.0) < 1e-6, f"Expected 1.0, got {hit_rate}"

    # Test case 2: Predictions within ±3 timesteps (should hit)
    # Convert index differences to normalized units
    # For seq_len=104, 3 timesteps = 3/104 = 0.0288 normalized units
    pred_center = torch.tensor([0.5])  # Position 52
    true_center = torch.tensor([0.5 + 3 / 104])  # Position 55 (within tolerance)

    hit_rate = hit_at_k(pred_center, true_center, k=3)
    print(f"Within +3 timesteps: Hit@3 = {hit_rate:.4f} (expected: 1.000)")
    assert abs(hit_rate - 1.0) < 1e-6, f"Expected 1.0, got {hit_rate}"

    # Test case 3: Predictions outside ±3 timesteps (should miss)
    pred_center = torch.tensor([0.5])  # Position 52
    true_center = torch.tensor([0.5 + 5 / 104])  # Position 57 (outside tolerance)

    hit_rate = hit_at_k(pred_center, true_center, k=3)
    print(f"Outside +3 timesteps: Hit@3 = {hit_rate:.4f} (expected: 0.000)")
    assert abs(hit_rate - 0.0) < 1e-6, f"Expected 0.0, got {hit_rate}"

    # Test case 4: Batch with mixed results
    pred_centers = torch.tensor([0.5, 0.5, 0.5, 0.5])  # All at position 52
    true_centers = torch.tensor([0.5, 0.5 + 2 / 104, 0.5 + 4 / 104, 0.5 + 6 / 104])  # 0, +2, +4, +6

    hit_rate = hit_at_k(pred_centers, true_centers, k=3)
    expected_hits = 2 / 4  # First two should hit, last two should miss
    print(f"Mixed batch (4 samples): Hit@3 = {hit_rate:.4f} (expected: {expected_hits:.4f})")
    assert abs(hit_rate - expected_hits) < 1e-6, f"Expected {expected_hits}, got {hit_rate}"

    print("✅ All hit metrics tests passed!")


def test_pointer_transforms():
    """Test center/length ↔ start/end conversions."""
    print("\n=== Testing Pointer Transforms ===")

    # Test case 1: Simple conversion
    start = torch.tensor([10.0])
    end = torch.tensor([50.0])

    center, length = start_end_to_center_length(start, end, seq_len=104)
    print(
        f"Start: {start.item()}, End: {end.item()} → Center: {center.item():.4f}, Length: {length.item():.4f}"
    )

    # Convert back
    recovered_start, recovered_end = center_length_to_start_end(center, length, seq_len=104)
    print(f"Recovered → Start: {recovered_start.item():.1f}, End: {recovered_end.item():.1f}")

    # Check recovery accuracy (allowing for small floating point errors)
    assert (
        abs(recovered_start.item() - start.item()) < 0.5
    ), f"Start recovery failed: {recovered_start} vs {start}"
    assert (
        abs(recovered_end.item() - end.item()) < 0.5
    ), f"End recovery failed: {recovered_end} vs {end}"

    # Test case 2: Edge case - minimum span
    start = torch.tensor([30.0])
    end = torch.tensor([30.0])  # Same position (length 1)

    center, length = start_end_to_center_length(start, end, seq_len=104)
    print(
        f"Min span - Start: {start.item()}, End: {end.item()} → Center: {center.item():.4f}, Length: {length.item():.4f}"
    )
    assert length.item() > 0, "Length should be positive"

    # Test case 3: Edge case - maximum span
    start = torch.tensor([0.0])
    end = torch.tensor([104.0])  # Full span

    center, length = start_end_to_center_length(start, end, seq_len=104)
    print(
        f"Max span - Start: {start.item()}, End: {end.item()} → Center: {center.item():.4f}, Length: {length.item():.4f}"
    )
    assert length.item() <= 1.0, "Length should not exceed 1.0"

    print("✅ All pointer transform tests passed!")


def test_uncertainty_bias():
    """Test Kendall bias initialization."""
    print("\n=== Testing Uncertainty Bias ===")

    from src.moola.loss.uncertainty_weighted import UncertaintyWeightedLoss

    # Test default initialization (should favor pointer)
    loss_fn = UncertaintyWeightedLoss()

    uncertainties = loss_fn.get_uncertainties()
    print(f"Initial σ_ptr: {uncertainties['sigma_ptr']:.4f} (expected: ~0.74)")
    print(f"Initial σ_type: {uncertainties['sigma_type']:.4f} (expected: 1.00)")
    print(f"Initial ptr_weight: {uncertainties['ptr_weight']:.4f}")
    print(f"Initial type_weight: {uncertainties['type_weight']:.4f}")
    print(
        f"Initial balance ratio (ptr/type): {loss_fn.get_task_balance_ratio():.4f} (expected: >1.0)"
    )

    # Verify pointer is favored (ratio > 1.0)
    assert loss_fn.get_task_balance_ratio() > 1.0, "Pointer task should be favored"
    assert (
        abs(uncertainties["sigma_ptr"] - 0.74) < 0.05
    ), f"Pointer sigma should be ~0.74, got {uncertainties['sigma_ptr']}"
    assert (
        abs(uncertainties["sigma_type"] - 1.00) < 0.1
    ), f"Type sigma should be ~1.00, got {uncertainties['sigma_type']}"

    print("✅ Uncertainty bias test passed!")


if __name__ == "__main__":
    test_hit_metrics()
    test_pointer_transforms()
    test_uncertainty_bias()
    print("\n🎉 All tests passed! Pointer-favoring patch is working correctly.")
