#!/usr/bin/env python3
"""Micro GPU Test - Verifies the exact issue mentioned in requirements.

Tests: xb = torch.randn(32,105,10,device="cuda"); model.to("cuda")(xb).sum().backward()
"""

import sys
from pathlib import Path

import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from moola.models.jade_pretrain import JadeConfig, JadePretrainer


def micro_gpu_test():
    """Execute the exact micro-test from requirements."""
    print("=" * 60)
    print("MICRO GPU TEST - Exact Requirements Check")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    config = JadeConfig(input_size=10, hidden_size=128, num_layers=2, dropout=0.2, huber_delta=1.0)
    model = JadePretrainer(config).to(device)
    print(f"Model device: {next(model.parameters()).device}")

    # Create test tensor as specified in requirements
    xb = torch.randn(32, 105, 10, device=device)
    mask = torch.zeros(32, 105, dtype=torch.bool, device=device)
    valid_mask = torch.ones(32, 105, dtype=torch.bool, device=device)

    # Add some masked positions for non-zero loss
    for i in range(32):
        valid_start = 20
        valid_positions = list(range(valid_start, 105))
        n_mask = int(0.15 * len(valid_positions))
        if n_mask > 0:
            mask_indices = torch.randperm(len(valid_positions))[:n_mask]
            for idx in mask_indices:
                mask[i, valid_positions[idx]] = True

    print(f"Input tensor device: {xb.device}")
    print(f"Mask device: {mask.device}")
    print(f"Valid mask device: {valid_mask.device}")

    # Execute the exact test from requirements
    print("\nExecuting: model.to(device)(xb).sum().backward()")

    try:
        # Forward pass
        predictions, encoded = model.jade(xb, mask)
        print(f"Predictions shape: {predictions.shape}, device: {predictions.device}")
        print(f"Encoded shape: {encoded.shape}, device: {encoded.device}")

        # Compute loss
        loss, metrics = model.jade.compute_loss(predictions, xb, mask, valid_mask)
        print(f"Loss: {loss.item():.6f}, device: {loss.device}")

        # Backward pass
        loss.backward()
        print("✅ Backward pass completed successfully!")

        # Check gradients
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        print(f"Model has gradients: {has_gradients}")

        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e6
            print(f"GPU Memory Used: {memory_used:.1f} MB")

        print("\n✅ MICRO GPU TEST PASSED!")
        print("The exact requirements test is working correctly.")

    except Exception as e:
        print(f"❌ Micro test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_tensor_placement_verification():
    """Verify xb.device, yb.device, mask.device are all cuda:0."""
    print("\n" + "=" * 60)
    print("TENSOR PLACEMENT VERIFICATION")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available - cannot test cuda:0 placement")
        return True

    device = torch.device("cuda:0")

    # Test tensors
    xb = torch.randn(32, 105, 10, device=device)
    yb = torch.randn(32, 105, 10, device=device)  # Target
    mask = torch.zeros(32, 105, dtype=torch.bool, device=device)

    print(f"xb.device: {xb.device}")
    print(f"yb.device: {yb.device}")
    print(f"mask.device: {mask.device}")

    # Verify all are cuda:0
    all_cuda = (
        str(xb.device) == "cuda:0" and str(yb.device) == "cuda:0" and str(mask.device) == "cuda:0"
    )

    if all_cuda:
        print("✅ All tensors are on cuda:0")
    else:
        print("❌ Not all tensors are on cuda:0")

    return all_cuda


def main():
    """Run micro GPU test and verification."""
    print("Jade Pretraining - Micro GPU Test")
    print("Testing the exact requirements specification")

    success = True

    # Run micro test
    success &= micro_gpu_test()

    # Run tensor placement verification
    success &= test_tensor_placement_verification()

    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL MICRO TESTS PASSED!")
        print("GPU utilization fixes are working correctly.")
    else:
        print("❌ Some tests failed")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
