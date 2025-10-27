#!/usr/bin/env python3
"""Quick GPU test for Jade pretraining."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

import time

import torch

from moola.models.jade_pretrain import JadeConfig, JadePretrainer


def test_gpu():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create model
    config = JadeConfig(input_size=10, hidden_size=128, num_layers=2, dropout=0.2, huber_delta=1.0)
    model = JadePretrainer(config)

    print(f"Model device: {next(model.parameters()).device}")

    # Test forward pass on GPU
    if torch.cuda.is_available():
        # Create dummy data on GPU
        batch_size = 32
        K, D = 105, 10

        X = torch.randn(batch_size, K, D, device="cuda")
        mask = torch.zeros(batch_size, K, dtype=torch.bool, device="cuda")
        valid_mask = torch.ones(batch_size, K, dtype=torch.bool, device="cuda")

        # Mask some positions
        mask[:, :15] = True

        batch = (X, mask, valid_mask)

        print("Testing forward pass on GPU...")
        start = time.time()
        loss, metrics = model(batch)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        print(f"Forward pass time: {elapsed:.3f}s")
        print(f"Loss: {loss.item():.6f}")
        print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

        # Test backward pass
        print("Testing backward pass...")
        start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"Backward pass time: {elapsed:.3f}s")

        print("✅ GPU test successful!")
    else:
        print("❌ CUDA not available")


if __name__ == "__main__":
    test_gpu()
