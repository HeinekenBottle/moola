#!/usr/bin/env python3
"""Simple GPU usage test."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

import time

import torch


def simple_gpu_test():
    print("=== Simple GPU Test ===")

    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

        # Test tensor operations on GPU
        print("Testing GPU tensor operations...")
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")

        start = time.time()
        z = torch.mm(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"GPU matrix multiplication (1000x1000): {elapsed:.4f}s")

        # Test CPU for comparison
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        start = time.time()
        z_cpu = torch.mm(x_cpu, y_cpu)
        elapsed_cpu = time.time() - start
        print(f"CPU matrix multiplication (1000x1000): {elapsed_cpu:.4f}s")

        print(f"GPU speedup: {elapsed_cpu/elapsed:.1f}x")

        # Test model forward pass
        print("\nTesting model forward pass...")
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
        ).to("cuda")

        batch = torch.randn(32, 10, device="cuda")

        start = time.time()
        for _ in range(100):
            output = model(batch)
            torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"Model forward pass (100 iterations): {elapsed:.4f}s")

        print("✅ GPU is working!")
    else:
        print("❌ CUDA not available")


if __name__ == "__main__":
    simple_gpu_test()
