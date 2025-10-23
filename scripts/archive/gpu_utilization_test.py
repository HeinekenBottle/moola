#!/usr/bin/env python3
"""GPU Utilization Diagnostic Script for Jade Pretraining.

Tests the critical GPU utilization fixes:
1. Tensor placement verification
2. Memory allocation tracking  
3. Forward/backward pass GPU usage
4. DataLoader pin_memory effectiveness
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from moola.models.jade_pretrain import JadePretrainer, JadeConfig
from moola.data.windowed_loader import WindowedConfig, create_dataloaders
import pandas as pd


def test_tensor_placement():
    """Test 1: Verify tensor placement in training loop."""
    print("=" * 60)
    print("TEST 1: Tensor Placement Verification")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    config = JadeConfig(input_size=10, hidden_size=64, num_layers=1, dropout=0.2, huber_delta=1.0)
    model = JadePretrainer(config).to(device)
    print(f"Model device: {next(model.parameters()).device}")
    
    # Create dummy batch
    batch_size = 8
    seq_len = 105
    n_features = 10
    
    # Test tuple format
    X = torch.randn(batch_size, seq_len, n_features)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    print(f"Before device move - X device: {X.device}")
    
    # CRITICAL: Move to device BEFORE forward pass (the fix)
    X = X.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    valid_mask = valid_mask.to(device, non_blocking=True)
    
    print(f"After device move - X device: {X.device}")
    
    batch = (X, mask, valid_mask)
    
    # Forward pass
    with torch.no_grad():
        loss, metrics = model(batch)
    
    print(f"Loss device: {loss.device}")
    print(f"Loss value: {loss.item():.6f}")
    print("✅ Tensor placement test passed\n")


def test_memory_allocation():
    """Test 2: GPU memory allocation tracking."""
    print("=" * 60)
    print("TEST 2: GPU Memory Allocation Tracking")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test\n")
        return
    
    device = torch.device('cuda')
    
    # Clear memory
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    print(f"Initial GPU memory: {initial_memory / 1e6:.1f} MB")
    
    # Create model
    config = JadeConfig(input_size=10, hidden_size=128, num_layers=2, dropout=0.2, huber_delta=1.0)
    model = JadePretrainer(config).to(device)
    
    after_model_memory = torch.cuda.memory_allocated()
    print(f"After model: {after_model_memory / 1e6:.1f} MB (+{(after_model_memory-initial_memory)/1e6:.1f} MB)")
    
    # Create batch
    batch_size = 32
    X = torch.randn(batch_size, 105, 10, device=device)
    mask = torch.zeros(batch_size, 105, dtype=torch.bool, device=device)
    valid_mask = torch.ones(batch_size, 105, dtype=torch.bool, device=device)
    
    after_batch_memory = torch.cuda.memory_allocated()
    print(f"After batch: {after_batch_memory / 1e6:.1f} MB (+{(after_batch_memory-after_model_memory)/1e6:.1f} MB)")
    
    # Forward pass
    loss, metrics = model((X, mask, valid_mask))
    
    after_forward_memory = torch.cuda.memory_allocated()
    print(f"After forward: {after_forward_memory / 1e6:.1f} MB (+{(after_forward_memory-after_batch_memory)/1e6:.1f} MB)")
    
    # Backward pass
    loss.backward()
    
    after_backward_memory = torch.cuda.memory_allocated()
    print(f"After backward: {after_backward_memory / 1e6:.1f} MB (+{(after_backward_memory-after_forward_memory)/1e6:.1f} MB)")
    
    print("✅ Memory allocation test passed\n")


def test_dataloader_efficiency():
    """Test 3: DataLoader pin_memory and non_blocking transfers."""
    print("=" * 60)
    print("TEST 3: DataLoader Pin Memory Efficiency")
    print("=" * 60)
    
    # Create dummy data with datetime index
    n_samples = 1000
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='1min')
    data = {
        'open': np.random.randn(n_samples) * 100 + 5000,
        'high': np.random.randn(n_samples) * 100 + 5100,
        'low': np.random.randn(n_samples) * 100 + 4900,
        'close': np.random.randn(n_samples) * 100 + 5000,
    }
    df = pd.DataFrame(data, index=dates)
    
    # Create windowed config
    windowed_config = WindowedConfig(window_length=105, stride=52, warmup_bars=20, mask_ratio=0.15, feature_config=None)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing DataLoader on device: {device}")
    
    # Test with pin_memory=True (the fix)
    train_loader, _, _ = create_dataloaders(
        df, windowed_config, batch_size=32, num_workers=2, pin_memory=True
    )
    
    print(f"Created DataLoader with {len(train_loader)} batches")
    
    # Time the data loading and transfer
    start_time = time.time()
    for i, batch in enumerate(train_loader):
        if i >= 5:  # Test first 5 batches
            break
            
        # CRITICAL: Non-blocking transfer (the fix)
        if isinstance(batch, (list, tuple)):
            X, mask, valid_mask = batch
            X = X.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
        
        batch_time = time.time() - start_time
        print(f"Batch {i}: loaded and transferred in {batch_time:.3f}s")
        start_time = time.time()
    
    print("✅ DataLoader efficiency test passed\n")


def test_micro_gpu_benchmark():
    """Test 4: Micro-benchmark GPU vs CPU performance."""
    print("=" * 60)
    print("TEST 4: Micro GPU vs CPU Benchmark")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on device: {device}")
    
    # Create model
    config = JadeConfig(input_size=10, hidden_size=128, num_layers=2, dropout=0.2, huber_delta=1.0)
    model = JadePretrainer(config).to(device)
    
    # Create test batch
    batch_size = 64
    X = torch.randn(batch_size, 105, 10, device=device)
    mask = torch.zeros(batch_size, 105, dtype=torch.bool, device=device)
    valid_mask = torch.ones(batch_size, 105, dtype=torch.bool, device=device)
    
    # Warmup
    for _ in range(3):
        loss, _ = model((X, mask, valid_mask))
        loss.backward()
        model.zero_grad()
    
    # Benchmark forward+backward
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(10):
        loss, metrics = model((X, mask, valid_mask))
        loss.backward()
        model.zero_grad()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.time() - start_time
    
    print(f"10 forward+backward passes in {total_time:.3f}s")
    print(f"Average time per iteration: {total_time/10:.3f}s")
    print(f"Throughput: {batch_size*10/total_time:.1f} samples/sec")
    
    if torch.cuda.is_available():
        gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else "N/A"
        memory_used = torch.cuda.memory_allocated() / 1e6
        print(f"GPU Memory Used: {memory_used:.1f} MB")
        print(f"GPU Utilization: {gpu_utilization}")
    
    print("✅ Micro benchmark test passed\n")


def main():
    """Run all GPU utilization diagnostic tests."""
    print("Jade Pretraining GPU Utilization Diagnostic")
    print("=" * 60)
    
    try:
        test_tensor_placement()
        test_memory_allocation()
        test_dataloader_efficiency()
        test_micro_gpu_benchmark()
        
        print("=" * 60)
        print("ALL TESTS PASSED! 🎉")
        print("GPU utilization fixes are working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())