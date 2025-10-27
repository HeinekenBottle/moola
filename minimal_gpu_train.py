#!/usr/bin/env python3
"""Minimal working Jade pretraining on GPU."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
import torch
from torch.optim import AdamW

from moola.data.windowed_loader import WindowedConfig, create_dataloaders
from moola.models.jade_pretrain import JadeConfig, JadePretrainer


def minimal_training():
    print("=== Minimal Jade Training on GPU ===")

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configs
    windowed_config = WindowedConfig(
        window_length=105, stride=52, warmup_bars=20, mask_ratio=0.15, feature_config=None
    )

    model_config = JadeConfig(
        input_size=10, hidden_size=128, num_layers=2, dropout=0.2, huber_delta=1.0
    )

    # Load data
    print("Loading data...")
    data_path = "data/raw/nq_5year.parquet"
    df = pd.read_parquet(data_path)

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, _, _ = create_dataloaders(
        df, windowed_config, batch_size=64, num_workers=0, pin_memory=True
    )

    # Create model and move to GPU
    print("Creating model...")
    model = JadePretrainer(model_config)
    model = model.to(device)
    print(f"Model device: {next(model.parameters()).device}")

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # Training loop
    print("Starting training...")
    model.train()

    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 5:  # Only 5 batches for testing
            break

        # Move batch to GPU explicitly
        if isinstance(batch, (list, tuple)):
            X, mask, valid_mask = batch
            X = X.to(device)
            mask = mask.to(device)
            valid_mask = valid_mask.to(device)
            batch = (X, mask, valid_mask)

        optimizer.zero_grad()

        # Forward pass
        loss, metrics = model(batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(f"Batch {batch_idx}: Loss {loss.item():.6f}")

        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e6
            print(f"  GPU Memory: {gpu_memory:.1f} MB")

    print(f"Average loss: {total_loss / 5:.6f}")
    print("✅ Training completed on GPU!")


if __name__ == "__main__":
    minimal_training()
