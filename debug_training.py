#!/usr/bin/env python3
"""Debug training script GPU usage."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import pandas as pd
from moola.data.windowed_loader import WindowedConfig, create_dataloaders
from moola.models.jade_pretrain import JadeConfig, JadePretrainer

def debug_training():
    print("=== Debug Training GPU Usage ===")
    
    # Create minimal config
    windowed_config = WindowedConfig(
        window_length=105,
        stride=52,
        warmup_bars=20,
        mask_ratio=0.15,
        feature_config=None
    )
    
    # Load data
    data_path = "data/raw/nq_5year.parquet"
    df = pd.read_parquet(data_path)
    
    print(f"Data shape: {df.shape}")
    
    # Create dataloaders with full dataset
    train_loader, _, _ = create_dataloaders(
        df, windowed_config, batch_size=8, num_workers=0, pin_memory=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    
    # Create model
    model_config = JadeConfig(
        input_size=10, hidden_size=128, num_layers=2, dropout=0.2, huber_delta=1.0
    )
    model = JadePretrainer(model_config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model device: {next(model.parameters()).device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Test one batch
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 2:  # Only test first 2 batches
            break
        print(f"\nBatch {batch_idx}:")
        print(f"  Batch type: {type(batch)}")
        print(f"  Batch length: {len(batch)}")
        
        if isinstance(batch, (list, tuple)):
            X, mask, valid_mask = batch
            print(f"  X device: {X.device}")
            print(f"  X shape: {X.shape}")
            print(f"  mask device: {mask.device}")
            print(f"  valid_mask device: {valid_mask.device}")
        
        # Forward pass
        with torch.cuda.amp.autocast():
            loss, metrics = model(batch)
        
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Loss device: {loss.device}")
        print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        
        # Backward pass
        loss.backward()
        print("  Backward pass completed")
        
        break  # Only test first batch
    
    print("\n✅ Debug completed successfully!")

if __name__ == "__main__":
    debug_training()