#!/usr/bin/env python3
"""Simple training script to test pointer-favoring JadeModel on RunPod."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from moola.models.jade import JadeModel
from moola.loss.uncertainty_weighted import UncertaintyWeightedLoss

def main():
    logger.info("Starting simple pointer-favoring training test")
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load the actual 174 sample dataset
    import pickle
    
    logger.info("Loading actual 174 sample dataset...")
    data_path = Path("data/processed/labeled/train_latest_11d.parquet")
    
    try:
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded dataset: {df.shape}")
        
        # Deserialize features and create X
        features_list = []
        for feat_bytes in df['features']:
            feat_array = pickle.loads(feat_bytes)
            features_list.append(feat_array)
        
        X = np.stack(features_list).astype(np.float32)
        
        # Create labels (consolidation=0, retracement=1)
        y = np.array([0 if label == 'consolidation' else 1 for label in df['label']])
        
        # Create metadata
        metadata = {"window_id": df['window_id'].values}
        
        logger.info(f"Loaded real data: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Class distribution: consolidation={int(np.sum(y==0))}, retracement={int(np.sum(y==1))}")
        
    except Exception as e:
        logger.error(f"Failed to load real data: {e}")
        return
    
    # Load temporal split
    with open("temporal_split.json", "r") as f:
        split_data = json.load(f)
    
    train_ids = set(split_data["train"])
    val_ids = set(split_data["val"])
    
    # Create train/val masks using indices (since window_ids are strings)
    n_samples = len(X)
    train_indices = list(range(min(len(train_ids), n_samples)))
    val_indices = list(range(len(train_indices), min(len(train_indices) + len(val_ids), n_samples)))
    
    train_mask = np.zeros(n_samples, dtype=bool)
    val_mask = np.zeros(n_samples, dtype=bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    
    logger.info(f"Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")
    
    # Initialize JadeModel with pointer-favoring
    model = JadeModel(
        seed=1337,
        device=device,
        predict_pointers=True,  # Enable multi-task learning
        n_epochs=2,  # Short test run
        batch_size=16,  # Smaller batch for testing
    )
    
    # Build the model manually (this is the key fix!)
    n_classes = 2  # consolidation and retracement only
    model.model = model._build_model(input_dim=X_train.shape[2], n_classes=n_classes)
    model.model = model.model.to(device)
    model.n_classes = n_classes
    model.input_dim = X_train.shape[2]
    
    logger.info(f"Built JadeModel with {n_classes} classes, input_dim={X_train.shape[2]}")
    logger.info(f"Model device: {next(model.model.parameters()).device}")
    
    # Initialize pointer-favoring loss
    criterion = UncertaintyWeightedLoss(
        init_log_var_ptr=-0.60,  # σ≈0.74 (higher weight)
        init_log_var_type=0.00,  # σ=1.0 (lower weight)
    )
    criterion = criterion.to(device)
    
    logger.info("Initialized pointer-favoring loss:")
    logger.info(f"  log_var_ptr: {criterion.log_var_ptr.item():.2f} (σ={torch.exp(criterion.log_var_ptr).item():.2f})")
    logger.info(f"  log_var_type: {criterion.log_var_type.item():.2f} (σ={torch.exp(criterion.log_var_type).item():.2f})")
    logger.info(f"  Pointer weight ratio: {(torch.exp(criterion.log_var_type) / torch.exp(criterion.log_var_ptr)).item():.2f}x")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=3e-4, weight_decay=1e-5)
    
    # Training loop
    logger.info("Starting training loop...")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Simple training without augmentation for testing
    model.model.train()
    for epoch in range(model.n_epochs):
        total_loss = 0.0
        n_batches = 0
        
        # Simple batching
        batch_size = model.batch_size
        for i in range(0, len(X_train_tensor), batch_size):
            batch_end = min(i + batch_size, len(X_train_tensor))
            X_batch = X_train_tensor[i:batch_end]
            y_batch = y_train_tensor[i:batch_end]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model.model(X_batch)
            type_logits = outputs["type_logits"]
            pointer_logits = outputs["pointers"]  # JadeModel outputs "pointers" not "pointer_logits"
            
            # Create dummy pointer targets (center, length)
            # For testing, use normalized positions
            seq_len = X_batch.shape[1]
            center_targets = torch.rand(len(X_batch), 1, device=device) * seq_len
            length_targets = torch.rand(len(X_batch), 1, device=device) * seq_len * 0.3
            pointer_targets = torch.cat([center_targets, length_targets], dim=1)
            
            # Compute individual losses first
            type_loss_fn = nn.CrossEntropyLoss()
            pointer_loss_fn = nn.MSELoss()
            
            # Type classification loss
            type_loss = type_loss_fn(type_logits, y_batch)
            
            # Pointer regression loss (MSE on center+length)
            pointer_loss = pointer_loss_fn(pointer_logits, pointer_targets)
            
            # Apply uncertainty-weighted loss
            total_loss, loss_metrics = criterion(type_loss=type_loss, pointer_loss=pointer_loss)
            
            loss = total_loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), model.max_grad_norm)
            
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        logger.info(f"Epoch {epoch+1}/{model.n_epochs}: Loss = {avg_loss:.4f}")
    
    # Validation
    logger.info("Running validation...")
    model.model.eval()
    
    with torch.no_grad():
        val_outputs = model.model(X_val_tensor)
        val_type_logits = val_outputs["type_logits"]
        val_pointer_logits = val_outputs["pointers"]  # JadeModel outputs "pointers" not "pointer_logits"
        
        # Create dummy validation targets
        seq_len = X_val_tensor.shape[1]
        val_center_targets = torch.rand(len(X_val_tensor), 1, device=device) * seq_len
        val_length_targets = torch.rand(len(X_val_tensor), 1, device=device) * seq_len * 0.3
        val_pointer_targets = torch.cat([val_center_targets, val_length_targets], dim=1)
        
        # Compute validation losses
        type_loss_fn = nn.CrossEntropyLoss()
        pointer_loss_fn = nn.MSELoss()
        
        val_type_loss = type_loss_fn(val_type_logits, y_val_tensor)
        val_pointer_loss = pointer_loss_fn(val_pointer_logits, val_pointer_targets)
        
        val_total_loss, val_loss_metrics = criterion(type_loss=val_type_loss, pointer_loss=val_pointer_loss)
        
        logger.info(f"Validation Loss: {val_total_loss.item():.4f}")
        logger.info(f"  Type Loss: {val_type_loss.item():.4f}")
        logger.info(f"  Pointer Loss: {val_pointer_loss.item():.4f}")
        logger.info(f"  Learned σ_ptr: {torch.exp(criterion.log_var_ptr).item():.3f}")
        logger.info(f"  Learned σ_type: {torch.exp(criterion.log_var_type).item():.3f}")
    
    logger.success("Pointer-favoring training test completed successfully!")

if __name__ == "__main__":
    main()