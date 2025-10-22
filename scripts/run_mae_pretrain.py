#!/usr/bin/env python3
"""MAE Pretraining Script for OHLC Data.

Specialized pretraining script that follows clean pipeline specifications:
- 11 months OHLC data only
- Float32 enforcement
- Light augmentation (jitter σ=0.01)
- Target: val MAE within 5-15% of train MAE
- Output: artifacts/encoders/pretrained/stones_encoder_mae.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from moola.data.pretrain_pipeline import PretrainPipeline, validate_pretrain_batch
from moola.encoder.bilstm_masked_autoencoder import BiLSTMMaskedAutoencoder
from moola.utils.training.training_utils import (
    enforce_float32_precision, 
    convert_batch_to_float32,
    initialize_model_biases
)
from moola.utils.seeds import set_seed, get_device
from moola.utils.early_stopping import EarlyStopping


def apply_light_augmentation(batch: dict, sigma: float = 0.01) -> dict:
    """Apply light jitter augmentation for pretraining.
    
    Args:
        batch: Batch dictionary with 'X' and 'target'
        sigma: Jitter standard deviation
        
    Returns:
        Augmented batch
    """
    X = batch['X']
    target = batch['target']
    
    # Apply light jitter to input only (not target)
    noise = torch.randn_like(X) * sigma
    X_aug = X + noise
    
    return {
        'X': X_aug,
        'target': target,
        'mask': batch['mask']
    }


def compute_mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked MAE loss.
    
    Args:
        pred: Predicted reconstruction [batch, seq_len, features]
        target: Target reconstruction [batch, seq_len, features]
        mask: Boolean mask [batch, seq_len]
        
    Returns:
        Masked MAE loss
    """
    # Compute absolute error
    abs_error = torch.abs(pred - target)  # [batch, seq_len, features]
    
    # Apply mask to timesteps
    mask_expanded = mask.unsqueeze(-1)  # [batch, seq_len, 1]
    masked_error = abs_error * mask_expanded
    
    # Compute mean over masked timesteps and features
    n_masked = mask_expanded.sum()
    if n_masked > 0:
        loss = masked_error.sum() / n_masked
    else:
        loss = torch.tensor(0.0, device=pred.device)
    
    return loss


def validate_pretrain_model(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Validate pretraining model.
    
    Args:
        model: Pretraining model
        val_loader: Validation dataloader
        device: Device
        
    Returns:
        Validation MAE loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Convert to float32 and validate
            batch = convert_batch_to_float32(batch)
            validate_pretrain_batch(batch)
            
            # Move to device
            X = batch['X'].to(device)
            target = batch['target'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass
            pred = model(X, mask)
            
            # Compute loss
            loss = compute_mae_loss(pred, target, mask)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="MAE Pretraining for OHLC Data")
    parser.add_argument("--data-path", type=str, required=True, help="Path to OHLC data parquet")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--mask-ratio", type=float, default=0.4, help="Mask ratio")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--jitter-sigma", type=float, default=0.01, help="Jitter augmentation sigma")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--output-dir", type=str, default="artifacts/encoders/pretrained", help="Output directory")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    enforce_float32_precision()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project="moola-pretrain",
            config=vars(args),
            name=f"mae_pretrain_ohlc_{args.epochs}ep"
        )
    
    logger.info("Starting MAE pretraining")
    logger.info(f"Device: {device}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output dir: {output_dir}")
    
    # Load data
    pipeline = PretrainPipeline(Path(args.data_path), months=11)
    ohlc_data = pipeline.load_ohlc_data()
    
    # Create datasets and dataloaders
    train_dataset, val_dataset = pipeline.create_datasets(
        ohlc_data, val_split=0.1, mask_ratio=args.mask_ratio
    )
    train_loader, val_loader = pipeline.create_dataloaders(
        train_dataset, val_dataset, batch_size=args.batch_size, num_workers=4
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset) if val_dataset else 0}")
    
    # Create model
    model = BiLSTMMaskedAutoencoder(
        input_dim=4,  # OHLC
        hidden_dim=128,
        num_layers=2,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {total_params:,}")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=1e-6,
        restore_best_weights=True,
        save_path=output_dir / "best_encoder.pt"
    )
    
    # Training loop
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in train_pbar:
            # Convert to float32 and validate
            batch = convert_batch_to_float32(batch)
            validate_pretrain_batch(batch)
            
            # Apply light augmentation
            batch = apply_light_augmentation(batch, sigma=args.jitter_sigma)
            
            # Move to device
            X = batch['X'].to(device)
            target = batch['target'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            pred = model(X, mask)
            
            # Compute loss
            loss = compute_mae_loss(pred, target, mask)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = train_loss / n_batches
        
        # Validation
        if val_loader:
            avg_val_loss = validate_pretrain_model(model, val_loader, device)
        else:
            avg_val_loss = 0.0
        
        # Update scheduler
        if val_loader:
            scheduler.step(avg_val_loss)
        
        # Track best losses
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        # Logging
        logger.info(f"Epoch {epoch+1}: Train MAE={avg_train_loss:.6f}, Val MAE={avg_val_loss:.6f}")
        
        if not args.no_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_mae": avg_train_loss,
                "val_mae": avg_val_loss,
                "lr": optimizer.param_groups[0]['lr']
            })
        
        # Early stopping
        if val_loader:
            if early_stopping(avg_val_loss, model.encoder):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Final validation
    if val_loader:
        val_gap_pct = ((avg_val_loss - avg_train_loss) / avg_train_loss) * 100
        logger.info(f"Final: Train MAE={best_train_loss:.6f}, Val MAE={best_val_loss:.6f}")
        logger.info(f"Validation gap: {val_gap_pct:.1f}%")
        
        # Check if validation gap is within target range (5-15%)
        if 5 <= val_gap_pct <= 15:
            logger.info("✓ Validation gap within target range (5-15%)")
        else:
            logger.warning(f"⚠ Validation gap {val_gap_pct:.1f}% outside target range (5-15%)")
    
    # Save final encoder
    final_encoder_path = output_dir / "stones_encoder_mae.pt"
    torch.save({
        'encoder_state_dict': model.encoder.state_dict(),
        'config': {
            'input_dim': 4,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': args.dropout
        },
        'train_loss': best_train_loss,
        'val_loss': best_val_loss,
        'epochs': epoch + 1
    }, final_encoder_path)
    
    logger.info(f"✓ Pretraining complete. Encoder saved to {final_encoder_path}")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()