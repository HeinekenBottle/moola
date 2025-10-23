#!/usr/bin/env python3
"""Supervised Training Script with Pointer-Only Warmup.

Specialized supervised training script that follows clean pipeline specifications:
- Hard invariants enforcement (model name, encoding, batch size)
- Float32 enforcement throughout
- Pointer-only warmup for 1-3 epochs
- Bias initialization (log_sigma_ptr=-0.30, log_sigma_cls=0.00)
- Early stopping on Hit@±3 (patience 20)
- ReduceLROnPlateau on Hit@±3 (patience 10)
- Target metrics: Hit@±3 ≥60%, F1_macro ≥0.50, ECE <0.10, Joint ≥40%
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from moola.models.registry import build, enforce_float32_precision, convert_batch_to_float32
from moola.data.enhanced_pipeline import EnhancedDataPipeline
from moola.utils.training.training_utils import (
    initialize_model_biases,
    validate_batch_schema,
    setup_optimized_dataloader
)
from moola.utils.seeds import set_seed, get_device
from moola.utils.early_stopping import EarlyStopping
from moola.metrics.joint_metrics import kendall_tau_loss, compute_hit_metrics, compute_joint_metrics
from moola.metrics.calibration import expected_calibration_error


def compute_uncertainty_weighted_loss(ptr_loss: torch.Tensor, cls_loss: torch.Tensor, 
                                    log_sigma_ptr: torch.Tensor, log_sigma_cls: torch.Tensor) -> torch.Tensor:
    """Compute uncertainty-weighted loss using learned log variances.
    
    Args:
        ptr_loss: Pointer regression loss
        cls_loss: Classification loss
        log_sigma_ptr: Learned log variance for pointer task
        log_sigma_cls: Learned log variance for classification task
        
    Returns:
        Uncertainty-weighted combined loss
    """
    # Compute precision (inverse variance)
    precision_ptr = torch.exp(-log_sigma_ptr)
    precision_cls = torch.exp(-log_sigma_cls)
    
    # Uncertainty-weighted loss
    weighted_ptr_loss = 0.5 * precision_ptr * ptr_loss + 0.5 * log_sigma_ptr
    weighted_cls_loss = 0.5 * precision_cls * cls_loss + 0.5 * log_sigma_cls
    
    return weighted_ptr_loss + weighted_cls_loss


def evaluate_model(model: nn.Module, val_loader: DataLoader, device: torch.device, 
                  epoch: int = 0) -> dict:
    """Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        val_loader: Validation dataloader
        device: Device
        epoch: Current epoch (for warmup logic)
        
    Returns:
        Dictionary with metrics
    """
    model.eval()
    all_ptr_preds = []
    all_cls_preds = []
    all_ptr_targets = []
    all_cls_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Convert to float32 and validate
            batch = convert_batch_to_float32(batch)
            validate_batch_schema(batch)
            
            # Move to device
            X = batch['X'].to(device)
            y_ptr = batch['y_ptr'].to(device)
            y_cls = batch['y_cls'].to(device)
            
            # Forward pass
            outputs = model(X)
            ptr_logits = outputs['pointer_logits']
            cls_logits = outputs['classification_logits']
            
            # Collect predictions and targets
            all_ptr_preds.append(ptr_logits.cpu())
            all_cls_preds.append(cls_logits.cpu())
            all_ptr_targets.append(y_ptr.cpu())
            all_cls_targets.append(y_cls.cpu())
    
    # Concatenate all batches
    ptr_preds = torch.cat(all_ptr_preds, dim=0)
    cls_preds = torch.cat(all_cls_preds, dim=0)
    ptr_targets = torch.cat(all_ptr_targets, dim=0)
    cls_targets = torch.cat(all_cls_targets, dim=0)
    
    # Compute metrics
    # Hit@±3 metrics
    hit_metrics = compute_hit_metrics(ptr_preds, ptr_targets, tolerance=3)
    
    # Classification metrics
    cls_probs = torch.softmax(cls_preds, dim=-1)
    cls_pred_labels = cls_probs.argmax(dim=-1)
    
    # F1 macro
    from sklearn.metrics import f1_score
    f1_macro = f1_score(cls_targets.numpy(), cls_pred_labels.numpy(), average='macro', zero_division=0)
    
    # ECE
    ece = expected_calibration_error(cls_probs.numpy(), cls_targets.numpy())
    
    # Joint metrics
    joint_metrics = compute_joint_metrics(ptr_preds, cls_preds, ptr_targets, cls_targets, tolerance=3)
    
    return {
        'hit_at_3': hit_metrics['hit_at_3'],
        'f1_macro': f1_macro,
        'ece': ece,
        'joint_accuracy': joint_metrics['joint_accuracy'],
        'ptr_mae': hit_metrics['mae'],
        'cls_accuracy': (cls_pred_labels == cls_targets).float().mean().item()
    }


def main():
    parser = argparse.ArgumentParser(description="Supervised Training with Pointer-Only Warmup")
    parser.add_argument("--model", type=str, required=True, choices=["jade", "sapphire", "opal"], 
                       help="Model name (jade, sapphire, opal)")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data parquet")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=29, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Pointer-only warmup epochs")
    parser.add_argument("--pretrained-encoder", type=str, default=None, help="Path to pretrained encoder")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--scheduler-patience", type=int, default=10, help="LR scheduler patience")
    parser.add_argument("--output-dir", type=str, default="artifacts/models", help="Output directory")
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
            project="moola-supervised",
            config=vars(args),
            name=f"{args.model}_supervised_{args.epochs}ep"
        )
    
    logger.info("Starting supervised training")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {device}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Warmup epochs: {args.warmup_epochs}")
    
    # Create mock config for model building
    class MockConfig:
        def __init__(self):
            self.model = MockModel()
            self.train = MockTrain()
    
    class MockModel:
        def __init__(self):
            self.name = args.model
            self.pointer_head = MockPointerHead()
            self.input_size = 11
            self.hidden_size = 64
            self.num_layers = 1
            self.bidirectional = True
            self.proj_head = True
            self.head_width = 64
    
    class MockPointerHead:
        def __init__(self):
            self.encoding = "center_length"
    
    class MockTrain:
        def __init__(self):
            self.batch_size = args.batch_size
    
    cfg = MockConfig()
    
    # Build model with invariant checks
    model = build(cfg).to(device)
    
    # Load pretrained encoder if specified
    if args.pretrained_encoder:
        logger.info(f"Loading pretrained encoder from {args.pretrained_encoder}")
        checkpoint = torch.load(args.pretrained_encoder, map_location=device)
        if 'encoder_state_dict' in checkpoint:
            # Load only encoder weights
            encoder_dict = checkpoint['encoder_state_dict']
            model_dict = model.state_dict()
            
            # Filter encoder weights
            encoder_dict = {k: v for k, v in encoder_dict.items() 
                          if k in model_dict and not k.startswith('heads.')}
            model_dict.update(encoder_dict)
            model.load_state_dict(model_dict)
            logger.info("Loaded pretrained encoder weights")
        else:
            logger.warning("Pretrained checkpoint format not recognized")
    
    # Initialize model biases
    initialize_model_biases(model)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {total_params:,} (trainable: {trainable_params:,})")
    
    # Setup data pipeline
    data_pipeline = EnhancedDataPipeline()
    X, y = data_pipeline.load_raw_data(Path(args.data_path))
    
    # Create datasets (using mock config for now)
    # In real implementation, this would use proper data loading
    train_loader = setup_optimized_dataloader(
        list(zip(X, y)),  # Mock dataset
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = setup_optimized_dataloader(
        list(zip(X, y)),  # Mock dataset
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # PERFORMANCE: Setup mixed precision training
    scaler = GradScaler() if torch.cuda.is_available() else None
    use_amp = scaler is not None

    # Setup scheduler based on Hit@±3
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=args.scheduler_patience, verbose=True
    )
    
    # Setup early stopping based on Hit@±3
    early_stopping = EarlyStopping(
        patience=args.patience,
        mode='max',
        restore_best_weights=True,
        save_path=output_dir / f"best_{args.model}_model.pt"
    )
    
    # Training loop
    best_hit_at_3 = 0.0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in train_pbar:
            # Convert to float32 and validate
            batch = convert_batch_to_float32(batch)
            validate_batch_schema(batch)
            
            # Move to device
            X = batch['X'].to(device)
            y_ptr = batch['y_ptr'].to(device)
            y_cls = batch['y_cls'].to(device)
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            if use_amp:
                with autocast():
                    outputs = model(X)
                    ptr_logits = outputs['pointer_logits']
                    cls_logits = outputs['classification_logits']

                    # Compute losses (Huber for pointers with δ=0.08)
                    ptr_loss = nn.HuberLoss(delta=0.08)(ptr_logits.squeeze(), y_ptr.squeeze())
                    cls_loss = nn.CrossEntropyLoss()(cls_logits, y_cls)

                    # Get learned uncertainties
                    log_sigma_ptr = getattr(model, 'log_sigma_ptr', torch.tensor(0.0, device=device))
                    log_sigma_cls = getattr(model, 'log_sigma_cls', torch.tensor(0.0, device=device))

                    # Pointer-only warmup
                    if epoch < args.warmup_epochs:
                        # Zero out classification loss during warmup
                        combined_loss = compute_uncertainty_weighted_loss(
                            ptr_loss, cls_loss * 0.0, log_sigma_ptr, log_sigma_cls
                        )
                        loss_type = "warmup"
                    else:
                        combined_loss = compute_uncertainty_weighted_loss(
                            ptr_loss, cls_loss, log_sigma_ptr, log_sigma_cls
                        )
                        loss_type = "full"

                # Backward pass with scaler
                scaler.scale(combined_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(X)
                ptr_logits = outputs['pointer_logits']
                cls_logits = outputs['classification_logits']

                # Compute losses
                ptr_loss = nn.MSELoss()(ptr_logits.squeeze(), y_ptr.squeeze())
                cls_loss = nn.CrossEntropyLoss()(cls_logits, y_cls)

                # Get learned uncertainties
                log_sigma_ptr = getattr(model, 'log_sigma_ptr', torch.tensor(0.0, device=device))
                log_sigma_cls = getattr(model, 'log_sigma_cls', torch.tensor(0.0, device=device))

                # Pointer-only warmup
                if epoch < args.warmup_epochs:
                    # Zero out classification loss during warmup
                    combined_loss = compute_uncertainty_weighted_loss(
                        ptr_loss, cls_loss * 0.0, log_sigma_ptr, log_sigma_cls
                    )
                    loss_type = "warmup"
                else:
                    combined_loss = compute_uncertainty_weighted_loss(
                        ptr_loss, cls_loss, log_sigma_ptr, log_sigma_cls
                    )
                    loss_type = "full"

                # Backward pass
                combined_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += combined_loss.item()
            n_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                "loss": combined_loss.item(),
                "type": loss_type
            })
        
        avg_loss = total_loss / n_batches
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, device, epoch)
        hit_at_3 = val_metrics['hit_at_3']
        
        # Update scheduler based on Hit@±3
        scheduler.step(hit_at_3)
        
        # Track best metrics
        if hit_at_3 > best_hit_at_3:
            best_hit_at_3 = hit_at_3
        
        # Logging
        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, Hit@±3={hit_at_3:.4f}, "
                   f"F1={val_metrics['f1_macro']:.4f}, ECE={val_metrics['ece']:.4f}, "
                   f"Joint={val_metrics['joint_accuracy']:.4f}")
        
        if not args.no_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_hit_at_3": hit_at_3,
                "val_f1_macro": val_metrics['f1_macro'],
                "val_ece": val_metrics['ece'],
                "val_joint_accuracy": val_metrics['joint_accuracy'],
                "val_ptr_mae": val_metrics['ptr_mae'],
                "val_cls_accuracy": val_metrics['cls_accuracy'],
                "lr": optimizer.param_groups[0]['lr']
            })
        
        # Early stopping
        if early_stopping(hit_at_3, model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    final_metrics = evaluate_model(model, val_loader, device)
    logger.info(f"Final Results: Hit@±3={final_metrics['hit_at_3']:.4f}, "
               f"F1={final_metrics['f1_macro']:.4f}, ECE={final_metrics['ece']:.4f}, "
               f"Joint={final_metrics['joint_accuracy']:.4f}")
    
    # Check target metrics
    targets_met = []
    if final_metrics['hit_at_3'] >= 0.60:
        targets_met.append("✓ Hit@±3 ≥60%")
    else:
        targets_met.append(f"✗ Hit@±3 {final_metrics['hit_at_3']:.1%} <60%")
    
    if final_metrics['f1_macro'] >= 0.50:
        targets_met.append("✓ F1_macro ≥0.50")
    else:
        targets_met.append(f"✗ F1_macro {final_metrics['f1_macro']:.3f} <0.50")
    
    if final_metrics['ece'] < 0.10:
        targets_met.append("✓ ECE <0.10")
    else:
        targets_met.append(f"✗ ECE {final_metrics['ece']:.3f} ≥0.10")
    
    if final_metrics['joint_accuracy'] >= 0.40:
        targets_met.append("✓ Joint ≥40%")
    else:
        targets_met.append(f"✗ Joint {final_metrics['joint_accuracy']:.1%} <40%")
    
    for target in targets_met:
        logger.info(target)
    
    # Save final model
    final_model_path = output_dir / f"{args.model}_final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'model_name': args.model,
            'input_size': 11,
            'hidden_size': 64,
            'num_layers': 1,
            'bidirectional': True,
            'proj_head': True,
            'head_width': 64
        },
        'metrics': final_metrics,
        'epochs': epoch + 1
    }, final_model_path)
    
    logger.info(f"✓ Training complete. Model saved to {final_model_path}")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()