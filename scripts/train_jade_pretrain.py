#!/usr/bin/env python3
"""Jade Pretraining Script.

Implements masked autoencoder pretraining on 5-year NQ data:
- Optim: AdamW lr=1e-3, betas=(0.9,0.999), wd=1e-2
- Scheduler: cosine decay, warmup 5 epochs  
- Epochs: 50
- Batch size: 256 windows
- Mixed precision: fp16 if available
- Seeds: 17 (and sweeps 13,23,29 for stability)

Usage:
    python scripts/train_jade_pretrain.py --config configs/windowed.yaml --data data/raw/nq_5year.parquet
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from pydantic import BaseModel

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from moola.data.windowed_loader import WindowedConfig, create_dataloaders, save_split_manifest
from moola.models.jade_pretrain import JadeConfig, JadePretrainer
from moola.utils.seeds import set_seed


class TrainingConfig(BaseModel):
    """Training configuration."""
    # Model config
    model: JadeConfig = JadeConfig(
        input_size=10,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        huber_delta=1.0
    )
    
    # Training hyperparameters (Grok's pre-train params)
    epochs: int = 20  # 10-20 epochs
    batch_size: int = 64  # Batch size 64 for pre-training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    warmup_epochs: int = 5
    
    # Optimization
    grad_clip: float = 1.0
    mixed_precision: bool = True
    
    # Seeds for stability
    seeds: List[int] = [17, 13, 23, 29]
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-6
    
    # Checkpointing
    save_top_k: int = 3
    eval_every: int = 1
    
    # Data
    num_workers: int = 4
    pin_memory: bool = True


class CosineWarmupScheduler:
    """Cosine scheduler with linear warmup."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int, 
                 total_epochs: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
        
        # Cosine scheduler after warmup
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=total_epochs - warmup_epochs,
            eta_min=base_lr * 0.1  # Final LR is 10% of base LR
        )
    
    def step(self):
        """Update learning rate."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Cosine decay
            self.cosine_scheduler.step()
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        """Get current learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class JadeTrainer:
    """Trainer for Jade pretraining."""
    
    def __init__(self, config: TrainingConfig, windowed_config: WindowedConfig):
        self.config = config
        self.windowed_config = windowed_config
        
        # Create output directory
        self.output_dir = Path("artifacts/jade_pretrain")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_seed = config.seeds[0]
        self.best_val_loss = float('inf')
        self.checkpoints = []
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
    def setup_model_and_optimizer(self) -> Tuple[JadePretrainer, torch.optim.Optimizer, CosineWarmupScheduler, Optional[GradScaler]]:
        """Setup model, optimizer, scheduler, and scaler."""
        # Create model
        model = JadePretrainer(self.config.model)
        
        # Ensure model is on GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(self.device)
        print(f"Model moved to device: {self.device}")
        print(f"Model device check: {next(model.parameters()).device}")
        
        # Create optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.config.weight_decay,
            eps=1e-8
        )
        
        # Create scheduler
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_epochs=self.config.warmup_epochs,
            total_epochs=self.config.epochs,
            base_lr=self.config.learning_rate
        )
        
        # Create gradient scaler for mixed precision - only enable if CUDA is available
        scaler = GradScaler() if (self.config.mixed_precision and torch.cuda.is_available()) else None
        
        return model, optimizer, scheduler, scaler
    
    def train_epoch(self, model: JadePretrainer, train_loader, optimizer: torch.optim.Optimizer,
                   scaler: Optional[GradScaler], epoch: int) -> Dict:
        """Train for one epoch."""
        model.train()
        epoch_metrics = []
        
        # Monitor GPU usage
        if torch.cuda.is_available():
            print(f"Training on GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory at start: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # CRITICAL FIX: Move batch tensors to device BEFORE forward pass
            if isinstance(batch, (list, tuple)):
                # Handle tuple format: (X, mask, valid_mask)
                X, mask, valid_mask = batch
                X = X.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True) 
                valid_mask = valid_mask.to(self.device, non_blocking=True)
                batch = (X, mask, valid_mask)
            elif isinstance(batch, dict):
                # Handle dict format
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            # Verify tensor placement
            if batch_idx == 0 and torch.cuda.is_available():
                if isinstance(batch, (list, tuple)):
                    print(f"Batch devices: X={batch[0].device}, mask={batch[1].device}, valid={batch[2].device}")
                else:
                    print(f"Batch devices: {[v.device for v in batch.values()]}")
            
            # Forward pass with mixed precision
            if scaler:
                with autocast():
                    loss, metrics = model(batch)
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, metrics = model(batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                
                optimizer.step()
            
            # Track metrics
            metrics.update({
                'batch': batch_idx,
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr']
            })
            epoch_metrics.append(metrics)
            
            # GPU monitoring every 10 batches
            if torch.cuda.is_available() and batch_idx % 10 == 0:
                gpu_memory = torch.cuda.memory_allocated() / 1e6
                print(f"  Batch {batch_idx}: Loss {loss.item():.6f}, GPU Memory {gpu_memory:.1f} MB")
        
        # Aggregate epoch metrics
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            if isinstance(epoch_metrics[0][key], (int, float)):
                avg_metrics[f'train_{key}'] = np.mean([m[key] for m in epoch_metrics])
        
        return avg_metrics
    
    def validate_epoch(self, model: JadePretrainer, val_loader, epoch: int) -> Dict:
        """Validate for one epoch."""
        model.eval()
        epoch_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # CRITICAL FIX: Move batch tensors to device BEFORE forward pass
                if isinstance(batch, (list, tuple)):
                    # Handle tuple format: (X, mask, valid_mask)
                    X, mask, valid_mask = batch
                    X = X.to(self.device, non_blocking=True)
                    mask = mask.to(self.device, non_blocking=True) 
                    valid_mask = valid_mask.to(self.device, non_blocking=True)
                    batch = (X, mask, valid_mask)
                elif isinstance(batch, dict):
                    # Handle dict format
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                
                loss, metrics = model(batch)
                
                metrics.update({
                    'batch': batch_idx,
                    'epoch': epoch
                })
                epoch_metrics.append(metrics)
        
        # Aggregate epoch metrics
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            if isinstance(epoch_metrics[0][key], (int, float)):
                avg_metrics[f'val_{key}'] = np.mean([m[key] for m in epoch_metrics])
        
        return avg_metrics
    
    def save_checkpoint(self, model: JadePretrainer, optimizer: torch.optim.Optimizer, 
                       epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.model_dump(),
            'windowed_config': self.windowed_config.model_dump(),
            'seed': self.current_seed
        }
        
        # Save latest
        checkpoint_path = self.output_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
        
        # Save in top-k list with unique path
        val_loss = metrics.get('val_loss', float('inf'))
        top_k_path = self.output_dir / f'checkpoint_epoch_{epoch}_loss_{val_loss:.6f}.pt'
        torch.save(checkpoint, top_k_path)
        self.checkpoints.append((val_loss, epoch, top_k_path))
        
        # Keep only top-k checkpoints
        self.checkpoints.sort(key=lambda x: x[0])
        if len(self.checkpoints) > self.config.save_top_k:
            _, _, old_path = self.checkpoints.pop()
            if old_path.exists():
                old_path.unlink()
        
        # Update top_k symbolic links
        for i, (_, _, path) in enumerate(self.checkpoints):
            new_path = self.output_dir / f'checkpoint_top_{i+1}.pt'
            if new_path.exists():
                new_path.unlink()
            new_path.symlink_to(path.name)
    
    def train(self, data_path: str):
        """Main training loop."""
        print(f"Starting Jade pretraining with seed {self.current_seed}")
        print(f"Output directory: {self.output_dir}")
        
        # Set seed
        set_seed(self.current_seed)
        
        # Create dataloaders
        print("Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            pd.read_parquet(data_path),
            self.windowed_config,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Setup model and optimizer
        model, optimizer, scheduler, scaler = self.setup_model_and_optimizer()
        
        print(f"Model info: {model.get_model_info()}")
        
        # Training loop
        print(f"Training for {self.config.epochs} epochs...")
        start_time = time.time()
        
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(model, train_loader, optimizer, scaler, epoch)
            
            # Validate
            if epoch % self.config.eval_every == 0:
                val_metrics = self.validate_epoch(model, val_loader, epoch)
            else:
                val_metrics = {}
            
            # Update scheduler
            scheduler.step()
            
            # Combine metrics
            combined_metrics = {**train_metrics, **val_metrics}
            combined_metrics['epoch'] = epoch
            combined_metrics['lr'] = scheduler.get_last_lr()[0]
            combined_metrics['epoch_time'] = time.time() - epoch_start
            
            # Track metrics
            self.train_metrics.append(train_metrics)
            if val_metrics:
                self.val_metrics.append(val_metrics)
            
            # Print progress
            val_loss = val_metrics.get('val_loss', 0.0)
            train_loss = train_metrics.get('train_loss', 0.0)
            
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {combined_metrics['lr']:.6f} | "
                  f"Time: {combined_metrics['epoch_time']:.1f}s")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                patience_counter = 0
                print(f"  *** New best validation loss: {val_loss:.6f} ***")
            else:
                patience_counter += 1
            
            self.save_checkpoint(model, optimizer, epoch, combined_metrics, is_best)
            
            # Early stopping
            if patience_counter >= self.config.patience:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        # Final evaluation on test set
        print("Evaluating on test set...")
        test_metrics = self.validate_epoch(model, test_loader, self.config.epochs)
        print(f"Test loss: {test_metrics.get('val_loss', 0.0):.6f}")
        
        # Save final results
        results = {
            'best_val_loss': self.best_val_loss,
            'test_metrics': test_metrics,
            'total_time': total_time,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'config': self.config.model_dump(),
            'windowed_config': self.windowed_config.model_dump(),
            'seed': self.current_seed
        }
        
        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_path}")
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pretrain Jade on NQ data")
    parser.add_argument("--config", required=True, help="Path to windowed config")
    parser.add_argument("--data", required=True, help="Path to NQ parquet file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--output-dir", default="artifacts/jade_pretrain", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        # Load configs
        with open(args.config, 'r') as f:
            windowed_config_dict = yaml.safe_load(f)
        windowed_config = WindowedConfig(**windowed_config_dict)
        
        # Create training config
        training_config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seeds=[args.seed]
        )
        
        # Create trainer
        trainer = JadeTrainer(training_config, windowed_config)
        trainer.output_dir = Path(args.output_dir)
        
        # Train
        results = trainer.train(args.data)
        
        print("Jade pretraining completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())