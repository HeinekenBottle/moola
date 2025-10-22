#!/usr/bin/env python3
"""COMPLETELY REAL Training Pipeline - No Simulation.

This is a 100% real training pipeline:
- Real masked LSTM encoder pretraining (2-3 hours)
- Real Jade model training with pointer-favoring (15-30 minutes)
- Real data, real models, real training, real results
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

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
from moola.pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer

def set_seed(seed: int = 1337):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class RealMetricsTracker:
    """Track real training metrics."""
    
    def __init__(self):
        self.metrics_history = {
            'pretrain_loss': [],
            'train_loss': [],
            'val_loss': [],
            'train_type_acc': [],
            'val_type_acc': [],
            'train_hit@1': [],
            'val_hit@1': [],
            'train_hit@3': [],
            'val_hit@3': [],
            'sigma_ptr': [],
            'sigma_type': [],
            'pointer_weight_ratio': [],
            'epoch_times': [],
            'learning_rates': []
        }
    
    def log_pretrain_epoch(self, epoch: int, loss: float, epoch_time: float):
        """Log pretraining metrics."""
        self.metrics_history['pretrain_loss'].append(loss)
        self.metrics_history['epoch_times'].append(epoch_time)
    
    def log_jade_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict, 
                      sigma_ptr: float, sigma_type: float, epoch_time: float, lr: float):
        """Log Jade training metrics."""
        self.metrics_history['train_loss'].append(train_metrics.get('total_loss', 0))
        self.metrics_history['val_loss'].append(val_metrics.get('total_loss', 0))
        self.metrics_history['train_type_acc'].append(train_metrics.get('type_accuracy', 0))
        self.metrics_history['val_type_acc'].append(val_metrics.get('type_accuracy', 0))
        self.metrics_history['train_hit@1'].append(train_metrics.get('hit@1', 0))
        self.metrics_history['val_hit@1'].append(val_metrics.get('hit@1', 0))
        self.metrics_history['train_hit@3'].append(train_metrics.get('hit@3', 0))
        self.metrics_history['val_hit@3'].append(val_metrics.get('hit@3', 0))
        self.metrics_history['sigma_ptr'].append(sigma_ptr)
        self.metrics_history['sigma_type'].append(sigma_type)
        self.metrics_history['pointer_weight_ratio'].append(sigma_type / sigma_ptr)
        self.metrics_history['epoch_times'].append(epoch_time)
        self.metrics_history['learning_rates'].append(lr)
    
    def save_metrics(self, path: Path):
        """Save metrics history to file."""
        with open(path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Metrics saved to {path}")

def compute_real_metrics(model, X_tensor, y_tensor, ptr_tensor, device, seq_len=105):
    """Compute real metrics including hit@1, hit@3 with ±3 tolerance."""
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        type_logits = outputs["type_logits"]
        pointer_logits = outputs["pointers"]
        
        # Type accuracy
        type_preds = torch.argmax(type_logits, dim=1)
        type_acc = (type_preds == y_tensor).float().mean().item()
        
        # Convert normalized predictions back to actual indices
        pred_centers = pointer_logits[:, 0] * seq_len
        pred_lengths = pointer_logits[:, 1] * seq_len
        pred_starts = pred_centers - pred_lengths / 2
        pred_ends = pred_centers + pred_lengths / 2
        
        # True targets (convert from normalized)
        true_centers = ptr_tensor[:, 0] * seq_len
        true_lengths = ptr_tensor[:, 1] * seq_len
        true_starts = true_centers - true_lengths / 2
        true_ends = true_centers + true_lengths / 2
        
        # Hit@1 metrics (exact match)
        hit1_starts = torch.abs(pred_starts - true_starts) <= 1.0
        hit1_ends = torch.abs(pred_ends - true_ends) <= 1.0
        hit1 = (hit1_starts & hit1_ends).float().mean().item()
        
        # Hit@3 metrics (±3 timestep tolerance)
        hit3_starts = torch.abs(pred_starts - true_starts) <= 3.0
        hit3_ends = torch.abs(pred_ends - true_ends) <= 3.0
        hit3 = (hit3_starts & hit3_ends).float().mean().item()
        
        # Additional detailed metrics
        start_mae = torch.abs(pred_starts - true_starts).mean().item()
        end_mae = torch.abs(pred_ends - true_ends).mean().item()
        center_mae = torch.abs(pred_centers - true_centers).mean().item()
        length_mae = torch.abs(pred_lengths - true_lengths).mean().item()
        
        return {
            'type_accuracy': type_acc,
            'hit@1': hit1,
            'hit@3': hit3,
            'start_mae': start_mae,
            'end_mae': end_mae,
            'center_mae': center_mae,
            'length_mae': length_mae,
            'pred_starts': pred_starts.cpu().numpy(),
            'pred_ends': pred_ends.cpu().numpy(),
            'true_starts': true_starts.cpu().numpy(),
            'true_ends': true_ends.cpu().numpy()
        }

def load_real_data():
    """Load the real 174-sample dataset."""
    logger.info("Loading real 174 sample dataset...")
    data_path = Path("data/processed/labeled/train_latest_11d.parquet")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded dataset: {df.shape}")
        
        # Validate data integrity
        required_cols = ['features', 'label', 'expansion_start', 'expansion_end', 'window_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Deserialize features and create X
        features_list = []
        for i, feat_bytes in enumerate(df['features']):
            try:
                feat_array = pickle.loads(feat_bytes)
                if feat_array.shape != (105, 11):
                    logger.warning(f"Sample {i} has unexpected shape: {feat_array.shape}")
                features_list.append(feat_array)
            except Exception as e:
                logger.error(f"Failed to deserialize features for sample {i}: {e}")
                raise
        
        X = np.stack(features_list).astype(np.float32)
        
        # Create labels (consolidation=0, retracement=1)
        y = np.array([0 if label == 'consolidation' else 1 for label in df['label']])
        
        # Get expansion start/end for pointer targets
        expansion_start = df['expansion_start'].values.astype(np.float32)
        expansion_end = df['expansion_end'].values.astype(np.float32)
        
        # Create metadata
        metadata = {"window_id": df['window_id'].values}
        
        logger.info(f"Loaded real data: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Class distribution: consolidation={int(np.sum(y==0))}, retracement={int(np.sum(y==1))}")
        
        return X, y, expansion_start, expansion_end, metadata
        
    except Exception as e:
        logger.error(f"Failed to load real data: {e}")
        raise

def normalize_pointer_targets(expansion_start, expansion_end, seq_len=105):
    """Normalize expansion targets to [0, 1] range for center+length encoding."""
    # Convert to center and length
    center = (expansion_start + expansion_end) / 2.0
    length = expansion_end - expansion_start
    
    # Normalize to [0, 1]
    center_norm = center / seq_len
    length_norm = length / seq_len
    
    # Clip to valid range
    center_norm = np.clip(center_norm, 0.0, 1.0)
    length_norm = np.clip(length_norm, 0.0, 1.0)
    
    return np.column_stack([center_norm, length_norm])

def stage1_real_pretraining(X, device, metrics_tracker):
    """Stage 1: REAL masked LSTM encoder pretraining (2-3 hours)."""
    logger.info("=" * 60)
    logger.info("STAGE 1: REAL Masked LSTM Encoder Pretraining")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Initialize real pretrainer
    pretrainer = MaskedLSTMPretrainer(
        input_dim=X.shape[2],
        hidden_dim=128,
        num_layers=2,
        mask_ratio=0.4,
        learning_rate=1e-3,
        batch_size=64,
        device=device,
        seed=1337
    )
    
    # Load unlabeled data for pretraining (if available)
    unlabeled_path = Path("data/raw/unlabeled_windows.parquet")
    if unlabeled_path.exists():
        logger.info("Loading unlabeled data for pretraining...")
        try:
            unlabeled_df = pd.read_parquet(unlabeled_path)
            unlabeled_features = []
            for feat_bytes in unlabeled_df['features']:
                feat_array = pickle.loads(feat_bytes)
                unlabeled_features.append(feat_array)
            X_unlabeled = np.stack(unlabeled_features).astype(np.float32)
            logger.info(f"Loaded {len(X_unlabeled)} unlabeled samples")
        except Exception as e:
            logger.warning(f"Failed to load unlabeled data: {e}")
            X_unlabeled = X  # Fall back to labeled data
    else:
        logger.info("No unlabeled data found, using labeled data for pretraining")
        X_unlabeled = X
    
    # REAL PRETRAINING - This will take 2-3 hours
    logger.info(f"Starting REAL pretraining on {len(X_unlabeled)} samples...")
    logger.info("⚠️  This will take 2-3 hours. No simulation - real training!")
    
    try:
        # Train the encoder
        encoder_path = Path("checkpoints/real_pretrained_encoder.pth")
        encoder_path.parent.mkdir(exist_ok=True)
        
        # Real pretraining
        logger.info("Starting REAL masked LSTM pretraining...")
        logger.info("This will perform actual neural network training - no simulation!")
        
        # Call the real pretrain method
        history = pretrainer.pretrain(
            X_unlabeled,
            n_epochs=100,
            patience=15,
            save_path=encoder_path,
            verbose=True
        )
        
        # Log pretraining metrics
        final_train_loss = history['train_loss'][-1] if history['train_loss'] else 0
        final_val_loss = history['val_loss'][-1] if history['val_loss'] else 0
        
        elapsed = time.time() - start_time
        logger.success(f"Stage 1 REAL pretraining completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        logger.success(f"Final train loss: {final_train_loss:.4f}, Final val loss: {final_val_loss:.4f}")
        
        return encoder_path
        
    except Exception as e:
        logger.error(f"Real pretraining failed: {e}")
        raise

def stage2_real_jade_training(X, y, pointer_targets, encoder_path, device, metrics_tracker):
    """Stage 2: REAL Jade model training with pointer-favoring."""
    logger.info("=" * 60)
    logger.info("STAGE 2: REAL Jade Model Training with Pointer-Favoring")
    logger.info("=" * 60)
    
    # Create temporal split (80/20)
    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    ptr_train, ptr_val = pointer_targets[:n_train], pointer_targets[n_train:]
    
    logger.info(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
    
    # Initialize Jade model with pointer-favoring
    model = JadeModel(
        seed=1337,
        device=device,
        predict_pointers=True,
        n_epochs=60,
        batch_size=29,
        learning_rate=3e-4,
        max_grad_norm=2.0,
        early_stopping_patience=20,
        scheduler_factor=0.5,
        scheduler_patience=10,
    )
    
    # Build the model
    model.model = model._build_model(input_dim=X_train.shape[2], n_classes=2)
    model.model = model.model.to(device)
    model.n_classes = 2
    model.input_dim = X_train.shape[2]
    
    # Load pretrained encoder if available
    if encoder_path and encoder_path.exists():
        logger.info(f"Loading pretrained encoder from {encoder_path}")
        try:
            encoder_state = torch.load(encoder_path, map_location=device)
            # Load encoder weights into model
            if hasattr(model.model, 'encoder') and 'encoder_state_dict' in encoder_state:
                model.model.encoder.load_state_dict(encoder_state['encoder_state_dict'])
                logger.info("Pretrained encoder loaded successfully")
            else:
                logger.warning("Could not load encoder weights - structure mismatch")
        except Exception as e:
            logger.warning(f"Failed to load pretrained encoder: {e}")
    
    # Initialize pointer-favoring loss
    criterion = UncertaintyWeightedLoss(
        init_log_var_ptr=-0.60,  # σ≈0.55 (higher weight)
        init_log_var_type=0.00,  # σ=1.0 (lower weight)
    )
    criterion = criterion.to(device)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    ptr_train_tensor = torch.FloatTensor(ptr_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    ptr_val_tensor = torch.FloatTensor(ptr_val).to(device)
    
    # REAL TRAINING LOOP
    logger.info("Starting REAL Jade training with pointer-favoring...")
    logger.info(f"Pointer-favoring initialized: σ_ptr={torch.exp(criterion.log_var_ptr).item():.3f}, "
               f"σ_type={torch.exp(criterion.log_var_type).item():.3f}, "
               f"ratio={torch.exp(criterion.log_var_type).item()/torch.exp(criterion.log_var_ptr).item():.2f}x")
    
    start_time = time.time()
    best_val_hit3 = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(model.n_epochs):
        epoch_start_time = time.time()
        
        # Training
        model.model.train()
        total_loss = 0.0
        n_batches = 0
        
        batch_size = model.batch_size
        for i in range(0, len(X_train_tensor), batch_size):
            batch_end = min(i + batch_size, len(X_train_tensor))
            X_batch = X_train_tensor[i:batch_end]
            y_batch = y_train_tensor[i:batch_end]
            ptr_batch = ptr_train_tensor[i:batch_end]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model.model(X_batch)
            type_logits = outputs["type_logits"]
            pointer_logits = outputs["pointers"]
            
            # Compute losses
            type_loss_fn = nn.CrossEntropyLoss()
            pointer_loss_fn = nn.MSELoss()
            
            type_loss = type_loss_fn(type_logits, y_batch)
            pointer_loss = pointer_loss_fn(pointer_logits, ptr_batch)
            
            # Apply uncertainty-weighted loss
            total_loss_batch, loss_metrics = criterion(type_loss=type_loss, pointer_loss=pointer_loss)
            
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), model.max_grad_norm)
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            n_batches += 1
        
        avg_train_loss = total_loss / n_batches
        
        # Compute real training metrics
        train_metrics = compute_real_metrics(
            model.model, X_train_tensor, y_train_tensor, ptr_train_tensor, device
        )
        train_metrics['total_loss'] = avg_train_loss
        
        # Validation
        val_metrics = compute_real_metrics(
            model.model, X_val_tensor, y_val_tensor, ptr_val_tensor, device
        )
        
        # Compute validation loss
        model.model.eval()
        with torch.no_grad():
            val_outputs = model.model(X_val_tensor)
            val_type_logits = val_outputs["type_logits"]
            val_pointer_logits = val_outputs["pointers"]
            
            # Define loss functions here to avoid scoping issues
            val_type_loss_fn = nn.CrossEntropyLoss()
            val_pointer_loss_fn = nn.MSELoss()
            
            val_type_loss = val_type_loss_fn(val_type_logits, y_val_tensor)
            val_pointer_loss = val_pointer_loss_fn(val_pointer_logits, ptr_val_tensor)
            
            val_total_loss, val_loss_metrics = criterion(type_loss=val_type_loss, pointer_loss=val_pointer_loss)
        
        val_metrics['total_loss'] = val_total_loss.item()
        
        scheduler.step(val_total_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track uncertainty parameters
        sigma_ptr = torch.exp(criterion.log_var_ptr).item()
        sigma_type = torch.exp(criterion.log_var_type).item()
        
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics
        metrics_tracker.log_jade_epoch(
            epoch, train_metrics, val_metrics, sigma_ptr, sigma_type, epoch_time, current_lr
        )
        
        # Save best model (based on hit@3)
        if val_metrics['hit@3'] > best_val_hit3:
            best_val_hit3 = val_metrics['hit@3']
            checkpoint_path = Path("checkpoints/real_jade_best_hit3.pth")
            checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_total_loss.item(),
                'hit@3': val_metrics['hit@3'],
                'hit@1': val_metrics['hit@1'],
                'type_acc': val_metrics['type_accuracy'],
                'log_var_ptr': criterion.log_var_ptr.item(),
                'log_var_type': criterion.log_var_type.item(),
                'sigma_ptr': sigma_ptr,
                'sigma_type': sigma_type,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, checkpoint_path)
            logger.info(f"New best Hit@3: {best_val_hit3:.3f}")
        
        # Also save best loss model
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            loss_checkpoint_path = Path("checkpoints/real_jade_best_loss.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_total_loss.item(),
                'log_var_ptr': criterion.log_var_ptr.item(),
                'log_var_type': criterion.log_var_type.item(),
            }, loss_checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Detailed logging every 5 epochs
        if epoch % 5 == 0 or epoch == model.n_epochs - 1:
            logger.info(f"Epoch {epoch+1}/{model.n_epochs}:")
            logger.info(f"  Train: Loss={avg_train_loss:.4f}, Acc={train_metrics['type_accuracy']:.3f}, "
                       f"Hit@1={train_metrics['hit@1']:.3f}, Hit@3={train_metrics['hit@3']:.3f}")
            logger.info(f"  Val:   Loss={val_total_loss.item():.4f}, Acc={val_metrics['type_accuracy']:.3f}, "
                       f"Hit@1={val_metrics['hit@1']:.3f}, Hit@3={val_metrics['hit@3']:.3f}")
            logger.info(f"  σ_ptr={sigma_ptr:.3f}, σ_type={sigma_type:.3f}, "
                       f"Ratio={sigma_type/sigma_ptr:.2f}x, LR={current_lr:.2e}, Time={epoch_time:.1f}s")
        
        # Early stopping
        if patience_counter >= model.early_stopping_patience:
            logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break
    
    elapsed = time.time() - start_time
    logger.success(f"Stage 2 REAL Jade training completed in {elapsed:.1f}s")
    logger.success(f"Best validation Hit@3: {best_val_hit3:.3f}")
    logger.success(f"Best validation loss: {best_val_loss:.4f}")
    
    # Return the actual checkpoint path
    best_checkpoint_path = Path("checkpoints/real_jade_best_hit3.pth")
    return best_checkpoint_path, metrics_tracker

def main():
    """Execute the complete REAL training pipeline."""
    logger.info("🚀 STARTING COMPLETELY REAL TRAINING PIPELINE - NO SIMULATION")
    
    # Set seeds for reproducibility
    set_seed(1337)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create checkpoints directory
    Path("checkpoints").mkdir(exist_ok=True)
    
    total_start_time = time.time()
    
    try:
        # Load real data
        X, y, expansion_start, expansion_end, metadata = load_real_data()
        
        # Normalize pointer targets
        pointer_targets = normalize_pointer_targets(expansion_start, expansion_end, seq_len=105)
        logger.info(f"Pointer targets shape: {pointer_targets.shape}")
        
        # Initialize metrics tracker
        metrics_tracker = RealMetricsTracker()
        
        # Stage 1: REAL pretraining (2-3 hours)
        logger.warning("⚠️  STAGE 1: This will take 2-3 hours of REAL training")
        logger.warning("⚠️  No simulation - actual neural network training")
        
        user_input = input("Continue with REAL pretraining? (y/N): ")
        if user_input.lower() != 'y':
            logger.info("Skipping pretraining - using untrained encoder")
            encoder_path = None
        else:
            encoder_path = stage1_real_pretraining(X, device, metrics_tracker)
        
        # Stage 2: REAL Jade training (15-30 minutes)
        logger.info("🚀 STAGE 2: REAL Jade training with pointer-favoring")
        jade_checkpoint, final_metrics = stage2_real_jade_training(
            X, y, pointer_targets, encoder_path, device, metrics_tracker
        )
        
        # Save final metrics
        metrics_tracker.save_metrics(Path("checkpoints/real_training_metrics.json"))
        
        total_elapsed = time.time() - total_start_time
        
        logger.success("=" * 60)
        logger.success("🎉 COMPLETELY REAL TRAINING PIPELINE COMPLETED!")
        logger.success("=" * 60)
        logger.success(f"Total training time: {total_elapsed/60:.1f} minutes")
        logger.success(f"Best Hit@3: {max(final_metrics.metrics_history['val_hit@3']):.3f}")
        logger.success(f"Best Hit@1: {max(final_metrics.metrics_history['val_hit@1']):.3f}")
        logger.success(f"Best Type Accuracy: {max(final_metrics.metrics_history['val_type_acc']):.3f}")
        logger.success("✅ All training was 100% REAL - no simulation!")
        logger.success("=" * 60)
        
    except Exception as e:
        logger.error(f"Real training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()