#!/usr/bin/env python3
"""Fast Jade Pretraining Script using pre-computed features.

PERFORMANCE IMPROVEMENT:
- Old workflow: 1-3 hours feature computation + X hours training
- New workflow: ~5 seconds feature loading + X hours training
- Speedup: 720-2160x faster iteration

Usage:
    # Train with pre-computed features
    python3 scripts/train_jade_pretrain_fast.py \
        --feature-dir data/processed/nq_features \
        --epochs 50 \
        --batch-size 256 \
        --lr 1e-3
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from moola.data.fast_windowed_loader import (
    create_fast_dataloaders,
    create_strided_dataloaders,
)
from moola.models.jade_pretrain import JadeConfig, JadePretrainer
from moola.utils.seeds import set_seed


class CosineWarmupScheduler:
    """Cosine scheduler with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.current_epoch = 0

        # Cosine scheduler after warmup
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=base_lr * 0.1
        )

    def step(self):
        """Update learning rate."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        else:
            # Cosine decay
            self.cosine_scheduler.step()

        self.current_epoch += 1

    def get_last_lr(self):
        """Get current learning rate."""
        return [param_group["lr"] for param_group in self.optimizer.param_groups]


def train_epoch(model, train_loader, optimizer, scaler, device, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        # Move batch to device
        X, mask, valid_mask = batch
        X = X.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)
        batch = (X, mask, valid_mask)

        # Forward pass with mixed precision
        if scaler:
            with autocast():
                loss, metrics = model(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, metrics = model(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # Print progress every 50 batches
        if batch_idx % 50 == 0:
            print(f"    Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.6f}")

    return total_loss / n_batches


def validate_epoch(model, val_loader, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            X, mask, valid_mask = batch
            X = X.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            batch = (X, mask, valid_mask)

            loss, metrics = model(batch)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fast Jade pretraining using pre-computed features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with pre-computed features (full dataset)
  python3 scripts/train_jade_pretrain_fast.py \\
      --feature-dir data/processed/nq_features \\
      --epochs 50 \\
      --batch-size 256

  # Train with strided windows (faster, 50% overlap)
  python3 scripts/train_jade_pretrain_fast.py \\
      --feature-dir data/processed/nq_features \\
      --stride 52 \\
      --epochs 50 \\
      --batch-size 256
        """,
    )
    parser.add_argument("--feature-dir", required=True, help="Pre-computed feature directory")
    parser.add_argument("--stride", type=int, help="Window stride (optional, for faster training)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--output-dir", default="artifacts/jade_pretrain", help="Output directory")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision")

    args = parser.parse_args()

    try:
        print("=" * 80)
        print("Fast Jade Pretraining")
        print("=" * 80)
        print(f"Feature directory: {args.feature_dir}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Seed: {args.seed}")

        # Set seed
        set_seed(args.seed)

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create dataloaders
        print("\n" + "=" * 80)
        print("Creating dataloaders...")
        print("=" * 80)

        load_start = time.time()

        if args.stride:
            train_loader, val_loader, test_loader = create_strided_dataloaders(
                args.feature_dir,
                stride=args.stride,
                batch_size=args.batch_size,
                num_workers=4,
                pin_memory=True,
                seed=args.seed,
            )
        else:
            train_loader, val_loader, test_loader = create_fast_dataloaders(
                args.feature_dir,
                batch_size=args.batch_size,
                num_workers=4,
                pin_memory=True,
                seed=args.seed,
            )

        load_time = time.time() - load_start

        print(f"\nDataloader creation time: {load_time:.2f}s")
        print(f"Speedup vs on-the-fly computation: ~{3600/load_time:.0f}x")

        # Create model
        print("\n" + "=" * 80)
        print("Creating model...")
        print("=" * 80)

        model_config = JadeConfig(
            input_size=10, hidden_size=128, num_layers=2, dropout=0.2, huber_delta=1.0
        )

        model = JadePretrainer(model_config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        print(f"Model: {model.get_model_info()}")
        print(f"Device: {device}")

        # Create optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
            eps=1e-8,
        )

        # Create scheduler
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs, base_lr=args.lr
        )

        # Create gradient scaler for mixed precision
        use_amp = torch.cuda.is_available() and not args.no_mixed_precision
        scaler = GradScaler() if use_amp else None

        print(f"Mixed precision: {use_amp}")

        # Training loop
        print("\n" + "=" * 80)
        print(f"Training for {args.epochs} epochs...")
        print("=" * 80)

        best_val_loss = float("inf")
        patience_counter = 0
        train_start = time.time()

        for epoch in range(args.epochs):
            epoch_start = time.time()

            # Train
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            train_loss = train_epoch(model, train_loader, optimizer, scaler, device, args.grad_clip)

            # Validate
            val_loss = validate_epoch(model, val_loader, device)

            # Update scheduler
            scheduler.step()

            epoch_time = time.time() - epoch_start
            lr = scheduler.get_last_lr()[0]

            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  LR:         {lr:.6f}")
            print(f"  Time:       {epoch_time:.1f}s")

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": model_config.model_dump(),
                    "seed": args.seed,
                }

                best_path = output_dir / "checkpoint_best.pt"
                torch.save(checkpoint, best_path)
                print(f"  *** New best validation loss: {val_loss:.6f} ***")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= args.patience:
                print(f"\nEarly stopping after {patience_counter} epochs without improvement")
                break

        total_time = time.time() - train_start

        # Final evaluation
        print("\n" + "=" * 80)
        print("Final Evaluation")
        print("=" * 80)

        test_loss = validate_epoch(model, test_loader, device)

        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Test loss:            {test_loss:.6f}")
        print(f"Total training time:  {total_time:.1f}s ({total_time/60:.1f}m)")

        # Save results
        results = {
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            "total_time": total_time,
            "epochs_trained": epoch + 1,
            "config": {
                "model": model_config.model_dump(),
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "seed": args.seed,
            },
        }

        results_path = output_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {results_path}")
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
