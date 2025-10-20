#!/usr/bin/env python3
"""Standalone TS-TCC pre-training script using unlabeled data.

This script pre-trains the TS-TCC encoder on 11,873 unlabeled windows
from data/raw/unlabeled_windows.parquet using contrastive learning.

Usage:
    python scripts/pretrain_tcc_unlabeled.py \
        --device cuda \
        --epochs 100 \
        --patience 15 \
        --batch-size 512

Output:
    models/ts_tcc/pretrained_encoder.pt - Pre-trained encoder weights
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.moola.models.ts_tcc import TSTCCPretrainer
from src.moola.utils.seeds import print_gpu_info


def main():
    parser = argparse.ArgumentParser(description="Pre-train TS-TCC encoder on unlabeled data")
    parser.add_argument(
        "--unlabeled-path",
        type=str,
        default="data/raw/unlabeled_windows.parquet",
        help="Path to unlabeled windows parquet file"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="models/ts_tcc/pretrained_encoder.pt",
        help="Path to save pre-trained encoder"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of pre-training epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed"
    )

    args = parser.parse_args()

    # Convert to absolute paths
    project_root = Path(__file__).parent.parent
    unlabeled_path = (project_root / args.unlabeled_path).resolve()
    output_path = (project_root / args.output_path).resolve()

    print("=" * 80)
    print("TS-TCC Pre-training on Unlabeled Data")
    print("=" * 80)
    print(f"Unlabeled data: {unlabeled_path}")
    print(f"Output path: {output_path}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Patience: {args.patience}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    # GPU verification
    if args.device == "cuda":
        print("\n[GPU INFO]")
        print_gpu_info()
        print()

    # Load unlabeled data
    print(f"\n[LOADING DATA]")
    if not unlabeled_path.exists():
        raise FileNotFoundError(f"Unlabeled data not found: {unlabeled_path}")

    df = pd.read_parquet(unlabeled_path)
    print(f"Loaded {len(df)} unlabeled samples")
    print(f"Columns: {list(df.columns)}")

    # Extract features
    # Features are stored as list of [O, H, L, C] bars
    # Convert to numpy array: [N, 105, 4]
    X_unlabeled = np.stack([np.stack(f) for f in df["features"]])
    print(f"Feature shape: {X_unlabeled.shape}")
    print(f"Feature dtype: {X_unlabeled.dtype}")
    print(f"Feature range (before norm): [{X_unlabeled.min():.2f}, {X_unlabeled.max():.2f}]")

    # Normalize data: standardize each sample independently
    # This is critical for contrastive learning stability
    print(f"\n[NORMALIZING DATA]")
    X_mean = X_unlabeled.mean(axis=(1, 2), keepdims=True)
    X_std = X_unlabeled.std(axis=(1, 2), keepdims=True)
    X_unlabeled = (X_unlabeled - X_mean) / (X_std + 1e-8)
    print(f"Feature range (after norm): [{X_unlabeled.min():.4f}, {X_unlabeled.max():.4f}]")
    print(f"Feature mean: {X_unlabeled.mean():.4f}, std: {X_unlabeled.std():.4f}")

    # Initialize TS-TCC pre-trainer
    print(f"\n[INITIALIZING PRE-TRAINER]")

    # Import TemporalAugmentation to configure it
    from src.moola.utils.temporal_augmentation import TemporalAugmentation

    # Create augmentation with time_warp disabled (has numpy/torch compatibility issues)
    augmentation = TemporalAugmentation(
        jitter_prob=0.8,
        jitter_sigma=0.03,
        scaling_prob=0.5,
        scaling_sigma=0.1,
        time_warp_prob=0.0,  # DISABLED - numpy/torch compatibility issue
        time_warp_sigma=0.2,
    )

    pretrainer = TSTCCPretrainer(
        input_dim=4,  # OHLC
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
        device=args.device,
        seed=args.seed,
        use_amp=False,  # Disable mixed precision to avoid NaN issues
        num_workers=16 if args.device == "cuda" else 0,
        temperature=0.7,  # Higher temperature for numerical stability
        learning_rate=3e-4,  # Lower learning rate to prevent explosion
    )

    # Override augmentation with our configured version
    pretrainer.augmentation = augmentation
    print(f"Augmentation config: jitter={augmentation.jitter_prob}, scaling={augmentation.scaling_prob}, time_warp={augmentation.time_warp_prob}")

    # Pre-train encoder
    print(f"\n[PRE-TRAINING]")
    print(f"Starting contrastive pre-training on {len(X_unlabeled)} samples...")
    print()

    history = pretrainer.pretrain(X_unlabeled)

    # Save pre-trained encoder
    print(f"\n[SAVING ENCODER]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pretrainer.save_encoder(output_path)

    # Print summary
    print("\n" + "=" * 80)
    print("PRE-TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best epoch: {history['best_epoch'] + 1}")
    print(f"Train loss: {history['train_loss'][-1]:.4f}")
    print(f"Val loss: {history['val_loss'][-1]:.4f}")
    print(f"Encoder saved to: {output_path}")
    print(f"Encoder file size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 80)

    # Print loss trajectory
    print("\n[LOSS TRAJECTORY]")
    print("Epoch | Train Loss | Val Loss")
    print("-" * 40)
    for i in range(0, len(history['train_loss']), max(1, len(history['train_loss']) // 10)):
        train_loss = history['train_loss'][i]
        val_loss = history['val_loss'][i] if i < len(history['val_loss']) else 0.0
        print(f"{i+1:5d} | {train_loss:10.4f} | {val_loss:8.4f}")
    print("-" * 40)

    print("\n✅ Pre-training complete! Encoder ready for fine-tuning.")


if __name__ == "__main__":
    main()
