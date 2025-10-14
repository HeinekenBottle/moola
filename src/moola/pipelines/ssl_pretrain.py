#!/usr/bin/env python3
"""Phase 1: SSL Pre-training with TS-TCC contrastive learning.

Pre-trains encoder on unlabeled data using InfoNCE contrastive loss.

Usage:
    python -m moola.pipelines.ssl_pretrain \
        --unlabeled data/raw/unlabeled_windows.parquet \
        --output data/artifacts/pretrained/encoder_weights.pt \
        --epochs 100 \
        --device cuda
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ..models.ts_tcc import TSTCCPretrainer


def load_unlabeled_data(path: Path) -> np.ndarray:
    """Load unlabeled windows from parquet.

    Args:
        path: Path to unlabeled windows parquet

    Returns:
        Unlabeled data array [N, 105, 4]
    """
    logger.info(f"Loading unlabeled data from {path}")
    df = pd.read_parquet(path)

    # Convert features to proper shape
    X = []
    for features in df['features']:
        # Convert array of arrays to 2D array
        window = np.array([np.array(bar) for bar in features])
        X.append(window)

    X = np.array(X)
    logger.info(f"Loaded {len(X):,} unlabeled samples with shape {X.shape}")

    return X


def main():
    parser = argparse.ArgumentParser(description="SSL Phase 1: Contrastive Pre-training")
    parser.add_argument('--unlabeled', type=Path, required=True, help='Unlabeled data parquet')
    parser.add_argument('--output', type=Path, required=True, help='Output encoder weights path')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--no-amp', action='store_true', help='Disable automatic mixed precision (FP16)')

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("SSL PHASE 1: CONTRASTIVE PRE-TRAINING")
    logger.info("="*70)
    logger.info(f"Unlabeled data: {args.unlabeled}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Mixed Precision: {not args.no_amp}")
    logger.info("")

    # Load data
    X_unlabeled = load_unlabeled_data(args.unlabeled)

    # Initialize pretrainer
    pretrainer = TSTCCPretrainer(
        input_dim=4,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        use_amp=not args.no_amp,
    )

    # Pre-train
    logger.info("Starting contrastive pre-training...")
    history = pretrainer.pretrain(X_unlabeled)

    # Save encoder
    pretrainer.save_encoder(args.output)

    logger.info("\n" + "="*70)
    logger.info("PRE-TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Best epoch: {history['best_epoch'] + 1}")
    logger.info(f"Encoder saved to: {args.output}")
    logger.info(f"\nNext: Run Phase 2 fine-tuning with:")
    logger.info(f"  python -m moola.pipelines.ssl_finetune \\")
    logger.info(f"    --encoder {args.output} \\")
    logger.info(f"    --labeled data/processed/train.parquet \\")
    logger.info(f"    --device {args.device}")


if __name__ == '__main__':
    main()
