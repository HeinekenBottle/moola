#!/usr/bin/env python3
"""Generate unlabeled sequences for masked LSTM pre-training.

Extracts unlabeled OHLC sequences from existing data and applies temporal
augmentation to expand the dataset for robust pre-training.

Target: 1000-5000 unlabeled sequences for RTX 4090 (24GB VRAM)
Pre-training time: ~30-40 minutes

Usage:
    python scripts/generate_unlabeled_data.py --output data/processed/unlabeled_pretrain.parquet
    python scripts/generate_unlabeled_data.py --target-count 5000 --augment-factor 4
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from moola.config.training_config import (
    MASKED_LSTM_AUG_JITTER_PROB,
    MASKED_LSTM_AUG_NUM_VERSIONS,
    MASKED_LSTM_AUG_TIME_WARP_PROB,
    MASKED_LSTM_AUG_VOLATILITY_SCALE_PROB,
    OHLC_DIMS,
    WINDOW_SIZE,
)
from moola.utils.temporal_augmentation import TemporalAugmentation


def extract_unlabeled_sequences(
    df: pd.DataFrame,
    sequence_column: str = "ohlc_sequence",
    target_count: int = 1000,
) -> np.ndarray:
    """Extract raw OHLC sequences without labels.

    Args:
        df: DataFrame with OHLC sequences
        sequence_column: Column name containing sequences
        target_count: Target number of unlabeled sequences

    Returns:
        [N, seq_len, 4] array of OHLC sequences
    """
    print(f"[DATA EXTRACTION]")
    print(f"  Source: {len(df)} samples")
    print(f"  Target: {target_count} unlabeled sequences")

    # Extract sequences
    sequences = []
    for idx, row in df.iterrows():
        seq = row[sequence_column]
        if isinstance(seq, np.ndarray):
            sequences.append(seq)
        elif isinstance(seq, (list, tuple)):
            sequences.append(np.array(seq))

    sequences = np.array(sequences)

    # Validate shape
    expected_shape = (len(df), WINDOW_SIZE, OHLC_DIMS)
    if sequences.shape != expected_shape:
        raise ValueError(
            f"Sequence shape mismatch. Expected {expected_shape}, got {sequences.shape}"
        )

    print(f"  Extracted: {len(sequences)} sequences")
    print(f"  Shape: {sequences.shape}")

    # Sample if we have more than target
    if len(sequences) > target_count:
        indices = np.random.choice(len(sequences), size=target_count, replace=False)
        sequences = sequences[indices]
        print(f"  Sampled: {len(sequences)} sequences")

    return sequences


def augment_sequences(
    sequences: np.ndarray,
    augment_factor: int = MASKED_LSTM_AUG_NUM_VERSIONS,
    jitter_prob: float = MASKED_LSTM_AUG_JITTER_PROB,
    time_warp_prob: float = MASKED_LSTM_AUG_TIME_WARP_PROB,
    volatility_scale_prob: float = MASKED_LSTM_AUG_VOLATILITY_SCALE_PROB,
) -> np.ndarray:
    """Apply temporal augmentation to expand unlabeled dataset.

    Generates multiple augmented versions per sequence to increase
    diversity and robustness of pre-training.

    Args:
        sequences: [N, seq_len, features] original sequences
        augment_factor: Number of augmented versions per sequence
        jitter_prob: Probability of jitter augmentation
        time_warp_prob: Probability of time warping
        volatility_scale_prob: Probability of volatility scaling

    Returns:
        [N*(1+augment_factor), seq_len, features] augmented sequences
    """
    print(f"\n[AUGMENTATION]")
    print(f"  Original count: {len(sequences)}")
    print(f"  Augment factor: {augment_factor}x")
    print(f"  Jitter prob: {jitter_prob}")
    print(f"  Time warp prob: {time_warp_prob}")
    print(f"  Volatility scale prob: {volatility_scale_prob}")

    # Initialize augmentation pipeline
    aug = TemporalAugmentation(
        jitter_prob=jitter_prob,
        jitter_sigma=0.05,  # 5% noise
        scaling_prob=volatility_scale_prob,
        scaling_sigma=0.1,  # 10% volatility variation
        time_warp_prob=time_warp_prob,
        time_warp_sigma=0.2,  # 20% temporal distortion
        permutation_prob=0.0,  # Disabled (breaks temporal order)
        rotation_prob=0.0,  # Disabled (OHLC order matters)
    )

    # Start with original sequences
    augmented = [sequences.copy()]

    # Generate augmented versions
    for version in range(augment_factor):
        print(f"  Generating augmentation version {version + 1}/{augment_factor}...")

        # Convert to torch for augmentation
        import torch
        sequences_torch = torch.FloatTensor(sequences)

        # Apply augmentation
        aug_sequences = aug.apply_augmentation(sequences_torch)

        # Convert back to numpy
        augmented.append(aug_sequences.cpu().numpy())

    # Concatenate all versions
    result = np.concatenate(augmented, axis=0)

    print(f"  Final count: {len(result)} sequences ({len(result) / len(sequences):.1f}x expansion)")

    return result


def save_unlabeled_data(
    sequences: np.ndarray,
    output_path: Path,
    metadata: dict = None,
) -> None:
    """Save unlabeled sequences to parquet file.

    Args:
        sequences: [N, seq_len, features] unlabeled sequences
        output_path: Path to save parquet file
        metadata: Optional metadata dictionary
    """
    print(f"\n[SAVING]")
    print(f"  Output: {output_path}")
    print(f"  Sequences: {len(sequences)}")
    print(f"  Shape: {sequences.shape}")

    # Create DataFrame
    # Store sequences as list of arrays for parquet compatibility
    df = pd.DataFrame({
        'ohlc_sequence': [seq for seq in sequences],
        'seq_index': np.arange(len(sequences)),
    })

    # Add metadata columns if provided
    if metadata:
        for key, value in metadata.items():
            df[f'meta_{key}'] = value

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"  ✓ Saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate unlabeled sequences for masked LSTM pre-training"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/train_pivot_134.parquet"),
        help="Input parquet file with labeled data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/unlabeled_pretrain.parquet"),
        help="Output parquet file for unlabeled sequences",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=1000,
        help="Target number of base unlabeled sequences (before augmentation)",
    )
    parser.add_argument(
        "--augment-factor",
        type=int,
        default=MASKED_LSTM_AUG_NUM_VERSIONS,
        help="Number of augmented versions per sequence",
    )
    parser.add_argument(
        "--jitter-prob",
        type=float,
        default=MASKED_LSTM_AUG_JITTER_PROB,
        help="Probability of jitter augmentation",
    )
    parser.add_argument(
        "--time-warp-prob",
        type=float,
        default=MASKED_LSTM_AUG_TIME_WARP_PROB,
        help="Probability of time warp augmentation",
    )
    parser.add_argument(
        "--volatility-scale-prob",
        type=float,
        default=MASKED_LSTM_AUG_VOLATILITY_SCALE_PROB,
        help="Probability of volatility scaling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Skip augmentation (only extract base sequences)",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    print("="*80)
    print("UNLABELED DATA GENERATION FOR MASKED LSTM PRE-TRAINING")
    print("="*80)

    # Load input data
    print(f"\n[LOADING]")
    print(f"  Input: {args.input}")

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_parquet(args.input)
    print(f"  Loaded: {len(df)} samples")

    # Extract unlabeled sequences
    sequences = extract_unlabeled_sequences(
        df,
        sequence_column="ohlc_sequence",
        target_count=args.target_count,
    )

    # Apply augmentation if enabled
    if not args.no_augment:
        sequences = augment_sequences(
            sequences,
            augment_factor=args.augment_factor,
            jitter_prob=args.jitter_prob,
            time_warp_prob=args.time_warp_prob,
            volatility_scale_prob=args.volatility_scale_prob,
        )
    else:
        print("\n[AUGMENTATION] Skipped (--no-augment)")

    # Save unlabeled data
    metadata = {
        'source': str(args.input),
        'target_count': args.target_count,
        'augment_factor': args.augment_factor if not args.no_augment else 0,
        'seed': args.seed,
    }

    save_unlabeled_data(sequences, args.output, metadata)

    print("\n" + "="*80)
    print("✅ UNLABELED DATA GENERATION COMPLETE")
    print("="*80)
    print(f"  Total sequences: {len(sequences)}")
    print(f"  Output: {args.output}")
    print(f"  Ready for masked LSTM pre-training!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
