#!/usr/bin/env python3
"""Precompute relativity features for 5-year NQ data.

This script builds features once and saves them to numpy arrays for fast loading.
Eliminates 1-3 hour feature computation bottleneck before each training run.

Usage:
    python3 scripts/precompute_nq_features.py \
        --data data/raw/nq_5year.parquet \
        --output data/processed/nq_features \
        --config configs/windowed.yaml

Output:
    data/processed/nq_features/
        features_12d.npy      # [N_windows, K, D] float32 features (D=12 with expansion_proxy + consol_proxy)
        valid_mask.npy        # [N_windows, K] bool mask
        metadata.json         # Feature info, config, timestamps
        splits.json           # Time split indices for train/val/test

Performance:
    - CPU-bound: ~100-500 bars/second depending on CPU clock speed
    - 1.8M bars = ~1-3 hours on typical CPU
    - Zigzag state is sequential, so multi-core parallelization is limited
    - Best hardware: High clock speed CPU (4-5 GHz) > Many cores
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from moola.features.relativity import build_relativity_features, RelativityConfig


class PrecomputeConfig(BaseModel):
    """Configuration for feature precomputation."""
    window_length: int = 105
    warmup_bars: int = 20
    train_end: str = "2024-12-31"
    val_end: str = "2025-03-31"
    test_end: str = "2025-06-30"

    # Feature config
    ohlc_eps: float = 1.0e-6
    ohlc_ema_range_period: int = 20
    atr_period: int = 10
    zigzag_k: float = 1.2
    zigzag_hybrid_confirm_lookback: int = 5
    zigzag_hybrid_min_retrace_atr: float = 0.5


def estimate_completion_time(n_bars: int, bars_per_second: float = 100.0) -> str:
    """Estimate completion time based on data size.

    Args:
        n_bars: Total number of bars to process
        bars_per_second: Estimated processing speed

    Returns:
        Human-readable time estimate
    """
    total_seconds = n_bars / bars_per_second

    if total_seconds < 60:
        return f"{total_seconds:.0f} seconds"
    elif total_seconds < 3600:
        return f"{total_seconds/60:.1f} minutes"
    else:
        hours = int(total_seconds / 3600)
        minutes = int((total_seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def precompute_features(data_path: str, output_dir: str, config: PrecomputeConfig) -> Dict:
    """Precompute features for full dataset.

    Args:
        data_path: Path to NQ parquet file
        output_dir: Directory to save features
        config: Precomputation configuration

    Returns:
        Metadata dictionary with processing stats
    """
    print("=" * 80)
    print("NQ Feature Precomputation Pipeline")
    print("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    # Load data
    print(f"\nLoading data from {data_path}...")
    load_start = time.time()
    df = pd.read_parquet(data_path)
    load_time = time.time() - load_start

    print(f"  Loaded {len(df):,} bars in {load_time:.1f}s")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Memory usage: {df.memory_usage().sum() / 1e6:.1f} MB")

    # Estimate completion time
    estimated_time = estimate_completion_time(len(df), bars_per_second=100)
    print(f"  Estimated processing time: {estimated_time}")

    # Create relativity config
    relativity_cfg = RelativityConfig(
        window_length=config.window_length,
        ohlc_eps=config.ohlc_eps,
        ohlc_ema_range_period=config.ohlc_ema_range_period,
        atr_period=config.atr_period,
        zigzag_k=config.zigzag_k,
        zigzag_hybrid_confirm_lookback=config.zigzag_hybrid_confirm_lookback,
        zigzag_hybrid_min_retrace_atr=config.zigzag_hybrid_min_retrace_atr
    )

    # Build features with progress tracking
    print(f"\nBuilding relativity features (window_length={config.window_length})...")
    print("  Features (11 dims):")
    print("    Candle (6): open_norm, close_norm, body_pct, upper_wick, lower_wick, range_z")
    print("    Swing (4):  dist_to_SH, dist_to_SL, bars_since_SH, bars_since_SL")
    print("    Expansion (1): expansion_proxy = range_z × leg_dir × body_pct")
    print("  Processing is sequential (zigzag state dependencies)...")

    build_start = time.time()

    # Wrap build_relativity_features to add progress tracking
    # For now, just call it directly (progress is internal to function)
    X_full, valid_mask_full, feature_meta = build_relativity_features(
        df[['open', 'high', 'low', 'close']],
        relativity_cfg.dict()
    )

    build_time = time.time() - build_start
    bars_per_second = len(df) / build_time

    print(f"\n  Feature building completed in {build_time:.1f}s ({bars_per_second:.0f} bars/s)")
    print(f"  Feature shape: {X_full.shape}")
    print(f"  Memory usage: {X_full.nbytes / 1e6:.1f} MB")
    print(f"  Valid ratio: {valid_mask_full.mean():.3f}")

    # Create time-based split indices
    print("\nCreating time-based split indices...")

    # Find split boundaries
    train_end_idx = df.index.get_indexer([config.train_end], method='ffill')[0]
    val_end_idx = df.index.get_indexer([config.val_end], method='ffill')[0]
    test_end_idx = df.index.get_indexer([config.test_end], method='ffill')[0]

    # Adjust for window length
    n_windows = len(X_full)
    train_windows = train_end_idx - config.window_length + 1
    val_start = train_windows
    val_windows = val_end_idx - config.window_length + 1
    test_start = val_windows
    test_windows = min(test_end_idx - config.window_length + 1, n_windows)

    print(f"  Train: windows [0, {train_windows}) = {train_windows:,} windows")
    print(f"  Val:   windows [{val_start}, {val_windows}) = {val_windows - val_start:,} windows")
    print(f"  Test:  windows [{test_start}, {test_windows}) = {test_windows - test_start:,} windows")

    # Save features
    print("\nSaving feature arrays...")
    save_start = time.time()

    features_path = output_path / "features_11d.npy"
    mask_path = output_path / "valid_mask.npy"

    np.save(features_path, X_full)
    np.save(mask_path, valid_mask_full)

    save_time = time.time() - save_start

    print(f"  Saved features to {features_path} ({X_full.nbytes / 1e6:.1f} MB)")
    print(f"  Saved valid mask to {mask_path} ({valid_mask_full.nbytes / 1e6:.1f} MB)")
    print(f"  Save time: {save_time:.1f}s")

    # Create metadata
    metadata = {
        "created_at": pd.Timestamp.now().isoformat(),
        "data_path": str(data_path),
        "output_dir": str(output_dir),

        # Data info
        "n_bars": len(df),
        "n_windows": n_windows,
        "date_range": {
            "start": str(df.index.min()),
            "end": str(df.index.max())
        },

        # Feature info
        "feature_shape": list(X_full.shape),
        "feature_dtype": str(X_full.dtype),
        "feature_names": feature_meta['feature_names'],
        "feature_ranges": feature_meta['feature_ranges'],

        # Mask info
        "valid_mask_shape": list(valid_mask_full.shape),
        "valid_ratio": float(valid_mask_full.mean()),

        # Split info
        "splits": {
            "train": {
                "window_range": [0, train_windows],
                "n_windows": train_windows,
                "date_range": [str(df.index[0]), config.train_end]
            },
            "val": {
                "window_range": [val_start, val_windows],
                "n_windows": val_windows - val_start,
                "date_range": [config.train_end, config.val_end]
            },
            "test": {
                "window_range": [test_start, test_windows],
                "n_windows": test_windows - test_start,
                "date_range": [config.val_end, config.test_end]
            }
        },

        # Config
        "config": config.dict(),
        "relativity_config": relativity_cfg.dict(),

        # Performance
        "timing": {
            "load_time": load_time,
            "build_time": build_time,
            "save_time": save_time,
            "total_time": load_time + build_time + save_time,
            "bars_per_second": bars_per_second
        }
    }

    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved metadata to {metadata_path}")

    # Save split indices separately for fast loading
    splits_path = output_path / "splits.json"
    splits_data = {
        "train_indices": [0, train_windows],
        "val_indices": [val_start, val_windows],
        "test_indices": [test_start, test_windows]
    }
    with open(splits_path, 'w') as f:
        json.dump(splits_data, f, indent=2)

    print(f"  Saved split indices to {splits_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("PRECOMPUTATION COMPLETE")
    print("=" * 80)
    total_time = metadata['timing']['total_time']
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"Processing speed: {bars_per_second:.0f} bars/s")
    print(f"\nOutput files:")
    print(f"  {features_path}")
    print(f"  {mask_path}")
    print(f"  {metadata_path}")
    print(f"  {splits_path}")

    print(f"\nNext steps:")
    print(f"  1. Verify features: python3 scripts/verify_precomputed_features.py")
    print(f"  2. Update training script to use precomputed features")
    print(f"  3. Run training with fast loader (5s load vs 1-3h compute)")

    return metadata


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Precompute relativity features for NQ data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Precompute with default config
  python3 scripts/precompute_nq_features.py \\
      --data data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet \\
      --output data/processed/nq_features

  # Precompute with custom config
  python3 scripts/precompute_nq_features.py \\
      --data data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet \\
      --output data/processed/nq_features \\
      --config configs/windowed.yaml
        """
    )
    parser.add_argument("--data", required=True, help="Path to NQ parquet file")
    parser.add_argument("--output", required=True, help="Output directory for features")
    parser.add_argument("--config", help="Path to windowed config (optional)")
    parser.add_argument("--window-length", type=int, default=105, help="Window length")
    parser.add_argument("--warmup-bars", type=int, default=20, help="Warmup bars to mask")

    args = parser.parse_args()

    try:
        # Load config if provided
        if args.config:
            with open(args.config, 'r') as f:
                config_dict = yaml.safe_load(f)
            config = PrecomputeConfig(**config_dict)
        else:
            config = PrecomputeConfig(
                window_length=args.window_length,
                warmup_bars=args.warmup_bars
            )

        # Run precomputation
        metadata = precompute_features(args.data, args.output, config)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
