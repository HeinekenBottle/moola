#!/usr/bin/env python3
"""Parallel Precompute Relativity Features (32-Core Optimized).

Shards 5-year NQ data by month, processes in parallel across 32 cores.
Reduces compute time from 1-3 hours → 5-10 minutes on high-compute pods.

Architecture:
    - Monthly sharding (60 chunks for 5 years)
    - 30-bar overlap for zigzag warmup
    - joblib Parallel with 32 workers
    - Memory-efficient chunked processing

Usage:
    python3 scripts/precompute_nq_features_parallel.py \
        --data data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet \
        --output data/processed/nq_features \
        --n-jobs 32

Hardware:
    - 32 vCPUs, 64GB RAM (EPYC 9754 or similar)
    - Ubuntu 22.04 LTS
    - NVMe SSD for fast I/O
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from pydantic import BaseModel
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from moola.features.relativity import build_relativity_features, RelativityConfig


class ParallelPrecomputeConfig(BaseModel):
    """Configuration for parallel precomputation."""
    window_length: int = 105
    warmup_bars: int = 20
    overlap_bars: int = 30  # Overlap between chunks for zigzag warmup
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


def shard_by_month(df: pd.DataFrame, overlap_bars: int = 30) -> List[Tuple[str, pd.DataFrame]]:
    """Shard DataFrame into monthly chunks with overlap.

    Args:
        df: Input DataFrame with DatetimeIndex
        overlap_bars: Number of bars to overlap between chunks (for warmup)

    Returns:
        List of (month_label, chunk_df) tuples
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    # Group by year-month
    grouped = df.groupby(pd.Grouper(freq='MS'))  # MS = month start

    shards = []
    prev_end_idx = None

    for month_start, group in grouped:
        if len(group) == 0:
            continue

        # Add overlap from previous month
        start_idx = group.index[0]
        if prev_end_idx is not None:
            # Find overlap_bars before start_idx
            overlap_start = max(0, df.index.get_loc(start_idx) - overlap_bars)
            start_idx = df.index[overlap_start]

        end_idx = group.index[-1]
        chunk = df.loc[start_idx:end_idx]

        month_label = month_start.strftime('%Y-%m')
        shards.append((month_label, chunk))

        prev_end_idx = end_idx

    return shards


def process_shard(shard_data: Tuple[str, pd.DataFrame],
                  cfg: RelativityConfig,
                  overlap_bars: int) -> Tuple[str, np.ndarray, np.ndarray, Dict]:
    """Process a single shard (month) of data.

    Args:
        shard_data: (month_label, chunk_df)
        cfg: Relativity configuration
        overlap_bars: Number of overlap bars to strip

    Returns:
        (month_label, features, mask, metadata)
    """
    month_label, chunk = shard_data

    # Build features for this chunk
    X, mask, meta = build_relativity_features(
        chunk[['open', 'high', 'low', 'close']],
        cfg.dict()
    )

    # Strip overlap windows (keep only non-overlap windows)
    # The first overlap_bars won't have complete windows, so strip them
    if overlap_bars > 0 and len(X) > overlap_bars:
        X = X[overlap_bars:]
        mask = mask[overlap_bars:]

    return month_label, X, mask, meta


def parallel_precompute(data_path: str,
                       output_dir: str,
                       config: ParallelPrecomputeConfig,
                       n_jobs: int = 32) -> Dict:
    """Precompute features using parallel processing.

    Args:
        data_path: Path to NQ parquet file
        output_dir: Directory to save features
        config: Precomputation configuration
        n_jobs: Number of parallel workers

    Returns:
        Metadata dictionary
    """
    print("=" * 80)
    print("NQ Parallel Feature Precomputation (32-Core Optimized)")
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

    print(f"  ✅ Loaded {len(df):,} bars in {load_time:.1f}s")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Memory usage: {df.memory_usage().sum() / 1e6:.1f} MB")

    # Shard by month
    print(f"\nSharding data by month (overlap={config.overlap_bars} bars)...")
    shard_start = time.time()
    shards = shard_by_month(df, overlap_bars=config.overlap_bars)
    shard_time = time.time() - shard_start

    print(f"  ✅ Created {len(shards)} monthly shards in {shard_time:.1f}s")
    print(f"  Avg bars per shard: {len(df) / len(shards):.0f}")

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

    # Process shards in parallel
    print(f"\nProcessing shards in parallel (n_jobs={n_jobs})...")
    print("  Features (11 dims):")
    print("    Candle (6): open_norm, close_norm, body_pct, upper_wick, lower_wick, range_z")
    print("    Swing (4):  dist_to_SH, dist_to_SL, bars_since_SH, bars_since_SL")
    print("    Expansion (1): expansion_proxy = range_z × leg_dir × body_pct")
    print(f"\n  🚀 Using {n_jobs} workers...")

    build_start = time.time()

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_shard)(shard, relativity_cfg, config.overlap_bars)
        for shard in shards
    )

    build_time = time.time() - build_start
    bars_per_second = len(df) / build_time

    print(f"\n  ✅ Parallel processing completed in {build_time:.1f}s ({bars_per_second:.0f} bars/s)")
    print(f"  Speedup: ~{(1800/build_time):.1f}x vs sequential (est.)")

    # Merge results
    print("\nMerging shard results...")
    merge_start = time.time()

    all_X = []
    all_masks = []

    for month_label, X, mask, meta in results:
        all_X.append(X)
        all_masks.append(mask)

    X_full = np.concatenate(all_X, axis=0)
    valid_mask_full = np.concatenate(all_masks, axis=0)

    # Get metadata from first shard (all have same structure)
    _, _, _, feature_meta = results[0]

    merge_time = time.time() - merge_start

    print(f"  ✅ Merged {len(results)} shards in {merge_time:.1f}s")
    print(f"  Final shape: {X_full.shape}")
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

    print(f"  ✅ Saved features to {features_path} ({X_full.nbytes / 1e6:.1f} MB)")
    print(f"  ✅ Saved valid mask to {mask_path} ({valid_mask_full.nbytes / 1e6:.1f} MB)")
    print(f"  Save time: {save_time:.1f}s")

    # Create metadata
    metadata = {
        "created_at": pd.Timestamp.now().isoformat(),
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "parallel_config": {
            "n_jobs": n_jobs,
            "n_shards": len(shards),
            "overlap_bars": config.overlap_bars
        },

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
            "shard_time": shard_time,
            "build_time": build_time,
            "merge_time": merge_time,
            "save_time": save_time,
            "total_time": load_time + shard_time + build_time + merge_time + save_time,
            "bars_per_second": bars_per_second
        }
    }

    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  ✅ Saved metadata to {metadata_path}")

    # Save split indices separately
    splits_path = output_path / "splits.json"
    splits_data = {
        "train_indices": [0, train_windows],
        "val_indices": [val_start, val_windows],
        "test_indices": [test_start, test_windows]
    }
    with open(splits_path, 'w') as f:
        json.dump(splits_data, f, indent=2)

    print(f"  ✅ Saved split indices to {splits_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("PARALLEL PRECOMPUTATION COMPLETE")
    print("=" * 80)
    total_time = metadata['timing']['total_time']
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"Processing speed: {bars_per_second:.0f} bars/s")
    print(f"Speedup vs sequential: ~{(1800/build_time):.1f}x")
    print(f"\nOutput files:")
    print(f"  {features_path}")
    print(f"  {mask_path}")
    print(f"  {metadata_path}")
    print(f"  {splits_path}")

    print(f"\nNext steps:")
    print(f"  1. Verify features: python3 scripts/verify_precomputed_features.py")
    print(f"  2. Check non-zero density (target >50%)")
    print(f"  3. Train Jade model with uncertainty weighting")

    return metadata


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parallel precompute relativity features (32-core optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 32-core high-compute pod
  python3 scripts/precompute_nq_features_parallel.py \\
      --data data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet \\
      --output data/processed/nq_features \\
      --n-jobs 32

  # 16-core pod
  python3 scripts/precompute_nq_features_parallel.py \\
      --data data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet \\
      --output data/processed/nq_features \\
      --n-jobs 16

Hardware:
  - Recommended: 32 vCPUs, 64GB RAM (EPYC 9754)
  - Ubuntu 22.04 LTS
  - NVMe SSD
  - Est. time: 5-10 minutes
        """
    )
    parser.add_argument("--data", required=True, help="Path to NQ parquet file")
    parser.add_argument("--output", required=True, help="Output directory for features")
    parser.add_argument("--config", help="Path to windowed config (optional)")
    parser.add_argument("--n-jobs", type=int, default=32, help="Number of parallel workers")
    parser.add_argument("--window-length", type=int, default=105, help="Window length")
    parser.add_argument("--overlap-bars", type=int, default=30, help="Overlap between chunks")

    args = parser.parse_args()

    try:
        # Load config if provided
        if args.config:
            with open(args.config, 'r') as f:
                config_dict = yaml.safe_load(f)
            config = ParallelPrecomputeConfig(**config_dict)
        else:
            config = ParallelPrecomputeConfig(
                window_length=args.window_length,
                overlap_bars=args.overlap_bars
            )

        # Run parallel precomputation
        metadata = parallel_precompute(args.data, args.output, config, n_jobs=args.n_jobs)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
