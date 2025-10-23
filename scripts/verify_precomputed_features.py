#!/usr/bin/env python3
"""Verify pre-computed features are correct and complete.

Performs sanity checks on pre-computed features:
1. File integrity (all files exist, correct shapes)
2. Feature value ranges (within expected bounds)
3. Mask consistency (valid mask makes sense)
4. Split integrity (no overlaps, correct time ordering)
5. Reproducibility check (sample windows match on-the-fly computation)

Usage:
    python3 scripts/verify_precomputed_features.py \
        --feature-dir data/processed/nq_features \
        --data data/archive/nq_ohlcv_1min_2020-09_2025-10_continuous.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from moola.features.relativity import build_relativity_features, RelativityConfig


def verify_file_integrity(feature_dir: str) -> bool:
    """Verify all required files exist and have correct format.

    Args:
        feature_dir: Directory containing pre-computed features

    Returns:
        True if all checks pass
    """
    print("=" * 80)
    print("1. File Integrity Check")
    print("=" * 80)

    feature_path = Path(feature_dir)
    required_files = [
        "features_10d.npy",
        "valid_mask.npy",
        "metadata.json",
        "splits.json"
    ]

    all_exist = True
    for filename in required_files:
        filepath = feature_path / filename
        exists = filepath.exists()
        size_mb = filepath.stat().st_size / 1e6 if exists else 0
        status = "✓" if exists else "✗"
        print(f"  {status} {filename:20s} ({size_mb:8.1f} MB)")
        all_exist = all_exist and exists

    if not all_exist:
        print("\n  FAILED: Missing required files")
        return False

    # Load and check shapes
    features = np.load(feature_path / "features_10d.npy")
    valid_mask = np.load(feature_path / "valid_mask.npy")

    print(f"\n  Features shape: {features.shape}")
    print(f"  Valid mask shape: {valid_mask.shape}")

    if len(features.shape) != 3:
        print(f"\n  FAILED: Features should be 3D [N, K, D], got {features.shape}")
        return False

    if features.shape[:2] != valid_mask.shape:
        print(f"\n  FAILED: Feature and mask shapes mismatch")
        return False

    if features.shape[2] != 10:
        print(f"\n  FAILED: Expected 10 features, got {features.shape[2]}")
        return False

    print("\n  PASSED: All files exist with correct shapes")
    return True


def verify_feature_ranges(feature_dir: str) -> bool:
    """Verify feature values are within expected ranges.

    Args:
        feature_dir: Directory containing pre-computed features

    Returns:
        True if all checks pass
    """
    print("\n" + "=" * 80)
    print("2. Feature Range Check")
    print("=" * 80)

    feature_path = Path(feature_dir)

    # Load features and metadata
    features = np.load(feature_path / "features_10d.npy")
    with open(feature_path / "metadata.json", 'r') as f:
        metadata = json.load(f)

    feature_names = metadata['feature_names']
    feature_ranges = metadata['feature_ranges']

    # Check each feature
    all_valid = True
    for i, name in enumerate(feature_names):
        values = features[:, :, i]
        min_val = values.min()
        max_val = values.max()
        mean_val = values.mean()
        std_val = values.std()

        expected_range = feature_ranges[name]

        # Parse expected range
        if expected_range == '[0, 1]':
            expected_min, expected_max = 0.0, 1.0
        elif expected_range == '[-1, 1]':
            expected_min, expected_max = -1.0, 1.0
        elif expected_range == '[0, 3]':
            expected_min, expected_max = 0.0, 3.0
        elif expected_range == '[-3, 3]':
            expected_min, expected_max = -3.0, 3.0
        else:
            expected_min, expected_max = -np.inf, np.inf

        # Allow small violations due to clipping
        tolerance = 0.01
        is_valid = (min_val >= expected_min - tolerance and
                   max_val <= expected_max + tolerance)

        status = "✓" if is_valid else "✗"
        print(f"  {status} {name:20s} range=[{min_val:6.3f}, {max_val:6.3f}] "
              f"mean={mean_val:6.3f} std={std_val:6.3f} (expected {expected_range})")

        all_valid = all_valid and is_valid

    if all_valid:
        print("\n  PASSED: All features within expected ranges")
    else:
        print("\n  FAILED: Some features outside expected ranges")

    return all_valid


def verify_mask_consistency(feature_dir: str) -> bool:
    """Verify valid mask is consistent.

    Args:
        feature_dir: Directory containing pre-computed features

    Returns:
        True if all checks pass
    """
    print("\n" + "=" * 80)
    print("3. Mask Consistency Check")
    print("=" * 80)

    feature_path = Path(feature_dir)

    # Load mask and metadata
    valid_mask = np.load(feature_path / "valid_mask.npy")
    with open(feature_path / "metadata.json", 'r') as f:
        metadata = json.load(f)

    # Check valid ratio
    valid_ratio = valid_mask.mean()
    expected_ratio = metadata['valid_ratio']

    print(f"  Valid ratio: {valid_ratio:.4f} (metadata: {expected_ratio:.4f})")

    if abs(valid_ratio - expected_ratio) > 0.001:
        print(f"\n  FAILED: Valid ratio mismatch")
        return False

    # Check that first few timesteps are typically masked (warmup period)
    warmup_bars = metadata['config'].get('warmup_bars', 20)
    first_timesteps_valid = valid_mask[:, :warmup_bars].mean()

    print(f"  First {warmup_bars} timesteps valid ratio: {first_timesteps_valid:.4f}")
    print(f"  (Should be low due to warmup period)")

    # Check that later timesteps are mostly valid
    later_timesteps_valid = valid_mask[:, warmup_bars:].mean()
    print(f"  Later timesteps valid ratio: {later_timesteps_valid:.4f}")
    print(f"  (Should be high, ~1.0)")

    if later_timesteps_valid < 0.95:
        print(f"\n  WARNING: Later timesteps have low valid ratio")

    print("\n  PASSED: Mask consistency checks")
    return True


def verify_split_integrity(feature_dir: str) -> bool:
    """Verify train/val/test splits are non-overlapping and time-ordered.

    Args:
        feature_dir: Directory containing pre-computed features

    Returns:
        True if all checks pass
    """
    print("\n" + "=" * 80)
    print("4. Split Integrity Check")
    print("=" * 80)

    feature_path = Path(feature_dir)

    # Load splits
    with open(feature_path / "splits.json", 'r') as f:
        splits = json.load(f)

    train_start, train_end = splits['train_indices']
    val_start, val_end = splits['val_indices']
    test_start, test_end = splits['test_indices']

    print(f"  Train: [{train_start:6d}, {train_end:6d}) = {train_end - train_start:,} windows")
    print(f"  Val:   [{val_start:6d}, {val_end:6d}) = {val_end - val_start:,} windows")
    print(f"  Test:  [{test_start:6d}, {test_end:6d}) = {test_end - test_start:,} windows")

    # Check non-overlapping
    all_valid = True

    if train_end > val_start:
        print(f"\n  FAILED: Train overlaps with val")
        all_valid = False

    if val_end > test_start:
        print(f"\n  FAILED: Val overlaps with test")
        all_valid = False

    # Check time ordering
    if not (train_start < train_end <= val_start < val_end <= test_start < test_end):
        print(f"\n  FAILED: Splits not in time order")
        all_valid = False

    if all_valid:
        print("\n  PASSED: Splits are non-overlapping and time-ordered")

    return all_valid


def verify_reproducibility(feature_dir: str, data_path: str, n_samples: int = 5) -> bool:
    """Verify features match on-the-fly computation (spot check).

    Args:
        feature_dir: Directory containing pre-computed features
        data_path: Path to original OHLC data
        n_samples: Number of random windows to check

    Returns:
        True if all checks pass
    """
    print("\n" + "=" * 80)
    print("5. Reproducibility Check (Spot Checking)")
    print("=" * 80)

    feature_path = Path(feature_dir)

    # Load pre-computed features
    features_precomputed = np.load(feature_path / "features_10d.npy")

    # Load metadata to get config
    with open(feature_path / "metadata.json", 'r') as f:
        metadata = json.load(f)

    # Load original data
    print(f"  Loading original data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Reconstruct config
    relativity_config = metadata['relativity_config']

    # Build features on-the-fly for comparison
    print(f"  Rebuilding features on-the-fly for comparison...")
    features_rebuilt, _, _ = build_relativity_features(
        df[['open', 'high', 'low', 'close']],
        relativity_config
    )

    # Check shapes match
    if features_precomputed.shape != features_rebuilt.shape:
        print(f"\n  FAILED: Shape mismatch")
        print(f"    Pre-computed: {features_precomputed.shape}")
        print(f"    Rebuilt: {features_rebuilt.shape}")
        return False

    # Spot check random windows
    print(f"\n  Checking {n_samples} random windows...")
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(features_precomputed), size=n_samples, replace=False)

    max_diff = 0.0
    all_match = True

    for i, idx in enumerate(sample_indices):
        precomp = features_precomputed[idx]
        rebuilt = features_rebuilt[idx]

        diff = np.abs(precomp - rebuilt).max()
        max_diff = max(max_diff, diff)

        match = diff < 1e-5  # Allow tiny numerical differences
        status = "✓" if match else "✗"

        print(f"    {status} Window {idx:6d}: max_diff = {diff:.2e}")

        all_match = all_match and match

    print(f"\n  Overall max difference: {max_diff:.2e}")

    if all_match:
        print("  PASSED: Pre-computed features match on-the-fly computation")
    else:
        print("  FAILED: Features don't match")

    return all_match


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify pre-computed features",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--feature-dir", required=True, help="Pre-computed feature directory")
    parser.add_argument("--data", help="Original OHLC parquet file (for reproducibility check)")
    parser.add_argument("--skip-reproducibility", action="store_true",
                       help="Skip reproducibility check (faster)")

    args = parser.parse_args()

    try:
        all_passed = True

        # Run checks
        all_passed &= verify_file_integrity(args.feature_dir)
        all_passed &= verify_feature_ranges(args.feature_dir)
        all_passed &= verify_mask_consistency(args.feature_dir)
        all_passed &= verify_split_integrity(args.feature_dir)

        if args.data and not args.skip_reproducibility:
            all_passed &= verify_reproducibility(args.feature_dir, args.data)

        # Print summary
        print("\n" + "=" * 80)
        if all_passed:
            print("ALL CHECKS PASSED ✓")
        else:
            print("SOME CHECKS FAILED ✗")
        print("=" * 80)

        return 0 if all_passed else 1

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
