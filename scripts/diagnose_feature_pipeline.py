#!/usr/bin/env python3
"""Diagnostic script to validate feature engineering pipeline fix.

Checks:
1. Zero rates per feature (should be low, indicating real data)
2. Swing detection counts (verify swings are being detected)
3. Valid mask ratio (should be ~85% valid after 15-bar warmup)
4. NaN counts (should be zero)
5. Feature distributions (min, max, mean, std)

Usage:
    python3 scripts/diagnose_feature_pipeline.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moola.features.relativity import build_features, RelativityConfig


def calculate_zero_rate(arr: np.ndarray) -> float:
    """Calculate percentage of zero values."""
    return (arr == 0).sum() / arr.size * 100


def calculate_nan_rate(arr: np.ndarray) -> float:
    """Calculate percentage of NaN values."""
    return np.isnan(arr).sum() / arr.size * 100


def analyze_feature_distribution(X: np.ndarray, mask: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Analyze distribution statistics for each feature.

    Args:
        X: Feature tensor [N, K, D]
        mask: Boolean mask [N, K]
        feature_names: List of feature names

    Returns:
        DataFrame with statistics per feature
    """
    n_windows, seq_len, n_features = X.shape

    stats = []
    for i, name in enumerate(feature_names):
        # Extract feature across all windows and timesteps
        feature_data = X[:, :, i]

        # Apply mask to get valid data only
        valid_data = feature_data[mask]

        stats.append({
            'feature': name,
            'zero_rate_%': calculate_zero_rate(valid_data),
            'nan_rate_%': calculate_nan_rate(valid_data),
            'min': np.nanmin(valid_data) if valid_data.size > 0 else np.nan,
            'max': np.nanmax(valid_data) if valid_data.size > 0 else np.nan,
            'mean': np.nanmean(valid_data) if valid_data.size > 0 else np.nan,
            'std': np.nanstd(valid_data) if valid_data.size > 0 else np.nan,
            'n_samples': valid_data.size
        })

    return pd.DataFrame(stats)


def count_swing_detections(X: np.ndarray, mask: np.ndarray) -> dict:
    """Count how many swings are detected across all windows.

    Args:
        X: Feature tensor [N, K, D]
        mask: Boolean mask [N, K]

    Returns:
        Dictionary with swing detection statistics
    """
    # Feature indices for swing features
    dist_to_SH_idx = 6  # dist_to_prev_SH
    dist_to_SL_idx = 7  # dist_to_prev_SL

    # Extract swing features
    dist_to_SH = X[:, :, dist_to_SH_idx]
    dist_to_SL = X[:, :, dist_to_SL_idx]

    # Apply mask
    valid_SH = dist_to_SH[mask]
    valid_SL = dist_to_SL[mask]

    # Count non-zero values (indicating swing was detected)
    n_SH_detected = (valid_SH != 0).sum()
    n_SL_detected = (valid_SL != 0).sum()

    total_valid = mask.sum()

    return {
        'total_valid_timesteps': total_valid,
        'n_swing_highs_detected': n_SH_detected,
        'n_swing_lows_detected': n_SL_detected,
        'SH_detection_rate_%': n_SH_detected / total_valid * 100 if total_valid > 0 else 0,
        'SL_detection_rate_%': n_SL_detected / total_valid * 100 if total_valid > 0 else 0,
    }


def analyze_mask(mask: np.ndarray) -> dict:
    """Analyze mask statistics.

    Args:
        mask: Boolean mask [N, K]

    Returns:
        Dictionary with mask statistics
    """
    n_windows, seq_len = mask.shape
    total_timesteps = n_windows * seq_len
    valid_timesteps = mask.sum()

    # Check first 15 bars of each window (should be masked)
    warmup_bars = 15
    warmup_mask = mask[:, :warmup_bars]
    warmup_valid_rate = warmup_mask.sum() / warmup_mask.size * 100

    # Check remaining bars (should be valid)
    active_mask = mask[:, warmup_bars:]
    active_valid_rate = active_mask.sum() / active_mask.size * 100

    return {
        'total_timesteps': total_timesteps,
        'valid_timesteps': valid_timesteps,
        'valid_ratio_%': valid_timesteps / total_timesteps * 100,
        'warmup_valid_rate_%': warmup_valid_rate,
        'active_valid_rate_%': active_valid_rate,
        'expected_valid_ratio_%': (seq_len - warmup_bars) / seq_len * 100
    }


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def main():
    """Run diagnostic on feature pipeline."""

    print("\n" + "=" * 80)
    print("  FEATURE PIPELINE DIAGNOSTIC")
    print("=" * 80)

    # Load labeled data (smaller, easier to work with)
    data_path = Path(__file__).parent.parent / "data/processed/labeled/train_latest.parquet"

    print(f"\n📊 Loading data from: {data_path}")

    if not data_path.exists():
        print(f"❌ ERROR: Data file not found: {data_path}")
        print("\nTrying alternative: loading first 1000 bars from unlabeled data...")

        alt_path = Path(__file__).parent.parent / "data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet"
        if not alt_path.exists():
            print(f"❌ ERROR: Alternative data file not found: {alt_path}")
            return 1

        df_full = pd.read_parquet(alt_path)
        df = df_full.head(1000)
        print(f"✅ Loaded {len(df)} bars from unlabeled data")
    else:
        df = pd.read_parquet(data_path)
        print(f"✅ Loaded {len(df)} labeled windows")

        # Check if this is windowed data or raw OHLC
        if 'open' not in df.columns:
            print("⚠️  Labeled data appears to be pre-windowed. Reconstructing OHLC...")
            # For now, try to load from raw unlabeled data instead
            alt_path = Path(__file__).parent.parent / "data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet"
            if alt_path.exists():
                df_full = pd.read_parquet(alt_path)
                df = df_full.head(1000)
                print(f"✅ Loaded {len(df)} bars from unlabeled data instead")
            else:
                print("❌ ERROR: Need raw OHLC data to run diagnostic")
                return 1

    print(f"   Columns: {list(df.columns)}")
    print(f"   Shape: {df.shape}")

    # Verify OHLC columns
    required = ['open', 'high', 'low', 'close']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"❌ ERROR: Missing required columns: {missing}")
        return 1

    print("\n🔧 Building features with RelativityConfig...")
    cfg = RelativityConfig()
    print(f"   Window length: {cfg.window_length}")
    print(f"   ATR period: {cfg.atr_period}")
    print(f"   ZigZag k: {cfg.zigzag_k}")

    try:
        X, mask, meta = build_features(df, cfg)
        print(f"✅ Features built successfully!")
        print(f"   Shape: {X.shape}")
        print(f"   Dtype: {X.dtype}")
        print(f"   Features: {meta['n_features']}")
    except Exception as e:
        print(f"❌ ERROR building features: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ========================================================================
    # MASK ANALYSIS
    # ========================================================================
    print_section("1. MASK ANALYSIS")

    mask_stats = analyze_mask(mask)
    print(f"Total timesteps:        {mask_stats['total_timesteps']:,}")
    print(f"Valid timesteps:        {mask_stats['valid_timesteps']:,}")
    print(f"Valid ratio:            {mask_stats['valid_ratio_%']:.2f}% (expected: {mask_stats['expected_valid_ratio_%']:.2f}%)")
    print(f"Warmup valid rate:      {mask_stats['warmup_valid_rate_%']:.2f}% (expected: 0%)")
    print(f"Active valid rate:      {mask_stats['active_valid_rate_%']:.2f}% (expected: 100%)")

    if abs(mask_stats['valid_ratio_%'] - mask_stats['expected_valid_ratio_%']) < 1.0:
        print("\n✅ PASS: Mask ratio matches expected value")
    else:
        print(f"\n⚠️  WARNING: Mask ratio deviates from expected")

    # ========================================================================
    # SWING DETECTION ANALYSIS
    # ========================================================================
    print_section("2. SWING DETECTION ANALYSIS")

    swing_stats = count_swing_detections(X, mask)
    print(f"Total valid timesteps:  {swing_stats['total_valid_timesteps']:,}")
    print(f"Swing highs detected:   {swing_stats['n_swing_highs_detected']:,} ({swing_stats['SH_detection_rate_%']:.2f}%)")
    print(f"Swing lows detected:    {swing_stats['n_swing_lows_detected']:,} ({swing_stats['SL_detection_rate_%']:.2f}%)")

    expected_swing_rate = 20  # Expect at least 20% detection rate
    if swing_stats['SH_detection_rate_%'] >= expected_swing_rate and swing_stats['SL_detection_rate_%'] >= expected_swing_rate:
        print(f"\n✅ PASS: Swing detection rates are healthy (>{expected_swing_rate}%)")
    else:
        print(f"\n⚠️  WARNING: Low swing detection rates (<{expected_swing_rate}%)")

    # ========================================================================
    # FEATURE DISTRIBUTION ANALYSIS
    # ========================================================================
    print_section("3. FEATURE DISTRIBUTION ANALYSIS")

    feature_names = meta['feature_names']
    dist_df = analyze_feature_distribution(X, mask, feature_names)

    # Format for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)

    print(dist_df.to_string(index=False))

    # ========================================================================
    # QUALITY CHECKS
    # ========================================================================
    print_section("4. QUALITY CHECKS")

    issues = []

    # Check for NaNs
    nan_features = dist_df[dist_df['nan_rate_%'] > 0]
    if len(nan_features) > 0:
        issues.append(f"❌ FAIL: Found NaNs in {len(nan_features)} features")
        for _, row in nan_features.iterrows():
            issues.append(f"   - {row['feature']}: {row['nan_rate_%']:.2f}% NaNs")
    else:
        print("✅ PASS: No NaN values detected")

    # Check for excessive zeros (candle features should be mostly non-zero)
    candle_features = ['open_norm', 'close_norm', 'body_pct', 'upper_wick_pct', 'lower_wick_pct', 'range_z']
    high_zero_candles = dist_df[
        (dist_df['feature'].isin(candle_features)) &
        (dist_df['zero_rate_%'] > 40)  # More than 40% zeros is suspicious
    ]

    if len(high_zero_candles) > 0:
        issues.append(f"⚠️  WARNING: High zero rates in candle features:")
        for _, row in high_zero_candles.iterrows():
            issues.append(f"   - {row['feature']}: {row['zero_rate_%']:.2f}% zeros")
    else:
        print("✅ PASS: Candle features have healthy non-zero rates")

    # Check swing features (should have some zeros when no swing detected)
    # Distance features should have very LOW zero rates (swings detected most of the time)
    # Bars-since features can have higher zero rates (recent swings)
    dist_features = ['dist_to_prev_SH', 'dist_to_prev_SL']
    dist_df_subset = dist_df[dist_df['feature'].isin(dist_features)]

    dist_zero_ok = True
    for _, row in dist_df_subset.iterrows():
        # For distance features, expect 70-95% NON-ZERO (i.e., 5-30% zeros)
        if row['zero_rate_%'] > 30:  # Too many zeros = swings not detected
            dist_zero_ok = False
            issues.append(f"⚠️  WARNING: High zero rate for {row['feature']}: {row['zero_rate_%']:.2f}%")

    if dist_zero_ok:
        print("✅ PASS: Swing distance features have healthy detection rates (<30% zeros)")

    # Check value ranges
    print("\n✅ Checking value ranges against expected bounds...")
    feature_ranges = meta['feature_ranges']

    range_issues = []
    for _, row in dist_df.iterrows():
        name = row['feature']
        expected_range = feature_ranges.get(name, 'unknown')
        actual_min = row['min']
        actual_max = row['max']

        # Parse expected range
        if expected_range != 'unknown':
            import re
            match = re.match(r'\[([-\d.]+),\s*([-\d.]+)\]', expected_range)
            if match:
                exp_min = float(match.group(1))
                exp_max = float(match.group(2))

                # Allow small tolerance for numerical precision
                tolerance = 0.01
                if actual_min < exp_min - tolerance or actual_max > exp_max + tolerance:
                    range_issues.append(
                        f"   - {name}: [{actual_min:.4f}, {actual_max:.4f}] outside expected {expected_range}"
                    )

    if range_issues:
        issues.append("⚠️  WARNING: Some features outside expected ranges:")
        issues.extend(range_issues)
    else:
        print("   All features within expected ranges")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_section("5. SUMMARY")

    if not issues:
        print("🎉 ALL CHECKS PASSED!")
        print("\nFeature pipeline is working correctly:")
        print("  ✅ No NaN values")
        print("  ✅ Healthy zero rates in candle features")
        print("  ✅ Swing detection functioning")
        print("  ✅ Mask ratio correct (~85% valid)")
        print("  ✅ Feature ranges within bounds")
        return 0
    else:
        print("⚠️  ISSUES DETECTED:\n")
        for issue in issues:
            print(issue)

        # Check if issues are critical
        critical = any('FAIL' in issue for issue in issues)
        if critical:
            print("\n❌ CRITICAL ISSUES FOUND - Pipeline needs attention")
            return 1
        else:
            print("\n⚠️  Non-critical warnings found - Pipeline mostly working")
            return 0


if __name__ == "__main__":
    sys.exit(main())
