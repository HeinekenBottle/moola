#!/usr/bin/env python3
"""
Test script to verify feature variance and expansion indices integration.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from moola.features.price_action_features import engineer_classical_features

def main():
    # Load training data
    data_path = Path(__file__).parent.parent / "data" / "processed" / "train.parquet"
    df = pd.read_parquet(data_path)

    print(f"Loaded {len(df)} training samples")
    print(f"Data shape: {df.shape}")

    # Extract OHLC data and expansion indices
    X_raw = np.stack([np.stack(f) for f in df["features"]])
    expansion_start = df["expansion_start"].values
    expansion_end = df["expansion_end"].values

    print(f"X_raw shape: {X_raw.shape}")
    print(f"Expansion start range: [{expansion_start.min()}, {expansion_start.max()}]")
    print(f"Expansion end range: [{expansion_end.min()}, {expansion_end.max()}]")

    # Extract features
    print("\n" + "="*80)
    print("EXTRACTING FEATURES WITH EXPANSION INDICES")
    print("="*80)

    X = engineer_classical_features(X_raw, expansion_start, expansion_end)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")

    # Calculate feature statistics
    print("\n" + "="*80)
    print("FEATURE VARIANCE ANALYSIS")
    print("="*80)

    feature_means = np.mean(X, axis=0)
    feature_stds = np.std(X, axis=0)
    feature_vars = np.var(X, axis=0)
    feature_mins = np.min(X, axis=0)
    feature_maxs = np.max(X, axis=0)

    print(f"\nOverall Statistics:")
    print(f"  Mean variance: {np.mean(feature_vars):.4f}")
    print(f"  Median variance: {np.median(feature_vars):.4f}")
    print(f"  Min variance: {np.min(feature_vars):.4f}")
    print(f"  Max variance: {np.max(feature_vars):.4f}")

    # Check features with variance > 0.1
    high_var_features = np.sum(feature_vars > 0.1)
    low_var_features = np.sum(feature_vars <= 0.1)
    zero_var_features = np.sum(feature_vars == 0.0)

    print(f"\nVariance Distribution:")
    print(f"  Features with variance > 0.1: {high_var_features} ({high_var_features/len(feature_vars)*100:.1f}%)")
    print(f"  Features with variance <= 0.1: {low_var_features} ({low_var_features/len(feature_vars)*100:.1f}%)")
    print(f"  Features with zero variance: {zero_var_features}")

    # Show top 10 features by variance
    print("\n" + "="*80)
    print("TOP 10 FEATURES BY VARIANCE")
    print("="*80)

    top_indices = np.argsort(feature_vars)[::-1][:10]
    for i, idx in enumerate(top_indices, 1):
        print(f"{i:2d}. Feature {idx:2d}: variance={feature_vars[idx]:.4f}, "
              f"mean={feature_means[idx]:.4f}, std={feature_stds[idx]:.4f}, "
              f"range=[{feature_mins[idx]:.4f}, {feature_maxs[idx]:.4f}]")

    # Show bottom 10 features by variance
    print("\n" + "="*80)
    print("BOTTOM 10 FEATURES BY VARIANCE")
    print("="*80)

    bottom_indices = np.argsort(feature_vars)[:10]
    for i, idx in enumerate(bottom_indices, 1):
        print(f"{i:2d}. Feature {idx:2d}: variance={feature_vars[idx]:.4f}, "
              f"mean={feature_means[idx]:.4f}, std={feature_stds[idx]:.4f}, "
              f"range=[{feature_mins[idx]:.4f}, {feature_maxs[idx]:.4f}]")

    # Check for NaN or Inf values
    print("\n" + "="*80)
    print("DATA QUALITY CHECKS")
    print("="*80)

    nan_count = np.sum(np.isnan(X))
    inf_count = np.sum(np.isinf(X))

    print(f"NaN values: {nan_count}")
    print(f"Inf values: {inf_count}")

    if nan_count > 0 or inf_count > 0:
        print("\n[WARNING] Found NaN or Inf values in features!")
        return False

    # Success criteria
    print("\n" + "="*80)
    print("SUCCESS CRITERIA CHECK")
    print("="*80)

    success = True

    # Check 1: At least 50% of features have variance > 0.1
    if high_var_features / len(feature_vars) >= 0.5:
        print(f"✓ PASS: {high_var_features/len(feature_vars)*100:.1f}% of features have variance > 0.1")
    else:
        print(f"✗ FAIL: Only {high_var_features/len(feature_vars)*100:.1f}% of features have variance > 0.1 (need >= 50%)")
        success = False

    # Check 2: No zero variance features (all features should capture some variation)
    if zero_var_features == 0:
        print(f"✓ PASS: No zero-variance features")
    else:
        print(f"✗ FAIL: {zero_var_features} features have zero variance")
        success = False

    # Check 3: Mean variance should be > 0.1
    if np.mean(feature_vars) > 0.1:
        print(f"✓ PASS: Mean feature variance is {np.mean(feature_vars):.4f} > 0.1")
    else:
        print(f"✗ FAIL: Mean feature variance is {np.mean(feature_vars):.4f} <= 0.1")
        success = False

    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
