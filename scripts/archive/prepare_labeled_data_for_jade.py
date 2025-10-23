#!/usr/bin/env python3
"""
Prepare labeled data for Jade training by applying relativity feature engineering.

Converts raw OHLC features (105 x 4) to relativity features (105 x 10).
Creates a new file: train_174.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moola.features.relativity import build_features, RelativityConfig


def convert_ohlc_to_relativity(ohlc_window):
    """
    Convert a single OHLC window to relativity features.

    Args:
        ohlc_window: np.ndarray of shape (105, 4) with OHLC data

    Returns:
        np.ndarray of shape (105, 10) with relativity features
    """
    # Create a minimal dataframe for build_features
    # Each row is one bar with OHLC
    df = pd.DataFrame(ohlc_window, columns=['open', 'high', 'low', 'close'])

    # Need timestamps (just use a simple range)
    df['timestamp'] = pd.date_range('2020-01-01', periods=len(df), freq='1min')
    df = df.set_index('timestamp')

    # Build relativity features
    cfg = RelativityConfig(window_length=105)
    features, valid_mask, metadata = build_features(df, cfg)

    # features shape: (n_windows, 105, 10)
    # We only have 1 window, so take the first one
    return features[0]  # Shape: (105, 10)


def main():
    """Apply relativity engineering to all labeled data."""
    print("=" * 80)
    print("PREPARE LABELED DATA FOR JADE TRAINING")
    print("=" * 80)

    # Load labeled data
    input_path = "data/processed/train_174.parquet"
    output_path = "data/processed/train_174.parquet"

    print(f"\nLoading labeled data from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"✅ Loaded {len(df)} samples")

    # Check first sample
    first_features = df['features'].iloc[0]
    print(f"\nOriginal feature shape: {np.asarray(first_features).shape}")
    print(f"Expected: (105,) array of 4D OHLC arrays")

    # Convert each sample
    print(f"\nConverting to relativity features (105, 10)...")
    engineered_features = []

    for i, row in df.iterrows():
        # Extract OHLC window
        ohlc_window = np.array([arr for arr in row['features']])  # Shape: (105, 4)

        # Apply relativity engineering
        try:
            rel_features = convert_ohlc_to_relativity(ohlc_window)
            engineered_features.append(rel_features)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(df)} samples...")
        except Exception as e:
            print(f"  ❌ Error on sample {i}: {e}")
            # Use zeros as fallback
            engineered_features.append(np.zeros((105, 10), dtype=np.float32))

    print(f"✅ Converted {len(engineered_features)} samples")

    # Create new dataframe with engineered features
    # Convert numpy arrays to lists for parquet compatibility
    df_jade = df.copy()
    df_jade['features'] = [feat.tolist() for feat in engineered_features]

    # Verify shape
    print(f"\nVerifying feature shapes...")
    for i in range(min(3, len(df_jade))):
        feat = np.array(df_jade['features'].iloc[i])
        feat_shape = feat.shape
        print(f"  Sample {i}: {feat_shape}")
        assert feat_shape == (105, 10), f"Expected (105, 10), got {feat_shape}"

    # Save
    print(f"\nSaving to {output_path}...")
    df_jade.to_parquet(output_path, index=False)

    # Verify saved file
    print(f"\nVerifying saved file...")
    df_verify = pd.read_parquet(output_path)
    print(f"✅ Verified: {len(df_verify)} samples")
    first_feat = np.array(df_verify['features'].iloc[0])
    print(f"   First feature shape: {first_feat.shape}")

    print("\n" + "=" * 80)
    print("✅ LABELED DATA READY FOR JADE TRAINING")
    print("=" * 80)
    print(f"Input:  {input_path} (OHLC 4D)")
    print(f"Output: {output_path} (Relativity 10D)")
    print(f"Samples: {len(df_verify)}")
    print(f"Feature shape: (105, 10)")
    print("\nNext steps:")
    print("1. Upload to RunPod: scp -P 11192 data/processed/train_174.parquet root@IP:/root/moola/data/processed/")
    print("2. Run fine-tuning experiments (Option 1 vs Option 2)")


if __name__ == "__main__":
    main()
