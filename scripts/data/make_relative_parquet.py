#!/usr/bin/env python3
"""Convert 4D OHLC parquet to 11D RelativeTransform parquet.

This script transforms absolute OHLC price data into scale-invariant relative features
using the RelativeFeatureTransform class.

Input: [N, 105, 4] OHLC data
Output: [N, 105, 11] relative features

Features generated:
- 4 log returns: log(price_t / price_t-1) for O, H, L, C
- 3 candle ratios: body/range, upper_wick/range, lower_wick/range
- 4 rolling z-scores: standardized values over 20-bar window for O, H, L, C

Usage:
    python scripts/data/make_relative_parquet.py \\
        --input data/processed/labeled/train_latest.parquet \\
        --output data/processed/labeled/train_latest_relative.parquet
"""

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

# Add moola to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from moola.features.relative_transform import RelativeFeatureTransform


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Input parquet file with 4D OHLC data",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    required=True,
    help="Output parquet file for 11D relative features",
)
@click.option(
    "--validate",
    is_flag=True,
    default=True,
    help="Validate output shape and feature ranges",
)
def main(input_path, output_path, validate):
    """Convert 4D OHLC parquet to 11D RelativeTransform parquet."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    print("=" * 70)
    print("CONVERT 4D OHLC → 11D RELATIVETRANSFORM")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    # Load 4D OHLC data
    print("[1/4] Loading 4D OHLC data...")
    df = pd.read_parquet(input_path)
    print(f"  ✓ Loaded {len(df)} samples")
    print(f"  Columns: {list(df.columns)}")

    # Extract OHLC features
    print("\n[2/4] Extracting OHLC features...")
    if "features" in df.columns:
        # New format: features column with nested arrays
        # Each row is an array of 105 timesteps, each timestep is an array of 4 OHLC values
        features_list = []
        for feat in df["features"]:
            # feat is shape (105,) where each element is an array of 4 values
            # Stack them to get (105, 4)
            stacked = np.stack([np.array(bar, dtype=np.float32) for bar in feat])
            features_list.append(stacked)
        ohlc_data = np.stack(features_list)
    else:
        # Old format: flat feature columns
        feature_cols = [c for c in df.columns if c not in {"window_id", "label"}]
        ohlc_raw = df[feature_cols].values
        N = len(ohlc_raw)
        ohlc_data = ohlc_raw.reshape(N, 105, 4)

    print(f"  ✓ OHLC shape: {ohlc_data.shape}")

    # Validate 4D shape
    if ohlc_data.shape[1:] != (105, 4):
        print(f"  ✗ ERROR: Expected shape [N, 105, 4], got {ohlc_data.shape}")
        sys.exit(1)

    # Transform to 11D relative features
    print("\n[3/4] Transforming to 11D relative features...")
    transformer = RelativeFeatureTransform(eps=1e-8)
    relative_data = transformer.transform(ohlc_data)
    print(f"  ✓ Relative shape: {relative_data.shape}")

    # Validate 11D shape
    if relative_data.shape != (len(df), 105, 11):
        print(f"  ✗ ERROR: Expected shape [{len(df)}, 105, 11], got {relative_data.shape}")
        sys.exit(1)

    # Create output dataframe
    print("\n[4/4] Creating output parquet...")
    df_output = df.copy()
    # Convert numpy arrays to nested lists for parquet compatibility
    # Each sample: (105, 11) → list of 105 timesteps, each with 11 features
    df_output["features"] = [
        [timestep.tolist() for timestep in sample]
        for sample in relative_data
    ]

    # Add metadata
    if "feature_type" not in df_output.columns:
        df_output["feature_type"] = "relative_11d"

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_parquet(output_path, index=False)
    print(f"  ✓ Saved to {output_path}")

    # Validation
    if validate:
        print("\n[VALIDATION]")
        print(f"  Feature names: {transformer.get_feature_names()}")
        print(f"  Shape: {relative_data.shape}")
        print(f"  Dtype: {relative_data.dtype}")
        print(f"  Range: [{relative_data.min():.4f}, {relative_data.max():.4f}]")
        print(f"  Mean: {relative_data.mean():.4f}")
        print(f"  Std: {relative_data.std():.4f}")
        print(f"  NaN count: {np.isnan(relative_data).sum()}")
        print(f"  Inf count: {np.isinf(relative_data).sum()}")

        # Check for issues
        if np.isnan(relative_data).any():
            print("  ⚠️  WARNING: NaN values detected")
        if np.isinf(relative_data).any():
            print("  ⚠️  WARNING: Inf values detected")

        # Feature-wise statistics
        print("\n  Feature-wise statistics:")
        feature_names = transformer.get_feature_names()
        for i, name in enumerate(feature_names):
            feat_data = relative_data[:, :, i]
            print(f"    {i:2d}. {name:25s} | "
                  f"range=[{feat_data.min():7.4f}, {feat_data.max():7.4f}] | "
                  f"mean={feat_data.mean():7.4f} | "
                  f"std={feat_data.std():7.4f}")

    print("\n" + "=" * 70)
    print("✅ SUCCESS! 11D RelativeTransform dataset created")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Pretrain encoder:")
    print(f"     python -m moola.cli pretrain-bilstm \\")
    print(f"       --input {output_path} \\")
    print(f"       --input-dim 11 \\")
    print(f"       --output artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt \\")
    print(f"       --device cuda --epochs 50 --seed 17")
    print(f"\n  2. Fine-tune model:")
    print(f"     python -m moola.cli train \\")
    print(f"       --model enhanced_simple_lstm \\")
    print(f"       --data {output_path} \\")
    print(f"       --split data/splits/fwd_chain_v3.json \\")
    print(f"       --pretrained-encoder artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt \\")
    print(f"       --freeze-encoder --epochs 5 --seed 17")
    print()


if __name__ == "__main__":
    main()

