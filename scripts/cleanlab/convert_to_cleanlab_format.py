#!/usr/bin/env python3
"""
Convert Moola prediction output to CleanLab Studio format.

Usage:
    python scripts/convert_to_cleanlab_format.py \
        data/artifacts/predictions_v2_raw.csv \
        data/artifacts/predictions_v2_cleanlab.csv
"""

import sys
import pandas as pd
from pathlib import Path


def convert_to_cleanlab_format(input_csv: Path, output_csv: Path):
    """
    Convert Moola predictions to CleanLab format.

    Input format:
        window_id, prediction, prob_class_0, prob_class_1

    Output format:
        id, label, prob_consolidation, prob_retracement
    """
    df = pd.read_csv(input_csv)

    print(f"Loaded {len(df)} predictions from {input_csv}")
    print(f"Columns: {list(df.columns)}")

    # Determine which prob_class_N corresponds to which label
    # Check first row to see label-to-probability mapping
    first_label = df['prediction'].iloc[0]
    first_prob_0 = df['prob_class_0'].iloc[0]
    first_prob_1 = df['prob_class_1'].iloc[0]

    # Assume the higher probability corresponds to the predicted label
    if first_prob_0 > first_prob_1:
        # prob_class_0 is the predicted label
        if first_label == 'consolidation':
            # class_0 = consolidation, class_1 = retracement
            prob_consolidation = df['prob_class_0']
            prob_retracement = df['prob_class_1']
        else:
            # class_0 = retracement, class_1 = consolidation
            prob_consolidation = df['prob_class_1']
            prob_retracement = df['prob_class_0']
    else:
        # prob_class_1 is the predicted label
        if first_label == 'consolidation':
            # class_1 = consolidation, class_0 = retracement
            prob_consolidation = df['prob_class_1']
            prob_retracement = df['prob_class_0']
        else:
            # class_1 = retracement, class_0 = consolidation
            prob_consolidation = df['prob_class_0']
            prob_retracement = df['prob_class_1']

    print(f"\nDetected mapping:")
    print(f"  First prediction: {first_label}")
    print(f"  prob_class_0: {first_prob_0:.3f}")
    print(f"  prob_class_1: {first_prob_1:.3f}")
    print(f"  → prob_consolidation: {prob_consolidation.iloc[0]:.3f}")
    print(f"  → prob_retracement: {prob_retracement.iloc[0]:.3f}")

    # Create CleanLab format DataFrame
    cleanlab_df = pd.DataFrame({
        'id': df['window_id'],
        'label': df['prediction'],
        'prob_consolidation': prob_consolidation,
        'prob_retracement': prob_retracement
    })

    # Save to CSV
    cleanlab_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved {len(cleanlab_df)} predictions to {output_csv}")
    print(f"\nFirst 5 rows:")
    print(cleanlab_df.head())

    # Verify probabilities sum to 1
    prob_sums = cleanlab_df['prob_consolidation'] + cleanlab_df['prob_retracement']
    if not prob_sums.between(0.99, 1.01).all():
        print("\n⚠️  WARNING: Some probabilities don't sum to 1.0")
        print(f"   Min sum: {prob_sums.min():.4f}")
        print(f"   Max sum: {prob_sums.max():.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_cleanlab_format.py <input_csv> <output_csv>")
        sys.exit(1)

    input_csv = Path(sys.argv[1])
    output_csv = Path(sys.argv[2])

    if not input_csv.exists():
        print(f"Error: Input file not found: {input_csv}")
        sys.exit(1)

    convert_to_cleanlab_format(input_csv, output_csv)
