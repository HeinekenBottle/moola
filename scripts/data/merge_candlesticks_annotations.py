#!/usr/bin/env python3
"""
Merge Candlesticks good quality annotations with existing training data.

Extracts A/B/C quality annotations from candlesticks and combines them
with the existing train.parquet to create an expanded training set.
"""

import json
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_good_candlesticks_annotations():
    """Load all good quality (A/B/C) annotations from candlesticks."""
    master_path = Path("/Users/jack/projects/moola/data/corrections/candlesticks_annotations/master_index.csv")
    annotation_dir = Path("/Users/jack/projects/moola/data/corrections/candlesticks_annotations")

    good_annotations = []

    with open(master_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            quality = row.get('quality_grade')
            if quality in ['A', 'B', 'C']:
                batch_file = row['batch_file']
                annotation_path = annotation_dir / batch_file

                if annotation_path.exists():
                    with open(annotation_path, 'r') as af:
                        annotation = json.load(af)
                        good_annotations.append(annotation)

    print(f"✅ Loaded {len(good_annotations)} good quality candlesticks annotations")
    return good_annotations


def convert_annotation_to_training_format(annotation):
    """Convert candlesticks annotation to training data format."""
    window_id = annotation['window_id']
    expansions = annotation.get('expansions', [])

    # Convert each expansion to a training sample
    samples = []

    for expansion in expansions:
        exp_type = expansion['expansion_type']
        start_idx = expansion['start_index']
        end_idx = expansion['end_index']

        # Use string labels to match existing training data format
        sample = {
            'window_id': window_id,
            'label': exp_type,  # Keep as string: 'consolidation', 'retracement', 'reversal'
            'expansion_start': start_idx,
            'expansion_end': end_idx,
            'quality': expansion['quality_grade'],
            'source': 'candlesticks'
        }

        samples.append(sample)

    return samples


def merge_with_existing_training_data(candlesticks_samples):
    """Merge candlesticks samples with existing training data."""
    existing_train_path = Path("/Users/jack/projects/moola/data/processed/train.parquet")

    # Load existing training data
    if existing_train_path.exists():
        existing_df = pd.read_parquet(existing_train_path)
        print(f"📊 Existing training data: {len(existing_df)} samples")
    else:
        existing_df = pd.DataFrame()
        print(f"⚠️  No existing training data found")

    # Convert candlesticks to DataFrame
    candlesticks_df = pd.DataFrame(candlesticks_samples)
    print(f"📊 Candlesticks data: {len(candlesticks_df)} samples")

    # Combine
    if len(existing_df) > 0:
        # Ensure compatible columns
        common_cols = list(set(existing_df.columns) & set(candlesticks_df.columns))

        if 'source' not in existing_df.columns:
            existing_df['source'] = 'original'

        combined_df = pd.concat([existing_df, candlesticks_df], ignore_index=True)
    else:
        combined_df = candlesticks_df

    print(f"📊 Combined data: {len(combined_df)} samples")

    return combined_df


def save_combined_training_data(df):
    """Save combined training data."""
    output_path = Path(f"/Users/jack/projects/moola/data/processed/train_combined_{len(df)}.parquet")

    df.to_parquet(output_path, index=False)
    print(f"✅ Saved combined training data to: {output_path}")

    # Create symlink
    symlink_path = Path("/Users/jack/projects/moola/data/processed/train_latest.parquet")
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()
    symlink_path.symlink_to(output_path.name)
    print(f"✅ Created symlink: train_latest.parquet -> {output_path.name}")

    return output_path


def main():
    """Main merge pipeline."""
    print("=" * 80)
    print("MERGING CANDLESTICKS ANNOTATIONS WITH TRAINING DATA")
    print("=" * 80)

    # Load candlesticks annotations
    annotations = load_good_candlesticks_annotations()

    # Convert to training format
    print("\nConverting to training format...")
    all_samples = []
    for annotation in annotations:
        samples = convert_annotation_to_training_format(annotation)
        all_samples.extend(samples)

    print(f"✅ Converted {len(all_samples)} training samples")

    # Distribution
    labels = [s['label'] for s in all_samples]
    print(f"\nLabel distribution:")
    print(f"   Consolidation (0): {labels.count(0)}")
    print(f"   Retracement (1): {labels.count(1)}")
    print(f"   Reversal (2): {labels.count(2)}")

    # Merge with existing
    print("\nMerging with existing training data...")
    combined_df = merge_with_existing_training_data(all_samples)

    # Save
    print("\nSaving combined dataset...")
    output_path = save_combined_training_data(combined_df)

    print("\n" + "=" * 80)
    print("✅ MERGE COMPLETE")
    print("=" * 80)
    print(f"Training data location: {output_path}")
    print(f"Symlink: /Users/jack/projects/moola/data/processed/train_latest.parquet")
    print(f"\nYou can now use this for training!")


if __name__ == "__main__":
    main()
