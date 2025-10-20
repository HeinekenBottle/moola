#!/usr/bin/env python3
"""
Fix missing features in train_latest.parquet by merging from source batch files.

The train_latest.parquet has 70 candlesticks samples with null features.
This script populates them from the original batch files:
- batch_200_clean_keepers.parquet (33 samples)
- batch_201.parquet (16 samples)
- batch_203_annotation_ready.parquet (5 samples)
- batch_204_annotation_ready.parquet (15 samples)
- Plus 1 mystery sample with window_id='1'
"""
import pandas as pd
from pathlib import Path


def main():
    # Paths
    train_path = Path("data/processed/train_latest.parquet")
    backup_path = Path("data/processed/train_latest_backup_pre_fix.parquet")

    batch_files = {
        "batch_200_keepers": Path("data/batches/batch_200_clean_keepers.parquet"),
        "batch_201": Path("data/batches/batch_201.parquet"),
        "batch_203": Path("data/batches/batch_203_annotation_ready.parquet"),
        "batch_204": Path("data/batches/batch_204_annotation_ready.parquet"),
    }

    print("=" * 70)
    print("🔧 FIXING MISSING FEATURES IN TRAIN_LATEST.PARQUET")
    print("=" * 70)

    # Load train_latest
    train = pd.read_parquet(train_path)
    print(f"\nLoaded train_latest: {len(train)} rows")

    # Backup original file
    train.to_parquet(backup_path)
    print(f"✅ Created backup: {backup_path.name}")

    # Count missing features
    cs_samples = train[train['source'] == 'candlesticks']
    missing_count = cs_samples['features'].isnull().sum()
    print(f"\nCandlesticks samples: {len(cs_samples)}")
    print(f"Missing features: {missing_count}")

    # Load all batch files into a combined lookup
    batch_data = {}
    for name, path in batch_files.items():
        if path.exists():
            df = pd.read_parquet(path)
            print(f"\nLoaded {name}: {len(df)} rows")
            # Index by window_id for fast lookup
            for idx, row in df.iterrows():
                window_id = row['window_id']
                batch_data[window_id] = row
        else:
            print(f"⚠️  {name} not found at {path}")

    print(f"\nTotal windows in batch lookup: {len(batch_data)}")

    # Fix missing features
    fixed_count = 0
    not_found = []

    for idx, row in train.iterrows():
        # Only process candlesticks samples with null features
        if row['source'] == 'candlesticks' and pd.isnull(row['features']):
            window_id = row['window_id']

            if window_id in batch_data:
                # Copy features from batch
                train.at[idx, 'features'] = batch_data[window_id]['features']
                fixed_count += 1
            else:
                not_found.append(window_id)

    print(f"\n--- RESULTS ---")
    print(f"Fixed: {fixed_count}/{missing_count}")
    print(f"Not found: {len(not_found)}")

    if not_found:
        print(f"\nWindow IDs not found in batches:")
        for wid in not_found[:10]:  # Show first 10
            print(f"  - {wid}")
        if len(not_found) > 10:
            print(f"  ... and {len(not_found) - 10} more")

    # Verify fix
    cs_samples_after = train[train['source'] == 'candlesticks']
    missing_after = cs_samples_after['features'].isnull().sum()

    print(f"\n--- VERIFICATION ---")
    print(f"Missing features before: {missing_count}")
    print(f"Missing features after: {missing_after}")
    print(f"Improvement: {missing_count - missing_after} features populated")

    # Save updated file
    if fixed_count > 0:
        train.to_parquet(train_path)
        print(f"\n✅ Saved updated train_latest.parquet")
        print(f"   Original backed up to: {backup_path.name}")
    else:
        print(f"\n❌ No features were fixed - not saving")

    print("\n" + "=" * 70)
    if missing_after == 0:
        print("✅ ALL FEATURES POPULATED! Ready for training.")
    elif missing_after < missing_count:
        print(f"⚠️  Partially fixed. {missing_after} features still missing.")
    else:
        print(f"❌ No improvement. Check batch files and window IDs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
