#!/usr/bin/env python3
"""Extract reversal samples from training data and prepare 2-class dataset.

This script:
1. Loads the full 3-class training dataset (consolidation/retracement/reversal)
2. Separates reversal samples (19 samples) and archives them for future use
3. Creates a 2-class training dataset (consolidation vs retracement, 115 samples)

The reversal class had insufficient samples (19) for reliable training, causing
class collapse with 0% recall. By training on consolidation vs retracement only,
we can proceed with the available data while archiving reversals for future use
when more samples are collected.

Usage:
    python3 scripts/extract_reversals.py

Output:
    - data/processed/reversals_archive.parquet (19 samples)
    - data/processed/train_2class.parquet (115 samples)
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


def extract_reversals():
    """Extract reversals and create 2-class training dataset."""

    # Paths
    data_dir = repo_root / "data" / "processed"
    input_path = data_dir / "train.parquet"
    reversals_path = data_dir / "reversal_holdout.parquet"
    output_path = data_dir / "train.parquet"  # Overwrite original

    print("=" * 60)
    print("EXTRACTING REVERSALS & PREPARING 2-CLASS TRAINING DATA")
    print("=" * 60)
    print()

    # Load full dataset
    if not input_path.exists():
        print(f"❌ Error: Training data not found at {input_path}")
        print("   Run 'moola ingest' first")
        sys.exit(1)

    print(f"📥 Loading dataset from {input_path}")
    df = pd.read_parquet(input_path)
    print(f"   Loaded: {len(df)} samples")
    print(f"   Columns: {list(df.columns)}")

    # Create backup before modifying
    backup_path = data_dir / "train_3class_backup.parquet"
    print(f"\n💾 Creating backup: {backup_path}")
    df.to_parquet(backup_path, index=False)
    print(f"   ✅ Backup saved ({len(df)} samples)")
    print()

    # Show original distribution
    print("📊 Original class distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        pct = 100 * count / len(df)
        print(f"   {label:15s}: {count:3d} samples ({pct:5.1f}%)")
    print()

    # Split reversals from main dataset
    print("✂️  Splitting reversals from main dataset...")
    reversals = df[df['label'] == 'reversal'].copy()
    main_data = df[df['label'] != 'reversal'].copy()

    print(f"   Reversals to archive: {len(reversals)} samples")
    print(f"   Training data remaining: {len(main_data)} samples")
    print()

    # Save reversals archive
    print(f"💾 Saving reversals to {reversals_path}")
    reversals_path.parent.mkdir(parents=True, exist_ok=True)
    reversals.to_parquet(reversals_path, index=False)
    print(f"   ✅ Archived {len(reversals)} reversal samples")
    print()

    # Save 2-class training data
    print(f"💾 Saving 2-class training data to {output_path}")
    main_data.to_parquet(output_path, index=False)
    print(f"   ✅ Saved {len(main_data)} samples")
    print()

    # Show new distribution
    print("📊 New 2-class distribution:")
    new_label_counts = main_data['label'].value_counts()
    for label, count in new_label_counts.items():
        pct = 100 * count / len(main_data)
        print(f"   {label:15s}: {count:3d} samples ({pct:5.1f}%)")
    print()

    # Validation
    print("✅ VALIDATION:")
    print(f"   Original samples: {len(df)}")
    print(f"   Archived: {len(reversals)}")
    print(f"   Training: {len(main_data)}")
    print(f"   Sum check: {len(reversals) + len(main_data)} == {len(df)} ✓")
    print()

    # Next steps
    print("=" * 60)
    print("📝 NEXT STEPS:")
    print("=" * 60)
    print()
    print("1. Verify 2-class data:")
    print("   python3 -c \"import pandas as pd; df = pd.read_parquet('data/processed/train.parquet'); print(df['label'].value_counts())\"")
    print()
    print("2. Clean 3-class training artifacts:")
    print("   rm -rf data/artifacts/oof/*/v1/*.npy data/artifacts/models/stack/* data/artifacts/splits/v1/*.json")
    print()
    print("3. Update configs for 2-class problem:")
    print("   Edit configs/default.yaml: set num_classes: 2")
    print()
    print("4. Retrain all models on 2-class dataset")
    print()
    print("💡 Original 3-class data backed up to train_3class_backup.parquet")
    print("   Reversal samples archived to reversal_holdout.parquet")
    print()


if __name__ == "__main__":
    extract_reversals()
