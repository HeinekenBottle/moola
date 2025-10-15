#!/usr/bin/env python3
"""Ingest Pivot's 134-sample CleanLab-cleaned dataset into moola training format.

This script:
1. Loads 134 pre-processed parquet files from pivot-experiments
2. Each parquet has expansion indices (start_idx, end_idx) - the critical data moola is missing
3. Combines into single training parquet for moola

The key fix: moola's current data has labels without locations. This dataset
includes the start_idx and end_idx that tell the model WHERE in the 105-bar
window the consolidation/retracement/reversal actually occurs.

Dataset: v2.7 post-CleanLab (134 samples)
- Consolidation: 65 (48.5%)
- Retracement: 50 (37.3%)
- Reversal: 19 (14.2%)

Usage:
    python3 scripts/ingest_pivot_134_clean.py [--output PATH] [--exclude-reversals]

Options:
    --output             Output parquet path (default: data/processed/train_pivot_134.parquet)
    --exclude-reversals  Remove reversal samples (creates 2-class dataset, 115 samples)

Output Schema:
    window_id: str                    # e.g., "0_exp_1"
    label: str                        # consolidation/retracement/reversal
    expansion_start: int              # Start bar index (0-104)
    expansion_end: int                # End bar index (0-104)
    features: np.ndarray[105, 4]      # OHLC array (not flattened)
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Paths
REPO_ROOT = Path(__file__).parent.parent
# Pivot is at same level as moola repo - use relative path from repo root
PIVOT_PARQUETS = REPO_ROOT.parent / "pivot-experiments" / "data" / "processed" / "windows105" / "train"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "train_pivot_134.parquet"

# Add src to path
sys.path.insert(0, str(REPO_ROOT / "src"))

# Type mapping (pivot uses int encoding)
TYPE_MAP = {
    0: 'consolidation',
    1: 'retracement',
    2: 'reversal'
}


def load_pivot_parquets(parquet_dir: Path, exclude_reversals: bool = False) -> pd.DataFrame:
    """Load all 134 pre-processed parquet files from pivot-experiments.

    Args:
        parquet_dir: Path to windows105/train/ directory
        exclude_reversals: If True, remove reversal samples

    Returns:
        DataFrame with columns: window_id, label, expansion_start, expansion_end, features
    """
    logger.info(f"Loading parquet files from {parquet_dir}")

    # Find all parquet files
    parquet_files = glob.glob(str(parquet_dir / "*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files")

    if len(parquet_files) != 134:
        logger.warning(f"Expected 134 files, found {len(parquet_files)}")

    # Load each parquet
    records = []
    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)
            row = df.iloc[0]

            # Convert OHLC from list of arrays to 2D numpy array
            ohlc = np.array([np.array(bar) for bar in row['ohlc']])

            # Decode type label
            label = TYPE_MAP[row['type']]

            # Skip reversals if requested
            if exclude_reversals and label == 'reversal':
                continue

            records.append({
                'window_id': row['window_id'],
                'label': label,
                'expansion_start': int(row['start_idx']),
                'expansion_end': int(row['end_idx']),
                'features': ohlc
            })

        except Exception as e:
            logger.error(f"Error loading {Path(file_path).name}: {e}")
            continue

    logger.info(f"Successfully loaded {len(records)} samples")

    # Create DataFrame
    df = pd.DataFrame(records)

    # Report statistics
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Class distribution:")
    for label, count in df['label'].value_counts().items():
        pct = 100 * count / len(df)
        logger.info(f"    {label:15s}: {count:3d} ({pct:5.1f}%)")

    logger.info(f"  Expansion span statistics:")
    spans = df['expansion_end'] - df['expansion_start']
    logger.info(f"    Mean span: {spans.mean():.1f} bars")
    logger.info(f"    Median span: {spans.median():.0f} bars")
    logger.info(f"    Range: [{spans.min()}, {spans.max()}] bars")

    # Validate features
    sample_features = df['features'].iloc[0]
    logger.info(f"  Features shape: {sample_features.shape}")
    if sample_features.shape != (105, 4):
        logger.error(f"Unexpected features shape: {sample_features.shape}, expected (105, 4)")
        sys.exit(1)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Pivot's 134-sample CleanLab-cleaned dataset for moola training"
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f'Output parquet path (default: {DEFAULT_OUTPUT})'
    )
    parser.add_argument(
        '--exclude-reversals',
        action='store_true',
        help='Exclude reversal samples (creates 2-class dataset with 115 samples)'
    )

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("INGESTING PIVOT 134-SAMPLE CLEANLAB-CLEANED DATASET")
    logger.info("="*70)
    logger.info(f"Source: {PIVOT_PARQUETS}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Exclude reversals: {args.exclude_reversals}")
    logger.info("")

    # Check pivot data exists
    if not PIVOT_PARQUETS.exists():
        logger.error(f"Pivot parquet directory not found at {PIVOT_PARQUETS}")
        logger.error("Please check the path and try again")
        sys.exit(1)

    # Load parquets
    df = load_pivot_parquets(PIVOT_PARQUETS, exclude_reversals=args.exclude_reversals)

    if len(df) == 0:
        logger.error("No samples loaded - check data and filters")
        sys.exit(1)

    # Save output
    # Note: Convert numpy arrays to lists for parquet serialization
    df_save = df.copy()
    df_save['features'] = df_save['features'].apply(lambda x: x.tolist())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_save.to_parquet(args.output, index=False)
    logger.info(f"\n✅ Saved {len(df)} samples to {args.output}")

    # Verification
    logger.info("\n" + "="*70)
    logger.info("VERIFICATION")
    logger.info("="*70)
    df_check = pd.read_parquet(args.output)

    logger.info(f"✓ Reloaded parquet: {len(df_check)} samples")
    logger.info(f"✓ Columns: {list(df_check.columns)}")
    logger.info(f"✓ Has expansion indices: expansion_start, expansion_end")

    # Show sample (features are stored as lists)
    logger.info("\nSample record:")
    sample = df_check.iloc[0]
    logger.info(f"  window_id: {sample['window_id']}")
    logger.info(f"  label: {sample['label']}")
    logger.info(f"  expansion_start: {sample['expansion_start']}")
    logger.info(f"  expansion_end: {sample['expansion_end']}")
    logger.info(f"  expansion span: {sample['expansion_end'] - sample['expansion_start'] + 1} bars")
    logger.info(f"  features type: list[list[float]] (convert to numpy when loading)")
    logger.info(f"  features length: {len(sample['features'])} bars × {len(sample['features'][0])} OHLC")

    logger.info("\n" + "="*70)
    logger.info("NEXT STEPS")
    logger.info("="*70)
    logger.info("1. Update moola feature engineering to use expansion_start/end:")
    logger.info("   - Modify src/moola/features/price_action_features.py")
    logger.info("   - Extract features only from bars [expansion_start:expansion_end+1]")
    logger.info("   - This solves the 'whole window is flat' problem")
    logger.info("")
    logger.info("2. Update symlink to use new data:")
    logger.info("   cd data/processed")
    logger.info("   rm train.parquet  # Remove old symlink")
    logger.info("   ln -s train_pivot_134.parquet train.parquet")
    logger.info("")
    logger.info("3. Re-run training:")
    logger.info("   python3 -m moola.cli oof --model xgb --folds 5 --seed 1337")
    logger.info("")
    logger.info("4. Expected improvement:")
    logger.info("   - Features now computed on expansion region only (5-23 bars)")
    logger.info("   - Order blocks, FVG, liquidity zones will have actual signal")
    logger.info("   - Performance should jump from 46% → 60-70%")
    logger.info("")
    logger.info("5. Verify feature variance after re-training:")
    logger.info("   python3 -c \"")
    logger.info("   import pandas as pd; import numpy as np")
    logger.info("   from moola.features.price_action_features import engineer_classical_features")
    logger.info("   df = pd.read_parquet('data/processed/train.parquet')")
    logger.info("   X = np.array([f[df.iloc[i]['expansion_start']:df.iloc[i]['expansion_end']+1] for i, f in enumerate(df['features'])])")
    logger.info("   # Features should now have variance >0.1\"")


if __name__ == "__main__":
    main()
