#!/usr/bin/env python3
"""Extract unlabeled 105-bar windows from raw OHLC data for SSL pre-training.

This script creates the ~118k unlabeled windows needed for TS-TCC contrastive pre-training.

Usage:
    python scripts/extract_unlabeled_windows.py \
        --input /path/to/raw_ohlc.csv \
        --output data/raw/unlabeled_windows.parquet \
        --window-size 105 \
        --stride 10 \
        --max-samples 120000 \
        --exclude data/processed/train.parquet

Output Schema:
    window_id: str              # e.g., "window_12345"
    features: list[list[float]]  # [105, 4] OHLC array (stored as list for parquet)

Notes:
    - Windows are extracted via sliding window (size=105, configurable stride)
    - Excludes any windows overlapping with labeled data (if --exclude provided)
    - Output format matches labeled data structure (minus 'label' column)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

# Paths
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_raw_ohlc(input_path: Path) -> pd.DataFrame:
    """Load raw OHLC data from CSV or parquet.

    Expected columns: timestamp, open, high, low, close (case-insensitive)

    Args:
        input_path: Path to input file (CSV or parquet)

    Returns:
        DataFrame with columns [timestamp, open, high, low, close]
    """
    logger.info(f"Loading raw OHLC from {input_path}")

    # Load based on extension
    if input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
    elif input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    # Normalize column names (case-insensitive)
    df.columns = [c.lower() for c in df.columns]

    # Validate required columns
    required = ['timestamp', 'open', 'high', 'low', 'close']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Loaded {len(df)} bars from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    return df[required]


def extract_sliding_windows(
    ohlc_df: pd.DataFrame,
    window_size: int = 105,
    stride: int = 1,
    max_samples: int = None
) -> list[dict]:
    """Extract sliding windows from OHLC time series.

    Args:
        ohlc_df: DataFrame with OHLC data
        window_size: Window size in bars (default: 105)
        stride: Stride for sliding window (default: 1, can increase for faster extraction)
        max_samples: Maximum number of windows to extract (None for all)

    Returns:
        List of window dictionaries with keys:
        - window_id: str
        - start_idx: int (index in original dataframe)
        - end_idx: int
        - timestamp_start: timestamp
        - timestamp_end: timestamp
        - features: np.ndarray[105, 4]
    """
    logger.info(f"Extracting windows: size={window_size}, stride={stride}, max={max_samples or 'all'}")

    ohlc_array = ohlc_df[['open', 'high', 'low', 'close']].values
    timestamps = ohlc_df['timestamp'].values

    windows = []
    n_windows = (len(ohlc_array) - window_size) // stride + 1

    if max_samples:
        n_windows = min(n_windows, max_samples)

    logger.info(f"Total windows to extract: {n_windows:,}")

    for i in tqdm(range(0, n_windows * stride, stride), desc="Extracting windows"):
        if i + window_size > len(ohlc_array):
            break

        # Extract window
        window_ohlc = ohlc_array[i:i + window_size]

        # Skip if window has NaN
        if np.any(np.isnan(window_ohlc)):
            continue

        windows.append({
            'window_id': f'window_{i}',
            'start_idx': i,
            'end_idx': i + window_size - 1,
            'timestamp_start': timestamps[i],
            'timestamp_end': timestamps[i + window_size - 1],
            'features': window_ohlc.copy()
        })

        if max_samples and len(windows) >= max_samples:
            break

    logger.info(f"Extracted {len(windows):,} valid windows (skipped {n_windows - len(windows)} with NaN)")

    return windows


def load_labeled_windows(labeled_path: Path) -> set:
    """Load labeled windows to exclude from unlabeled set.

    Args:
        labeled_path: Path to labeled data parquet

    Returns:
        Set of (start_timestamp, end_timestamp) tuples to exclude
    """
    if not labeled_path.exists():
        logger.warning(f"Labeled data not found at {labeled_path}, skipping exclusion")
        return set()

    logger.info(f"Loading labeled windows from {labeled_path}")
    df = pd.read_parquet(labeled_path)

    # Check if has timestamp columns
    if 'timestamp_start' not in df.columns or 'timestamp_end' not in df.columns:
        logger.warning("Labeled data missing timestamp columns, cannot exclude windows")
        return set()

    excluded = set(zip(df['timestamp_start'], df['timestamp_end']))
    logger.info(f"Will exclude {len(excluded)} labeled windows")

    return excluded


def filter_labeled_windows(windows: list[dict], excluded_timestamps: set) -> list[dict]:
    """Filter out windows that overlap with labeled data.

    Args:
        windows: List of window dictionaries
        excluded_timestamps: Set of (start_timestamp, end_timestamp) tuples

    Returns:
        Filtered list of windows
    """
    if not excluded_timestamps:
        return windows

    logger.info(f"Filtering {len(windows):,} windows to exclude labeled data")

    filtered = []
    excluded_count = 0

    for window in windows:
        # Check if window timestamps match any labeled window
        window_ts = (window['timestamp_start'], window['timestamp_end'])

        if window_ts not in excluded_timestamps:
            filtered.append(window)
        else:
            excluded_count += 1

    logger.info(f"Kept {len(filtered):,} windows, excluded {excluded_count} labeled windows")

    return filtered


def save_windows(windows: list[dict], output_path: Path):
    """Save unlabeled windows to parquet.

    Args:
        windows: List of window dictionaries
        output_path: Output parquet path
    """
    logger.info(f"Saving {len(windows):,} windows to {output_path}")

    # Convert to DataFrame
    # Store features as lists for parquet serialization
    df = pd.DataFrame([
        {
            'window_id': w['window_id'],
            'features': w['features'].tolist()  # Convert numpy array to list
        }
        for w in windows
    ])

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    df.to_parquet(output_path, index=False)

    logger.info(f"✅ Saved {len(df):,} windows to {output_path}")

    # Verify
    df_check = pd.read_parquet(output_path)
    sample_features = np.array(df_check['features'].iloc[0])

    logger.info(f"\nVerification:")
    logger.info(f"  Total windows: {len(df_check):,}")
    logger.info(f"  Columns: {list(df_check.columns)}")
    logger.info(f"  Sample features shape: {sample_features.shape}")
    logger.info(f"  Sample features type: {type(sample_features)}")

    if sample_features.shape != (105, 4):
        logger.error(f"❌ Unexpected features shape: {sample_features.shape}, expected (105, 4)")
    else:
        logger.info(f"  ✓ Features shape correct: (105, 4)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract unlabeled 105-bar windows from raw OHLC for SSL pre-training"
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input OHLC file (CSV or parquet with columns: timestamp, open, high, low, close)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "unlabeled_windows.parquet",
        help='Output parquet path (default: data/raw/unlabeled_windows.parquet)'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=105,
        help='Window size in bars (default: 105)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=10,
        help='Sliding window stride (default: 10 for faster extraction, use 1 for maximum windows)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=120000,
        help='Maximum number of windows to extract (default: 120000, use None for all)'
    )
    parser.add_argument(
        '--exclude',
        type=Path,
        help='Labeled data parquet to exclude windows from (optional)'
    )

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("EXTRACTING UNLABELED WINDOWS FOR SSL PRE-TRAINING")
    logger.info("="*70)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Window size: {args.window_size}")
    logger.info(f"Stride: {args.stride}")
    logger.info(f"Max samples: {args.max_samples or 'unlimited'}")
    logger.info(f"Exclude labeled: {args.exclude or 'none'}")
    logger.info("")

    # Validate input exists
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Load raw OHLC
    ohlc_df = load_raw_ohlc(args.input)

    # Extract windows
    windows = extract_sliding_windows(
        ohlc_df,
        window_size=args.window_size,
        stride=args.stride,
        max_samples=args.max_samples
    )

    if len(windows) == 0:
        logger.error("No windows extracted - check input data and parameters")
        sys.exit(1)

    # Exclude labeled windows if specified
    if args.exclude:
        excluded_timestamps = load_labeled_windows(args.exclude)
        windows = filter_labeled_windows(windows, excluded_timestamps)

    if len(windows) == 0:
        logger.error("No unlabeled windows remaining after exclusion")
        sys.exit(1)

    # Save to parquet
    save_windows(windows, args.output)

    logger.info("\n" + "="*70)
    logger.info("NEXT STEPS")
    logger.info("="*70)
    logger.info("1. Verify unlabeled data:")
    logger.info(f"   python -c \"")
    logger.info(f"   import pandas as pd; import numpy as np")
    logger.info(f"   df = pd.read_parquet('{args.output}')")
    logger.info(f"   print(f'Total samples: {{len(df)}}')\"")
    logger.info("")
    logger.info("2. Run SSL pre-training:")
    logger.info(f"   moola ssl-pretrain \\")
    logger.info(f"     --unlabeled {args.output} \\")
    logger.info(f"     --output data/artifacts/pretrained/encoder_weights.pt \\")
    logger.info(f"     --epochs 100 --device cuda")
    logger.info("")
    logger.info("3. Expected: ~2-3 hours GPU time for pre-training")


if __name__ == "__main__":
    main()
