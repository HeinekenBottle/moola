"""Generate overlapping windows from raw NQ data using Candlesticks timestamps.

Maps train_latest windows to their original timestamps in Candlesticks annotations,
then extracts overlapping windows at stride=52 from raw NQ data to double effective
training data: 174 → ~350 windows.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

import json
import re
from typing import Optional

import numpy as np
import pandas as pd


def load_candlesticks_annotations(annotations_dir: str) -> pd.DataFrame:
    """Load all Candlesticks annotations into a DataFrame.

    Args:
        annotations_dir: Path to multi_expansion_annotations_v2 directory

    Returns:
        DataFrame with columns: window_id, center_timestamp, expansions
    """
    master_index = pd.read_csv(f"{annotations_dir}/master_index.csv")
    master_index["center_timestamp"] = pd.to_datetime(
        master_index["center_timestamp"], format="mixed", utc=True
    )

    # Load expansion details from batch files
    annotations = []
    for idx, row in master_index.iterrows():
        batch_file = f"{annotations_dir}/{row['batch_file']}"
        try:
            with open(batch_file) as f:
                batch_data = json.load(f)

            # Each batch file is a list with one annotation
            if isinstance(batch_data, list) and len(batch_data) > 0:
                anno = batch_data[0]
                annotations.append(
                    {
                        "window_id": row["window_id"],
                        "center_timestamp": row["center_timestamp"],
                        "window_quality": row["window_quality"],
                        "expansions": anno.get("expansions", []),
                    }
                )
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Couldn't load {batch_file}: {e}")

    return pd.DataFrame(annotations)


def parse_window_id(window_id: str) -> tuple[int, int]:
    """Parse window_id like '0_exp_1' to (base_id, exp_num).

    Args:
        window_id: Window ID string

    Returns:
        (base_window_id, expansion_number)
    """
    match = re.match(r"(\d+)_exp_(\d+)", str(window_id))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def find_window_in_raw(
    raw_df: pd.DataFrame, center_timestamp: pd.Timestamp, window_length: int = 105
) -> Optional[int]:
    """Find the center index of a window in raw NQ data.

    Args:
        raw_df: Raw NQ OHLCV data with timestamp index
        center_timestamp: Center timestamp of the window
        window_length: Window size (default: 105)

    Returns:
        Center index in raw_df, or None if not found
    """
    # Ensure raw_df has datetime index
    if not isinstance(raw_df.index, pd.DatetimeIndex):
        if "timestamp" in raw_df.columns:
            raw_df = raw_df.set_index("timestamp")
        else:
            raise ValueError("raw_df must have datetime index or timestamp column")

    # Find closest timestamp
    try:
        center_idx = raw_df.index.get_indexer([center_timestamp], method="nearest")[0]

        # Verify we have enough data around this center
        half_window = window_length // 2
        if center_idx < half_window or center_idx >= len(raw_df) - half_window:
            return None

        return center_idx
    except (KeyError, IndexError):
        return None


def extract_overlapping_windows(
    raw_df: pd.DataFrame,
    center_idx: int,
    expansion_start: int,
    expansion_end: int,
    label: str,
    window_id: str,
    window_length: int = 105,
    stride: int = 52,
    max_overlaps: int = 2,
) -> list[dict]:
    """Extract overlapping windows around a center position.

    Args:
        raw_df: Raw NQ data
        center_idx: Center index in raw_df
        expansion_start: Expansion start in original 105-bar window
        expansion_end: Expansion end in original 105-bar window
        label: Pattern label (consolidation/retracement)
        window_id: Original window ID
        window_length: Window size (default: 105)
        stride: Overlap stride (default: 52 = 50%)
        max_overlaps: Maximum overlaps in each direction

    Returns:
        List of window dictionaries with OHLC features and adjusted labels
    """
    half_window = window_length // 2
    windows = []

    # Original window (offset=0)
    start_idx = center_idx - half_window
    end_idx = start_idx + window_length

    if start_idx >= 0 and end_idx <= len(raw_df):
        ohlc = raw_df.iloc[start_idx:end_idx][["open", "high", "low", "close"]].values
        windows.append(
            {
                "window_id": f"{window_id}_offset0",
                "base_window_id": window_id,
                "offset": 0,
                "features": ohlc,
                "label": label,
                "expansion_start": expansion_start,
                "expansion_end": expansion_end,
                "overlap_fraction": 1.0,
            }
        )

    # Forward overlaps
    for i in range(1, max_overlaps + 1):
        offset = i * stride
        new_start_idx = start_idx + offset
        new_end_idx = new_start_idx + window_length

        if new_end_idx <= len(raw_df):
            # Adjust expansion pointers
            new_exp_start = expansion_start - offset
            new_exp_end = expansion_end - offset

            # Calculate overlap fraction
            if new_exp_end < 0 or new_exp_start >= window_length:
                overlap_frac = 0.0
            else:
                new_exp_start_clipped = max(0, new_exp_start)
                new_exp_end_clipped = min(window_length - 1, new_exp_end)
                visible_length = new_exp_end_clipped - new_exp_start_clipped + 1
                original_length = expansion_end - expansion_start + 1
                overlap_frac = visible_length / original_length if original_length > 0 else 0.0

            # Only keep if overlap is meaningful (>30%)
            if overlap_frac >= 0.3:
                ohlc = raw_df.iloc[new_start_idx:new_end_idx][
                    ["open", "high", "low", "close"]
                ].values
                windows.append(
                    {
                        "window_id": f"{window_id}_offset+{offset}",
                        "base_window_id": window_id,
                        "offset": offset,
                        "features": ohlc,
                        "label": label,
                        "expansion_start": max(0, new_exp_start),
                        "expansion_end": min(window_length - 1, new_exp_end),
                        "overlap_fraction": overlap_frac,
                    }
                )

    # Backward overlaps
    for i in range(1, max_overlaps + 1):
        offset = -i * stride
        new_start_idx = start_idx + offset
        new_end_idx = new_start_idx + window_length

        if new_start_idx >= 0:
            # Adjust expansion pointers
            new_exp_start = expansion_start - offset
            new_exp_end = expansion_end - offset

            # Calculate overlap fraction
            if new_exp_end < 0 or new_exp_start >= window_length:
                overlap_frac = 0.0
            else:
                new_exp_start_clipped = max(0, new_exp_start)
                new_exp_end_clipped = min(window_length - 1, new_exp_end)
                visible_length = new_exp_end_clipped - new_exp_start_clipped + 1
                original_length = expansion_end - expansion_start + 1
                overlap_frac = visible_length / original_length if original_length > 0 else 0.0

            # Only keep if overlap is meaningful (>30%)
            if overlap_frac >= 0.3:
                ohlc = raw_df.iloc[new_start_idx:new_end_idx][
                    ["open", "high", "low", "close"]
                ].values
                windows.append(
                    {
                        "window_id": f"{window_id}_offset{offset}",
                        "base_window_id": window_id,
                        "offset": offset,
                        "features": ohlc,
                        "label": label,
                        "expansion_start": max(0, new_exp_start),
                        "expansion_end": min(window_length - 1, new_exp_end),
                        "overlap_fraction": overlap_frac,
                    }
                )

    return windows


def main():
    print("=" * 80)
    print("OVERLAPPING WINDOWS FROM RAW NQ DATA")
    print("=" * 80)

    # Paths
    train_latest_path = "data/processed/labeled/train_latest.parquet"
    annotations_dir = (
        "/Users/jack/projects/candlesticks/data/corrections/multi_expansion_annotations_v2"
    )
    raw_nq_path = "data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet"
    output_path = "data/processed/labeled/train_latest_overlaps_v2.parquet"

    # Load data
    print(f"\n1. Loading train_latest ({train_latest_path})...")
    train_df = pd.read_parquet(train_latest_path)
    print(f"   {len(train_df)} windows")

    print(f"\n2. Loading Candlesticks annotations ({annotations_dir})...")
    annotations_df = load_candlesticks_annotations(annotations_dir)
    print(f"   {len(annotations_df)} annotated windows")

    print(f"\n3. Loading raw NQ data ({raw_nq_path})...")
    raw_df = pd.read_parquet(raw_nq_path)
    if "timestamp" in raw_df.columns:
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True)
        raw_df = raw_df.set_index("timestamp")
    else:
        # Index is already datetime, convert to UTC
        raw_df.index = pd.to_datetime(raw_df.index, utc=True)
    print(f"   {len(raw_df)} bars from {raw_df.index[0]} to {raw_df.index[-1]}")

    # Map train_latest windows to annotations
    print("\n4. Mapping train_latest windows to annotations...")
    expanded_windows = []
    matched_count = 0
    skipped_count = 0

    for idx, row in train_df.iterrows():
        window_id = row["window_id"]
        base_id, exp_num = parse_window_id(window_id)

        if base_id is None:
            print(f"   Warning: Couldn't parse window_id '{window_id}'")
            skipped_count += 1
            continue

        # Find annotation
        anno_row = annotations_df[annotations_df["window_id"] == base_id]
        if anno_row.empty:
            print(f"   Warning: No annotation found for base_id {base_id}")
            skipped_count += 1
            continue

        center_timestamp = anno_row.iloc[0]["center_timestamp"]
        expansions = anno_row.iloc[0]["expansions"]

        # Find the specific expansion (exp_num is 1-indexed)
        if exp_num > len(expansions):
            print(
                f"   Warning: exp_num {exp_num} > {len(expansions)} expansions in window {base_id}"
            )
            skipped_count += 1
            continue

        # Find center in raw NQ
        center_idx = find_window_in_raw(raw_df, center_timestamp)
        if center_idx is None:
            print(f"   Warning: Couldn't find timestamp {center_timestamp} in raw NQ")
            skipped_count += 1
            continue

        # Extract overlapping windows
        windows = extract_overlapping_windows(
            raw_df,
            center_idx,
            row["expansion_start"],
            row["expansion_end"],
            row["label"],
            window_id,
            stride=52,
            max_overlaps=2,
        )

        expanded_windows.extend(windows)
        matched_count += 1

        if (matched_count + skipped_count) % 50 == 0:
            print(f"   Processed {matched_count + skipped_count}/{len(train_df)} windows...")

    print(f"\n   Matched: {matched_count}, Skipped: {skipped_count}")
    print(
        f"   Generated: {len(expanded_windows)} total windows ({len(expanded_windows) / len(train_df):.2f}x)"
    )

    # Create DataFrame
    print("\n5. Creating expanded dataset...")
    expanded_df = pd.DataFrame(expanded_windows)

    # Convert numpy arrays to lists for parquet compatibility
    expanded_df["features"] = expanded_df["features"].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )

    # Add source and quality columns to match train_latest schema
    expanded_df["source"] = "overlapped"
    expanded_df["quality"] = "auto"

    # Statistics
    print("\n=== Overlap Statistics ===")
    print(f"Original windows: {(expanded_df['offset'] == 0).sum()}")
    print(f"Forward overlaps: {(expanded_df['offset'] > 0).sum()}")
    print(f"Backward overlaps: {(expanded_df['offset'] < 0).sum()}")

    overlap_stats = expanded_df["overlap_fraction"].describe()
    print("\nOverlap fraction distribution:")
    print(f"  Mean: {overlap_stats['mean']:.3f}")
    print(f"  Median: {overlap_stats['50%']:.3f}")
    print(f"  Min: {overlap_stats['min']:.3f}")
    print(f"  Max: {overlap_stats['max']:.3f}")

    # Label distribution
    print("\n=== Label Distribution ===")
    label_counts = expanded_df["label"].value_counts()
    for label, count in label_counts.items():
        pct = count / len(expanded_df) * 100
        print(f"{label}: {count} ({pct:.1f}%)")

    # Save
    print(f"\n6. Saving to: {output_path}")
    expanded_df.to_parquet(output_path, index=False)

    print("\n" + "=" * 80)
    print("✓ OVERLAPPING WINDOWS GENERATED FROM RAW NQ")
    print("=" * 80)
    print(f"\nNext: Test training on {len(expanded_df)} windows")
    print("Expected: +15-20% F1 from temporal augmentation")


if __name__ == "__main__":
    main()
