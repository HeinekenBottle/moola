"""Generate overlapping windows from base labeled dataset.

Doubles effective training data: 174 base → ~350 windows via stride=52 (50% overlap).
Each overlap shares ~53 bars, inherits prorated labels for partial expansions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))


import pandas as pd


def prorate_expansion_labels(
    expansion_start: int, expansion_end: int, window_offset: int, window_length: int = 105
) -> tuple[int, int, float]:
    """Prorate expansion pointers for overlapping window.

    Args:
        expansion_start: Original expansion start in base window
        expansion_end: Original expansion end in base window
        window_offset: How many bars this window is shifted from base
        window_length: Window size (default: 105)

    Returns:
        new_start: Adjusted start (clipped to [0, 104])
        new_end: Adjusted end (clipped to [0, 104])
        overlap_fraction: What fraction of expansion is visible (0-1)
    """
    # Shift pointers by window offset
    new_start = expansion_start - window_offset
    new_end = expansion_end - window_offset

    # Calculate overlap before clipping
    original_length = expansion_end - expansion_start + 1

    # Clip to window bounds
    new_start_clipped = max(0, min(window_length - 1, new_start))
    new_end_clipped = max(0, min(window_length - 1, new_end))

    # Calculate visible length after clipping
    if new_end < 0 or new_start >= window_length:
        # Expansion completely outside window
        visible_length = 0
    else:
        visible_length = new_end_clipped - new_start_clipped + 1

    # Overlap fraction (used for weighting in loss)
    overlap_fraction = visible_length / original_length if original_length > 0 else 0.0

    return new_start_clipped, new_end_clipped, overlap_fraction


def generate_overlapping_windows(
    base_df: pd.DataFrame, stride: int = 52, min_overlap_fraction: float = 0.3
) -> pd.DataFrame:
    """Generate overlapping windows from base dataset.

    Args:
        base_df: Base labeled windows (174 samples)
        stride: Window stride (52 = 50% overlap with 105-bar windows)
        min_overlap_fraction: Minimum expansion overlap to keep window

    Returns:
        Expanded dataset with overlapping windows
    """
    expanded_rows = []

    for idx, row in base_df.iterrows():
        features = row["features"]  # List of 105 OHLC arrays
        expansion_start = row["expansion_start"]
        expansion_end = row["expansion_end"]
        label = row["label"]
        window_id = row["window_id"]

        # Always include original window (offset=0)
        expanded_rows.append(
            {
                "window_id": f"{window_id}_offset0",
                "base_window_id": window_id,
                "offset": 0,
                "features": features,
                "label": label,
                "expansion_start": expansion_start,
                "expansion_end": expansion_end,
                "overlap_fraction": 1.0,
            }
        )

        # Generate forward overlaps (stride forward)
        offset = stride
        while offset < len(features):
            new_start, new_end, overlap_frac = prorate_expansion_labels(
                expansion_start, expansion_end, offset
            )

            if overlap_frac >= min_overlap_fraction:
                # Extract overlapping window features
                overlap_features = features[offset : offset + 105]

                # Pad if needed (edge case: near end of original)
                if len(overlap_features) < 105:
                    # Skip incomplete windows
                    break

                expanded_rows.append(
                    {
                        "window_id": f"{window_id}_offset{offset}",
                        "base_window_id": window_id,
                        "offset": offset,
                        "features": overlap_features,
                        "label": label,
                        "expansion_start": new_start,
                        "expansion_end": new_end,
                        "overlap_fraction": overlap_frac,
                    }
                )

            offset += stride

        # Generate backward overlaps (stride backward)
        offset = -stride
        while abs(offset) < len(features):
            new_start, new_end, overlap_frac = prorate_expansion_labels(
                expansion_start, expansion_end, offset
            )

            if overlap_frac >= min_overlap_fraction:
                # Extract overlapping window features
                start_idx = max(0, offset)
                overlap_features = features[start_idx : start_idx + 105]

                # Pad if needed (edge case: near start of original)
                if len(overlap_features) < 105:
                    # Skip incomplete windows
                    break

                expanded_rows.append(
                    {
                        "window_id": f"{window_id}_offset{offset}",
                        "base_window_id": window_id,
                        "offset": offset,
                        "features": overlap_features,
                        "label": label,
                        "expansion_start": new_start,
                        "expansion_end": new_end,
                        "overlap_fraction": overlap_frac,
                    }
                )

            offset -= stride

    return pd.DataFrame(expanded_rows)


def main():
    print("=" * 80)
    print("OVERLAPPING WINDOW GENERATION")
    print("=" * 80)

    # Load base dataset
    input_path = "data/processed/labeled/train_latest.parquet"
    output_path = "data/processed/labeled/train_latest_overlaps.parquet"

    print(f"\nLoading base dataset: {input_path}")
    base_df = pd.read_parquet(input_path)
    print(f"Base windows: {len(base_df)}")

    # Generate overlapping windows
    print("\nGenerating overlaps (stride=52, min_overlap=0.3)...")
    expanded_df = generate_overlapping_windows(base_df, stride=52, min_overlap_fraction=0.3)

    print(f"Expanded windows: {len(expanded_df)} ({len(expanded_df) / len(base_df):.2f}x)")

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
    print(f"\nSaving to: {output_path}")
    expanded_df.to_parquet(output_path, index=False)

    print("\n" + "=" * 80)
    print("✓ OVERLAPPING WINDOWS GENERATED")
    print("=" * 80)
    print(f"\nNext: Test training on {len(expanded_df)} windows")
    print("Expected: +15-20% F1 from temporal augmentation")


if __name__ == "__main__":
    main()
