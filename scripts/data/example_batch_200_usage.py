#!/usr/bin/env python3
"""
Example usage of batch_200 extraction outputs.
Demonstrates loading, inspecting, and analyzing extracted windows.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent


def example_load_and_inspect():
    """Load batch and print basic info."""
    print("=" * 80)
    print("Example 1: Load and Inspect Batch")
    print("=" * 80)

    # Load batch parquet
    df = pd.read_parquet(PROJECT_ROOT / "data/batches/batch_200.parquet")

    print(f"\nTotal windows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst window:")
    print(f"  ID: {df.iloc[0]['window_id']}")
    print(f"  Session: {df.iloc[0]['session']}")
    print(f"  Volatility: {df.iloc[0]['volatility_bucket']}")
    print(f"  Time range: {df.iloc[0]['start_ts']} to {df.iloc[0]['end_ts']}")
    print(f"  Features shape: {df.iloc[0]['features'].shape}")

    # Load manifest
    with open(PROJECT_ROOT / "data/batches/batch_200_manifest.json") as f:
        manifest = json.load(f)

    print(f"\nManifest info:")
    print(f"  Script version: {manifest['script_version']}")
    print(f"  Extraction time: {manifest['extraction_datetime']}")
    print(f"  Seed: {manifest['seed']}")
    print(f"  Total windows: {manifest['total_windows']}")
    print(f"  Rejections: {manifest['rejected_count']}")


def example_session_distribution():
    """Show distribution across sessions and volatility."""
    print("\n" + "=" * 80)
    print("Example 2: Session and Volatility Distribution")
    print("=" * 80)

    df = pd.read_parquet(PROJECT_ROOT / "data/batches/batch_200.parquet")

    # Session × Volatility distribution
    dist = df.groupby(["session", "volatility_bucket"]).size()
    print("\nWindows per cell:")
    print(dist)

    # Diversity metrics summary
    print("\nDiversity metrics summary:")
    print(df[["rv", "range_norm", "trend_mag"]].describe())


def example_access_ohlc_data():
    """Access and display OHLC data for a window."""
    print("\n" + "=" * 80)
    print("Example 3: Access OHLC Data")
    print("=" * 80)

    df = pd.read_parquet(PROJECT_ROOT / "data/batches/batch_200.parquet")

    # Get first window
    window = df.iloc[0]
    ohlc = window["features"]  # shape: (105, 4)

    print(f"\nWindow: {window['window_id']}")
    print(f"Session: {window['session']}, Volatility: {window['volatility_bucket']}")
    print(f"\nOHLC data (first 5 bars):")
    print("       Open      High       Low     Close")
    for i in range(5):
        o, h, l, c = ohlc[i]
        print(f"Bar {i:2d}: {o:8.2f} {h:8.2f} {l:8.2f} {c:8.2f}")

    print(f"\nOHLC data (last 5 bars):")
    print("       Open      High       Low     Close")
    for i in range(100, 105):
        o, h, l, c = ohlc[i]
        print(f"Bar {i:2d}: {o:8.2f} {h:8.2f} {l:8.2f} {c:8.2f}")

    # Compute simple stats
    close_prices = ohlc[:, 3]
    print(f"\nClose price stats:")
    print(f"  Min: {close_prices.min():.2f}")
    print(f"  Max: {close_prices.max():.2f}")
    print(f"  Mean: {close_prices.mean():.2f}")
    print(f"  Range: {close_prices.max() - close_prices.min():.2f}")


def example_trace_to_raw_data():
    """Trace a window back to original raw data."""
    print("\n" + "=" * 80)
    print("Example 4: Trace Window to Raw Data")
    print("=" * 80)

    # Load batch and raw data
    batch_df = pd.read_parquet(PROJECT_ROOT / "data/batches/batch_200.parquet")
    raw_df = pd.read_parquet(PROJECT_ROOT / "candlesticks/data/raw/nq_1min_raw.parquet")

    # Get a window
    window = batch_df.iloc[10]  # Arbitrary window
    print(f"\nWindow: {window['window_id']}")

    # Extract from raw using indices
    start_idx = window["raw_start_idx"]
    end_idx = window["raw_end_idx"]
    original_bars = raw_df.iloc[start_idx : end_idx + 1]

    print(f"\nRaw data indices: {start_idx} to {end_idx}")
    print(f"Number of bars: {len(original_bars)}")

    # Verify timestamps
    print(f"\nTimestamp verification:")
    print(f"  Stored start_ts: {window['start_ts']}")
    print(f"  Raw start_ts:    {original_bars.iloc[0]['timestamp']}")
    print(f"  Match: {window['start_ts'] == original_bars.iloc[0]['timestamp']}")

    print(f"\n  Stored end_ts:   {window['end_ts']}")
    print(f"  Raw end_ts:      {original_bars.iloc[-1]['timestamp']}")
    print(f"  Match: {window['end_ts'] == original_bars.iloc[-1]['timestamp']}")

    # Verify OHLC values
    original_ohlc = original_bars[["open", "high", "low", "close"]].values
    stored_ohlc = window["features"]
    match = np.allclose(original_ohlc, stored_ohlc)

    print(f"\nOHLC values match: {match}")


def example_filter_by_criteria():
    """Filter windows by session, volatility, or diversity metrics."""
    print("\n" + "=" * 80)
    print("Example 5: Filter Windows by Criteria")
    print("=" * 80)

    df = pd.read_parquet(PROJECT_ROOT / "data/batches/batch_200.parquet")

    # Filter: Session A, high volatility
    filtered = df[(df["session"] == "A") & (df["volatility_bucket"] == "high")]
    print(f"\nSession A, high volatility: {len(filtered)} windows")
    if len(filtered) > 0:
        print(f"  Example: {filtered.iloc[0]['window_id']}")

    # Filter: High trend magnitude (top 20%)
    trend_threshold = df["trend_mag"].quantile(0.8)
    high_trend = df[df["trend_mag"] >= trend_threshold]
    print(f"\nHigh trend magnitude (top 20%): {len(high_trend)} windows")
    print(f"  Threshold: {trend_threshold:.4f}")
    if len(high_trend) > 0:
        print(f"  Example: {high_trend.iloc[0]['window_id']}")

    # Filter: Wide range (top 20%)
    range_threshold = df["range_norm"].quantile(0.8)
    wide_range = df[df["range_norm"] >= range_threshold]
    print(f"\nWide range (top 20%): {len(wide_range)} windows")
    print(f"  Threshold: {range_threshold:.4f}")
    if len(wide_range) > 0:
        print(f"  Example: {wide_range.iloc[0]['window_id']}")

    # Filter: Multiple criteria
    complex_filter = df[
        (df["session"].isin(["A", "D"]))  # Evening sessions
        & (df["volatility_bucket"] == "high")  # High vol
        & (df["rv"] > df["rv"].median())  # Above median RV
    ]
    print(f"\nComplex filter (sessions A/D, high vol, above median RV): {len(complex_filter)} windows")


def example_annotation_workflow():
    """Demonstrate annotation workflow with master index."""
    print("\n" + "=" * 80)
    print("Example 6: Annotation Workflow")
    print("=" * 80)

    # Load batch
    batch_df = pd.read_parquet(PROJECT_ROOT / "data/batches/batch_200.parquet")

    # Load master index
    master_index = pd.read_csv(
        PROJECT_ROOT / "data/corrections/candlesticks_annotations/master_index.csv"
    )

    # Filter to batch_200
    batch_200_index = master_index[master_index["batch_file"] == "batch_200.parquet"]

    print(f"\nBatch windows in master index: {len(batch_200_index)}")
    print(f"Total windows in master index: {len(master_index)}")

    # Show annotation status
    print(f"\nAnnotation status for batch_200:")
    print(f"  Quality grades assigned: {batch_200_index['quality_grade'].notna().sum()}")
    print(f"  Pending annotation: {batch_200_index['quality_grade'].isna().sum()}")

    # Example: simulate adding annotation
    print(f"\nExample annotation workflow:")
    example_window_id = batch_df.iloc[0]["window_id"]
    print(f"  1. Load window: {example_window_id}")
    print(f"  2. Review OHLC chart manually")
    print(f"  3. Assign label (e.g., 'consolidation', 'breakout')")
    print(f"  4. Update quality_grade in master_index.csv")
    print(f"  5. Update expansion_count if needed")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("BATCH_200 USAGE EXAMPLES")
    print("=" * 80)

    try:
        example_load_and_inspect()
        example_session_distribution()
        example_access_ohlc_data()
        example_trace_to_raw_data()
        example_filter_by_criteria()
        example_annotation_workflow()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80 + "\n")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Make sure you've run extract_batch_200.py first!")
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
