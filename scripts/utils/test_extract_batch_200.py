#!/usr/bin/env python3
"""
Quick test script for extract_batch_200.py
Run this AFTER running extract_batch_200.py to validate outputs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent


def test_batch_parquet():
    """Test the batch parquet file."""
    print("=" * 80)
    print("Testing batch_200.parquet")
    print("=" * 80)

    path = PROJECT_ROOT / "data/batches/batch_200.parquet"
    if not path.exists():
        print(f"ERROR: File not found at {path}")
        return False

    df = pd.read_parquet(path)
    print(f"✓ Loaded {len(df)} rows")

    # Expected columns
    expected_cols = [
        "window_id",
        "features",
        "raw_start_idx",
        "raw_end_idx",
        "start_ts",
        "end_ts",
        "session",
        "volatility_bucket",
        "rv",
        "range_norm",
        "trend_mag",
    ]

    for col in expected_cols:
        if col not in df.columns:
            print(f"✗ Missing column: {col}")
            return False
    print(f"✓ All expected columns present: {len(expected_cols)}")

    # Check row count
    if len(df) != 200:
        print(f"✗ Expected 200 rows, got {len(df)}")
        return False
    print("✓ Correct number of rows (200)")

    # Check features shape
    sample_features = df.iloc[0]["features"]
    if sample_features.shape != (105, 4):
        print(f"✗ Expected features shape (105, 4), got {sample_features.shape}")
        return False
    print("✓ Features have correct shape (105, 4)")

    # Check for NaN in features
    has_nan = any(np.any(np.isnan(feat)) for feat in df["features"])
    if has_nan:
        print("✗ Features contain NaN values")
        return False
    print("✓ No NaN values in features")

    # Check for duplicate window IDs
    n_unique = df["window_id"].nunique()
    if n_unique != 200:
        print(f"✗ Expected 200 unique IDs, got {n_unique}")
        return False
    print("✓ All window IDs are unique")

    # Check session distribution
    print("\nSession × Volatility distribution:")
    dist = df.groupby(["session", "volatility_bucket"]).size()
    print(dist)

    # Check all sessions present
    sessions = df["session"].unique()
    expected_sessions = {"A", "B", "C", "D"}
    if set(sessions) != expected_sessions:
        print(f"✗ Expected sessions {expected_sessions}, got {set(sessions)}")
        return False
    print(f"✓ All sessions present: {sorted(sessions)}")

    # Check volatility buckets
    vol_buckets = df["volatility_bucket"].unique()
    expected_vol = {"low", "high"}
    if set(vol_buckets) != expected_vol:
        print(f"✗ Expected vol buckets {expected_vol}, got {set(vol_buckets)}")
        return False
    print(f"✓ Both volatility buckets present: {sorted(vol_buckets)}")

    print("\n✓ batch_200.parquet PASSED all tests")
    return True


def test_manifest():
    """Test the manifest JSON file."""
    print("\n" + "=" * 80)
    print("Testing batch_200_manifest.json")
    print("=" * 80)

    import json

    path = PROJECT_ROOT / "data/batches/batch_200_manifest.json"
    if not path.exists():
        print(f"ERROR: File not found at {path}")
        return False

    with open(path) as f:
        manifest = json.load(f)

    print(f"✓ Loaded manifest")

    # Check required keys
    required_keys = [
        "script_version",
        "extraction_datetime",
        "seed",
        "total_windows",
        "source_file",
        "global_stats",
        "session_counts",
        "rejected_count",
        "rejection_reasons",
    ]

    for key in required_keys:
        if key not in manifest:
            print(f"✗ Missing key: {key}")
            return False
    print(f"✓ All required keys present: {len(required_keys)}")

    # Check values
    if manifest["script_version"] != "1.0.0":
        print(f"✗ Unexpected version: {manifest['script_version']}")
        return False
    print(f"✓ Script version: {manifest['script_version']}")

    if manifest["seed"] != 17:
        print(f"✗ Expected seed 17, got {manifest['seed']}")
        return False
    print(f"✓ Seed: {manifest['seed']}")

    if manifest["total_windows"] != 200:
        print(f"✗ Expected 200 windows, got {manifest['total_windows']}")
        return False
    print(f"✓ Total windows: {manifest['total_windows']}")

    print(f"\nGlobal stats:")
    for key, val in manifest["global_stats"].items():
        print(f"  {key}: {val}")

    print(f"\nSession counts:")
    for key, val in manifest["session_counts"].items():
        print(f"  {key}: {val}")

    print(f"\nRejections: {manifest['rejected_count']}")

    print("\n✓ batch_200_manifest.json PASSED all tests")
    return True


def test_master_index():
    """Test the master index update."""
    print("\n" + "=" * 80)
    print("Testing master_index.csv update")
    print("=" * 80)

    path = PROJECT_ROOT / "data/corrections/candlesticks_annotations/master_index.csv"
    if not path.exists():
        print(f"ERROR: File not found at {path}")
        return False

    df = pd.read_csv(path)
    print(f"✓ Loaded master index with {len(df)} rows")

    # Check for batch_200.parquet entries
    batch_200_rows = df[df["batch_file"] == "batch_200.parquet"]
    if len(batch_200_rows) != 200:
        print(f"✗ Expected 200 batch_200.parquet rows, got {len(batch_200_rows)}")
        return False
    print(f"✓ Found 200 rows for batch_200.parquet")

    # Check columns
    expected_cols = ["window_id", "batch_file", "annotation_date", "quality_grade", "expansion_count"]
    for col in expected_cols:
        if col not in df.columns:
            print(f"✗ Missing column: {col}")
            return False
    print(f"✓ All expected columns present")

    # Check window IDs are unique globally
    n_unique = df["window_id"].nunique()
    if n_unique != len(df):
        print(f"✗ Duplicate window IDs found ({n_unique} unique out of {len(df)} total)")
        return False
    print(f"✓ All window IDs are unique across all batches")

    print("\n✓ master_index.csv PASSED all tests")
    return True


def test_traceability():
    """Test traceability back to raw data."""
    print("\n" + "=" * 80)
    print("Testing traceability to raw data")
    print("=" * 80)

    batch_df = pd.read_parquet(PROJECT_ROOT / "data/batches/batch_200.parquet")
    raw_df = pd.read_parquet(PROJECT_ROOT / "candlesticks/data/raw/nq_1min_raw.parquet")

    print(f"✓ Loaded batch and raw data")

    # Test first window
    window = batch_df.iloc[0]
    start_idx = window["raw_start_idx"]
    end_idx = window["raw_end_idx"]

    # Extract from raw
    original_bars = raw_df.iloc[start_idx : end_idx + 1]

    if len(original_bars) != 105:
        print(f"✗ Expected 105 bars, got {len(original_bars)}")
        return False
    print(f"✓ Correct number of bars traced (105)")

    # Check timestamps match
    if original_bars.iloc[0]["timestamp"] != window["start_ts"]:
        print("✗ Start timestamp mismatch")
        return False
    if original_bars.iloc[-1]["timestamp"] != window["end_ts"]:
        print("✗ End timestamp mismatch")
        return False
    print("✓ Timestamps match raw data")

    # Check OHLC values match
    original_ohlc = original_bars[["open", "high", "low", "close"]].values
    stored_ohlc = window["features"]

    if not np.allclose(original_ohlc, stored_ohlc):
        print("✗ OHLC values don't match raw data")
        return False
    print("✓ OHLC values match raw data")

    print(f"\nExample window: {window['window_id']}")
    print(f"  Raw indices: {start_idx} to {end_idx}")
    print(f"  Time range: {window['start_ts']} to {window['end_ts']}")
    print(f"  Session: {window['session']}, Vol: {window['volatility_bucket']}")

    print("\n✓ Traceability PASSED all tests")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("EXTRACT_BATCH_200 OUTPUT VALIDATION")
    print("=" * 80 + "\n")

    results = []
    results.append(("Batch Parquet", test_batch_parquet()))
    results.append(("Manifest JSON", test_manifest()))
    results.append(("Master Index", test_master_index()))
    results.append(("Traceability", test_traceability()))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")

    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n✓ ALL TESTS PASSED - Batch extraction successful!")
    else:
        print("\n✗ SOME TESTS FAILED - Review output above")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
