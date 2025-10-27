#!/usr/bin/env python3
"""
Replace September 2025 data with fresh download to ensure quality.
"""


import databento as db
import pandas as pd


def fix_september_2025():
    """Replace September 2025 with fresh data."""

    print("🔧 FIXING SEPTEMBER 2025 DATA")
    print("=" * 50)

    # Load current dataset
    current_path = "data/raw/nq_ohlcv_1min_2020-09_2025-09_continuous.parquet"
    df = pd.read_parquet(current_path)

    print(f"Current dataset: {len(df):,} records")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Load fresh September 2025 data
    sep_2025_path = "/Users/jack/projects/moola/data/raw/nq_sep_2025_complete.dbn.zst"
    print(f"\nLoading fresh September 2025 from: {sep_2025_path}")

    store = db.DBNStore.from_file(sep_2025_path)
    sep_2025_fresh = store.to_df()

    # Convert timezone to match existing data (make timezone-naive)
    if sep_2025_fresh.index.tz is not None:
        sep_2025_fresh.index = sep_2025_fresh.index.tz_localize(None)

    print(f"Fresh September 2025: {len(sep_2025_fresh):,} records")
    print(f"Date range: {sep_2025_fresh.index.min()} to {sep_2025_fresh.index.max()}")

    # Remove old September 2025 and add fresh data
    print("\nRemoving old September 2025...")
    df_without_sep = df[df.index < "2025-09-01"]  # Keep data before September 2025

    print(f"Records before Sep 2025: {len(df_without_sep):,}")

    # Merge with fresh September 2025
    print("Adding fresh September 2025...")
    df_fixed = pd.concat([df_without_sep, sep_2025_fresh], ignore_index=False)
    df_fixed = df_fixed.sort_index()

    # Remove any duplicates (shouldn't be any, but just in case)
    before_dedup = len(df_fixed)
    df_fixed = df_fixed[~df_fixed.index.duplicated(keep="first")]
    after_dedup = len(df_fixed)

    if before_dedup != after_dedup:
        print(f"Removed {before_dedup - after_dedup} duplicate timestamps")

    # Save fixed dataset
    output_path = "data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet"
    df_fixed.to_parquet(output_path, index=True, engine="pyarrow")

    print(f"\n✅ Fixed dataset saved to: {output_path}")
    print(f"Final record count: {len(df_fixed):,}")
    print(f"Final date range: {df_fixed.index.min()} to {df_fixed.index.max()}")

    # Validate September 2025 in fixed dataset
    sep_2025_fixed = df_fixed.loc["2025-09-01":"2025-09-30"]
    print("\n📊 September 2025 validation:")
    print(f"  Records: {len(sep_2025_fixed):,}")
    print(f"  Days with data: {(sep_2025_fixed.resample('D').size() > 0).sum()}/30")

    # Quick quality check
    extreme_moves = sep_2025_fixed["close"].pct_change().abs() > 0.05
    inconsistent = (
        (sep_2025_fixed["high"] < sep_2025_fixed["low"])
        | (sep_2025_fixed["high"] < sep_2025_fixed["open"])
        | (sep_2025_fixed["high"] < sep_2025_fixed["close"])
    )

    print(f"  Extreme price moves: {extreme_moves.sum()}")
    print(f"  Inconsistent candles: {inconsistent.sum()}")

    print("\n🎯 September 2025 data replaced and validated!")

    return output_path


if __name__ == "__main__":
    fix_september_2025()
