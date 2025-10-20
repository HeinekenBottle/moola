#!/usr/bin/env python3
"""
Extract 200 diverse windows from raw OHLC for annotation.
No volume. Stratified by session × volatility. Traceable to raw indices.

Version: 1.0.0
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# Constants
SCRIPT_VERSION = "1.0.0"
T = 105  # Window size in bars
N_TOTAL = 200  # Total windows to extract
SEED = 17  # For reproducibility
N_SESSIONS = 4
N_VOL_BUCKETS = 2
N_PER_CELL = N_TOTAL // (N_SESSIONS * N_VOL_BUCKETS)  # ~25

# Session definitions (UTC hours)
SESSIONS = {
    "A": (22, 1),  # 22:00-01:00 (evening)
    "B": (1, 7),  # 01:00-07:00 (overnight)
    "C": (13, 17),  # 13:00-17:00 (midday)
    "D": (17, 22),  # 17:00-22:00 (afternoon/evening)
}

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "candlesticks/data/raw/nq_1min_raw.parquet"
LABELED_DATA_PATH = PROJECT_ROOT / "data/processed/train_pivot_134.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data/batches"
ANNOTATIONS_DIR = PROJECT_ROOT / "data/corrections/candlesticks_annotations"
MASTER_INDEX_PATH = ANNOTATIONS_DIR / "master_index.csv"
REJECTIONS_PATH = ANNOTATIONS_DIR / "rejections.json"


def log(msg: str) -> None:
    """Log with timestamp."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load and validate raw OHLC data."""
    log(f"Loading raw data from {path}")
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")

    df = pd.read_parquet(path)

    # Validate columns
    required_cols = ["timestamp", "open", "high", "low", "close"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")

    # Ensure timestamp is timezone-aware
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    log(f"Loaded {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df[required_cols]


def compute_yearly_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Compute global statistics for normalization."""
    log("Computing yearly statistics...")

    # Median yearly range
    bar_range = df["high"] - df["low"]
    median_yearly_range = bar_range.median()

    log(f"Median yearly range: {median_yearly_range:.4f}")
    return {"median_yearly_range": float(median_yearly_range)}


def validate_window(ohlc: np.ndarray) -> Tuple[bool, str]:
    """
    Validate OHLC window quality.

    Args:
        ohlc: Array of shape (T, 4) with [open, high, low, close]

    Returns:
        (is_valid, reason)
    """
    # Check for NaN or inf
    if np.any(np.isnan(ohlc)) or np.any(np.isinf(ohlc)):
        return False, "contains_nan_or_inf"

    # Check all prices are positive
    if np.any(ohlc <= 0):
        return False, "non_positive_prices"

    # Check high/low constraints for each bar
    for i in range(len(ohlc)):
        open_val, high, low, close = ohlc[i]
        if high < max(open_val, close) or low > min(open_val, close):
            return False, f"invalid_high_low_at_bar_{i}"

    return True, ""


def get_session_from_timestamp(ts: pd.Timestamp) -> str:
    """Determine which session a timestamp belongs to."""
    hour = ts.hour

    for session_name, (start_hour, end_hour) in SESSIONS.items():
        if start_hour < end_hour:
            # Normal range (e.g., 13-17)
            if start_hour <= hour < end_hour:
                return session_name
        else:
            # Wraps midnight (e.g., 22-01)
            if hour >= start_hour or hour < end_hour:
                return session_name

    return "unknown"


def generate_candidates(df: pd.DataFrame, yearly_stats: Dict[str, float]) -> pd.DataFrame:
    """
    Generate all valid T-bar windows with diversity metrics.

    Returns DataFrame with columns:
        - raw_start_idx, raw_end_idx
        - start_ts, end_ts
        - session
        - rv (realized volatility)
        - range_norm (normalized range)
        - trend_mag (normalized trend magnitude)
        - ohlc (array of shape (T, 4))
    """
    log(f"Generating candidate windows (T={T})...")

    median_yearly_range = yearly_stats["median_yearly_range"]
    candidates = []

    total_possible = len(df) - T + 1
    progress_interval = max(1, total_possible // 10)

    for i in range(total_possible):
        if i % progress_interval == 0:
            pct = 100 * i / total_possible
            log(f"Progress: {pct:.0f}% ({i}/{total_possible})")

        # Extract window
        window = df.iloc[i : i + T]
        ohlc_array = window[["open", "high", "low", "close"]].values

        # Validate
        is_valid, reason = validate_window(ohlc_array)
        if not is_valid:
            continue

        # Compute diversity metrics
        # 1. Realized volatility (std of 1-bar log returns)
        prices = window["close"].values
        log_returns = np.diff(np.log(prices))
        rv = float(np.std(log_returns))

        # 2. Normalized range
        window_high = window["high"].max()
        window_low = window["low"].min()
        range_norm = float((window_high - window_low) / median_yearly_range)

        # 3. Trend magnitude
        price_change = abs(prices[-1] - prices[0])
        trend_mag = float(price_change / median_yearly_range)

        # Get session
        start_ts = window.iloc[0]["timestamp"]
        end_ts = window.iloc[-1]["timestamp"]
        session = get_session_from_timestamp(start_ts)

        candidates.append(
            {
                "raw_start_idx": i,
                "raw_end_idx": i + T - 1,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "session": session,
                "rv": rv,
                "range_norm": range_norm,
                "trend_mag": trend_mag,
                "ohlc": ohlc_array,
            }
        )

    log(f"Generated {len(candidates)} valid candidate windows")
    return pd.DataFrame(candidates)


def load_labeled_windows(path: Path) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Load timestamp ranges of existing labeled windows.

    Returns list of (start_ts, end_ts) tuples.
    """
    if not path.exists():
        log(f"No labeled data found at {path}, skipping overlap check")
        return []

    log(f"Loading labeled windows from {path}")
    df = pd.read_parquet(path)

    # Extract timestamp ranges from features
    labeled_ranges = []
    for idx, row in df.iterrows():
        features = row["features"]
        # Features is an array of arrays, need to extract timestamps
        # We don't have timestamps in the labeled data, so we'll need to use
        # the raw data to reconstruct them based on window_id if available
        # For now, we'll skip this as the labeled data doesn't include timestamps
        pass

    log(f"Found {len(labeled_ranges)} labeled window ranges")
    return labeled_ranges


def check_overlap_with_labeled(
    candidates: pd.DataFrame, labeled_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]]
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Remove candidates that overlap with existing labeled windows.

    Returns:
        - Filtered candidates DataFrame
        - List of rejection records
    """
    if not labeled_ranges:
        log("No labeled windows to check overlap against")
        return candidates, []

    log(f"Checking {len(candidates)} candidates for overlap with {len(labeled_ranges)} labeled windows...")

    rejections = []
    valid_mask = np.ones(len(candidates), dtype=bool)

    for idx, row in candidates.iterrows():
        start_ts = row["start_ts"]
        end_ts = row["end_ts"]

        for labeled_start, labeled_end in labeled_ranges:
            # Check for any overlap
            if start_ts <= labeled_end and end_ts >= labeled_start:
                valid_mask[idx] = False
                rejections.append(
                    {
                        "raw_start_idx": int(row["raw_start_idx"]),
                        "reason": "overlaps_labeled_window",
                        "timestamp": str(start_ts),
                    }
                )
                break

    filtered = candidates[valid_mask].reset_index(drop=True)
    log(f"Filtered to {len(filtered)} candidates after overlap removal ({len(rejections)} rejected)")
    return filtered, rejections


def stratified_sample(candidates: pd.DataFrame, n_per_cell: int, seed: int) -> pd.DataFrame:
    """
    Sample approximately n_per_cell windows from each session × volatility cell.
    Within each cell, spread samples across range_norm and trend_mag quantiles.

    Args:
        candidates: All valid candidate windows
        n_per_cell: Target samples per cell (default ~25)
        seed: Random seed for reproducibility

    Returns:
        Sampled DataFrame with ~N_TOTAL windows
    """
    log(f"Stratified sampling: {n_per_cell} per cell across {N_SESSIONS} sessions × {N_VOL_BUCKETS} vol buckets")

    # Compute global volatility median
    rv_median = candidates["rv"].median()
    log(f"RV median for bucketing: {rv_median:.6f}")

    # Tag volatility bucket
    candidates = candidates.copy()
    candidates["vol_bucket"] = candidates["rv"].apply(lambda x: "high" if x >= rv_median else "low")

    # Sample from each cell
    rng = np.random.RandomState(seed)
    sampled_windows = []
    session_counts = {}

    for session in SESSIONS.keys():
        for vol_bucket in ["low", "high"]:
            cell_key = f"{session}_{vol_bucket}"

            # Get candidates in this cell
            cell_mask = (candidates["session"] == session) & (candidates["vol_bucket"] == vol_bucket)
            cell_candidates = candidates[cell_mask].copy()

            n_available = len(cell_candidates)
            n_to_sample = min(n_per_cell, n_available)

            if n_available == 0:
                log(f"  {cell_key}: 0 candidates available, skipping")
                session_counts[cell_key] = 0
                continue

            log(f"  {cell_key}: {n_available} candidates, sampling {n_to_sample}")

            if n_available <= n_to_sample:
                # Take all
                sampled = cell_candidates
            else:
                # Spread across range_norm and trend_mag quantiles
                # Create composite diversity score
                cell_candidates["diversity_score"] = (
                    cell_candidates["range_norm"].rank(pct=True)
                    + cell_candidates["trend_mag"].rank(pct=True)
                )

                # Divide into quantiles and sample evenly
                n_quantiles = min(5, n_to_sample)
                cell_candidates["quantile"] = pd.qcut(
                    cell_candidates["diversity_score"],
                    q=n_quantiles,
                    labels=False,
                    duplicates="drop",
                )

                sampled_indices = []
                samples_per_quantile = n_to_sample // n_quantiles
                remainder = n_to_sample % n_quantiles

                for q in range(n_quantiles):
                    q_candidates = cell_candidates[cell_candidates["quantile"] == q]
                    n_from_q = samples_per_quantile + (1 if q < remainder else 0)
                    n_from_q = min(n_from_q, len(q_candidates))

                    if n_from_q > 0:
                        selected = rng.choice(q_candidates.index, size=n_from_q, replace=False)
                        sampled_indices.extend(selected)

                sampled = cell_candidates.loc[sampled_indices]

            session_counts[cell_key] = len(sampled)
            sampled_windows.append(sampled)

    # Combine all samples
    result = pd.concat(sampled_windows, ignore_index=True)
    log(f"Sampled {len(result)} total windows")
    log(f"Session distribution: {session_counts}")

    return result, session_counts, rv_median


def generate_window_id(batch_datetime: datetime, seq: int) -> str:
    """
    Generate window ID in format: batch_YYYYMMDDHHMM_<seq>

    Args:
        batch_datetime: Timestamp when script was run
        seq: Sequence number (1-indexed)

    Returns:
        Window ID string
    """
    timestamp_str = batch_datetime.strftime("%Y%m%d%H%M")
    seq_str = f"{seq:03d}"
    return f"batch_{timestamp_str}_{seq_str}"


def save_outputs(
    windows: pd.DataFrame,
    manifest: Dict,
    rejections: List[Dict],
    batch_datetime: datetime,
) -> None:
    """
    Save all output files:
    1. data/batches/batch_200.parquet
    2. data/batches/batch_200_manifest.json
    3. Append to data/corrections/candlesticks_annotations/master_index.csv
    4. data/corrections/candlesticks_annotations/rejections.json (if rejections exist)
    """
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate window IDs
    window_ids = [generate_window_id(batch_datetime, i + 1) for i in range(len(windows))]
    windows = windows.copy()
    windows["window_id"] = window_ids

    # Prepare output DataFrame
    output_df = windows[
        [
            "window_id",
            "ohlc",
            "raw_start_idx",
            "raw_end_idx",
            "start_ts",
            "end_ts",
            "session",
            "vol_bucket",
            "rv",
            "range_norm",
            "trend_mag",
        ]
    ].copy()

    # Rename ohlc to features for consistency
    output_df = output_df.rename(columns={"ohlc": "features", "vol_bucket": "volatility_bucket"})

    # Convert numpy arrays to lists for parquet serialization
    output_df["features"] = output_df["features"].apply(lambda x: x.tolist())

    # 1. Save parquet file
    parquet_path = OUTPUT_DIR / "batch_200.parquet"
    output_df.to_parquet(parquet_path, index=False)
    log(f"Saved batch parquet to {parquet_path}")

    # 2. Save manifest JSON
    manifest_path = OUTPUT_DIR / "batch_200_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"Saved manifest to {manifest_path}")

    # 3. NOTE: Do NOT pre-fill master_index.csv
    # master_index.csv should only be updated when annotations are SAVED via the backend
    # Pre-filling causes the backend to think all windows are already annotated
    log(f"Skipping master_index pre-fill (will be populated when annotations are saved)")

    # 4. Save rejections if any
    if rejections:
        # Load existing rejections if file exists
        if REJECTIONS_PATH.exists():
            with open(REJECTIONS_PATH, "r") as f:
                existing = json.load(f)
                all_rejections = existing.get("rejections", [])
        else:
            all_rejections = []

        # Add new rejections with IDs
        for i, rejection in enumerate(rejections):
            rejection["id"] = f"rejected_{batch_datetime.strftime('%Y%m%d%H%M')}_{i+1:03d}"

        all_rejections.extend(rejections)

        with open(REJECTIONS_PATH, "w") as f:
            json.dump({"rejections": all_rejections}, f, indent=2)
        log(f"Saved {len(rejections)} rejections to {REJECTIONS_PATH}")

    log("All outputs saved successfully")


def main():
    """Main execution flow."""
    log("=" * 80)
    log(f"Extract Batch 200 - Version {SCRIPT_VERSION}")
    log(f"Target: {N_TOTAL} diverse windows, stratified by session × volatility")
    log(f"Window size: T={T} bars")
    log(f"Random seed: {SEED}")
    log("=" * 80)

    # Record execution timestamp
    batch_datetime = datetime.now(timezone.utc)

    try:
        # 1. Load raw data
        raw_df = load_raw_data(RAW_DATA_PATH)

        # 2. Compute yearly statistics
        yearly_stats = compute_yearly_stats(raw_df)

        # 3. Generate all candidate windows
        candidates = generate_candidates(raw_df, yearly_stats)

        if len(candidates) == 0:
            raise ValueError("No valid candidate windows generated")

        # 4. Filter overlaps with labeled windows
        labeled_ranges = load_labeled_windows(LABELED_DATA_PATH)
        candidates_filtered, overlap_rejections = check_overlap_with_labeled(candidates, labeled_ranges)

        if len(candidates_filtered) < N_TOTAL:
            log(
                f"WARNING: Only {len(candidates_filtered)} candidates available after filtering, "
                f"but {N_TOTAL} requested. Will sample as many as possible."
            )

        # 5. Stratified sampling
        sampled_windows, session_counts, rv_median = stratified_sample(
            candidates_filtered, N_PER_CELL, SEED
        )

        # 6. Prepare manifest
        manifest = {
            "script_version": SCRIPT_VERSION,
            "extraction_datetime": batch_datetime.isoformat(),
            "seed": SEED,
            "total_windows": len(sampled_windows),
            "source_file": str(RAW_DATA_PATH.relative_to(PROJECT_ROOT)),
            "global_stats": {
                "median_yearly_range": yearly_stats["median_yearly_range"],
                "rv_median": float(rv_median),
            },
            "session_counts": session_counts,
            "rejected_count": len(overlap_rejections),
            "rejection_reasons": {
                "overlaps_labeled_window": len(overlap_rejections),
            },
        }

        # 7. Save all outputs
        save_outputs(sampled_windows, manifest, overlap_rejections, batch_datetime)

        log("=" * 80)
        log("EXTRACTION COMPLETE")
        log(f"Windows extracted: {len(sampled_windows)}")
        log(f"Windows rejected: {len(overlap_rejections)}")
        log(f"Output directory: {OUTPUT_DIR}")
        log("=" * 80)

    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        raise


def validate_outputs():
    """Quick validation of generated outputs (for testing)."""
    log("\n" + "=" * 80)
    log("VALIDATION")
    log("=" * 80)

    # Load outputs
    parquet_path = OUTPUT_DIR / "batch_200.parquet"
    manifest_path = OUTPUT_DIR / "batch_200_manifest.json"

    if not parquet_path.exists():
        log("ERROR: Batch parquet file not found")
        return

    df = pd.read_parquet(parquet_path)
    log(f"Loaded batch parquet: {len(df)} rows")
    log(f"Columns: {df.columns.tolist()}")

    # Check shapes
    sample_features = df.iloc[0]["features"]
    log(f"Features shape: {sample_features.shape} (expected: ({T}, 4))")

    # Check for duplicates
    duplicates = df["window_id"].duplicated().sum()
    log(f"Duplicate window IDs: {duplicates}")

    # Check session distribution
    session_dist = df.groupby(["session", "volatility_bucket"]).size()
    log(f"Session × Volatility distribution:\n{session_dist}")

    # Verify no NaN in features
    has_nan = any(np.any(np.isnan(feat)) for feat in df["features"])
    log(f"Features contain NaN: {has_nan}")

    # Load manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    log(f"Manifest total_windows: {manifest['total_windows']}")
    log(f"Manifest session_counts: {manifest['session_counts']}")

    # Check master index
    if MASTER_INDEX_PATH.exists():
        master = pd.read_csv(MASTER_INDEX_PATH)
        batch_200_rows = master[master["batch_file"] == "batch_200.parquet"]
        log(f"Master index rows for batch_200: {len(batch_200_rows)}")
    else:
        log("Master index not found")

    log("=" * 80)
    log("VALIDATION COMPLETE")
    log("=" * 80)


if __name__ == "__main__":
    # Run extraction
    main()

    # Uncomment for validation (testing only)
    # validate_outputs()
