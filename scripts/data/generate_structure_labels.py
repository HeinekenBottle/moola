"""Automatic Structure Label Generation for Unlabeled OHLC Data.

This script generates three types of structural labels for unlabeled time-series data:

1. **Expansion Detection** (binary per bar):
   - Uses Adaptive ATR-based epsilon
   - Labels bars as in_expansion (1) or not (0)
   - Algorithm: Check if retracement < epsilon

2. **Swing Point Classification** (3-class per bar):
   - 0 = neither
   - 1 = swing high (local maximum)
   - 2 = swing low (local minimum)
   - Uses configurable window (default 5-bar: 2 before, 2 after)

3. **Candlestick Pattern** (4-class per bar):
   - 0 = bullish (close > open)
   - 1 = bearish (close < open)
   - 2 = neutral (small body)
   - 3 = doji (open ≈ close)

Features:
- Vectorized NumPy operations (handles 118k samples in <5 min)
- Deterministic (no randomness)
- Comprehensive logging with loguru
- CLI interface with argparse
- Output to parquet format with schema validation
- Integrates with existing data infrastructure

Usage:
    python3 scripts/generate_structure_labels.py \\
        --input data/raw/unlabeled_windows.parquet \\
        --output data/processed/unlabeled_with_labels.parquet \\
        --epsilon-factor 2.0 \\
        --swing-window 5 \\
        --doji-threshold 0.1

Example:
    >>> from generate_structure_labels import generate_all_labels
    >>> from pathlib import Path
    >>> generate_all_labels(
    ...     data_path=Path("data/raw/unlabeled_windows.parquet"),
    ...     output_path=Path("data/processed/unlabeled_with_labels.parquet"),
    ...     epsilon_factor=2.0
    ... )
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Configure logger
logger.add(
    "logs/generate_structure_labels_{time}.log",
    rotation="100 MB",
    retention="30 days",
    level="INFO",
)


# ============================================================================
# ATR (Average True Range) Computation
# ============================================================================


def compute_atr(ohlc: np.ndarray, window: int = 14) -> np.ndarray:
    """Compute Average True Range for volatility-adaptive epsilon.

    ATR measures volatility by calculating the average of true ranges over
    a specified window. Used to create adaptive thresholds for expansion detection.

    Args:
        ohlc: [N, T, 4] array of OHLC data (open, high, low, close)
        window: Rolling window size for ATR calculation (default: 14)

    Returns:
        [N, T] array of ATR values

    Algorithm:
        1. True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        2. ATR = rolling average of True Range over window
    """
    high = ohlc[:, :, 1]
    low = ohlc[:, :, 2]
    close = ohlc[:, :, 3]

    # Shift close prices by 1 timestep to get previous close
    prev_close = np.roll(close, shift=1, axis=1)
    prev_close[:, 0] = close[:, 0]  # First bar has no previous close

    # Compute three components of True Range
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    # True Range is the maximum of the three
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))

    # Compute rolling average of True Range
    atr = np.zeros_like(true_range)
    for t in range(window, true_range.shape[1]):
        atr[:, t] = true_range[:, t - window + 1 : t + 1].mean(axis=1)

    # For initial bars (before window), use simple average up to that point
    for t in range(1, window):
        atr[:, t] = true_range[:, : t + 1].mean(axis=1)

    return atr


# ============================================================================
# Expansion Detection
# ============================================================================


def label_expansion(ohlc: np.ndarray, epsilon_factor: float = 2.0) -> np.ndarray:
    """Label expansion bars using adaptive ATR-based epsilon.

    An expansion occurs when price moves significantly beyond recent range,
    indicating a breakout or strong directional move. Uses ATR to adapt
    threshold to current volatility levels.

    Args:
        ohlc: [N, T, 4] array of OHLC data
        epsilon_factor: Multiplier for ATR to determine epsilon threshold
                       (default: 2.0 = 2x ATR)

    Returns:
        [N, T] array of binary expansion labels (1 = in expansion, 0 = not)

    Algorithm:
        1. Compute ATR(14) for each bar
        2. epsilon = epsilon_factor * ATR
        3. For each bar, check if move exceeds epsilon from recent range
        4. Label as expansion (1) if retracement < epsilon, else 0
    """
    logger.info(f"Computing expansion labels (epsilon_factor={epsilon_factor})...")

    high = ohlc[:, :, 1]
    low = ohlc[:, :, 2]
    close = ohlc[:, :, 3]

    # Compute adaptive epsilon based on ATR
    atr = compute_atr(ohlc, window=14)
    epsilon = epsilon_factor * atr

    # Initialize expansion labels
    expansion = np.zeros((ohlc.shape[0], ohlc.shape[1]), dtype=np.int32)

    # For each bar, check if it's in expansion
    # Expansion: current price significantly beyond recent range
    for t in range(1, ohlc.shape[1]):
        # Get recent range (last 5 bars)
        lookback = min(5, t)
        recent_high = high[:, t - lookback : t].max(axis=1)
        recent_low = low[:, t - lookback : t].min(axis=1)

        # Check if current bar breaks out of recent range by more than epsilon
        # Use high/low for more sensitivity to breakouts
        breakout_up = high[:, t] > (recent_high + epsilon[:, t])
        breakout_down = low[:, t] < (recent_low - epsilon[:, t])

        expansion[:, t] = (breakout_up | breakout_down).astype(np.int32)

    expansion_pct = (expansion.sum() / expansion.size) * 100
    logger.success(
        f"Expansion detection complete: {expansion_pct:.2f}% bars labeled as expansion"
    )

    return expansion


# ============================================================================
# Swing Point Classification
# ============================================================================


def label_swing_points(ohlc: np.ndarray, window: int = 5) -> np.ndarray:
    """Label swing highs and swing lows using local extrema detection.

    Swing points are local maxima (swing highs) or minima (swing lows) that
    represent potential support/resistance levels and trend reversal points.

    Args:
        ohlc: [N, T, 4] array of OHLC data
        window: Window size for local extrema detection (default: 5)
                Uses (window//2) bars before and after

    Returns:
        [N, T] array of swing point labels:
            0 = neither
            1 = swing high
            2 = swing low

    Algorithm:
        For each bar at position t:
        1. Check if high[t] is max in window [t-w//2, t+w//2] → swing high
        2. Check if low[t] is min in window [t-w//2, t+w//2] → swing low
        3. Otherwise → neither
    """
    logger.info(f"Computing swing point labels (window={window})...")

    high = ohlc[:, :, 1]
    low = ohlc[:, :, 2]

    swing_labels = np.zeros((ohlc.shape[0], ohlc.shape[1]), dtype=np.int32)
    half_window = window // 2

    # Iterate through timesteps (excluding edges that don't have full window)
    for t in range(half_window, ohlc.shape[1] - half_window):
        # Extract window around current timestep
        window_high = high[:, t - half_window : t + half_window + 1]
        window_low = low[:, t - half_window : t + half_window + 1]

        # Check if current bar is swing high (local maximum)
        is_swing_high = high[:, t] == window_high.max(axis=1)

        # Check if current bar is swing low (local minimum)
        is_swing_low = low[:, t] == window_low.min(axis=1)

        # Assign labels (swing high takes precedence if both)
        swing_labels[:, t] = np.where(
            is_swing_high, 1, np.where(is_swing_low, 2, 0)
        )

    # Count distribution
    neither_count = (swing_labels == 0).sum()
    high_count = (swing_labels == 1).sum()
    low_count = (swing_labels == 2).sum()
    total = swing_labels.size

    logger.success(
        f"Swing point detection complete: "
        f"neither={neither_count/total:.2%}, "
        f"high={high_count/total:.2%}, "
        f"low={low_count/total:.2%}"
    )

    return swing_labels


# ============================================================================
# Candlestick Pattern Classification
# ============================================================================


def label_candlestick(ohlc: np.ndarray, doji_threshold: float = 0.1) -> np.ndarray:
    """Label candlestick patterns based on body size and direction.

    Classifies each candlestick into one of four categories based on the
    relationship between open and close prices.

    Args:
        ohlc: [N, T, 4] array of OHLC data
        doji_threshold: Threshold for doji detection as fraction of range
                       (default: 0.1 = 10% of high-low range)

    Returns:
        [N, T] array of candlestick pattern labels:
            0 = bullish (close > open)
            1 = bearish (close < open)
            2 = neutral (small body, but not doji)
            3 = doji (open ≈ close)

    Algorithm:
        1. body_size = |close - open|
        2. total_range = high - low
        3. body_ratio = body_size / total_range
        4. Classify:
           - doji: body_ratio < doji_threshold
           - neutral: body_ratio < 0.3 and not doji
           - bullish: close > open
           - bearish: close < open
    """
    logger.info(f"Computing candlestick pattern labels (doji_threshold={doji_threshold})...")

    open_p = ohlc[:, :, 0]
    high = ohlc[:, :, 1]
    low = ohlc[:, :, 2]
    close = ohlc[:, :, 3]

    # Compute body characteristics
    body_size = np.abs(close - open_p)
    total_range = high - low + 1e-8  # Avoid division by zero
    body_ratio = body_size / total_range

    # Initialize labels
    candle_labels = np.zeros((ohlc.shape[0], ohlc.shape[1]), dtype=np.int32)

    # Classify candlesticks
    is_doji = body_ratio < doji_threshold
    is_neutral = (body_ratio < 0.3) & ~is_doji
    is_bullish = (close > open_p) & ~is_doji & ~is_neutral
    is_bearish = (close < open_p) & ~is_doji & ~is_neutral

    # Assign labels
    candle_labels = np.where(
        is_doji,
        3,
        np.where(
            is_neutral,
            2,
            np.where(is_bullish, 0, 1)
        )
    ).astype(np.int32)

    # Count distribution
    bullish_count = (candle_labels == 0).sum()
    bearish_count = (candle_labels == 1).sum()
    neutral_count = (candle_labels == 2).sum()
    doji_count = (candle_labels == 3).sum()
    total = candle_labels.size

    logger.success(
        f"Candlestick pattern detection complete: "
        f"bullish={bullish_count/total:.2%}, "
        f"bearish={bearish_count/total:.2%}, "
        f"neutral={neutral_count/total:.2%}, "
        f"doji={doji_count/total:.2%}"
    )

    return candle_labels


# ============================================================================
# Main Label Generation Pipeline
# ============================================================================


def generate_all_labels(
    data_path: Path,
    output_path: Path,
    epsilon_factor: float = 2.0,
    swing_window: int = 5,
    doji_threshold: float = 0.1,
) -> Tuple[pd.DataFrame, dict]:
    """Generate all structural labels for unlabeled dataset.

    This is the main entry point for label generation. It loads unlabeled
    OHLC data, applies all three labeling algorithms, and saves the enriched
    dataset to parquet format.

    Args:
        data_path: Path to input parquet file with unlabeled windows
        output_path: Path to save labeled output parquet file
        epsilon_factor: ATR multiplier for expansion detection (default: 2.0)
        swing_window: Window size for swing point detection (default: 5)
        doji_threshold: Body ratio threshold for doji detection (default: 0.1)

    Returns:
        Tuple of (labeled_dataframe, statistics_dict)

    Input Format:
        Parquet file with columns:
        - window_id: str, unique identifier
        - features: list of [105, 4] OHLC arrays

    Output Format:
        Parquet file with columns:
        - window_id: str
        - features: list of [105, 4] OHLC arrays
        - expansion_labels: list of [105] binary labels
        - swing_labels: list of [105] 3-class labels
        - candle_labels: list of [105] 4-class labels

    Example:
        >>> stats = generate_all_labels(
        ...     data_path=Path("data/raw/unlabeled_windows.parquet"),
        ...     output_path=Path("data/processed/unlabeled_with_labels.parquet")
        ... )
        >>> print(f"Processed {stats['total_windows']} windows")
        >>> print(f"Total bars: {stats['total_bars']}")
    """
    logger.info("=" * 80)
    logger.info("AUTOMATIC STRUCTURE LABEL GENERATION")
    logger.info("=" * 80)
    logger.info(f"Input: {data_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Config: epsilon_factor={epsilon_factor}, swing_window={swing_window}, "
                f"doji_threshold={doji_threshold}")
    logger.info("=" * 80)

    # Load unlabeled data
    logger.info("Loading unlabeled data...")
    df = pd.read_parquet(data_path)
    logger.success(f"Loaded {len(df)} windows with columns: {df.columns.tolist()}")

    # Convert features to numpy array [N, T, 4]
    logger.info("Converting features to numpy array...")
    # Each row's features is a numpy array (105,) where each element is array (4,)
    # Need to convert each to proper 2D array then stack
    ohlc = np.array([np.array([bar for bar in window]) for window in df["features"]], dtype=np.float32)
    logger.success(f"Converted to numpy array with shape: {ohlc.shape}")

    # Validate shape
    if ohlc.ndim != 3 or ohlc.shape[-1] != 4:
        raise ValueError(
            f"Expected OHLC array shape [N, T, 4], got {ohlc.shape}"
        )

    # Generate labels
    logger.info("")
    logger.info("Generating structural labels...")
    logger.info("-" * 80)

    # 1. Expansion detection
    expansion_labels = label_expansion(ohlc, epsilon_factor=epsilon_factor)

    # 2. Swing point classification
    swing_labels = label_swing_points(ohlc, window=swing_window)

    # 3. Candlestick pattern classification
    candle_labels = label_candlestick(ohlc, doji_threshold=doji_threshold)

    logger.info("-" * 80)

    # Add labels to dataframe
    logger.info("Adding labels to dataframe...")
    df["expansion_labels"] = list(expansion_labels)
    df["swing_labels"] = list(swing_labels)
    df["candle_labels"] = list(candle_labels)

    # Compute statistics
    total_bars = ohlc.shape[0] * ohlc.shape[1]
    stats = {
        "total_windows": len(df),
        "total_bars": total_bars,
        "bars_per_window": ohlc.shape[1],
        "expansion_rate": (expansion_labels.sum() / total_bars),
        "swing_high_rate": ((swing_labels == 1).sum() / total_bars),
        "swing_low_rate": ((swing_labels == 2).sum() / total_bars),
        "bullish_rate": ((candle_labels == 0).sum() / total_bars),
        "bearish_rate": ((candle_labels == 1).sum() / total_bars),
        "neutral_rate": ((candle_labels == 2).sum() / total_bars),
        "doji_rate": ((candle_labels == 3).sum() / total_bars),
    }

    # Save to parquet
    logger.info("Saving labeled data to parquet...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, compression="snappy", index=False)
    logger.success(f"Saved {len(df)} labeled windows to {output_path}")

    # Print summary statistics
    logger.info("")
    logger.info("=" * 80)
    logger.info("LABEL GENERATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total windows: {stats['total_windows']:,}")
    logger.info(f"Total bars: {stats['total_bars']:,}")
    logger.info(f"Bars per window: {stats['bars_per_window']}")
    logger.info("")
    logger.info("Expansion Detection:")
    logger.info(f"  - Expansion rate: {stats['expansion_rate']:.2%}")
    logger.info("")
    logger.info("Swing Points:")
    logger.info(f"  - Swing high rate: {stats['swing_high_rate']:.2%}")
    logger.info(f"  - Swing low rate: {stats['swing_low_rate']:.2%}")
    logger.info("")
    logger.info("Candlestick Patterns:")
    logger.info(f"  - Bullish: {stats['bullish_rate']:.2%}")
    logger.info(f"  - Bearish: {stats['bearish_rate']:.2%}")
    logger.info(f"  - Neutral: {stats['neutral_rate']:.2%}")
    logger.info(f"  - Doji: {stats['doji_rate']:.2%}")
    logger.info("=" * 80)

    return df, stats


# ============================================================================
# CLI Interface
# ============================================================================


def main():
    """Command-line interface for structure label generation."""
    parser = argparse.ArgumentParser(
        description="Generate automatic structure labels for unlabeled OHLC data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python3 scripts/generate_structure_labels.py \\
      --input data/raw/unlabeled_windows.parquet \\
      --output data/processed/unlabeled_with_labels.parquet

  # Custom parameters for more sensitive expansion detection
  python3 scripts/generate_structure_labels.py \\
      --input data/raw/unlabeled_windows.parquet \\
      --output data/processed/unlabeled_sensitive.parquet \\
      --epsilon-factor 1.5 \\
      --swing-window 7 \\
      --doji-threshold 0.05

  # Less sensitive (fewer expansions/swings detected)
  python3 scripts/generate_structure_labels.py \\
      --input data/raw/unlabeled_windows.parquet \\
      --output data/processed/unlabeled_conservative.parquet \\
      --epsilon-factor 3.0 \\
      --swing-window 3 \\
      --doji-threshold 0.15
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input parquet file with unlabeled windows",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save labeled output parquet file",
    )

    parser.add_argument(
        "--epsilon-factor",
        type=float,
        default=2.0,
        help="ATR multiplier for expansion detection (default: 2.0). "
             "Lower = more sensitive (more expansions detected)",
    )

    parser.add_argument(
        "--swing-window",
        type=int,
        default=5,
        help="Window size for swing point detection (default: 5). "
             "Larger = fewer, stronger swing points",
    )

    parser.add_argument(
        "--doji-threshold",
        type=float,
        default=0.1,
        help="Body ratio threshold for doji detection (default: 0.1). "
             "Lower = stricter doji definition",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks on generated labels",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    if args.epsilon_factor <= 0:
        logger.error("epsilon-factor must be positive")
        return 1

    if args.swing_window < 3 or args.swing_window % 2 == 0:
        logger.error("swing-window must be odd and >= 3")
        return 1

    if args.doji_threshold <= 0 or args.doji_threshold >= 1:
        logger.error("doji-threshold must be between 0 and 1")
        return 1

    try:
        # Generate labels
        df, stats = generate_all_labels(
            data_path=args.input,
            output_path=args.output,
            epsilon_factor=args.epsilon_factor,
            swing_window=args.swing_window,
            doji_threshold=args.doji_threshold,
        )

        # Optional validation
        if args.validate:
            logger.info("")
            logger.info("Running validation checks...")
            validate_labels(df)

        logger.success("Label generation completed successfully!")
        return 0

    except Exception as e:
        logger.exception(f"Label generation failed: {e}")
        return 1


def validate_labels(df: pd.DataFrame):
    """Run validation checks on generated labels.

    Args:
        df: DataFrame with generated labels

    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating label integrity...")

    # Check all required columns exist
    required_cols = ["window_id", "features", "expansion_labels", "swing_labels", "candle_labels"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check label shapes match feature shapes
    for idx, row in df.iterrows():
        feature_len = len(row["features"])
        expansion_len = len(row["expansion_labels"])
        swing_len = len(row["swing_labels"])
        candle_len = len(row["candle_labels"])

        if not (feature_len == expansion_len == swing_len == candle_len):
            raise ValueError(
                f"Label length mismatch at row {idx}: "
                f"features={feature_len}, expansion={expansion_len}, "
                f"swing={swing_len}, candle={candle_len}"
            )

    # Check label value ranges
    for idx, row in df.iterrows():
        expansion = np.array(row["expansion_labels"])
        swing = np.array(row["swing_labels"])
        candle = np.array(row["candle_labels"])

        if not np.all((expansion >= 0) & (expansion <= 1)):
            raise ValueError(f"Invalid expansion labels at row {idx}")

        if not np.all((swing >= 0) & (swing <= 2)):
            raise ValueError(f"Invalid swing labels at row {idx}")

        if not np.all((candle >= 0) & (candle <= 3)):
            raise ValueError(f"Invalid candle labels at row {idx}")

    logger.success("All validation checks passed!")


if __name__ == "__main__":
    exit(main())
