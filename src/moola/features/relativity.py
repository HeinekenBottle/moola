"""Relativity Features - Price-relative transformations for Moola.

Implements AGENTS.md Section 6 requirements:
- No absolute price leakage
- Features in [0,1] range for relative values
- Invariant to linear price scaling
- Causal: uses only closed candle information

Usage:
    >>> from moola.features.relativity import build_features
    >>> X, mask, meta = build_features(df, cfg)
"""

from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

from .zigzag import CausalZigZag


class RelativityConfig(BaseModel):
    """Configuration for relativity feature builder.

    AGENTS.md compliance: All parameters configurable via YAML.
    """

    ohlc_eps: float = Field(1.0e-6, description="Small constant to prevent division by zero")
    ohlc_ema_range_period: int = Field(20, description="EMA period for range normalization")
    atr_period: int = Field(10, description="ATR calculation period")
    zigzag_k: float = Field(1.2, description="ATR multiplier for swing confirmation")
    zigzag_hybrid_confirm_lookback: int = Field(5, description="Lookback for hybrid confirmation")
    zigzag_hybrid_min_retrace_atr: float = Field(
        0.5, description="Minimum retrace for hybrid confirmation"
    )
    window_length: int = Field(105, description="Fixed sequence length")
    window_overlap: float = Field(0.5, description="Window overlap fraction")

    @validator("window_length")
    def validate_window_length(cls, v):
        if v < 10 or v > 500:
            raise ValueError("window_length must be between 10 and 500")
        return v


def candle_shape(
    open_price: float, high: float, low: float, close: float, ema_range: float, eps: float = 1e-6
) -> tuple[float, float, float, float, float, float]:
    """Calculate candle shape features.

    Args:
        open_price, high, low, close: OHLC prices for completed bar
        ema_range: EMA of recent price ranges for normalization
        eps: Small constant to prevent division by zero

    Returns:
        Tuple of (open_norm, close_norm, body_pct, upper_wick_pct, lower_wick_pct, range_z)
    """
    rng = max(high - low, eps)
    open_norm = (open_price - low) / rng
    close_norm = (close - low) / rng
    body_pct = (close - open_price) / rng
    upper_wick_pct = (high - max(open_price, close)) / rng
    lower_wick_pct = (min(open_price, close) - low) / rng
    range_z = (high - low) / max(ema_range, eps)

    # Clip to reasonable bounds
    open_norm = np.clip(open_norm, 0, 1)
    close_norm = np.clip(close_norm, 0, 1)
    body_pct = np.clip(body_pct, -1, 1)  # Can be negative for red candles
    upper_wick_pct = np.clip(upper_wick_pct, 0, 1)
    lower_wick_pct = np.clip(lower_wick_pct, 0, 1)
    range_z = np.clip(range_z, 0, 3)

    return open_norm, close_norm, body_pct, upper_wick_pct, lower_wick_pct, range_z


def build_features(df: pd.DataFrame, cfg: RelativityConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build relativity features with causal zigzag integration.

    Sequential pass over CLOSED candles only, no absolute price in outputs.

    Args:
        df: OHLC DataFrame (no volume)
        cfg: Relativity configuration

    Returns:
        X: Feature tensor [N, K, D] with float32 dtype, D=12 (includes expansion_proxy and consol_proxy)
        mask: Boolean mask [N, K] (False for warmup period)
        meta: Metadata dictionary
    """
    # Validate required columns (OHLC only for minimal system)
    required_cols = ["open", "high", "low", "close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n_bars = len(df)
    window_length = cfg.window_length

    if n_bars < window_length:
        raise ValueError(f"Data length {n_bars} less than window length {window_length}")

    # Initialize outputs (11 core features + expansion_proxy + consol_proxy + position_encoding)
    n_features = 13  # Added position_encoding for ICT structure awareness
    n_windows = n_bars - window_length + 1
    X = np.zeros((n_windows, window_length, n_features), dtype=np.float32)
    mask = np.ones((n_windows, window_length), dtype=bool)

    # Initialize zigzag detector
    zigzag = CausalZigZag(
        atr_period=cfg.atr_period,
        k=cfg.zigzag_k,
        hybrid_lb=cfg.zigzag_hybrid_confirm_lookback,
        hybrid_min_atr=cfg.zigzag_hybrid_min_retrace_atr,
    )

    # Initialize EMA for range
    ema_range = None
    alpha = 2.0 / (cfg.ohlc_ema_range_period + 1)

    # Sequential pass over data
    for i in range(n_bars):
        row = df.iloc[i]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]

        # Update EMA of range
        current_range = h - l
        if ema_range is None:
            ema_range = current_range
        else:
            ema_range = alpha * current_range + (1 - alpha) * ema_range

        # Update zigzag detector
        prev_SH, prev_SL = zigzag.update(o, h, l, c)

        # Get zigzag state
        state = zigzag.get_state()
        atr = state["current_atr"]

        # Calculate bars since swings
        bars_since_SH = i - state["prev_SH"][0] if state["prev_SH"] else 0
        bars_since_SL = i - state["prev_SL"][0] if state["prev_SL"] else 0

        # Calculate features for EVERY bar (not just bars >= window_length - 1)
        # Candle shape features (6 dims)
        open_norm, close_norm, body_pct, upper_wick_pct, lower_wick_pct, range_z = candle_shape(
            o, h, l, c, ema_range, cfg.ohlc_eps
        )

        # Swing-relative features (4 dims)
        from .zigzag import swing_relative

        dist_to_prev_SH, dist_to_prev_SL, bars_since_SH_norm, bars_since_SL_norm = swing_relative(
            c,
            state["prev_SH"][1] if state["prev_SH"] else None,
            state["prev_SL"][1] if state["prev_SL"] else None,
            atr,
            bars_since_SH,
            bars_since_SL,
            K=window_length,  # Normalize by window length
        )

        # Expansion proxy (1 dim): Synthetic surge detector
        # expansion_proxy = range_z × leg_dir × body_pct
        # Flags ICT expansions (5-6 bar surges) without absolute prices
        current_trend = state.get("current_trend")
        leg_dir = 1.0 if current_trend == "up" else (-1.0 if current_trend == "down" else 0.0)
        expansion_proxy = range_z * leg_dir * body_pct
        expansion_proxy = np.clip(expansion_proxy, -2.0, 2.0)  # Bounded ~[-2, 2]

        # Consolidation proxy (1 dim): Low-volatility consolidation detector
        # consol_proxy = (1 - range_z) × (1 - |body_pct|) × bars_since_swing_norm
        # HIGH values during consolidations (low volatility, small bodies, time since swing)
        bars_since_swing_norm = (bars_since_SH_norm + bars_since_SL_norm) / 2.0
        consol_proxy = (1.0 - range_z / 3.0) * (1.0 - abs(body_pct)) * bars_since_swing_norm
        consol_proxy = np.clip(consol_proxy, 0.0, 3.0)  # Bounded [0, 3]

        # Store core features (12 features, position encoding added below per-window)
        base_features = [
            open_norm,
            close_norm,
            body_pct,
            upper_wick_pct,
            lower_wick_pct,
            range_z,
            dist_to_prev_SH,
            dist_to_prev_SL,
            bars_since_SH_norm,
            bars_since_SL_norm,
            expansion_proxy,
            consol_proxy,
        ]

        # Populate ALL windows that contain this bar
        # Bar i appears in windows that have started (first_window >= 0) and contain bar i
        first_window = max(0, i - window_length + 1)
        last_window = min(i, n_windows - 1)

        for win_idx in range(first_window, last_window + 1):
            # Position of bar i within window win_idx
            timestep_in_window = i - win_idx

            # Add position encoding (t / K-1) for ICT window structure awareness
            position_encoding = timestep_in_window / (window_length - 1)  # [0, 1]

            # Store all 13 features
            X[win_idx, timestep_in_window, :12] = base_features
            X[win_idx, timestep_in_window, 12] = position_encoding

            # Set mask for warmup period (first 15 bars of each window)
            warmup_bars = 15
            if timestep_in_window < warmup_bars:
                mask[win_idx, timestep_in_window] = False

    # Shift features to align with causal structure
    # Each window should contain features for bars [i-window_length+1, i]
    # This is already correct from our construction above

    # Prepare metadata
    feature_names = [
        "open_norm",
        "close_norm",
        "body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "range_z",
        "dist_to_prev_SH",
        "dist_to_prev_SL",
        "bars_since_SH_norm",
        "bars_since_SL_norm",
        "expansion_proxy",
        "consol_proxy",
        "position_encoding",  # Linear position t/(K-1) for ICT window structure
    ]
    feature_ranges = {
        "open_norm": "[0, 1]",
        "close_norm": "[0, 1]",
        "body_pct": "[-1, 1]",
        "upper_wick_pct": "[0, 1]",
        "lower_wick_pct": "[0, 1]",
        "range_z": "[0, 3]",
        "dist_to_prev_SH": "[-3, 3]",
        "dist_to_prev_SL": "[-3, 3]",
        "bars_since_SH_norm": "[0, 3]",
        "bars_since_SL_norm": "[0, 3]",
        "expansion_proxy": "[-2, 2]",
        "consol_proxy": "[0, 3]",
        "position_encoding": "[0, 1]",  # t/(K-1)
    }

    meta = {
        "n_features": n_features,
        "feature_names": feature_names,
        "feature_ranges": feature_ranges,
        "properties": [
            "Price-relative (no absolute leakage)",
            "Volatility-adjusted (ATR-normalized)",
            "Scaling invariant",
            "Causal (no future information)",
            "Zigzag swing detection",
        ],
        "window_length": window_length,
        "warmup_bars": 15,
        "config": cfg.dict(),
    }

    return X, mask, meta


# Convenience wrapper for the main build_features function
def build_relativity_features(
    data: pd.DataFrame, config: Optional[dict] = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build relativity features with output contract.

    AGENTS.md Section 15: Every feature builder returns X, mask, meta.

    Args:
        data: OHLC DataFrame
        config: Configuration dictionary

    Returns:
        X: Feature tensor [N, K, D] with float32 dtype
        mask: Boolean mask [N, K] (all True for valid windows)
        meta: Metadata dictionary
    """
    cfg = RelativityConfig(**config) if config else RelativityConfig()
    return build_features(data, cfg)


# AGENTS.md CLI integration
def main():
    """CLI entry point for relativity features.

    Usage: moola features --config configs/features/relativity.yaml --in data.parquet --out features.parquet
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Build relativity features from OHLC data")
    parser.add_argument("--config", required=True, help="Path to relativity config")
    parser.add_argument("--in", required=True, dest="input", help="Input parquet file")
    parser.add_argument("--out", required=True, dest="output", help="Output parquet file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")

    args = parser.parse_args()

    try:
        # Load data
        import time

        start_time = time.time()

        data = pd.read_parquet(args.input)
        if args.verbose:
            print(f"Loaded data: {args.input}")
            print(f"Shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")

        # Load config
        import yaml

        with open(args.config) as f:
            config = yaml.safe_load(f)

        # Build features
        X, mask, meta = build_relativity_features(data, config.get("relativity", {}))

        # Save results

        # Convert to DataFrame for saving
        feature_names = meta["feature_names"]
        n_samples, seq_len, n_features = X.shape

        # Flatten for parquet storage
        df_data = {}
        for i, name in enumerate(feature_names):
            df_data[name] = X[:, :, i].reshape(-1)  # Flatten

        df_data["window_id"] = np.repeat(np.arange(n_samples), seq_len)
        df_data["timestep"] = np.tile(np.arange(seq_len), n_samples)

        out_df = pd.DataFrame(df_data)
        out_df.to_parquet(args.output)

        # Print summary (AGENTS.md Section 15)
        elapsed = time.time() - start_time
        print(f"Input path: {args.input}")
        print(f"Output path: {args.output}")
        print(f"Rows processed: {len(data)}")
        print(f"Windows created: {n_samples}")
        print(f"Features per window: {n_features}")
        print(f"Window size: {seq_len}")
        print(f"Wall time: {elapsed:.2f}s")

        if args.verbose:
            print(f"Feature info: {meta}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
