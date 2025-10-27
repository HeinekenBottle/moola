#!/usr/bin/env python3
"""Reverse Engineer Expansion Proxies from 210 Labeled Samples.

Extracts mathematical rules from human-labeled expansions to create soft pseudo-labels
for pre-training on unlabeled data. Targets >0.15 correlation with human labels.

Strategy:
- Compute per-bar stats: momentum, streak, volatility, return
- Fit formula: prob = clip((cumulative_return * streak) / volatility, 0, 1)
- Grid-search thresholds on validation set
- Generate 500 synthetic windows with jitter (σ=0.03) for pre-training

Usage:
    python3 scripts/reverse_engineer_proxies.py \
        --data data/processed/labeled/train_latest_overlaps_v2.parquet \
        --output artifacts/reverse_proxies/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split


def compute_temporal_features(ohlc_df: pd.DataFrame) -> pd.DataFrame:
    """Compute temporal features for reverse engineering.

    Args:
        ohlc_df: DataFrame with ['open', 'high', 'low', 'close'] columns

    Returns:
        DataFrame with temporal features
    """
    df = ohlc_df.copy()

    # 1. Momentum (3-bar trailing)
    df["momentum_3"] = (df["close"] - df["close"].shift(3)) / (
        df["high"].rolling(4).max() - df["low"].rolling(4).min() + 1e-9
    )

    # 2. Directional streak (consecutive same-sign bars)
    df["bar_sign"] = (df["close"] > df["open"]).astype(int)
    df["direction_streak"] = df["bar_sign"].rolling(10).sum()  # Count bullish in last 10

    # 3. Acceleration (rate of change of momentum)
    df["accel"] = df["momentum_3"].diff()

    # 4. Cumulative return (5-bar)
    df["cum_return_5"] = df["close"].pct_change().rolling(5).sum()

    # 5. Volatility (20-bar std of range)
    df["volatility"] = (df["high"] - df["low"]).rolling(20).std()

    # Fill NaNs with 0
    df = df.fillna(0)

    return df


def extract_expansion_stats(row, temporal_features: pd.DataFrame) -> dict:
    """Extract statistics from a labeled expansion window.

    Args:
        row: Row from labeled dataset with expansion_start, expansion_end
        temporal_features: DataFrame with computed temporal features

    Returns:
        Dict with expansion statistics
    """
    start = row["expansion_start"]
    end = row["expansion_end"]

    # Stats for bars WITHIN the expansion
    in_expansion = temporal_features.iloc[start : end + 1]

    stats = {
        "momentum_mean": in_expansion["momentum_3"].mean(),
        "momentum_std": in_expansion["momentum_3"].std(),
        "streak_mean": in_expansion["direction_streak"].mean(),
        "streak_max": in_expansion["direction_streak"].max(),
        "cum_return_mean": in_expansion["cum_return_5"].mean(),
        "cum_return_total": in_expansion["cum_return_5"].sum(),
        "vol_mean": in_expansion["volatility"].mean(),
        "accel_mean": in_expansion["accel"].mean(),
        "expansion_length": end - start + 1,
    }

    return stats


def fit_proxy_formula(expansion_stats: list, thresholds: dict = None) -> dict:
    """Fit formula for expansion proxy from collected statistics.

    Formula: prob = clip((cum_return * streak) / (volatility + eps), 0, 1)

    Args:
        expansion_stats: List of dicts with expansion statistics
        thresholds: Optional dict with 'return_weight', 'streak_weight', 'vol_epsilon'

    Returns:
        Dict with fitted formula parameters
    """
    if thresholds is None:
        # Default thresholds (to be grid-searched)
        thresholds = {
            "return_weight": 1.0,
            "streak_weight": 0.1,
            "vol_epsilon": 0.01,
            "clip_min": 0.0,
            "clip_max": 1.0,
        }

    # Compute means across all expansions
    avg_stats = {
        "momentum": np.mean([s["momentum_mean"] for s in expansion_stats]),
        "streak": np.mean([s["streak_mean"] for s in expansion_stats]),
        "cum_return": np.mean([s["cum_return_mean"] for s in expansion_stats]),
        "volatility": np.mean([s["vol_mean"] for s in expansion_stats]),
        "expansion_length": np.mean([s["expansion_length"] for s in expansion_stats]),
    }

    formula = {
        "type": "cumulative_return_streak_ratio",
        "equation": "prob = clip((return * w_return + streak * w_streak) / (vol + eps), min, max)",
        "parameters": thresholds,
        "calibration_stats": avg_stats,
    }

    return formula


def apply_proxy_formula(temporal_features: pd.DataFrame, formula: dict) -> np.ndarray:
    """Apply proxy formula to generate soft labels.

    Args:
        temporal_features: DataFrame with temporal features
        formula: Dict with formula parameters

    Returns:
        Array of soft probabilities [0, 1] for each timestep
    """
    params = formula["parameters"]

    cum_return = temporal_features["cum_return_5"].values
    streak = temporal_features["direction_streak"].values / 10.0  # Normalize to [0, 1]
    vol = temporal_features["volatility"].values

    prob = (cum_return * params["return_weight"] + streak * params["streak_weight"]) / (
        vol + params["vol_epsilon"]
    )

    prob = np.clip(prob, params["clip_min"], params["clip_max"])

    return prob


def grid_search_formula(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Grid search over formula parameters to maximize correlation with human labels.

    Args:
        train_df: Training set with labels
        val_df: Validation set with labels

    Returns:
        Dict with best formula parameters
    """
    print("Grid searching formula parameters...")

    param_grid = {
        "return_weight": [0.5, 1.0, 2.0],
        "streak_weight": [0.05, 0.1, 0.2],
        "vol_epsilon": [0.001, 0.01, 0.1],
    }

    best_corr = -1.0
    best_params = None

    total_combinations = (
        len(param_grid["return_weight"])
        * len(param_grid["streak_weight"])
        * len(param_grid["vol_epsilon"])
    )

    print(f"Testing {total_combinations} parameter combinations...")

    for return_w in param_grid["return_weight"]:
        for streak_w in param_grid["streak_weight"]:
            for vol_eps in param_grid["vol_epsilon"]:
                params = {
                    "return_weight": return_w,
                    "streak_weight": streak_w,
                    "vol_epsilon": vol_eps,
                    "clip_min": 0.0,
                    "clip_max": 1.0,
                }

                # Compute correlation on validation set
                all_probs = []
                all_labels = []

                for _, row in val_df.iterrows():
                    # Build temporal features
                    ohlc_arrays = [arr for arr in row["features"]]
                    ohlc_df = pd.DataFrame(ohlc_arrays, columns=["open", "high", "low", "close"])
                    temporal_feat = compute_temporal_features(ohlc_df)

                    # Apply formula
                    formula = {"type": "test", "parameters": params}
                    probs = apply_proxy_formula(temporal_feat, formula)

                    # True labels
                    binary_mask = np.zeros(105)
                    binary_mask[row["expansion_start"] : row["expansion_end"] + 1] = 1.0

                    all_probs.extend(probs)
                    all_labels.extend(binary_mask)

                # Compute correlation
                corr, _ = pearsonr(all_probs, all_labels)

                if corr > best_corr:
                    best_corr = corr
                    best_params = params

    print(f"Best correlation: {best_corr:.4f}")
    print(f"Best parameters: {best_params}")

    return {"parameters": best_params, "validation_correlation": best_corr}


def main():
    parser = argparse.ArgumentParser(description="Reverse engineer expansion proxies")
    parser.add_argument("--data", type=str, required=True, help="Path to labeled data")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples for debugging")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("REVERSE ENGINEERING EXPANSION PROXIES")
    print("=" * 80)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print()

    # Load labeled data
    print("Loading labeled data...")
    df = pd.read_parquet(args.data)

    if args.max_samples:
        df = df.head(args.max_samples)

    print(f"Total samples: {len(df)}")
    print()

    # Split for grid search
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    print()

    # Extract expansion statistics
    print("Extracting expansion statistics from training set...")
    expansion_stats = []

    for _, row in train_df.iterrows():
        # Build temporal features
        ohlc_arrays = [arr for arr in row["features"]]
        ohlc_df = pd.DataFrame(ohlc_arrays, columns=["open", "high", "low", "close"])
        temporal_feat = compute_temporal_features(ohlc_df)

        # Extract stats
        stats = extract_expansion_stats(row, temporal_feat)
        expansion_stats.append(stats)

    print(f"Extracted stats from {len(expansion_stats)} expansions")
    print()

    # Display summary stats
    print("Summary Statistics:")
    print("-" * 80)
    print(f"  Avg momentum:      {np.mean([s['momentum_mean'] for s in expansion_stats]):.4f}")
    print(f"  Avg streak:        {np.mean([s['streak_mean'] for s in expansion_stats]):.2f}")
    print(f"  Avg cum return:    {np.mean([s['cum_return_mean'] for s in expansion_stats]):.4f}")
    print(f"  Avg volatility:    {np.mean([s['vol_mean'] for s in expansion_stats]):.4f}")
    print(
        f"  Avg length:        {np.mean([s['expansion_length'] for s in expansion_stats]):.1f} bars"
    )
    print()

    # Grid search for best formula
    best_formula = grid_search_formula(train_df, val_df)

    # Fit final formula
    formula = fit_proxy_formula(expansion_stats, best_formula["parameters"])
    formula["validation_correlation"] = best_formula["validation_correlation"]

    print()
    print("Final Formula:")
    print("-" * 80)
    print(f"  Type: {formula['type']}")
    print(f"  Equation: {formula['equation']}")
    print("  Parameters:")
    for k, v in formula["parameters"].items():
        print(f"    {k}: {v}")
    print(f"  Validation correlation: {formula['validation_correlation']:.4f}")
    print()

    # Save formula
    formula_path = output_dir / "proxy_formula.json"
    with open(formula_path, "w") as f:
        json.dump(formula, f, indent=2)
    print(f"✓ Saved formula: {formula_path}")
    print()

    # Save expansion stats for reference
    stats_df = pd.DataFrame(expansion_stats)
    stats_path = output_dir / "expansion_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"✓ Saved expansion stats: {stats_path}")
    print()

    # Visualize proxy vs labels on validation set
    print("Creating visualizations...")

    all_probs = []
    all_labels = []

    for _, row in val_df.iterrows():
        ohlc_arrays = [arr for arr in row["features"]]
        ohlc_df = pd.DataFrame(ohlc_arrays, columns=["open", "high", "low", "close"])
        temporal_feat = compute_temporal_features(ohlc_df)

        probs = apply_proxy_formula(temporal_feat, formula)

        binary_mask = np.zeros(105)
        binary_mask[row["expansion_start"] : row["expansion_end"] + 1] = 1.0

        all_probs.extend(probs)
        all_labels.extend(binary_mask)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(all_labels, all_probs, alpha=0.1, s=1)
    ax1.plot([0, 1], [0, 1], "r--", label="Perfect correlation")
    ax1.set_xlabel("Human Labels (0/1)")
    ax1.set_ylabel("Proxy Probabilities")
    ax1.set_title(f'Proxy vs Labels (corr={formula["validation_correlation"]:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Histogram comparison
    in_expansion_probs = all_probs[all_labels == 1]
    out_expansion_probs = all_probs[all_labels == 0]

    ax2.hist(in_expansion_probs, bins=50, alpha=0.5, label="In-Expansion", density=True)
    ax2.hist(out_expansion_probs, bins=50, alpha=0.5, label="Out-Expansion", density=True)
    ax2.set_xlabel("Proxy Probability")
    ax2.set_ylabel("Density")
    ax2.set_title("Proxy Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    viz_path = output_dir / "proxy_validation.png"
    plt.savefig(viz_path, dpi=150)
    print(f"✓ Saved visualization: {viz_path}")
    print()

    print("=" * 80)
    print("REVERSE ENGINEERING COMPLETE")
    print("=" * 80)
    print()
    print(f"Correlation achieved: {formula['validation_correlation']:.4f}")

    if formula["validation_correlation"] > 0.15:
        print("✅ SUCCESS: Correlation > 0.15 - proxy is viable for pre-training!")
    elif formula["validation_correlation"] > 0.10:
        print("⚠️  MARGINAL: Correlation > 0.10 - may provide weak signal for pre-training")
    else:
        print("❌ INSUFFICIENT: Correlation < 0.10 - proxy too weak, need better features")

    print()
    print("Next steps:")
    print("  1. If corr > 0.15: Use proxy for generating soft labels on unlabeled data")
    print("  2. Generate 500 synthetic windows with jitter (σ=0.03)")
    print("  3. Pre-train BiLSTM encoder with MAE on soft targets")
    print("  4. Transfer to supervised task")


if __name__ == "__main__":
    main()
