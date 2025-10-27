#!/usr/bin/env python3
"""Validate Pre-training Features Against Labeled Expansions.

Cross-checks heuristic features (expansion_proxy, range_z, etc.) against
human-annotated expansion regions to:
1. Measure feature-label alignment (correlation, overlap)
2. Identify where features miss expansions or create false positives
3. Suggest threshold/formula adjustments for better pre-training signals
4. Generate diagnostic visualizations

Usage:
    python3 scripts/validate_pretraining_features.py \
        --data data/processed/labeled/train_latest_overlaps_v2.parquet \
        --output artifacts/feature_validation/
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

from moola.features.relativity import RelativityConfig, build_relativity_features


def load_labeled_data(data_path: Path) -> pd.DataFrame:
    """Load labeled expansion windows with ground truth."""
    df = pd.read_parquet(data_path)
    print(f"✓ Loaded {len(df)} labeled windows from {data_path}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
    return df


def extract_features_from_window(row: pd.Series) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract 12D relativity features from a single window.

    Returns:
        features: (105, 12) array of features
        ohlc_df: (105, 4) DataFrame of OHLC for reference
    """
    # Convert features list to OHLC DataFrame
    ohlc_arrays = [arr for arr in row["features"]]
    ohlc_df = pd.DataFrame(ohlc_arrays, columns=["open", "high", "low", "close"])

    # Build relativity features
    cfg = RelativityConfig()
    X_12d, mask, meta = build_relativity_features(ohlc_df, cfg.model_dump())

    # X_12d is (1, 105, 12) - squeeze to (105, 12)
    features = X_12d[0]

    return features, ohlc_df


def create_expansion_mask(
    expansion_start: int, expansion_end: int, length: int = 105
) -> np.ndarray:
    """Create binary mask for expansion region."""
    mask = np.zeros(length, dtype=bool)
    mask[expansion_start : expansion_end + 1] = True
    return mask


def compute_feature_alignment(
    features: np.ndarray,
    expansion_mask: np.ndarray,
    feature_idx: int,
    feature_name: str,
    threshold: float = None,
) -> dict:
    """Compute alignment between a feature and expansion labels.

    Args:
        features: (105, 12) feature array
        expansion_mask: (105,) binary mask of expansion region
        feature_idx: Index of feature to analyze (e.g., 10 for expansion_proxy)
        feature_name: Name of feature for reporting
        threshold: Optional threshold to binarize feature

    Returns:
        Alignment metrics dictionary
    """
    feature_values = features[:, feature_idx]

    # Compute basic statistics
    in_expansion_mean = feature_values[expansion_mask].mean() if expansion_mask.any() else 0.0
    out_expansion_mean = feature_values[~expansion_mask].mean() if (~expansion_mask).any() else 0.0
    separation = in_expansion_mean - out_expansion_mean

    # If threshold provided, compute binary metrics
    if threshold is not None:
        feature_binary = feature_values > threshold

        # True positives: Feature high AND in expansion
        tp = (feature_binary & expansion_mask).sum()

        # False positives: Feature high BUT not in expansion
        fp = (feature_binary & ~expansion_mask).sum()

        # False negatives: Feature low BUT in expansion
        fn = (~feature_binary & expansion_mask).sum()

        # True negatives: Feature low AND not in expansion
        tn = (~feature_binary & ~expansion_mask).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "feature_name": feature_name,
            "threshold": threshold,
            "in_expansion_mean": in_expansion_mean,
            "out_expansion_mean": out_expansion_mean,
            "separation": separation,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    else:
        # Just correlation without thresholding
        correlation = np.corrcoef(feature_values, expansion_mask.astype(float))[0, 1]
        return {
            "feature_name": feature_name,
            "in_expansion_mean": in_expansion_mean,
            "out_expansion_mean": out_expansion_mean,
            "separation": separation,
            "correlation": correlation,
        }


def analyze_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze all 12 features against labeled expansions.

    Returns:
        DataFrame with alignment metrics for each feature
    """
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
    ]

    # Feature thresholds to test (only for relevant features)
    thresholds = {
        "range_z": [0.2, 0.3, 0.4, 0.5],  # High range indicates expansion
        "expansion_proxy": [0.1, 0.2, 0.3, 0.5],  # Positive values indicate expansion
        "body_pct": [0.05, 0.1, 0.15],  # Large bodies
        "consol_proxy": [0.5, 1.0, 1.5],  # High values indicate consolidation (inverse)
    }

    results = []

    for idx, row in df.iterrows():
        features, ohlc = extract_features_from_window(row)
        expansion_mask = create_expansion_mask(row["expansion_start"], row["expansion_end"])

        # Analyze each feature
        for feat_idx, feat_name in enumerate(feature_names):
            # First, compute correlation without threshold
            corr_metrics = compute_feature_alignment(
                features, expansion_mask, feat_idx, feat_name, threshold=None
            )
            corr_metrics["window_id"] = idx
            corr_metrics["label"] = row["label"]
            results.append(corr_metrics)

            # If feature has thresholds to test, compute F1 for each
            if feat_name in thresholds:
                for thresh in thresholds[feat_name]:
                    thresh_metrics = compute_feature_alignment(
                        features, expansion_mask, feat_idx, feat_name, threshold=thresh
                    )
                    thresh_metrics["window_id"] = idx
                    thresh_metrics["label"] = row["label"]
                    results.append(thresh_metrics)

    return pd.DataFrame(results)


def plot_feature_alignment(results_df: pd.DataFrame, output_dir: Path):
    """Generate diagnostic visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature separation (in-expansion vs out-of-expansion means)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get correlation-only rows (no threshold)
    corr_only = results_df[results_df["threshold"].isna()].copy()

    # Aggregate by feature (mean separation across all windows)
    feature_sep = (
        corr_only.groupby("feature_name")
        .agg(
            {
                "separation": "mean",
                "correlation": "mean",
                "in_expansion_mean": "mean",
                "out_expansion_mean": "mean",
            }
        )
        .reset_index()
    )

    feature_sep = feature_sep.sort_values("separation", ascending=False)

    ax.barh(feature_sep["feature_name"], feature_sep["separation"], color="steelblue")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean Separation (In-Expansion - Out-of-Expansion)")
    ax.set_title("Feature Alignment: Which Features Separate Expansions?")
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    fig.savefig(output_dir / "feature_separation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved: {output_dir / 'feature_separation.png'}")

    # 2. Threshold optimization for key features
    threshold_features = ["range_z", "expansion_proxy", "body_pct", "consol_proxy"]

    for feat_name in threshold_features:
        feat_data = results_df[
            (results_df["feature_name"] == feat_name) & (results_df["threshold"].notna())
        ].copy()

        if len(feat_data) == 0:
            continue

        # Aggregate by threshold
        thresh_summary = (
            feat_data.groupby("threshold")
            .agg(
                {
                    "f1": "mean",
                    "precision": "mean",
                    "recall": "mean",
                }
            )
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            thresh_summary["threshold"], thresh_summary["f1"], marker="o", label="F1", linewidth=2
        )
        ax.plot(
            thresh_summary["threshold"],
            thresh_summary["precision"],
            marker="s",
            label="Precision",
            linewidth=2,
        )
        ax.plot(
            thresh_summary["threshold"],
            thresh_summary["recall"],
            marker="^",
            label="Recall",
            linewidth=2,
        )

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Metric Value")
        ax.set_title(f"{feat_name}: Threshold Optimization for Expansion Detection")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_dir / f"threshold_opt_{feat_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"✓ Saved: {output_dir / f'threshold_opt_{feat_name}.png'}")

    # 3. Correlation heatmap
    corr_pivot = corr_only.pivot_table(
        index="window_id",
        columns="feature_name",
        values="correlation",
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_pivot.T, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5, ax=ax)
    ax.set_title("Feature-Expansion Correlation per Window")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Feature")

    plt.tight_layout()
    fig.savefig(output_dir / "feature_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved: {output_dir / 'feature_correlation_heatmap.png'}")


def generate_recommendations(results_df: pd.DataFrame) -> list[str]:
    """Generate feature adjustment recommendations based on alignment analysis."""
    recommendations = []

    # Get correlation-only rows
    corr_only = results_df[results_df["threshold"].isna()].copy()
    feature_agg = (
        corr_only.groupby("feature_name")
        .agg(
            {
                "separation": "mean",
                "correlation": "mean",
            }
        )
        .reset_index()
    )

    # Sort by correlation strength
    feature_agg["abs_corr"] = feature_agg["correlation"].abs()
    feature_agg = feature_agg.sort_values("abs_corr", ascending=False)

    print("\n" + "=" * 80)
    print("FEATURE-EXPANSION ALIGNMENT SUMMARY")
    print("=" * 80)
    print(feature_agg[["feature_name", "correlation", "separation"]].to_string(index=False))
    print()

    # Recommendations for top features
    for _, row in feature_agg.head(5).iterrows():
        feat = row["feature_name"]
        corr = row["correlation"]
        sep = row["separation"]

        if corr > 0.2:
            recommendations.append(
                f"✅ {feat}: Strong positive correlation ({corr:.3f}) - KEEP in pre-training"
            )
        elif corr < -0.2:
            recommendations.append(
                f"⚠️ {feat}: Negative correlation ({corr:.3f}) - Consider INVERTING or REMOVING"
            )
        elif abs(corr) < 0.1:
            recommendations.append(
                f"❌ {feat}: Weak correlation ({corr:.3f}) - Low value for pre-training"
            )

    # Check threshold-based features
    threshold_features = ["range_z", "expansion_proxy", "body_pct", "consol_proxy"]

    for feat_name in threshold_features:
        feat_data = results_df[
            (results_df["feature_name"] == feat_name) & (results_df["threshold"].notna())
        ].copy()

        if len(feat_data) == 0:
            continue

        # Find best threshold
        best_row = feat_data.loc[feat_data["f1"].idxmax()]

        recommendations.append(
            f"🎯 {feat_name}: Best threshold = {best_row['threshold']:.2f} "
            f"(F1={best_row['f1']:.3f}, P={best_row['precision']:.3f}, R={best_row['recall']:.3f})"
        )

    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description="Validate pre-training features against labeled expansions"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/labeled/train_latest_overlaps_v2.parquet",
        help="Path to labeled data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/feature_validation",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (for quick testing)",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output)

    print("=" * 80)
    print("PRE-TRAINING FEATURE VALIDATION")
    print("=" * 80)
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print()

    # Load data
    df = load_labeled_data(data_path)

    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"  Limiting to {len(df)} samples for testing")

    # Analyze features
    print("\n🔬 Analyzing feature alignment...")
    results_df = analyze_all_features(df)

    # Save raw results
    results_csv = output_dir / "feature_alignment_results.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_csv, index=False)
    print(f"✓ Saved raw results: {results_csv}")

    # Generate visualizations
    print("\n📊 Generating diagnostic plots...")
    plot_feature_alignment(results_df, output_dir)

    # Generate recommendations
    print("\n💡 RECOMMENDATIONS")
    print("=" * 80)
    recommendations = generate_recommendations(results_df)

    for rec in recommendations:
        print(rec)

    # Save recommendations
    rec_file = output_dir / "recommendations.txt"
    with open(rec_file, "w") as f:
        f.write("\n".join(recommendations))
    print(f"\n✓ Saved recommendations: {rec_file}")

    print("\n" + "=" * 80)
    print("✅ VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nReview diagnostics in: {output_dir}")
    print("\nNext steps:")
    print("1. Review feature_separation.png to see which features align best")
    print("2. Check threshold_opt_*.png for optimal feature thresholds")
    print("3. Adjust feature formulas in src/moola/features/relativity.py if needed")
    print("4. Use validated features for pre-training on unlabeled data")


if __name__ == "__main__":
    main()
