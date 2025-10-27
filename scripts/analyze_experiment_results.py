#!/usr/bin/env python3
"""Quick analysis script for experiment results.

Parses results from both experiments and generates summary report.

Usage:
    python3 scripts/analyze_experiment_results.py \\
        --threshold-csv results/threshold_grid.csv \\
        --augmentation-dir artifacts/augmentation_exp/
"""

import argparse
from pathlib import Path

import pandas as pd


def analyze_threshold_experiment(csv_path):
    """Analyze Experiment A results."""
    df = pd.read_csv(csv_path)

    print("=" * 80)
    print("EXPERIMENT A: THRESHOLD PRECISION TUNING")
    print("=" * 80)
    print()

    # Find best by F1
    best_f1_idx = df["f1"].idxmax()
    best_f1 = df.iloc[best_f1_idx]

    # Find best meeting targets (F1 > 0.23, recall >= 0.40)
    targets_met = df[(df["f1"] > 0.23) & (df["recall"] >= 0.40)]

    print("Best F1 Score:")
    print(f"  Threshold: {best_f1['threshold']:.2f}")
    print(f"  F1:        {best_f1['f1']:.4f}")
    print(f"  Precision: {best_f1['precision']:.4f}")
    print(f"  Recall:    {best_f1['recall']:.4f}")
    print()

    if len(targets_met) > 0:
        print(
            f"✅ TARGET MET: {len(targets_met)} thresholds meet criteria (F1 > 0.23, Recall >= 0.40)"
        )
        print()
        print("Thresholds meeting target:")
        for idx, row in targets_met.iterrows():
            print(
                f"  {row['threshold']:.2f}: F1={row['f1']:.4f}, P={row['precision']:.4f}, R={row['recall']:.4f}"
            )
        print()

        # Recommend highest F1 among those meeting targets
        best_target = targets_met.loc[targets_met["f1"].idxmax()]
        print(f"RECOMMENDED THRESHOLD: {best_target['threshold']:.2f}")
        print(f"  - Highest F1 ({best_target['f1']:.4f}) while meeting target constraints")
    else:
        print("⚠️  TARGET NOT MET: No thresholds meet both F1 > 0.23 and Recall >= 0.40")
        print()
        print("Closest to target:")
        # Find threshold with best F1 score
        print(f"  Threshold {best_f1['threshold']:.2f}: F1={best_f1['f1']:.4f}")

    print()

    # F1 vs threshold trend
    print("F1 Trend:")
    for idx, row in df.iterrows():
        bar_length = int(row["f1"] * 100)
        bar = "█" * bar_length
        target_marker = " ✓" if row["f1"] > 0.23 and row["recall"] >= 0.40 else ""
        print(f"  {row['threshold']:.2f}: {bar} {row['f1']:.4f}{target_marker}")

    print()


def analyze_augmentation_experiment(output_dir):
    """Analyze Experiment B results."""
    output_path = Path(output_dir)
    history_path = output_path / "training_history.csv"

    if not history_path.exists():
        print(f"❌ Training history not found: {history_path}")
        return

    df = pd.read_csv(history_path)

    print("=" * 80)
    print("EXPERIMENT B: DATA AUGMENTATION STRATEGY")
    print("=" * 80)
    print()

    # Best and final metrics
    best_f1_epoch = df["span_f1"].idxmax() + 1
    best_f1 = df["span_f1"].max()
    final_f1 = df["span_f1"].iloc[-1]

    print("F1 Score Summary:")
    print(f"  Best:  {best_f1:.4f} (epoch {best_f1_epoch})")
    print(f"  Final: {final_f1:.4f} (epoch {len(df)})")
    print()

    # Check target
    if best_f1 >= 0.25:
        print("✅ TARGET MET: F1 >= 0.25")
        improvement = (best_f1 - 0.22) / 0.22 * 100  # vs baseline ~0.22
        print(f"   Improvement over baseline: +{improvement:.1f}%")
    else:
        print(f"⚠️  TARGET NOT MET: F1 {best_f1:.4f} < 0.25")
        print(f"   Shortfall: {(0.25 - best_f1) * 100:.1f} percentage points")

    print()

    # Precision/Recall at best F1 epoch
    best_epoch_data = df.iloc[best_f1_epoch - 1]
    print(f"Metrics at Best F1 (Epoch {best_f1_epoch}):")
    print(f"  F1:        {best_epoch_data['span_f1']:.4f}")
    print(f"  Precision: {best_epoch_data['span_precision']:.4f}")
    print(f"  Recall:    {best_epoch_data['span_recall']:.4f}")
    print()

    # Loss trend (overfitting check)
    train_loss_trend = df["train_loss"].iloc[-5:].mean() - df["train_loss"].iloc[:5].mean()
    val_loss_trend = df["val_loss"].iloc[-5:].mean() - df["val_loss"].iloc[:5].mean()

    print("Loss Trends (last 5 vs first 5 epochs):")
    print(
        f"  Train loss: {train_loss_trend:+.4f} ({'decreasing' if train_loss_trend < 0 else 'increasing'})"
    )
    print(
        f"  Val loss:   {val_loss_trend:+.4f} ({'decreasing' if val_loss_trend < 0 else 'increasing'})"
    )

    if train_loss_trend < 0 and val_loss_trend > 0:
        print("  ⚠️  Warning: Possible overfitting (train decreasing, val increasing)")
    elif train_loss_trend < 0 and val_loss_trend < 0:
        print("  ✓ Healthy: Both losses decreasing")
    else:
        print("  ⚠️  Unusual pattern: Review training curves")

    print()

    # F1 progression
    print("F1 Progression (every 5 epochs):")
    for i in range(0, len(df), 5):
        epoch = i + 1
        f1 = df.iloc[i]["span_f1"]
        bar_length = int(f1 * 100)
        bar = "█" * bar_length
        target_marker = " ✓" if f1 >= 0.25 else ""
        print(f"  Epoch {epoch:2d}: {bar} {f1:.4f}{target_marker}")

    print()


def compare_experiments(threshold_csv, augmentation_dir):
    """Compare results from both experiments."""
    threshold_path = Path(threshold_csv)
    augmentation_path = Path(augmentation_dir) / "training_history.csv"

    if not threshold_path.exists() or not augmentation_path.exists():
        return

    print("=" * 80)
    print("COMPARISON: THRESHOLD TUNING vs DATA AUGMENTATION")
    print("=" * 80)
    print()

    # Threshold results
    threshold_df = pd.read_csv(threshold_path)
    best_threshold_f1 = threshold_df["f1"].max()
    best_threshold_val = threshold_df.loc[threshold_df["f1"].idxmax(), "threshold"]

    # Augmentation results
    augmentation_df = pd.read_csv(augmentation_path)
    best_augmentation_f1 = augmentation_df["span_f1"].max()

    baseline_f1 = 0.22  # Assumed baseline from 100-epoch training

    print(f"Baseline (100 epochs, no aug):       F1 = {baseline_f1:.4f}")
    print(
        f"Experiment A (threshold {best_threshold_val:.2f}): F1 = {best_threshold_f1:.4f} ({(best_threshold_f1 - baseline_f1) / baseline_f1 * 100:+.1f}%)"
    )
    print(
        f"Experiment B (augmentation 3x):      F1 = {best_augmentation_f1:.4f} ({(best_augmentation_f1 - baseline_f1) / baseline_f1 * 100:+.1f}%)"
    )
    print()

    # Combined estimate (if both improve)
    if best_threshold_f1 > baseline_f1 and best_augmentation_f1 > baseline_f1:
        # Conservative estimate: assume effects are partially independent
        threshold_gain = best_threshold_f1 - baseline_f1
        augmentation_gain = best_augmentation_f1 - baseline_f1
        combined_estimate = (
            baseline_f1 + threshold_gain + (augmentation_gain * 0.7)
        )  # 70% of aug gain on top

        print(
            f"Estimated Combined (A + B):           F1 ≈ {combined_estimate:.4f} ({(combined_estimate - baseline_f1) / baseline_f1 * 100:+.1f}%)"
        )
        print("  (Conservative estimate: assumes 70% of augmentation gain remains)")
        print()

    # Recommendations
    print("RECOMMENDATIONS:")
    if best_threshold_f1 > baseline_f1:
        print(f"  ✓ Deploy optimal threshold {best_threshold_val:.2f} to inference")
    if best_augmentation_f1 >= 0.25:
        print("  ✓ Deploy augmentation (3x, σ=0.03) to training pipeline")
    if best_threshold_f1 > baseline_f1 and best_augmentation_f1 >= 0.25:
        print("  ✓ Combine both strategies for maximum gain")

    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--threshold-csv",
        type=str,
        default="results/threshold_grid.csv",
        help="Path to threshold grid CSV",
    )
    parser.add_argument(
        "--augmentation-dir",
        type=str,
        default="artifacts/augmentation_exp",
        help="Path to augmentation output directory",
    )

    args = parser.parse_args()

    print()

    # Analyze Experiment A
    if Path(args.threshold_csv).exists():
        analyze_threshold_experiment(args.threshold_csv)
    else:
        print(f"❌ Threshold results not found: {args.threshold_csv}")
        print()

    # Analyze Experiment B
    if Path(args.augmentation_dir).exists():
        analyze_augmentation_experiment(args.augmentation_dir)
    else:
        print(f"❌ Augmentation results not found: {args.augmentation_dir}")
        print()

    # Compare
    compare_experiments(args.threshold_csv, args.augmentation_dir)


if __name__ == "__main__":
    main()
