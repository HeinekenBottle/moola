#!/usr/bin/env python3
"""Experiment A: Threshold Precision Tuning (5 min).

Grid search thresholds 0.30-0.40 (step 0.02) on validation set to find
optimal operating point for expansion span detection.

Target: Find threshold yielding F1 > 0.23 without recall drop below 0.40
Expected runtime: 5 minutes on GPU

Usage:
    python3 scripts/experiment_a_threshold_grid.py \\
        --checkpoint artifacts/baseline_100ep/best_model.pt \\
        --data data/processed/labeled/train_latest_overlaps_v2.parquet \\
        --output results/threshold_grid.csv \\
        --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path.cwd() / "src"))

from moola.features.relativity import RelativityConfig, build_relativity_features
from moola.models.jade_core import JadeCompact


def create_expansion_labels(expansion_start, expansion_end, window_length=105):
    """Create binary expansion mask from pointers."""
    binary_mask = np.zeros(window_length, dtype=np.float32)
    binary_mask[expansion_start : expansion_end + 1] = 1.0
    return binary_mask


class ExpansionDataset(Dataset):
    """Dataset with expansion labels."""

    def __init__(self, data_path, max_samples=None):
        self.df = pd.read_parquet(data_path)
        if max_samples:
            self.df = self.df.head(max_samples)

        self.label_map = {"consolidation": 0, "retracement": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Build 13D features (12D relativity + position_encoding)
        ohlc_arrays = [arr for arr in row["features"]]
        ohlc_df = pd.DataFrame(ohlc_arrays, columns=["open", "high", "low", "close"])

        cfg = RelativityConfig()
        X_13d, _, _ = build_relativity_features(ohlc_df, cfg.model_dump())

        # Binary expansion label
        binary = create_expansion_labels(row["expansion_start"], row["expansion_end"])

        # Pattern label
        label = self.label_map.get(row["label"], 0)

        return {
            "features": torch.from_numpy(X_13d[0]).float(),
            "binary": torch.from_numpy(binary).float(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def evaluate_threshold(model, loader, device, threshold=0.5):
    """Evaluate F1, precision, recall at specific threshold."""
    model.eval()
    all_pred_spans = []
    all_true_spans = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            binary = batch["binary"].to(device)

            output = model(features)
            pred_probs = output["expansion_binary"]  # (batch, 105)

            # Binarize predictions
            pred_binary = (pred_probs > threshold).float()

            # Flatten for metrics
            all_pred_spans.append(pred_binary.cpu().numpy().flatten())
            all_true_spans.append(binary.cpu().numpy().flatten())

    # Concatenate all samples
    all_pred_spans = np.concatenate(all_pred_spans)
    all_true_spans = np.concatenate(all_true_spans)

    # Compute metrics
    f1 = f1_score(all_true_spans, all_pred_spans, zero_division=0)
    precision = precision_score(all_true_spans, all_pred_spans, zero_division=0)
    recall = recall_score(all_true_spans, all_pred_spans, zero_division=0)

    n_true_positive = int(all_true_spans.sum())
    n_pred_positive = int(all_pred_spans.sum())

    return {
        "threshold": threshold,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "n_true_positive": n_true_positive,
        "n_pred_positive": n_pred_positive,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment A: Threshold Grid Search")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)",
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
        default="results/threshold_grid.csv",
        help="Output CSV path for results",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.30,
        help="Minimum threshold",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=0.40,
        help="Maximum threshold",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.02,
        help="Threshold step size",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device: cpu or cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("=" * 80)
    print("EXPERIMENT A: THRESHOLD PRECISION TUNING")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data}")
    print(
        f"Threshold range: [{args.min_threshold:.2f}, {args.max_threshold:.2f}], step={args.step:.2f}"
    )
    print(f"Device: {args.device}")
    print()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    print("Loading data...")
    dataset = ExpansionDataset(args.data)
    _, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=args.seed)

    val_loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx),
    )

    print(f"Validation: {len(val_idx)} samples")

    # Load model
    print("\nLoading model...")
    device = torch.device(args.device)
    model = JadeCompact(
        input_size=13,  # 13 features with position_encoding
        predict_pointers=True,
        predict_expansion_sequence=True,
        use_crf=False,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Grid search
    print("\nTesting thresholds...")
    print(f"{'Threshold':<12} {'F1':<8} {'Precision':<12} {'Recall':<8} {'Target Met'}")
    print("-" * 80)

    thresholds = np.arange(args.min_threshold, args.max_threshold + args.step / 2, args.step)
    results = []

    start_time = time.time()

    for threshold in thresholds:
        metrics = evaluate_threshold(model, val_loader, device, threshold)
        results.append(metrics)

        # Check if target met: F1 > 0.23, recall >= 0.40
        target_met = metrics["f1"] > 0.23 and metrics["recall"] >= 0.40
        target_status = "✓ YES" if target_met else "  no"

        print(
            f"{threshold:<12.2f} {metrics['f1']:<8.4f} {metrics['precision']:<12.4f} "
            f"{metrics['recall']:<8.4f} {target_status}"
        )

    elapsed = time.time() - start_time

    # Find best threshold
    best_result = max(results, key=lambda x: x["f1"])

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Best threshold: {best_result['threshold']:.2f}")
    print(f"  F1 Score:  {best_result['f1']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall:    {best_result['recall']:.4f}")
    print()

    # Check if target met
    if best_result["f1"] > 0.23 and best_result["recall"] >= 0.40:
        print("✅ TARGET MET: F1 > 0.23 and Recall >= 0.40")
    else:
        print("⚠️  TARGET NOT MET")
        if best_result["f1"] <= 0.23:
            print(f"   - F1 {best_result['f1']:.4f} <= 0.23")
        if best_result["recall"] < 0.40:
            print(f"   - Recall {best_result['recall']:.4f} < 0.40")

    print()
    print(f"Runtime: {elapsed:.1f}s")
    print("=" * 80)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved results to {output_path}")

    # Save summary
    summary_path = output_path.parent / "threshold_summary.txt"
    with open(summary_path, "w") as f:
        f.write("EXPERIMENT A: THRESHOLD PRECISION TUNING\n")
        f.write("=" * 80 + "\n")
        f.write(f"Runtime: {elapsed:.1f}s\n")
        f.write(f"Thresholds tested: {len(thresholds)}\n")
        f.write(f"\nBest threshold: {best_result['threshold']:.2f}\n")
        f.write(f"  F1 Score:  {best_result['f1']:.4f}\n")
        f.write(f"  Precision: {best_result['precision']:.4f}\n")
        f.write(f"  Recall:    {best_result['recall']:.4f}\n")
        f.write("\nTarget (F1 > 0.23, Recall >= 0.40): ")
        if best_result["f1"] > 0.23 and best_result["recall"] >= 0.40:
            f.write("MET ✓\n")
        else:
            f.write("NOT MET ✗\n")

    print(f"✓ Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
