#!/usr/bin/env python3
"""Optimize span prediction threshold to maximize F1 score.

Tests thresholds from 0.3-0.7 on validation set and identifies the threshold
that achieves the best F1 score for hard span extraction from soft predictions.

This validates that the soft span loss approach is learning meaningful continuous
predictions that can be converted to high-quality hard spans via thresholding.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from moola.models.jade_core import JadeCompact, compute_span_f1, compute_span_metrics
from moola.features.relativity import build_relativity_features, RelativityConfig


def create_expansion_labels(expansion_start, expansion_end, window_length=105):
    """Create binary expansion label."""
    binary_mask = np.zeros(window_length, dtype=np.float32)
    binary_mask[expansion_start : expansion_end + 1] = 1.0
    return binary_mask


class ExpansionDataset(Dataset):
    """Dataset with expansion labels (matches train_expansion_local.py format)."""

    def __init__(self, data_path, max_samples=None):
        self.df = pd.read_parquet(data_path)
        if max_samples:
            self.df = self.df.head(max_samples)

        self.label_map = {"consolidation": 0, "retracement": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Build 12D features from raw OHLC (stored as list of [o, h, l, c])
        ohlc_arrays = [arr for arr in row["features"]]
        ohlc_df = pd.DataFrame(ohlc_arrays, columns=["open", "high", "low", "close"])

        cfg = RelativityConfig()
        X_12d, _, _ = build_relativity_features(ohlc_df, cfg.model_dump())

        # Binary expansion label (0-1 mask over 105 timesteps)
        binary = create_expansion_labels(row["expansion_start"], row["expansion_end"])

        # Pattern label
        label = self.label_map.get(row["label"], 0)

        return {
            "features": torch.from_numpy(X_12d[0]).float(),
            "binary": torch.FloatTensor(binary),
            "label": torch.tensor(label, dtype=torch.long),
        }


def extract_hard_spans(soft_probs, threshold=0.5, min_length=1):
    """Extract hard spans from soft probabilities using threshold.

    Args:
        soft_probs: (seq_len,) array of soft probabilities [0, 1]
        threshold: Threshold for binarization
        min_length: Minimum span length to keep

    Returns:
        (seq_len,) binary array with hard spans
    """
    binary = (soft_probs > threshold).astype(np.float32)

    # Remove short spans (noise)
    if min_length > 1:
        # Find connected components
        changes = np.diff(np.concatenate([[0], binary, [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        for start, end in zip(starts, ends):
            if end - start < min_length:
                binary[start:end] = 0

    return binary


def evaluate_threshold(model, loader, device, threshold=0.5, min_length=1):
    """Evaluate F1 score for a specific threshold."""
    model.eval()
    all_pred_spans = []
    all_true_spans = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            binary = batch["binary"].to(device)

            output = model(features)
            pred_probs = output["expansion_binary"]  # (batch, seq_len)

            # Extract hard spans for each sample
            for i in range(pred_probs.shape[0]):
                pred_soft = pred_probs[i].cpu().numpy()
                true_soft = binary[i].cpu().numpy()

                # Convert soft → hard using threshold
                pred_hard = extract_hard_spans(pred_soft, threshold=threshold, min_length=min_length)
                true_hard = extract_hard_spans(true_soft, threshold=0.5, min_length=min_length)

                all_pred_spans.append(pred_hard)
                all_true_spans.append(true_hard)

    # Concatenate all predictions
    all_pred_spans = np.concatenate(all_pred_spans)
    all_true_spans = np.concatenate(all_true_spans)

    # Calculate metrics
    if all_true_spans.sum() > 0:  # Only if there are true positives
        f1 = f1_score(all_true_spans, all_pred_spans, zero_division=0)
        precision = precision_score(all_true_spans, all_pred_spans, zero_division=0)
        recall = recall_score(all_true_spans, all_pred_spans, zero_division=0)
    else:
        f1 = precision = recall = 0.0

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "n_true_spans": int(all_true_spans.sum()),
        "n_pred_spans": int(all_pred_spans.sum()),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Optimize span extraction threshold")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/labeled/train_latest_overlaps_v2.parquet",
        help="Path to training data",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cpu or cuda",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.3,
        help="Minimum threshold to test",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=0.7,
        help="Maximum threshold to test",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Threshold step size",
    )
    parser.add_argument(
        "--min-span-length",
        type=int,
        default=1,
        help="Minimum span length to keep",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("THRESHOLD OPTIMIZATION FOR SOFT SPAN MASKS")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    dataset = ExpansionDataset(args.data_path)
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    val_loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx),
    )

    print(f"✓ Loaded {len(val_idx)} validation samples")

    # Create and initialize model (freshly instantiated, not trained)
    print("\nCreating model...")
    device = torch.device(args.device)
    model = JadeCompact(
        input_size=12,
        predict_pointers=True,
        predict_expansion_sequence=True,
        use_crf=False,
    ).to(device)

    print(f"✓ Model created ({model.get_num_parameters()['total']:,} parameters)")

    # Test thresholds
    print("\nTesting thresholds...")
    print(f"{'Threshold':<12} {'F1':<8} {'Precision':<12} {'Recall':<8} {'Pred Spans':<12} {'True Spans'}")
    print("-" * 80)

    thresholds = np.arange(args.min_threshold, args.max_threshold + args.step, args.step)
    results = []

    for threshold in thresholds:
        metrics = evaluate_threshold(
            model, val_loader, device, threshold=threshold, min_length=args.min_span_length
        )
        results.append(
            {
                "threshold": threshold,
                **metrics,
            }
        )

        print(
            f"{threshold:<12.2f} {metrics['f1']:<8.4f} {metrics['precision']:<12.4f} "
            f"{metrics['recall']:<8.4f} {metrics['n_pred_spans']:<12} {metrics['n_true_spans']}"
        )

    # Find optimal threshold
    best_result = max(results, key=lambda x: x["f1"])
    print("\n" + "=" * 80)
    print(f"OPTIMAL THRESHOLD: {best_result['threshold']:.2f}")
    print(f"  F1 Score:  {best_result['f1']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall:    {best_result['recall']:.4f}")
    print(f"  Predicted: {best_result['n_pred_spans']} spans")
    print(f"  True:      {best_result['n_true_spans']} spans")
    print("=" * 80)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("threshold_optimization_results.csv", index=False)
    print(f"\n✓ Results saved to threshold_optimization_results.csv")

    return best_result["threshold"]


if __name__ == "__main__":
    main()
