#!/usr/bin/env python3
"""Threshold optimization for soft span mask predictions.

Extracts hard spans from soft mask predictions using different thresholds
and calculates F1 scores to find the optimal threshold for span detection.

This validates that the model is learning meaningful soft masks that can
be converted to accurate hard span predictions.
"""

import argparse
import sys
from pathlib import Path
import pickle

sys.path.insert(0, str(Path.cwd() / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from moola.models.jade_core import JadeCompact, soft_span_loss, compute_span_f1, compute_span_metrics
from moola.features.relativity import build_relativity_features, RelativityConfig


class ExpansionDataset(Dataset):
    """Dataset for expansion detection with soft span masks."""

    def __init__(self, parquet_path, max_samples=None):
        """Load parquet file and build features."""
        import pyarrow.parquet as pq

        df = pq.read_table(parquet_path).to_pandas()
        if max_samples:
            df = df.iloc[:max_samples]

        self.data = []
        config = RelativityConfig()

        for idx, row in df.iterrows():
            ohlc = np.array([row["open"], row["high"], row["low"], row["close"]], dtype=np.float32)
            features = build_relativity_features(ohlc, config)

            # Soft span mask (continuous 0-1)
            span_mask = np.array(row["soft_span_mask"], dtype=np.float32)

            # Binary targets
            binary = np.float32(row.get("expansion_binary", 0))
            countdown = np.float32(row.get("countdown_bars", 0))

            self.data.append({
                "features": torch.FloatTensor(features),
                "binary": torch.FloatTensor([binary]),
                "span_mask": torch.FloatTensor(span_mask),
                "countdown": torch.FloatTensor([countdown]),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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


def evaluate_threshold(model, loader, device, threshold=0.5):
    """Evaluate F1 score for a specific threshold."""
    model.eval()
    all_pred_spans = []
    all_true_spans = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            span_masks = batch["span_mask"].to(device)

            output = model(features)
            pred_probs = output["expansion_span"]  # (batch, seq_len)

            # Extract hard spans for each sample
            for i in range(pred_probs.shape[0]):
                pred_soft = pred_probs[i].cpu().numpy()
                true_soft = span_masks[i].cpu().numpy()

                # Convert soft → hard using threshold
                pred_hard = extract_hard_spans(pred_soft, threshold=threshold, min_length=1)
                true_hard = extract_hard_spans(true_soft, threshold=0.5, min_length=1)

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
    parser = argparse.ArgumentParser(description="Threshold optimization for soft span masks")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/labeled/train_latest_overlaps_v2.parquet",
        help="Path to validation data",
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
        default=0.1,
        help="Minimum threshold to test",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=0.9,
        help="Maximum threshold to test",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Threshold step size",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("THRESHOLD OPTIMIZATION FOR SOFT SPAN MASKS")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model = JadeCompact(
        input_size=12,
        predict_pointers=True,
        predict_expansion_sequence=True,
        use_crf=False,
    ).to(args.device)

    try:
        checkpoint = torch.load(args.model_path, map_location=args.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Loaded checkpoint from {args.model_path}")
    except FileNotFoundError:
        print(f"⚠️  No checkpoint found at {args.model_path}, using random weights")

    # Load data
    print("\nLoading data...")
    dataset = ExpansionDataset(args.data_path)
    train_idx, val_idx = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )

    val_loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx),
    )

    print(f"✓ Loaded {len(val_idx)} validation samples")

    # Test thresholds
    print("\nTesting thresholds...")
    print(f"{'Threshold':<12} {'F1':<8} {'Precision':<12} {'Recall':<8} {'Pred Spans':<12} {'True Spans'}")
    print("-" * 80)

    thresholds = np.arange(args.min_threshold, args.max_threshold + args.step, args.step)
    results = []

    for threshold in thresholds:
        metrics = evaluate_threshold(model, val_loader, args.device, threshold=threshold)
        results.append({
            "threshold": threshold,
            **metrics,
        })

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
