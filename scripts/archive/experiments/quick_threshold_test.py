#!/usr/bin/env python3
"""Quick threshold optimization on trained model.

Tests thresholds 0.30-0.50 to find optimal F1 without re-training.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path.cwd() / "src"))

from moola.features.relativity import RelativityConfig, build_relativity_features
from moola.models.jade_core import JadeCompact, compute_span_metrics


def create_expansion_labels(expansion_start, expansion_end, window_length=105):
    """Create soft span mask and countdown from pointers."""
    binary_mask = np.zeros(window_length, dtype=np.float32)
    binary_mask[expansion_start : expansion_end + 1] = 1.0

    countdown = np.arange(window_length, dtype=np.float32) - expansion_start
    countdown = -countdown
    countdown = (countdown + 20) / 40.0
    countdown = np.clip(countdown, 0, 1)

    return binary_mask, countdown


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

        # Build 12D features
        ohlc_arrays = [arr for arr in row["features"]]
        ohlc_df = pd.DataFrame(ohlc_arrays, columns=["open", "high", "low", "close"])
        cfg = RelativityConfig()
        X_12d, _, _ = build_relativity_features(ohlc_df, cfg.model_dump())

        label = self.label_map.get(row["label"], 0)
        center = (row["expansion_start"] + row["expansion_end"]) / 2.0 / 105.0
        length = (row["expansion_end"] - row["expansion_start"]) / 105.0
        pointers = np.array([center, length], dtype=np.float32)
        binary, countdown = create_expansion_labels(row["expansion_start"], row["expansion_end"])

        return {
            "features": torch.from_numpy(X_12d[0]).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "pointers": torch.from_numpy(pointers).float(),
            "binary": torch.from_numpy(binary).float(),
            "countdown": torch.from_numpy(countdown).float(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device)

    # Extract model state dict from checkpoint
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = JadeCompact(
        input_size=13,  # Position encoding adds 13th feature
        hidden_size=96,
        num_layers=1,
        num_classes=3,
        dropout=0.7,
        predict_pointers=True,
        predict_expansion_sequence=True,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load data
    print(f"Loading data from {args.data}...")
    dataset = ExpansionDataset(args.data)
    _, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Collect all predictions
    all_pred_probs = []
    all_targets = []

    print("Collecting predictions...")
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            binary = batch["binary"]

            output = model(features)
            pred_probs = output["expansion_binary"].cpu()  # (batch, 105)

            all_pred_probs.append(pred_probs)
            all_targets.append(binary)

    all_pred_probs = torch.cat(all_pred_probs, dim=0)  # (N, 105)
    all_targets = torch.cat(all_targets, dim=0)  # (N, 105)

    # Test thresholds
    print("\n🎯 Threshold Optimization Results:\n")
    print(f"{'Threshold':<12} {'F1':<8} {'Precision':<12} {'Recall':<8}")
    print("-" * 45)

    best_f1 = 0
    best_threshold = 0.5

    for threshold in np.arange(0.30, 0.51, 0.05):
        # Compute metrics at this threshold
        metrics = compute_span_metrics(all_pred_probs, all_targets, threshold=threshold)

        f1 = metrics["f1"]
        precision = metrics["precision"]
        recall = metrics["recall"]

        marker = " ✓" if f1 > best_f1 else ""
        print(f"{threshold:<12.2f} {f1:<8.4f} {precision:<12.4f} {recall:<8.4f}{marker}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print("-" * 45)
    print(f"\n✅ Best threshold: {best_threshold:.2f} (F1 = {best_f1:.4f})")

    # Show improvement
    baseline_metrics = compute_span_metrics(all_pred_probs, all_targets, threshold=0.5)
    print("\nImprovement over threshold=0.5:")
    print(
        f"  F1:        {baseline_metrics['f1']:.4f} → {best_f1:.4f} (+{best_f1 - baseline_metrics['f1']:.4f})"
    )
    print(f"  Precision: {baseline_metrics['precision']:.4f} → {metrics['precision']:.4f}")
    print(f"  Recall:    {baseline_metrics['recall']:.4f} → {metrics['recall']:.4f}")


if __name__ == "__main__":
    main()
