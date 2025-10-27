#!/usr/bin/env python3
"""Exp B: Data Augmentation via Jitter (σ=0.03).

Expands 210 samples to 630 (3x) via Gaussian jitter on 13D features.
Trains 20 epochs with pos_weight=13.1, uncertainty weighting.
Target: F1 >= 0.25 by epoch 20.

Usage:
    python3 scripts/train_augmented_20ep.py \
        --data data/processed/labeled/train_latest_overlaps_v2.parquet \
        --output artifacts/augmentation_exp/ \
        --epochs 20 \
        --device cuda
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path.cwd() / "src"))

from moola.features.relativity import RelativityConfig, build_relativity_features
from moola.models.jade_core import JadeCompact, compute_span_metrics, soft_span_loss


def create_expansion_labels(expansion_start, expansion_end, window_length=105):
    """Create soft span mask and countdown from pointers."""
    binary_mask = np.zeros(window_length, dtype=np.float32)
    binary_mask[expansion_start : expansion_end + 1] = 1.0

    countdown = np.arange(window_length, dtype=np.float32) - expansion_start
    countdown = -countdown
    countdown = (countdown + 20) / 40.0
    countdown = np.clip(countdown, 0, 1)

    return binary_mask, countdown


class AugmentedExpansionDataset(Dataset):
    """Dataset with Gaussian jitter augmentation."""

    def __init__(self, data_path, augment_factor=2, jitter_sigma=0.03, max_samples=None):
        self.df = pd.read_parquet(data_path)
        if max_samples:
            self.df = self.df.head(max_samples)

        self.augment_factor = augment_factor
        self.jitter_sigma = jitter_sigma
        self.label_map = {"consolidation": 0, "retracement": 1}

    def __len__(self):
        return len(self.df) * (1 + self.augment_factor)

    def __getitem__(self, idx):
        # Determine if this is original or augmented
        original_idx = idx // (1 + self.augment_factor)
        aug_idx = idx % (1 + self.augment_factor)

        row = self.df.iloc[original_idx]

        # Build features
        ohlc_arrays = [arr for arr in row["features"]]
        ohlc_df = pd.DataFrame(ohlc_arrays, columns=["open", "high", "low", "close"])
        cfg = RelativityConfig()
        X_12d, _, _ = build_relativity_features(ohlc_df, cfg.model_dump())

        # Apply jitter if augmented
        if aug_idx > 0:
            noise = np.random.normal(0, self.jitter_sigma, X_12d.shape)
            X_12d = X_12d + noise
            X_12d = np.clip(X_12d, -3, 3)  # Prevent extreme values

        label = self.label_map.get(row["label"], 0)
        center = (row["expansion_start"] + row["expansion_end"]) / 2.0 / 105.0
        length = (row["expansion_end"] - row["expansion_start"]) / 105.0
        pointers = np.array([center, length], dtype=np.float32)
        binary, countdown = create_expansion_labels(row["expansion_start"], row["expansion_end"])

        # Extract only first 12 features (exclude position encoding which failed validation)
        features_12d = X_12d[0, :, :12]  # Shape: (105, 12)

        return {
            "features": torch.from_numpy(features_12d).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "pointers": torch.from_numpy(pointers).float(),
            "binary": torch.from_numpy(binary).float(),
            "countdown": torch.from_numpy(countdown).float(),
        }


def train_epoch(model, loader, optimizer, logger, epoch, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    loss_components_sum = defaultdict(float)

    for batch_idx, batch in enumerate(loader):
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        pointers = batch["pointers"].to(device)
        binary = batch["binary"].to(device)
        countdown = batch["countdown"].to(device)

        output = model(features)

        # Compute losses
        loss_type = F.cross_entropy(output["logits"], labels)
        loss_ptr = F.huber_loss(output["pointers"], pointers, delta=0.08)
        pred_probs = output["expansion_binary"]
        loss_span = soft_span_loss(pred_probs, binary, reduction="mean", pos_weight=13.1)
        loss_countdown = F.huber_loss(output["expansion_countdown"], countdown, delta=1.0)

        loss_components_sum["type"] += loss_type.item()
        loss_components_sum["ptr"] += loss_ptr.item()
        loss_components_sum["span"] += loss_span.item()
        loss_components_sum["countdown"] += loss_countdown.item()

        # Uncertainty weighting
        sigma_ptr = torch.exp(model.log_sigma_ptr)
        sigma_type = torch.exp(model.log_sigma_type)
        sigma_span = torch.exp(model.log_sigma_span)
        sigma_countdown = torch.exp(model.log_sigma_countdown)

        loss = (
            (1 / (2 * sigma_ptr**2)) * loss_ptr
            + torch.log(sigma_ptr)
            + (1 / (2 * sigma_type**2)) * loss_type
            + torch.log(sigma_type)
            + (1 / (2 * sigma_span**2)) * loss_span
            + torch.log(sigma_span)
            + (1 / (2 * sigma_countdown**2)) * loss_countdown
            + torch.log(sigma_countdown)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


def eval_epoch(model, loader, device):
    """Evaluate one epoch."""
    model.eval()
    total_loss = 0
    span_f1_scores = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            pointers = batch["pointers"].to(device)
            binary = batch["binary"].to(device)
            countdown = batch["countdown"].to(device)

            output = model(features)

            loss_type = F.cross_entropy(output["logits"], labels)
            loss_ptr = F.huber_loss(output["pointers"], pointers, delta=0.08)
            pred_probs = output["expansion_binary"]
            loss_span = soft_span_loss(pred_probs, binary, reduction="mean", pos_weight=13.1)
            loss_countdown = F.huber_loss(output["expansion_countdown"], countdown, delta=1.0)

            sigma_ptr = torch.exp(model.log_sigma_ptr)
            sigma_type = torch.exp(model.log_sigma_type)
            sigma_span = torch.exp(model.log_sigma_span)
            sigma_countdown = torch.exp(model.log_sigma_countdown)

            loss = (
                (1 / (2 * sigma_ptr**2)) * loss_ptr
                + torch.log(sigma_ptr)
                + (1 / (2 * sigma_type**2)) * loss_type
                + torch.log(sigma_type)
                + (1 / (2 * sigma_span**2)) * loss_span
                + torch.log(sigma_span)
                + (1 / (2 * sigma_countdown**2)) * loss_countdown
                + torch.log(sigma_countdown)
            )

            total_loss += loss.item()

            # Compute span metrics
            for i in range(pred_probs.size(0)):
                metrics = compute_span_metrics(pred_probs[i], binary[i], threshold=0.5)
                span_f1_scores.append(metrics["f1"])

    avg_loss = total_loss / len(loader)
    avg_f1 = np.mean(span_f1_scores) if span_f1_scores else 0.0

    return avg_loss, avg_f1


def main():
    parser = argparse.ArgumentParser(description="Train with data augmentation (Exp B)")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="artifacts/augmentation_exp/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--n-augment", type=int, default=2, help="Augmentation factor (2 = 3x data)"
    )
    parser.add_argument("--sigma", type=float, default=0.03, help="Jitter std dev")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data}...")
    dataset = AugmentedExpansionDataset(
        args.data, augment_factor=args.n_augment, jitter_sigma=args.sigma
    )

    # Split (on original indices before augmentation)
    n_original = len(dataset.df)
    _, val_indices = train_test_split(range(n_original), test_size=0.2, random_state=42)

    # Map validation indices to augmented dataset (only original samples, no augmentation)
    val_dataset_indices = [i * (1 + args.n_augment) for i in val_indices]  # Original only
    train_dataset_indices = [
        i * (1 + args.n_augment) + j
        for i in range(n_original)
        if i not in val_indices
        for j in range(1 + args.n_augment)
    ]

    train_dataset = torch.utils.data.Subset(dataset, train_dataset_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_dataset_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(
        f"Train: {len(train_dataset)} samples (augmented), Val: {len(val_dataset)} samples (original only)"
    )

    # Create model
    print("Creating model...")
    model = JadeCompact(
        input_size=12,
        hidden_size=96,
        num_layers=1,
        num_classes=3,
        dropout=0.7,
        predict_pointers=True,
        predict_expansion_sequence=True,
    )
    model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []

    print(f"\nTraining for {args.epochs} epochs...")
    best_f1 = 0
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, None, epoch, device)
        val_loss, val_f1 = eval_epoch(model, val_loader, device)

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "f1": val_f1}
        )

        status = "✓" if val_f1 > best_f1 else ""
        print(
            f"Epoch {epoch:2d}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, F1={val_f1:.4f} {status}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, best_model_path)

    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Final F1: {history[-1]['f1']:.4f}")
    print("Target: 0.25")
    print(f"Status: {'✅ TARGET MET' if best_f1 >= 0.25 else '❌ TARGET MISSED'}")
    print(f"Saved: {best_model_path}")


if __name__ == "__main__":
    main()
