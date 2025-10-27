#!/usr/bin/env python3
"""Fine-tune Position Encoding Model - 20 Epochs.

Loads best position encoding checkpoint (epoch 95) and fine-tunes without CRF
to improve span F1 and IoU metrics.

Usage:
    python3 scripts/finetune_position_20ep.py \
        --checkpoint artifacts/baseline_100ep_position/best_model.pt \
        --data data/processed/labeled/train_latest_overlaps_v2.parquet \
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

        # Build 13D features (12 base + position encoding)
        ohlc_arrays = [arr for arr in row["features"]]
        ohlc_df = pd.DataFrame(ohlc_arrays, columns=["open", "high", "low", "close"])
        cfg = RelativityConfig()
        X_12d, _, _ = build_relativity_features(ohlc_df, cfg.model_dump())

        # Extract 12D features and add position encoding
        X_base = X_12d[0, :, :12]  # (105, 12)
        position_encoding = np.linspace(0, 1, 105, dtype=np.float32)  # (105,)

        # Stack: (105, 13)
        X_13d = np.column_stack([X_base, position_encoding])

        label = self.label_map.get(row["label"], 0)
        center = (row["expansion_start"] + row["expansion_end"]) / 2.0 / 105.0
        length = (row["expansion_end"] - row["expansion_start"]) / 105.0
        pointers = np.array([center, length], dtype=np.float32)
        binary, countdown = create_expansion_labels(row["expansion_start"], row["expansion_end"])

        return {
            "features": torch.from_numpy(X_13d).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "pointers": torch.from_numpy(pointers).float(),
            "binary": torch.from_numpy(binary).float(),
            "countdown": torch.from_numpy(countdown).float(),
        }


def compute_iou(pred_binary, target_binary, threshold=0.5):
    """Compute IoU (Intersection over Union) for span predictions."""
    pred_mask = (pred_binary > threshold).float()
    target_mask = target_binary.float()

    intersection = (pred_mask * target_mask).sum(dim=1)
    union = ((pred_mask + target_mask) > 0).float().sum(dim=1)

    iou = intersection / (union + 1e-7)
    return iou.mean().item()


def train_epoch(model, loader, optimizer, epoch, device):
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

        # Loss 1: Type classification
        loss_type = F.cross_entropy(output["logits"], labels)

        # Loss 2: Pointer regression
        loss_ptr = F.huber_loss(output["pointers"], pointers, delta=0.08)

        # Loss 3: Span detection (soft span loss with class weighting)
        pred_probs = output["expansion_binary"]
        loss_span = soft_span_loss(pred_probs, binary, reduction="mean", pos_weight=13.1)

        # Loss 4: Countdown regression
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
    span_iou_scores = []

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

            # Use soft probabilities for F1/IoU computation
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

            # Compute IoU
            iou = compute_iou(pred_probs, binary, threshold=0.5)
            span_iou_scores.append(iou)

    avg_loss = total_loss / len(loader)
    avg_f1 = np.mean(span_f1_scores) if span_f1_scores else 0.0
    avg_iou = np.mean(span_iou_scores) if span_iou_scores else 0.0

    return avg_loss, avg_f1, avg_iou


def main():
    parser = argparse.ArgumentParser(description="Fine-tune position encoding model (20 epochs)")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to position encoding checkpoint"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--output", type=str, default="artifacts/position_finetuned/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Extract model state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Create model WITHOUT CRF (matching checkpoint architecture)
    print("Creating JadeCompact (no CRF)...")
    model = JadeCompact(
        input_size=13,  # 12 base features + position encoding
        hidden_size=96,
        num_layers=1,
        num_classes=3,
        dropout=0.7,
        predict_pointers=True,
        predict_expansion_sequence=True,
        use_crf=False,  # No CRF - use soft span loss
    )

    # Load checkpoint weights
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    print(f"Loaded checkpoint. Parameters: {sum(p.numel() for p in model.parameters())}")

    # Load data
    print(f"Loading data from {args.data}...")
    dataset = ExpansionDataset(args.data)

    # Split (on original indices)
    n_original = len(dataset)
    _, val_indices = train_test_split(range(n_original), test_size=0.2, random_state=42)

    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    train_indices = [i for i in range(n_original) if i not in val_indices]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []

    print(f"\nFine-tuning for {args.epochs} epochs...")
    best_f1 = 0
    best_model_path = output_dir / "best_model.pt"
    best_iou = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, epoch, device)
        val_loss, val_f1, val_iou = eval_epoch(model, val_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "span_f1": val_f1,
                "span_iou": val_iou,
            }
        )

        status = "✓" if val_f1 > best_f1 else ""
        print(
            f"Epoch {epoch:2d}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, F1={val_f1:.4f}, IoU={val_iou:.4f} {status}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_iou = val_iou
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, best_model_path)

    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "metrics.csv", index=False)

    print(f"\n{'='*70}")
    print("✅ FINE-TUNING COMPLETE")
    print(f"{'='*70}")
    print(f"Best F1:     {best_f1:.4f} (with IoU={best_iou:.4f})")
    print(f"Final F1:    {history[-1]['span_f1']:.4f}")
    print("Target:      0.25 (baseline was 0.220)")
    print(
        f"Status:      {'✅ TARGET MET' if best_f1 >= 0.25 else '⚠️ APPROACHING TARGET' if best_f1 >= 0.23 else '❌ NEEDS MORE WORK'}"
    )
    print(f"Saved:       {best_model_path}")
    print(f"Metrics:     {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
