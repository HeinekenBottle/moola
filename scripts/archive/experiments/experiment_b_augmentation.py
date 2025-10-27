#!/usr/bin/env python3
"""Experiment B: Data Augmentation Strategy (20 epochs).

Jitter augmentation: σ=0.03 on 210 training samples → 630 total (3x).
Train for 20 epochs with augmented data to test if diversity improves F1.

Expected: F1 0.25+ via diversity (Synth PDF: dropout=0.65 prevents overfit)
Expected runtime: 20-25 minutes on GPU

Usage:
    python3 scripts/experiment_b_augmentation.py \\
        --data data/processed/labeled/train_latest_overlaps_v2.parquet \\
        --output artifacts/augmentation_exp/ \\
        --epochs 20 \\
        --sigma 0.03 \\
        --n-augment 2 \\
        --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path.cwd() / "src"))

from moola.features.relativity import RelativityConfig, build_relativity_features
from moola.models.jade_core import JadeCompact, soft_span_loss


def create_expansion_labels(expansion_start, expansion_end, window_length=105):
    """Create soft span mask and normalized countdown from pointers."""
    binary_mask = np.zeros(window_length, dtype=np.float32)
    binary_mask[expansion_start : expansion_end + 1] = 1.0

    countdown = np.arange(window_length, dtype=np.float32) - expansion_start
    countdown = -countdown
    countdown = (countdown + 20) / 40.0  # Map [-20, 20] → [0, 1]
    countdown = np.clip(countdown, 0, 1)

    return binary_mask, countdown


def jitter_features(features: np.ndarray, sigma: float = 0.03, seed: int = None) -> np.ndarray:
    """Apply Gaussian jitter to features for augmentation.

    Args:
        features: (seq_len, n_features) array
        sigma: Standard deviation of Gaussian noise (0.03 = 3% of normalized range)
        seed: Random seed for reproducibility

    Returns:
        Augmented features with same shape
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    noise = rng.normal(0, sigma, features.shape).astype(np.float32)
    return features + noise


class AugmentedExpansionDataset(Dataset):
    """Dataset with expansion labels + jitter augmentation."""

    def __init__(self, data_path, n_augment=0, sigma=0.03, seed=42, max_samples=None):
        """
        Args:
            data_path: Path to parquet file
            n_augment: Number of augmented copies per sample (0 = no augmentation)
            sigma: Jitter standard deviation
            seed: Random seed for reproducibility
            max_samples: Max samples to load (for testing)
        """
        self.df = pd.read_parquet(data_path)
        if max_samples:
            self.df = self.df.head(max_samples)

        self.n_augment = n_augment
        self.sigma = sigma
        self.seed = seed
        self.label_map = {"consolidation": 0, "retracement": 1}

        # Total samples = original + augmented
        self.n_original = len(self.df)
        self.n_total = self.n_original * (1 + n_augment)

        print(
            f"Dataset: {self.n_original} original → {self.n_total} total (augmentation {n_augment}x)"
        )

    def __len__(self):
        return self.n_total

    def __getitem__(self, idx):
        # Determine if this is original or augmented
        original_idx = idx % self.n_original
        augment_idx = idx // self.n_original  # 0 = original, 1+ = augmented

        row = self.df.iloc[original_idx]

        # Build 13D features from raw OHLC
        ohlc_arrays = [arr for arr in row["features"]]
        ohlc_df = pd.DataFrame(ohlc_arrays, columns=["open", "high", "low", "close"])

        cfg = RelativityConfig()
        X_13d, _, _ = build_relativity_features(ohlc_df, cfg.model_dump())

        features = X_13d[0]  # (105, 13)

        # Apply jitter if augmented sample
        if augment_idx > 0:
            # Use deterministic seed for reproducibility
            augment_seed = self.seed + original_idx * 1000 + augment_idx
            features = jitter_features(features, sigma=self.sigma, seed=augment_seed)

        # Labels (unchanged by augmentation)
        label = self.label_map.get(row["label"], 0)
        center = (row["expansion_start"] + row["expansion_end"]) / 2.0 / 105.0
        length = (row["expansion_end"] - row["expansion_start"]) / 105.0
        pointers = np.array([center, length], dtype=np.float32)

        binary, countdown = create_expansion_labels(row["expansion_start"], row["expansion_end"])

        return {
            "features": torch.from_numpy(features).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "pointers": torch.from_numpy(pointers).float(),
            "binary": torch.from_numpy(binary).float(),
            "countdown": torch.from_numpy(countdown).float(),
        }


def train_epoch(model, loader, optimizer, device, pos_weight=13.1):
    """Train one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        pointers = batch["pointers"].to(device)
        binary = batch["binary"].to(device)
        countdown = batch["countdown"].to(device)

        # Forward
        output = model(features)

        # Compute raw losses
        loss_type = F.cross_entropy(output["logits"], labels)
        loss_ptr = F.huber_loss(output["pointers"], pointers, delta=0.08)
        pred_probs = output["expansion_binary"]
        loss_span = soft_span_loss(pred_probs, binary, reduction="mean", pos_weight=pos_weight)
        loss_countdown = F.huber_loss(output["expansion_countdown"], countdown, delta=1.0)

        # Uncertainty weighting
        sigma_ptr = output["sigma_ptr"]
        sigma_type = output["sigma_type"]
        sigma_span = output.get("sigma_span", torch.tensor(1.0, device=device))
        sigma_countdown = output.get("sigma_countdown", torch.tensor(1.0, device=device))

        loss = (
            (1.0 / (2 * sigma_ptr**2)) * loss_ptr
            + torch.log(sigma_ptr)
            + (1.0 / (2 * sigma_type**2)) * loss_type
            + torch.log(sigma_type)
            + (1.0 / (2 * sigma_span**2)) * loss_span
            + torch.log(sigma_span)
            + (1.0 / (2 * sigma_countdown**2)) * loss_countdown
            + torch.log(sigma_countdown)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate_epoch(model, loader, device, pos_weight=13.1):
    """Validate one epoch with F1 computation."""
    model.eval()
    total_loss = 0
    n_batches = 0

    # Collect predictions for F1
    all_pred_spans = []
    all_true_spans = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            pointers = batch["pointers"].to(device)
            binary = batch["binary"].to(device)
            countdown = batch["countdown"].to(device)

            output = model(features)

            # Losses
            loss_type = F.cross_entropy(output["logits"], labels)
            loss_ptr = F.huber_loss(output["pointers"], pointers, delta=0.08)
            pred_probs = output["expansion_binary"]
            loss_span = soft_span_loss(pred_probs, binary, reduction="mean", pos_weight=pos_weight)
            loss_countdown = F.huber_loss(output["expansion_countdown"], countdown, delta=1.0)

            total_loss += (loss_type + loss_ptr + loss_span + loss_countdown).item()
            n_batches += 1

            # Collect span predictions (threshold 0.5)
            pred_binary = (pred_probs > 0.5).float()
            all_pred_spans.append(pred_binary.cpu().numpy().flatten())
            all_true_spans.append(binary.cpu().numpy().flatten())

    # Compute span F1
    from sklearn.metrics import f1_score, precision_score, recall_score

    all_pred_spans = np.concatenate(all_pred_spans)
    all_true_spans = np.concatenate(all_true_spans)

    f1 = f1_score(all_true_spans, all_pred_spans, zero_division=0)
    precision = precision_score(all_true_spans, all_pred_spans, zero_division=0)
    recall = recall_score(all_true_spans, all_pred_spans, zero_division=0)

    return {
        "val_loss": total_loss / n_batches,
        "span_f1": f1,
        "span_precision": precision,
        "span_recall": recall,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment B: Data Augmentation")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/labeled/train_latest_overlaps_v2.parquet",
        help="Path to labeled data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/augmentation_exp",
        help="Output directory for logs and checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument(
        "--n-augment", type=int, default=2, help="Augmented copies per sample (0 = no augmentation)"
    )
    parser.add_argument("--sigma", type=float, default=0.03, help="Jitter standard deviation")
    parser.add_argument(
        "--pos-weight", type=float, default=13.1, help="Positive class weight for span loss"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cpu or cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("=" * 80)
    print("EXPERIMENT B: DATA AUGMENTATION STRATEGY")
    print("=" * 80)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Augmentation: {args.n_augment}x copies, sigma={args.sigma}")
    print(f"Positive weight: {args.pos_weight}")
    print(f"Device: {args.device}")
    print()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data with augmentation
    print("Loading data with augmentation...")
    full_dataset = AugmentedExpansionDataset(
        args.data,
        n_augment=args.n_augment,
        sigma=args.sigma,
        seed=args.seed,
    )

    # Split train/val (split on ORIGINAL indices to avoid data leakage)
    n_original = full_dataset.n_original
    train_orig_idx, val_orig_idx = train_test_split(
        range(n_original), test_size=0.2, random_state=args.seed
    )

    # Map to augmented indices
    # Training: use original + augmented for train split
    train_idx = []
    for orig_idx in train_orig_idx:
        for aug_idx in range(1 + args.n_augment):
            train_idx.append(orig_idx + aug_idx * n_original)

    # Validation: use only original samples (no augmentation)
    val_idx = list(val_orig_idx)

    print(
        f"Train: {len(train_idx)} samples ({len(train_orig_idx)} original × {1 + args.n_augment})"
    )
    print(f"Val: {len(val_idx)} samples (original only, no augmentation)")

    train_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
    )
    val_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx),
    )

    # Model
    print("\nCreating model...")
    device = torch.device(args.device)
    model = JadeCompact(
        input_size=13,  # 13 features with position_encoding
        predict_pointers=True,
        predict_expansion_sequence=True,
        use_crf=False,
    ).to(device)

    params = model.get_num_parameters()
    print(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Save metadata
    metadata = {
        "experiment": "B_data_augmentation",
        "args": vars(args),
        "model_params": params,
        "n_original_samples": n_original,
        "n_augmented_samples": len(train_idx),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata: {output_dir / 'metadata.json'}")

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 80)

    history = {
        "train_loss": [],
        "val_loss": [],
        "span_f1": [],
        "span_precision": [],
        "span_recall": [],
    }
    best_f1 = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, args.pos_weight)

        # Validate
        val_metrics = validate_epoch(model, val_loader, device, args.pos_weight)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["val_loss"])
        history["span_f1"].append(val_metrics["span_f1"])
        history["span_precision"].append(val_metrics["span_precision"])
        history["span_recall"].append(val_metrics["span_recall"])

        # Print progress
        print(
            f"Epoch {epoch:2d}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['val_loss']:.4f}, "
            f"F1={val_metrics['span_f1']:.3f}, "
            f"P={val_metrics['span_precision']:.3f}, "
            f"R={val_metrics['span_recall']:.3f}, "
            f"time={time.time() - epoch_start:.1f}s"
        )

        # Save best model
        if val_metrics["span_f1"] > best_f1:
            best_f1 = val_metrics["span_f1"]
            best_model_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "span_f1": best_f1,
                },
                best_model_path,
            )

    total_time = time.time() - start_time

    # Final results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Final F1 (epoch {args.epochs}): {history['span_f1'][-1]:.4f}")
    print()

    # Check if target met
    if best_f1 >= 0.25:
        print("✅ TARGET MET: F1 >= 0.25")
    else:
        print(f"⚠️  TARGET NOT MET: F1 {best_f1:.4f} < 0.25")

    print("=" * 80)

    # Save training curves
    print("\n📊 Generating training curves...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss curves
    axes[0, 0].plot(history["train_loss"], label="Train Loss")
    axes[0, 0].plot(history["val_loss"], label="Val Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # F1 curve
    axes[0, 1].plot(history["span_f1"], label="Validation F1", color="green")
    axes[0, 1].axhline(y=0.25, color="red", linestyle="--", label="Target (0.25)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("F1 Score")
    axes[0, 1].set_title("Span F1 Score")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Precision curve
    axes[1, 0].plot(history["span_precision"], label="Precision", color="blue")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Span Precision")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Recall curve
    axes[1, 1].plot(history["span_recall"], label="Recall", color="orange")
    axes[1, 1].axhline(y=0.40, color="red", linestyle="--", label="Target (0.40)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_title("Span Recall")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plot_path = output_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150)
    print(f"✓ Saved training curves: {plot_path}")

    # Save history CSV
    history_df = pd.DataFrame(history)
    history_df.insert(0, "epoch", range(1, args.epochs + 1))
    history_csv = output_dir / "training_history.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"✓ Saved training history: {history_csv}")

    print("\n✅ Experiment B complete!")


if __name__ == "__main__":
    main()
