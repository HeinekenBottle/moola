#!/usr/bin/env python3
"""Fast Baseline Training - Stones Detection Only (No Countdown).

Optimized for speed with larger batch sizes and GPU utilization.
Comprehensive logging for surgical analysis.

Key changes from baseline_100ep:
- REMOVED countdown task (was causing 91% of loss)
- INCREASED batch size: 32 → 128 (better GPU utilization)
- 3 tasks only: Classification + Pointers + Span

Usage:
    python3 scripts/train_stones_only.py \
        --data data/processed/labeled/train_latest_overlaps_v2.parquet \
        --output artifacts/stones_only \
        --epochs 100 \
        --batch-size 128 \
        --device cuda
"""

import argparse
import json
import sys
import time
from datetime import datetime
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


class MetricsLogger:
    """Comprehensive metrics logging."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epoch_metrics = []
        self.loss_components = []
        self.uncertainty_params = []
        self.probability_stats = []
        self.feature_stats = []
        self.gradient_stats = []

    def log_epoch_metrics(self, epoch: int, metrics: dict):
        """Log overall epoch-level metrics."""
        record = {"epoch": epoch, **metrics}
        self.epoch_metrics.append(record)

    def log_loss_components(
        self, epoch: int, phase: str, loss_type: float, loss_ptr: float, loss_span: float
    ):
        """Log individual loss components (3 tasks only)."""
        self.loss_components.append(
            {
                "epoch": epoch,
                "phase": phase,
                "loss_type": loss_type,
                "loss_ptr": loss_ptr,
                "loss_span": loss_span,
                "total_loss": loss_type + loss_ptr + loss_span,
            }
        )

    def log_uncertainty_params(
        self, epoch: int, sigma_ptr: float, sigma_type: float, sigma_span: float
    ):
        """Log learned uncertainty parameters (3 tasks)."""
        inv_var_ptr = 1.0 / (sigma_ptr**2)
        inv_var_type = 1.0 / (sigma_type**2)
        inv_var_span = 1.0 / (sigma_span**2)
        total_inv_var = inv_var_ptr + inv_var_type + inv_var_span

        self.uncertainty_params.append(
            {
                "epoch": epoch,
                "sigma_ptr": sigma_ptr,
                "sigma_type": sigma_type,
                "sigma_span": sigma_span,
                "weight_ptr": inv_var_ptr / total_inv_var,
                "weight_type": inv_var_type / total_inv_var,
                "weight_span": inv_var_span / total_inv_var,
            }
        )

    def log_probability_stats(
        self, epoch: int, phase: str, in_span_probs: np.ndarray, out_span_probs: np.ndarray
    ):
        """Log probability distribution statistics."""
        self.probability_stats.append(
            {
                "epoch": epoch,
                "phase": phase,
                "in_span_mean": in_span_probs.mean(),
                "in_span_std": in_span_probs.std(),
                "out_span_mean": out_span_probs.mean(),
                "out_span_std": out_span_probs.std(),
                "separation": in_span_probs.mean() - out_span_probs.mean(),
            }
        )

    def log_feature_stats(self, epoch: int, phase: str, features: np.ndarray, feature_names: list):
        """Log per-feature statistics."""
        for feat_idx, feat_name in enumerate(feature_names):
            feat_vals = features[:, :, feat_idx].flatten()
            self.feature_stats.append(
                {
                    "epoch": epoch,
                    "phase": phase,
                    "feature_name": feat_name,
                    "mean": feat_vals.mean(),
                    "std": feat_vals.std(),
                    "min": feat_vals.min(),
                    "max": feat_vals.max(),
                    "median": np.median(feat_vals),
                }
            )

    def log_gradient_stats(self, epoch: int, model):
        """Log gradient norms."""
        total_norm = 0.0
        grad_dict = {"epoch": epoch}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                grad_dict[name] = param_norm
                total_norm += param_norm**2

        grad_dict["total_grad_norm"] = total_norm**0.5
        self.gradient_stats.append(grad_dict)

    def save_all(self):
        """Save all metrics to CSV files."""
        pd.DataFrame(self.epoch_metrics).to_csv(self.output_dir / "epoch_metrics.csv", index=False)
        pd.DataFrame(self.loss_components).to_csv(
            self.output_dir / "loss_components.csv", index=False
        )
        pd.DataFrame(self.uncertainty_params).to_csv(
            self.output_dir / "uncertainty_params.csv", index=False
        )
        pd.DataFrame(self.probability_stats).to_csv(
            self.output_dir / "probability_stats.csv", index=False
        )
        pd.DataFrame(self.feature_stats).to_csv(self.output_dir / "feature_stats.csv", index=False)
        pd.DataFrame(self.gradient_stats).to_csv(
            self.output_dir / "gradient_stats.csv", index=False
        )


def create_expansion_labels(expansion_start, expansion_end, window_length=105):
    """Create soft span mask from pointers (no countdown)."""
    binary_mask = np.zeros(window_length, dtype=np.float32)
    binary_mask[expansion_start : expansion_end + 1] = 1.0
    return binary_mask


class ExpansionDataset(Dataset):
    """Dataset with expansion labels (no countdown)."""

    def __init__(self, df, feature_names):
        self.df = df.reset_index(drop=True)
        self.feature_names = feature_names

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Build 12D features from raw OHLC
        ohlc_arrays = [arr for arr in row["features"]]
        ohlc_df = pd.DataFrame(ohlc_arrays, columns=["open", "high", "low", "close"])

        cfg = RelativityConfig()
        X_12d, _, _ = build_relativity_features(ohlc_df, cfg.model_dump())
        # Take first window explicitly (matches baseline approach)

        # Labels
        label_map = {"consolidation": 0, "retracement": 1}
        label = label_map.get(row["label"], 0)

        # Expansion binary mask
        binary = create_expansion_labels(row["expansion_start"], row["expansion_end"])

        return {
            "features": torch.from_numpy(X_12d[0]).float(),  # (105, 12)
            "label": torch.tensor(label, dtype=torch.long),
            "expansion_start": torch.tensor(row["expansion_start"], dtype=torch.long),
            "expansion_end": torch.tensor(row["expansion_end"], dtype=torch.long),
            "binary_mask": torch.from_numpy(binary).float(),
        }


def train_epoch(model, loader, optimizer, device, logger, epoch):
    """Train for one epoch (3 tasks: type, ptr, span)."""
    model.train()
    loss_components_sum = {"type": 0.0, "ptr": 0.0, "span": 0.0}
    total_loss_sum = 0.0

    all_probs = []
    all_masks = []

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        expansion_start = batch["expansion_start"].to(device)
        expansion_end = batch["expansion_end"].to(device)
        binary_mask = batch["binary_mask"].to(device)

        optimizer.zero_grad()

        output = model(features)

        # Compute individual losses
        loss_type = F.cross_entropy(output["logits"], labels)
        loss_ptr = F.huber_loss(
            output["expansion_center"], (expansion_start + expansion_end) / 2.0, delta=0.08
        )
        loss_ptr += 0.8 * F.huber_loss(
            output["expansion_length"], (expansion_end - expansion_start).float(), delta=0.08
        )
        loss_span = soft_span_loss(output["expansion_probs"], binary_mask)

        loss_components_sum["type"] += loss_type.item()
        loss_components_sum["ptr"] += loss_ptr.item()
        loss_components_sum["span"] += loss_span.item()

        # Uncertainty-weighted loss (3 tasks)
        sigma_type = output.get("sigma_type", torch.tensor(1.0, device=device))
        sigma_ptr = output.get("sigma_ptr", torch.tensor(1.0, device=device))
        sigma_span = output.get("sigma_span", torch.tensor(1.0, device=device))

        loss = (
            (1.0 / (2 * sigma_type**2)) * loss_type
            + torch.log(sigma_type)
            + (1.0 / (2 * sigma_ptr**2)) * loss_ptr
            + torch.log(sigma_ptr)
            + (1.0 / (2 * sigma_span**2)) * loss_span
            + torch.log(sigma_span)
        )

        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item()

        # Collect probabilities for stats
        all_probs.append(output["expansion_probs"].detach().cpu().numpy())
        all_masks.append(binary_mask.cpu().numpy())

    # Average losses
    n_batches = len(loader)
    avg_loss = total_loss_sum / n_batches
    avg_loss_components = {k: v / n_batches for k, v in loss_components_sum.items()}

    # Log loss components
    logger.log_loss_components(
        epoch,
        "train",
        avg_loss_components["type"],
        avg_loss_components["ptr"],
        avg_loss_components["span"],
    )

    # Probability stats
    all_probs = np.concatenate(all_probs, axis=0)  # (N, 105)
    all_masks = np.concatenate(all_masks, axis=0)

    in_span_probs = all_probs[all_masks == 1]
    out_span_probs = all_probs[all_masks == 0]

    logger.log_probability_stats(epoch, "train", in_span_probs, out_span_probs)

    return avg_loss


def eval_epoch(model, loader, device, logger, epoch, feature_names):
    """Evaluate for one epoch."""
    model.eval()
    loss_components_sum = {"type": 0.0, "ptr": 0.0, "span": 0.0}
    total_loss_sum = 0.0

    all_probs = []
    all_masks = []
    all_features = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            expansion_start = batch["expansion_start"].to(device)
            expansion_end = batch["expansion_end"].to(device)
            binary_mask = batch["binary_mask"].to(device)

            output = model(features)

            loss_type = F.cross_entropy(output["logits"], labels)
            loss_ptr = F.huber_loss(
                output["expansion_center"], (expansion_start + expansion_end) / 2.0, delta=0.08
            )
            loss_ptr += 0.8 * F.huber_loss(
                output["expansion_length"], (expansion_end - expansion_start).float(), delta=0.08
            )
            loss_span = soft_span_loss(output["expansion_probs"], binary_mask)

            loss_components_sum["type"] += loss_type.item()
            loss_components_sum["ptr"] += loss_ptr.item()
            loss_components_sum["span"] += loss_span.item()

            total_loss_sum += (loss_type + loss_ptr + loss_span).item()

            all_probs.append(output["expansion_probs"].cpu().numpy())
            all_masks.append(binary_mask.cpu().numpy())
            all_features.append(features.cpu().numpy())

    n_batches = len(loader)
    avg_loss = total_loss_sum / n_batches
    avg_loss_components = {k: v / n_batches for k, v in loss_components_sum.items()}

    logger.log_loss_components(
        epoch,
        "val",
        avg_loss_components["type"],
        avg_loss_components["ptr"],
        avg_loss_components["span"],
    )

    # Compute span F1
    all_probs = np.concatenate(all_probs, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    all_features = np.concatenate(all_features, axis=0)

    span_f1, span_precision, span_recall = compute_span_metrics(all_probs, all_masks, threshold=0.5)

    in_span_probs = all_probs[all_masks == 1]
    out_span_probs = all_probs[all_masks == 0]

    logger.log_probability_stats(epoch, "val", in_span_probs, out_span_probs)

    # Feature stats (every 5 epochs)
    if epoch % 5 == 0:
        logger.log_feature_stats(epoch, "val", all_features, feature_names)

    return avg_loss, span_f1, span_precision, span_recall


def main():
    parser = argparse.ArgumentParser(description="Fast Stones-Only Training")
    parser.add_argument("--data", type=str, required=True, help="Path to training data parquet")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples (for debugging)")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size (increased from 32)"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    print("=" * 80)
    print("STONES-ONLY TRAINING (NO COUNTDOWN) - OPTIMIZED FOR SPEED")
    print("=" * 80)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} (4x larger for better GPU utilization)")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print()

    # Load data
    print("Loading data...")
    df = pd.read_parquet(args.data)

    if args.max_samples:
        df = df.head(args.max_samples)

    print(f"Total samples: {len(df)}")

    # Split
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df["label"]
    )
    print(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples")
    print()

    # Feature names
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

    # Datasets
    train_dataset = ExpansionDataset(train_df, feature_names)
    val_dataset = ExpansionDataset(val_df, feature_names)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    print("Creating model...")
    model = JadeCompact(
        input_size=12,
        hidden_size=128,
        num_layers=2,
        dropout=0.7,
        input_dropout=0.3,
        predict_pointers=True,  # Enable pointer heads for expansion boundaries
        predict_expansion_sequence=True,  # Enable span prediction
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "model_params": {"total": total_params, "trainable": trainable_params},
        "dataset_size": len(df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "feature_names": feature_names,
        "note": "Countdown task REMOVED (was 91% of loss in baseline_100ep)",
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {output_dir / 'metadata.json'}")
    print()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Logger
    logger = MetricsLogger(output_dir)

    # Training loop
    print(f"Training for {args.epochs} epochs...")
    print("-" * 80)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device, logger, epoch)
        val_loss, span_f1, span_precision, span_recall = eval_epoch(
            model, val_loader, device, logger, epoch, feature_names
        )

        epoch_time = time.time() - epoch_start

        # Log epoch metrics
        logger.log_epoch_metrics(
            epoch,
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "span_f1": span_f1,
                "span_precision": span_precision,
                "span_recall": span_recall,
                "epoch_time": epoch_time,
            },
        )

        # Log uncertainty params (every 5 epochs)
        if epoch % 5 == 0:
            with torch.no_grad():
                dummy_input = torch.randn(1, 105, 12).to(device)
                dummy_output = model(dummy_input)
                sigma_ptr = dummy_output.get("sigma_ptr", torch.tensor(1.0)).item()
                sigma_type = dummy_output.get("sigma_type", torch.tensor(1.0)).item()
                sigma_span = dummy_output.get("sigma_span", torch.tensor(1.0)).item()
                logger.log_uncertainty_params(epoch, sigma_ptr, sigma_type, sigma_span)

        # Log gradient stats (every 10 epochs)
        if epoch % 10 == 0:
            logger.log_gradient_stats(epoch, model)

        print(
            f"Epoch {epoch:3d}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"span_F1={span_f1:.3f}, P={span_precision:.3f}, R={span_recall:.3f}, "
            f"time={epoch_time:.1f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss},
                output_dir / "best_model.pt",
            )

        # Save checkpoints every 10 epochs
        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss},
                output_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    total_time = time.time() - start_time
    print("-" * 80)
    print(f"Training complete! Total time: {total_time / 60:.1f} minutes")
    print()

    # Save all metrics
    print("Saving metrics...")
    logger.save_all()
    print(f"✓ Saved all metrics to {output_dir}")
    print()

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
