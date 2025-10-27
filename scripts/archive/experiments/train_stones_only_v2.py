#!/usr/bin/env python3
"""Baseline Training Run with Comprehensive Logging - 100 Epochs.

This script runs an extensive baseline training session with detailed metric collection
at all stages to enable surgical analysis of model behavior.

Logging includes:
- Per-epoch metrics (loss components, F1, precision, recall)
- Uncertainty parameter evolution (σ values)
- Probability distribution statistics (in-span vs out-of-span)
- Per-feature statistics (mean, std, min, max)
- Gradient norms (total and per-layer)
- Model checkpoints (every 10 epochs + best model)
- Training metadata (hyperparameters, dataset info, reproducibility seeds)

Usage:
    python3 scripts/train_baseline_100ep.py \
        --data data/processed/labeled/train_latest_overlaps_v2.parquet \
        --output artifacts/baseline_100ep/ \
        --epochs 100 \
        --device cuda
"""

import argparse
import json
import sys
import time
from collections import defaultdict
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
from moola.models.jade_core import (
    JadeCompact,
    compute_span_f1,
    compute_span_metrics,
    soft_span_loss,
)


class MetricsLogger:
    """Comprehensive metrics logging for baseline run."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CSV files for different metric types
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
        self,
        epoch: int,
        phase: str,
        loss_type: float,
        loss_ptr: float,
        loss_span: float,
        loss_countdown: float,
    ):
        """Log individual loss components."""
        self.loss_components.append(
            {
                "epoch": epoch,
                "phase": phase,
                "loss_type": loss_type,
                "loss_ptr": loss_ptr,
                "loss_span": loss_span,
                "loss_countdown": loss_countdown,
                "total_loss": loss_type + loss_ptr + loss_span + loss_countdown,
            }
        )

    def log_uncertainty_params(
        self,
        epoch: int,
        sigma_ptr: float,
        sigma_type: float,
        sigma_span: float,
        sigma_countdown: float,
    ):
        """Log learned uncertainty parameters."""
        # Compute effective task weights
        inv_var_ptr = 1.0 / (sigma_ptr**2)
        inv_var_type = 1.0 / (sigma_type**2)
        inv_var_span = 1.0 / (sigma_span**2)
        inv_var_countdown = 1.0 / (sigma_countdown**2)
        total_inv_var = inv_var_ptr + inv_var_type + inv_var_span + inv_var_countdown

        self.uncertainty_params.append(
            {
                "epoch": epoch,
                "sigma_ptr": sigma_ptr,
                "sigma_type": sigma_type,
                "sigma_span": sigma_span,
                "sigma_countdown": sigma_countdown,
                "weight_ptr": inv_var_ptr / total_inv_var,
                "weight_type": inv_var_type / total_inv_var,
                "weight_span": inv_var_span / total_inv_var,
                "weight_countdown": inv_var_countdown / total_inv_var,
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
                "in_span_min": in_span_probs.min(),
                "in_span_max": in_span_probs.max(),
                "in_span_median": np.median(in_span_probs),
                "out_span_mean": out_span_probs.mean(),
                "out_span_std": out_span_probs.std(),
                "out_span_min": out_span_probs.min(),
                "out_span_max": out_span_probs.max(),
                "out_span_median": np.median(out_span_probs),
                "separation": in_span_probs.mean() - out_span_probs.mean(),
            }
        )

    def log_feature_stats(
        self, epoch: int, phase: str, features: torch.Tensor, feature_names: list
    ):
        """Log per-feature statistics."""
        # features: (batch, 105, 12)
        features_np = features.detach().cpu().numpy()

        for feat_idx, feat_name in enumerate(feature_names):
            feat_values = features_np[:, :, feat_idx].flatten()
            self.feature_stats.append(
                {
                    "epoch": epoch,
                    "phase": phase,
                    "feature_name": feat_name,
                    "mean": feat_values.mean(),
                    "std": feat_values.std(),
                    "min": feat_values.min(),
                    "max": feat_values.max(),
                    "median": np.median(feat_values),
                }
            )

    def log_gradient_stats(self, epoch: int, model: torch.nn.Module):
        """Log gradient norm statistics."""
        total_norm = 0.0
        layer_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                layer_norms[name] = param_norm
                total_norm += param_norm**2

        total_norm = total_norm**0.5

        self.gradient_stats.append(
            {
                "epoch": epoch,
                "total_grad_norm": total_norm,
                **layer_norms,
            }
        )

    def save_all(self):
        """Save all logged metrics to CSV files."""
        if self.epoch_metrics:
            pd.DataFrame(self.epoch_metrics).to_csv(
                self.output_dir / "epoch_metrics.csv", index=False
            )
            print(f"✓ Saved: {self.output_dir / 'epoch_metrics.csv'}")

        if self.loss_components:
            pd.DataFrame(self.loss_components).to_csv(
                self.output_dir / "loss_components.csv", index=False
            )
            print(f"✓ Saved: {self.output_dir / 'loss_components.csv'}")

        if self.uncertainty_params:
            pd.DataFrame(self.uncertainty_params).to_csv(
                self.output_dir / "uncertainty_params.csv", index=False
            )
            print(f"✓ Saved: {self.output_dir / 'uncertainty_params.csv'}")

        if self.probability_stats:
            pd.DataFrame(self.probability_stats).to_csv(
                self.output_dir / "probability_stats.csv", index=False
            )
            print(f"✓ Saved: {self.output_dir / 'probability_stats.csv'}")

        if self.feature_stats:
            pd.DataFrame(self.feature_stats).to_csv(
                self.output_dir / "feature_stats.csv", index=False
            )
            print(f"✓ Saved: {self.output_dir / 'feature_stats.csv'}")

        if self.gradient_stats:
            pd.DataFrame(self.gradient_stats).to_csv(
                self.output_dir / "gradient_stats.csv", index=False
            )
            print(f"✓ Saved: {self.output_dir / 'gradient_stats.csv'}")


def create_expansion_labels(expansion_start, expansion_end, window_length=105):
    """Create soft span mask and countdown from pointers."""
    binary_mask = np.zeros(window_length, dtype=np.float32)
    binary_mask[expansion_start : expansion_end + 1] = 1.0

    countdown = np.arange(window_length, dtype=np.float32) - expansion_start
    countdown = -countdown
    countdown = np.clip(countdown, -20, 20)

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

        # Build 12D features from raw OHLC
        ohlc_arrays = [arr for arr in row["features"]]
        ohlc_df = pd.DataFrame(ohlc_arrays, columns=["open", "high", "low", "close"])

        cfg = RelativityConfig()
        X_12d, _, _ = build_relativity_features(ohlc_df, cfg.model_dump())

        # Labels
        label = self.label_map.get(row["label"], 0)

        # Pointers (normalized to [0, 1])
        center = (row["expansion_start"] + row["expansion_end"]) / 2.0 / 105.0
        length = (row["expansion_end"] - row["expansion_start"]) / 105.0
        pointers = np.array([center, length], dtype=np.float32)

        # Expansion labels
        binary, countdown = create_expansion_labels(row["expansion_start"], row["expansion_end"])

        return {
            "features": torch.from_numpy(X_12d[0]).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "pointers": torch.from_numpy(pointers).float(),
            "binary": torch.from_numpy(binary).float(),
            "countdown": torch.from_numpy(countdown).float(),
        }


def train_epoch(model, loader, optimizer, logger, epoch, device, feature_names):
    """Train one epoch with comprehensive logging."""
    model.train()
    total_loss = 0
    loss_components_sum = defaultdict(float)

    # Collect all features for statistics
    all_features = []

    for batch_idx, batch in enumerate(loader):
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        pointers = batch["pointers"].to(device)
        binary = batch["binary"].to(device)
        countdown = batch["countdown"].to(device)

        # Store features for statistics (every 10 batches to save memory)
        if batch_idx % 10 == 0:
            all_features.append(features.cpu())

        # Forward
        output = model(features)

        # Compute raw losses
        loss_type = F.cross_entropy(output["logits"], labels)
        loss_ptr = F.huber_loss(output["pointers"], pointers, delta=0.08)
        pred_probs = output["expansion_binary"]
        loss_span = soft_span_loss(pred_probs, binary, reduction="mean")
        loss_countdown = F.huber_loss(output["expansion_countdown"], countdown, delta=1.0)

        # Accumulate loss components
        loss_components_sum["type"] += loss_type.item()
        loss_components_sum["ptr"] += loss_ptr.item()
        loss_components_sum["span"] += loss_span.item()
        loss_components_sum["countdown"] += loss_countdown.item()

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

        # Log gradient stats (every 10 epochs)
        if epoch % 10 == 0 and batch_idx == 0:
            logger.log_gradient_stats(epoch, model)

        optimizer.step()

        total_loss += loss.item()

    # Average loss components
    n_batches = len(loader)
    avg_loss_components = {k: v / n_batches for k, v in loss_components_sum.items()}

    # Log loss components
    logger.log_loss_components(
        epoch,
        "train",
        avg_loss_components["type"],
        avg_loss_components["ptr"],
        avg_loss_components["span"],
        avg_loss_components["countdown"],
    )

    # Log feature statistics (every 5 epochs)
    if epoch % 5 == 0 and all_features:
        all_features_tensor = torch.cat(all_features, dim=0)
        logger.log_feature_stats(epoch, "train", all_features_tensor, feature_names)

    return total_loss / n_batches


def validate_epoch(model, loader, logger, epoch, device, feature_names):
    """Validate one epoch with comprehensive logging."""
    model.eval()
    total_loss = 0
    span_f1_scores = []
    span_metrics_list = []
    loss_components_sum = defaultdict(float)

    # Collect probability distributions
    all_in_span_probs = []
    all_out_span_probs = []

    # Collect all features
    all_features = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            pointers = batch["pointers"].to(device)
            binary = batch["binary"].to(device)
            countdown = batch["countdown"].to(device)

            all_features.append(features.cpu())

            output = model(features)

            # Losses
            loss_type = F.cross_entropy(output["logits"], labels)
            loss_ptr = F.huber_loss(output["pointers"], pointers, delta=0.08)
            pred_probs = output["expansion_binary"]
            loss_span = soft_span_loss(pred_probs, binary, reduction="mean")
            loss_countdown = F.huber_loss(output["expansion_countdown"], countdown, delta=1.0)

            loss_components_sum["type"] += loss_type.item()
            loss_components_sum["ptr"] += loss_ptr.item()
            loss_components_sum["span"] += loss_span.item()
            loss_components_sum["countdown"] += loss_countdown.item()

            total_loss += (loss_type + loss_ptr + loss_span + loss_countdown).item()

            # Collect probability distributions
            pred_probs_np = pred_probs.cpu().numpy()
            binary_np = binary.cpu().numpy()

            for i in range(pred_probs_np.shape[0]):
                in_span_mask = binary_np[i] > 0.5
                if in_span_mask.any():
                    all_in_span_probs.append(pred_probs_np[i][in_span_mask])
                if (~in_span_mask).any():
                    all_out_span_probs.append(pred_probs_np[i][~in_span_mask])

            # Span F1 metrics
            for i in range(pred_probs.size(0)):
                f1 = compute_span_f1(pred_probs[i], binary[i], threshold=0.5)
                span_f1_scores.append(f1)

                metrics = compute_span_metrics(pred_probs[i], binary[i], threshold=0.5)
                span_metrics_list.append(metrics)

    # Average metrics
    n_batches = len(loader)
    avg_loss = total_loss / n_batches
    avg_span_f1 = np.mean(span_f1_scores) if span_f1_scores else 0.0

    avg_loss_components = {k: v / n_batches for k, v in loss_components_sum.items()}

    # Log loss components
    logger.log_loss_components(
        epoch,
        "val",
        avg_loss_components["type"],
        avg_loss_components["ptr"],
        avg_loss_components["span"],
        avg_loss_components["countdown"],
    )

    # Log probability statistics
    if all_in_span_probs and all_out_span_probs:
        in_span_probs = np.concatenate(all_in_span_probs)
        out_span_probs = np.concatenate(all_out_span_probs)
        logger.log_probability_stats(epoch, "val", in_span_probs, out_span_probs)

    # Log feature statistics (every 5 epochs)
    if epoch % 5 == 0 and all_features:
        all_features_tensor = torch.cat(all_features, dim=0)
        logger.log_feature_stats(epoch, "val", all_features_tensor, feature_names)

    # Aggregate span metrics
    if span_metrics_list:
        mean_precision = np.mean([m["precision"] for m in span_metrics_list])
        mean_recall = np.mean([m["recall"] for m in span_metrics_list])
    else:
        mean_precision = 0.0
        mean_recall = 0.0

    return {
        "val_loss": avg_loss,
        "span_f1": avg_span_f1,
        "span_precision": mean_precision,
        "span_recall": mean_recall,
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline training with comprehensive logging")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/labeled/train_latest_overlaps_v2.parquet",
        help="Path to labeled data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/baseline_100ep",
        help="Output directory for logs and checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples for testing")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BASELINE TRAINING RUN - 100 EPOCHS WITH COMPREHENSIVE LOGGING")
    print("=" * 80)
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print()

    # Initialize logger
    logger = MetricsLogger(output_dir)

    # Feature names for logging
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

    # Load data
    print("Loading data...")
    dataset = ExpansionDataset(args.data, max_samples=args.max_samples)
    train_idx, val_idx = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=args.seed
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
    )
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx),
    )

    print(f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

    # Model
    print("\nCreating model...")
    model = JadeCompact(
        input_size=12,
        predict_pointers=True,
        predict_expansion_sequence=True,
        use_crf=False,
    ).to(args.device)

    params = model.get_num_parameters()
    print(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "model_params": params,
        "dataset_size": len(dataset),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "feature_names": feature_names,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata: {output_dir / 'metadata.json'}")

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 80)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, logger, epoch, args.device, feature_names
        )

        # Validate
        val_metrics = validate_epoch(model, val_loader, logger, epoch, args.device, feature_names)

        # Log uncertainty parameters (every 5 epochs)
        if epoch % 5 == 0:
            with torch.no_grad():
                dummy_batch = next(iter(train_loader))
                dummy_output = model(dummy_batch["features"][:1].to(args.device))

                sigma_ptr = dummy_output["sigma_ptr"].item()
                sigma_type = dummy_output["sigma_type"].item()
                sigma_span = dummy_output.get("sigma_span", torch.tensor(1.0)).item()
                sigma_countdown = dummy_output.get("sigma_countdown", torch.tensor(1.0)).item()

                logger.log_uncertainty_params(
                    epoch, sigma_ptr, sigma_type, sigma_span, sigma_countdown
                )

        # Log epoch metrics
        logger.log_epoch_metrics(
            epoch,
            {
                "train_loss": train_loss,
                **val_metrics,
                "epoch_time": time.time() - epoch_start,
            },
        )

        # Print progress
        print(
            f"Epoch {epoch:3d}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['val_loss']:.4f}, "
            f"span_F1={val_metrics['span_f1']:.3f}, "
            f"span_P={val_metrics['span_precision']:.3f}, "
            f"span_R={val_metrics['span_recall']:.3f}, "
            f"time={time.time() - epoch_start:.1f}s"
        )

        # Save checkpoint (every 10 epochs + best model)
        if epoch % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_metrics["val_loss"],
                },
                checkpoint_path,
            )
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_model_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                best_model_path,
            )

    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save all logged metrics
    print("\n💾 Saving metrics...")
    logger.save_all()

    print("\n📊 Generated artifacts:")
    print(f"  - Metrics CSVs: {output_dir}")
    print(f"  - Best model: {output_dir / 'best_model.pt'}")
    print(f"  - Checkpoints: {output_dir / 'checkpoint_epoch_*.pt'}")
    print(f"  - Metadata: {output_dir / 'metadata.json'}")

    print("\n✅ Baseline established! Ready for analysis.")


if __name__ == "__main__":
    main()
