"""Quick local test of expansion-focused training with soft span loss.

Test on small dataset before full RunPod deployment.

Changes from binary BCE:
- Uses soft_span_loss for gradient-balanced span prediction
- Computes span F1 metrics during validation
- Generates probability histograms for diagnostics
- Supports optional CRF mode via --use-crf flag
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from moola.features.relativity import RelativityConfig, build_relativity_features
from moola.models.jade_core import (
    JadeCompact,
    compute_span_f1,
    compute_span_metrics,
    crf_span_loss,
    soft_span_loss,
)


class LossNormalizer:
    """Normalize losses by running mean for fair multi-task weighting."""

    def __init__(self, momentum=0.95, warmup_steps=10):
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.running_means = {}

    def normalize(self, losses):
        normalized = {}
        self.step_count += 1

        for name, loss in losses.items():
            loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss

            if name not in self.running_means:
                self.running_means[name] = loss_value

            if self.step_count > self.warmup_steps:
                self.running_means[name] = (
                    self.momentum * self.running_means[name] + (1 - self.momentum) * loss_value
                )
            else:
                self.running_means[name] = (
                    self.running_means[name] * (self.step_count - 1) + loss_value
                ) / self.step_count

            mean = self.running_means[name]
            normalized[name] = loss / mean if mean > 1e-8 else loss

        return normalized


def create_expansion_labels(expansion_start, expansion_end, window_length=105):
    """Create soft span mask and countdown from pointers.

    Note: Binary mask is kept as 0/1 for now, but soft_span_loss supports
    soft targets in [0, 1] for future use (e.g., weighted by annotation quality).
    """
    # Binary mask (can be soft in future: 0.2, 0.8, 0.9, etc.)
    binary_mask = np.zeros(window_length, dtype=np.float32)
    binary_mask[expansion_start : expansion_end + 1] = 1.0

    # Countdown (regression target)
    countdown = np.arange(window_length, dtype=np.float32) - expansion_start
    countdown = -countdown
    countdown = np.clip(countdown, -20, 20)  # Clip to prevent MSE explosion

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


def train_epoch(model, loader, optimizer, normalizer, device, use_crf=False):
    """Train one epoch with normalized multi-task loss."""
    model.train()
    total_loss = 0

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

        # Soft span loss (replaces binary_cross_entropy_with_logits)
        if use_crf:
            # CRF mode: Use CRF negative log-likelihood
            emissions = output["span_emissions"]
            target_tags = (binary > 0.5).long()  # Convert to tag indices
            loss_span = crf_span_loss(model, emissions, target_tags)
        else:
            # Soft span mode: Use soft_span_loss with sigmoid probabilities
            pred_probs = output["expansion_binary"]  # Already sigmoid
            loss_span = soft_span_loss(pred_probs, binary, reduction="mean")

        loss_countdown = F.huber_loss(output["expansion_countdown"], countdown, delta=1.0)

        # Uncertainty weighting (Kendall et al., CVPR 2018)
        # L_total = (1/2σ_ptr²)L_ptr + (1/2σ_type²)L_type + (1/2σ_span²)L_span + (1/2σ_countdown²)L_countdown + log(σ_ptr × σ_type × σ_span × σ_countdown)
        # This auto-balances task importance by learning optimal σ parameters

        # Get learned uncertainty parameters from model
        sigma_ptr = output["sigma_ptr"]
        sigma_type = output["sigma_type"]
        sigma_span = output.get("sigma_span", torch.tensor(1.0, device=device))
        sigma_countdown = output.get("sigma_countdown", torch.tensor(1.0, device=device))

        # Uncertainty-weighted loss (prevents manual tuning of λ weights)
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

    return total_loss / len(loader)


def validate_epoch(model, loader, device, use_crf=False):
    """Validate one epoch with span F1 metrics."""
    model.eval()
    val_loss = 0
    span_f1_scores = []
    span_metrics_list = []

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

            if use_crf:
                emissions = output["span_emissions"]
                target_tags = (binary > 0.5).long()
                loss_span = crf_span_loss(model, emissions, target_tags)
            else:
                pred_probs = output["expansion_binary"]
                loss_span = soft_span_loss(pred_probs, binary, reduction="mean")

            loss_countdown = F.huber_loss(output["expansion_countdown"], countdown, delta=1.0)

            val_loss += (loss_type + loss_ptr + loss_span + loss_countdown).item()

            # Span F1 metrics
            pred_probs = output["expansion_binary"]
            for i in range(pred_probs.size(0)):
                f1 = compute_span_f1(pred_probs[i], binary[i], threshold=0.5)
                span_f1_scores.append(f1)

                metrics = compute_span_metrics(pred_probs[i], binary[i], threshold=0.5)
                span_metrics_list.append(metrics)

    val_loss /= len(loader)
    mean_span_f1 = np.mean(span_f1_scores) if span_f1_scores else 0.0

    # Aggregate span metrics
    if span_metrics_list:
        mean_precision = np.mean([m["precision"] for m in span_metrics_list])
        mean_recall = np.mean([m["recall"] for m in span_metrics_list])
    else:
        mean_precision = 0.0
        mean_recall = 0.0

    return {
        "val_loss": val_loss,
        "span_f1": mean_span_f1,
        "span_precision": mean_precision,
        "span_recall": mean_recall,
    }


def generate_diagnostics(model, loader, device, output_dir, use_crf=False):
    """Generate probability histograms for span predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_pred_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            binary = batch["binary"].to(device)

            output = model(features)
            pred_probs = output["expansion_binary"]

            all_pred_probs.append(pred_probs.cpu().numpy())
            all_targets.append(binary.cpu().numpy())

    all_pred_probs = np.concatenate(all_pred_probs, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    # Separate positive and negative examples
    pred_pos = all_pred_probs[all_targets > 0.5]
    pred_neg = all_pred_probs[all_targets < 0.5]

    # Create histogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(pred_pos, bins=50, alpha=0.6, label="In-span (target=1)", color="green", range=(0, 1))
    ax.hist(pred_neg, bins=50, alpha=0.6, label="Out-of-span (target=0)", color="red", range=(0, 1))
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Span Probability Distribution ({'CRF' if use_crf else 'Soft Span Loss'})")
    ax.legend()
    ax.grid(alpha=0.3)

    mode_str = "crf" if use_crf else "soft"
    fig.savefig(output_dir / f"span_probs_{mode_str}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n✓ Diagnostics saved to {output_dir}")
    print(f"  - Histogram: span_probs_{mode_str}.png")
    print(f"  - In-span mean: {pred_pos.mean():.3f}, Out-of-span mean: {pred_neg.mean():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Local test of expansion-focused training with soft span loss"
    )
    parser.add_argument(
        "--use-crf", action="store_true", help="Enable CRF layer for span prediction"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Max samples to use for quick testing (default: 50)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device: cpu or cuda (default: cpu)"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    args = parser.parse_args()

    print("=" * 80)
    print("EXPANSION-FOCUSED TRAINING - LOCAL TEST")
    print(f"Mode: {'CRF' if args.use_crf else 'Soft Span Loss'}")
    print("=" * 80)

    # Config
    device = args.device
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    # Load data
    print("\nLoading data...")
    dataset = ExpansionDataset(
        "data/processed/labeled/train_latest_overlaps_v2.parquet",
        max_samples=args.max_samples,
    )
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx),
    )

    print(f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

    # Model
    print("\nCreating model...")
    model = JadeCompact(
        input_size=12,
        predict_pointers=True,
        predict_expansion_sequence=True,  # Enable expansion heads
        use_crf=args.use_crf,  # Enable CRF if requested
    ).to(device)

    params = model.get_num_parameters()
    print(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Loss normalizer
    normalizer = LossNormalizer(momentum=0.95, warmup_steps=5)

    # Training
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, normalizer, device, use_crf=args.use_crf
        )

        # Validation with span F1 metrics
        val_metrics = validate_epoch(model, val_loader, device, use_crf=args.use_crf)

        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['val_loss']:.4f}, "
            f"span_F1={val_metrics['span_f1']:.3f}, "
            f"span_P={val_metrics['span_precision']:.3f}, "
            f"span_R={val_metrics['span_recall']:.3f}"
        )

        # Show learned uncertainty parameters on final epoch
        if epoch == epochs - 1:
            print("\nLearned uncertainty parameters (σ):")
            with torch.no_grad():
                # Get sigma values from model (use single sample)
                dummy_batch = next(iter(train_loader))
                dummy_output = model(dummy_batch["features"][:1].to(device))

                sigma_ptr = dummy_output["sigma_ptr"].item()
                sigma_type = dummy_output["sigma_type"].item()
                sigma_span = dummy_output.get("sigma_span", torch.tensor(1.0)).item()
                sigma_countdown = dummy_output.get("sigma_countdown", torch.tensor(1.0)).item()

                print(f"  σ_ptr: {sigma_ptr:.4f}")
                print(f"  σ_type: {sigma_type:.4f}")
                print(f"  σ_span: {sigma_span:.4f}")
                print(f"  σ_countdown: {sigma_countdown:.4f}")

                # Effective task weights (inversely proportional to σ²)
                total_inv_var = (
                    (1 / sigma_ptr**2)
                    + (1 / sigma_type**2)
                    + (1 / sigma_span**2)
                    + (1 / sigma_countdown**2)
                )
                print("\nAuto-learned task weights:")
                print(f"  Pointers: {(1/sigma_ptr**2)/total_inv_var:.1%}")
                print(f"  Classification: {(1/sigma_type**2)/total_inv_var:.1%}")
                print(f"  Span: {(1/sigma_span**2)/total_inv_var:.1%}")
                print(f"  Countdown: {(1/sigma_countdown**2)/total_inv_var:.1%}")

    # Generate diagnostics
    print("\nGenerating diagnostics...")
    generate_diagnostics(model, val_loader, device, "artifacts/diagnostics", use_crf=args.use_crf)

    print("\n" + "=" * 80)
    print("✓ LOCAL TEST COMPLETE")
    print("=" * 80)
    print("\nNext: Deploy to RunPod for full training on 174 samples")
    print(
        f"  - Loss function: {'CRF negative log-likelihood' if args.use_crf else 'Soft span loss'}"
    )
    print("  - Metrics: Span F1, precision, recall computed")
    print("  - Diagnostics: Probability histograms saved")


if __name__ == "__main__":
    main()
