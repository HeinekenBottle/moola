"""Audit pre-trained Jade encoder quality before fine-tuning.

Validates reconstruction quality with feature-specific MAE thresholds:
- Normalized features [0,1]: MAE < 0.01
- Range features [-3,3]: MAE < 0.03
- Weighted average: MAE < 0.02

Also computes MC Dropout uncertainty (50 passes) for calibration (ECE < 0.10).

Usage:
    python3 scripts/audit_pretrained_encoder.py \
        --checkpoint artifacts/jade_pretrain_20ep/checkpoint_best.pt \
        --data data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet \
        --n-samples 1000
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

from moola.data.windowed_loader import WindowedConfig, WindowedDataset
from moola.features.relativity import build_relativity_features, RelativityConfig
from moola.models.jade_pretrain import JadeConfig, JadePretrainer
from rich.progress import track

console = Console()


# Feature specifications (from relativity.py)
FEATURE_SPECS = {
    # Normalized features [0,1]
    "open_norm": {"range": [0, 1], "mae_threshold": 0.01, "type": "normalized"},
    "close_norm": {"range": [0, 1], "mae_threshold": 0.01, "type": "normalized"},
    "upper_wick_pct": {"range": [0, 1], "mae_threshold": 0.01, "type": "normalized"},
    "lower_wick_pct": {"range": [0, 1], "mae_threshold": 0.01, "type": "normalized"},
    # Signed features [-1,1]
    "body_pct": {"range": [-1, 1], "mae_threshold": 0.01, "type": "normalized"},
    # Range features [0,3] or [-3,3]
    "range_z": {"range": [0, 3], "mae_threshold": 0.03, "type": "range"},
    "dist_to_prev_SH": {"range": [-3, 3], "mae_threshold": 0.03, "type": "range"},
    "dist_to_prev_SL": {"range": [-3, 3], "mae_threshold": 0.03, "type": "range"},
    "bars_since_SH_norm": {"range": [0, 3], "mae_threshold": 0.03, "type": "range"},
    "bars_since_SL_norm": {"range": [0, 3], "mae_threshold": 0.03, "type": "range"},
    "expansion_proxy": {"range": [-2, 2], "mae_threshold": 0.02, "type": "range"},
}

FEATURE_NAMES = list(FEATURE_SPECS.keys())
WEIGHTED_MAE_THRESHOLD = 0.02


def load_checkpoint(checkpoint_path: str) -> Tuple[JadePretrainer, Dict]:
    """Load pre-trained model checkpoint."""
    console.print(f"[blue]Loading checkpoint: {checkpoint_path}[/blue]")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Reconstruct config
    model_config = checkpoint.get("model_config", {})
    config = JadeConfig(
        input_size=model_config.get("input_size", 12),  # Default to 12 (new consol_proxy)
        hidden_size=model_config.get("hidden_size", 128),
        num_layers=model_config.get("num_layers", 2),
        dropout=model_config.get("dropout", 0.65),
    )

    # Create model and load weights
    model = JadePretrainer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Extract metadata
    metadata = {
        "epoch": checkpoint.get("epoch", -1),
        "val_loss": checkpoint.get("val_loss", -1.0),
        "learned_sigma": torch.exp(model.log_sigma).item(),
    }

    console.print(f"[green]✓ Loaded epoch {metadata['epoch']}, val_loss={metadata['val_loss']:.6f}[/green]")
    console.print(f"[green]✓ Learned σ = {metadata['learned_sigma']:.4f}[/green]")

    return model, metadata


def sample_contiguous_windows(
    df: pd.DataFrame, n_samples: int = 1000, temporal_mix: float = 0.2, window_len: int = 105
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample contiguous 105-bar windows by start indices for temporal integrity.

    Args:
        df: Full OHLC DataFrame (must be contiguous in time)
        n_samples: Number of windows to sample
        temporal_mix: Fraction of samples from recent period (2025)
        window_len: Window length (default 105)

    Returns:
        X: Features tensor [n_samples, window_len, 11]
        valid_mask: Valid mask [n_samples, window_len]
    """
    # Era splits (df assumed sorted by time)
    recent_mask = df.index.year >= 2025
    train_end = np.where(recent_mask)[0]
    train_end = train_end[0] if len(train_end) > 0 else len(df)

    n_train_era = int(n_samples * (1 - temporal_mix))
    n_recent = n_samples - n_train_era

    # Valid start indices
    starts_train = np.arange(0, max(0, train_end - window_len + 1))
    starts_recent = np.arange(train_end, max(train_end, len(df) - window_len + 1))

    # Sample (with replacement if needed)
    if len(starts_train) >= n_train_era:
        starts_train_sample = np.random.choice(starts_train, n_train_era, replace=False)
    else:
        starts_train_sample = np.random.choice(starts_train, n_train_era, replace=True)

    if len(starts_recent) >= n_recent:
        starts_recent_sample = np.random.choice(starts_recent, n_recent, replace=False)
    else:
        starts_recent_sample = np.random.choice(starts_recent, len(starts_recent), replace=True)
        # Pad with train-era if needed
        extra = n_recent - len(starts_recent_sample)
        starts_train_sample = np.append(starts_train_sample, np.random.choice(starts_train, extra, replace=False))
        n_train_era += extra  # Adjust count

    all_starts = np.concatenate([starts_train_sample, starts_recent_sample])[:n_samples]
    np.random.shuffle(all_starts)  # Random order, but contiguous windows

    # Extract windows + features
    X_list, valid_mask_list = [], []
    relativity_cfg = RelativityConfig()
    for start in track(all_starts, description="Extracting contiguous windows"):
        if start + window_len > len(df):
            continue  # Edge case
        window_df = df.iloc[start:start + window_len][["open", "high", "low", "close"]].copy()
        X_12d, valid_mask, _ = build_relativity_features(window_df, relativity_cfg.dict())
        # Ensure proper array indexing and conversion to tensor
        X_np = X_12d[0] if (isinstance(X_12d, np.ndarray) and X_12d.ndim > 2) else X_12d
        mask_np = valid_mask[0] if (isinstance(valid_mask, np.ndarray) and valid_mask.ndim > 1) else valid_mask
        X_list.append(torch.as_tensor(X_np, dtype=torch.float32))  # [window_len, 12]
        valid_mask_list.append(torch.as_tensor(mask_np, dtype=torch.float32))  # [window_len]

    X = torch.stack(X_list)
    valid_mask = torch.stack(valid_mask_list)

    console.print(f"[blue]✓ {len(X)} contiguous windows: ~{n_train_era} train-era, ~{n_recent} recent[/blue]")
    return X, valid_mask


def compute_reconstruction_metrics(
    model: JadePretrainer,
    X: torch.Tensor,
    valid_mask: torch.Tensor,
    device: str = "cpu"
) -> Dict[str, float]:
    """Compute reconstruction metrics on validation data.

    Args:
        model: Pre-trained JadePretrainer
        X: Features [batch, K, D=11]
        valid_mask: Valid positions [batch, K]
        device: Device to run on

    Returns:
        Dict of metrics (per-feature MAE, weighted MAE, per-feature std, etc.)
    """
    model = model.to(device)
    X = X.to(device)
    valid_mask = valid_mask.to(device).bool()  # Convert to boolean for indexing

    with torch.no_grad():
        # Forward pass
        encoded, _ = model.encoder(X)
        reconstructed = model.decoder(encoded)

        # Extract valid positions
        valid_X = X[valid_mask]  # [n_valid, D]
        valid_recon = reconstructed[valid_mask]  # [n_valid, D]

        # Compute per-feature MAE
        per_feature_mae = (valid_recon - valid_X).abs().mean(dim=0).cpu().numpy()  # [D]

        # Compute per-feature std (reconstruction variance)
        per_feature_std = (valid_recon - valid_X).std(dim=0).cpu().numpy()  # [D]

        # Weighted MAE (normalized features weight=1, range features weight=1.5)
        feature_weights = torch.tensor(
            [1.0 if FEATURE_SPECS[name]["type"] == "normalized" else 1.5
             for name in FEATURE_NAMES],
            device=device
        )
        weighted_mae = ((valid_recon - valid_X).abs() * feature_weights).mean().item()

        # Non-zero activations (check for dead neurons)
        encoder_activations = encoded.abs().mean(dim=1)  # [batch, hidden*2]
        non_zero_ratio = (encoder_activations > 1e-6).float().mean().item()

    return {
        "per_feature_mae": per_feature_mae.tolist(),
        "per_feature_std": per_feature_std.tolist(),
        "weighted_mae": weighted_mae,
        "non_zero_activation_ratio": non_zero_ratio,
    }


def compute_mc_dropout_ece(
    model: JadePretrainer,
    X: torch.Tensor,
    valid_mask: torch.Tensor,
    n_passes: int = 50,
    device: str = "cpu"
) -> Dict[str, float]:
    """Compute MC Dropout uncertainty and Expected Calibration Error.

    Args:
        model: Pre-trained JadePretrainer
        X: Features [batch, K, D=11]
        valid_mask: Valid positions [batch, K]
        n_passes: Number of MC forward passes
        device: Device to run on

    Returns:
        Dict with mean_variance, epistemic_uncertainty, ece
    """
    model = model.to(device)
    X = X.to(device)
    valid_mask_bool = valid_mask.to(device).bool()  # Convert to boolean for indexing

    # Run MC Dropout (note: model.compute_mc_dropout_uncertainty expects train mode)
    mean_recon, variance = model.compute_mc_dropout_uncertainty(
        X, valid_mask.cpu().numpy(), n_passes=n_passes, dropout_rate=0.1
    )

    # Extract valid positions
    valid_variance = variance[valid_mask_bool.cpu()]  # [n_valid, D]

    # Epistemic uncertainty (mean variance across features)
    epistemic_uncertainty = valid_variance.mean().item()

    # Simple ECE approximation (variance should correlate with error)
    # Low variance → high confidence → should have low error
    # High variance → low confidence → allowed higher error
    with torch.no_grad():
        encoded, _ = model.encoder(X)
        reconstructed = model.decoder(encoded)
        errors = (reconstructed - X).abs()[valid_mask_bool]  # [n_valid, D]

        # Bin by variance percentiles (move everything to CPU for indexing)
        variance_flat = valid_variance.flatten().cpu()
        error_flat = errors.cpu().flatten()

        # Sort by variance, compute error in bins
        sorted_indices = torch.argsort(variance_flat)
        n_bins = 10
        bin_size = len(sorted_indices) // n_bins

        ece = 0.0
        for i in range(n_bins):
            bin_indices = sorted_indices[i * bin_size : (i + 1) * bin_size]
            bin_variance = variance_flat[bin_indices].mean()
            bin_error = error_flat[bin_indices].mean()

            # ECE: |confidence - accuracy| where confidence = 1 - normalized_variance
            confidence = 1.0 - torch.clamp(bin_variance / 0.1, 0, 1)  # Normalize by expected max
            accuracy = 1.0 - torch.clamp(bin_error / 0.1, 0, 1)
            ece += (confidence - accuracy).abs().item()

        ece /= n_bins

    return {
        "mean_variance": valid_variance.mean().item(),
        "epistemic_uncertainty": epistemic_uncertainty,
        "ece": ece,
    }


def print_results(metrics: Dict, mc_metrics: Dict, metadata: Dict):
    """Print audit results in formatted table."""
    console.print("\n[bold cyan]═══ Pre-Training Audit Results ═══[/bold cyan]\n")

    # Checkpoint metadata
    console.print(f"[bold]Checkpoint Metadata:[/bold]")
    console.print(f"  Epoch: {metadata['epoch']}")
    console.print(f"  Val Loss: {metadata['val_loss']:.6f}")
    console.print(f"  Learned σ: {metadata['learned_sigma']:.4f} {'✓' if 0.5 <= metadata['learned_sigma'] <= 2.0 else '⚠️'}")

    # Per-feature reconstruction
    console.print("\n[bold]Per-Feature Reconstruction:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Feature", style="cyan", width=20)
    table.add_column("MAE", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Status", justify="center")

    for i, feature_name in enumerate(FEATURE_NAMES):
        mae = metrics["per_feature_mae"][i]
        std = metrics["per_feature_std"][i]
        threshold = FEATURE_SPECS[feature_name]["mae_threshold"]

        status = "✓" if mae < threshold and std < 0.05 else "⚠️"

        table.add_row(
            feature_name,
            f"{mae:.4f}",
            f"{threshold:.4f}",
            f"{std:.4f}",
            status
        )

    console.print(table)

    # Overall metrics
    console.print("\n[bold]Overall Metrics:[/bold]")
    weighted_mae_status = "✓" if metrics["weighted_mae"] < WEIGHTED_MAE_THRESHOLD else "⚠️"
    non_zero_status = "✓" if metrics["non_zero_activation_ratio"] > 0.5 else "⚠️"
    ece_status = "✓" if mc_metrics["ece"] < 0.10 else "⚠️"

    console.print(f"  Weighted MAE: {metrics['weighted_mae']:.4f} (threshold: {WEIGHTED_MAE_THRESHOLD}) {weighted_mae_status}")
    console.print(f"  Non-zero Activations: {metrics['non_zero_activation_ratio']:.2%} (threshold: 50%) {non_zero_status}")
    console.print(f"  Epistemic Uncertainty: {mc_metrics['epistemic_uncertainty']:.4f}")
    console.print(f"  ECE (calibration): {mc_metrics['ece']:.4f} (threshold: 0.10) {ece_status}")

    # Pass/fail summary
    all_passed = (
        weighted_mae_status == "✓"
        and non_zero_status == "✓"
        and ece_status == "✓"
        and 0.5 <= metadata['learned_sigma'] <= 2.0
    )

    if all_passed:
        console.print("\n[bold green]✓ Encoder quality: PASS - Ready for fine-tuning[/bold green]")
    else:
        console.print("\n[bold yellow]⚠️ Encoder quality: MARGINAL - Review failed metrics before fine-tuning[/bold yellow]")


def main():
    parser = argparse.ArgumentParser(description="Audit pre-trained Jade encoder")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--data", required=True, help="Path to raw OHLC parquet")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of validation windows")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mc-passes", type=int, default=50, help="MC Dropout passes")

    args = parser.parse_args()

    # Load checkpoint
    model, metadata = load_checkpoint(args.checkpoint)

    # Load data and sample contiguous windows
    console.print(f"[blue]Loading data: {args.data}[/blue]")
    df = pd.read_parquet(args.data)
    X, valid_mask = sample_contiguous_windows(df, args.n_samples)

    console.print(f"[blue]Computing reconstruction on {len(X)} windows...[/blue]")

    # Compute metrics
    metrics = compute_reconstruction_metrics(model, X, valid_mask, args.device)

    console.print(f"[blue]Computing MC Dropout uncertainty ({args.mc_passes} passes)...[/blue]")
    mc_metrics = compute_mc_dropout_ece(model, X, valid_mask, args.mc_passes, args.device)

    # Print results
    print_results(metrics, mc_metrics, metadata)

    # Save to JSON
    output_path = Path(args.checkpoint).parent / "audit_results.json"
    results = {
        "checkpoint": str(args.checkpoint),
        "metadata": metadata,
        "reconstruction_metrics": metrics,
        "mc_dropout_metrics": mc_metrics,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]✓ Saved results to {output_path}[/green]")


if __name__ == "__main__":
    main()
