"""Fine-tune position encoding baseline with CRF layer for span detection.

This script:
1. Loads best position encoding checkpoint (100 epochs, 13D features)
2. Initializes JadeCompact with CRF enabled (use_crf=True)
3. Fine-tunes for 20 epochs with learning rate 1e-4
4. Tracks F1 and IoU metrics for span boundaries
5. Saves best model and comprehensive metrics

Expected Performance:
- Baseline (no CRF): F1 = 0.2196 (position encoding, 100 epochs)
- Target (with CRF): F1 > 0.25 (CRF should improve contiguous span detection)

Usage:
    python3 scripts/finetune_position_crf.py

Output:
    - artifacts/position_crf_finetuned/best_model.pt
    - artifacts/position_crf_finetuned/metrics.csv
    - artifacts/position_crf_finetuned/checkpoint_epoch_{5,10,15,20}.pt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add moola to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moola.models.jade_core import JadeCompact, compute_span_metrics, crf_span_loss

# ============================================================================
# IoU Metric
# ============================================================================


def compute_iou(
    pred_binary: torch.Tensor, target_binary: torch.Tensor, threshold: float = 0.5
) -> float:
    """Compute IoU (Intersection over Union) for span predictions.

    Args:
        pred_binary: Predicted probabilities, shape (batch, seq_len)
        target_binary: Target binary labels, shape (batch, seq_len)
        threshold: Threshold for converting predictions to binary

    Returns:
        Mean IoU across all windows
    """
    # Convert to binary masks
    pred_mask = (pred_binary > threshold).float()
    target_mask = target_binary.float()

    # Compute IoU for each window
    intersection = (pred_mask * target_mask).sum(dim=1)  # (batch,)
    union = ((pred_mask + target_mask) > 0).float().sum(dim=1)  # (batch,)

    # Avoid division by zero
    iou = intersection / (union + 1e-7)

    return iou.mean().item()


def compute_batch_iou_per_sample(
    pred_binary: torch.Tensor, target_binary: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Compute IoU per sample (for detailed tracking).

    Args:
        pred_binary: Predicted probabilities, shape (batch, seq_len)
        target_binary: Target binary labels, shape (batch, seq_len)
        threshold: Threshold for converting predictions to binary

    Returns:
        IoU per sample, shape (batch,)
    """
    pred_mask = (pred_binary > threshold).float()
    target_mask = target_binary.float()

    intersection = (pred_mask * target_mask).sum(dim=1)
    union = ((pred_mask + target_mask) > 0).float().sum(dim=1)

    iou = intersection / (union + 1e-7)
    return iou


# ============================================================================
# Data Loading
# ============================================================================


def load_training_data(data_path: Path):
    """Load training data with 13D features (12 base + position encoding).

    Returns:
        X_train: (n_samples, 105, 13)
        y_train: (n_samples,) - pattern type labels
        expansion_binary_train: (n_samples, 105) - binary span labels
        expansion_start_train: (n_samples,) - expansion start indices
        expansion_end_train: (n_samples,) - expansion end indices
    """
    import pandas as pd

    df = pd.read_parquet(data_path)

    # Extract features (12D base features)
    feature_cols = [
        "open_norm",
        "high_norm",
        "low_norm",
        "close_norm",
        "range_z",
        "body_z",
        "dist_to_SH",
        "dist_to_SL",
        "swing_high_proxy",
        "swing_low_proxy",
        "expansion_proxy",
        "consol_proxy",
    ]

    n_samples = len(df)
    window_length = 105

    # Initialize 13D features (12 base + position encoding)
    X = np.zeros((n_samples, window_length, 13), dtype=np.float32)

    # Load 12D base features
    for i, row in enumerate(df.itertuples()):
        for j, col in enumerate(feature_cols):
            feature_array = getattr(row, col)
            if isinstance(feature_array, (list, np.ndarray)):
                X[i, :, j] = feature_array
            else:
                # Single value repeated
                X[i, :, j] = feature_array

    # Add position encoding as 13th feature
    position_encoding = np.linspace(0, 1, window_length).astype(np.float32)
    X[:, :, 12] = position_encoding  # Broadcast to all samples

    # Extract labels
    y = df["target"].values.astype(np.int64)

    # Create binary expansion masks from expansion_start and expansion_end
    expansion_start = df["expansion_start"].values
    expansion_end = df["expansion_end"].values

    expansion_binary = np.zeros((n_samples, window_length), dtype=np.float32)
    for i in range(n_samples):
        start = int(expansion_start[i])
        end = int(expansion_end[i])
        expansion_binary[i, start : end + 1] = 1.0

    return X, y, expansion_binary, expansion_start, expansion_end


# ============================================================================
# Training Loop
# ============================================================================


def train_one_epoch(model, train_loader, optimizer, device, use_crf=True):
    """Train for one epoch with CRF loss."""
    model.train()
    total_loss = 0.0
    total_span_loss = 0.0
    total_ptr_loss = 0.0
    total_type_loss = 0.0

    for batch_idx, (X_batch, y_batch, span_batch, ptr_batch) in enumerate(train_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        span_batch = span_batch.to(device)
        ptr_batch = ptr_batch.to(device)  # (batch, 2) - [center, length]

        optimizer.zero_grad()

        # Forward pass
        output = model(X_batch)

        # 1. CRF loss for span detection
        if use_crf:
            emissions = output["span_emissions"]  # (batch, 105, 2)
            target_tags = (span_batch > 0.5).long()  # Convert to binary tags (0/1)
            span_loss = crf_span_loss(model, emissions, target_tags)
        else:
            # Fallback: soft span loss (shouldn't be used if CRF is enabled)
            raise ValueError("CRF must be enabled for this training script")

        # 2. Pointer loss (Huber, δ=0.08)
        pointers = output["pointers"]  # (batch, 2) - [center, length]
        ptr_loss_fn = nn.HuberLoss(delta=0.08)
        ptr_loss = ptr_loss_fn(pointers, ptr_batch)

        # 3. Type loss (CrossEntropy)
        logits = output["logits"]  # (batch, num_classes)
        type_loss_fn = nn.CrossEntropyLoss()
        type_loss = type_loss_fn(logits, y_batch)

        # 4. Uncertainty weighting
        # L_total = (1/2σ_span²)L_span + (1/2σ_ptr²)L_ptr + (1/2σ_type²)L_type + log(σ_span × σ_ptr × σ_type)
        sigma_span = output["sigma_span"]
        sigma_ptr = output["sigma_ptr"]
        sigma_type = output["sigma_type"]

        weighted_span_loss = 0.5 * span_loss / (sigma_span**2)
        weighted_ptr_loss = 0.5 * ptr_loss / (sigma_ptr**2)
        weighted_type_loss = 0.5 * type_loss / (sigma_type**2)
        uncertainty_reg = torch.log(sigma_span * sigma_ptr * sigma_type)

        total_batch_loss = (
            weighted_span_loss + weighted_ptr_loss + weighted_type_loss + uncertainty_reg
        )

        # Backward pass
        total_batch_loss.backward()
        optimizer.step()

        # Track losses
        total_loss += total_batch_loss.item()
        total_span_loss += span_loss.item()
        total_ptr_loss += ptr_loss.item()
        total_type_loss += type_loss.item()

    n_batches = len(train_loader)
    return {
        "train_loss": total_loss / n_batches,
        "train_span_loss": total_span_loss / n_batches,
        "train_ptr_loss": total_ptr_loss / n_batches,
        "train_type_loss": total_type_loss / n_batches,
    }


def validate(model, val_loader, device, use_crf=True):
    """Validate and compute F1 + IoU metrics."""
    model.eval()
    total_loss = 0.0
    total_span_loss = 0.0
    total_ptr_loss = 0.0
    total_type_loss = 0.0

    all_pred_spans = []
    all_target_spans = []
    all_pred_probs = []

    with torch.no_grad():
        for X_batch, y_batch, span_batch, ptr_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            span_batch = span_batch.to(device)
            ptr_batch = ptr_batch.to(device)

            # Forward pass
            output = model(X_batch)

            # 1. CRF loss
            if use_crf:
                emissions = output["span_emissions"]
                target_tags = (span_batch > 0.5).long()
                span_loss = crf_span_loss(model, emissions, target_tags)

                # Get Viterbi decoded spans (already done in forward during inference)
                pred_spans = output["expansion_spans"]  # (batch, 105) with 0/1
                pred_probs = output["expansion_binary"]  # Soft probabilities
            else:
                raise ValueError("CRF must be enabled")

            # 2. Pointer loss
            pointers = output["pointers"]
            ptr_loss_fn = nn.HuberLoss(delta=0.08)
            ptr_loss = ptr_loss_fn(pointers, ptr_batch)

            # 3. Type loss
            logits = output["logits"]
            type_loss_fn = nn.CrossEntropyLoss()
            type_loss = type_loss_fn(logits, y_batch)

            # 4. Total loss (uncertainty weighted)
            sigma_span = output["sigma_span"]
            sigma_ptr = output["sigma_ptr"]
            sigma_type = output["sigma_type"]

            weighted_span_loss = 0.5 * span_loss / (sigma_span**2)
            weighted_ptr_loss = 0.5 * ptr_loss / (sigma_ptr**2)
            weighted_type_loss = 0.5 * type_loss / (sigma_type**2)
            uncertainty_reg = torch.log(sigma_span * sigma_ptr * sigma_type)

            total_batch_loss = (
                weighted_span_loss + weighted_ptr_loss + weighted_type_loss + uncertainty_reg
            )

            total_loss += total_batch_loss.item()
            total_span_loss += span_loss.item()
            total_ptr_loss += ptr_loss.item()
            total_type_loss += type_loss.item()

            # Collect predictions for metrics
            all_pred_spans.append(pred_spans.cpu())
            all_target_spans.append(span_batch.cpu())
            all_pred_probs.append(pred_probs.cpu())

    # Concatenate all batches
    all_pred_spans = torch.cat(all_pred_spans, dim=0)  # (n_samples, 105)
    all_target_spans = torch.cat(all_target_spans, dim=0)  # (n_samples, 105)
    all_pred_probs = torch.cat(all_pred_probs, dim=0)  # (n_samples, 105)

    # Compute F1
    span_metrics = compute_span_metrics(
        all_pred_probs.flatten(), all_target_spans.flatten(), threshold=0.5
    )
    span_f1 = span_metrics["f1"]

    # Compute IoU
    span_iou = compute_iou(all_pred_probs, all_target_spans, threshold=0.5)

    n_batches = len(val_loader)
    return {
        "val_loss": total_loss / n_batches,
        "val_span_loss": total_span_loss / n_batches,
        "val_ptr_loss": total_ptr_loss / n_batches,
        "val_type_loss": total_type_loss / n_batches,
        "span_f1": span_f1,
        "span_iou": span_iou,
        "span_precision": span_metrics["precision"],
        "span_recall": span_metrics["recall"],
    }


# ============================================================================
# Main Training Script
# ============================================================================


def main():
    """Fine-tune position encoding baseline with CRF layer."""
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Fine-tune position encoding baseline with CRF layer"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to baseline checkpoint")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to training data parquet file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for fine-tuned model"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of fine-tuning epochs")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate for fine-tuning"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    # Config
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    data_path = Path(args.data)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading training data (13D features with position encoding)...")
    X, y, expansion_binary, expansion_start, expansion_end = load_training_data(data_path)
    print(f"Data shape: X={X.shape}, y={y.shape}, expansion_binary={expansion_binary.shape}")

    # Compute center and length for pointer targets
    center = ((expansion_start + expansion_end) / 2) / 104.0  # Normalize to [0, 1]
    length = (expansion_end - expansion_start) / 104.0  # Normalize to [0, 1]
    pointers = np.stack([center, length], axis=1).astype(np.float32)  # (n_samples, 2)

    # Train/val split (80/20)
    n_samples = len(X)
    n_train = int(0.8 * n_samples)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    span_train, span_val = expansion_binary[train_idx], expansion_binary[val_idx]
    ptr_train, ptr_val = pointers[train_idx], pointers[val_idx]

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train),
        torch.tensor(span_train),
        torch.tensor(ptr_train),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val),
        torch.tensor(y_val),
        torch.tensor(span_val),
        torch.tensor(ptr_val),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model with CRF
    print("\nInitializing JadeCompact with CRF enabled...")
    model = JadeCompact(
        input_size=13,  # 12 base features + position encoding
        hidden_size=96,
        num_layers=1,
        num_classes=3,
        predict_pointers=True,
        predict_expansion_sequence=True,
        use_crf=True,  # ← KEY: Enable CRF layer
        dropout=0.7,
        input_dropout=0.3,
        dense_dropout=0.6,
    )

    # Load checkpoint weights (if available)
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Load weights with strict=False (CRF layer is new, won't exist in checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded checkpoint weights (CRF layer initialized randomly)")
    else:
        print(f"⚠️ Checkpoint not found at {checkpoint_path}, training from scratch")

    model = model.to(device)

    # Print model info
    param_info = model.get_num_parameters()
    print(f"Total parameters: {param_info['total']:,}")
    print(f"Trainable parameters: {param_info['trainable']:,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    print("\n" + "=" * 80)
    print(f"Starting fine-tuning ({args.epochs} epochs)")
    print("=" * 80)

    best_f1 = 0.0
    metrics_history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, use_crf=True)
        print(
            f"Train - Loss: {train_metrics['train_loss']:.4f}, "
            f"Span: {train_metrics['train_span_loss']:.4f}, "
            f"Ptr: {train_metrics['train_ptr_loss']:.4f}, "
            f"Type: {train_metrics['train_type_loss']:.4f}"
        )

        # Validate
        val_metrics = validate(model, val_loader, device, use_crf=True)
        print(
            f"Val   - Loss: {val_metrics['val_loss']:.4f}, "
            f"F1: {val_metrics['span_f1']:.4f}, "
            f"IoU: {val_metrics['span_iou']:.4f}, "
            f"P: {val_metrics['span_precision']:.4f}, "
            f"R: {val_metrics['span_recall']:.4f}"
        )

        # Compute pointer MAE for center and length
        with torch.no_grad():
            model.eval()
            X_val_tensor = torch.tensor(X_val).to(device)
            output = model(X_val_tensor)
            pred_pointers = output["pointers"].cpu().numpy()
            true_pointers = ptr_val

            center_mae = np.abs(pred_pointers[:, 0] - true_pointers[:, 0]).mean()
            length_mae = np.abs(pred_pointers[:, 1] - true_pointers[:, 1]).mean()

        print(f"Ptr   - Center MAE: {center_mae:.4f}, Length MAE: {length_mae:.4f}")

        # Save metrics
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "val_loss": val_metrics["val_loss"],
            "span_f1": val_metrics["span_f1"],
            "span_iou": val_metrics["span_iou"],
            "span_precision": val_metrics["span_precision"],
            "span_recall": val_metrics["span_recall"],
            "pointer_mae_center": center_mae,
            "pointer_mae_length": length_mae,
        }
        metrics_history.append(epoch_metrics)

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_save_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": epoch_metrics,
                },
                checkpoint_save_path,
            )
            print(f"✓ Saved checkpoint: {checkpoint_save_path.name}")

        # Save best model (by F1)
        if val_metrics["span_f1"] > best_f1:
            best_f1 = val_metrics["span_f1"]
            best_model_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": epoch_metrics,
                    "model_config": {
                        "input_size": 13,
                        "hidden_size": 96,
                        "num_layers": 1,
                        "num_classes": 3,
                        "predict_pointers": True,
                        "predict_expansion_sequence": True,
                        "use_crf": True,
                    },
                },
                best_model_path,
            )
            print(f"✅ New best F1: {best_f1:.4f} (saved to {best_model_path.name})")

    # Save metrics CSV
    metrics_df = pd.DataFrame(metrics_history)
    metrics_csv_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\n✓ Saved metrics to {metrics_csv_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("Fine-tuning Complete")
    print("=" * 80)
    print(f"Best F1: {best_f1:.4f}")
    print(f"Best model: {output_dir / 'best_model.pt'}")
    print(f"Metrics: {metrics_csv_path}")


if __name__ == "__main__":
    main()
