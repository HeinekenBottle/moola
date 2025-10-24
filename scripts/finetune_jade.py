"""Fine-tune Jade on 174 labeled samples with pre-trained encoder.

Implements:
- Pre-trained encoder loading with freeze/unfreeze strategy
- Uncertainty-weighted multi-task loss (Kendall et al., CVPR 2018)
- Focal loss for class imbalance (γ=2)
- WeightedRandomSampler for balanced batch exposure
- Data preprocessing: Raw OHLC → 11-D relativity features
- Pointer encoding: expansion_start/end → normalized center/length [0,1]

Usage:
    # With pre-trained encoder (recommended)
    python3 scripts/finetune_jade.py \
        --data data/processed/labeled/train_latest.parquet \
        --pretrained-encoder artifacts/jade_pretrain_20ep/checkpoint_best.pt \
        --freeze-encoder \
        --epochs 20 \
        --device cuda

    # From scratch (baseline comparison)
    python3 scripts/finetune_jade.py \
        --data data/processed/labeled/train_latest.parquet \
        --epochs 20 \
        --device cuda
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.progress import track
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from moola.features.relativity import build_relativity_features, RelativityConfig
from moola.models.jade_core import JadeCompact

console = Console()


class OHLCAugmenter:
    """Jitter augmentation to match pre-training distribution (σ=0.03)."""

    def __init__(self, jitter_sigma: float = 0.03):
        self.sigma = jitter_sigma

    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        """Apply jitter to features [105, 11]."""
        noise = torch.randn_like(feats) * self.sigma
        return feats + noise


class FocalLoss(nn.Module):
    """Focal loss for class imbalance (Lin et al., 2017).

    Formula: FL = -α(1-p)^γ * log(p)
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Predictions [batch, num_classes]
            targets: Ground truth [batch] (class indices)

        Returns:
            Scalar loss
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        return focal_loss.mean()


class JadeDataset(Dataset):
    """Dataset for Jade fine-tuning with 11-D relativity features."""

    def __init__(
        self,
        df: pd.DataFrame,
        label_map: Dict[str, int] = None,
        window_length: int = 105,
        use_augmentation: bool = False,
        augmentation_multiplier: int = 3,
    ):
        """Initialize dataset.

        Args:
            df: DataFrame with columns [window_id, label, expansion_start, expansion_end, features]
            label_map: Mapping from label strings to class indices
            window_length: Expected window length (default 105)
            use_augmentation: Apply jitter augmentation to match pre-training distribution
            augmentation_multiplier: Effective dataset size multiplier (default 3x)
        """
        self.df = df.reset_index(drop=True)
        self.window_length = window_length
        self.use_augmentation = use_augmentation
        self.multiplier = augmentation_multiplier if use_augmentation else 1

        # Default label map
        if label_map is None:
            self.label_map = {"consolidation": 0, "retracement": 1}
        else:
            self.label_map = label_map

        # Augmenter (σ=0.03 to match pre-training)
        self.augmenter = OHLCAugmenter(jitter_sigma=0.03) if use_augmentation else None

        # Build relativity features for all samples
        console.print(f"[blue]Building 11-D relativity features for {len(df)} samples...[/blue]")
        if use_augmentation:
            console.print(f"[blue]  Augmentation enabled: {augmentation_multiplier}x multiplier (σ=0.03 jitter)[/blue]")
        self.features_11d = self._build_all_features()

        console.print(f"[green]✓ Dataset ready: {len(self)} samples, feature shape: {self.features_11d[0].shape}[/green]")

    def _build_all_features(self) -> List[np.ndarray]:
        """Build 11-D relativity features from raw OHLC."""
        features_list = []

        for idx in track(range(len(self.df)), description="Building features"):
            # Get raw OHLC features [K, 4]
            raw_ohlc = self.df.iloc[idx]["features"]

            # Handle ragged array (stored as array of arrays)
            if isinstance(raw_ohlc[0], np.ndarray):
                raw_ohlc = np.stack(raw_ohlc)  # Convert (105,) with [4] elements to [105, 4]

            # Convert to DataFrame for relativity pipeline
            ohlc_df = pd.DataFrame(
                raw_ohlc,
                columns=["open", "high", "low", "close"]
            )

            # Build 11-D relativity features
            relativity_cfg = RelativityConfig()
            X_11d, valid_mask, _ = build_relativity_features(ohlc_df, relativity_cfg.dict())

            # X_11d shape: [1, K, 11] (only 1 window from this OHLC)
            features_list.append(X_11d[0])  # Extract [K, 11]

        return features_list

    def __len__(self) -> int:
        return len(self.df) * self.multiplier

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Get sample (features, label, pointer_targets).

        Returns:
            features: [K, 11] relativity features
            label: Class index
            pointer_targets: [2] normalized (center, length) in [0, 1]
        """
        # Map augmented idx back to original idx
        original_idx = idx % len(self.df)
        row = self.df.iloc[original_idx]

        # Features [K, 11]
        features = torch.from_numpy(self.features_11d[original_idx]).float()

        # Apply augmentation if not first replica (idx % multiplier != 0)
        if self.use_augmentation and (idx % self.multiplier != 0):
            features = self.augmenter(features)

        # Label
        label = self.label_map[row["label"]]

        # Pointer targets: convert start/end → center/length, normalize to [0, 1]
        expansion_start = row["expansion_start"]
        expansion_end = row["expansion_end"]

        center = (expansion_start + expansion_end) / 2.0 / self.window_length
        length = (expansion_end - expansion_start) / self.window_length

        pointer_targets = torch.tensor([center, length], dtype=torch.float32)

        return features, label, pointer_targets


def create_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Create weighted sampler for balanced batch exposure."""
    unique, counts = np.unique(labels, return_counts=True)
    class_weights = 1.0 / counts
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    console.print(f"[blue]Class distribution: {dict(zip(unique, counts))}[/blue]")
    console.print(f"[blue]Class weights: {dict(zip(unique, class_weights))}[/blue]")

    return sampler


def compute_multi_task_loss(
    logits: torch.Tensor,
    pointers: torch.Tensor,
    targets_label: torch.Tensor,
    targets_pointer: torch.Tensor,
    sigma_ptr: torch.Tensor,
    sigma_type: torch.Tensor,
    focal_gamma: float = 2.0,
    use_uncertainty_weighting: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute uncertainty-weighted multi-task loss.

    Args:
        logits: Classification logits [batch, num_classes]
        pointers: Pointer predictions [batch, 2] (center, length)
        targets_label: Label targets [batch]
        targets_pointer: Pointer targets [batch, 2]
        sigma_ptr: Learned pointer uncertainty
        sigma_type: Learned type uncertainty
        focal_gamma: Focal loss gamma
        use_uncertainty_weighting: Use Kendall uncertainty weighting

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    # Classification loss with focal loss
    focal_loss_fn = FocalLoss(gamma=focal_gamma)
    loss_type = focal_loss_fn(logits, targets_label)

    # Pointer loss with Huber (δ=0.08, ~8 timesteps transition tolerance)
    loss_ptr = F.huber_loss(pointers, targets_pointer, delta=0.08)

    # Uncertainty weighting (Kendall et al., CVPR 2018)
    if use_uncertainty_weighting:
        # L = (1/2σ_ptr²)L_ptr + (1/2σ_type²)L_type + log(σ_ptr) + log(σ_type)
        weighted_loss_ptr = loss_ptr / (2 * sigma_ptr ** 2) + torch.log(sigma_ptr)
        weighted_loss_type = loss_type / (2 * sigma_type ** 2) + torch.log(sigma_type)
        total_loss = weighted_loss_ptr + weighted_loss_type
    else:
        # Fixed weights (fallback)
        total_loss = 0.7 * loss_type + 0.3 * loss_ptr

    metrics = {
        "loss_total": total_loss.item(),
        "loss_type": loss_type.item(),
        "loss_ptr": loss_ptr.item(),
        "sigma_ptr": sigma_ptr.item() if use_uncertainty_weighting else 1.0,
        "sigma_type": sigma_type.item() if use_uncertainty_weighting else 1.0,
    }

    return total_loss, metrics


def train_epoch(
    model: JadeCompact,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    focal_gamma: float = 0.0,
    use_uncertainty_weighting: bool = True,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_count = 0

    for features, labels, pointer_targets in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        pointer_targets = pointer_targets.to(device)

        # Forward pass
        output = model(features)
        logits = output["logits"]
        pointers = output["pointers"]
        sigma_ptr = output.get("sigma_ptr", torch.tensor(1.0))
        sigma_type = output.get("sigma_type", torch.tensor(1.0))

        # Compute loss
        loss, metrics_dict = compute_multi_task_loss(
            logits, pointers, labels, pointer_targets,
            sigma_ptr, sigma_type,
            focal_gamma=focal_gamma,
            use_uncertainty_weighting=use_uncertainty_weighting
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)

        # Per-task monitoring every 10 batches
        batch_count += 1
        if batch_count % 10 == 0:
            console.print(
                f"[blue]Batch {batch_count}: L_ptr={metrics_dict['loss_ptr']:.4f}, "
                f"L_type={metrics_dict['loss_type']:.4f}, "
                f"σ_ptr={metrics_dict['sigma_ptr']:.2f}, "
                f"σ_type={metrics_dict['sigma_type']:.2f}[/blue]"
            )
            # Collapse detection
            if metrics_dict['loss_ptr'] < 1e-4 or metrics_dict['sigma_ptr'] > 10:
                console.print("[yellow]⚠️ Ptr collapse: L_ptr near 0 or σ_ptr >10[/yellow]")
            if metrics_dict['loss_type'] < 1e-4 or metrics_dict['sigma_type'] > 10:
                console.print("[yellow]⚠️ Type collapse: L_type near 0 or σ_type >10[/yellow]")

    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }

    return metrics


@torch.no_grad()
def evaluate(
    model: JadeCompact,
    dataloader: DataLoader,
    device: str,
    focal_gamma: float = 0.0,
    use_uncertainty_weighting: bool = True,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    for features, labels, pointer_targets in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        pointer_targets = pointer_targets.to(device)

        # Forward pass
        output = model(features)
        logits = output["logits"]
        pointers = output["pointers"]
        sigma_ptr = output.get("sigma_ptr", torch.tensor(1.0))
        sigma_type = output.get("sigma_type", torch.tensor(1.0))

        # Compute loss
        loss, _ = compute_multi_task_loss(
            logits, pointers, labels, pointer_targets,
            sigma_ptr, sigma_type,
            focal_gamma=focal_gamma,
            use_uncertainty_weighting=use_uncertainty_weighting
        )

        # Metrics
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Compute F1 scores
    from sklearn.metrics import f1_score

    f1_macro = f1_score(all_labels, all_preds, average="macro")

    # Per-class report
    report = classification_report(
        all_labels, all_preds,
        labels=[0, 1],
        target_names=["consolidation", "retracement"],
        zero_division=0
    )
    console.print(f"[blue]Per-class Report:\n{report}[/blue]")

    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "f1_macro": f1_macro,
    }

    return metrics


def train_with_cv(
    data_path: str,
    pretrained_encoder: str = None,
    freeze_encoder: bool = True,
    unfreeze_epoch: int = None,
    lr_unfrozen: float = 1e-5,
    l2_encoder: float = 1e-3,
    n_epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    focal_gamma: float = 0.0,
    use_augmentation: bool = False,
    device: str = "cuda",
    output_dir: str = "artifacts/jade_finetuned",
    n_folds: int = 5,
):
    """Train Jade with cross-validation."""
    # Load data
    console.print(f"[bold cyan]═══ Jade Fine-Tuning ═══[/bold cyan]\n")
    console.print(f"[blue]Loading data: {data_path}[/blue]")
    df = pd.read_parquet(data_path)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    labels = df["label"].map({"consolidation": 0, "retracement": 1}).values

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, labels)):
        console.print(f"\n[bold magenta]Fold {fold_idx + 1}/{n_folds}[/bold magenta]")

        # Split data
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        console.print(f"  Train: {len(train_df)} samples")
        console.print(f"  Val: {len(val_df)} samples")

        # Create datasets
        train_dataset = JadeDataset(train_df, use_augmentation=use_augmentation)
        val_dataset = JadeDataset(val_df, use_augmentation=False)  # No aug for validation

        # Create weighted sampler for training
        train_labels = train_df["label"].map({"consolidation": 0, "retracement": 1}).values
        sampler = create_weighted_sampler(train_labels)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True if device == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if device == "cuda" else False,
        )

        # Create model
        if pretrained_encoder:
            console.print(f"[blue]Loading pre-trained encoder: {pretrained_encoder}[/blue]")
            model = JadeCompact.from_pretrained(
                pretrained_encoder,
                predict_pointers=True,
                freeze_encoder=freeze_encoder,
            )
        else:
            console.print("[yellow]Training from scratch (no pre-training)[/yellow]")
            model = JadeCompact(
                input_size=11,
                hidden_size=96,
                num_layers=1,
                predict_pointers=True,
            )

        model = model.to(device)

        # Optimizer (will be recreated if unfreezing)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=1e-4
        )

        # Training loop
        best_f1 = 0.0
        patience_counter = 0
        patience = 5

        for epoch in range(n_epochs):
            # Unfreezing logic
            if unfreeze_epoch is not None and epoch == unfreeze_epoch and pretrained_encoder:
                console.print(f"\n[bold yellow]Unfreezing last encoder layer at epoch {epoch}[/bold yellow]")

                # Unfreeze last LSTM layer
                for name, param in model.named_parameters():
                    if "encoder.lstm.weight_ih_l1" in name or "encoder.lstm.weight_hh_l1" in name:
                        param.requires_grad = True
                        console.print(f"  [green]Unfroze: {name}[/green]")

                # Recreate optimizer with hierarchical weight decay
                param_groups = [
                    {'params': [p for n, p in model.named_parameters() if 'encoder' in n and p.requires_grad],
                     'lr': lr_unfrozen, 'weight_decay': l2_encoder},
                    {'params': [p for n, p in model.named_parameters() if 'encoder' not in n and p.requires_grad],
                     'lr': learning_rate, 'weight_decay': 1e-4},
                ]
                optimizer = torch.optim.AdamW(param_groups)
                console.print(f"  [green]Encoder LR: {lr_unfrozen}, weight_decay: {l2_encoder}[/green]")
                console.print(f"  [green]Heads LR: {learning_rate}, weight_decay: 1e-4[/green]")

            train_metrics = train_epoch(model, train_loader, optimizer, device, focal_gamma=focal_gamma)
            val_metrics = evaluate(model, val_loader, device, focal_gamma=focal_gamma)

            console.print(
                f"  Epoch {epoch+1}/{n_epochs}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_acc={val_metrics['accuracy']:.3f}, "
                f"val_f1={val_metrics['f1_macro']:.3f}"
            )

            # Early stopping
            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                patience_counter = 0

                # Save best checkpoint
                checkpoint_path = output_path / f"fold{fold_idx}_best.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "epoch": epoch,
                }, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    console.print(f"[yellow]Early stopping at epoch {epoch+1}[/yellow]")
                    break

        # Store fold results
        fold_results.append({
            "fold": fold_idx + 1,
            "best_f1": best_f1,
            "final_val_metrics": val_metrics,
        })

        console.print(f"[green]✓ Fold {fold_idx + 1} best F1: {best_f1:.3f}[/green]")

    # Average results
    avg_f1 = np.mean([r["best_f1"] for r in fold_results])
    console.print(f"\n[bold green]Average F1 across {n_folds} folds: {avg_f1:.3f}[/bold green]")

    # Save summary
    summary = {
        "avg_f1_macro": avg_f1,
        "fold_results": fold_results,
        "config": {
            "pretrained_encoder": pretrained_encoder,
            "freeze_encoder": freeze_encoder,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "focal_gamma": focal_gamma,
            "use_augmentation": use_augmentation,
        }
    }

    summary_path = output_path / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"[green]✓ Saved summary to {summary_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Jade on labeled data")
    parser.add_argument("--data", required=True, help="Path to train_latest.parquet")
    parser.add_argument("--pretrained-encoder", default=None, help="Path to pre-trained encoder checkpoint")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder weights")
    parser.add_argument("--unfreeze-epoch", type=int, default=None, help="Epoch to unfreeze last encoder layer (None=keep frozen)")
    parser.add_argument("--lr-unfrozen", type=float, default=1e-5, help="LR for unfrozen encoder")
    parser.add_argument("--l2-encoder", type=float, default=1e-3, help="L2 regularization for encoder")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--focal-gamma", type=float, default=0.0, help="Focal loss gamma (0=CE)")
    parser.add_argument("--use-augmentation", action="store_true", help="Enable jitter augmentation (σ=0.03, 3x multiplier)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="artifacts/jade_finetuned", help="Output directory")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")

    args = parser.parse_args()

    train_with_cv(
        data_path=args.data,
        pretrained_encoder=args.pretrained_encoder,
        freeze_encoder=args.freeze_encoder,
        unfreeze_epoch=args.unfreeze_epoch,
        lr_unfrozen=args.lr_unfrozen,
        l2_encoder=args.l2_encoder,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        focal_gamma=args.focal_gamma,
        use_augmentation=args.use_augmentation,
        device=args.device,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()
