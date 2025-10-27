"""Fine-tune Jade on 174 labeled samples with pre-trained encoder.

Implements:
- Pre-trained encoder loading with freeze/unfreeze strategy
- Uncertainty-weighted multi-task loss (Kendall et al., CVPR 2018)
- Focal loss for class imbalance (γ=2)
- WeightedRandomSampler for balanced batch exposure
- Data preprocessing: Raw OHLC → 12-D relativity features (incl. consol_proxy)
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
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.progress import track
from rich.table import Table
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from moola.features.relativity import RelativityConfig, build_relativity_features
from moola.models.jade_core import JadeCompact

console = Console()


class GradientConflictMonitor:
    """Monitor gradient conflicts between tasks during multi-task learning.

    Tracks:
    - Cosine similarity between task gradients
    - Conflict frequency (% of updates with cos < threshold)
    - Average gradient magnitudes per task
    - PCGrad projection frequency
    """

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.reset()

    def reset(self):
        """Reset all monitoring statistics."""
        self.step_count = 0
        self.conflict_count = 0
        self.projection_count = 0
        self.cos_similarities = []
        self.grad_mag_task1 = []
        self.grad_mag_task2 = []

    def record(
        self,
        cos_sim: float,
        grad_mag_1: float,
        grad_mag_2: float,
        was_projected: bool,
        is_conflict: bool,
    ):
        """Record gradient statistics for a single step."""
        self.step_count += 1
        self.cos_similarities.append(cos_sim)
        self.grad_mag_task1.append(grad_mag_1)
        self.grad_mag_task2.append(grad_mag_2)

        if is_conflict:
            self.conflict_count += 1
        if was_projected:
            self.projection_count += 1

    def should_log(self) -> bool:
        """Check if we should log statistics."""
        return self.step_count > 0 and self.step_count % self.log_interval == 0

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics."""
        if self.step_count == 0:
            return {}

        return {
            "conflict_rate": self.conflict_count / self.step_count,
            "projection_rate": self.projection_count / self.step_count,
            "avg_cos_sim": np.mean(self.cos_similarities),
            "min_cos_sim": np.min(self.cos_similarities),
            "avg_grad_mag_task1": np.mean(self.grad_mag_task1),
            "avg_grad_mag_task2": np.mean(self.grad_mag_task2),
            "total_steps": self.step_count,
        }

    def log_summary(self):
        """Log summary statistics with rich formatting."""
        summary = self.get_summary()
        if not summary:
            return

        table = Table(
            title="Gradient Conflict Analysis",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Steps", f"{summary['total_steps']}")
        table.add_row("Conflict Rate", f"{summary['conflict_rate']:.2%}")
        table.add_row("Projection Rate", f"{summary['projection_rate']:.2%}")
        table.add_row("Avg Cos Similarity", f"{summary['avg_cos_sim']:.4f}")
        table.add_row("Min Cos Similarity", f"{summary['min_cos_sim']:.4f}")
        table.add_row("Avg |∇L_type|", f"{summary['avg_grad_mag_task1']:.6f}")
        table.add_row("Avg |∇L_ptr|", f"{summary['avg_grad_mag_task2']:.6f}")

        console.print(table)


def project_conflicting_gradients(
    grads_task1: list[torch.Tensor],
    grads_task2: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor], bool, float, float, float]:
    """Project gradients orthogonally if they conflict (negative dot product).

    Implements PCGrad (Yu et al., NeurIPS 2020):
    When two task gradients conflict (cos < 0), project each gradient onto
    the normal plane of the other to remove the conflicting component.

    Args:
        grads_task1: List of gradients for task 1 (classification)
        grads_task2: List of gradients for task 2 (pointer regression)

    Returns:
        Tuple of:
        - projected_grads_task1: Projected gradients for task 1
        - projected_grads_task2: Projected gradients for task 2
        - conflict_detected: True if gradients conflict (cos < 0)
        - cos_similarity: Cosine similarity between gradients
        - grad_mag_1: Magnitude of task 1 gradient
        - grad_mag_2: Magnitude of task 2 gradient
    """
    # Flatten gradients to 1D vectors
    flat_grad1 = torch.cat([g.flatten() for g in grads_task1 if g is not None])
    flat_grad2 = torch.cat([g.flatten() for g in grads_task2 if g is not None])

    # Compute gradient magnitudes
    grad_mag_1 = torch.norm(flat_grad1).item()
    grad_mag_2 = torch.norm(flat_grad2).item()

    # Compute cosine similarity
    dot_product = torch.dot(flat_grad1, flat_grad2)
    cos_similarity = (dot_product / (grad_mag_1 * grad_mag_2 + 1e-8)).item()

    # Check for conflict
    conflict_detected = cos_similarity < 0

    if not conflict_detected:
        # No conflict - return original gradients
        return grads_task1, grads_task2, False, cos_similarity, grad_mag_1, grad_mag_2

    # Project gradients orthogonally
    # g1_proj = g1 - (g1 · g2 / ||g2||²) * g2
    # g2_proj = g2 - (g2 · g1 / ||g1||²) * g1

    # Compute projection coefficients
    coef_1_on_2 = dot_product / (torch.norm(flat_grad2) ** 2 + 1e-8)
    coef_2_on_1 = dot_product / (torch.norm(flat_grad1) ** 2 + 1e-8)

    # Project flat gradients
    flat_grad1_proj = flat_grad1 - coef_1_on_2 * flat_grad2
    flat_grad2_proj = flat_grad2 - coef_2_on_1 * flat_grad1

    # Reshape back to original gradient structure
    projected_grads_task1 = []
    projected_grads_task2 = []

    offset = 0
    for g1, g2 in zip(grads_task1, grads_task2):
        if g1 is not None:
            numel = g1.numel()
            projected_grads_task1.append(flat_grad1_proj[offset : offset + numel].reshape_as(g1))
            projected_grads_task2.append(flat_grad2_proj[offset : offset + numel].reshape_as(g2))
            offset += numel
        else:
            projected_grads_task1.append(None)
            projected_grads_task2.append(None)

    return (
        projected_grads_task1,
        projected_grads_task2,
        True,
        cos_similarity,
        grad_mag_1,
        grad_mag_2,
    )


class OHLCAugmenter:
    """Jitter augmentation to match pre-training distribution (σ=0.03)."""

    def __init__(self, jitter_sigma: float = 0.03):
        self.sigma = jitter_sigma

    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        """Apply jitter to features [105, 12]."""
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
    """Dataset for Jade fine-tuning with 12-D relativity features (incl. consol_proxy)."""

    def __init__(
        self,
        df: pd.DataFrame,
        label_map: dict[str, int] = None,
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
        console.print(f"[blue]Building 12-D relativity features for {len(df)} samples...[/blue]")
        if use_augmentation:
            console.print(
                f"[blue]  Augmentation enabled: {augmentation_multiplier}x multiplier (σ=0.03 jitter)[/blue]"
            )
        self.features_12d = self._build_all_features()

        console.print(
            f"[green]✓ Dataset ready: {len(self)} samples, feature shape: {self.features_12d[0].shape}[/green]"
        )

    def _build_all_features(self) -> list[np.ndarray]:
        """Build 12-D relativity features from raw OHLC (incl. consol_proxy)."""
        features_list = []

        for idx in track(range(len(self.df)), description="Building features"):
            # Get raw OHLC features [K, 4]
            raw_ohlc = self.df.iloc[idx]["features"]

            # Handle ragged array (stored as array of arrays)
            if isinstance(raw_ohlc[0], np.ndarray):
                raw_ohlc = np.stack(raw_ohlc)  # Convert (105,) with [4] elements to [105, 4]

            # Convert to DataFrame for relativity pipeline
            ohlc_df = pd.DataFrame(raw_ohlc, columns=["open", "high", "low", "close"])

            # Build 12-D relativity features (incl. consol_proxy)
            relativity_cfg = RelativityConfig()
            X_12d, valid_mask, _ = build_relativity_features(ohlc_df, relativity_cfg.dict())

            # X_12d shape: [1, K, 12] (only 1 window from this OHLC)
            features_list.append(X_12d[0])  # Extract [K, 12]

        return features_list

    def __len__(self) -> int:
        return len(self.df) * self.multiplier

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Get sample (features, label, pointer_targets).

        Returns:
            features: [K, 12] relativity features (incl. consol_proxy)
            label: Class index
            pointer_targets: [2] normalized (center, length) in [0, 1]
        """
        # Map augmented idx back to original idx
        original_idx = idx % len(self.df)
        row = self.df.iloc[original_idx]

        # Features [K, 12]
        features = torch.from_numpy(self.features_12d[original_idx]).float()

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
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    console.print(f"[blue]Class distribution: {dict(zip(unique, counts))}[/blue]")
    console.print(f"[blue]Class weights: {dict(zip(unique, class_weights))}[/blue]")

    return sampler


def compute_pointer_metrics(
    pred_centers: np.ndarray,
    pred_lengths: np.ndarray,
    true_centers: np.ndarray,
    true_lengths: np.ndarray,
    window_length: int = 105,
) -> dict[str, float]:
    """Compute pointer regression metrics.

    Args:
        pred_centers: Predicted centers (normalized [0, 1])
        pred_lengths: Predicted lengths (normalized [0, 1])
        true_centers: True centers (normalized [0, 1])
        true_lengths: True lengths (normalized [0, 1])
        window_length: Window length for denormalization (default 105)

    Returns:
        Dictionary of pointer metrics
    """
    # Denormalize to timestep units
    pred_centers_ts = pred_centers * window_length
    pred_lengths_ts = pred_lengths * window_length
    true_centers_ts = true_centers * window_length
    true_lengths_ts = true_lengths * window_length

    # MAE for center and length
    center_mae = np.mean(np.abs(pred_centers_ts - true_centers_ts))
    length_mae = np.mean(np.abs(pred_lengths_ts - true_lengths_ts))

    # Hit@±3 and Hit@±5 (center within ±N bars)
    center_errors = np.abs(pred_centers_ts - true_centers_ts)
    hit_at_3 = np.mean(center_errors <= 3.0)
    hit_at_5 = np.mean(center_errors <= 5.0)

    return {
        "center_mae": center_mae,
        "length_mae": length_mae,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,
    }


def compute_calibration_metrics(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """Compute calibration metrics (ECE, MCE, Brier score).

    Args:
        probs: Predicted probabilities for positive class [N]
        labels: Binary labels [N]
        n_bins: Number of bins for ECE computation

    Returns:
        Dictionary of calibration metrics
    """
    # Brier score
    brier = brier_score_loss(labels, probs)

    # ECE and MCE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            # Compute accuracy and confidence in this bin
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])

            # Calibration error for this bin
            bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)

            # ECE: weighted by proportion of samples in bin
            ece += prop_in_bin * bin_error

            # MCE: maximum calibration error
            mce = max(mce, bin_error)

    return {
        "ece": ece,
        "mce": mce,
        "brier": brier,
    }


def compute_joint_success(
    preds: np.ndarray,
    labels: np.ndarray,
    pred_centers: np.ndarray,
    true_centers: np.ndarray,
    window_length: int = 105,
    center_tolerance: int = 3,
) -> float:
    """Compute joint success rate (correct classification AND correct pointer).

    Args:
        preds: Predicted class labels
        labels: True class labels
        pred_centers: Predicted centers (normalized [0, 1])
        true_centers: True centers (normalized [0, 1])
        window_length: Window length for denormalization
        center_tolerance: Tolerance in timesteps for center prediction (default ±3)

    Returns:
        Joint success rate (fraction of samples with both tasks correct)
    """
    # Classification correct
    classification_correct = preds == labels

    # Pointer correct (center within tolerance)
    pred_centers_ts = pred_centers * window_length
    true_centers_ts = true_centers * window_length
    center_errors = np.abs(pred_centers_ts - true_centers_ts)
    pointer_correct = center_errors <= center_tolerance

    # Both correct
    joint_correct = classification_correct & pointer_correct

    return np.mean(joint_correct)


def create_metrics_table(metrics: dict[str, float], title: str = "Validation Metrics") -> Table:
    """Create a rich Table for displaying metrics.

    Args:
        metrics: Dictionary of metric name -> value
        title: Table title

    Returns:
        Rich Table object
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="green", justify="right", width=15)

    # Organize metrics by category
    categories = {
        "Classification": [
            "accuracy",
            "f1_macro",
            "precision_cons",
            "recall_cons",
            "f1_cons",
            "precision_retr",
            "recall_retr",
            "f1_retr",
            "auroc",
            "auprc",
        ],
        "Pointer Regression": ["center_mae", "length_mae", "hit_at_3", "hit_at_5"],
        "Calibration": ["ece", "mce", "brier"],
        "Multi-Task": ["joint_success_at_3", "joint_success_at_5"],
        "Loss": ["loss", "loss_type", "loss_ptr"],
    }

    for category, metric_keys in categories.items():
        # Add category header
        table.add_row(f"[bold yellow]{category}[/bold yellow]", "")

        # Add metrics in this category
        for key in metric_keys:
            if key in metrics:
                value = metrics[key]

                # Format based on metric type
                if "mae" in key or "loss" in key:
                    formatted_value = f"{value:.4f}"
                elif (
                    "hit" in key
                    or "success" in key
                    or key in ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]
                ):
                    formatted_value = f"{value:.3f} ({value*100:.1f}%)"
                elif key in ["ece", "mce", "brier"]:
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = f"{value:.4f}"

                table.add_row(f"  {key}", formatted_value)

        # Add spacing between categories
        table.add_row("", "")

    return table


def compute_multi_task_loss(
    logits: torch.Tensor,
    pointers: torch.Tensor,
    targets_label: torch.Tensor,
    targets_pointer: torch.Tensor,
    sigma_ptr: torch.Tensor,
    sigma_type: torch.Tensor,
    focal_gamma: float = 2.0,
    use_uncertainty_weighting: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
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
        weighted_loss_ptr = loss_ptr / (2 * sigma_ptr**2) + torch.log(sigma_ptr)
        weighted_loss_type = loss_type / (2 * sigma_type**2) + torch.log(sigma_type)
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
    use_pcgrad: bool = True,
    pcgrad_threshold: float = -0.3,
    conflict_monitor: Optional[GradientConflictMonitor] = None,
) -> dict[str, float]:
    """Train for one epoch with optional PCGrad projection.

    Args:
        model: JadeCompact model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device (cuda/cpu)
        focal_gamma: Focal loss gamma parameter
        use_uncertainty_weighting: Use Kendall uncertainty weighting
        use_pcgrad: Apply PCGrad when gradients conflict
        pcgrad_threshold: Apply PCGrad when cos similarity < threshold (default -0.3)
        conflict_monitor: Optional monitor for gradient conflict statistics

    Returns:
        Dictionary of training metrics
    """
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

        # Compute individual task losses
        focal_loss_fn = FocalLoss(gamma=focal_gamma)
        loss_type = focal_loss_fn(logits, labels)
        loss_ptr = F.huber_loss(pointers, pointer_targets, delta=0.08)

        if use_pcgrad:
            # Compute gradients for each task separately
            optimizer.zero_grad()

            # Task 1: Classification gradients
            loss_type.backward(retain_graph=True)
            # Collect only SHARED encoder gradients (not task-specific heads)
            grads_task1 = []
            for name, p in model.named_parameters():
                if "lstm" in name or "projection" in name or "input_dropout" in name:
                    grads_task1.append(p.grad.clone() if p.grad is not None else None)

            # Clear gradients
            optimizer.zero_grad()

            # Task 2: Pointer gradients
            loss_ptr.backward(retain_graph=True)
            # Collect only SHARED encoder gradients (not task-specific heads)
            grads_task2 = []
            for name, p in model.named_parameters():
                if "lstm" in name or "projection" in name or "input_dropout" in name:
                    grads_task2.append(p.grad.clone() if p.grad is not None else None)

            # Clear gradients again
            optimizer.zero_grad()

            # Apply PCGrad projection
            (
                proj_grads_task1,
                proj_grads_task2,
                was_projected,
                cos_sim,
                grad_mag_1,
                grad_mag_2,
            ) = project_conflicting_gradients(grads_task1, grads_task2)

            # Decide whether to apply projection based on threshold
            should_project = use_pcgrad and cos_sim < pcgrad_threshold
            is_conflict = cos_sim < 0

            # Record conflict statistics
            if conflict_monitor is not None:
                conflict_monitor.record(
                    cos_sim, grad_mag_1, grad_mag_2, should_project, is_conflict
                )

            # Use projected gradients if conflict is strong enough
            if should_project:
                # Combine projected gradients with uncertainty weighting
                if use_uncertainty_weighting:
                    weight_type = 1.0 / (2 * sigma_type**2)
                    weight_ptr = 1.0 / (2 * sigma_ptr**2)
                else:
                    weight_type = 0.7
                    weight_ptr = 0.3

                # Apply projected gradients to encoder, standard gradients to task heads
                # First, set encoder gradients to projected values
                encoder_params = [
                    (name, p)
                    for name, p in model.named_parameters()
                    if "lstm" in name or "projection" in name or "input_dropout" in name
                ]
                for (name, param), g1, g2 in zip(
                    encoder_params, proj_grads_task1, proj_grads_task2
                ):
                    if param.requires_grad and g1 is not None and g2 is not None:
                        param.grad = weight_type * g1 + weight_ptr * g2

                # Then compute standard combined loss for task-specific heads
                combined_loss = weight_type * loss_type + weight_ptr * loss_ptr
                combined_loss.backward()  # Adds gradients for classifier + pointer heads
            else:
                # No strong conflict - use standard combined loss
                loss, _ = compute_multi_task_loss(
                    logits,
                    pointers,
                    labels,
                    pointer_targets,
                    sigma_ptr,
                    sigma_type,
                    focal_gamma=focal_gamma,
                    use_uncertainty_weighting=use_uncertainty_weighting,
                )
                loss.backward()

            # Update parameters
            optimizer.step()

            # Compute metrics for logging
            metrics_dict = {
                "loss_type": loss_type.item(),
                "loss_ptr": loss_ptr.item(),
                "sigma_ptr": sigma_ptr.item() if use_uncertainty_weighting else 1.0,
                "sigma_type": sigma_type.item() if use_uncertainty_weighting else 1.0,
            }
            total_loss += (loss_type.item() + loss_ptr.item()) * len(labels)

        else:
            # Standard training without PCGrad
            loss, metrics_dict = compute_multi_task_loss(
                logits,
                pointers,
                labels,
                pointer_targets,
                sigma_ptr,
                sigma_type,
                focal_gamma=focal_gamma,
                use_uncertainty_weighting=use_uncertainty_weighting,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)

        # Metrics
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)

        # Per-task monitoring every 10 batches
        batch_count += 1
        if batch_count % 10 == 0:
            pcgrad_status = ""
            if use_pcgrad and conflict_monitor is not None:
                rate = conflict_monitor.projection_count / conflict_monitor.step_count
                pcgrad_status = f" | PCGrad: {conflict_monitor.projection_count}/{conflict_monitor.step_count} ({rate:.1%})"

            console.print(
                f"[blue]Batch {batch_count}: L_ptr={metrics_dict['loss_ptr']:.4f}, "
                f"L_type={metrics_dict['loss_type']:.4f}, "
                f"σ_ptr={metrics_dict['sigma_ptr']:.2f}, "
                f"σ_type={metrics_dict['sigma_type']:.2f}{pcgrad_status}[/blue]"
            )
            # Collapse detection
            if metrics_dict["loss_ptr"] < 1e-4 or metrics_dict["sigma_ptr"] > 10:
                console.print("[yellow]⚠️ Ptr collapse: L_ptr near 0 or σ_ptr >10[/yellow]")
            if metrics_dict["loss_type"] < 1e-4 or metrics_dict["sigma_type"] > 10:
                console.print("[yellow]⚠️ Type collapse: L_type near 0 or σ_type >10[/yellow]")

        # Log conflict analysis periodically
        if use_pcgrad and conflict_monitor is not None and conflict_monitor.should_log():
            conflict_monitor.log_summary()

    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }

    # Add conflict statistics to final metrics
    if use_pcgrad and conflict_monitor is not None:
        conflict_summary = conflict_monitor.get_summary()
        if conflict_summary:
            metrics.update(
                {
                    "conflict_rate": conflict_summary["conflict_rate"],
                    "projection_rate": conflict_summary["projection_rate"],
                    "avg_cos_sim": conflict_summary["avg_cos_sim"],
                }
            )

    return metrics


@torch.no_grad()
def evaluate(
    model: JadeCompact,
    dataloader: DataLoader,
    device: str,
    focal_gamma: float = 0.0,
    use_uncertainty_weighting: bool = True,
    window_length: int = 105,
) -> dict[str, float]:
    """Evaluate model on validation set with comprehensive metrics."""
    model.eval()

    total_loss = 0.0
    total_loss_type = 0.0
    total_loss_ptr = 0.0
    total_correct = 0
    total_samples = 0

    # Collect predictions and targets
    all_preds = []
    all_labels = []
    all_probs = []  # For calibration metrics
    all_pred_centers = []
    all_pred_lengths = []
    all_true_centers = []
    all_true_lengths = []

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
        loss, loss_dict = compute_multi_task_loss(
            logits,
            pointers,
            labels,
            pointer_targets,
            sigma_ptr,
            sigma_type,
            focal_gamma=focal_gamma,
            use_uncertainty_weighting=use_uncertainty_weighting,
        )

        # Accumulate losses
        total_loss += loss.item() * len(labels)
        total_loss_type += loss_dict["loss_type"] * len(labels)
        total_loss_ptr += loss_dict["loss_ptr"] * len(labels)

        # Predictions
        preds = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)[:, 1]  # Probability of retracement (class 1)

        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)

        # Collect for metrics
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_pred_centers.extend(pointers[:, 0].cpu().numpy())
        all_pred_lengths.extend(pointers[:, 1].cpu().numpy())
        all_true_centers.extend(pointer_targets[:, 0].cpu().numpy())
        all_true_lengths.extend(pointer_targets[:, 1].cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_pred_centers = np.array(all_pred_centers)
    all_pred_lengths = np.array(all_pred_lengths)
    all_true_centers = np.array(all_true_centers)
    all_true_lengths = np.array(all_true_lengths)

    # === Classification Metrics ===
    accuracy = total_correct / total_samples
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Per-class precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1], zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    # AUROC and AUPRC (if both classes present)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    except ValueError:
        # Only one class in validation set
        auroc = 0.0
        auprc = 0.0

    # === Pointer Regression Metrics ===
    pointer_metrics = compute_pointer_metrics(
        all_pred_centers,
        all_pred_lengths,
        all_true_centers,
        all_true_lengths,
        window_length=window_length,
    )

    # === Calibration Metrics ===
    calibration_metrics = compute_calibration_metrics(all_probs, all_labels)

    # === Joint Success Metrics ===
    joint_success_at_3 = compute_joint_success(
        all_preds,
        all_labels,
        all_pred_centers,
        all_true_centers,
        window_length=window_length,
        center_tolerance=3,
    )

    joint_success_at_5 = compute_joint_success(
        all_preds,
        all_labels,
        all_pred_centers,
        all_true_centers,
        window_length=window_length,
        center_tolerance=5,
    )

    # Compile all metrics
    metrics = {
        # Loss
        "loss": total_loss / total_samples,
        "loss_type": total_loss_type / total_samples,
        "loss_ptr": total_loss_ptr / total_samples,
        # Classification
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "precision_cons": precision[0],
        "recall_cons": recall[0],
        "f1_cons": f1[0],
        "precision_retr": precision[1],
        "recall_retr": recall[1],
        "f1_retr": f1[1],
        "auroc": auroc,
        "auprc": auprc,
        # Pointer regression
        "center_mae": pointer_metrics["center_mae"],
        "length_mae": pointer_metrics["length_mae"],
        "hit_at_3": pointer_metrics["hit_at_3"],
        "hit_at_5": pointer_metrics["hit_at_5"],
        # Calibration
        "ece": calibration_metrics["ece"],
        "mce": calibration_metrics["mce"],
        "brier": calibration_metrics["brier"],
        # Multi-task joint success
        "joint_success_at_3": joint_success_at_3,
        "joint_success_at_5": joint_success_at_5,
    }

    # Print comprehensive metrics table
    table = create_metrics_table(metrics)
    console.print(table)

    # Print confusion matrix
    console.print("\n[blue]Confusion Matrix:[/blue]")
    console.print(f"  True Cons / Pred Cons: {cm[0, 0]}")
    console.print(f"  True Cons / Pred Retr: {cm[0, 1]}")
    console.print(f"  True Retr / Pred Cons: {cm[1, 0]}")
    console.print(f"  True Retr / Pred Retr: {cm[1, 1]}")

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
    use_pcgrad: bool = True,
    pcgrad_threshold: float = -0.3,
    device: str = "cuda",
    output_dir: str = "artifacts/jade_finetuned",
    n_folds: int = 5,
):
    """Train Jade with cross-validation."""
    # Load data
    console.print("[bold cyan]═══ Jade Fine-Tuning ═══[/bold cyan]\n")
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
                input_size=12,
                hidden_size=96,
                num_layers=1,
                predict_pointers=True,
            )

        model = model.to(device)

        # Optimizer (will be recreated if unfreezing)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=1e-4
        )

        # Create gradient conflict monitor for PCGrad
        conflict_monitor = None
        if use_pcgrad:
            conflict_monitor = GradientConflictMonitor(log_interval=100)
            console.print(f"[blue]PCGrad enabled: threshold={pcgrad_threshold}[/blue]")

        # Training loop
        best_f1 = 0.0
        best_metrics = None
        patience_counter = 0
        patience = 5

        for epoch in range(n_epochs):
            # Unfreezing logic
            if unfreeze_epoch is not None and epoch == unfreeze_epoch and pretrained_encoder:
                console.print(
                    f"\n[bold yellow]Unfreezing last encoder layer at epoch {epoch}[/bold yellow]"
                )

                # Unfreeze last LSTM layer
                for name, param in model.named_parameters():
                    if "encoder.lstm.weight_ih_l1" in name or "encoder.lstm.weight_hh_l1" in name:
                        param.requires_grad = True
                        console.print(f"  [green]Unfroze: {name}[/green]")

                # Recreate optimizer with hierarchical weight decay
                param_groups = [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if "encoder" in n and p.requires_grad
                        ],
                        "lr": lr_unfrozen,
                        "weight_decay": l2_encoder,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if "encoder" not in n and p.requires_grad
                        ],
                        "lr": learning_rate,
                        "weight_decay": 1e-4,
                    },
                ]
                optimizer = torch.optim.AdamW(param_groups)
                console.print(
                    f"  [green]Encoder LR: {lr_unfrozen}, weight_decay: {l2_encoder}[/green]"
                )
                console.print(f"  [green]Heads LR: {learning_rate}, weight_decay: 1e-4[/green]")

            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                focal_gamma=focal_gamma,
                use_pcgrad=use_pcgrad,
                pcgrad_threshold=pcgrad_threshold,
                conflict_monitor=conflict_monitor,
            )
            val_metrics = evaluate(model, val_loader, device, focal_gamma=focal_gamma)

            console.print(
                f"  Epoch {epoch+1}/{n_epochs}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_acc={val_metrics['accuracy']:.3f}, "
                f"val_f1={val_metrics['f1_macro']:.3f}, "
                f"joint@3={val_metrics['joint_success_at_3']:.3f}"
            )

            # Early stopping
            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                best_metrics = val_metrics.copy()
                patience_counter = 0

                # Save best checkpoint
                checkpoint_path = output_path / f"fold{fold_idx}_best.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "val_metrics": val_metrics,
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    console.print(f"[yellow]Early stopping at epoch {epoch+1}[/yellow]")
                    break

        # Log final conflict analysis for this fold
        if use_pcgrad and conflict_monitor is not None:
            console.print(
                f"\n[bold yellow]Final Gradient Conflict Analysis - Fold {fold_idx + 1}[/bold yellow]"
            )
            conflict_monitor.log_summary()

        # Store fold results with comprehensive metrics
        fold_result = {
            "fold": fold_idx + 1,
            "best_metrics": best_metrics,
        }

        # Add conflict statistics if PCGrad was used
        if use_pcgrad and conflict_monitor is not None:
            conflict_summary = conflict_monitor.get_summary()
            fold_result["conflict_stats"] = conflict_summary

        fold_results.append(fold_result)

        console.print(f"[green]✓ Fold {fold_idx + 1} complete[/green]")
        console.print(
            f"[green]  Best F1: {best_metrics['f1_macro']:.3f}, Joint@3: {best_metrics['joint_success_at_3']:.3f}[/green]"
        )

    # Compute average metrics across all folds
    metric_keys = list(fold_results[0]["best_metrics"].keys())
    avg_metrics = {}

    for key in metric_keys:
        values = [r["best_metrics"][key] for r in fold_results]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f"{key}_std"] = np.std(values)

    # Print comprehensive cross-validation summary
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]Cross-Validation Summary ({n_folds} folds)[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

    summary_table = create_metrics_table(avg_metrics, title="Average Metrics Across Folds")
    console.print(summary_table)

    # Highlight key metrics
    console.print("\n[bold green]Key Metrics (mean ± std):[/bold green]")
    console.print(f"  F1 Macro: {avg_metrics['f1_macro']:.3f} ± {avg_metrics['f1_macro_std']:.3f}")
    console.print(f"  Accuracy: {avg_metrics['accuracy']:.3f} ± {avg_metrics['accuracy_std']:.3f}")
    console.print(
        f"  Joint Success@3: {avg_metrics['joint_success_at_3']:.3f} ± {avg_metrics['joint_success_at_3_std']:.3f}"
    )
    console.print(
        f"  Center MAE: {avg_metrics['center_mae']:.2f} ± {avg_metrics['center_mae_std']:.2f} bars"
    )
    console.print(f"  Hit@±3: {avg_metrics['hit_at_3']:.3f} ± {avg_metrics['hit_at_3_std']:.3f}")
    console.print(f"  AUROC: {avg_metrics['auroc']:.3f} ± {avg_metrics['auroc_std']:.3f}")
    console.print(f"  ECE: {avg_metrics['ece']:.4f} ± {avg_metrics['ece_std']:.4f}")

    # Save comprehensive summary
    summary = {
        "avg_metrics": avg_metrics,
        "fold_results": fold_results,
        "config": {
            "pretrained_encoder": pretrained_encoder,
            "freeze_encoder": freeze_encoder,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "focal_gamma": focal_gamma,
            "use_augmentation": use_augmentation,
            "use_pcgrad": use_pcgrad,
            "pcgrad_threshold": pcgrad_threshold,
        },
    }

    summary_path = output_path / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[green]✓ Saved comprehensive summary to {summary_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Jade on labeled data")
    parser.add_argument("--data", required=True, help="Path to train_latest.parquet")
    parser.add_argument(
        "--pretrained-encoder", default=None, help="Path to pre-trained encoder checkpoint"
    )
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder weights")
    parser.add_argument(
        "--unfreeze-epoch",
        type=int,
        default=None,
        help="Epoch to unfreeze last encoder layer (None=keep frozen)",
    )
    parser.add_argument("--lr-unfrozen", type=float, default=1e-5, help="LR for unfrozen encoder")
    parser.add_argument(
        "--l2-encoder", type=float, default=1e-3, help="L2 regularization for encoder"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--focal-gamma", type=float, default=0.0, help="Focal loss gamma (0=CE)")
    parser.add_argument(
        "--use-augmentation",
        action="store_true",
        help="Enable jitter augmentation (σ=0.03, 3x multiplier)",
    )
    parser.add_argument(
        "--use-pcgrad",
        action="store_true",
        default=True,
        help="Enable PCGrad (Projecting Conflicting Gradients) for multi-task learning (default: True)",
    )
    parser.add_argument(
        "--no-pcgrad", dest="use_pcgrad", action="store_false", help="Disable PCGrad"
    )
    parser.add_argument(
        "--pcgrad-threshold",
        type=float,
        default=-0.3,
        help="Apply PCGrad when cosine similarity < threshold (default: -0.3)",
    )
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
        use_pcgrad=args.use_pcgrad,
        pcgrad_threshold=args.pcgrad_threshold,
        device=args.device,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()
