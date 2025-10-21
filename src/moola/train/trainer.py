"""Training pipeline for multi-task models.

Implements training loops with uncertainty-weighted loss and augmentation.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader


# Placeholder imports until modules are properly integrated
class UncertaintyWeightedLoss:
    def __init__(self, init_log_var=0.0):
        self.log_var_type = torch.tensor(init_log_var, requires_grad=True)
        self.log_var_pointer = torch.tensor(init_log_var, requires_grad=True)

    def __call__(self, type_loss, pointer_loss):
        weighted_type = (1.0 / (2.0 * torch.exp(self.log_var_type))) * type_loss
        weighted_pointer = (1.0 / (2.0 * torch.exp(self.log_var_pointer))) * pointer_loss
        total_loss = weighted_type + self.log_var_pointer + weighted_pointer + self.log_var_type
        loss_metrics = {
            "type_sigma": torch.exp(self.log_var_type).item(),
            "pointer_sigma": torch.exp(self.log_var_pointer).item(),
        }
        return total_loss, loss_metrics


class Jitter:
    def __init__(self, sigma=0.03, prob=0.8):
        self.sigma = sigma
        self.prob = prob
        self.training = True

    def __call__(self, x):
        if self.training and torch.rand(1) < self.prob:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x


class MagnitudeWarp:
    def __init__(self, sigma=0.2, knots=4, prob=0.5):
        self.sigma = sigma
        self.knots = knots
        self.prob = prob
        self.training = True

    def __call__(self, x):
        if self.training and torch.rand(1) < self.prob:
            # Simple magnitude scaling for now
            scale = 1 + torch.randn(1) * self.sigma
            return x * scale
        return x


def compute_joint_success_metrics(
    type_pred, type_true, pred_center, pred_length, true_center, true_length, center_tolerance=0.03
):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    type_pred_labels = type_pred.argmax(dim=1).cpu().numpy()
    type_true_np = type_true.cpu().numpy()

    accuracy = accuracy_score(type_true_np, type_pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        type_true_np, type_pred_labels, average="binary", zero_division="0"
    )

    center_error = torch.abs(pred_center.cpu() - true_center.cpu())
    length_error = torch.abs(pred_length.cpu() - true_length.cpu())

    type_correct = type_pred_labels == type_true_np
    center_correct = center_error <= center_tolerance
    length_correct = length_error <= 0.1

    joint_success = type_correct & center_correct & length_correct

    return {
        "type_accuracy": accuracy,
        "center_hit_rate": center_correct.float().mean().item(),
        "length_hit_rate": length_correct.float().mean().item(),
        "joint_success_rate": joint_success.mean(),
    }


class MultiTaskTrainer:
    """Trainer for multi-task models with uncertainty weighting.

    Handles classification + pointer regression with proper augmentation
    and loss balancing.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        use_amp: bool = True,
        jitter_prob: float = 0.8,
        jitter_sigma: float = 0.03,
        mag_warp_prob: float = 0.5,
        mag_warp_sigma: float = 0.2,
    ):
        """Initialize trainer.

        Args:
            model: Multi-task model
            device: Training device
            use_amp: Use automatic mixed precision
            jitter_prob: Probability of jitter augmentation
            jitter_sigma: Jitter noise std
            mag_warp_prob: Probability of magnitude warping
            mag_warp_sigma: Magnitude warp std
        """
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()

        # Augmentation
        self.jitter = Jitter(sigma=jitter_sigma, prob=jitter_prob)
        self.mag_warp = MagnitudeWarp(sigma=mag_warp_sigma, prob=mag_warp_prob)

        # Loss function
        self.uncertainty_loss = UncertaintyWeightedLoss()

        # AMP scaler
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            epoch: Current epoch number

        Returns:
            Training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_type_loss = 0.0
        total_pointer_loss = 0.0
        n_batches = 0
        last_loss_metrics = {}

        for batch_idx, batch in enumerate(train_loader):
            # Extract data (adjust for your data format)
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs = batch.x
                targets = batch.y

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Extract multi-task targets
            if hasattr(targets, "type"):
                type_targets = targets.type
                pointer_targets = targets.pointers
            else:
                # Assume targets format: [type, center, length]
                type_targets = targets[:, 0].long()
                pointer_targets = targets[:, 1:3]

            optimizer.zero_grad()

            # Forward pass with augmentation
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # Apply augmentation
                    inputs_aug = self.jitter(inputs)
                    inputs_aug = self.mag_warp(inputs_aug)

                    # Model forward
                    outputs = self.model(inputs_aug)

                    # Extract outputs
                    if isinstance(outputs, dict):
                        type_logits = outputs["type"]
                        pointer_pred = outputs["pointers"]
                    else:
                        # Assume tuple format: (type_logits, pointer_pred)
                        type_logits, pointer_pred = outputs

                    # Compute losses
                    type_loss = nn.CrossEntropyLoss()(type_logits, type_targets)
                    pointer_loss = nn.MSELoss()(pointer_pred, pointer_targets)

                    # Uncertainty-weighted combined loss
                    total_loss_batch, loss_metrics = self.uncertainty_loss(type_loss, pointer_loss)
                    last_loss_metrics = loss_metrics

                # Backward pass with AMP
                self.scaler.scale(total_loss_batch).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.5)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Apply augmentation
                inputs_aug = self.jitter(inputs)
                inputs_aug = self.mag_warp(inputs_aug)

                # Model forward
                outputs = self.model(inputs_aug)

                # Extract outputs
                if isinstance(outputs, dict):
                    type_logits = outputs["type"]
                    pointer_pred = outputs["pointers"]
                else:
                    type_logits, pointer_pred = outputs

                # Compute losses
                type_loss = nn.CrossEntropyLoss()(type_logits, type_targets)
                pointer_loss = nn.MSELoss()(pointer_pred, pointer_targets)

                # Uncertainty-weighted combined loss
                total_loss_batch, loss_metrics = self.uncertainty_loss(type_loss, pointer_loss)

                # Backward pass
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.5)
                optimizer.step()

            # Accumulate metrics
            total_loss += total_loss_batch.item()
            total_type_loss += type_loss.item()
            total_pointer_loss += pointer_loss.item()
            n_batches += 1

            # Log progress
            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {total_loss_batch.item():.4f} "
                    f"Type: {type_loss.item():.4f} "
                    f"Pointer: {pointer_loss.item():.4f}"
                )

        # Return epoch metrics
        return {
            "loss": total_loss / n_batches,
            "type_loss": total_type_loss / n_batches,
            "pointer_loss": total_pointer_loss / n_batches,
            "type_sigma": last_loss_metrics.get("type_sigma", 0.0),
            "pointer_sigma": last_loss_metrics.get("pointer_sigma", 0.0),
        }

    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        all_type_preds = []
        all_type_targets = []
        all_pointer_preds = []
        all_pointer_targets = []

        with torch.no_grad():
            for batch in val_loader:
                # Extract data
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs = batch.x
                    targets = batch.y

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Extract targets
                if hasattr(targets, "type"):
                    type_targets = targets.type
                    pointer_targets = targets.pointers
                else:
                    type_targets = targets[:, 0].long()
                    pointer_targets = targets[:, 1:3]

                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                # Extract outputs
                if isinstance(outputs, dict):
                    type_logits = outputs["type"]
                    pointer_pred = outputs["pointers"]
                else:
                    type_logits, pointer_pred = outputs

                # Compute losses
                type_loss = nn.CrossEntropyLoss()(type_logits, type_targets)
                pointer_loss = nn.MSELoss()(pointer_pred, pointer_targets)
                total_loss_batch, _ = self.uncertainty_loss(type_loss, pointer_loss)

                # Accumulate
                total_loss += total_loss_batch.item()
                all_type_preds.append(type_logits.cpu())
                all_type_targets.append(type_targets.cpu())
                all_pointer_preds.append(pointer_pred.cpu())
                all_pointer_targets.append(pointer_targets.cpu())

        # Concatenate all predictions
        all_type_preds = torch.cat(all_type_preds, dim=0)
        all_type_targets = torch.cat(all_type_targets, dim=0)
        all_pointer_preds = torch.cat(all_pointer_preds, dim=0)
        all_pointer_targets = torch.cat(all_pointer_targets, dim=0)

        # Compute metrics
        joint_metrics = compute_joint_success_metrics(
            all_type_preds,
            all_type_targets,
            all_pointer_preds[:, 0],
            all_pointer_preds[:, 1],
            all_pointer_targets[:, 0],
            all_pointer_targets[:, 1],
        )

        return {
            "loss": total_loss / len(val_loader),
            "accuracy": joint_metrics["type_accuracy"],
            "joint_success_rate": joint_metrics["joint_success_rate"],
            "center_hit_rate": joint_metrics["center_hit_rate"],
            "length_hit_rate": joint_metrics["length_hit_rate"],
        }
