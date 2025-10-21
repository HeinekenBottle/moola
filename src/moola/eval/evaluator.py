"""Evaluation protocols for multi-task models.

Implements comprehensive evaluation including calibration, uncertainty,
and task-specific metrics.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

# import matplotlib.pyplot as plt


# Placeholder imports until modules are properly integrated
def hit_at_k(pred_center, true_center, k=3):
    pred_center_np = pred_center.detach().cpu().numpy()
    true_center_np = true_center.detach().cpu().numpy()
    pred_idx = (pred_center_np * 104).astype(int)
    true_idx = (true_center_np * 104).astype(int)
    hits = np.abs(pred_idx - true_idx) <= k
    return hits.mean()


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


class MultiTaskEvaluator:
    """Comprehensive evaluator for multi-task models.

    Evaluates classification, pointer regression, and joint performance.
    """

    def __init__(self, device: str = "cuda"):
        """Initialize evaluator.

        Args:
            device: Device to run evaluation on
        """
        self.device = device

    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        return_predictions: bool = False,
    ) -> Dict[str, Any]:
        """Comprehensive evaluation.

        Args:
            model: Trained model
            test_loader: Test data loader
            return_predictions: Whether to return raw predictions

        Returns:
            Comprehensive evaluation results
        """
        model.eval()

        # Collect all predictions and targets
        all_type_preds = []
        all_type_targets = []
        all_pointer_preds = []
        all_pointer_targets = []
        all_type_probs = []

        with torch.no_grad():
            for batch in test_loader:
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
                outputs = model(inputs)

                # Extract outputs
                if isinstance(outputs, dict):
                    type_logits = outputs["type"]
                    pointer_pred = outputs["pointers"]
                else:
                    type_logits, pointer_pred = outputs

                # Convert to probabilities
                type_probs = torch.softmax(type_logits, dim=1)
                type_pred_labels = type_logits.argmax(dim=1)

                # Accumulate
                all_type_preds.append(type_pred_labels.cpu())
                all_type_targets.append(type_targets.cpu())
                all_pointer_preds.append(pointer_pred.cpu())
                all_pointer_targets.append(pointer_targets.cpu())
                all_type_probs.append(type_probs.cpu())

        # Concatenate all predictions
        all_type_preds = torch.cat(all_type_preds, dim=0)
        all_type_targets = torch.cat(all_type_targets, dim=0)
        all_pointer_preds = torch.cat(all_pointer_preds, dim=0)
        all_pointer_targets = torch.cat(all_pointer_targets, dim=0)
        all_type_probs = torch.cat(all_type_probs, dim=0)

        # Compute comprehensive metrics
        results = self._compute_metrics(
            all_type_preds, all_type_targets, all_pointer_preds, all_pointer_targets, all_type_probs
        )

        if return_predictions:
            results["predictions"] = {
                "type_preds": all_type_preds.numpy(),
                "type_targets": all_type_targets.numpy(),
                "type_probs": all_type_probs.numpy(),
                "pointer_preds": all_pointer_preds.numpy(),
                "pointer_targets": all_pointer_targets.numpy(),
            }

        return results

    def _compute_metrics(
        self,
        type_preds: torch.Tensor,
        type_targets: torch.Tensor,
        pointer_preds: torch.Tensor,
        pointer_targets: torch.Tensor,
        type_probs: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compute all evaluation metrics.

        Args:
            type_preds: Type predictions [batch]
            type_targets: Type targets [batch]
            pointer_preds: Pointer predictions [batch, 2]
            pointer_targets: Pointer targets [batch, 2]
            type_probs: Type probabilities [batch, n_classes]

        Returns:
            Dictionary of all metrics
        """
        # Convert to numpy
        type_preds_np = type_preds.numpy()
        type_targets_np = type_targets.numpy()
        pointer_preds_np = pointer_preds.numpy()
        pointer_targets_np = pointer_targets.numpy()
        type_probs_np = type_probs.numpy()

        # Basic classification metrics
        accuracy = (type_preds_np == type_targets_np).mean()

        # Detailed classification report
        class_report = classification_report(
            type_targets_np,
            type_preds_np,
            target_names=["Consolidation", "Retracement"],
            output_dict=True,
            zero_division="0",
        )

        # Confusion matrix
        cm = confusion_matrix(type_targets_np, type_preds_np)

        # Pointer metrics
        center_pred = pointer_preds[:, 0]
        length_pred = pointer_preds[:, 1]
        center_true = pointer_targets[:, 0]
        length_true = pointer_targets[:, 1]

        # Hit accuracies at different tolerances
        hit_1 = hit_at_k(center_pred, center_true, k=1)
        hit_3 = hit_at_k(center_pred, center_true, k=3)
        hit_5 = hit_at_k(center_pred, center_true, k=5)

        # Pointer regression errors
        center_mae = np.abs(center_pred - center_true).mean()
        length_mae = np.abs(length_pred - length_true).mean()

        # Joint success metrics
        joint_metrics = compute_joint_success_metrics(
            type_probs, type_targets, center_pred, length_pred, center_true, length_true
        )

        # Class-specific performance
        class_0_mask = type_targets_np == 0
        class_1_mask = type_targets_np == 1

        class_0_accuracy = (
            (type_preds_np[class_0_mask] == type_targets_np[class_0_mask]).mean()
            if class_0_mask.any()
            else 0.0
        )
        class_1_accuracy = (
            (type_preds_np[class_1_mask] == type_targets_np[class_1_mask]).mean()
            if class_1_mask.any()
            else 0.0
        )

        # Pointer performance by class
        class_0_center_mae = (
            np.abs(center_pred[class_0_mask] - center_true[class_0_mask]).mean()
            if class_0_mask.any()
            else 0.0
        )
        class_1_center_mae = (
            np.abs(center_pred[class_1_mask] - center_true[class_1_mask]).mean()
            if class_1_mask.any()
            else 0.0
        )

        return {
            # Overall metrics
            "accuracy": accuracy,
            "joint_success_rate": joint_metrics["joint_success_rate"],
            # Classification metrics
            "classification_report": class_report,
            "confusion_matrix": cm.tolist(),
            "class_0_accuracy": class_0_accuracy,  # Consolidation
            "class_1_accuracy": class_1_accuracy,  # Retracement
            # Pointer metrics
            "center_mae": center_mae,
            "length_mae": length_mae,
            "hit@±1": hit_1,
            "hit@±3": hit_3,
            "hit@±5": hit_5,
            # Class-specific pointer performance
            "class_0_center_mae": class_0_center_mae,
            "class_1_center_mae": class_1_center_mae,
            # Joint metrics
            "joint_metrics": joint_metrics,
        }

    def evaluate_calibration(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """Evaluate model calibration.

        Args:
            model: Trained model
            val_loader: Validation data loader
            n_bins: Number of confidence bins

        Returns:
            Calibration metrics
        """
        model.eval()

        all_probs = []
        all_targets = []

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
                else:
                    type_targets = targets[:, 0].long()

                # Forward pass
                outputs = model(inputs)

                # Extract outputs
                if isinstance(outputs, dict):
                    type_logits = outputs["type"]
                else:
                    type_logits, _ = outputs

                # Convert to probabilities
                type_probs = torch.softmax(type_logits, dim=1)

                all_probs.append(type_probs.cpu())
                all_targets.append(type_targets.cpu())

        # Concatenate
        all_probs = torch.cat(all_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute Expected Calibration Error (ECE)
        ece = self._compute_ece(all_probs, all_targets, n_bins)

        # Compute Maximum Calibration Error (MCE)
        mce = self._compute_mce(all_probs, all_targets, n_bins)

        return {
            "ece": ece,
            "mce": mce,
            "n_bins": n_bins,
        }

    def _compute_ece(self, probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        confidence, pred_class = probs.max(dim=1)
        accuracy = (pred_class == targets).float()

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracy[in_bin].mean()
                avg_confidence_in_bin = confidence[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(ece)

    def _compute_mce(self, probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
        """Compute Maximum Calibration Error."""
        confidence, pred_class = probs.max(dim=1)
        accuracy = (pred_class == targets).float()

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracy[in_bin].mean()
                avg_confidence_in_bin = confidence[in_bin].mean()
                bin_error = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                mce = max(mce, float(bin_error))

        return float(mce)
