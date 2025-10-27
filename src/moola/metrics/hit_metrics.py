"""Hit accuracy metrics for pointer evaluation.

Computes Hit@±3 and other temporal localization metrics.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def hit_at_k(pred_center: torch.Tensor, true_center: torch.Tensor, k: int = 3) -> float:
    """Compute Hit@±k accuracy for center predictions.

    Args:
        pred_center: Predicted center values [batch] in range [0, 1]
        true_center: True center values [batch] in range [0, 1]
        k: Tolerance window in timesteps (default: 3)

    Returns:
        Hit@±k accuracy
    """
    pred_center_np = pred_center.detach().cpu().numpy()
    true_center_np = true_center.detach().cpu().numpy()

    # Convert tolerance from timesteps to normalized units
    # seq_len = 104 for 105 timesteps (0-based indices)
    tolerance = k / 104.0

    # Check if prediction is within ±k timesteps of true position
    hits = np.abs(pred_center_np - true_center_np) <= tolerance
    return hits.mean()


def compute_pointer_metrics(
    pred_center: torch.Tensor,
    pred_length: torch.Tensor,
    true_center: torch.Tensor,
    true_length: torch.Tensor,
    center_weight: float = 1.0,
    length_weight: float = 0.8,
) -> dict:
    """Compute comprehensive pointer regression metrics.

    Args:
        pred_center: Predicted center values [batch]
        pred_length: Predicted length values [batch]
        true_center: True center values [batch]
        true_length: True length values [batch]
        center_weight: Weight for center error
        length_weight: Weight for length error

    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    pred_center = pred_center.cpu().numpy()
    pred_length = pred_length.cpu().numpy()
    true_center = true_center.cpu().numpy()
    true_length = true_length.cpu().numpy()

    # Compute errors
    center_error = np.abs(pred_center - true_center)
    length_error = np.abs(pred_length - true_length)

    # Weighted combined error
    combined_error = center_weight * center_error + length_weight * length_error

    # Hit accuracies at different tolerances (convert timesteps to normalized units)
    hit_1 = (center_error <= (1 / 104)).mean()  # ±1 timestep
    hit_3 = (center_error <= (3 / 104)).mean()  # ±3 timesteps
    hit_5 = (center_error <= (5 / 104)).mean()  # ±5 timesteps

    return {
        "center_mae": center_error.mean(),
        "length_mae": length_error.mean(),
        "combined_mae": combined_error.mean(),
        "hit@±1": hit_1,
        "hit@±3": hit_3,
        "hit@±5": hit_5,
        "center_rmse": np.sqrt((center_error**2).mean()),
        "length_rmse": np.sqrt((length_error**2).mean()),
    }


def compute_joint_success_metrics(
    type_pred: torch.Tensor,
    type_true: torch.Tensor,
    pred_center: torch.Tensor,
    pred_length: torch.Tensor,
    true_center: torch.Tensor,
    true_length: torch.Tensor,
    center_tolerance: float = 0.03,  # ±3 timesteps
) -> dict:
    """Compute joint success metrics for both tasks.

    Both classification and localization must be correct.

    Args:
        type_pred: Type predictions [batch, n_classes]
        type_true: True type labels [batch]
        pred_center: Predicted center values [batch]
        pred_length: Predicted length values [batch]
        true_center: True center values [batch]
        true_length: True length values [batch]
        center_tolerance: Tolerance for center prediction

    Returns:
        Dictionary of joint metrics
    """
    # Classification metrics
    type_pred_labels = type_pred.argmax(dim=1).cpu().numpy()
    type_true_np = type_true.cpu().numpy()

    accuracy = accuracy_score(type_true_np, type_pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        type_true_np, type_pred_labels, average="binary", zero_division="0"
    )

    # Pointer metrics
    center_error = np.abs(pred_center.cpu().numpy() - true_center.cpu().numpy())
    length_error = np.abs(pred_length.cpu().numpy() - true_length.cpu().numpy())

    # Joint success: both correct
    type_correct = type_pred_labels == type_true_np
    center_correct = center_error <= center_tolerance
    length_correct = length_error <= 0.1  # 10% tolerance for length

    joint_success = type_correct & center_correct & length_correct

    return {
        "type_accuracy": accuracy,
        "type_precision": precision,
        "type_recall": recall,
        "type_f1": f1,
        "center_hit_rate": center_correct.mean(),
        "length_hit_rate": length_correct.mean(),
        "joint_success_rate": joint_success.mean(),
        "joint_accuracy": joint_success.mean(),  # Alias for consistency
    }
