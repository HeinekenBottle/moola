"""Joint metrics for dual-task model evaluation (pointer + type classification).

Provides metrics where BOTH tasks must be correct for a prediction to count as correct.
This is critical for dual-task models where partial correctness is insufficient.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_joint_hit_accuracy(
    pred_start: np.ndarray,
    pred_end: np.ndarray,
    true_start: np.ndarray,
    true_end: np.ndarray,
    pred_type: np.ndarray,
    true_type: np.ndarray,
    tolerance: int = 3,
) -> Dict[str, float]:
    """Compute joint accuracy where BOTH pointer and type must be correct.

    A prediction is correct only if:
    1. Pointer start/end within tolerance (hit@±tolerance)
    2. Type classification is correct

    Args:
        pred_start: Predicted start positions (n_samples,)
        pred_end: Predicted end positions (n_samples,)
        true_start: True start positions (n_samples,)
        true_end: True end positions (n_samples,)
        pred_type: Predicted type labels (n_samples,)
        true_type: True type labels (n_samples,)
        tolerance: Hit tolerance in bars (default: 3)

    Returns:
        Dictionary with joint accuracy and component accuracies
    """
    # Pointer hit (both start and end within tolerance)
    start_hit = np.abs(pred_start - true_start) <= tolerance
    end_hit = np.abs(pred_end - true_end) <= tolerance
    pointer_hit = start_hit & end_hit

    # Type correctness
    type_correct = pred_type == true_type

    # Joint correctness (both must be correct)
    joint_correct = pointer_hit & type_correct

    return {
        "joint_accuracy": float(joint_correct.mean()),
        "pointer_hit_rate": float(pointer_hit.mean()),
        "type_accuracy": float(type_correct.mean()),
        "joint_correct_count": int(joint_correct.sum()),
        "total_samples": int(len(pred_start)),
    }


def compute_joint_hit_f1(
    pred_start: np.ndarray,
    pred_end: np.ndarray,
    true_start: np.ndarray,
    true_end: np.ndarray,
    pred_type: np.ndarray,
    true_type: np.ndarray,
    tolerance: int = 3,
    average: str = "weighted",
) -> Dict[str, float]:
    """Compute F1 score for joint task (pointer hit + type classification).

    Treats each combination of (type, pointer_hit) as a separate class:
    - Class 0: type=0, pointer_miss
    - Class 1: type=0, pointer_hit
    - Class 2: type=1, pointer_miss
    - Class 3: type=1, pointer_hit

    Args:
        pred_start: Predicted start positions
        pred_end: Predicted end positions
        true_start: True start positions
        true_end: True end positions
        pred_type: Predicted type labels (0 or 1)
        true_type: True type labels (0 or 1)
        tolerance: Hit tolerance in bars
        average: 'weighted', 'macro', or 'micro' for multi-class F1

    Returns:
        Dictionary with joint F1, precision, recall
    """
    # Compute pointer hits
    start_hit = np.abs(pred_start - true_start) <= tolerance
    end_hit = np.abs(pred_end - true_end) <= tolerance
    pred_pointer_hit = (start_hit & end_hit).astype(int)

    # True pointer hits (always hit for ground truth)
    true_pointer_hit = np.ones_like(true_start, dtype=int)

    # Create joint labels: (type * 2 + pointer_hit)
    # Class 0: type=0, pointer_miss
    # Class 1: type=0, pointer_hit
    # Class 2: type=1, pointer_miss
    # Class 3: type=1, pointer_hit
    pred_joint = pred_type * 2 + pred_pointer_hit
    true_joint = true_type * 2 + true_pointer_hit  # Always class 1 or 3 (hits)

    # Compute F1, precision, recall
    f1 = f1_score(true_joint, pred_joint, average=average, zero_division=0)
    precision = precision_score(true_joint, pred_joint, average=average, zero_division=0)
    recall = recall_score(true_joint, pred_joint, average=average, zero_division=0)

    return {
        "joint_f1": float(f1),
        "joint_precision": float(precision),
        "joint_recall": float(recall),
        "average": average,
    }


def compute_task_contribution_analysis(
    pred_start: np.ndarray,
    pred_end: np.ndarray,
    true_start: np.ndarray,
    true_end: np.ndarray,
    pred_type: np.ndarray,
    true_type: np.ndarray,
    tolerance: int = 3,
) -> Dict[str, Dict[str, float]]:
    """Analyze contribution of each task to joint performance.

    Breaks down joint accuracy by:
    - Both correct
    - Pointer correct, type wrong
    - Type correct, pointer wrong
    - Both wrong

    Args:
        pred_start, pred_end, true_start, true_end: Pointer predictions/targets
        pred_type, true_type: Type predictions/targets
        tolerance: Hit tolerance

    Returns:
        Dictionary with breakdown statistics
    """
    # Compute hits
    start_hit = np.abs(pred_start - true_start) <= tolerance
    end_hit = np.abs(pred_end - true_end) <= tolerance
    pointer_hit = start_hit & end_hit
    type_correct = pred_type == true_type

    # Four categories
    both_correct = pointer_hit & type_correct
    pointer_only = pointer_hit & ~type_correct
    type_only = ~pointer_hit & type_correct
    both_wrong = ~pointer_hit & ~type_correct

    n_samples = len(pred_start)

    return {
        "both_correct": {
            "count": int(both_correct.sum()),
            "fraction": float(both_correct.mean()),
        },
        "pointer_only_correct": {
            "count": int(pointer_only.sum()),
            "fraction": float(pointer_only.mean()),
        },
        "type_only_correct": {
            "count": int(type_only.sum()),
            "fraction": float(type_only.mean()),
        },
        "both_wrong": {
            "count": int(both_wrong.sum()),
            "fraction": float(both_wrong.mean()),
        },
        "total_samples": n_samples,
    }


def select_best_model_by_joint_metric(
    results: list,
    metric_name: str = "joint_f1",
    min_pointer_hit: float = 0.4,
    min_type_acc: float = 0.5,
) -> Tuple[dict, int]:
    """Select best model based on joint metric with minimum task thresholds.

    Args:
        results: List of result dictionaries with joint metrics
        metric_name: Joint metric to optimize ('joint_f1' or 'joint_accuracy')
        min_pointer_hit: Minimum pointer hit rate threshold
        min_type_acc: Minimum type accuracy threshold

    Returns:
        (best_result, best_index) tuple

    Raises:
        ValueError: If no results meet minimum thresholds
    """
    valid_results = []
    valid_indices = []

    for i, result in enumerate(results):
        # Check minimum thresholds
        if (
            result.get("pointer_hit_rate", 0) >= min_pointer_hit
            and result.get("type_accuracy", 0) >= min_type_acc
        ):
            valid_results.append(result)
            valid_indices.append(i)

    if not valid_results:
        raise ValueError(
            f"No results meet minimum thresholds: "
            f"pointer_hit>={min_pointer_hit}, type_acc>={min_type_acc}"
        )

    # Find best by joint metric
    best_idx = np.argmax([r[metric_name] for r in valid_results])

    return valid_results[best_idx], valid_indices[best_idx]
