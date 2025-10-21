"""Pointer regression metrics for dual-task BiLSTM model.

Metrics for evaluating pointer head performance in expansion start/end prediction.
Used to monitor training progress and identify task silencing issues.

Usage:
    >>> from moola.utils.metrics.pointer_regression import compute_pointer_regression_metrics
    >>> metrics = compute_pointer_regression_metrics(
    ...     pred_start=pred[:, 0],
    ...     pred_end=pred[:, 1],
    ...     true_start=targets[:, 0],
    ...     true_end=targets[:, 1],
    ...     tolerance=3
    ... )
    >>> print(f"Hit@±3: {metrics['hit_at_pm3']:.1%}")
"""

import numpy as np


def compute_pointer_regression_metrics(
    pred_start: np.ndarray,
    pred_end: np.ndarray,
    true_start: np.ndarray,
    true_end: np.ndarray,
    tolerance: int = 3,
) -> dict:
    """Compute metrics for pointer regression task.

    Args:
        pred_start: Predicted start indices [N]
        pred_end: Predicted end indices [N]
        true_start: True start indices [N]
        true_end: True end indices [N]
        tolerance: Tolerance for hit rate calculation (default: 3 timesteps)

    Returns:
        dict with keys:
            - start_mae: Mean absolute error for start pointer
            - end_mae: Mean absolute error for end pointer
            - center_mae: MAE for center position (more interpretable)
            - length_mae: MAE for span length (detects scaling issues)
            - hit_at_pm3: % of predictions within ±3 timesteps (both start AND end)
            - hit_at_pm5: % within ±5 timesteps (both start AND end)
            - exact_match: % exact matches (both start AND end correct)

    Example:
        >>> pred_start = np.array([10, 20, 30])
        >>> pred_end = np.array([50, 60, 70])
        >>> true_start = np.array([12, 19, 31])
        >>> true_end = np.array([52, 58, 69])
        >>> metrics = compute_pointer_regression_metrics(
        ...     pred_start, pred_end, true_start, true_end, tolerance=3
        ... )
        >>> assert 0 <= metrics['hit_at_pm3'] <= 1
    """
    # MAE for raw predictions
    start_mae = np.mean(np.abs(pred_start - true_start))
    end_mae = np.mean(np.abs(pred_end - true_end))

    # Center and length MAE (more interpretable)
    pred_center = (pred_start + pred_end) / 2
    true_center = (true_start + true_end) / 2
    center_mae = np.mean(np.abs(pred_center - true_center))

    pred_length = pred_end - pred_start
    true_length = true_end - true_start
    length_mae = np.mean(np.abs(pred_length - true_length))

    # Hit rates (IoU-style: both pointers must be within tolerance)
    start_hits_3 = np.abs(pred_start - true_start) <= 3
    end_hits_3 = np.abs(pred_end - true_end) <= 3
    both_hits_3 = start_hits_3 & end_hits_3
    hit_at_pm3 = np.mean(both_hits_3)

    start_hits_5 = np.abs(pred_start - true_start) <= 5
    end_hits_5 = np.abs(pred_end - true_end) <= 5
    both_hits_5 = start_hits_5 & end_hits_5
    hit_at_pm5 = np.mean(both_hits_5)

    exact_match = np.mean((pred_start == true_start) & (pred_end == true_end))

    return {
        "start_mae": float(start_mae),
        "end_mae": float(end_mae),
        "center_mae": float(center_mae),
        "length_mae": float(length_mae),
        "hit_at_pm3": float(hit_at_pm3),
        "hit_at_pm5": float(hit_at_pm5),
        "exact_match": float(exact_match),
    }
