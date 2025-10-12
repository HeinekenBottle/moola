"""Windowing utilities for time series prediction regions.

This module handles the proper windowing structure for financial time series:
- Left buffer (0-29): Context only, not for prediction
- Inner window (30-74): Active prediction zone (45 bars)
- Right buffer (75-104): Future context only

This ensures models are evaluated fairly on regions with sufficient context.
"""

from typing import Tuple

import numpy as np


# Window constants
BUFFER_LEFT = 30
INNER_WINDOW = 45
BUFFER_RIGHT = 30
TOTAL_WINDOW = 105


def get_prediction_indices() -> Tuple[int, int]:
    """Return (start, end) indices for inner prediction window.

    Returns:
        Tuple of (start_idx, end_idx) for the prediction region.
        For standard 105-bar windows: (30, 75)
    """
    return (BUFFER_LEFT, BUFFER_LEFT + INNER_WINDOW)


def mask_predictions(predictions: np.ndarray, window_size: int = 105) -> np.ndarray:
    """Mask predictions to only evaluate inner window.

    This function zeros out predictions outside the valid prediction region.
    Used for models that produce per-timestep predictions (like pointer networks).

    Args:
        predictions: Predictions array. Can be:
            - Shape [N, window_size, 2] for pointer predictions (start/end logits)
            - Shape [N, window_size] for per-timestep predictions
            - Shape [N, C] for already-aggregated class predictions (returned unchanged)
        window_size: Total window size (default: 105)

    Returns:
        Masked predictions with same shape as input.
        For per-timestep predictions, boundaries are zeroed out.
        For aggregated predictions, returned unchanged.
    """
    if predictions.ndim == 2 and predictions.shape[1] <= 10:
        # Already aggregated to class predictions [N, C] where C is small
        # No masking needed
        return predictions

    start_idx, end_idx = get_prediction_indices()

    # Create a copy to avoid modifying input
    masked = predictions.copy()

    if predictions.ndim == 3:
        # Pointer predictions [N, window_size, 2]
        # Zero out left buffer [0:start_idx] and right buffer [end_idx:]
        masked[:, :start_idx, :] = 0
        masked[:, end_idx:, :] = 0
    elif predictions.ndim == 2 and predictions.shape[1] == window_size:
        # Per-timestep predictions [N, window_size]
        masked[:, :start_idx] = 0
        masked[:, end_idx:] = 0

    return masked


def validate_expansion_indices(start_idx: int, end_idx: int) -> bool:
    """Check if expansion start/end are within valid inner window [30, 75).

    Args:
        start_idx: Starting index of expansion
        end_idx: Ending index of expansion (exclusive)

    Returns:
        True if expansion is fully within inner window, False otherwise.
    """
    pred_start, pred_end = get_prediction_indices()

    # Check if both indices are within the valid prediction region
    if start_idx < pred_start or start_idx >= pred_end:
        return False
    if end_idx <= pred_start or end_idx > pred_end:
        return False

    # Check that start comes before end
    if start_idx >= end_idx:
        return False

    return True


def get_window_regions(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split time series data into three regions: left buffer, inner window, right buffer.

    Args:
        data: Time series data of shape [N, 105, F] or [105, F]

    Returns:
        Tuple of (left_buffer, inner_window, right_buffer)
        Each has shape [N, region_size, F] or [region_size, F]
    """
    if data.ndim == 2:
        # Single sample [105, F]
        left_buffer = data[:BUFFER_LEFT, :]
        inner_window = data[BUFFER_LEFT:BUFFER_LEFT + INNER_WINDOW, :]
        right_buffer = data[BUFFER_LEFT + INNER_WINDOW:, :]
    elif data.ndim == 3:
        # Batch [N, 105, F]
        left_buffer = data[:, :BUFFER_LEFT, :]
        inner_window = data[:, BUFFER_LEFT:BUFFER_LEFT + INNER_WINDOW, :]
        right_buffer = data[:, BUFFER_LEFT + INNER_WINDOW:, :]
    else:
        raise ValueError(f"Expected 2D or 3D data, got shape {data.shape}")

    return left_buffer, inner_window, right_buffer


def compute_window_weights(window_size: int = 105) -> np.ndarray:
    """Compute attention weights for windowing.

    Returns weights that boost the inner prediction window relative to buffers.

    Args:
        window_size: Total window size (default: 105)

    Returns:
        Weights array of shape [window_size] with values:
        - 1.0 for buffer regions
        - 1.5 for inner prediction window (50% boost)
    """
    weights = np.ones(window_size, dtype=np.float32)
    start_idx, end_idx = get_prediction_indices()
    weights[start_idx:end_idx] = 1.5
    return weights
