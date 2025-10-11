"""Metric computation utilities for model evaluation and calibration."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score


def calculate_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error.

    ECE measures how well predicted probabilities match actual frequencies.
    Lower ECE indicates better calibration.

    Args:
        y_true: True labels [N]
        y_proba: Predicted probabilities [N, C]
        n_bins: Number of bins for calibration curve (default 10)

    Returns:
        ECE score (0 to 1, lower is better)
    """
    # Get predicted class and confidence
    y_pred_class = np.argmax(y_proba, axis=1)
    y_pred_conf = np.max(y_proba, axis=1)

    # Convert string labels to numeric if needed
    if y_true.dtype.kind in ['U', 'S', 'O']:  # Unicode, bytes, or object
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_true_encoded = le.fit_transform(y_true)
        y_pred_correct = (y_pred_class == y_true_encoded).astype(int)
    else:
        y_pred_correct = (y_pred_class == y_true).astype(int)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Samples in this confidence bin
        in_bin = (y_pred_conf > bin_lower) & (y_pred_conf <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            # Accuracy in this bin
            accuracy_in_bin = y_pred_correct[in_bin].mean()
            # Average confidence in this bin
            avg_conf_in_bin = y_pred_conf[in_bin].mean()
            # ECE contribution from this bin
            ece += np.abs(avg_conf_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
    """Calculate comprehensive evaluation metrics.

    Args:
        y_true: True labels [N]
        y_pred: Predicted labels [N]
        y_proba: Predicted probabilities [N, C] (optional, for ECE and log loss)

    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - precision: Macro-averaged precision
            - recall: Macro-averaged recall
            - f1: Macro-averaged F1 score
            - ece: Expected calibration error (if y_proba provided)
            - logloss: Log loss (if y_proba provided)
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    if y_proba is not None:
        metrics["ece"] = calculate_ece(y_true, y_proba)
        try:
            metrics["logloss"] = log_loss(y_true, y_proba)
        except ValueError:
            # Handle case where y_true and y_proba have mismatched classes
            metrics["logloss"] = None

    return metrics
