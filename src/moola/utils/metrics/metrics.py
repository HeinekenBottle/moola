"""Metric computation utilities for model evaluation and calibration."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
    if y_true.dtype.kind in ["U", "S", "O"]:  # Unicode, bytes, or object
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


def calculate_metrics_pack(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    class_names: list = None,
) -> dict:
    """Calculate comprehensive evaluation metrics pack.

    Args:
        y_true: True labels (shape: N,)
        y_pred: Predicted labels (shape: N,)
        y_proba: Predicted probabilities (shape: N, n_classes) - required for calibration
        class_names: Optional class names for reporting

    Returns:
        Dictionary with:
            - accuracy: Overall accuracy
            - precision_macro: Macro-averaged precision
            - recall_macro: Macro-averaged recall
            - f1_macro: Macro-averaged F1 score
            - f1_per_class: F1 score per class (list)
            - pr_auc: Precision-Recall AUC (macro-averaged)
            - brier: Brier score (calibration quality)
            - ece: Expected Calibration Error
            - log_loss: Log loss (cross-entropy)
    """
    import logging

    from sklearn.metrics import auc as compute_auc
    from sklearn.metrics import brier_score_loss, precision_recall_curve

    logger = logging.getLogger(__name__)

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class F1
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "f1_per_class": f1_per_class,
    }

    # Add class names if provided
    if class_names:
        metrics["class_names"] = class_names
        metrics["f1_by_class"] = {name: float(f1) for name, f1 in zip(class_names, f1_per_class)}

    # Probability-based metrics (require y_proba)
    if y_proba is not None:
        # Ensure y_proba is 2D
        if y_proba.ndim == 1:
            # Binary case: convert to 2D
            y_proba = np.column_stack([1 - y_proba, y_proba])

        n_classes = y_proba.shape[1]

        # PR-AUC (macro-averaged for multiclass)
        pr_auc_scores = []
        for class_idx in range(n_classes):
            # One-vs-rest for this class
            y_true_binary = (y_true == class_idx).astype(int)
            y_proba_class = y_proba[:, class_idx]

            precision, recall, _ = precision_recall_curve(y_true_binary, y_proba_class)
            pr_auc = compute_auc(recall, precision)
            pr_auc_scores.append(pr_auc)

        metrics["pr_auc"] = float(np.mean(pr_auc_scores))
        metrics["pr_auc_per_class"] = [float(x) for x in pr_auc_scores]

        # Brier score (multiclass)
        # For multiclass, use mean over classes
        y_true_onehot = np.eye(n_classes)[y_true]
        brier = np.mean((y_proba - y_true_onehot) ** 2)
        metrics["brier"] = float(brier)

        # ECE (Expected Calibration Error)
        ece = calculate_ece(y_true, y_proba)
        metrics["ece"] = float(ece)

        # Log loss
        try:
            logloss = log_loss(y_true, y_proba)
            metrics["log_loss"] = float(logloss)
        except Exception as e:
            logger.warning(f"Could not compute log loss: {e}")
            metrics["log_loss"] = None

    else:
        logger.warning("y_proba not provided - skipping calibration metrics (PR-AUC, Brier, ECE)")
        metrics["pr_auc"] = None
        metrics["brier"] = None
        metrics["ece"] = None
        metrics["log_loss"] = None

    return metrics


def compute_pointer_metrics(
    start_preds: np.ndarray,
    end_preds: np.ndarray,
    start_true: np.ndarray,
    end_true: np.ndarray,
    k: int = 3,
) -> dict:
    """Compute comprehensive pointer detection metrics for expansion localization.

    This function evaluates how well a model can identify the start and end points
    of market expansions within a sliding window. Multiple complementary metrics
    provide a complete picture of pointer prediction quality.

    Args:
        start_preds: Predicted start probabilities [N, 45]
            Each row contains probability distribution over 45 timesteps
            Output from sigmoid(model_logits), values in [0, 1]
        end_preds: Predicted end probabilities [N, 45]
            Same format as start_preds
        start_true: True start indices [N]
            Ground truth pointer locations, integers in range [0, 44]
        end_true: True end indices [N]
            Ground truth pointer locations, integers in range [0, 44]
        k: Top-k for precision@k metric (default 3)
            Number of top predictions to consider for "near miss" scoring

    Returns:
        Dictionary containing 8 pointer prediction metrics:
        {
            # Ranking metrics (how well can model discriminate pointer vs non-pointer?)
            'start_auc': float [0-1],       # AUC-ROC for start detection
            'end_auc': float [0-1],         # AUC-ROC for end detection

            # Top-k metrics (is true pointer in top-k predictions?)
            'start_precision_at_k': float [0-1],  # % samples with true start in top-k
            'end_precision_at_k': float [0-1],    # % samples with true end in top-k

            # Exact accuracy (did model predict exact timestep?)
            'start_exact_accuracy': float [0-1],  # % exact start matches
            'end_exact_accuracy': float [0-1],    # % exact end matches

            # Localization error (how far off is prediction?)
            'avg_start_error': float,      # Mean absolute error in timesteps
            'avg_end_error': float         # Mean absolute error in timesteps
        }

    Metrics Explained:
        AUC (Area Under ROC Curve):
            - Treats each timestep as binary classification: "Is this the pointer?"
            - Measures ranking quality: Can model rank true pointer higher than others?
            - 0.5 = random guessing, 1.0 = perfect ranking
            - Robust to class imbalance (1 positive vs 44 negatives per sample)

        Precision@k (Top-k Accuracy):
            - More lenient than exact match: "Is true pointer in top-k predictions?"
            - Useful for applications where approximate localization is acceptable
            - k=3 means: did model identify correct region (within ~7% of window)?

        Exact Accuracy:
            - Strictest metric: predicted timestep must match ground truth exactly
            - Equivalent to argmax(predictions) == true_index
            - Baseline for random guessing: 1/45 ≈ 2.2%

        Average Error:
            - Measures localization precision in physical units (timesteps)
            - |argmax(predictions) - true_index|
            - Lower is better, 0 = perfect localization
            - More interpretable than probability-based metrics

    Example:
        >>> # Model predictions for batch of 10 samples
        >>> start_preds = np.random.rand(10, 45)  # Sigmoid outputs
        >>> end_preds = np.random.rand(10, 45)
        >>> start_true = np.array([5, 12, 8, 15, 3, 20, 10, 7, 18, 25])
        >>> end_true = np.array([20, 30, 25, 35, 18, 40, 28, 22, 35, 42])
        >>> metrics = compute_pointer_metrics(start_preds, end_preds, start_true, end_true, k=3)
        >>> print(f"Start AUC: {metrics['start_auc']:.3f}")
        >>> print(f"End Precision@3: {metrics['end_precision_at_k']:.1%}")

    Notes:
        - All predictions should be probabilities (post-sigmoid), not logits
        - Pointer indices are relative to inner window [0, 45)
        - Metrics are averaged across all samples in batch
        - Handles edge cases (empty predictions, ties in argmax)
        - For random baseline: AUC≈0.5, Precision@3≈6.7%, Exact≈2.2%
    """
    n_samples = start_true.shape[0]
    inner_window_size = 45

    # Validate inputs
    assert start_preds.shape == (
        n_samples,
        inner_window_size,
    ), f"start_preds shape mismatch: expected ({n_samples}, {inner_window_size}), got {start_preds.shape}"
    assert end_preds.shape == (
        n_samples,
        inner_window_size,
    ), f"end_preds shape mismatch: expected ({n_samples}, {inner_window_size}), got {end_preds.shape}"
    assert start_true.shape == (
        n_samples,
    ), f"start_true shape mismatch: expected ({n_samples},), got {start_true.shape}"
    assert end_true.shape == (
        n_samples,
    ), f"end_true shape mismatch: expected ({n_samples},), got {end_true.shape}"

    # 1. AUC Computation (ranking quality)
    # Flatten to [N*45] for sklearn
    start_preds_flat = start_preds.flatten()
    end_preds_flat = end_preds.flatten()

    # Create binary labels: 1 at true pointer index, 0 elsewhere
    start_labels_flat = np.zeros(n_samples * inner_window_size)
    end_labels_flat = np.zeros(n_samples * inner_window_size)

    for i in range(n_samples):
        start_labels_flat[i * inner_window_size + start_true[i]] = 1
        end_labels_flat[i * inner_window_size + end_true[i]] = 1

    # Compute AUC (handles rare case of all same label gracefully)
    try:
        start_auc = roc_auc_score(start_labels_flat, start_preds_flat)
    except ValueError:
        # All labels are same class (shouldn't happen with proper data)
        start_auc = 0.5

    try:
        end_auc = roc_auc_score(end_labels_flat, end_preds_flat)
    except ValueError:
        end_auc = 0.5

    # 2. Precision@k (top-k accuracy)
    # Get top-k predictions for each sample
    start_topk_indices = np.argsort(start_preds, axis=1)[:, -k:]  # [N, k]
    end_topk_indices = np.argsort(end_preds, axis=1)[:, -k:]

    # Check if true index is in top-k
    start_in_topk = np.array([start_true[i] in start_topk_indices[i] for i in range(n_samples)])
    end_in_topk = np.array([end_true[i] in end_topk_indices[i] for i in range(n_samples)])

    start_precision_at_k = start_in_topk.mean()
    end_precision_at_k = end_in_topk.mean()

    # 3. Exact Accuracy (argmax match)
    start_predictions = np.argmax(start_preds, axis=1)  # [N]
    end_predictions = np.argmax(end_preds, axis=1)

    start_exact_accuracy = (start_predictions == start_true).mean()
    end_exact_accuracy = (end_predictions == end_true).mean()

    # 4. Average Error (localization precision)
    avg_start_error = np.abs(start_predictions - start_true).mean()
    avg_end_error = np.abs(end_predictions - end_true).mean()

    return {
        # Ranking metrics
        "start_auc": start_auc,
        "end_auc": end_auc,
        # Top-k metrics
        "start_precision_at_k": start_precision_at_k,
        "end_precision_at_k": end_precision_at_k,
        # Exact accuracy
        "start_exact_accuracy": start_exact_accuracy,
        "end_exact_accuracy": end_exact_accuracy,
        # Localization error
        "avg_start_error": avg_start_error,
        "avg_end_error": avg_end_error,
    }
