"""Bootstrap confidence intervals for robust performance estimation on small datasets.

Implements bootstrap resampling and confidence interval computation for:
- Classification accuracy
- Pointer regression metrics (MAE, hit rates)
- Calibration metrics (ECE, Brier score)

Critical for small validation sets (e.g., 34 samples) where point estimates are unreliable.

Usage:
    >>> from moola.utils.metrics.bootstrap import bootstrap_accuracy
    >>> result = bootstrap_accuracy(y_true, y_pred, n_resamples=1000, confidence_level=0.95)
    >>> print(f"Accuracy: {result['mean']:.3f} [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
"""

import numpy as np
from typing import Dict, Callable, Tuple


def bootstrap_resample(
    y_true: np.ndarray, y_pred: np.ndarray, n_resamples: int = 1000, random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate bootstrap resamples for uncertainty estimation.

    Samples with replacement to create alternative datasets for computing confidence intervals.

    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        n_resamples: Number of bootstrap resamples
        random_seed: Random seed for reproducibility

    Returns:
        (resampled_true, resampled_pred) arrays of shape (n_resamples, n_samples)

    Example:
        >>> y_true = np.array([0, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 1, 1])
        >>> resampled_true, resampled_pred = bootstrap_resample(y_true, y_pred, n_resamples=10)
        >>> assert resampled_true.shape == (10, 4)
    """
    np.random.seed(random_seed)
    n_samples = len(y_true)

    resampled_true = []
    resampled_pred = []

    for _ in range(n_resamples):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        resampled_true.append(y_true[indices])
        resampled_pred.append(y_pred[indices])

    return np.array(resampled_true), np.array(resampled_pred)


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for a metric.

    Generic bootstrap wrapper for any metric function.

    Args:
        y_true: True labels
        y_pred: Predictions
        metric_fn: Function computing metric from (y_true, y_pred) -> float
        n_resamples: Number of bootstrap resamples
        confidence_level: Confidence level (default 95%)
        random_seed: Random seed

    Returns:
        Dictionary with mean, std, median, CI lower/upper bounds

    Example:
        >>> def accuracy(y_t, y_p): return (y_t == y_p).mean()
        >>> result = bootstrap_metric(y_true, y_pred, accuracy, n_resamples=100)
        >>> assert 'mean' in result and 'ci_lower' in result
    """
    # Generate resamples
    resampled_true, resampled_pred = bootstrap_resample(y_true, y_pred, n_resamples, random_seed)

    # Compute metric on each resample
    metric_values = []
    for true_sample, pred_sample in zip(resampled_true, resampled_pred):
        try:
            value = metric_fn(true_sample, pred_sample)
            metric_values.append(value)
        except Exception:
            # Skip invalid resamples (e.g., all same class for precision/recall)
            continue

    metric_values = np.array(metric_values)

    # Compute statistics
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    return {
        "mean": float(metric_values.mean()),
        "std": float(metric_values.std()),
        "median": float(np.median(metric_values)),
        "ci_lower": float(np.percentile(metric_values, lower_percentile)),
        "ci_upper": float(np.percentile(metric_values, upper_percentile)),
        "confidence_level": float(confidence_level),
    }


def bootstrap_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """Bootstrap confidence interval for classification accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_resamples: Number of bootstrap resamples
        confidence_level: Confidence level

    Returns:
        Dictionary with accuracy statistics and CI

    Example:
        >>> y_true = np.array([0, 1, 0, 1, 0])
        >>> y_pred = np.array([0, 1, 1, 1, 0])
        >>> result = bootstrap_accuracy(y_true, y_pred, n_resamples=100)
        >>> print(f"Accuracy: {result['mean']:.3f} [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
    """

    def accuracy(y_t, y_p):
        return (y_t == y_p).mean()

    return bootstrap_metric(y_true, y_pred, accuracy, n_resamples, confidence_level)


def bootstrap_pointer_metrics(
    pred_start: np.ndarray,
    pred_end: np.ndarray,
    true_start: np.ndarray,
    true_end: np.ndarray,
    tolerance: int = 3,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap confidence intervals for pointer regression metrics.

    Provides uncertainty estimates for dual-task BiLSTM pointer head performance.

    Args:
        pred_start: Predicted start positions
        pred_end: Predicted end positions
        true_start: True start positions
        true_end: True end positions
        tolerance: Hit tolerance in bars
        n_resamples: Number of bootstrap resamples
        confidence_level: Confidence level

    Returns:
        Dictionary of metrics, each with mean, std, median, CI bounds

    Example:
        >>> pred_start = np.array([10, 20, 30])
        >>> pred_end = np.array([50, 60, 70])
        >>> true_start = np.array([12, 19, 31])
        >>> true_end = np.array([52, 58, 69])
        >>> results = bootstrap_pointer_metrics(pred_start, pred_end, true_start, true_end)
        >>> print(f"Start MAE: {results['start_mae']['mean']:.2f}")
    """
    from moola.utils.metrics.pointer_regression import compute_pointer_regression_metrics

    n_samples = len(pred_start)
    np.random.seed(42)

    # Initialize metric collections
    metrics_collection = {
        "start_mae": [],
        "end_mae": [],
        "center_mae": [],
        "length_mae": [],
        "hit_at_pm3": [],
        "hit_at_pm5": [],
        "exact_match": [],
    }

    for _ in range(n_resamples):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # Compute metrics on resample
        metrics = compute_pointer_regression_metrics(
            pred_start[indices],
            pred_end[indices],
            true_start[indices],
            true_end[indices],
            tolerance=tolerance,
        )

        for key in metrics_collection:
            metrics_collection[key].append(metrics[key])

    # Compute statistics for each metric
    results = {}
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    for metric_name, values in metrics_collection.items():
        values = np.array(values)
        results[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "ci_lower": float(np.percentile(values, lower_percentile)),
            "ci_upper": float(np.percentile(values, upper_percentile)),
        }

    return results


def bootstrap_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap confidence intervals for calibration metrics.

    Quantifies uncertainty in ECE, MCE, and Brier score for small validation sets.

    Args:
        probs: Predicted probabilities
        labels: True labels
        n_resamples: Number of bootstrap resamples
        confidence_level: Confidence level

    Returns:
        Dictionary with ECE, MCE, Brier score statistics

    Example:
        >>> probs = np.array([0.1, 0.4, 0.6, 0.9])
        >>> labels = np.array([0, 0, 1, 1])
        >>> results = bootstrap_calibration_metrics(probs, labels, n_resamples=100)
        >>> print(f"ECE: {results['ece']['mean']:.4f} ± {results['ece']['std']:.4f}")
    """
    from moola.utils.metrics.calibration import compute_smooth_ece, compute_brier_score

    n_samples = len(probs)
    np.random.seed(42)

    ece_values = []
    mce_values = []
    brier_values = []

    for _ in range(n_resamples):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # Compute calibration metrics
        ece_result = compute_smooth_ece(probs[indices], labels[indices], n_bins=15)
        brier = compute_brier_score(probs[indices], labels[indices])

        ece_values.append(ece_result["ece"])
        mce_values.append(ece_result["mce"])
        brier_values.append(brier)

    # Compute statistics
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    results = {}
    for metric_name, values in [("ece", ece_values), ("mce", mce_values), ("brier", brier_values)]:
        values = np.array(values)
        results[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "ci_lower": float(np.percentile(values, lower_percentile)),
            "ci_upper": float(np.percentile(values, upper_percentile)),
        }

    return results


def format_bootstrap_result(metric_name: str, result: Dict[str, float]) -> str:
    """Format bootstrap result for pretty printing.

    Args:
        metric_name: Name of the metric
        result: Bootstrap result dictionary

    Returns:
        Formatted string with mean and confidence interval

    Example:
        >>> result = {'mean': 0.8523, 'ci_lower': 0.7845, 'ci_upper': 0.9201, 'confidence_level': 0.95}
        >>> formatted = format_bootstrap_result('Accuracy', result)
        >>> assert 'Accuracy' in formatted
        >>> assert '0.8523' in formatted
    """
    mean = result["mean"]
    ci_lower = result["ci_lower"]
    ci_upper = result["ci_upper"]
    confidence = result.get("confidence_level", 0.95)

    return (
        f"{metric_name}: {mean:.4f} " f"[{confidence*100:.0f}% CI: {ci_lower:.4f} - {ci_upper:.4f}]"
    )
