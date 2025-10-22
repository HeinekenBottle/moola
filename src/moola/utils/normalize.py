"""Price normalization utilities for cross-period generalization.

Implements price relevance scaling to improve model generalization across
different market regimes and time periods.

Key Insight:
    Raw OHLC prices vary significantly across time periods (e.g., NQ futures
    at 12,000 in 2020 vs 18,000 in 2024). Normalizing to [0,1] within each
    window makes patterns more comparable across periods.

Usage:
    >>> from moola.utils.normalize import price_relevance
    >>> import numpy as np
    >>> 
    >>> # Load raw OHLC data
    >>> X = np.random.randn(100, 105, 11)  # [N, T, D]
    >>> 
    >>> # Apply price relevance scaling
    >>> X_norm = price_relevance(X)
    >>> 
    >>> # First 4 channels (OHLC) are now in [0, 1] per window
    >>> assert X_norm[..., :4].min() >= 0
    >>> assert X_norm[..., :4].max() <= 1
"""

import numpy as np
from loguru import logger


def price_relevance(X: np.ndarray, *, ohlc_channels: int = 4) -> np.ndarray:
    """Apply price relevance scaling to OHLC columns.
    
    Normalizes OHLC (first N channels) to [0, 1] within each window
    to improve cross-period generalization.
    
    Formula:
        X_norm = (X - X_min) / (X_max - X_min + eps)
    
    Where min/max are computed per window (axis=1) for OHLC channels only.
    
    Args:
        X: Input array of shape (N, T, D) where D >= ohlc_channels
           First `ohlc_channels` are OHLC
        ohlc_channels: Number of OHLC channels to normalize (default: 4)
    
    Returns:
        Normalized array with OHLC scaled to [0, 1] per window
    
    Raises:
        ValueError: If X has fewer than ohlc_channels features
    
    Examples:
        >>> X = np.random.randn(100, 105, 11)
        >>> X_norm = price_relevance(X)
        >>> assert X_norm.shape == X.shape
        >>> assert X_norm[..., :4].min() >= 0
        >>> assert X_norm[..., :4].max() <= 1
    """
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array (N, T, D), got shape {X.shape}")
    
    if X.shape[-1] < ohlc_channels:
        raise ValueError(
            f"Expected at least {ohlc_channels} features for OHLC, got {X.shape[-1]}"
        )
    
    # Extract OHLC channels
    ohlc = X[..., :ohlc_channels]
    
    # Compute min/max per window (axis=1)
    ohlc_min = ohlc.min(axis=1, keepdims=True)
    ohlc_max = ohlc.max(axis=1, keepdims=True)
    
    # Normalize to [0, 1]
    rng = ohlc_max - ohlc_min + 1e-8  # Add epsilon to avoid division by zero
    ohlc_norm = (ohlc - ohlc_min) / rng
    
    # Create normalized array (copy to avoid modifying original)
    X_norm = X.copy()
    X_norm[..., :ohlc_channels] = ohlc_norm
    
    logger.debug(
        f"Applied price relevance scaling to {ohlc_channels} OHLC channels | "
        f"Range: [{ohlc_norm.min():.4f}, {ohlc_norm.max():.4f}]"
    )
    
    return X_norm


def z_score_normalize(X: np.ndarray, *, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    """Apply z-score normalization (mean=0, std=1).
    
    Args:
        X: Input array of shape (N, T, D)
        axis: Axis to compute statistics over (default: 1 for per-window)
        eps: Epsilon to avoid division by zero (default: 1e-8)
    
    Returns:
        Z-score normalized array
    
    Examples:
        >>> X = np.random.randn(100, 105, 11)
        >>> X_norm = z_score_normalize(X)
        >>> assert np.abs(X_norm.mean(axis=1)).max() < 0.1  # Mean ≈ 0
        >>> assert np.abs(X_norm.std(axis=1) - 1).max() < 0.1  # Std ≈ 1
    """
    mu = X.mean(axis=axis, keepdims=True)
    sd = X.std(axis=axis, keepdims=True) + eps
    return (X - mu) / sd


def min_max_normalize(
    X: np.ndarray, *, axis: int = 1, feature_range: tuple = (0, 1), eps: float = 1e-8
) -> np.ndarray:
    """Apply min-max normalization to specified range.
    
    Args:
        X: Input array of shape (N, T, D)
        axis: Axis to compute statistics over (default: 1 for per-window)
        feature_range: Target range (default: (0, 1))
        eps: Epsilon to avoid division by zero (default: 1e-8)
    
    Returns:
        Min-max normalized array
    
    Examples:
        >>> X = np.random.randn(100, 105, 11)
        >>> X_norm = min_max_normalize(X, feature_range=(-1, 1))
        >>> assert X_norm.min() >= -1
        >>> assert X_norm.max() <= 1
    """
    x_min = X.min(axis=axis, keepdims=True)
    x_max = X.max(axis=axis, keepdims=True)
    
    # Normalize to [0, 1]
    X_std = (X - x_min) / (x_max - x_min + eps)
    
    # Scale to feature_range
    min_val, max_val = feature_range
    X_scaled = X_std * (max_val - min_val) + min_val
    
    return X_scaled


def robust_normalize(X: np.ndarray, *, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    """Apply robust normalization using median and IQR.
    
    More robust to outliers than z-score normalization.
    
    Args:
        X: Input array of shape (N, T, D)
        axis: Axis to compute statistics over (default: 1 for per-window)
        eps: Epsilon to avoid division by zero (default: 1e-8)
    
    Returns:
        Robust normalized array
    
    Examples:
        >>> X = np.random.randn(100, 105, 11)
        >>> X_norm = robust_normalize(X)
        >>> # Median ≈ 0, IQR ≈ 1
    """
    median = np.median(X, axis=axis, keepdims=True)
    q75 = np.percentile(X, 75, axis=axis, keepdims=True)
    q25 = np.percentile(X, 25, axis=axis, keepdims=True)
    iqr = q75 - q25 + eps
    
    return (X - median) / iqr


def normalize_batch(
    X: np.ndarray,
    *,
    method: str = "price_relevance",
    ohlc_channels: int = 4,
    **kwargs,
) -> np.ndarray:
    """Normalize batch using specified method.
    
    Args:
        X: Input array of shape (N, T, D)
        method: Normalization method (default: 'price_relevance')
            - 'price_relevance': OHLC to [0,1] per window
            - 'z_score': Mean=0, std=1
            - 'min_max': Min-max to [0,1] or custom range
            - 'robust': Median=0, IQR=1
            - 'none': No normalization
        ohlc_channels: Number of OHLC channels (default: 4)
        **kwargs: Additional arguments for normalization method
    
    Returns:
        Normalized array
    
    Raises:
        ValueError: If method is unknown
    
    Examples:
        >>> X = np.random.randn(100, 105, 11)
        >>> X_norm = normalize_batch(X, method='price_relevance')
        >>> X_norm = normalize_batch(X, method='z_score')
        >>> X_norm = normalize_batch(X, method='min_max', feature_range=(-1, 1))
    """
    if method == "price_relevance":
        return price_relevance(X, ohlc_channels=ohlc_channels)
    elif method == "z_score":
        return z_score_normalize(X, **kwargs)
    elif method == "min_max":
        return min_max_normalize(X, **kwargs)
    elif method == "robust":
        return robust_normalize(X, **kwargs)
    elif method == "none":
        return X
    else:
        raise ValueError(
            f"Unknown normalization method: {method}. "
            f"Choose from: price_relevance, z_score, min_max, robust, none"
        )

