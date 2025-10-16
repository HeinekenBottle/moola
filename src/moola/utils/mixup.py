"""Mixup augmentation utility for shallow ML models.

Provides mixup data augmentation for non-deep learning models (LogReg, RF, XGBoost).
Mixup creates synthetic training examples by interpolating between pairs of samples.

Reference:
    - mixup: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
"""

import numpy as np
from typing import Tuple


def mixup_data(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply mixup augmentation to training data.

    Mixup creates virtual training examples by interpolating between pairs
    of samples. For each sample pair (x_i, y_i) and (x_j, y_j), creates:
        x_mixed = lambda * x_i + (1 - lambda) * x_j

    For classification, returns both label sets for weighted loss computation.

    Args:
        X: Feature matrix of shape [N, D]
           Can be flattened time series [N, T*F] or feature vectors
        y: Target labels of shape [N]
           Integer class labels for classification
        alpha: Beta distribution parameter (higher = more mixing)
               Recommended: 0.4 for small datasets, 0.2-0.8 generally

    Returns:
        Tuple of (X_mixed, y_a, y_b, lam) where:
            - X_mixed: Mixed features [N, D]
            - y_a: Original labels [N]
            - y_b: Shuffled labels [N]
            - lam: Mixing coefficients [N] (one per sample)

    Example:
        >>> from moola.utils.mixup import mixup_data
        >>> X = np.random.randn(100, 420)  # 100 samples, 420 features
        >>> y = np.array([0, 1] * 50)       # Binary labels
        >>> X_mixed, y_a, y_b, lam = mixup_data(X, y, alpha=0.4)
        >>>
        >>> # For sklearn models, use both label sets in custom loss
        >>> # Or train on mixed data with soft labels (if supported)
        >>> # For XGBoost, can pass both y_a and y_b with weighted objectives
    """
    if alpha > 0:
        # Sample mixing coefficients from Beta distribution
        lam = np.random.beta(alpha, alpha, size=X.shape[0])
    else:
        # No mixing (lambda = 1.0)
        lam = np.ones(X.shape[0])

    batch_size = X.shape[0]
    index = np.random.permutation(batch_size)

    # Broadcast lambda for feature-wise mixing
    # Shape: [N, 1] to multiply with [N, D]
    lam_expanded = lam[:, np.newaxis]

    # Mix inputs: X_mixed = lambda * X_i + (1 - lambda) * X_j
    X_mixed = lam_expanded * X + (1 - lam_expanded) * X[index]

    # Return both label sets for loss computation
    y_a = y
    y_b = y[index]

    return X_mixed, y_a, y_b, lam


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    n_augmented: int,
    alpha: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate augmented training set using mixup.

    Creates additional synthetic samples to expand the training set.
    Useful for small datasets where each additional sample matters.

    Args:
        X: Original feature matrix [N, D]
        y: Original labels [N]
        n_augmented: Number of synthetic samples to generate
        alpha: Beta distribution parameter for mixup (default: 0.4)

    Returns:
        Tuple of (X_augmented, y_augmented) where:
            - X_augmented: Combined original + synthetic features [N + n_augmented, D]
            - y_augmented: Combined labels [N + n_augmented]
                          Synthetic labels are hard labels (argmax of mixed probabilities)

    Example:
        >>> # Original dataset: 100 samples
        >>> X_train = np.random.randn(100, 420)
        >>> y_train = np.array([0, 1] * 50)
        >>>
        >>> # Augment with 50 synthetic samples (50% increase)
        >>> X_aug, y_aug = augment_dataset(X_train, y_train, n_augmented=50, alpha=0.4)
        >>> print(X_aug.shape)  # (150, 420)
    """
    if n_augmented <= 0:
        return X, y

    # Generate synthetic samples
    synthetic_samples = []
    synthetic_labels = []

    # Sample pairs randomly with replacement
    n_samples = X.shape[0]
    for _ in range(n_augmented):
        # Sample two random indices
        idx_a = np.random.randint(0, n_samples)
        idx_b = np.random.randint(0, n_samples)

        # Sample mixing coefficient
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        # Mix samples
        x_mixed = lam * X[idx_a] + (1 - lam) * X[idx_b]

        # For classification, use the label with higher weight
        # This creates "hard" labels for standard classifiers
        y_mixed = y[idx_a] if lam > 0.5 else y[idx_b]

        synthetic_samples.append(x_mixed)
        synthetic_labels.append(y_mixed)

    # Combine original and synthetic data
    X_augmented = np.vstack([X, np.array(synthetic_samples)])
    y_augmented = np.concatenate([y, np.array(synthetic_labels)])

    return X_augmented, y_augmented


def mixup_criterion_sklearn(
    y_true_a: np.ndarray,
    y_true_b: np.ndarray,
    y_pred_proba: np.ndarray,
    lam: np.ndarray,
) -> float:
    """Compute mixup loss for sklearn models with predict_proba.

    Computes weighted cross-entropy loss for mixup samples:
        L = lambda * CE(y_pred, y_a) + (1 - lambda) * CE(y_pred, y_b)

    Args:
        y_true_a: First set of true labels [N]
        y_true_b: Second set of true labels [N]
        y_pred_proba: Predicted probabilities [N, n_classes]
        lam: Mixing coefficients [N]

    Returns:
        Mean mixup loss (scalar)

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from moola.utils.mixup import mixup_data, mixup_criterion_sklearn
        >>>
        >>> # Train with mixup
        >>> X_mixed, y_a, y_b, lam = mixup_data(X_train, y_train, alpha=0.4)
        >>> model = LogisticRegression()
        >>> model.fit(X_mixed, y_a)  # Train on mixed data with primary labels
        >>>
        >>> # Evaluate mixup loss
        >>> y_pred_proba = model.predict_proba(X_val)
        >>> loss = mixup_criterion_sklearn(y_val, y_val, y_pred_proba, np.ones(len(y_val)))
    """
    # Convert labels to one-hot
    n_samples = y_pred_proba.shape[0]
    n_classes = y_pred_proba.shape[1]

    y_a_onehot = np.zeros((n_samples, n_classes))
    y_b_onehot = np.zeros((n_samples, n_classes))

    y_a_onehot[np.arange(n_samples), y_true_a] = 1
    y_b_onehot[np.arange(n_samples), y_true_b] = 1

    # Compute mixed targets
    lam_expanded = lam[:, np.newaxis]
    y_mixed = lam_expanded * y_a_onehot + (1 - lam_expanded) * y_b_onehot

    # Compute cross-entropy loss
    # CE = -sum(y_true * log(y_pred))
    epsilon = 1e-15  # For numerical stability
    y_pred_proba_clipped = np.clip(y_pred_proba, epsilon, 1 - epsilon)

    ce_loss = -np.sum(y_mixed * np.log(y_pred_proba_clipped), axis=1)

    return np.mean(ce_loss)
