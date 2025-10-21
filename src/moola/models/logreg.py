"""Logistic Regression model implementation."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BaseModel


class LogRegModel(BaseModel):
    """Logistic Regression classifier.

    Simple linear model for baseline classification.
    Uses scikit-learn's LogisticRegression with default L2 regularization.
    """

    def __init__(self, seed: int = 1337, max_iter: int = 1000, device: str = "cpu", **kwargs):
        """Initialize LogisticRegression model.

        Args:
            seed: Random seed for reproducibility
            max_iter: Maximum iterations for solver convergence
            device: Device parameter (ignored for sklearn models, kept for API consistency)
            **kwargs: Additional sklearn LogisticRegression parameters
        """
        super().__init__(seed=seed)
        self.max_iter = max_iter
        self.kwargs = kwargs
        self.model = LogisticRegression(
            random_state=self.seed, max_iter=self.max_iter, **self.kwargs
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        expansion_start: np.ndarray = None,
        expansion_end: np.ndarray = None,
    ) -> "LogRegModel":
        """Train logistic regression model.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, F]
            y: Target labels of shape [N]
            expansion_start: Optional expansion start indices of shape [N] (unused for classical models)
            expansion_end: Optional expansion end indices of shape [N] (unused for classical models)

        Returns:
            Self for method chaining
        """
        from ..features.price_action_features import engineer_classical_features

        # Transform raw OHLC to engineered features
        if X.shape[1] == 420:  # Flattened [105*4]
            X = X.reshape(-1, 105, 4)

        if X.ndim == 3:  # [N, T, F] format
            X_engineered = engineer_classical_features(
                X, expansion_start=expansion_start, expansion_end=expansion_end
            )
        else:
            X_engineered = X

        self.model.fit(X_engineered, y)
        self.is_fitted = True
        return self

    def predict(
        self, X: np.ndarray, expansion_start: np.ndarray = None, expansion_end: np.ndarray = None
    ) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, F]
            expansion_start: Optional expansion start indices of shape [N] (unused for classical models)
            expansion_end: Optional expansion end indices of shape [N] (unused for classical models)

        Returns:
            Predicted labels of shape [N]
        """
        from ..features.price_action_features import engineer_classical_features

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if X.shape[1] == 420:
            X = X.reshape(-1, 105, 4)

        if X.ndim == 3:
            X_engineered = engineer_classical_features(
                X, expansion_start=expansion_start, expansion_end=expansion_end
            )
        else:
            X_engineered = X

        return self.model.predict(X_engineered)

    def predict_proba(
        self, X: np.ndarray, expansion_start: np.ndarray = None, expansion_end: np.ndarray = None
    ) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, F]
            expansion_start: Optional expansion start indices of shape [N] (unused for classical models)
            expansion_end: Optional expansion end indices of shape [N] (unused for classical models)

        Returns:
            Class probabilities of shape [N, C]
        """
        from ..features.price_action_features import engineer_classical_features

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if X.shape[1] == 420:
            X = X.reshape(-1, 105, 4)

        if X.ndim == 3:
            X_engineered = engineer_classical_features(
                X, expansion_start=expansion_start, expansion_end=expansion_end
            )
        else:
            X_engineered = X

        return self.model.predict_proba(X_engineered)
