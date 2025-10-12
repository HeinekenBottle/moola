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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogRegModel":
        """Train logistic regression model.

        Args:
            X: Feature matrix of shape [N, D]
            y: Target labels of shape [N]

        Returns:
            Self for method chaining
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix of shape [N, D]

        Returns:
            Predicted labels of shape [N]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape [N, D]

        Returns:
            Class probabilities of shape [N, C]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
