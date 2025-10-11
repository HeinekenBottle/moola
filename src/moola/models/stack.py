"""Stacking meta-learner model implementation.

The stack model trains on concatenated out-of-fold predictions from base models
to learn optimal ensemble weights and calibration.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BaseModel


class StackModel(BaseModel):
    """Stacking meta-learner using Logistic Regression.

    Trains on concatenated OOF predictions [N, 3*C] from base models
    (logreg, rf, xgb) to produce calibrated ensemble predictions.

    Uses LogisticRegression with balanced class weights for robustness
    to class imbalance and improved calibration.
    """

    def __init__(self, seed: int = 1337, C: float = 1.0, **kwargs):
        """Initialize stacking meta-learner.

        Args:
            seed: Random seed for reproducibility
            C: Inverse of regularization strength (default 1.0)
            **kwargs: Additional sklearn LogisticRegression parameters
        """
        super().__init__(seed=seed)
        self.C = C
        self.kwargs = kwargs
        self.model = LogisticRegression(
            C=self.C,
            class_weight="balanced",
            random_state=self.seed,
            max_iter=1000,
            **self.kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackModel":
        """Train meta-learner on concatenated OOF predictions.

        Args:
            X: Concatenated OOF predictions of shape [N, 3*C] where
               - N is number of samples
               - 3 is number of base models
               - C is number of classes
            y: Target labels of shape [N]

        Returns:
            Self for method chaining
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using meta-learner.

        Args:
            X: Concatenated base model predictions of shape [N, 3*C]

        Returns:
            Predicted labels of shape [N]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated class probabilities.

        Args:
            X: Concatenated base model predictions of shape [N, 3*C]

        Returns:
            Calibrated class probabilities of shape [N, C]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
