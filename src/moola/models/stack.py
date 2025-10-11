"""Stacking meta-learner model implementation.

The stack model trains on concatenated out-of-fold predictions from base models
to learn optimal ensemble weights and calibration.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaseModel


class StackModel(BaseModel):
    """Stacking meta-learner using RandomForest.

    Trains on concatenated OOF predictions [N, 3*C] from base models
    (logreg, rf, xgb, rwkv_ts, cnn_transformer) to produce calibrated ensemble predictions.

    Uses RandomForest with balanced_subsample class weights for robustness
    to class imbalance and improved performance on small datasets.
    """

    def __init__(self, seed: int = 1337, n_estimators: int = 1000, **kwargs):
        """Initialize stacking meta-learner.

        Args:
            seed: Random seed for reproducibility
            n_estimators: Number of trees in the forest (default 1000)
            **kwargs: Additional sklearn RandomForestClassifier parameters
        """
        super().__init__(seed=seed)
        self.n_estimators = n_estimators
        self.kwargs = kwargs
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight="balanced_subsample",
            random_state=self.seed,
            max_features="sqrt",
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
