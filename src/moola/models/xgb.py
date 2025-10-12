"""XGBoost model implementation."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from .base import BaseModel


class XGBModel(BaseModel):
    """XGBoost gradient boosting classifier.

    High-performance gradient boosting implementation with regularization.
    Uses histogram-based tree construction for speed and memory efficiency.
    Automatically handles string labels via LabelEncoder.
    """

    def __init__(
        self,
        seed: int = 1337,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        eval_metric: str = "logloss",
        tree_method: str = "hist",
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize XGBoost model.

        Args:
            seed: Random seed for reproducibility
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (controls model complexity)
            learning_rate: Step size shrinkage (smaller = more conservative)
            subsample: Fraction of samples for training each tree
            colsample_bytree: Fraction of features for training each tree
            reg_lambda: L2 regularization term on weights
            eval_metric: Evaluation metric (logloss, error, auc)
            tree_method: Tree construction algorithm ('hist' for speed)
            device: Device parameter ('cpu' or 'cuda', XGBoost supports GPU)
            **kwargs: Additional xgboost parameters
        """
        super().__init__(seed=seed)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.eval_metric = eval_metric
        self.tree_method = tree_method
        self.kwargs = kwargs
        self.label_encoder = LabelEncoder()

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            eval_metric=self.eval_metric,
            tree_method=self.tree_method,
            random_state=self.seed,
            **self.kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBModel":
        """Train XGBoost model.

        Args:
            X: Feature matrix of shape [N, D]
            y: Target labels of shape [N] (can be strings or numeric)

        Returns:
            Self for method chaining
        """
        # Encode labels if they are strings
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix of shape [N, D]

        Returns:
            Predicted labels of shape [N] (in original label space)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)

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

    def save(self, path: Path) -> None:
        """Save model and label encoder to disk.

        Args:
            path: Path to save model file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save both model and label encoder together
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "label_encoder": self.label_encoder}, f)

    def load(self, path: Path) -> "XGBModel":
        """Load model and label encoder from disk.

        Args:
            path: Path to model file

        Returns:
            Self with loaded model and label encoder
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.label_encoder = data["label_encoder"]
        self.is_fitted = True
        return self
