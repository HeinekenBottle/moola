"""Random Forest model implementation."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaseModel


class RFModel(BaseModel):
    """Random Forest classifier.

    Ensemble of decision trees with bagging for robust predictions.
    Uses scikit-learn's RandomForestClassifier with balanced class weights.
    """

    def __init__(
        self,
        seed: int = 1337,
        n_estimators: int = 1000,
        max_depth: int = None,
        min_samples_split: int = 2,
        class_weight: str = "balanced_subsample",
        oob_score: bool = True,
        n_jobs: int = -1,
        device: str = "cpu",  # sklearn models ignore device parameter
        **kwargs,
    ):
        """Initialize RandomForest model.

        Args:
            seed: Random seed for reproducibility
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples required to split node
            class_weight: Class weighting strategy ('balanced_subsample' handles imbalance)
            oob_score: Whether to use out-of-bag samples for generalization estimate
            n_jobs: Number of parallel jobs (-1 = use all cores)
            device: Device parameter (ignored for sklearn models, kept for API consistency)
            **kwargs: Additional sklearn RandomForestClassifier parameters
        """
        super().__init__(seed=seed)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.kwargs = kwargs

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            class_weight=self.class_weight,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.seed,
            **self.kwargs,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        expansion_start: np.ndarray = None,
        expansion_end: np.ndarray = None,
    ) -> "RFModel":
        """Train random forest model.

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
