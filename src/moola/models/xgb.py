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
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 5,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        eval_metric: str = "logloss",
        tree_method: str = "hist",
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize XGBoost model.

        Args:
            seed: Random seed for reproducibility
            n_estimators: Number of boosting rounds (reduced to 200 to prevent overfitting)
            max_depth: Maximum tree depth (reduced to 4 for better generalization)
            learning_rate: Step size shrinkage (increased to 0.1 for faster convergence)
            subsample: Fraction of samples for training each tree
            colsample_bytree: Fraction of features for training each tree
            min_child_weight: Minimum sum of instance weight needed in a child (5 for small dataset)
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            eval_metric: Evaluation metric (logloss, error, auc)
            tree_method: Tree construction algorithm ('hist' for CPU, 'gpu_hist' for GPU)
            device: Device parameter ('cpu' or 'cuda')
            **kwargs: Additional xgboost parameters
        """
        super().__init__(seed=seed)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.eval_metric = eval_metric
        self.device = device
        self.kwargs = kwargs
        self.label_encoder = LabelEncoder()

        # Auto-configure GPU settings
        if device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    self.tree_method = "gpu_hist"
                    # XGBoost uses 'cuda' for device parameter
                    self.device_param = "cuda"
                    print(f"[GPU] XGBoost using GPU acceleration: {torch.cuda.get_device_name(0)}")
                else:
                    print("[WARNING] CUDA requested but not available. Falling back to CPU.")
                    self.tree_method = tree_method
                    self.device_param = "cpu"
            except ImportError:
                print("[WARNING] PyTorch not found. Cannot detect CUDA. Using CPU.")
                self.tree_method = tree_method
                self.device_param = "cpu"
        else:
            self.tree_method = tree_method
            self.device_param = "cpu"

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            eval_metric=self.eval_metric,
            tree_method=self.tree_method,
            device=self.device_param,
            random_state=self.seed,
            **self.kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, expansion_start: np.ndarray = None, expansion_end: np.ndarray = None) -> "XGBModel":
        """Train XGBoost model with HopSketch 15-feature extraction.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, F]
            y: Target labels of shape [N] (can be strings or numeric)
            expansion_start: Optional expansion start indices of shape [N] (UNUSED - kept for API compatibility)
            expansion_end: Optional expansion end indices of shape [N] (UNUSED - kept for API compatibility)

        Returns:
            Self for method chaining
        """
        from ..features.price_action_features import extract_hopsketch_features

        # Transform raw OHLC to HopSketch features
        if X.shape[1] == 420:  # Flattened [105*4]
            X = X.reshape(-1, 105, 4)

        if X.ndim == 3:  # [N, T, F] format
            # Extract 15 features per bar from FULL 105-bar window
            X_hopsketch = extract_hopsketch_features(X)  # [N, 1575]

            # Reshape to [N, 105, 15] for temporal aggregation
            X_hopsketch = X_hopsketch.reshape(-1, 105, 15)

            # Aggregate across time dimension: [N, 60] features
            # 4 statistics (mean, std, min, max) × 15 features = 60 total
            X_engineered = np.column_stack([
                X_hopsketch.mean(axis=1),   # [N, 15] mean per feature
                X_hopsketch.std(axis=1),    # [N, 15] std per feature
                X_hopsketch.min(axis=1),    # [N, 15] min per feature
                X_hopsketch.max(axis=1),    # [N, 15] max per feature
            ])  # [N, 60] total
        else:
            X_engineered = X

        # Ensure X_engineered is numeric
        X_engineered = np.array(X_engineered, dtype=np.float64)
        print(f"[HOPSKETCH] Aggregated features shape: {X_engineered.shape}, dtype: {X_engineered.dtype}")

        # Encode labels if they are strings
        y_encoded = self.label_encoder.fit_transform(y)

        # SMOTE removed per Phase 1c - use controlled augmentation with KS p-value validation instead
        # Use sample weighting for class imbalance
        unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
        n_samples = len(y_encoded)
        n_classes = len(unique_classes)

        # Calculate balanced class weights: n_samples / (n_classes * class_count)
        class_weights = n_samples / (n_classes * class_counts)

        # Map weights to each sample
        sample_weights = np.array([class_weights[cls] for cls in y_encoded])

        print(f"[CLASS BALANCE] Class weights: {dict(zip(unique_classes, class_weights))}")
        print(f"[CLASS BALANCE] Class distribution: {dict(zip(unique_classes, class_counts))}")

        # Fit with sample weights (preferred over SMOTE for XGBoost)
        self.model.fit(X_engineered, y_encoded, sample_weight=sample_weights)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, expansion_start: np.ndarray = None, expansion_end: np.ndarray = None) -> np.ndarray:
        """Predict class labels with HopSketch features.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, F]
            expansion_start: Optional expansion start indices of shape [N] (UNUSED - kept for API compatibility)
            expansion_end: Optional expansion end indices of shape [N] (UNUSED - kept for API compatibility)

        Returns:
            Predicted labels of shape [N] (in original label space)
        """
        from ..features.price_action_features import extract_hopsketch_features

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if X.shape[1] == 420:
            X = X.reshape(-1, 105, 4)

        if X.ndim == 3:
            # Extract and aggregate HopSketch features
            X_hopsketch = extract_hopsketch_features(X)  # [N, 1575]
            X_hopsketch = X_hopsketch.reshape(-1, 105, 15)

            # Aggregate to [N, 60]
            X_engineered = np.column_stack([
                X_hopsketch.mean(axis=1),
                X_hopsketch.std(axis=1),
                X_hopsketch.min(axis=1),
                X_hopsketch.max(axis=1),
            ])
        else:
            X_engineered = X

        y_pred_encoded = self.model.predict(X_engineered)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X: np.ndarray, expansion_start: np.ndarray = None, expansion_end: np.ndarray = None) -> np.ndarray:
        """Predict class probabilities with HopSketch features.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, F]
            expansion_start: Optional expansion start indices of shape [N] (UNUSED - kept for API compatibility)
            expansion_end: Optional expansion end indices of shape [N] (UNUSED - kept for API compatibility)

        Returns:
            Class probabilities of shape [N, C]
        """
        from ..features.price_action_features import extract_hopsketch_features

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if X.shape[1] == 420:
            X = X.reshape(-1, 105, 4)

        if X.ndim == 3:
            # Extract and aggregate HopSketch features
            X_hopsketch = extract_hopsketch_features(X)  # [N, 1575]
            X_hopsketch = X_hopsketch.reshape(-1, 105, 15)

            # Aggregate to [N, 60]
            X_engineered = np.column_stack([
                X_hopsketch.mean(axis=1),
                X_hopsketch.std(axis=1),
                X_hopsketch.min(axis=1),
                X_hopsketch.max(axis=1),
            ])
        else:
            X_engineered = X

        return self.model.predict_proba(X_engineered)

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
