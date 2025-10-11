"""Base model interface for all ML models in Moola."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Abstract base class for all ML models.

    All models must implement:
    - fit(X, y): Train the model
    - predict(X): Return class predictions
    - predict_proba(X): Return class probabilities [N, C]
    - save(path): Serialize model to disk
    - load(path): Deserialize model from disk
    """

    def __init__(self, seed: int = 1337, **kwargs):
        """Initialize base model.

        Args:
            seed: Random seed for reproducibility
            **kwargs: Additional model-specific hyperparameters
        """
        self.seed = seed
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Train the model on data.

        Args:
            X: Feature matrix of shape [N, D]
            y: Target labels of shape [N]

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix of shape [N, D]

        Returns:
            Predicted labels of shape [N]
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape [N, D]

        Returns:
            Class probabilities of shape [N, C] where C is number of classes
        """
        pass

    def save(self, path: Path) -> None:
        """Save model to disk using pickle.

        Args:
            path: Path to save model file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: Path) -> "BaseModel":
        """Load model from disk.

        Args:
            path: Path to model file

        Returns:
            Self with loaded model
        """
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        return self

    def __repr__(self) -> str:
        """String representation of model."""
        return f"{self.__class__.__name__}(seed={self.seed}, fitted={self.is_fitted})"
