"""Stones Ensemble - Combines multiple trained models.

Simple ensemble implementation that can work with any compatible models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


class StonesEnsemble:
    """Simple ensemble of compatible models.

    Combines predictions from multiple models using weighted averaging.
    """

    def __init__(self):
        self.models: Dict[str, any] = {}
        self.weights: Dict[str, float] = {}
        self.n_classes: int = 2
        self.is_fitted: bool = False

    def add_model(self, name: str, model: any, weight: float = 1.0) -> "StonesEnsemble":
        """Add a model to the ensemble.

        Args:
            name: Model name/identifier
            model: Trained model with predict() and predict_proba() methods
            weight: Model weight for ensemble averaging

        Returns:
            Self for method chaining
        """
        self.models[name] = model
        self.weights[name] = weight
        
        # Get n_classes from first model
        if hasattr(model, 'n_classes'):
            self.n_classes = model.n_classes
            
        logger.info(f"Added model '{name}' with weight {weight}")
        return self

    def set_weights(self, weights: Dict[str, float]) -> "StonesEnsemble":
        """Set ensemble weights manually.

        Args:
            weights: Dictionary mapping model names to weights

        Returns:
            Self for method chaining
        """
        # Validate weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")

        # Normalize weights
        self.weights = {k: v / total_weight for k, v in weights.items()}
        logger.info(f"Set ensemble weights: {self.weights}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using ensemble.

        Args:
            X: Feature matrix [N, T, D]

        Returns:
            Class probabilities [N, n_classes]
        """
        if not self.models:
            raise ValueError("No models in ensemble. Add models first.")

        # Collect predictions from all models
        all_probs = []
        model_weights = []

        for name, model in self.models.items():
            try:
                probs = model.predict_proba(X)
                all_probs.append(probs)
                weight = self.weights.get(name, 1.0)
                model_weights.append(weight)
                logger.debug(f"{name}: weight={weight:.3f}")
            except Exception as e:
                logger.warning(f"Failed to get predictions from {name}: {e}")
                continue

        if not all_probs:
            raise ValueError("No model predictions available")

        # Weighted average of probabilities
        model_weights = np.array(model_weights)
        model_weights = model_weights / model_weights.sum()  # Renormalize

        ensemble_probs = np.zeros_like(all_probs[0])
        for probs, weight in zip(all_probs, model_weights):
            ensemble_probs += weight * probs

        return ensemble_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using ensemble.

        Args:
            X: Feature matrix [N, T, D]

        Returns:
            Predicted class labels [N]
        """
        probs = self.predict_proba(X)
        predictions = np.argmax(probs, axis=1)

        # Convert back to original labels if available
        if self.models:
            first_model = next(iter(self.models.values()))
            if hasattr(first_model, 'idx_to_label'):
                predictions = np.array([first_model.idx_to_label[idx] for idx in predictions])

        return predictions

    def evaluate_ensemble(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate ensemble performance.

        Args:
            X: Feature matrix [N, T, D]
            y: True labels [N]

        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        # Classification metrics
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average="binary", zero_division="0"
        )

        metrics = {
            "ensemble_accuracy": accuracy,
            "ensemble_precision": precision,
            "ensemble_recall": recall,
            "ensemble_f1": f1,
            "n_models": len(self.models),
        }

        return metrics

    def save_ensemble_info(self, path: Path) -> None:
        """Save ensemble configuration.

        Args:
            path: Path to save ensemble metadata
        """
        import json

        ensemble_info = {
            "models": list(self.models.keys()),
            "weights": self.weights,
            "n_classes": self.n_classes,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(ensemble_info, f, indent=2)

        logger.info(f"Saved ensemble configuration to {path}")


def load_jade_ensemble() -> StonesEnsemble:
    """Load ensemble with available Jade models.

    Returns:
        Configured StonesEnsemble instance with available models
    """
    from .jade import JadeModel

    ensemble = StonesEnsemble()

    # Try to load available models
    model_dir = Path("data/artifacts/models")
    
    # Try Jade
    jade_path = model_dir / "jade" / "model.pkl"
    if jade_path.exists():
        try:
            jade_model = JadeModel()
            jade_model.load(jade_path)
            ensemble.add_model("jade", jade_model, weight=1.0)
            logger.info("Loaded Jade model")
        except Exception as e:
            logger.warning(f"Failed to load Jade model: {e}")

    if not ensemble.models:
        raise ValueError("No models could be loaded")

    return ensemble