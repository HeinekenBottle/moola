"""Model registry for Moola ML pipeline.

Provides centralized model instantiation via get_model() function.
All models implement the BaseModel interface.
"""

from typing import Type

from .base import BaseModel
from .logreg import LogRegModel
from .rf import RFModel
from .xgb import XGBModel

# Model registry mapping names to classes
_MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    "logreg": LogRegModel,
    "rf": RFModel,
    "xgb": XGBModel,
}


def get_model(name: str, **kwargs) -> BaseModel:
    """Get model instance by name.

    Args:
        name: Model name (logreg, rf, xgb, stack)
        **kwargs: Model-specific hyperparameters (seed, max_iter, etc.)

    Returns:
        Instantiated model implementing BaseModel interface

    Raises:
        ValueError: If model name not found in registry

    Examples:
        >>> model = get_model("logreg", seed=1337, max_iter=1000)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")

    model_class = _MODEL_REGISTRY[name]
    return model_class(**kwargs)


def list_models() -> list[str]:
    """List all registered model names.

    Returns:
        List of available model names
    """
    return list(_MODEL_REGISTRY.keys())


__all__ = ["BaseModel", "LogRegModel", "get_model", "list_models"]
