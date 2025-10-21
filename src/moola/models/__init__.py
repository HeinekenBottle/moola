"""Model registry for Moola ML pipeline.

Provides centralized model instantiation via get_model() function.
All models implement the BaseModel interface.

Stones collection with codename system - PAPER-STRICT COMPLIANCE:
- Jade: moola-lstm-m-v1.0 // codename: Jade
- Sapphire: moola-lstm-sf-v1.0 // codename: Sapphire  
- Opal: moola-preenc-ma-v1.0 // codename: Opal

ONLY THESE MODELS ARE ALLOWED FOR PAPER EXPERIMENTS.
"""

from typing import Type
import sys
import os

from .base import BaseModel
from .jade import JadeModel
from .registry import (
    registry,
    get_model as get_model_by_id,
    get_jade,
    get_sapphire,
    get_opal,
    list_available_models,
    get_stones_collection,
)

# PAPER-STRICT: Only Stones models allowed
_MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    "jade": JadeModel,      # PRODUCTION: Jade architecture with Stones non-negotiables
    "sapphire": JadeModel,  # TRANSFER: Frozen encoder
    "opal": JadeModel,      # ADAPTIVE: Fine-tuning
}


def get_model(name: str, **kwargs) -> BaseModel:
    """Get model instance by name (PAPER-STRICT: Stones models only).

    Args:
        name: Model name (jade, sapphire, opal ONLY)
        **kwargs: Model-specific hyperparameters (seed, device, predict_pointers, etc.)

    Returns:
        Instantiated model implementing BaseModel interface

    Raises:
        ValueError: If model name not found in Stones registry

    Examples:
        >>> model = get_model("jade", seed=1337, device="cuda", predict_pointers=True)
        >>> model.fit(X_train, y_train, expansion_start=starts, expansion_end=ends)
        >>> predictions = model.predict(X_test)

        >>> # Sapphire with frozen encoder
        >>> model = get_model("sapphire", device="cuda", predict_pointers=True)

        >>> # Opal with adaptive fine-tuning  
        >>> model = get_model("opal", device="cuda", predict_pointers=True)
    """
    # PAPER-STRICT: Only allow Stones models
    allowed_models = {"jade", "sapphire", "opal"}
    if name not in allowed_models:
        raise ValueError(
            f"PAPER-STRICT VIOLATION: Model '{name}' not allowed. "
            f"Only Stones models allowed: {sorted(allowed_models)}"
        )

    if name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")

    model_class = _MODEL_REGISTRY[name]

    # PAPER-STRICT: Use registry configuration for Stones models
    if name == "jade":
        return get_jade(**kwargs)
    elif name == "sapphire":
        return get_sapphire(**kwargs)
    elif name == "opal":
        return get_opal(**kwargs)
    else:
        # Fallback (should not reach due to validation above)
        return model_class(**kwargs)


def list_models() -> list[str]:
    """List all registered model names.

    Returns:
        List of available model names
    """
    return list(_MODEL_REGISTRY.keys())


__all__ = [
    "BaseModel",
    "JadeModel",
    "get_model",
    "list_models",
    "_MODEL_REGISTRY",  # For testing and introspection
    # Registry functions
    "get_jade",
    "get_sapphire", 
    "get_opal",
    "list_available_models",
    "get_stones_collection",
]

# Alias for backward compatibility and testing
REGISTRY = _MODEL_REGISTRY
