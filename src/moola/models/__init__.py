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
# Simplified registry - direct model building from config

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
        **kwargs: Model-specific hyperparameters (input_size, hidden_size, etc.)

    Returns:
        Instantiated model implementing BaseModel interface

    Raises:
        ValueError: If model name not found in Stones registry

    Examples:
        >>> model = get_model("jade", input_size=11, hidden_size=96)
        >>> # All models use Jade_Compact architecture with different configurations
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
    
    # Default parameters for Jade_Compact
    default_kwargs = {
        "input_size": kwargs.get("input_size", 11),
        "hidden_size": 64,
        "num_layers": 1,
        "bidirectional": True,
        "proj_head": True,
        "head_width": 64,
        "pointer_encoding": "center_length"
    }
    
    # Override with user-provided kwargs
    default_kwargs.update(kwargs)
    
    return model_class(**default_kwargs)


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
]

# Alias for backward compatibility and testing
REGISTRY = _MODEL_REGISTRY
