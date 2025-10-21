"""Model registry for Moola ML pipeline.

Provides centralized model instantiation via get_model() function.
All models implement the BaseModel interface.

Includes Stones collection with codename system:
- Jade: moola-lstm-m-v1.0 // codename: Jade
- Sapphire: moola-lstm-s-fr-v1.0 // codename: Sapphire
- Opal: moola-preenc-ad-m-v1.0 // codename: Opal
"""

from typing import Type

from .base import BaseModel
from .enhanced_simple_lstm import EnhancedSimpleLSTMModel
from .jade import JadeModel
from .logreg import LogRegModel
from .registry import (
    registry,
    get_model as get_model_by_id,
    get_jade,
    get_sapphire,
    get_opal,
    list_available_models,
    get_stones_collection,
)
from .rf import RFModel
from .simple_lstm import SimpleLSTMModel
from .stack import StackModel
from .xgb import XGBModel

# Model registry mapping names to classes (Stones-only)
_MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    # Deep learning models (production)
    "jade": JadeModel,  # PRODUCTION: Jade architecture with Stones non-negotiables
    "enhanced_simple_lstm": EnhancedSimpleLSTMModel,  # LEGACY: BiLSTM + attention with pretrained support
    "simple_lstm": SimpleLSTMModel,  # BASELINE: Lightweight for smoke tests
    # Classical ML models (for stacking)
    "logreg": LogRegModel,
    "rf": RFModel,
    "xgb": XGBModel,
    "stack": StackModel,
}


def get_model(name: str, **kwargs) -> BaseModel:
    """Get model instance by name.

    Args:
        name: Model name (jade, enhanced_simple_lstm, simple_lstm, cnn_transformer, rwkv_ts, logreg, rf, xgb, stack)
        **kwargs: Model-specific hyperparameters (seed, max_iter, device, etc.)
                 For enhanced_simple_lstm: load_pretrained_encoder (Path) loads encoder after init
                 For cnn_transformer: load_pretrained_encoder (Path) loads encoder after init
                 For jade: predict_pointers (bool) enables multi-task learning

    Returns:
        Instantiated model implementing BaseModel interface

    Raises:
        ValueError: If model name not found in registry

    Examples:
        >>> model = get_model("jade", seed=1337, device="cuda", predict_pointers=True)
        >>> model.fit(X_train, y_train, expansion_start=starts, expansion_end=ends)
        >>> predictions = model.predict(X_test)

        >>> # Load pre-trained encoder for enhanced_simple_lstm
        >>> model = get_model("enhanced_simple_lstm", device="cuda",
        ...                   load_pretrained_encoder="artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt")
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")

    model_class = _MODEL_REGISTRY[name]

    # Extract load_pretrained_encoder parameter (for models that support it)
    load_pretrained_encoder = kwargs.pop("load_pretrained_encoder", None)

    # Instantiate model
    model = model_class(**kwargs)

    # Load pre-trained encoder if specified (for models that support it)
    if load_pretrained_encoder and hasattr(model_class, "load_pretrained_encoder"):
        from pathlib import Path

        encoder_path = Path(load_pretrained_encoder)
        # Note: load_pretrained_encoder will be called during fit() after model is built
        # Store the path for later use in fit()
        if hasattr(model, "_pretrained_encoder_path"):
            model._pretrained_encoder_path = encoder_path

    return model


def list_models() -> list[str]:
    """List all registered model names.

    Returns:
        List of available model names
    """
    return list(_MODEL_REGISTRY.keys())


__all__ = [
    "BaseModel",
    "JadeModel",
    "EnhancedSimpleLSTMModel",
    "SimpleLSTMModel",
    "LogRegModel",
    "RFModel",
    "XGBModel",
    "StackModel",
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
