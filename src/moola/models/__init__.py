"""Model registry for Moola ML pipeline.

Provides centralized model instantiation via get_model() function.
All models implement the BaseModel interface.
"""

from typing import Type

from .base import BaseModel
from .cnn_transformer import CnnTransformerModel
from .enhanced_simple_lstm import EnhancedSimpleLSTMModel
from .logreg import LogRegModel
from .rf import RFModel
from .relative_transform_lstm import RelativeTransformLSTMModel
from .rwkv_ts import RWKVTSModel
from .simple_lstm import SimpleLSTMModel
from .stack import StackModel
from .xgb import XGBModel

# Model registry mapping names to classes
_MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    # Deep learning models (production)
    "enhanced_simple_lstm": EnhancedSimpleLSTMModel,  # PRIMARY: BiLSTM + attention with pretrained support
    "simple_lstm": SimpleLSTMModel,  # BASELINE: Lightweight for smoke tests
    "relative_transform_lstm": RelativeTransformLSTMModel,  # RELATIVE: 11-dim RelativeTransform features
    "cnn_transformer": CnnTransformerModel,  # EXPERIMENTAL: Multi-task CNN-Transformer
    "rwkv_ts": RWKVTSModel,  # EXPERIMENTAL: RWKV for time series
    # Classical ML models (for stacking)
    "logreg": LogRegModel,
    "rf": RFModel,
    "xgb": XGBModel,
    "stack": StackModel,
}


def get_model(name: str, **kwargs) -> BaseModel:
    """Get model instance by name.

    Args:
        name: Model name (enhanced_simple_lstm, simple_lstm, cnn_transformer, rwkv_ts, logreg, rf, xgb, stack)
        **kwargs: Model-specific hyperparameters (seed, max_iter, device, etc.)
                 For enhanced_simple_lstm: load_pretrained_encoder (Path) loads encoder after init
                 For cnn_transformer: load_pretrained_encoder (Path) loads encoder after init

    Returns:
        Instantiated model implementing BaseModel interface

    Raises:
        ValueError: If model name not found in registry

    Examples:
        >>> model = get_model("enhanced_simple_lstm", seed=1337, device="cuda")
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)

        >>> # Load pre-trained encoder for enhanced_simple_lstm
        >>> model = get_model("enhanced_simple_lstm", device="cuda",
        ...                   load_pretrained_encoder="artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt")
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")

    model_class = _MODEL_REGISTRY[name]

    # Extract load_pretrained_encoder parameter (only for cnn_transformer)
    load_pretrained_encoder = kwargs.pop("load_pretrained_encoder", None)

    # Instantiate model
    model = model_class(**kwargs)

    # Load pre-trained encoder if specified (only for cnn_transformer)
    if load_pretrained_encoder and name == "cnn_transformer":
        from pathlib import Path
        encoder_path = Path(load_pretrained_encoder)
        # Note: load_pretrained_encoder will be called during fit() after model is built
        # Store the path for later use in fit()
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
    "EnhancedSimpleLSTMModel",
    "SimpleLSTMModel",
    "CnnTransformerModel",
    "RWKVTSModel",
    "LogRegModel",
    "RFModel",
    "XGBModel",
    "StackModel",
    "get_model",
    "list_models",
    "_MODEL_REGISTRY",  # For testing and introspection
]

# Alias for backward compatibility and testing
REGISTRY = _MODEL_REGISTRY
