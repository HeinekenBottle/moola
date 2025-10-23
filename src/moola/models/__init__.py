"""Model registry for Moola ML pipeline.

Provides centralized model instantiation via get_model() function.
Clean architecture with separation of concerns:
- JadeCompact: Pure PyTorch nn.Module (architecture only)
- ModuleAdapter: Thin wrapper for fit/predict/save (backward compatibility)

Stones collection with codename system - PAPER-STRICT COMPLIANCE:
- JadeCompact: moola-lstm-s-v1.1 // codename: Jade-Compact

ONLY THIS MODEL IS ALLOWED FOR PAPER EXPERIMENTS.
"""

from typing import Optional

from .adapters import ModuleAdapter, TrainCfg
from .jade_core import JadeCompact

__all__ = ["get_model", "list_models", "ModuleAdapter", "TrainCfg", "JadeCompact"]


def get_model(name: str, **kwargs) -> ModuleAdapter:
    """Get Stones model by name (jade only).

    Args:
        name: Model name (jade ONLY)
        **kwargs: Model-specific hyperparameters
            - input_size: Input feature dimension (default: 10)
            - hidden_size: LSTM hidden dimension (default: 96)
            - num_layers: Number of LSTM layers (default: 1)
            - dropout: Recurrent dropout (default: 0.7)
            - predict_pointers: Enable pointer prediction (default: False)
            - trainer: TrainCfg instance for training configuration

    Returns:
        ModuleAdapter wrapping JadeCompact model

    Raises:
        ValueError: If model name not 'jade'

    Examples:
        >>> # JadeCompact model
        >>> model = get_model("jade", input_size=10, hidden_size=96)
        >>>
        >>> # With training config
        >>> from moola.models import TrainCfg
        >>> cfg = TrainCfg(epochs=60, lr=3e-4, device="cuda")
        >>> model = get_model("jade", input_size=10, trainer=cfg)
    """
    # PAPER-STRICT: Only allow Jade
    allowed_models = {"jade"}
    n = name.lower()
    if n not in allowed_models:
        raise ValueError(
            f"PAPER-STRICT VIOLATION: Model '{name}' not allowed. "
            f"Only JadeCompact allowed: {sorted(allowed_models)}"
        )

    # Extract trainer config if provided
    trainer_cfg = kwargs.pop("trainer", None)

    # Default parameters for JadeCompact
    default_kwargs = {
        "input_size": 10,
        "hidden_size": 96,
        "num_layers": 1,
        "dropout": 0.7,
        "input_dropout": 0.3,
        "dense_dropout": 0.6,
        "num_classes": 3,
        "predict_pointers": False,
        "proj_head": True,
        "head_width": 64,
    }

    # Override with user-provided kwargs
    default_kwargs.update(kwargs)

    # Build JadeCompact model
    core = JadeCompact(**default_kwargs)

    # Wrap in adapter
    return ModuleAdapter(core, cfg=trainer_cfg)


def list_models() -> list[str]:
    """List all registered model names.

    Returns:
        List of available Stones model names
    """
    return ["jade"]

