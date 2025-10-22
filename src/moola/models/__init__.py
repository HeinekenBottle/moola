"""Model registry for Moola ML pipeline.

Provides centralized model instantiation via get_model() function.
Clean architecture with separation of concerns:
- JadeCore/JadeCompact: Pure PyTorch nn.Module (architecture only)
- ModuleAdapter: Thin wrapper for fit/predict/save (backward compatibility)

Stones collection with codename system - PAPER-STRICT COMPLIANCE:
- Jade: moola-lstm-m-v1.0 // codename: Jade
- Sapphire: moola-lstm-sf-v1.0 // codename: Sapphire  
- Opal: moola-preenc-ma-v1.0 // codename: Opal

ONLY THESE MODELS ARE ALLOWED FOR PAPER EXPERIMENTS.
"""

from typing import Optional

from .adapters import ModuleAdapter, TrainCfg
from .jade_core import JadeCompact, JadeCore

__all__ = ["get_model", "list_models", "ModuleAdapter", "TrainCfg", "JadeCore", "JadeCompact"]


def get_model(name: str, **kwargs) -> ModuleAdapter:
    """Get Stones model by name (jade/opal/sapphire only).
    
    Args:
        name: Model name (jade, sapphire, opal ONLY)
        **kwargs: Model-specific hyperparameters
            - input_size: Input feature dimension (default: 11)
            - hidden_size: LSTM hidden dimension (default: 128 for jade, 96 for compact)
            - num_layers: Number of LSTM layers (default: 2 for jade, 1 for compact)
            - dropout: Recurrent dropout (default: 0.65)
            - predict_pointers: Enable pointer prediction (default: False)
            - trainer: TrainCfg instance for training configuration
            - use_compact: Use JadeCompact variant (default: False)
    
    Returns:
        ModuleAdapter wrapping the requested model
    
    Raises:
        ValueError: If model name not in Stones registry
    
    Examples:
        >>> # Standard Jade model
        >>> model = get_model("jade", input_size=11, hidden_size=128)
        >>> 
        >>> # Compact variant for small datasets
        >>> model = get_model("jade", input_size=11, use_compact=True)
        >>> 
        >>> # With training config
        >>> from moola.models import TrainCfg
        >>> cfg = TrainCfg(epochs=60, lr=3e-4, device="cuda")
        >>> model = get_model("jade", input_size=11, trainer=cfg)
    """
    # PAPER-STRICT: Only allow Stones models
    allowed_models = {"jade", "sapphire", "opal"}
    n = name.lower()
    if n not in allowed_models:
        raise ValueError(
            f"PAPER-STRICT VIOLATION: Model '{name}' not allowed. "
            f"Only Stones models allowed: {sorted(allowed_models)}"
        )
    
    # Extract trainer config if provided
    trainer_cfg = kwargs.pop("trainer", None)
    
    # Determine if using compact variant
    use_compact = kwargs.pop("use_compact", False)
    
    # Default parameters based on model variant
    if use_compact:
        # Jade-Compact: 1-layer, 96 hidden, projection head
        default_kwargs = {
            "input_size": 11,
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
        model_cls = JadeCompact
    else:
        # Standard Jade: 2-layer, 128 hidden
        default_kwargs = {
            "input_size": 11,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.65,
            "input_dropout": 0.25,
            "dense_dropout": 0.5,
            "num_classes": 3,
            "predict_pointers": False,
        }
        model_cls = JadeCore
    
    # Override with user-provided kwargs
    default_kwargs.update(kwargs)
    
    # Build core model
    core = model_cls(**default_kwargs)
    
    # Wrap in adapter
    return ModuleAdapter(core, cfg=trainer_cfg)


def list_models() -> list[str]:
    """List all registered model names.
    
    Returns:
        List of available Stones model names
    """
    return ["jade", "sapphire", "opal"]

