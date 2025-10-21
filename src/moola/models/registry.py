"""Model Registry System for MOOLA.

Provides centralized model management with codename system.
Registry format: moola-{family}-{size}{variant}-v{semver} // codename: {Stone}
"""

import re
from typing import Dict, Optional, Type, Any
from dataclasses import dataclass
from pathlib import Path
import torch

try:
    from .enhanced_simple_lstm import EnhancedSimpleLSTMModel as JadeModel
except ImportError:
    # Fallback for development
    from .enhanced_simple_lstm import EnhancedSimpleLSTMModel as JadeModel


@dataclass
class ModelInfo:
    """Model information for registry."""

    model_id: str  # e.g., "moola-lstm-m-v1.0"
    codename: str  # e.g., "Jade"
    family: str  # e.g., "lstm"
    size: str  # e.g., "m" for medium
    variant: str  # e.g., "" for base, "fr" for frozen encoder
    version: str  # e.g., "v1.0"
    description: str
    model_class: Type
    default_params: Dict[str, Any]
    stones_compliant: bool = True


class ModelRegistry:
    """Centralized model registry with codename system."""

    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._codenames: Dict[str, str] = {}  # codename -> model_id
        self._register_builtin_models()

    def _register_builtin_models(self):
        """Register built-in MOOLA models."""

        # Jade: Production BiLSTM with Multi-task Learning
        self.register(
            model_id="moola-lstm-m-v1.0",
            codename="Jade",
            family="lstm",
            size="m",  # Medium
            variant="",  # Base variant
            version="v1.0",
            description="Production BiLSTM with uncertainty-weighted multi-task learning",
            model_class=JadeModel,
            default_params={
                "hidden_size": 128,
                "num_layers": 2,
                "predict_pointers": False,
                "use_uncertainty_weighting": True,  # Default for Jade
                "max_grad_norm": 2.0,
                "early_stopping_patience": 20,
            },
        )

        # Sapphire: Pre-trained encoder with frozen weights
        self.register(
            model_id="moola-lstm-sf-v1.0",
            codename="Sapphire",
            family="lstm",  # LSTM family
            size="s",  # Small
            variant="f",  # Frozen encoder
            version="v1.0",
            description="Pre-trained encoder with frozen weights for transfer learning",
            model_class=JadeModel,  # Same class, different config
            default_params={
                "hidden_size": 128,
                "num_layers": 2,
                "predict_pointers": False,
                "use_uncertainty_weighting": True,
                "max_grad_norm": 2.0,
                "early_stopping_patience": 20,
                "freeze_encoder": True,
                "pretrained_encoder_path": "artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt",
            },
        )

        # Opal: Pre-trained encoder with adaptive fine-tuning
        self.register(
            model_id="moola-preenc-ma-v1.0",
            codename="Opal",
            family="preenc",  # Pre-trained encoder
            size="m",  # Medium
            variant="a",  # Adaptive fine-tuning
            version="v1.0",
            description="Pre-trained encoder with adaptive fine-tuning for optimal transfer",
            model_class=JadeModel,  # Same class, different config
            default_params={
                "hidden_size": 128,
                "num_layers": 2,
                "predict_pointers": False,
                "use_uncertainty_weighting": True,
                "max_grad_norm": 2.0,
                "early_stopping_patience": 20,
                "freeze_encoder": False,
                "unfreeze_encoder_after": 10,
                "pretrained_encoder_path": "artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt",
            },
        )

    def register(
        self,
        model_id: str,
        codename: str,
        family: str,
        size: str,
        variant: str,
        version: str,
        description: str,
        model_class: Type,
        default_params: Dict[str, Any],
        stones_compliant: bool = True,
    ) -> None:
        """Register a model in the registry.

        Args:
            model_id: Model ID in format moola-{family}-{size}{variant}-v{semver}
            codename: Stone codename (e.g., Jade, Sapphire, Opal)
            family: Model family (e.g., lstm, preenc)
            size: Model size (s, m, l, xl)
            variant: Variant identifier (fr, ad, etc.)
            version: Semantic version (v1.0, v1.1, etc.)
            description: Model description
            model_class: Model class
            default_params: Default hyperparameters
            stones_compliant: Whether model follows Stones specifications
        """
        # Validate model_id format
        pattern = r"^moola-[a-z]+-[smlxl][a-z]*-v\d+\.\d+$"
        if not re.match(pattern, model_id):
            raise ValueError(
                f"Invalid model_id format: {model_id}. Expected: moola-{{family}}-{{size}}{{variant}}-v{{semver}}"
            )

        # Check for duplicates
        if model_id in self._models:
            raise ValueError(f"Model ID already registered: {model_id}")
        if codename in self._codenames:
            raise ValueError(f"Codename already registered: {codename}")

        model_info = ModelInfo(
            model_id=model_id,
            codename=codename,
            family=family,
            size=size,
            variant=variant,
            version=version,
            description=description,
            model_class=model_class,
            default_params=default_params,
            stones_compliant=stones_compliant,
        )

        self._models[model_id] = model_info
        self._codenames[codename] = model_id

    def get(self, identifier: str) -> ModelInfo:
        """Get model info by ID or codename.

        Args:
            identifier: Model ID or codename

        Returns:
            ModelInfo object
        """
        if identifier in self._models:
            return self._models[identifier]
        elif identifier in self._codenames:
            return self._models[self._codenames[identifier]]
        else:
            raise ValueError(f"Model not found: {identifier}")

    def create_model(self, identifier: str, **kwargs) -> Any:
        """Create model instance by ID or codename.

        Args:
            identifier: Model ID or codename
            **kwargs: Override default parameters

        Returns:
            Model instance
        """
        model_info = self.get(identifier)

        # Merge default params with overrides
        params = model_info.default_params.copy()
        params.update(kwargs)

        return model_info.model_class(**params)

    def list_models(self) -> Dict[str, ModelInfo]:
        """List all registered models.

        Returns:
            Dictionary of model_id -> ModelInfo
        """
        return self._models.copy()

    def list_stones_compliant(self) -> Dict[str, ModelInfo]:
        """List Stones-compliant models.

        Returns:
            Dictionary of model_id -> ModelInfo for Stones-compliant models
        """
        return {model_id: info for model_id, info in self._models.items() if info.stones_compliant}

    def search(self, **filters) -> Dict[str, ModelInfo]:
        """Search models by criteria.

        Args:
            **filters: Filter criteria (family, size, variant, stones_compliant)

        Returns:
            Dictionary of matching model_id -> ModelInfo
        """
        results = {}

        for model_id, info in self._models.items():
            match = True

            for key, value in filters.items():
                if hasattr(info, key):
                    if getattr(info, key) != value:
                        match = False
                        break
                else:
                    match = False
                    break

            if match:
                results[model_id] = info

        return results

    def get_stones_models(self) -> Dict[str, ModelInfo]:
        """Get Stones collection models (Jade, Sapphire, Opal).

        Returns:
            Dictionary of codename -> ModelInfo for Stones models
        """
        stones_codenames = ["Jade", "Sapphire", "Opal"]
        return {
            info.codename: info
            for info in self._models.values()
            if info.codename in stones_codenames
        }


# Global registry instance
registry = ModelRegistry()


def get_model(identifier: str, **kwargs) -> Any:
    """Get model instance by ID or codename.

    Args:
        identifier: Model ID or codename
        **kwargs: Override default parameters

    Returns:
        Model instance
    """
    return registry.create_model(identifier, **kwargs)


def list_available_models() -> Dict[str, str]:
    """List available models with descriptions.

    Returns:
        Dictionary of identifier -> description
    """
    models = registry.list_models()
    result = {}

    for model_id, info in models.items():
        result[model_id] = info.description
        result[info.codename] = info.description

    return result


def get_stones_collection() -> Dict[str, str]:
    """Get Stones collection models.

    Returns:
        Dictionary of codename -> model_id
    """
    stones_models = registry.get_stones_models()
    return {codename: info.model_id for codename, info in stones_models.items()}


def validate_model_id(model_id: str) -> bool:
    """Validate model ID format.

    Args:
        model_id: Model ID to validate

    Returns:
        True if valid format
    """
    pattern = r"^moola-[a-z]+-[xsmlxl]*[a-z]*-v\d+\.\d+$"
    return bool(re.match(pattern, model_id))


# Convenience functions for Stones models
def get_jade(**kwargs) -> JadeModel:
    """Get Jade model.

    Args:
        **kwargs: Override default parameters

    Returns:
        JadeModel instance
    """
    return get_model("Jade", **kwargs)


def get_sapphire(**kwargs) -> JadeModel:
    """Get Sapphire model.

    Args:
        **kwargs: Override default parameters

    Returns:
        JadeModel instance configured for Sapphire
    """
    return get_model("Sapphire", **kwargs)


def get_opal(**kwargs) -> JadeModel:
    """Get Opal model.

    Args:
        **kwargs: Override default parameters

    Returns:
        JadeModel instance configured for Opal
    """
    return get_model("Opal", **kwargs)
