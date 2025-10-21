"""Model architecture and compatibility specifications.

Defines expected input shapes, architecture parameters, and device compatibility
for all supported models in the Moola pipeline.

Philosophy:
- Single registry of all models and their specs
- Enables validation before training
- Prevents model-data mismatch issues
"""

# ============================================================================
# MODEL ARCHITECTURE REGISTRY
# ============================================================================

MODEL_ARCHITECTURES = {
    "cnn_transformer": {
        "name": "CNN-Transformer",
        "type": "deep_learning",
        "input_dim": 4,  # OHLC: Open, High, Low, Close
        "supports_multiclass": True,
        "supports_pointer_prediction": True,
        "cnn_channels": [64, 128, 128],
        "cnn_kernels": [3, 5, 9],
        "transformer_layers": 3,
        "transformer_heads": 4,
        "description": "Hybrid CNN→Transformer for time series classification with optional pointer prediction",
    },
    "rwkv_ts": {
        "name": "RWKV-TS",
        "type": "deep_learning",
        "input_dim": 4,
        "supports_multiclass": True,
        "supports_pointer_prediction": False,
        "hidden_dim": 128,
        "description": "Recurrent Vision Transformer for time series",
    },
    "simple_lstm": {
        "name": "Simple LSTM",
        "type": "deep_learning",
        "input_dim": 4,
        "supports_multiclass": True,
        "supports_pointer_prediction": False,
        "hidden_dim": 64,
        "num_layers": 2,
        "description": "Simple LSTM-based time series classifier",
    },
    "logreg": {
        "name": "Logistic Regression",
        "type": "traditional_ml",
        "supports_multiclass": True,
        "supports_pointer_prediction": False,
        "description": "Baseline linear classifier",
    },
    "rf": {
        "name": "Random Forest",
        "type": "tree_based",
        "supports_multiclass": True,
        "supports_pointer_prediction": False,
        "n_estimators": 100,
        "description": "Ensemble tree-based classifier",
    },
    "xgb": {
        "name": "XGBoost",
        "type": "tree_based",
        "supports_multiclass": True,
        "supports_pointer_prediction": False,
        "n_estimators": 100,
        "description": "Gradient boosted decision trees",
    },
    "stack": {
        "name": "Stacking Meta-Learner",
        "type": "ensemble_meta_learner",
        "supports_multiclass": True,
        "supports_pointer_prediction": False,
        "base_models": ["logreg", "rf", "xgb", "rwkv_ts", "cnn_transformer"],
        "description": "Random Forest meta-learner trained on base model predictions",
    },
}


# ============================================================================
# DEVICE COMPATIBILITY
# ============================================================================

MODEL_DEVICE_COMPATIBILITY = {
    "cnn_transformer": ["cpu", "cuda"],  # GPU-recommended but works on CPU
    "rwkv_ts": ["cpu", "cuda"],
    "simple_lstm": ["cpu", "cuda"],
    "logreg": ["cpu"],  # No GPU benefit
    "rf": ["cpu"],  # sklearn does not support GPU
    "xgb": ["cpu"],  # XGBoost GPU support is optional
    "stack": ["cpu"],  # Meta-learner always CPU
}


# ============================================================================
# INPUT VALIDATION SPECS
# ============================================================================

REQUIRED_INPUT_SHAPES = {
    "cnn_transformer": {
        "description": "3D array [N_samples, 105, 4] for OHLC time series",
        "min_samples": 20,  # Minimum samples for training
        "max_samples": None,  # No upper limit
        "sequence_length": 105,
        "feature_dim": 4,
    },
    "rwkv_ts": {
        "description": "3D array [N_samples, seq_len, 4] for time series",
        "min_samples": 20,
        "max_samples": None,
        "sequence_length": None,  # Variable length supported
        "feature_dim": 4,
    },
    "simple_lstm": {
        "description": "3D array [N_samples, seq_len, 4] for time series",
        "min_samples": 20,
        "max_samples": None,
        "sequence_length": None,
        "feature_dim": 4,
    },
    "logreg": {
        "description": "2D array [N_samples, N_features] for tabular data",
        "min_samples": 10,
        "max_samples": None,
        "sequence_length": None,
        "feature_dim": None,
    },
    "rf": {
        "description": "2D array [N_samples, N_features] for tabular data",
        "min_samples": 10,
        "max_samples": None,
        "sequence_length": None,
        "feature_dim": None,
    },
    "xgb": {
        "description": "2D array [N_samples, N_features] for tabular data",
        "min_samples": 10,
        "max_samples": None,
        "sequence_length": None,
        "feature_dim": None,
    },
    "stack": {
        "description": "Concatenated predictions from 5 base models",
        "min_samples": 10,
        "max_samples": None,
        "sequence_length": None,
        "feature_dim": 15,  # 5 models × 3 classes
    },
}


# ============================================================================
# LABEL SPECIFICATIONS
# ============================================================================

VALID_LABELS = ["consolidation", "retracement", "expansion"]
"""Valid class labels for classification. Dynamically updated during training."""

MIN_SAMPLES_PER_CLASS = 2
"""Minimum samples required per class to avoid SMOTE failures."""

MIN_CLASSES = 2
"""Minimum number of classes (binary or multi-class)."""


# ============================================================================
# OUTPUT SPECIFICATIONS
# ============================================================================

OUTPUT_SPECS = {
    "predictions": {
        "description": "Class label predictions",
        "shape": "[N_samples]",
        "dtype": "object (strings) or int (indices)",
    },
    "probabilities": {
        "description": "Class probability estimates",
        "shape": "[N_samples, N_classes]",
        "dtype": "float32",
        "sum_to_one": True,
    },
    "pointer_start": {
        "description": "Start position within inner window [30:75]",
        "shape": "[N_samples]",
        "dtype": "int64",
        "range": "[0, 44]",
    },
    "pointer_end": {
        "description": "End position within inner window [30:75]",
        "shape": "[N_samples]",
        "dtype": "int64",
        "range": "[0, 44]",
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_model_spec(model_name: str) -> dict:
    """Get architecture specification for a model.

    Args:
        model_name: Model name from MODEL_ARCHITECTURES keys

    Returns:
        Dictionary with model specifications

    Raises:
        ValueError: If model_name not in registry
    """
    if model_name not in MODEL_ARCHITECTURES:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_ARCHITECTURES.keys())}"
        )
    return MODEL_ARCHITECTURES[model_name]


def supports_gpu(model_name: str) -> bool:
    """Check if model has GPU support.

    Args:
        model_name: Model name

    Returns:
        True if model can use CUDA, False otherwise
    """
    if model_name not in MODEL_DEVICE_COMPATIBILITY:
        return False
    return "cuda" in MODEL_DEVICE_COMPATIBILITY[model_name]


def supports_multiclass(model_name: str) -> bool:
    """Check if model supports multi-class classification.

    Args:
        model_name: Model name

    Returns:
        True if model supports multi-class, False for binary only
    """
    spec = get_model_spec(model_name)
    return spec.get("supports_multiclass", False)


def supports_pointer_prediction(model_name: str) -> bool:
    """Check if model supports pointer start/end prediction.

    Args:
        model_name: Model name

    Returns:
        True if model supports multi-task pointer prediction
    """
    spec = get_model_spec(model_name)
    return spec.get("supports_pointer_prediction", False)


# ============================================================================
# EXPORT ALL SPECIFICATIONS
# ============================================================================

__all__ = [
    "MODEL_ARCHITECTURES",
    "MODEL_DEVICE_COMPATIBILITY",
    "REQUIRED_INPUT_SHAPES",
    "VALID_LABELS",
    "MIN_SAMPLES_PER_CLASS",
    "MIN_CLASSES",
    "OUTPUT_SPECS",
    "get_model_spec",
    "supports_gpu",
    "supports_multiclass",
    "supports_pointer_prediction",
]
