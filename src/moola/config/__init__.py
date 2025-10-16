"""Centralized configuration system for Moola ML pipeline.

This module provides standardized hyperparameters, model specifications, and
data configurations to ensure reproducibility and prevent misconfigurations.

Usage:
    from moola.config import training_config, model_config, data_config

    # Access training hyperparameters
    batch_size = training_config.DEFAULT_BATCH_SIZE
    learning_rate = training_config.CNNTR_LEARNING_RATE

    # Access model specs
    arch = model_config.MODEL_ARCHITECTURES['cnn_transformer']

    # Access data specs
    window_size = data_config.EXPECTED_WINDOW_LENGTH
"""

from . import data_config, model_config, training_config

__all__ = [
    "training_config",
    "model_config",
    "data_config",
]
