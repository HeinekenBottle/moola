"""Backward compatibility shim for training_utils module.

This module re-exports TrainingSetup from its new location at
utils/training/training_utils.py to maintain backward compatibility.
"""

from .training.training_utils import TrainingSetup

__all__ = ["TrainingSetup"]
