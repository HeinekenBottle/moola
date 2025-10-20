"""Backward compatibility shim for early_stopping module.

This module re-exports EarlyStopping from its new location at
utils/training/early_stopping.py to maintain backward compatibility.
"""

from .training.early_stopping import EarlyStopping

__all__ = ["EarlyStopping"]
