"""Backward compatibility shim for temporal_augmentation module.

This module re-exports TemporalAugmentation from its new location at
utils/augmentation/temporal_augmentation.py to maintain backward compatibility.
"""

from .augmentation.temporal_augmentation import TemporalAugmentation

__all__ = ["TemporalAugmentation"]
