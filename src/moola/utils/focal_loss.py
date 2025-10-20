"""Backward compatibility shim for focal_loss module.

This module re-exports FocalLoss from its new location at
utils/metrics/focal_loss.py to maintain backward compatibility.
"""

from .metrics.focal_loss import FocalLoss

__all__ = ["FocalLoss"]
