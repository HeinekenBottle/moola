"""Backward compatibility shim for losses module.

This module re-exports losses from its new location at
utils/metrics/losses.py to maintain backward compatibility.
"""

from .metrics.losses import compute_multitask_loss

__all__ = ["compute_multitask_loss"]
