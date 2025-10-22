"""Metrics module for MOOLA evaluation.

Stones-only metrics:
- Hit@±3 ≥60% threshold
- Pointer accuracy metrics
"""

from .hit_metrics import (
    compute_joint_success_metrics,
    compute_pointer_metrics,
    hit_at_k,
)

__all__ = [
    "hit_at_k",
    "compute_pointer_metrics",
    "compute_joint_success_metrics",
]
