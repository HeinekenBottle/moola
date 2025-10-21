"""Monitoring utilities for ML training diagnostics.

This module provides tools for monitoring gradient flow, detecting task collapse,
and tracking training health in multi-task learning scenarios.
"""

from .gradient_diagnostics import (
    GradientMonitor,
    compute_gradient_statistics,
    compute_layer_gradient_norms,
    compute_task_gradient_ratio,
    detect_exploding_gradients,
    detect_task_collapse,
    detect_vanishing_gradients,
)

__all__ = [
    "GradientMonitor",
    "compute_gradient_statistics",
    "compute_layer_gradient_norms",
    "compute_task_gradient_ratio",
    "detect_vanishing_gradients",
    "detect_exploding_gradients",
    "detect_task_collapse",
]
