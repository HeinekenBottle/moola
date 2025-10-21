"""Gradient monitoring and task collapse detection for multi-task learning.

This module provides utilities for monitoring gradient flow during training and detecting
task collapse in multi-task learning scenarios where one task dominates and silences others.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def compute_gradient_statistics(model: nn.Module) -> Dict[str, float]:
    """Compute gradient statistics across all model parameters.

    Args:
        model: PyTorch model with computed gradients

    Returns:
        Dictionary with gradient statistics:
            - grad_norm_mean: Mean of gradient norms across all parameters
            - grad_norm_max: Maximum gradient norm
            - grad_norm_min: Minimum gradient norm
            - grad_value_max: Maximum gradient value
            - grad_value_min: Minimum gradient value
    """
    grad_norms = []
    grad_max = []
    grad_min = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            grad_norms.append(grad.norm().item())
            grad_max.append(grad.max().item())
            grad_min.append(grad.min().item())

    return {
        "grad_norm_mean": np.mean(grad_norms) if grad_norms else 0.0,
        "grad_norm_max": np.max(grad_norms) if grad_norms else 0.0,
        "grad_norm_min": np.min(grad_norms) if grad_norms else 0.0,
        "grad_value_max": np.max(grad_max) if grad_max else 0.0,
        "grad_value_min": np.min(grad_min) if grad_min else 0.0,
    }


def compute_layer_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """Compute gradient norms per layer/module.

    Args:
        model: PyTorch model with computed gradients

    Returns:
        Dictionary mapping layer names to gradient norms
    """
    layer_norms = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_norms[name] = param.grad.norm().item()

    return layer_norms


def detect_vanishing_gradients(layer_norms: Dict[str, float], threshold: float = 1e-7) -> List[str]:
    """Detect layers with vanishing gradients.

    Args:
        layer_norms: Dictionary of layer names to gradient norms
        threshold: Threshold below which gradients are considered vanishing

    Returns:
        List of layer names with vanishing gradients
    """
    vanishing = []
    for layer, norm in layer_norms.items():
        if norm < threshold:
            vanishing.append(layer)
    return vanishing


def detect_exploding_gradients(layer_norms: Dict[str, float], threshold: float = 10.0) -> List[str]:
    """Detect layers with exploding gradients.

    Args:
        layer_norms: Dictionary of layer names to gradient norms
        threshold: Threshold above which gradients are considered exploding

    Returns:
        List of layer names with exploding gradients
    """
    exploding = []
    for layer, norm in layer_norms.items():
        if norm > threshold:
            exploding.append(layer)
    return exploding


def compute_task_gradient_ratio(
    pointer_loss: torch.Tensor, type_loss: torch.Tensor, model: nn.Module
) -> Dict[str, float]:
    """Compute gradient magnitude ratio between tasks.

    Helps detect task collapse where one task dominates training by computing
    the ratio of gradient norms from each task's loss.

    Args:
        pointer_loss: Pointer regression loss (detached, requires_grad=True)
        type_loss: Type classification loss (detached, requires_grad=True)
        model: Model to compute gradients for

    Returns:
        Dictionary with task gradient statistics:
            - pointer_grad_norm: L2 norm of pointer task gradients
            - type_grad_norm: L2 norm of type task gradients
            - grad_ratio: Ratio of pointer/type gradient norms
            - balanced: True if ratio is in healthy range [0.5, 2.0]
    """
    # Compute pointer task gradients
    model.zero_grad()
    pointer_loss.backward(retain_graph=True)
    pointer_grad_norm = (
        sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    )

    # Compute type task gradients
    model.zero_grad()
    type_loss.backward(retain_graph=True)
    type_grad_norm = (
        sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    )

    # Clear gradients
    model.zero_grad()

    # Compute ratio
    ratio = pointer_grad_norm / (type_grad_norm + 1e-8)

    return {
        "pointer_grad_norm": pointer_grad_norm,
        "type_grad_norm": type_grad_norm,
        "grad_ratio": ratio,
        "balanced": 0.5 <= ratio <= 2.0,  # Healthy range
    }


def detect_task_collapse(
    task_grad_ratios: List[float], window_size: int = 10, collapse_threshold: float = 5.0
) -> Tuple[bool, str]:
    """Detect if one task is dominating training (task collapse).

    Task collapse occurs when one task's gradients consistently overwhelm another's,
    preventing the weaker task from learning effectively.

    Args:
        task_grad_ratios: List of gradient ratios (pointer/type) over epochs
        window_size: Number of recent epochs to check
        collapse_threshold: Ratio threshold indicating collapse (e.g., 5.0 means 5x imbalance)

    Returns:
        (is_collapsed, message) tuple where:
            - is_collapsed: True if task collapse detected
            - message: Description of the task balance state
    """
    if len(task_grad_ratios) < window_size:
        return False, "Not enough data"

    recent_ratios = task_grad_ratios[-window_size:]
    mean_ratio = np.mean(recent_ratios)

    if mean_ratio > collapse_threshold:
        return (
            True,
            f"Pointer task dominating (ratio={mean_ratio:.2f} > {collapse_threshold})",
        )
    elif mean_ratio < 1.0 / collapse_threshold:
        return (
            True,
            f"Type task dominating (ratio={mean_ratio:.2f} < {1.0/collapse_threshold:.2f})",
        )
    else:
        return False, f"Tasks balanced (ratio={mean_ratio:.2f})"


class GradientMonitor:
    """Monitor gradient statistics during training.

    Tracks gradient norms, detects vanishing/exploding gradients, and maintains
    a history of gradient statistics for post-training analysis.
    """

    def __init__(self, log_frequency: int = 10):
        """Initialize gradient monitor.

        Args:
            log_frequency: How often to log gradient statistics (in epochs)
        """
        self.log_frequency = log_frequency
        self.history = defaultdict(list)

    def update(self, epoch: int, model: nn.Module):
        """Update gradient statistics.

        Args:
            epoch: Current epoch number
            model: Model with computed gradients
        """
        stats = compute_gradient_statistics(model)
        layer_norms = compute_layer_gradient_norms(model)

        # Store history
        for key, value in stats.items():
            self.history[key].append(value)

        # Log periodically
        if epoch % self.log_frequency == 0:
            print(f"Epoch {epoch} Gradient Stats:")
            print(f"  Mean norm: {stats['grad_norm_mean']:.4f}")
            print(f"  Max norm:  {stats['grad_norm_max']:.4f}")

            # Check for issues
            vanishing = detect_vanishing_gradients(layer_norms)
            if vanishing:
                print(f"  WARNING: Vanishing gradients in {len(vanishing)} layers")

            exploding = detect_exploding_gradients(layer_norms)
            if exploding:
                print(f"  WARNING: Exploding gradients in {len(exploding)} layers")

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics over all epochs.

        Returns:
            Dictionary mapping metric names to summary statistics (mean, std, min, max)
        """
        return {
            key: {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
            for key, values in self.history.items()
        }
