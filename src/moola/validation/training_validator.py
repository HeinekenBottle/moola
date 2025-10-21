"""Training validation utilities for debugging and verification.

This module provides pre-flight checks and runtime validation for ML training:
- Encoder weight verification (SSL transfer learning)
- Class collapse detection (imbalanced classification)
- Gradient flow verification (frozen layers)
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def validate_encoder_loading(
    model: nn.Module, encoder_path: Path, tolerance: float = 1e-6
) -> Dict[str, int]:
    """Verify encoder weights loaded correctly from checkpoint.

    This function compares the encoder weights in the model against the checkpoint
    to ensure they match exactly (within tolerance). Critical for SSL transfer learning.

    Args:
        model: PyTorch model with loaded encoder weights
        encoder_path: Path to encoder checkpoint (.pt file)
        tolerance: Numerical tolerance for weight comparison (default: 1e-6)

    Returns:
        Dictionary with verification statistics:
        {
            'total_layers': int,
            'matched_layers': int,
            'mismatched_layers': int,
            'missing_layers': int
        }

    Raises:
        FileNotFoundError: If encoder checkpoint doesn't exist
        AssertionError: If critical layers fail to match

    Example:
        >>> model = CnnTransformerModel()
        >>> model.load_pretrained_encoder('encoder.pt')
        >>> stats = validate_encoder_loading(model.model, 'encoder.pt')
        >>> assert stats['matched_layers'] > 0, "No encoder weights loaded!"
    """
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")

    # Load checkpoint
    checkpoint = torch.load(encoder_path, map_location="cpu")
    encoder_state = checkpoint["encoder_state_dict"]

    # Get model state dict
    model_state = model.state_dict()

    # Track verification stats
    stats = {
        "total_layers": len(encoder_state),
        "matched_layers": 0,
        "mismatched_layers": 0,
        "missing_layers": 0,
    }

    # Compare each layer
    for key, encoder_weight in encoder_state.items():
        if key not in model_state:
            print(f"[VALIDATION] ⚠️  Layer {key} missing in model")
            stats["missing_layers"] += 1
            continue

        model_weight = model_state[key]

        # Check shape match
        if encoder_weight.shape != model_weight.shape:
            print(f"[VALIDATION] ❌ Shape mismatch: {key}")
            print(f"              Encoder: {encoder_weight.shape}, Model: {model_weight.shape}")
            stats["mismatched_layers"] += 1
            continue

        # Check weight values match
        if not torch.allclose(encoder_weight, model_weight, atol=tolerance):
            print(f"[VALIDATION] ❌ Weight mismatch: {key}")
            max_diff = (encoder_weight - model_weight).abs().max().item()
            print(f"              Max difference: {max_diff:.2e} (tolerance: {tolerance:.2e})")
            stats["mismatched_layers"] += 1
            continue

        stats["matched_layers"] += 1

    # Print summary
    print(f"\n[VALIDATION] Encoder Loading Verification:")
    print(f"  ✓ Matched layers: {stats['matched_layers']}/{stats['total_layers']}")
    if stats["mismatched_layers"] > 0:
        print(f"  ❌ Mismatched layers: {stats['mismatched_layers']}")
    if stats["missing_layers"] > 0:
        print(f"  ⚠️  Missing layers: {stats['missing_layers']}")

    # Assert critical layers loaded
    if stats["matched_layers"] == 0:
        raise AssertionError("CRITICAL: No encoder weights loaded successfully!")

    return stats


def detect_class_collapse(
    predictions: np.ndarray,
    labels: np.ndarray,
    epoch: int,
    threshold: float = 0.1,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[int, float]:
    """Detect class collapse during training (early warning for imbalanced learning).

    Class collapse occurs when a model stops predicting certain classes, often due to:
    - Extreme class imbalance
    - Frozen encoder preventing learning
    - Poor loss weighting

    Args:
        predictions: Predicted class indices [N]
        labels: True class labels [N]
        epoch: Current training epoch (for logging)
        threshold: Accuracy threshold below which to warn (default: 0.1 = 10%)
        class_names: Optional mapping {class_idx -> name} for readable output

    Returns:
        Dictionary mapping class index to per-class accuracy:
        {
            0: 0.85,  # Class 0: 85% accuracy
            1: 0.03,  # Class 1: 3% accuracy (COLLAPSED!)
        }

    Example:
        >>> preds = model.predict(X_val)
        >>> class_accs = detect_class_collapse(preds, y_val, epoch=15)
        >>> if class_accs[1] < 0.1:
        >>>     print("WARNING: Class 1 collapsed!")
    """
    unique_classes = np.unique(labels)
    class_accs = {}

    print(f"\n[CLASS BALANCE] Epoch {epoch} - Per-class Accuracy:")

    for cls in unique_classes:
        # Get samples for this class
        mask = labels == cls
        n_samples = mask.sum()

        if n_samples == 0:
            continue

        # Calculate accuracy for this class
        correct = (predictions[mask] == cls).sum()
        accuracy = correct / n_samples
        class_accs[cls] = accuracy

        # Format class name
        if class_names and cls in class_names:
            class_label = f"Class {cls} ({class_names[cls]})"
        else:
            class_label = f"Class {cls}"

        # Warn if below threshold
        if accuracy < threshold:
            print(
                f"  ⚠️  {class_label}: {accuracy:.1%} ({correct}/{n_samples}) - COLLAPSE DETECTED!"
            )
        else:
            print(f"  ✓ {class_label}: {accuracy:.1%} ({correct}/{n_samples})")

    return class_accs


def verify_gradient_flow(model: nn.Module, phase: str = "training") -> Dict[str, Dict[str, int]]:
    """Verify gradients are flowing correctly through the model.

    Use this to debug:
    - Frozen layers (should have requires_grad=False and no gradients)
    - Trainable layers (should have requires_grad=True and gradients after backward)
    - Vanishing/exploding gradients

    Args:
        model: PyTorch model to inspect
        phase: Training phase for logging (e.g., "frozen", "unfrozen_stage1")

    Returns:
        Dictionary with gradient flow statistics:
        {
            'frozen': {'count': int, 'params': [names]},
            'trainable': {'count': int, 'params': [names]},
            'gradients': {'count': int, 'params': [names]},  # After backward pass
        }

    Example:
        >>> model.freeze_encoder()
        >>> stats = verify_gradient_flow(model.model, phase="frozen")
        >>> assert stats['frozen']['count'] > 0, "No layers frozen!"
    """
    frozen_params = []
    trainable_params = []
    gradient_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen_params.append(name)
        else:
            trainable_params.append(name)
            if param.grad is not None:
                gradient_params.append(name)

    stats = {
        "frozen": {"count": len(frozen_params), "params": frozen_params[:5]},  # Show first 5
        "trainable": {"count": len(trainable_params), "params": trainable_params[:5]},
        "gradients": {"count": len(gradient_params), "params": gradient_params[:5]},
    }

    # Print summary
    print(f"\n[GRADIENT FLOW] Phase: {phase}")
    print(f"  Frozen params: {stats['frozen']['count']}")
    if stats["frozen"]["count"] > 0:
        print(f"    Examples: {', '.join(stats['frozen']['params'][:3])}")

    print(f"  Trainable params: {stats['trainable']['count']}")
    if stats["trainable"]["count"] > 0:
        print(f"    Examples: {', '.join(stats['trainable']['params'][:3])}")

    print(f"  Params with gradients: {stats['gradients']['count']}")
    if stats["gradients"]["count"] > 0:
        print(f"    Examples: {', '.join(stats['gradients']['params'][:3])}")

    # Warnings
    if stats["trainable"]["count"] == 0:
        print("  ⚠️  WARNING: No trainable parameters! Model won't learn.")

    if stats["gradients"]["count"] == 0 and phase != "frozen":
        print("  ⚠️  WARNING: No gradients computed! Call backward() first.")

    return stats
