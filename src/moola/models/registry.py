"""Model Registry System for MOOLA.

Provides centralized model management with codename system.
Registry format: moola-{family}-{size}{variant}-v{semver} // codename: {Stone}
"""

from .jade_core import JadeCompact

ALLOWED = {"jade"}


def build(cfg):
    """Build Stones model from Hydra config.

    Args:
        cfg: Hydra config with model, train, and data sections

    Returns:
        JadeCompact instance
    """
    # Hard invariant checks
    assert cfg.model.name in ALLOWED, f"Model name must be one of {ALLOWED}, got {cfg.model.name}"
    assert (
        cfg.model.pointer_head.encoding == "center_length"
    ), f"Pointer encoding must be 'center_length', got {cfg.model.pointer_head.encoding}"
    # Batch size updated to 32-64 range for better GPU utilization
    assert 32 <= cfg.train.batch_size <= 64, f"Batch size must be 32-64, got {cfg.train.batch_size}"

    # Build JadeCompact model
    model = JadeCompact(
        input_size=10,  # Fixed for RelativeTransform (6 candle + 4 swing features)
        hidden_size=getattr(cfg.model, "hidden_size", 96),
        num_layers=getattr(cfg.model, "num_layers", 1),
        dropout=getattr(cfg.model, "dropout", 0.7),
        input_dropout=getattr(cfg.model, "input_dropout", 0.3),
        dense_dropout=getattr(cfg.model, "dense_dropout", 0.6),
        num_classes=getattr(cfg.model, "num_classes", 3),
        predict_pointers=getattr(cfg.model, "predict_pointers", False),
        proj_head=False,  # Disable projection to reduce params
    )

    # Print parameter count
    param_stats = model.get_num_parameters()
    print(
        f"Model: {cfg.model.name} (JadeCompact) | "
        f"Total params: {param_stats['total']:,} | "
        f"Trainable: {param_stats['trainable']:,}"
    )

    # Expected range for JadeCompact (adjusted for input_size=10)
    assert (
        80000 <= param_stats["total"] <= 100000
    ), f"JadeCompact should have 80-100K params, got {param_stats['total']:,}"

    return model


def enforce_float32_precision():
    """Enforce float32 precision throughout the pipeline."""
    import torch

    torch.set_float32_matmul_precision("high")


def convert_batch_to_float32(batch):
    """Convert all tensors in batch to float32."""
    import torch

    if isinstance(batch, dict):
        return {k: v.float() if torch.is_tensor(v) else v for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [v.float() if torch.is_tensor(v) else v for v in batch]
    elif torch.is_tensor(batch):
        return batch.float()
    else:
        return batch


def initialize_model_biases(model):
    """Initialize model biases for better convergence.

    Implements logit bias initialization:
    - log_sigma_ptr = -0.30
    - log_sigma_cls = 0.00
    """
    import math

    # Initialize uncertainty parameters if they exist
    if hasattr(model, "log_sigma_ptr"):
        model.log_sigma_ptr.data.fill_(-0.30)
    if hasattr(model, "log_sigma_cls"):
        model.log_sigma_cls.data.fill_(0.00)

    # Initialize classification head biases for balanced classes
    if hasattr(model, "classifier") and hasattr(model.classifier, "bias"):
        # For 3-class problem, initialize to log(1/3) ≈ -1.1
        model.classifier.bias.data.fill_(-math.log(3))

    print("Model biases initialized for better convergence")
