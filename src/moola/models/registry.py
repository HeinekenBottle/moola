"""Model Registry System for MOOLA.

Provides centralized model management with codename system.
Registry format: moola-{family}-{size}{variant}-v{semver} // codename: {Stone}
"""

from .jade_core import JadeCore, JadeCompact

ALLOWED = {"jade", "sapphire", "opal"}

def build(cfg):
    """Build Stones model from Hydra config.

    Args:
        cfg: Hydra config with model, train, and data sections

    Returns:
        JadeCore or JadeCompact instance
    """
    # Hard invariant checks
    assert cfg.model.name in ALLOWED, f"Model name must be one of {ALLOWED}, got {cfg.model.name}"
    assert cfg.model.pointer_head.encoding == "center_length", f"Pointer encoding must be 'center_length', got {cfg.model.pointer_head.encoding}"
    assert cfg.train.batch_size == 29, f"Batch size must be 29, got {cfg.train.batch_size}"

    # Determine model variant
    use_compact = getattr(cfg.model, 'use_compact', False)
    model_cls = JadeCompact if use_compact else JadeCore

    # Build model with correct parameters
    model = model_cls(
        input_size=11,  # Fixed for RelativeTransform
        hidden_size=getattr(cfg.model, 'hidden_size', 96 if use_compact else 128),
        num_layers=getattr(cfg.model, 'num_layers', 1 if use_compact else 2),
        dropout=getattr(cfg.model, 'dropout', 0.7 if use_compact else 0.65),
        input_dropout=getattr(cfg.model, 'input_dropout', 0.3 if use_compact else 0.25),
        dense_dropout=getattr(cfg.model, 'dense_dropout', 0.6 if use_compact else 0.5),
        num_classes=getattr(cfg.model, 'num_classes', 3),
        predict_pointers=getattr(cfg.model, 'predict_pointers', False),
    )

    # Print parameter count
    param_stats = model.get_num_parameters()
    print(f"Model: {cfg.model.name} ({model_cls.__name__}) | "
          f"Total params: {param_stats['total']:,} | "
          f"Trainable: {param_stats['trainable']:,}")

    # Expected ranges for Jade variants (adjusted for input_size=11)
    if cfg.model.name == "jade":
        if use_compact:
            assert 40000 <= param_stats['total'] <= 80000, \
                f"Jade-Compact should have 40-80K params, got {param_stats['total']:,}"
        else:
            assert 80000 <= param_stats['total'] <= 150000, \
                f"Jade should have 80-150K params, got {param_stats['total']:,}"

    return model

def enforce_float32_precision():
    """Enforce float32 precision throughout the pipeline."""
    import torch
    torch.set_float32_matmul_precision('high')

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