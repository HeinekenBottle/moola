"""Model Registry System for MOOLA.

Provides centralized model management with codename system.
Registry format: moola-{family}-{size}{variant}-v{semver} // codename: {Stone}
"""

from .jade import JadeModel

ALLOWED = {"jade","sapphire","opal"}

def build(cfg):
    # Hard invariant checks
    assert cfg.model.name in ALLOWED, f"Model name must be one of {ALLOWED}, got {cfg.model.name}"
    assert cfg.model.pointer_head.encoding == "center_length", f"Pointer encoding must be 'center_length', got {cfg.model.pointer_head.encoding}"
    assert cfg.train.batch_size == 29, f"Batch size must be 29, got {cfg.train.batch_size}"
    
    # Build model with correct parameters for JadeModel
    model = JadeModel(
        input_size=11,  # Fixed for RelativeTransform
        hidden_size=getattr(cfg.model, 'hidden_size', 96),
        num_layers=getattr(cfg.model, 'num_layers', 1),
        bidirectional=getattr(cfg.model, 'bidirectional', True),
        proj_head=getattr(cfg.model, 'proj_head', True),
        head_width=getattr(cfg.model, 'head_width', 64),
        pointer_encoding=cfg.model.pointer_head.encoding
    )
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg.model.name} | Total params: {total_params:,} | Trainable: {trainable_params:,}")
    
    # Expected ranges for Jade_Compact (adjusted for input_size=11)
    if cfg.model.name == "jade":
        assert 40000 <= total_params <= 120000, f"Jade_Compact should have 40-120K params, got {total_params:,}"
    
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