"""Pretrained weight loading with strict validation and reporting."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger


def load_pretrained_strict(
    model: nn.Module,
    checkpoint_path: str,
    freeze_encoder: bool = True,
    min_match_ratio: float = 0.20,  # Lower for encoder-only loading (multi-task models have extra heads)
    allow_shape_mismatch: bool = False,
) -> dict[str, Any]:
    """Load pretrained weights with strict validation.

    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to .pt checkpoint file
        freeze_encoder: If True, freeze encoder parameters
        min_match_ratio: Minimum ratio of matched tensors (default 0.80)
        allow_shape_mismatch: If True, allow shape mismatches (default False)

    Returns:
        Dictionary with:
            - matched: List of matched tensor names
            - missing: List of missing tensor names
            - mismatched: List of (name, checkpoint_shape, model_shape) tuples
            - n_matched: Count of matched tensors
            - n_missing: Count of missing tensors
            - n_mismatched: Count of shape mismatches
            - match_ratio: matched / total_model_tensors
            - n_frozen: Count of frozen parameters

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        AssertionError: If match_ratio < min_match_ratio or shape mismatches exist
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint_path}\n"
            f"Available encoders should be in: artifacts/pretrained/"
        )

    # Load checkpoint
    logger.info(f"Loading pretrained weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "encoder_state_dict" in checkpoint:
        # Handle encoder-only checkpoints (from pretraining)
        state_dict = checkpoint["encoder_state_dict"]
    else:
        state_dict = checkpoint

    # Get model state
    model_state = model.state_dict()

    # Match tensors
    matched = []
    missing = []
    mismatched = []

    for name, param in state_dict.items():
        # Try direct match first
        model_key = name
        if model_key not in model_state:
            # Try adding 'lstm.' prefix for encoder checkpoints
            model_key = f"lstm.{name}"

        if model_key in model_state:
            if param.shape == model_state[model_key].shape:
                matched.append((name, model_key))
            else:
                mismatched.append((name, param.shape, model_state[model_key].shape))
        else:
            # Tensor in checkpoint but not in model (OK - model might be subset)
            pass

    # Check for missing tensors (in model but not in checkpoint)
    for name in model_state:
        # Check if this tensor was matched
        if not any(model_key == name for _, model_key in matched):
            missing.append(name)

    # Calculate stats
    n_matched = len(matched)
    n_missing = len(missing)
    n_mismatched = len(mismatched)
    total_model_tensors = len(model_state)
    match_ratio = n_matched / total_model_tensors if total_model_tensors > 0 else 0.0

    # Report
    logger.info("=" * 80)
    logger.info("PRETRAINED LOAD REPORT")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Model tensors: {total_model_tensors}")
    logger.info(f"Matched: {n_matched} tensors ({match_ratio:.1%})")
    logger.info(f"Missing: {n_missing} tensors (will be trained from scratch)")
    logger.info(f"Shape mismatches: {n_mismatched}")

    if matched:
        logger.info(f"Matched tensors (first 5): {[m[0] for m in matched[:5]]}")

    if missing and len(missing) <= 10:
        logger.info(f"Missing tensors: {missing}")
    elif missing:
        logger.info(f"Missing tensors (first 10): {missing[:10]}")

    if mismatched:
        logger.warning("Shape mismatches detected:")
        for name, ckpt_shape, model_shape in mismatched[:5]:
            logger.warning(f"  {name}: checkpoint {ckpt_shape} vs model {model_shape}")

    # STRICT VALIDATION
    if match_ratio < min_match_ratio:
        raise AssertionError(
            f"Pretrained load FAILED: match ratio {match_ratio:.1%} < {min_match_ratio:.1%}\n"
            f"Matched: {n_matched}/{total_model_tensors} tensors\n"
            f"This model may be incompatible with the encoder.\n"
            f"Mismatches:\n" + "\n".join(f"  {n}: {cs} vs {ms}" for n, cs, ms in mismatched[:10])
        )

    if n_mismatched > 0 and not allow_shape_mismatch:
        raise AssertionError(
            f"Pretrained load FAILED: {n_mismatched} shape mismatches detected\n"
            f"Shape mismatches:\n" + "\n".join(f"  {n}: {cs} vs {ms}" for n, cs, ms in mismatched)
        )

    # Load matched tensors
    # Create a new state dict with model keys
    load_dict = {}
    for ckpt_name, model_key in matched:
        load_dict[model_key] = state_dict[ckpt_name]

    model.load_state_dict(load_dict, strict=False)
    logger.info(f"✓ Loaded {n_matched} tensors into model")

    # Freeze encoder if requested
    n_frozen = 0
    if freeze_encoder:
        for name, param in model.named_parameters():
            # Freeze encoder layers (common patterns)
            if any(keyword in name.lower() for keyword in ["encoder", "ohlc", "lstm", "rnn"]):
                param.requires_grad = False
                n_frozen += 1

        logger.info(f"✓ Froze {n_frozen} encoder parameters")

    logger.info("=" * 80)

    return {
        "matched": [m[0] for m in matched],
        "missing": missing,
        "mismatched": mismatched,
        "n_matched": n_matched,
        "n_missing": n_missing,
        "n_mismatched": n_mismatched,
        "match_ratio": match_ratio,
        "n_frozen": n_frozen,
    }
