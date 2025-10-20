"""Quick verification that encoder was frozen during fine-tuning.

This script proves the root cause by comparing encoder weights before and
after fine-tuning. If weights are identical, encoder was frozen.
"""

from pathlib import Path

import torch
from loguru import logger


def main():
    """Compare encoder weights before and after fine-tuning."""
    # Paths
    artifacts_dir = Path("/Users/jack/projects/moola/artifacts/runpod_results")
    encoder_path = artifacts_dir / "multitask_encoder.pt"
    pretrained_path = artifacts_dir / "simple_lstm_with_pretrained_encoder.pkl"

    # Load checkpoints
    encoder_checkpoint = torch.load(encoder_path, map_location="cpu")
    model_checkpoint = torch.load(pretrained_path, map_location="cpu")

    # Extract LSTM weights
    encoder_weights = encoder_checkpoint["encoder_state_dict"]
    model_weights = model_checkpoint["model_state_dict"]

    logger.info("="*70)
    logger.info("FROZEN ENCODER VERIFICATION")
    logger.info("="*70)

    # Check each weight tensor
    all_frozen = True
    all_equal = True

    for encoder_key in encoder_weights.keys():
        if "_l1" in encoder_key:
            continue  # Skip layer 1 (not loaded into SimpleLSTM)

        model_key = f"lstm.{encoder_key}"

        if model_key not in model_weights:
            logger.warning(f"Key {model_key} not found in model")
            continue

        encoder_tensor = encoder_weights[encoder_key]
        model_tensor = model_weights[model_key]

        # Check if tensors are exactly equal
        is_equal = torch.equal(encoder_tensor, model_tensor)
        max_diff = torch.abs(encoder_tensor - model_tensor).max().item()

        if is_equal:
            status = "🔴 FROZEN (no change)"
        elif max_diff < 1e-6:
            status = "🟡 MINIMAL (< 1e-6)"
            all_frozen = False
        elif max_diff < 1e-3:
            status = "🟢 UPDATED (< 1e-3)"
            all_frozen = False
            all_equal = False
        else:
            status = "🟢 TRAINED (> 1e-3)"
            all_frozen = False
            all_equal = False

        logger.info(f"{encoder_key:30s} max_diff={max_diff:.10f} {status}")

    logger.info("="*70)

    if all_equal:
        logger.error("❌ CRITICAL: ALL weights are identical!")
        logger.error("Encoder was FROZEN during entire fine-tuning process.")
        logger.error("This explains why pretrained and baseline models have identical accuracy.")
        logger.error("")
        logger.error("Root cause: freeze_encoder=True with unfreeze_encoder_after=0")
        logger.error("Fix: Set unfreeze_encoder_after=10 for two-phase training")
    elif all_frozen:
        logger.warning("⚠️ WARNING: Weights changed but by negligible amounts (<1e-6)")
        logger.warning("Encoder likely frozen or learning rate too low.")
    else:
        logger.success("✅ SUCCESS: Encoder weights updated during fine-tuning!")
        logger.success("Transfer learning is working as expected.")

    logger.info("="*70)


if __name__ == "__main__":
    main()
