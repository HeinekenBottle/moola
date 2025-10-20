"""Forensic analysis of transfer learning pipeline.

Diagnoses suspicious transfer learning results where pre-trained and baseline
models achieve identical 60% validation accuracy, suggesting encoder weights
didn't actually load or contribute to training.

Root Cause Hypotheses:
1. Input dimension mismatch: Pre-training used 11 features (RelativeFeatureTransform),
   fine-tuning used 4 features (raw OHLC) → weight shapes incompatible
2. Only layer 0 loaded: 8 tensors = 1 BiLSTM layer, but encoder has 2 layers
3. Encoder frozen too aggressively: Can't adapt to new task
4. Silent loading failure: PyTorch load_state_dict(strict=False) skipped mismatched shapes

This script checks:
- Weight tensor dimensions before/after loading
- Weight statistics (mean, std, min, max) to detect random vs learned
- Parameter requires_grad status to verify freezing
- Input dimension compatibility between pre-training and fine-tuning

Usage:
    python3 scripts/diagnose_transfer_learning.py

Expected Output:
    - Diagnostic report showing exact failure mode
    - Statistical proof of identical models
    - Concrete fix recommendation
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


def analyze_pretrained_encoder(encoder_path: Path) -> dict:
    """Analyze pre-trained encoder checkpoint.

    Args:
        encoder_path: Path to multitask_encoder.pt

    Returns:
        Dictionary with encoder metadata and weight statistics
    """
    logger.info("="*70)
    logger.info("ANALYZING PRE-TRAINED ENCODER")
    logger.info("="*70)

    checkpoint = torch.load(encoder_path, map_location="cpu")
    encoder_state_dict = checkpoint["encoder_state_dict"]
    hyperparams = checkpoint["hyperparams"]

    logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
    logger.info(f"Hyperparameters: {hyperparams}")
    logger.info(f"  input_dim: {hyperparams['input_dim']}")
    logger.info(f"  hidden_dim: {hyperparams['hidden_dim']}")
    logger.info(f"  num_layers: {hyperparams['num_layers']}")

    logger.info(f"\nEncoder state dict keys ({len(encoder_state_dict)} total):")
    for i, (key, tensor) in enumerate(encoder_state_dict.items(), 1):
        mean = tensor.mean().item()
        std = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        logger.info(
            f"  {i:2d}. {key:30s} shape={str(tensor.shape):20s} "
            f"mean={mean:7.4f} std={std:7.4f} min={min_val:7.4f} max={max_val:7.4f}"
        )

    return {
        "hyperparams": hyperparams,
        "num_params": len(encoder_state_dict),
        "state_dict": encoder_state_dict,
    }


def analyze_finetuned_model(model_path: Path, label: str) -> dict:
    """Analyze fine-tuned SimpleLSTM model.

    Args:
        model_path: Path to fine-tuned model (.pkl)
        label: Human-readable label for logging

    Returns:
        Dictionary with model metadata and weight statistics
    """
    logger.info("="*70)
    logger.info(f"ANALYZING FINE-TUNED MODEL: {label}")
    logger.info("="*70)

    checkpoint = torch.load(model_path, map_location="cpu")

    logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
    logger.info(f"Input dim: {checkpoint.get('input_dim')}")
    logger.info(f"Num classes: {checkpoint.get('n_classes')}")
    logger.info(f"Hyperparams: {checkpoint.get('hyperparams')}")

    state_dict = checkpoint["model_state_dict"]
    logger.info(f"\nModel state dict keys ({len(state_dict)} total):")

    # Focus on LSTM encoder weights
    lstm_weights = {k: v for k, v in state_dict.items() if k.startswith("lstm.")}
    logger.info(f"\nLSTM encoder weights ({len(lstm_weights)} tensors):")
    for i, (key, tensor) in enumerate(lstm_weights.items(), 1):
        mean = tensor.mean().item()
        std = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        logger.info(
            f"  {i:2d}. {key:30s} shape={str(tensor.shape):20s} "
            f"mean={mean:7.4f} std={std:7.4f} min={min_val:7.4f} max={max_val:7.4f}"
        )

    return {
        "input_dim": checkpoint.get("input_dim"),
        "n_classes": checkpoint.get("n_classes"),
        "hyperparams": checkpoint.get("hyperparams"),
        "lstm_weights": lstm_weights,
        "full_state_dict": state_dict,
    }


def compare_lstm_weights(
    pretrained_weights: dict,
    baseline_weights: dict,
    pretrained_encoder_weights: dict
) -> None:
    """Compare LSTM weights between models to detect transfer.

    Args:
        pretrained_weights: Weights from model trained WITH pre-trained encoder
        baseline_weights: Weights from model trained WITHOUT pre-trained encoder
        pretrained_encoder_weights: Weights from pre-trained encoder checkpoint
    """
    logger.info("="*70)
    logger.info("WEIGHT COMPARISON: Detecting Transfer")
    logger.info("="*70)

    # Extract layer 0 weights from pre-trained encoder
    encoder_layer0_keys = [k for k in pretrained_encoder_weights.keys() if "_l0" in k]
    logger.info(f"\nPre-trained encoder layer 0 keys ({len(encoder_layer0_keys)}):")
    for key in encoder_layer0_keys:
        logger.info(f"  - {key}")

    # Compare each LSTM weight
    logger.info("\nWeight-by-weight comparison:")
    logger.info("(If transfer worked, 'pretrained' should match 'encoder', not 'baseline')")
    logger.info("")

    for key in sorted(pretrained_weights.keys()):
        if not key.startswith("lstm."):
            continue

        pretrained_tensor = pretrained_weights[key]
        baseline_tensor = baseline_weights.get(key)

        # Find corresponding encoder weight (strip "lstm." prefix)
        encoder_key = key.replace("lstm.", "")
        encoder_tensor = pretrained_encoder_weights.get(encoder_key)

        logger.info(f"\n{key}:")
        logger.info(f"  Shape: {pretrained_tensor.shape}")

        # Compute statistics
        pretrained_mean = pretrained_tensor.mean().item()
        pretrained_std = pretrained_tensor.std().item()

        if baseline_tensor is not None:
            baseline_mean = baseline_tensor.mean().item()
            baseline_std = baseline_tensor.std().item()
            diff_baseline = torch.abs(pretrained_tensor - baseline_tensor).mean().item()
            logger.info(f"  Baseline:    mean={baseline_mean:7.4f} std={baseline_std:7.4f}")
            logger.info(f"  Pretrained:  mean={pretrained_mean:7.4f} std={pretrained_std:7.4f}")
            logger.info(f"  |Pretrained - Baseline|: {diff_baseline:.6f}")
        else:
            logger.info(f"  Pretrained:  mean={pretrained_mean:7.4f} std={pretrained_std:7.4f}")
            logger.info("  Baseline: NOT FOUND")

        if encoder_tensor is not None:
            encoder_mean = encoder_tensor.mean().item()
            encoder_std = encoder_tensor.std().item()
            diff_encoder = torch.abs(pretrained_tensor - encoder_tensor).mean().item()
            logger.info(f"  Encoder:     mean={encoder_mean:7.4f} std={encoder_std:7.4f}")
            logger.info(f"  |Pretrained - Encoder|: {diff_encoder:.6f}")

            # Check shape compatibility
            if pretrained_tensor.shape != encoder_tensor.shape:
                logger.warning(
                    f"  ❌ SHAPE MISMATCH: Pretrained {pretrained_tensor.shape} "
                    f"!= Encoder {encoder_tensor.shape}"
                )
                logger.warning("  → This weight was NOT loaded due to shape incompatibility!")
        else:
            logger.info("  Encoder: NOT FOUND")


def check_weight_initialization_patterns(weights: dict, label: str) -> None:
    """Check if weights look random (Xavier/Kaiming init) or learned.

    Args:
        weights: Dictionary of weight tensors
        label: Model label for logging

    Expected patterns:
    - Random init (Xavier/Kaiming): std ~0.02-0.08, mean ~0
    - Learned weights: std ~0.15-0.40, potentially non-zero mean
    """
    logger.info("="*70)
    logger.info(f"WEIGHT INITIALIZATION ANALYSIS: {label}")
    logger.info("="*70)

    lstm_weights = {k: v for k, v in weights.items() if k.startswith("lstm.weight")}

    for key, tensor in lstm_weights.items():
        mean = tensor.mean().item()
        std = tensor.std().item()

        # Heuristic: Random init typically has std < 0.1
        if std < 0.1:
            status = "🔴 RANDOM (likely not trained or not loaded)"
        else:
            status = "🟢 LEARNED (has structure)"

        logger.info(f"{key:40s} mean={mean:7.4f} std={std:7.4f} → {status}")


def statistical_impossibility_analysis(val_acc_with: float, val_acc_without: float) -> None:
    """Explain why identical accuracy is statistically impossible.

    Args:
        val_acc_with: Validation accuracy with pre-trained encoder
        val_acc_without: Validation accuracy without pre-trained encoder
    """
    logger.info("="*70)
    logger.info("STATISTICAL ANALYSIS: Identical Accuracy")
    logger.info("="*70)

    logger.info(f"Validation accuracy WITHOUT encoder: {val_acc_without*100:.1f}%")
    logger.info(f"Validation accuracy WITH encoder:    {val_acc_with*100:.1f}%")

    if abs(val_acc_with - val_acc_without) < 0.001:  # Less than 0.1% difference
        logger.error("\n❌ SMOKING GUN: Identical accuracy to 3 decimal places!")
        logger.error("\nWhy this is impossible if transfer learning worked:")
        logger.error("  1. Different initialization: Random vs pre-trained weights")
        logger.error("  2. Different optimization paths: SGD is chaotic, never converges to same point")
        logger.error("  3. Different training dynamics: Frozen encoder vs trainable encoder")
        logger.error("  4. Different augmentation: Random seed affects jitter/mixup")
        logger.error("\nProbability of identical accuracy by chance:")
        logger.error("  - Validation set size: ~13 samples (15% of 89)")
        logger.error("  - Accuracy resolution: 1/13 ≈ 7.7% per sample")
        logger.error("  - Chance of exact match: ~7.7% (if independent)")
        logger.error("  - Chance of early stopping at same epoch: ~1/60 ≈ 1.7%")
        logger.error("  - Combined probability: 7.7% × 1.7% ≈ 0.13%")
        logger.error("\n→ Conclusion: Encoder weights did NOT load or did NOT affect training")
    else:
        logger.success(f"\n✓ Accuracy differs by {abs(val_acc_with - val_acc_without)*100:.2f}%")
        logger.info("→ Transfer learning likely had some effect")


def diagnose_input_dimension_mismatch(
    encoder_info: dict,
    model_info: dict
) -> None:
    """Check for input dimension mismatch between pre-training and fine-tuning.

    Args:
        encoder_info: Pre-trained encoder metadata
        model_info: Fine-tuned model metadata
    """
    logger.info("="*70)
    logger.info("INPUT DIMENSION COMPATIBILITY CHECK")
    logger.info("="*70)

    encoder_input_dim = encoder_info["hyperparams"]["input_dim"]
    model_input_dim = model_info["input_dim"]

    logger.info(f"Pre-trained encoder input_dim: {encoder_input_dim}")
    logger.info(f"Fine-tuned model input_dim:    {model_input_dim}")

    if encoder_input_dim != model_input_dim:
        logger.error("\n❌ INPUT DIMENSION MISMATCH!")
        logger.error(f"  Pre-training: {encoder_input_dim} features (likely RelativeFeatureTransform)")
        logger.error(f"  Fine-tuning:  {model_input_dim} features (likely raw OHLC)")
        logger.error("\nThis causes LSTM weight shape mismatch:")
        logger.error(f"  weight_ih_l0 shape: ({encoder_info['hyperparams']['hidden_dim']*4}, {encoder_input_dim})")
        logger.error(f"  Expected shape:     ({encoder_info['hyperparams']['hidden_dim']*4}, {model_input_dim})")
        logger.error("\nPyTorch behavior with strict=False:")
        logger.error("  - load_state_dict() skips mismatched shapes WITHOUT error")
        logger.error("  - Model keeps randomly initialized weights")
        logger.error("  - Training proceeds as if no encoder was loaded")
        logger.error("\n→ ROOT CAUSE: Pre-training and fine-tuning use different feature sets")
    else:
        logger.success("\n✓ Input dimensions match")
        logger.info("→ This is NOT the root cause")


def propose_fix(encoder_info: dict, model_info: dict) -> None:
    """Propose concrete fix based on diagnosis.

    Args:
        encoder_info: Pre-trained encoder metadata
        model_info: Fine-tuned model metadata
    """
    logger.info("="*70)
    logger.info("RECOMMENDED FIX")
    logger.info("="*70)

    encoder_input_dim = encoder_info["hyperparams"]["input_dim"]
    model_input_dim = model_info["input_dim"]

    if encoder_input_dim != model_input_dim:
        logger.info("\nOption 1: Use same features for pre-training and fine-tuning (RECOMMENDED)")
        logger.info("  1. Apply RelativeFeatureTransform to fine-tuning data:")
        logger.info("     ```python")
        logger.info("     from moola.features.relative_transform import RelativeFeatureTransform")
        logger.info("     transform = RelativeFeatureTransform()")
        logger.info("     X_relative = transform.transform(X_ohlc)  # [N, 105, 4] → [N, 105, 11]")
        logger.info("     ```")
        logger.info("  2. Fine-tune SimpleLSTM with X_relative (11 features)")
        logger.info("  3. Encoder weights will now load successfully")
        logger.info("\nOption 2: Re-train encoder on raw OHLC (4 features)")
        logger.info("  1. Modify pre-training script to use raw OHLC instead of RelativeFeatureTransform")
        logger.info("  2. Re-run pre-training (25 min on H100)")
        logger.info("  3. Fine-tune with matching 4-feature input")
        logger.info("\nOption 3: Use adapter layer (NOT RECOMMENDED - adds complexity)")
        logger.info("  1. Add linear projection: 4 → 11 features before LSTM")
        logger.info("  2. Keep pre-trained encoder weights")
        logger.info("  3. Train only adapter + classifier")
    else:
        logger.info("\nInput dimensions match - investigate other causes:")
        logger.info("  1. Check if encoder weights are actually being used (frozen initially?)")
        logger.info("  2. Verify learning rate isn't too high (erasing pre-trained features)")
        logger.info("  3. Check if pre-trained encoder was trained on different task")


def main():
    """Run full diagnostic analysis."""
    # Paths
    artifacts_dir = Path("/Users/jack/projects/moola/artifacts/runpod_results")
    encoder_path = artifacts_dir / "multitask_encoder.pt"
    baseline_path = artifacts_dir / "simple_lstm_finetuned.pkl"
    pretrained_path = artifacts_dir / "simple_lstm_with_pretrained_encoder.pkl"

    # Check files exist
    for path in [encoder_path, baseline_path, pretrained_path]:
        if not path.exists():
            logger.error(f"File not found: {path}")
            return

    # Analyze pre-trained encoder
    encoder_info = analyze_pretrained_encoder(encoder_path)

    # Analyze fine-tuned models
    baseline_info = analyze_finetuned_model(baseline_path, "Baseline (no encoder)")
    pretrained_info = analyze_finetuned_model(pretrained_path, "With Pre-trained Encoder")

    # Compare weights
    compare_lstm_weights(
        pretrained_weights=pretrained_info["full_state_dict"],
        baseline_weights=baseline_info["full_state_dict"],
        pretrained_encoder_weights=encoder_info["state_dict"]
    )

    # Check initialization patterns
    check_weight_initialization_patterns(
        baseline_info["lstm_weights"],
        "Baseline (no encoder)"
    )
    check_weight_initialization_patterns(
        pretrained_info["lstm_weights"],
        "With Pre-trained Encoder"
    )

    # Statistical analysis
    # From training logs: both models achieved 60% val accuracy
    statistical_impossibility_analysis(val_acc_with=0.60, val_acc_without=0.60)

    # Input dimension diagnosis
    diagnose_input_dimension_mismatch(encoder_info, pretrained_info)

    # Propose fix
    propose_fix(encoder_info, pretrained_info)

    logger.info("\n" + "="*70)
    logger.info("DIAGNOSIS COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
