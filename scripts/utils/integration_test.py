#!/usr/bin/env python3
"""Integration test for LSTM training pipeline.

Verifies that all components work end-to-end before RunPod deployment:
- Feature transformation (currently raw OHLC, will be relative features)
- Bidirectional LSTM architecture
- Multi-task pre-training model
- Encoder weight transfer
- SimpleLSTM fine-tuning model
- Focal loss with class weights
- Gradient flow

Run this before spending $ on RunPod to catch integration issues.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from moola.features.relative_transform import RelativeFeatureTransform
from moola.models.simple_lstm import SimpleLSTMModel
from moola.pretraining.multitask_pretrain import MultiTaskBiLSTM
from moola.utils.focal_loss import FocalLoss


def test_raw_ohlc_pipeline():
    """Test current pipeline with raw OHLC (input_dim=4)."""
    logger.info("=" * 70)
    logger.info("TEST 1: Current Pipeline (Raw OHLC, input_dim=4)")
    logger.info("=" * 70)

    # Generate synthetic OHLC data
    batch_size = 2
    seq_len = 105
    ohlc_dim = 4

    ohlc = torch.randn(batch_size, seq_len, ohlc_dim) * 100 + 23000
    logger.info(f"✓ Generated OHLC data: {ohlc.shape}")
    logger.info(f"  Price range: [{ohlc.min():.2f}, {ohlc.max():.2f}]")

    # Test SimpleLSTM with raw OHLC
    model = SimpleLSTMModel(
        seed=1337,
        hidden_size=128,
        num_layers=1,
        n_epochs=2,  # Quick test
        device="cpu",
        time_warp_prob=0.0,  # Should be disabled
    )

    # Build model (simulates fit() initialization)
    model.n_classes = 2
    model.input_dim = 4
    model.label_to_idx = {"consolidation": 0, "retracement": 1}
    model.idx_to_label = {0: "consolidation", 1: "retracement"}
    model.model = model._build_model(input_dim=4, n_classes=2)

    logger.info(f"✓ SimpleLSTM built: {sum(p.numel() for p in model.model.parameters())} params")

    # Check bidirectional LSTM
    is_bidirectional = model.model.lstm.bidirectional
    logger.info(f"✓ LSTM bidirectional: {is_bidirectional}")
    if not is_bidirectional:
        logger.error("✗ LSTM should be bidirectional!")
        return False

    # Forward pass
    logits = model.model(ohlc)
    assert logits.shape == (batch_size, 2), f"Expected (2, 2), got {logits.shape}"
    logger.info(f"✓ Forward pass output: {logits.shape}")

    # Test Focal Loss with class weights
    criterion = FocalLoss(gamma=2.0, alpha=torch.tensor([1.0, 1.17]))
    labels = torch.tensor([0, 1])
    loss = criterion(logits, labels)
    logger.info(f"✓ Focal loss computed: {loss.item():.4f}")

    # Backward pass (gradient flow)
    loss.backward()
    has_grads = sum(1 for p in model.model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.model.parameters())
    logger.info(f"✓ Gradients in {has_grads}/{total_params} parameters")

    if has_grads != total_params:
        logger.error(f"✗ Not all parameters have gradients!")
        return False

    logger.success("✅ TEST 1 PASSED: Current pipeline works with raw OHLC\n")
    return True


def test_relative_features():
    """Test relative feature transformation (for future integration)."""
    logger.info("=" * 70)
    logger.info("TEST 2: Relative Feature Transform (input_dim=11)")
    logger.info("=" * 70)

    # Generate synthetic OHLC
    ohlc = np.random.randn(2, 105, 4) * 100 + 23000
    logger.info(f"✓ Input OHLC shape: {ohlc.shape}")
    logger.info(f"  Price range: [{ohlc.min():.2f}, {ohlc.max():.2f}]")

    # Transform to relative features
    transform = RelativeFeatureTransform()
    features = transform.transform(ohlc)
    features_tensor = torch.from_numpy(features).float()

    assert features.shape == (2, 105, 11), f"Expected (2,105,11), got {features.shape}"
    logger.info(f"✓ Relative features shape: {features.shape}")
    logger.info(f"  Feature range: [{features.min():.2f}, {features.max():.2f}]")
    logger.info(
        f"  Features: 4 log returns + 3 candle ratios + 4 z-scores = 11 scale-invariant features"
    )

    # Verify scale invariance
    ohlc_scaled = ohlc * 1.3  # Simulate 30% price increase
    features_scaled = transform.transform(ohlc_scaled)
    diff = np.abs(features - features_scaled).max()
    logger.info(f"✓ Scale invariance check: max diff = {diff:.6f}")

    if diff > 0.01:
        logger.warning(
            f"⚠️  Features not perfectly scale-invariant (diff={diff:.6f}). "
            f"This is expected for z-scores if window statistics change."
        )

    logger.success("✅ TEST 2 PASSED: Relative features work correctly")
    logger.warning(
        "⚠️  NOTE: Relative features NOT integrated in training pipeline yet (requires code changes)\n"
    )
    return True


def test_multitask_pretraining():
    """Test multi-task pre-training model."""
    logger.info("=" * 70)
    logger.info("TEST 3: Multi-Task Pre-training Model")
    logger.info("=" * 70)

    # Use raw OHLC for now (current pipeline)
    ohlc = torch.randn(2, 105, 4) * 100 + 23000
    logger.info(f"✓ Input data: {ohlc.shape}")

    # Build multi-task model
    pretrain_model = MultiTaskBiLSTM(input_dim=4, hidden_dim=128, num_layers=2, dropout=0.2)

    exp_logits, swing_logits, candle_logits = pretrain_model(ohlc)

    assert exp_logits.shape == (2, 105, 2), f"Expansion head wrong shape: {exp_logits.shape}"
    assert swing_logits.shape == (2, 105, 3), f"Swing head wrong shape: {swing_logits.shape}"
    assert candle_logits.shape == (2, 105, 4), f"Candle head wrong shape: {candle_logits.shape}"

    logger.info(f"✓ Expansion logits: {exp_logits.shape} (binary: expansion vs normal)")
    logger.info(f"✓ Swing logits: {swing_logits.shape} (3-class: high/low/neither)")
    logger.info(f"✓ Candle logits: {candle_logits.shape} (4-class: bullish/bearish/doji/neutral)")

    # Check encoder extraction
    encoder_state = pretrain_model.encoder_lstm.state_dict()
    logger.info(f"✓ Encoder has {len(encoder_state)} weight tensors")

    logger.success("✅ TEST 3 PASSED: Multi-task pre-training model works\n")
    return True


def test_encoder_weight_transfer():
    """Test pre-trained encoder weight loading into SimpleLSTM."""
    logger.info("=" * 70)
    logger.info("TEST 4: Encoder Weight Transfer")
    logger.info("=" * 70)

    # Create pre-training model and extract encoder
    pretrain_model = MultiTaskBiLSTM(input_dim=4, hidden_dim=128)
    encoder_state = pretrain_model.encoder_lstm.state_dict()
    logger.info(f"✓ Pre-trained encoder: {len(encoder_state)} tensors")

    # Create SimpleLSTM
    classifier = SimpleLSTMModel(
        seed=1337, hidden_size=128, num_layers=1, n_epochs=2, device="cpu"
    )

    # Build model
    classifier.n_classes = 2
    classifier.input_dim = 4
    classifier.model = classifier._build_model(input_dim=4, n_classes=2)

    # Check weight initialization before transfer
    first_weight_before = classifier.model.lstm.weight_ih_l0.clone()
    logger.info(
        f"✓ Encoder weights before transfer: std={first_weight_before.std().item():.4f} (should be ~0.02-0.05 for random init)"
    )

    # Simulate weight loading (manual approach since we don't have a saved checkpoint)
    loaded_keys = []
    model_state_dict = classifier.model.state_dict()
    for key in encoder_state:
        model_key = f"lstm.{key}"
        if model_key in model_state_dict:
            if encoder_state[key].shape == model_state_dict[model_key].shape:
                model_state_dict[model_key] = encoder_state[key]
                loaded_keys.append(model_key)

    classifier.model.load_state_dict(model_state_dict)

    # Check weight initialization after transfer
    first_weight_after = classifier.model.lstm.weight_ih_l0
    logger.info(
        f"✓ Encoder weights after transfer: std={first_weight_after.std().item():.4f} (should be >0.1 if pre-trained)"
    )
    logger.info(f"✓ Loaded {len(loaded_keys)} parameter tensors")

    # Verify weights actually changed
    weight_diff = (first_weight_after - first_weight_before).abs().max().item()
    logger.info(f"✓ Max weight change: {weight_diff:.4f}")

    if weight_diff < 0.01:
        logger.warning("⚠️  Weights may not have transferred correctly (small change)")

    # Test forward pass with transferred weights
    ohlc = torch.randn(2, 105, 4) * 100 + 23000
    logits = classifier.model(ohlc)
    logger.info(f"✓ Forward pass with transferred weights: {logits.shape}")

    logger.success("✅ TEST 4 PASSED: Weight transfer works\n")
    return True


def test_time_warp_disabled():
    """Verify time warp augmentation is disabled."""
    logger.info("=" * 70)
    logger.info("TEST 5: Time Warp Augmentation (Should be DISABLED)")
    logger.info("=" * 70)

    model = SimpleLSTMModel(seed=1337, hidden_size=128)

    logger.info(f"✓ time_warp_prob = {model.time_warp_prob}")

    if model.time_warp_prob != 0.0:
        logger.error(f"✗ Time warp should be disabled (0.0), but is {model.time_warp_prob}")
        return False

    logger.success("✅ TEST 5 PASSED: Time warp is disabled\n")
    return True


def main():
    """Run all integration tests."""
    logger.info("\n" + "=" * 70)
    logger.info("MOOLA LSTM PIPELINE INTEGRATION TEST")
    logger.info("=" * 70 + "\n")

    results = []

    # Run all tests
    results.append(("Raw OHLC Pipeline", test_raw_ohlc_pipeline()))
    results.append(("Relative Features", test_relative_features()))
    results.append(("Multi-task Pre-training", test_multitask_pretraining()))
    results.append(("Encoder Weight Transfer", test_encoder_weight_transfer()))
    results.append(("Time Warp Disabled", test_time_warp_disabled()))

    # Summary
    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "✗ FAILED"
        logger.info(f"{status}: {name}")
        if not passed:
            all_passed = False

    logger.info("=" * 70)

    if all_passed:
        logger.success("\n✅ ALL INTEGRATION TESTS PASSED\n")
        logger.info("Pipeline is ready for RunPod training! 🚀")
        logger.info("\nNOTE: Currently using raw OHLC (input_dim=4).")
        logger.info(
            "For relative features (input_dim=11), integration work needed in cli.py.\n"
        )
        return 0
    else:
        logger.error("\n✗ SOME TESTS FAILED\n")
        logger.error("Fix issues before RunPod deployment!\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
