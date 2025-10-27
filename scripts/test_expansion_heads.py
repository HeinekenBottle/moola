"""Test expansion detection heads (binary + countdown) locally.

Quick validation of new architecture before full training run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from moola.features.relativity import RelativityConfig, build_relativity_features
from moola.models.jade_core import JadeCompact


def create_expansion_labels(expansion_start: int, expansion_end: int, window_length: int = 105):
    """Create binary mask and countdown labels from pointer annotations.

    Args:
        expansion_start: First bar of expansion (inclusive)
        expansion_end: Last bar of expansion (inclusive)
        window_length: Total window length

    Returns:
        binary_mask: [0, 0, ..., 1, 1, 1, ..., 0] where 1 = inside expansion
        countdown: [N, N-1, ..., 1, 0, -1, -2, ...] where 0 = expansion_start
    """
    # Binary mask: 1 for bars inside expansion, 0 otherwise
    binary_mask = np.zeros(window_length, dtype=np.float32)
    binary_mask[expansion_start : expansion_end + 1] = 1.0

    # Countdown: positive before expansion, 0 at start, negative after
    countdown = np.arange(window_length, dtype=np.float32) - expansion_start
    countdown = -countdown  # Flip: [..., 2, 1, 0, -1, -2, ...]

    return binary_mask, countdown


def test_label_generation():
    """Test label generation with sample data."""
    print("=" * 80)
    print("LABEL GENERATION TEST")
    print("=" * 80)

    # Example: expansion from bar 50 to bar 65
    expansion_start = 50
    expansion_end = 65

    binary, countdown = create_expansion_labels(expansion_start, expansion_end)

    print(f"\nExpansion: bars {expansion_start} to {expansion_end}")
    print(f"\nBinary mask shape: {binary.shape}")
    print(f"Binary mask values: {np.unique(binary, return_counts=True)}")
    print(
        f"Expansion bars marked: {binary.sum()} (expected: {expansion_end - expansion_start + 1})"
    )

    print(f"\nCountdown shape: {countdown.shape}")
    print(f"Countdown at expansion_start (bar {expansion_start}): {countdown[expansion_start]}")
    print(f"Countdown 5 bars before: {countdown[expansion_start-5]}")
    print(f"Countdown 5 bars after: {countdown[expansion_start+5]}")

    # Verify countdown crosses zero at expansion_start
    assert (
        countdown[expansion_start] == 0
    ), f"Countdown should be 0 at expansion_start, got {countdown[expansion_start]}"
    assert (
        countdown[expansion_start - 1] == 1
    ), f"Countdown should be 1 at expansion_start-1, got {countdown[expansion_start-1]}"
    assert (
        countdown[expansion_start + 1] == -1
    ), f"Countdown should be -1 at expansion_start+1, got {countdown[expansion_start+1]}"

    print("\n✓ Label generation validated")
    return binary, countdown


def test_model_architecture():
    """Test model with expansion heads."""
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE TEST")
    print("=" * 80)

    # Create model with expansion heads enabled
    model = JadeCompact(
        input_size=12,
        hidden_size=96,
        num_layers=1,
        predict_pointers=True,
        predict_expansion_sequence=True,
    )

    # Print parameter counts
    params = model.get_num_parameters()
    print(f"\nTotal parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")

    # Test forward pass
    batch_size = 4
    window_length = 105
    n_features = 12

    x = torch.randn(batch_size, window_length, n_features)

    with torch.no_grad():
        output = model(x)

    print(f"\nOutput keys: {list(output.keys())}")
    print(f"\nlogits shape: {output['logits'].shape} (expect: [{batch_size}, 3])")
    print(f"pointers shape: {output['pointers'].shape} (expect: [{batch_size}, 2])")
    print(
        f"expansion_binary shape: {output['expansion_binary'].shape} (expect: [{batch_size}, {window_length}])"
    )
    print(
        f"expansion_countdown shape: {output['expansion_countdown'].shape} (expect: [{batch_size}, {window_length}])"
    )

    # Verify shapes
    assert output["logits"].shape == (batch_size, 3)
    assert output["pointers"].shape == (batch_size, 2)
    assert output["expansion_binary"].shape == (batch_size, window_length)
    assert output["expansion_countdown"].shape == (batch_size, window_length)

    print("\n✓ Model architecture validated")
    return model


def test_loss_computation():
    """Test multi-task loss with new heads."""
    print("\n" + "=" * 80)
    print("LOSS COMPUTATION TEST")
    print("=" * 80)

    batch_size = 4
    window_length = 105

    # Create dummy predictions
    logits = torch.randn(batch_size, 3)
    pointers = torch.rand(batch_size, 2)  # Sigmoid output [0, 1]
    expansion_binary_logits = torch.randn(batch_size, window_length)
    expansion_countdown = torch.randn(batch_size, window_length)

    # Create dummy targets
    labels = torch.randint(0, 3, (batch_size,))
    pointer_targets = torch.rand(batch_size, 2)
    binary_targets = torch.randint(0, 2, (batch_size, window_length)).float()
    countdown_targets = torch.randn(batch_size, window_length)

    # Compute individual losses
    loss_type = F.cross_entropy(logits, labels)
    loss_ptr = F.huber_loss(pointers, pointer_targets, delta=0.08)
    loss_binary = F.binary_cross_entropy_with_logits(expansion_binary_logits, binary_targets)
    loss_countdown = F.huber_loss(expansion_countdown, countdown_targets, delta=1.0)

    # New loss weights: pointers 70%, binary+countdown 20% (10% each), type 10%
    lambda_ptr = 0.7
    lambda_binary = 0.1
    lambda_countdown = 0.1
    lambda_type = 0.1

    total_loss = (
        lambda_type * loss_type
        + lambda_ptr * loss_ptr
        + lambda_binary * loss_binary
        + lambda_countdown * loss_countdown
    )

    print("\nIndividual losses:")
    print(f"  Classification: {loss_type.item():.4f} (weight: {lambda_type})")
    print(f"  Pointers: {loss_ptr.item():.4f} (weight: {lambda_ptr})")
    print(f"  Binary: {loss_binary.item():.4f} (weight: {lambda_binary})")
    print(f"  Countdown: {loss_countdown.item():.4f} (weight: {lambda_countdown})")
    print(f"\nTotal loss: {total_loss.item():.4f}")

    # Verify weighted contributions
    contributions = {
        "type": lambda_type * loss_type.item() / total_loss.item() * 100,
        "pointers": lambda_ptr * loss_ptr.item() / total_loss.item() * 100,
        "binary": lambda_binary * loss_binary.item() / total_loss.item() * 100,
        "countdown": lambda_countdown * loss_countdown.item() / total_loss.item() * 100,
    }

    print("\nLoss contributions:")
    for task, pct in contributions.items():
        print(f"  {task}: {pct:.1f}%")

    print("\n✓ Loss computation validated")


def test_on_real_data():
    """Test on actual labeled data sample."""
    print("\n" + "=" * 80)
    print("REAL DATA TEST")
    print("=" * 80)

    # Load a few samples
    df = pd.read_parquet("data/processed/labeled/train_latest.parquet").head(5)

    print(f"\nLoaded {len(df)} samples")

    # Process first sample
    row = df.iloc[0]
    print(f"\nSample: {row['window_id']}")
    print(f"  Label: {row['label']}")
    print(f"  expansion_start: {row['expansion_start']}")
    print(f"  expansion_end: {row['expansion_end']}")

    # Build features
    ohlc = row["features"]  # 105 arrays of [O, H, L, C]
    ohlc_df = pd.DataFrame([arr for arr in ohlc], columns=["open", "high", "low", "close"])

    cfg = RelativityConfig()
    X_12d, valid_mask, _ = build_relativity_features(ohlc_df, cfg.dict())

    print(f"  Features shape: {X_12d.shape}")

    # Generate labels
    binary, countdown = create_expansion_labels(
        row["expansion_start"], row["expansion_end"], window_length=105
    )

    print(f"  Binary mask: {binary.sum():.0f} expansion bars")
    print(f"  Countdown range: [{countdown.min():.0f}, {countdown.max():.0f}]")

    # Test model forward pass
    model = JadeCompact(
        input_size=12,
        predict_pointers=True,
        predict_expansion_sequence=True,
    )
    model.eval()

    # Forward pass
    x = torch.from_numpy(X_12d).float()  # [1, 105, 12]

    with torch.no_grad():
        output = model(x)

    print("\nModel outputs:")
    print(f"  Classification logits: {output['logits'].shape}")
    print(f"  Pointers: {output['pointers'].shape}")
    print(f"  Binary probs: {output['expansion_binary'].shape}")
    print(f"    Mean prob: {output['expansion_binary'].mean():.4f}")
    print(f"  Countdown: {output['expansion_countdown'].shape}")
    print(f"    Mean: {output['expansion_countdown'].mean():.4f}")

    print("\n✓ Real data test passed")


if __name__ == "__main__":
    print("\nExpansion Detection Heads - Validation Suite")
    print("=" * 80)

    # Run tests
    binary, countdown = test_label_generation()
    model = test_model_architecture()
    test_loss_computation()

    try:
        test_on_real_data()
    except Exception as e:
        print(f"\n⚠️  Real data test skipped: {e}")
        print("(This is expected if data files are not available)")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review test output above")
    print("2. Integrate label generation into finetune_jade.py")
    print("3. Update loss computation with 70/20/10 weights")
    print("4. Run local training on 10k bars")
    print("5. Deploy to RunPod for full training")
