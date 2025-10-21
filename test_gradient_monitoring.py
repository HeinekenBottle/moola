"""Test script for Phase 4 gradient monitoring implementation.

This script demonstrates how gradient monitoring and task collapse detection work
during training of the EnhancedSimpleLSTM dual-task model.
"""

import numpy as np
import torch
from src.moola.models.enhanced_simple_lstm import EnhancedSimpleLSTMModel


def test_gradient_monitoring():
    """Test gradient monitoring with synthetic data."""
    print("=" * 80)
    print("Phase 4 Gradient Monitoring Test")
    print("=" * 80)

    # Create synthetic multi-task data
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 100
    seq_len = 105
    n_features = 4

    # Create dummy OHLC data
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)

    # Create binary labels (consolidation vs retracement)
    y = np.random.choice(["consolidation", "retracement"], size=n_samples)

    # Create pointer labels (expansion start/end indices)
    expansion_start = np.random.randint(20, 40, size=n_samples).astype(np.float32)
    expansion_end = np.random.randint(60, 80, size=n_samples).astype(np.float32)

    print(f"\nDataset:")
    print(f"  Samples: {n_samples}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {np.unique(y)}")
    print(f"  Pointer range: start=[{expansion_start.min():.0f}, {expansion_start.max():.0f}], "
          f"end=[{expansion_end.min():.0f}, {expansion_end.max():.0f}]")

    # Initialize model with gradient monitoring enabled
    print("\n" + "=" * 80)
    print("Initializing EnhancedSimpleLSTM with gradient monitoring...")
    print("=" * 80)

    model = EnhancedSimpleLSTMModel(
        n_epochs=20,  # Reduced for quick test
        batch_size=16,
        learning_rate=1e-3,
        val_split=0.2,
        device="cpu",  # Use CPU for test
        seed=42,
        predict_pointers=True,  # Enable multi-task learning
        loss_alpha=1.0,  # Classification task weight
        loss_beta=1.0,  # Pointer regression task weight
    )

    # Train with gradient monitoring
    print("\nTraining with gradient monitoring enabled...")
    print("-" * 80)

    model.fit(
        X=X,
        y=y,
        expansion_start=expansion_start,
        expansion_end=expansion_end,
        monitor_gradients=True,
        gradient_log_freq=5,  # Log every 5 epochs
    )

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    # Test predictions
    print("\nTesting predictions...")
    predictions = model.predict_with_pointers(X[:5], expansion_start[:5], expansion_end[:5])

    print("\nSample predictions:")
    for i in range(5):
        pred_class = predictions["type_predictions"][i]
        pred_start = predictions["pointer_predictions"]["start"][i]
        pred_end = predictions["pointer_predictions"]["end"][i]
        true_start = expansion_start[i]
        true_end = expansion_end[i]

        print(f"  Sample {i}:")
        print(f"    True label: {y[i]}, Predicted: {pred_class}")
        print(f"    True pointer: [{true_start:.1f}, {true_end:.1f}]")
        print(f"    Pred pointer: [{pred_start:.1f}, {pred_end:.1f}]")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_gradient_monitoring()
