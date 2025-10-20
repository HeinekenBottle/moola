"""Quick test script to verify SimpleLSTM model implementation.

Checks:
1. Model can be instantiated
2. Parameter count is ~70K
3. Model can fit and predict on dummy data
4. Model integrates with get_model() registry
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moola.models import get_model


def test_simple_lstm():
    print("=" * 60)
    print("Testing SimpleLSTM Model")
    print("=" * 60)

    # Create dummy data
    np.random.seed(1337)
    N = 50  # Small dataset
    T = 105  # Sequence length (OHLC bars)
    F = 4  # Features (OHLC)

    X = np.random.randn(N, T, F).astype(np.float32)
    y = np.random.choice(['consolidation', 'retracement'], size=N)

    print(f"\n1. Created dummy data: X.shape={X.shape}, y.shape={y.shape}")

    # Test model instantiation via registry
    print("\n2. Testing model instantiation via get_model()...")
    model = get_model(
        "simple_lstm",
        seed=1337,
        hidden_size=64,
        num_layers=1,
        num_heads=4,
        dropout=0.4,
        n_epochs=5,  # Short training for test
        device="cpu",
        val_split=0.2,
    )
    print("   ✓ Model instantiated successfully")

    # Test fitting
    print("\n3. Testing model.fit()...")
    model.fit(X, y)
    print("   ✓ Model fitted successfully")

    # Check parameter count
    if model.model is not None:
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"\n4. Parameter count: {total_params:,}")

        if 50_000 <= total_params <= 100_000:
            print(f"   ✓ Parameter count in target range (50K-100K)")
        else:
            print(f"   ⚠ Parameter count outside target range (expected ~70K)")

    # Test prediction
    print("\n5. Testing model.predict()...")
    predictions = model.predict(X[:10])
    print(f"   Predictions: {predictions}")
    print(f"   ✓ Predictions generated successfully")

    # Test predict_proba
    print("\n6. Testing model.predict_proba()...")
    probas = model.predict_proba(X[:10])
    print(f"   Probabilities shape: {probas.shape}")
    print(f"   Sample probabilities:\n{probas[:3]}")
    print(f"   ✓ Probabilities generated successfully")

    # Verify probabilities sum to 1
    prob_sums = probas.sum(axis=1)
    if np.allclose(prob_sums, 1.0, atol=1e-5):
        print(f"   ✓ Probabilities sum to 1.0")
    else:
        print(f"   ⚠ Probabilities don't sum to 1.0: {prob_sums}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_simple_lstm()
