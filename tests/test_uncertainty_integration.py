"""Integration test for uncertainty-weighted loss with EnhancedSimpleLSTM."""

import numpy as np
import pytest
import torch

from moola.models.enhanced_simple_lstm import EnhancedSimpleLSTMModel


def test_uncertainty_weighted_loss_integration():
    """Test that uncertainty-weighted loss works with EnhancedSimpleLSTM."""
    # Create synthetic data
    np.random.seed(42)
    torch.manual_seed(42)

    # Small batch for testing
    batch_size = 8
    seq_len = 105
    n_features = 11  # RelativeTransform features

    X = torch.randn(batch_size, seq_len, n_features)
    y = torch.randint(0, 2, (batch_size,)).float()
    ptr_start = torch.randint(0, 50, (batch_size,)).float()
    ptr_end = torch.randint(50, 104, (batch_size,)).float()

    # Create model with uncertainty weighting enabled (default)
    model = EnhancedSimpleLSTMModel(
        predict_pointers=True,
        use_uncertainty_weighting=True,  # This is the default now
        device="cpu",
        n_epochs=2,  # Very short training for testing
        batch_size=batch_size,
        learning_rate=1e-3,
    )

    # Test that model initializes correctly
    assert hasattr(model, "use_uncertainty_weighting")
    assert model.use_uncertainty_weighting is True

    # Test forward pass
    model.fit(X, y, expansion_start=ptr_start, expansion_end=ptr_end)

    # Test that training completed without errors
    assert hasattr(model, "model")
    assert model.model is not None

    # Test prediction
    predictions = model.predict(X)
    assert predictions.shape == (batch_size,)

    # Test pointer prediction
    ptr_predictions = model.predict_with_pointers(X)
    assert "pointers" in ptr_predictions
    assert ptr_predictions["pointers"].shape == (batch_size, 2)

    print("✅ Uncertainty-weighted loss integration test passed!")


if __name__ == "__main__":
    test_uncertainty_weighted_loss_integration()
