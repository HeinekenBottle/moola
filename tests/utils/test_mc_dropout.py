"""Tests for Monte Carlo Dropout and Temperature Scaling."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from moola.utils.uncertainty.mc_dropout import (
    TemperatureScaling,
    apply_temperature_scaling,
    enable_dropout,
    get_uncertainty_threshold,
    mc_dropout_predict,
)


class DummyModel(nn.Module):
    """Dummy model for testing MC Dropout."""

    def __init__(self, input_dim=11, hidden_dim=64, n_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim * 2, n_classes)
        self.pointer_head = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        """Forward pass returning dict with type_logits and pointers."""
        lstm_out, _ = self.lstm(x)  # [B, T, hidden*2]
        last_hidden = lstm_out[:, -1, :]  # [B, hidden*2]
        last_hidden = self.dropout(last_hidden)

        type_logits = self.classifier(last_hidden)
        pointers = torch.sigmoid(self.pointer_head(last_hidden))

        return {"type_logits": type_logits, "pointers": pointers}


def test_enable_dropout():
    """Test that enable_dropout switches dropout layers to training mode."""
    model = DummyModel()
    model.eval()

    # Initially, dropout should be in eval mode
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            assert not module.training

    # Enable dropout
    enable_dropout(model)

    # Now dropout should be in training mode
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            assert module.training


def test_mc_dropout_predict():
    """Test MC Dropout inference on dummy model."""
    model = DummyModel()
    model.eval()

    # Create dummy input
    batch_size = 4
    seq_len = 105
    input_dim = 11
    x = torch.randn(batch_size, seq_len, input_dim)

    # Run MC Dropout
    n_passes = 10
    results = mc_dropout_predict(model, x, n_passes=n_passes, dropout_rate=0.15)

    # Check output shapes
    assert results["type_probs_mean"].shape == (batch_size, 2)
    assert results["type_probs_std"].shape == (batch_size, 2)
    assert results["type_entropy"].shape == (batch_size,)
    assert results["pointer_mean"].shape == (batch_size, 2)
    assert results["pointer_std"].shape == (batch_size, 2)
    assert results["all_type_probs"].shape == (n_passes, batch_size, 2)
    assert results["all_pointers"].shape == (n_passes, batch_size, 2)

    # Check that probabilities sum to 1
    assert np.allclose(results["type_probs_mean"].sum(axis=1), 1.0, atol=1e-5)

    # Check that entropy is non-negative
    assert (results["type_entropy"] >= 0).all()

    # Check that std dev is non-negative
    assert (results["type_probs_std"] >= 0).all()
    assert (results["pointer_std"] >= 0).all()


def test_get_uncertainty_threshold():
    """Test uncertainty threshold computation."""
    entropy = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # 90th percentile should be close to 0.9 (numpy uses linear interpolation)
    threshold_90 = get_uncertainty_threshold(entropy, percentile=90)
    assert threshold_90 >= 0.85  # Should be in high range

    # 50th percentile should be median (0.55 due to linear interpolation)
    threshold_50 = get_uncertainty_threshold(entropy, percentile=50)
    assert threshold_50 == pytest.approx(0.55, abs=1e-5)


def test_temperature_scaling_fit():
    """Test temperature scaling fitting."""
    # Create dummy logits and labels
    batch_size = 32
    n_classes = 2
    logits = torch.randn(batch_size, n_classes) * 2  # Overconfident logits
    labels = torch.randint(0, n_classes, (batch_size,))

    # Fit temperature scaling
    temp_scaler = TemperatureScaling()
    optimal_temp = temp_scaler.fit(logits, labels, max_iter=10)

    # Temperature should be positive
    assert optimal_temp > 0

    # Check that forward pass works
    scaled_logits = temp_scaler(logits)
    assert scaled_logits.shape == logits.shape


def test_temperature_scaling_forward():
    """Test temperature scaling forward pass."""
    temp_scaler = TemperatureScaling()
    temp_scaler.temperature = nn.Parameter(torch.tensor([2.0]))

    logits = torch.tensor([[2.0, 1.0], [3.0, 0.5]])
    scaled_logits = temp_scaler(logits)

    # Check that logits are scaled by temperature
    expected = logits / 2.0
    assert torch.allclose(scaled_logits, expected, atol=1e-5)


def test_apply_temperature_scaling():
    """Test end-to-end temperature scaling application."""
    model = DummyModel()
    model.eval()

    # Create dummy dataloader
    batch_size = 8
    seq_len = 105
    input_dim = 11
    n_samples = 32

    X = torch.randn(n_samples, seq_len, input_dim)
    y = torch.randint(0, 2, (n_samples,))
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Apply temperature scaling
    temp_scaler, optimal_temp = apply_temperature_scaling(model, dataloader, device="cpu")

    # Check that temperature is positive
    assert optimal_temp > 0

    # Check that temp_scaler is a TemperatureScaling instance
    assert isinstance(temp_scaler, TemperatureScaling)


def test_mc_dropout_uncertainty_variance():
    """Test that MC Dropout produces variance in predictions."""
    model = DummyModel()
    model.eval()

    # Create dummy input
    x = torch.randn(2, 105, 11)

    # Run MC Dropout with many passes
    results = mc_dropout_predict(model, x, n_passes=50, dropout_rate=0.5)

    # Standard deviation should be non-zero (dropout creates variance)
    assert (results["type_probs_std"] > 0).any()
    assert (results["pointer_std"] > 0).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
