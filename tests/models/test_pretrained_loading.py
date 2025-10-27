"""Unit tests for strict pretrained weight loading."""

import pytest
import torch
import torch.nn as nn

from moola.models.pretrained_utils import load_pretrained_strict


class DummyModel(nn.Module):
    """Dummy model for testing pretrained loading."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 20)
        self.head = nn.Linear(20, 2)


class DummyLSTMModel(nn.Module):
    """Dummy LSTM model that mimics SimpleLSTM structure."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, 128, 1, batch_first=True, bidirectional=True)
        self.head = nn.Linear(256, 2)


def test_load_pretrained_strict_success(tmp_path):
    """Test successful pretrained loading with 100% match."""
    model = DummyModel()

    # Save checkpoint
    checkpoint_path = tmp_path / "encoder.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    # Create new model and load
    new_model = DummyModel()
    stats = load_pretrained_strict(
        new_model,
        str(checkpoint_path),
        freeze_encoder=False,
        min_match_ratio=0.80,
    )

    assert stats["n_matched"] == 4  # encoder.weight, encoder.bias, head.weight, head.bias
    assert stats["match_ratio"] == 1.0
    assert stats["n_mismatched"] == 0
    assert stats["n_missing"] == 0


def test_load_pretrained_strict_with_freezing(tmp_path):
    """Test that freezing works correctly."""
    model = DummyModel()

    # Save checkpoint
    checkpoint_path = tmp_path / "encoder.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    # Create new model and load with freezing
    new_model = DummyModel()
    stats = load_pretrained_strict(
        new_model,
        str(checkpoint_path),
        freeze_encoder=True,
        min_match_ratio=0.80,
    )

    # Check that encoder parameters are frozen
    assert not new_model.encoder.weight.requires_grad
    assert not new_model.encoder.bias.requires_grad
    assert stats["n_frozen"] > 0


def test_load_pretrained_strict_low_match_ratio(tmp_path):
    """Test that low match ratio aborts with assertion."""
    model = DummyModel()
    checkpoint_path = tmp_path / "encoder.pt"

    # Save only partial state dict (just encoder, not head)
    partial_state = {
        "encoder.weight": model.encoder.weight,
        "encoder.bias": model.encoder.bias,
    }
    torch.save({"model_state_dict": partial_state}, checkpoint_path)

    # Create new model with all parameters
    new_model = DummyModel()

    # Should fail because only 2/4 parameters match (50% < 80%)
    with pytest.raises(AssertionError, match="match ratio"):
        load_pretrained_strict(
            new_model,
            str(checkpoint_path),
            min_match_ratio=0.80,
        )


def test_load_pretrained_strict_shape_mismatch(tmp_path):
    """Test that shape mismatches abort with assertion."""
    model = DummyModel()
    checkpoint_path = tmp_path / "encoder.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    # Create model with different shape
    class DifferentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(10, 30)  # Different size!
            self.head = nn.Linear(30, 2)

    new_model = DifferentModel()

    # Should fail due to low match ratio (shape mismatches reduce matches)
    # The error message will mention "match ratio" not "shape mismatch" because
    # the first check (match_ratio < min) fires before the shape check
    with pytest.raises(AssertionError, match="match ratio|shape mismatch"):
        load_pretrained_strict(
            new_model,
            str(checkpoint_path),
            allow_shape_mismatch=False,
        )


def test_load_pretrained_strict_file_not_found():
    """Test that missing checkpoint file raises FileNotFoundError."""
    model = DummyModel()
    nonexistent_path = "/tmp/this_file_does_not_exist_12345.pt"

    with pytest.raises(FileNotFoundError, match="Pretrained checkpoint not found"):
        load_pretrained_strict(
            model,
            nonexistent_path,
        )


def test_load_pretrained_lstm_encoder_format(tmp_path):
    """Test loading encoder-only checkpoint (from pretraining)."""
    # Create encoder checkpoint in pretraining format
    lstm = nn.LSTM(4, 128, 1, batch_first=True, bidirectional=True)
    encoder_state = {k: v for k, v in lstm.state_dict().items()}

    checkpoint = {
        "encoder_state_dict": encoder_state,
        "hyperparams": {
            "hidden_dim": 128,
            "num_layers": 1,
            "input_dim": 4,
        },
    }

    checkpoint_path = tmp_path / "bilstm_encoder.pt"
    torch.save(checkpoint, checkpoint_path)

    # Load into model with lstm module
    model = DummyLSTMModel()
    stats = load_pretrained_strict(
        model,
        str(checkpoint_path),
        freeze_encoder=True,
        min_match_ratio=0.80,
    )

    # Should match LSTM parameters
    assert stats["n_matched"] > 0
    assert stats["match_ratio"] >= 0.80
    assert stats["n_mismatched"] == 0

    # Check LSTM is frozen
    for param in model.lstm.parameters():
        assert not param.requires_grad


def test_load_pretrained_with_missing_tensors_ok(tmp_path):
    """Test that missing tensors (model has more than checkpoint) is OK if ratio met."""
    # Save encoder only
    encoder = nn.Linear(10, 20)
    checkpoint_path = tmp_path / "encoder_only.pt"
    torch.save(
        {"model_state_dict": {"encoder.weight": encoder.weight, "encoder.bias": encoder.bias}},
        checkpoint_path,
    )

    # Load into full model
    model = DummyModel()

    # Should pass with lower threshold (encoder is 2/4 = 50%)
    stats = load_pretrained_strict(
        model,
        str(checkpoint_path),
        min_match_ratio=0.40,  # Lower threshold
        freeze_encoder=False,
    )

    assert stats["n_matched"] == 2
    assert stats["n_missing"] == 2  # head.weight and head.bias
    assert stats["match_ratio"] == 0.5


def test_load_pretrained_different_state_dict_keys(tmp_path):
    """Test handling of different state dict key formats."""
    model = DummyModel()

    # Test different checkpoint formats
    formats = [
        {"model_state_dict": model.state_dict()},  # Standard format
        {"state_dict": model.state_dict()},  # Alternative format
        model.state_dict(),  # Direct state dict
    ]

    for i, checkpoint_format in enumerate(formats):
        checkpoint_path = tmp_path / f"checkpoint_{i}.pt"
        torch.save(checkpoint_format, checkpoint_path)

        new_model = DummyModel()
        stats = load_pretrained_strict(
            new_model,
            str(checkpoint_path),
            min_match_ratio=0.80,
        )

        assert stats["match_ratio"] >= 0.80, f"Format {i} failed"


def test_load_pretrained_reports_correctly(tmp_path):
    """Test that pretrained loading returns correct stats."""
    model = DummyModel()
    checkpoint_path = tmp_path / "encoder.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    new_model = DummyModel()
    stats = load_pretrained_strict(
        new_model,
        str(checkpoint_path),
        freeze_encoder=True,
        min_match_ratio=0.80,
    )

    # Check that stats are correct
    assert stats["n_matched"] == 4
    assert stats["n_missing"] == 0
    assert stats["n_mismatched"] == 0
    assert stats["match_ratio"] == 1.0
    assert stats["n_frozen"] > 0
    assert "matched" in stats
    assert "missing" in stats
    assert "mismatched" in stats
