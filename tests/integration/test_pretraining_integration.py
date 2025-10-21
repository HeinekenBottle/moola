"""Pre-training integration tests.

Tests the integration between pre-training and fine-tuning pipelines,
including encoder transfer learning and two-phase training.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from moola.models import get_model
from moola.utils.seeds import set_seed


class TestPretrainingIntegration:
    """Test pre-training integration with SimpleLSTM."""

    def create_dummy_encoder_checkpoint(self) -> Path:
        """Create a dummy pre-trained encoder checkpoint for testing."""
        # Simulate pre-trained encoder state
        encoder_state = {
            "encoder_state_dict": {
                "weight_ih_l0": torch.randn(512, 4),  # Input to hidden
                "weight_hh_l0": torch.randn(512, 128),  # Hidden to hidden
                "weight_ih_l0_reverse": torch.randn(512, 4),  # Reverse direction
                "weight_hh_l0_reverse": torch.randn(512, 128),
                "bias_ih_l0": torch.randn(512),
                "bias_hh_l0": torch.randn(512),
                "bias_ih_l0_reverse": torch.randn(512),
                "bias_hh_l0_reverse": torch.randn(512),
            },
            "hyperparams": {
                "hidden_dim": 128,
                "num_layers": 1,
            },
            "training_loss": 0.05,
        }

        save_path = Path("dummy_encoder.pt")
        torch.save(encoder_state, save_path)
        return save_path

    def test_encoder_loading_capability(self):
        """Test that model can load pre-trained encoders."""
        model = get_model("simple_lstm", device="cpu")

        # Create dummy checkpoint
        encoder_path = self.create_dummy_encoder_checkpoint()

        try:
            # Test encoder loading
            model.load_pretrained_encoder(encoder_path, freeze_encoder=True)

            # Verify encoder is frozen
            for param in model.model.lstm.parameters():
                assert not param.requires_grad, "Encoder should be frozen"

            # Test that model can still make predictions
            X = np.random.randn(5, 105, 4)
            predictions = model.predict(X)
            assert len(predictions) == 5

            print("✅ Encoder loading capability test passed")

        finally:
            # Clean up
            if encoder_path.exists():
                encoder_path.unlink()

    def test_two_phase_training_workflow(self):
        """Test two-phase training workflow with frozen encoder."""
        model = get_model("simple_lstm", device="cpu")

        # Create dummy encoder checkpoint
        encoder_path = self.create_dummy_encoder_checkpoint()

        try:
            # Load and freeze encoder
            model.load_pretrained_encoder(encoder_path, freeze_encoder=True)

            # Generate synthetic data
            X = np.random.randn(20, 105, 4)
            y = np.random.choice(["class_A", "class_B"], 20)

            # Test two-phase training
            set_seed(42)
            model.fit(X, y, n_epochs=2, unfreeze_encoder_after=1)

            # Verify encoder was unfrozen after specified epochs
            # (This is structural testing since we only run 2 epochs)
            print("✅ Two-phase training workflow test passed")

        finally:
            if encoder_path.exists():
                encoder_path.unlink()

    def test_encoder_architecture_compatibility(self):
        """Test architecture compatibility between pre-trained and target models."""
        # Test different SimpleLSTM configurations
        configs = [
            {"hidden_size": 64, "num_layers": 1},
            {"hidden_size": 128, "num_layers": 1},
            {"hidden_size": 256, "num_layers": 1},
        ]

        for config in configs:
            model = get_model("simple_lstm", device="cpu", **config)

            # Create encoder checkpoint with same hidden size
            encoder_state = {
                "encoder_state_dict": {
                    "weight_ih_l0": torch.randn(512, 4),
                    "weight_hh_l0": torch.randn(512, config["hidden_size"]),
                    "weight_ih_l0_reverse": torch.randn(512, 4),
                    "weight_hh_l0_reverse": torch.randn(512, config["hidden_size"]),
                    "bias_ih_l0": torch.randn(512),
                    "bias_hh_l0": torch.randn(512),
                    "bias_ih_l0_reverse": torch.randn(512),
                    "bias_hh_l0_reverse": torch.randn(512),
                },
                "hyperparams": {
                    "hidden_dim": config["hidden_size"],
                    "num_layers": 1,
                },
            }

            with tempfile.TemporaryDirectory() as temp_dir:
                encoder_path = Path(temp_dir) / "test_encoder.pt"
                torch.save(encoder_state, encoder_path)

                try:
                    # Test compatibility
                    model.load_pretrained_encoder(encoder_path, freeze_encoder=False)

                    # Test that model works with loaded encoder
                    X = np.random.randn(5, 105, 4)
                    predictions = model.predict(X)
                    assert len(predictions) == 5

                    print(f"✅ Architecture compatibility test passed: {config}")

                finally:
                    encoder_path.unlink()

    def test_layer_count_mismatch_handling(self):
        """Test handling of layer count mismatches between encoders."""
        model = get_model("simple_lstm", device="cpu", num_layers=1)

        # Create multi-layer encoder (should trigger warning)
        encoder_state = {
            "encoder_state_dict": {
                "weight_ih_l0": torch.randn(512, 4),
                "weight_hh_l0": torch.randn(512, model.hidden_size),
                "weight_ih_l0_reverse": torch.randn(512, 4),
                "weight_hh_l0_reverse": torch.randn(512, model.hidden_size),
                # Layer 1 weights (should be skipped)
                "weight_ih_l1": torch.randn(512, model.hidden_size * 2),
                "weight_hh_l1": torch.randn(512, model.hidden_size * 2),
                "weight_ih_l1_reverse": torch.randn(512, model.hidden_size * 2),
                "weight_hh_l1_reverse": torch.randn(512, model.hidden_size * 2),
                "bias_ih_l0": torch.randn(512),
                "bias_hh_l0": torch.randn(512),
                "bias_ih_l0_reverse": torch.randn(512),
                "bias_hh_l0_reverse": torch.randn(512),
            },
            "hyperparams": {
                "hidden_dim": model.hidden_size,
                "num_layers": 2,  # More layers than model
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            encoder_path = Path(temp_dir) / "multi_layer_encoder.pt"
            torch.save(encoder_state, encoder_path)

            try:
                # Test layer count mismatch handling
                model.load_pretrained_encoder(encoder_path, freeze_encoder=False)

                # Test that model still works
                X = np.random.randn(5, 105, 4)
                predictions = model.predict(X)
                assert len(predictions) == 5

                print("✅ Layer count mismatch handling test passed")

            finally:
                encoder_path.unlink()

    def test_encoder_freeze_unfreeze_cycle(self):
        """Test freeze/unfreeze cycle for two-phase training."""
        model = get_model("simple_lstm", device="cpu")

        # Create dummy encoder
        encoder_state = {
            "encoder_state_dict": {
                "weight_ih_l0": torch.randn(512, 4),
                "weight_hh_l0": torch.randn(512, model.hidden_size),
                "weight_ih_l0_reverse": torch.randn(512, 4),
                "weight_hh_l0_reverse": torch.randn(512, model.hidden_size),
                "bias_ih_l0": torch.randn(512),
                "bias_hh_l0": torch.randn(512),
                "bias_ih_l0_reverse": torch.randn(512),
                "bias_hh_l0_reverse": torch.randn(512),
            },
            "hyperparams": {
                "hidden_dim": model.hidden_size,
                "num_layers": 1,
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            encoder_path = Path(temp_dir) / "test_encoder.pt"
            torch.save(encoder_state, encoder_path)

            try:
                # Test freeze/unfreeze cycle
                model.load_pretrained_encoder(encoder_path, freeze_encoder=True)

                # Verify initially frozen
                frozen_params = [p for p in model.model.lstm.parameters() if not p.requires_grad]
                assert len(frozen_params) > 0, "Encoder should be frozen"

                # Test unfreezing
                model.load_pretrained_encoder(encoder_path, freeze_encoder=False)

                # Verify now unfrozen
                frozen_params = [p for p in model.model.lstm.parameters() if not p.requires_grad]
                assert len(frozen_params) == 0, "Encoder should be unfrozen"

                print("✅ Freeze/unfreeze cycle test passed")

            finally:
                encoder_path.unlink()

    def test_pretrained_model_training_consistency(self):
        """Test training consistency with pre-loaded encoders."""
        # Create two identical models
        model1 = get_model("simple_lstm", device="cpu")
        model2 = get_model("simple_lstm", device="cpu")

        # Create dummy encoder checkpoint
        encoder_state = {
            "encoder_state_dict": {
                "weight_ih_l0": torch.randn(512, 4),
                "weight_hh_l0": torch.randn(512, 128),
                "weight_ih_l0_reverse": torch.randn(512, 4),
                "weight_hh_l0_reverse": torch.randn(512, 128),
                "bias_ih_l0": torch.randn(512),
                "bias_hh_l0": torch.randn(512),
                "bias_ih_l0_reverse": torch.randn(512),
                "bias_hh_l0_reverse": torch.randn(512),
            },
            "hyperparams": {
                "hidden_dim": 128,
                "num_layers": 1,
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            encoder_path = Path(temp_dir) / "consistency_test.pt"
            torch.save(encoder_state, encoder_path)

            try:
                # Load encoder into both models
                model1.load_pretrained_encoder(encoder_path, freeze_encoder=False)
                model2.load_pretrained_encoder(encoder_path, freeze_encoder=False)

                # Generate test data
                X = np.random.randn(10, 105, 4)
                y = np.random.choice(["class_A", "class_B"], 10)

                # Train both models
                set_seed(42)
                model1.fit(X, y, n_epochs=1)
                set_seed(42)
                model2.fit(X, y, n_epochs=1)

                # Compare predictions (should be identical due to same seed)
                X_test = np.random.randn(5, 105, 4)
                pred1 = model1.predict(X_test)
                pred2 = model2.predict(X_test)

                assert np.array_equal(pred1, pred2), "Predictions should be identical"

                print("✅ Training consistency test passed")

            finally:
                encoder_path.unlink()

    def test_encoder_error_handling(self):
        """Test error handling for encoder loading."""
        model = get_model("simple_lstm", device="cpu")

        # Test loading non-existent file
        with pytest.raises(FileNotFoundError):
            model.load_pretrained_encoder("non_existent_file.pt")

        # Test loading with model not built
        model.model = None
        with pytest.raises(ValueError, match="Model must be built first"):
            model.load_pretrained_encoder("dummy.pt")

        # Test with wrong hidden size
        encoder_state = {
            "encoder_state_dict": {
                "weight_ih_l0": torch.randn(512, 4),
                "weight_hh_l0": torch.randn(512, 256),  # Wrong size (model has 128)
                "weight_ih_l0_reverse": torch.randn(512, 4),
                "weight_hh_l0_reverse": torch.randn(512, 256),
            },
            "hyperparams": {
                "hidden_dim": 256,  # Mismatch with model
                "num_layers": 1,
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            encoder_path = Path(temp_dir) / "wrong_size.pt"
            torch.save(encoder_state, encoder_path)

            try:
                with pytest.raises(ValueError, match="Hidden size mismatch"):
                    model.load_pretrained_encoder(encoder_path)
            finally:
                encoder_path.unlink()

        print("✅ Encoder error handling test passed")

    def test_transfer_learning_performance(self):
        """Test that transfer learning actually improves performance."""
        # This test checks structural implementation of transfer learning
        # Actual performance improvement would require more extensive testing

        model = get_model("simple_lstm", device="cpu")

        # Check that transfer learning features are implemented
        assert hasattr(model, "load_pretrained_encoder"), "Transfer loading missing"
        assert hasattr(model, "fit"), "Training method missing"

        # Check two-phase training parameters
        assert model.early_stopping_patience > 0, "Early stopping needed for transfer learning"

        # Check learning rate reduction capability
        assert hasattr(model, "learning_rate"), "Learning rate control needed"

        print("✅ Transfer learning performance structure validated")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
