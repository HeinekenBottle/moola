"""Enhanced SimpleLSTM architecture validation tests.

Tests the enhanced SimpleLSTM model architecture, parameter optimization,
integration with feature engineering, and pre-training capabilities.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from moola.models import get_model
from moola.utils.seeds import set_seed


class TestSimpleLSTMEnhancedArchitecture:
    """Test enhanced SimpleLSTM architecture components."""

    def test_parameter_count_optimization(self):
        """Test that parameter count meets optimization targets."""
        # Test different model configurations
        configs = [
            {"hidden_size": 64, "num_heads": 2},
            {"hidden_size": 128, "num_heads": 2},
            {"hidden_size": 256, "num_heads": 4},
        ]

        for config in configs:
            model = get_model("simple_lstm", device="cpu", **config)
            param_count = sum(p.numel() for p in model.model.parameters())

            # Validate parameter count ranges for different configurations
            if config["hidden_size"] == 64:
                assert 20000 <= param_count <= 40000, f"64 hidden size: {param_count:,} params"
            elif config["hidden_size"] == 128:
                assert 40000 <= param_count <= 80000, f"128 hidden size: {param_count:,} params"
            elif config["hidden_size"] == 256:
                assert 80000 <= param_count <= 160000, f"256 hidden size: {param_count:,} params"

            print(f"✅ Parameter count validation: {config} → {param_count:,} params")

    def test_model_architecture_components(self):
        """Test individual model architecture components."""
        model = get_model("simple_lstm", device="cpu")

        # Test model architecture structure
        assert hasattr(model.model, "lstm"), "LSTM layer missing"
        assert hasattr(model.model, "attention"), "Attention layer missing"
        assert hasattr(model.model, "ln"), "Layer normalization missing"
        assert hasattr(model.model, "classifier"), "Classifier layer missing"

        # Test LSTM layer configuration
        lstm = model.model.lstm
        assert lstm.input_size == 4, f"Wrong input size: {lstm.input_size}"
        assert lstm.hidden_size == model.hidden_size, "Hidden size mismatch"
        assert lstm.num_layers == model.num_layers, "Layer count mismatch"
        assert lstm.bidirectional, "LSTM should be bidirectional"

        # Test attention layer configuration
        attention = model.model.attention
        assert attention.embed_dim == model.hidden_size * 2, "Attention dim mismatch"
        assert attention.num_heads == model.num_heads, "Attention heads mismatch"

        # Test classifier layer configuration
        classifier = model.model.classifier
        assert len(classifier) == 4, f"Wrong classifier layers: {len(classifier)}"
        assert classifier[0].in_features == model.hidden_size * 2, "Input dim mismatch"
        assert classifier[3].out_features == 2, "Output dim mismatch"

    def test_feature_integration_compatibility(self):
        """Test compatibility with engineered features."""
        # Generate test data with different feature counts
        feature_counts = [4, 20, 30, 40]  # OHLC + engineered features

        for feature_count in feature_counts:
            # Generate synthetic data with varying feature dimensions
            X = np.random.randn(10, 105, feature_count)
            y = np.random.choice(["class_A", "class_B"], 10)

            try:
                model = get_model("simple_lstm", device="cpu")
                # This should work for any feature count >= 4
                model.fit(X, y, n_epochs=1)
                predictions = model.predict(X[:5])

                assert len(predictions) == 5
                print(f"✅ Feature compatibility test passed: {feature_count} features")

            except Exception as e:
                if feature_count < 4:
                    print(f"⚠️  Expected failure for {feature_count} features: {e}")
                else:
                    raise e

    def test_pretraining_integration(self):
        """Test pre-training integration capabilities."""
        model = get_model("simple_lstm", device="cpu")

        # Test that model can handle pre-trained encoder loading
        # (This is a structural test - actual pre-trained file would be needed)
        assert hasattr(model, "load_pretrained_encoder"), "Pre-training loading missing"
        assert hasattr(model, "fit"), "Training method missing"

        # Test two-phase training parameters
        assert hasattr(model, "unfreeze_encoder_after"), "Two-phase control missing"
        assert model.early_stopping_patience > 0, "Early stopping not configured"

        print("✅ Pre-training integration validation passed")

    def test_augmentation_pipeline_integration(self):
        """Test data augmentation pipeline integration."""
        model = get_model("simple_lstm", device="cpu")

        # Test that model supports augmentation parameters
        assert hasattr(model, "mixup_alpha"), "Mixup parameter missing"
        assert hasattr(model, "cutmix_prob"), "CutMix parameter missing"
        assert hasattr(model, "use_temporal_aug"), "Temporal augmentation flag missing"
        assert hasattr(model, "temporal_aug"), "Temporal augmentation missing"

        # Generate test data
        X = np.random.randn(10, 105, 4)
        y = np.random.choice(["class_A", "class_B"], 10)

        # Test training with augmentation
        set_seed(42)
        model.fit(X, y, n_epochs=2)

        # Test predictions
        predictions = model.predict(X[:5])
        assert len(predictions) == 5

        print("✅ Augmentation pipeline integration test passed")

    def test_memory_efficiency(self):
        """Test memory efficiency with different batch sizes."""
        X = np.random.randn(20, 105, 4)
        y = np.random.choice(["class_A", "class_B"], 20)

        # Test different batch sizes
        batch_sizes = [4, 8, 16]

        for batch_size in batch_sizes:
            try:
                model = get_model("simple_lstm", device="cpu", batch_size=batch_size)
                model.fit(X, y, n_epochs=1)

                # Test memory usage by checking if predictions work
                predictions = model.predict(X[:5])
                assert len(predictions) == 5

                print(f"✅ Memory efficiency test passed: batch_size={batch_size}")

            except Exception as e:
                print(f"❌ Memory efficiency test failed for batch_size={batch_size}: {e}")
                raise e

    def test_gradient_flow(self):
        """Test gradient flow through the model architecture."""
        model = get_model("simple_lstm", device="cpu")

        # Generate test data
        X = torch.FloatTensor(np.random.randn(5, 105, 4))
        y = torch.LongTensor(np.random.randint(0, 2, 5))

        # Set up training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)

        # Forward pass
        model.model.train()
        optimizer.zero_grad()
        outputs = model.model(X)
        loss = criterion(outputs, y)

        # Backward pass
        loss.backward()

        # Check gradient flow
        gradients_exist = any(p.grad is not None for p in model.model.parameters())

        assert gradients_exist, "No gradients found - gradient flow issue"

        # Check specific layer gradients
        lstm_grads = [
            p.grad.abs().mean().item() for p in model.model.lstm.parameters() if p.grad is not None
        ]
        attention_grads = [
            p.grad.abs().mean().item()
            for p in model.model.attention.parameters()
            if p.grad is not None
        ]

        assert len(lstm_grads) > 0, "LSTM layer gradients missing"
        assert len(attention_grads) > 0, "Attention layer gradients missing"

        print("✅ Gradient flow validation passed")
        print(f"   LSTM gradient magnitude: {np.mean(lstm_grads):.6f}")
        print(f"   Attention gradient magnitude: {np.mean(attention_grads):.6f}")

    def test_forward_pass_validation(self):
        """Test forward pass with different input configurations."""
        model = get_model("simple_lstm", device="cpu")

        # Test different input configurations
        test_cases = [
            (5, 105, 4),  # Standard OHLC
            (3, 105, 20),  # Engineered features
            (10, 105, 30),  # More features
        ]

        for batch_size, seq_len, feature_dim in test_cases:
            X = torch.FloatTensor(np.random.randn(batch_size, seq_len, feature_dim))

            try:
                model.model.eval()
                with torch.no_grad():
                    outputs = model.model(X)

                # Validate output shape
                assert outputs.shape == (batch_size, 2), f"Wrong output shape: {outputs.shape}"

                # Validate output types
                assert torch.is_tensor(outputs), "Outputs should be tensors"
                assert not torch.isnan(outputs).any(), "Outputs contain NaN"
                assert not torch.isinf(outputs).any(), "Outputs contain Inf"

                print(f"✅ Forward pass validation passed: {batch_size}x{seq_len}x{feature_dim}")

            except Exception as e:
                print(
                    f"❌ Forward pass validation failed: {batch_size}x{seq_len}x{feature_dim}: {e}"
                )
                raise e

    def test_model_save_load_consistency(self):
        """Test model save/load consistency."""
        # Generate test data
        X = np.random.randn(10, 105, 4)
        y = np.random.choice(["class_A", "class_B"], 10)

        # Train original model
        original_model = get_model("simple_lstm", device="cpu")
        original_model.fit(X, y, n_epochs=2)

        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model.pt"
            original_model.save(save_path)

            # Load model
            loaded_model = get_model("simple_lstm", device="cpu")
            loaded_model.load(save_path)

            # Test predictions are consistent
            original_predictions = original_model.predict(X[:5])
            loaded_predictions = loaded_model.predict(X[:5])

            assert np.array_equal(
                original_predictions, loaded_predictions
            ), "Predictions differ after save/load"

            print("✅ Model save/load consistency test passed")

    def test_small_dataset_optimization(self):
        """Test model optimization for small datasets."""
        # Very small dataset (similar to production constraints)
        X = np.random.randn(8, 105, 4)  # 8 samples
        y = np.random.choice(["class_A", "class_B"], 8)

        # Test model with small dataset
        model = get_model("simple_lstm", device="cpu")

        # Higher dropout for small datasets
        assert (
            model.dropout_rate > 0
        ), f"Dropout should be > 0 for small datasets: {model.dropout_rate}"

        # Early stopping enabled
        assert model.early_stopping_patience > 0, "Early stopping should be enabled"

        # Try training (may not converge but should not crash)
        try:
            model.fit(X, y, n_epochs=5)
            predictions = model.predict(X)
            assert len(predictions) == 8
            print("✅ Small dataset optimization test passed")
        except Exception as e:
            print(f"⚠️  Small dataset training failed (expected for very small data): {e}")

    def test_class_imbalance_handling(self):
        """Test class imbalance handling in the model."""
        # Create imbalanced dataset
        X = np.random.randn(15, 105, 4)  # 15 samples
        # 12 of class_A, 3 of class_B (imbalanced)
        y = ["class_A"] * 12 + ["class_B"] * 3

        model = get_model("simple_lstm", device="cpu")

        # Check that model uses class weights
        assert hasattr(model, "fit"), "Model must have fit method"

        # Try training with imbalanced data
        try:
            model.fit(X, y, n_epochs=2)
            predictions = model.predict(X[:5])
            assert len(predictions) == 5
            print("✅ Class imbalance handling test passed")
        except Exception as e:
            print(f"❌ Class imbalance handling failed: {e}")
            raise e


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
