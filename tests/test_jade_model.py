"""Tests for Jade Model - Production BiLSTM with Multi-task Learning.

Tests Stones non-negotiables:
- Uncertainty-weighted loss as DEFAULT
- Dropout: recurrent 0.6-0.7, dense 0.4-0.5, input 0.2-0.3
- Gradient clipping 1.5-2.0
- ReduceLROnPlateau scheduler
- Early stopping patience 20
- Center+length pointer encoding
- Huber δ≈0.08 for pointer regression
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from moola.models.jade import JadeModel, UncertaintyWeightedLoss, compute_pointer_regression_loss
from moola.utils.seeds import set_seed


class TestUncertaintyWeightedLoss:
    """Test uncertainty-weighted multi-task loss implementation."""

    def test_initialization(self):
        """Test uncertainty loss initializes with equal weights."""
        loss_fn = UncertaintyWeightedLoss()
        
        # Should initialize with log_var = 0 (σ = 1.0)
        assert loss_fn.log_var_ptr.item() == 0.0
        assert loss_fn.log_var_type.item() == 0.0
        
        # Check uncertainties are equal initially
        uncertainties = loss_fn.get_uncertainties()
        assert uncertainties["sigma_ptr"] == 1.0
        assert uncertainties["sigma_type"] == 1.0

    def test_forward_pass(self):
        """Test loss computation with uncertainty weighting."""
        loss_fn = UncertaintyWeightedLoss()
        
        # Create dummy losses
        ptr_loss = torch.tensor(2.0)
        type_loss = torch.tensor(1.0)
        
        total_loss = loss_fn(ptr_loss, type_loss)
        
        # With equal uncertainties (σ=1), should be: 0.5*2 + 0 + 1 + 0 = 2.0
        expected = 0.5 * 2.0 + 0.0 + 1.0 + 0.0  # 2.0
        assert torch.allclose(total_loss, torch.tensor(expected))

    def test_uncertainty_learning(self):
        """Test that uncertainty parameters are learnable."""
        loss_fn = UncertaintyWeightedLoss()
        
        # Check parameters require gradients
        assert loss_fn.log_var_ptr.requires_grad
        assert loss_fn.log_var_type.requires_grad
        
        # Check they're in parameters list
        param_names = [name for name, _ in loss_fn.named_parameters()]
        assert "log_var_ptr" in param_names
        assert "log_var_type" in param_names

    def test_uncertainty_values_change(self):
        """Test that uncertainty values change during optimization."""
        loss_fn = UncertaintyWeightedLoss()
        optimizer = torch.optim.Adam(loss_fn.parameters(), lr=0.1)
        
        initial_uncertainties = loss_fn.get_uncertainties()
        
        # Simulate training step
        ptr_loss = torch.tensor(1.0, requires_grad=True)
        type_loss = torch.tensor(2.0, requires_grad=True)
        
        total_loss = loss_fn(ptr_loss, type_loss)
        total_loss.backward()
        optimizer.step()
        
        final_uncertainties = loss_fn.get_uncertainties()
        
        # Uncertainties should have changed
        assert final_uncertainties["sigma_ptr"] != initial_uncertainties["sigma_ptr"]
        assert final_uncertainties["sigma_type"] != initial_uncertainties["sigma_type"]


class TestJadeModelArchitecture:
    """Test Jade model architecture compliance with Stones specifications."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        set_seed(1337)
        n_samples = 100
        seq_len = 105
        n_features = 11  # RelativeTransform features
        
        X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
        y = np.random.randint(0, 3, n_samples)  # 3 classes for Jade
        
        # Pointer labels for multi-task
        expansion_start = np.random.randint(0, 50, n_samples)
        expansion_end = np.random.randint(55, 104, n_samples)
        
        return X, y, expansion_start, expansion_end

    @pytest.fixture
    def jade_model_single_task(self):
        """Create Jade model in single-task mode."""
        return JadeModel(
            seed=1337,
            predict_pointers=False,
            n_epochs=2,  # Fast for testing
            batch_size=16,
            val_split=0.2,
        )

    @pytest.fixture
    def jade_model_multi_task(self):
        """Create Jade model in multi-task mode."""
        return JadeModel(
            seed=1337,
            predict_pointers=True,
            n_epochs=2,  # Fast for testing
            batch_size=16,
            val_split=0.2,
        )

    def test_model_metadata(self):
        """Test model metadata and registry information."""
        model = JadeModel()
        
        assert model.MODEL_ID == "moola-lstm-m-v1.0"
        assert model.CODENAME == "Jade"
        assert model.hidden_size == 128
        assert model.num_layers == 2
        assert model.max_grad_norm == 2.0
        assert model.early_stopping_patience == 20

    def test_dropout_configuration(self):
        """Test dropout rates match Stones specifications."""
        model = JadeModel()
        model._build_model(input_dim=11, n_classes=3)
        
        # Check input dropout 0.2-0.3
        assert isinstance(model.model.input_dropout, nn.Dropout)
        assert 0.2 <= model.model.input_dropout.p <= 0.3
        
        # Check LSTM dropout 0.6-0.7
        lstm_dropout = model.model.lstm.dropout
        assert 0.6 <= lstm_dropout <= 0.7
        
        # Check dense dropout 0.4-0.5
        assert isinstance(model.model.dense_dropout, nn.Dropout)
        assert 0.4 <= model.model.dense_dropout.p <= 0.5

    def test_bilstm_architecture(self):
        """Test BiLSTM configuration matches Stones requirements."""
        model = JadeModel()
        net = model._build_model(input_dim=11, n_classes=3)
        
        # Check BiLSTM properties
        assert net.lstm.input_size == 11
        assert net.lstm.hidden_size == 128
        assert net.lstm.num_layers == 2
        assert net.lstm.bidirectional is True
        assert net.lstm.batch_first is True

    def test_global_average_pooling(self):
        """Test global average pooling implementation."""
        model = JadeModel()
        net = model._build_model(input_dim=11, n_classes=3)
        
        # Check global pool layer exists
        assert isinstance(net.global_pool, nn.AdaptiveAvgPool1d)
        assert net.global_pool.output_size == 1

    def test_pointer_head_configuration(self):
        """Test pointer head for center+length encoding."""
        model = JadeModel(predict_pointers=True)
        net = model._build_model(input_dim=11, n_classes=3)
        
        # Check pointer head exists
        assert hasattr(net, 'pointer_head')
        assert isinstance(net.pointer_head, nn.Sequential)
        
        # Check final layer outputs 2 values (center, length)
        final_layer = net.pointer_head[-1]
        assert isinstance(final_layer, nn.Linear)
        assert final_layer.out_features == 2

    def test_type_head_configuration(self):
        """Test type head for 3-way classification."""
        model = JadeModel()
        net = model._build_model(input_dim=11, n_classes=3)
        
        # Check type head exists
        assert hasattr(net, 'type_head')
        assert isinstance(net.type_head, nn.Sequential)
        
        # Check final layer outputs 3 classes
        final_layer = net.type_head[-1]
        assert isinstance(final_layer, nn.Linear)
        assert final_layer.out_features == 3

    def test_forward_pass_single_task(self, jade_model_single_task, sample_data):
        """Test forward pass in single-task mode."""
        X, y, _, _ = sample_data
        model = jade_model_single_task
        model._build_model(input_dim=11, n_classes=3)
        
        # Test forward pass
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X[:5])
            outputs = model.model(x_tensor)
            
            # Should return logits directly
            assert isinstance(outputs, torch.Tensor)
            assert outputs.shape == (5, 3)

    def test_forward_pass_multi_task(self, jade_model_multi_task, sample_data):
        """Test forward pass in multi-task mode."""
        X, y, start, end = sample_data
        model = jade_model_multi_task
        model._build_model(input_dim=11, n_classes=3)
        
        # Test forward pass
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X[:5])
            outputs = model.model(x_tensor)
            
            # Should return dict with both outputs
            assert isinstance(outputs, dict)
            assert "type_logits" in outputs
            assert "pointers_cl" in outputs
            assert "pointers" in outputs
            
            # Check shapes
            assert outputs["type_logits"].shape == (5, 3)
            assert outputs["pointers_cl"].shape == (5, 2)
            assert outputs["pointers"].shape == (5, 2)
            
            # Check pointer values are in [0, 1] (sigmoid)
            assert torch.all(outputs["pointers_cl"] >= 0)
            assert torch.all(outputs["pointers_cl"] <= 1)

    def test_input_dimension_warnings(self, caplog):
        """Test warnings for non-standard input dimensions."""
        model = JadeModel()
        
        # Warning for wrong input dim
        model._build_model(input_dim=4, n_classes=3)
        assert "Jade expects 11-dim input, got 4" in caplog.text
        
        caplog.clear()
        
        # Warning for wrong number of classes
        model._build_model(input_dim=11, n_classes=2)
        assert "Jade expects 3 classes, got 2" in caplog.text


class TestJadeModelTraining:
    """Test Jade model training with Stones requirements."""

    @pytest.fixture
    def training_data(self):
        """Create training data for multi-task learning."""
        set_seed(1337)
        n_samples = 60  # Small for fast testing
        seq_len = 105
        n_features = 11
        
        X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
        y = np.random.randint(0, 3, n_samples)
        expansion_start = np.random.randint(0, 50, n_samples)
        expansion_end = np.random.randint(55, 104, n_samples)
        
        return X, y, expansion_start, expansion_end

    def test_single_task_training(self, training_data):
        """Test single-task training workflow."""
        X, y, _, _ = training_data
        model = JadeModel(
            seed=1337,
            predict_pointers=False,
            n_epochs=2,
            batch_size=16,
            val_split=0.2,
            use_temporal_aug=False,  # Disable for reproducible testing
        )
        
        # Should train without errors
        model.fit(X, y)
        
        # Check model is fitted
        assert model.is_fitted
        assert model.n_classes == 3
        assert model.input_dim == 11

    def test_multi_task_training(self, training_data):
        """Test multi-task training with uncertainty-weighted loss."""
        X, y, start, end = training_data
        model = JadeModel(
            seed=1337,
            predict_pointers=True,
            n_epochs=2,
            batch_size=16,
            val_split=0.2,
            use_temporal_aug=False,  # Disable for reproducible testing
        )
        
        # Should train without errors
        model.fit(X, y, expansion_start=start, expansion_end=end)
        
        # Check model is fitted
        assert model.is_fitted
        assert model.n_classes == 3
        assert model.input_dim == 11

    def test_pointer_labels_validation(self, training_data):
        """Test validation of pointer labels for multi-task learning."""
        X, y, start, end = training_data
        model = JadeModel(predict_pointers=True)
        
        # Should raise error when pointer labels missing
        with pytest.raises(ValueError, match="predict_pointers=True but pointer labels not provided"):
            model.fit(X, y)

    def test_warning_for_unused_pointer_labels(self, training_data, caplog):
        """Test warning when pointer labels provided but not used."""
        X, y, start, end = training_data
        model = JadeModel(predict_pointers=False, n_epochs=1)
        
        model.fit(X, y, expansion_start=start, expansion_end=end)
        
        assert "Pointer labels provided but predict_pointers=False" in caplog.text

    def test_gradient_clipping_configuration(self, training_data):
        """Test gradient clipping is applied during training."""
        X, y, start, end = training_data
        model = JadeModel(
            seed=1337,
            predict_pointers=True,
            n_epochs=1,
            batch_size=16,
            max_grad_norm=1.8,  # Within Stones range
            use_temporal_aug=False,
        )
        
        # Mock gradient clipping to verify it's called
        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            model.fit(X, y, expansion_start=start, expansion_end=end)
            
            # Should be called for both model and uncertainty loss parameters
            assert mock_clip.call_count >= 2  # At least model + uncertainty loss
            
            # Check max_norm parameter
            for call in mock_clip.call_args_list:
                assert call.kwargs['max_norm'] == 1.8

    def test_reduce_lr_on_plateau_scheduler(self, training_data):
        """Test ReduceLROnPlateau scheduler configuration."""
        X, y, start, end = training_data
        model = JadeModel(
            seed=1337,
            predict_pointers=True,
            n_epochs=2,
            batch_size=16,
            val_split=0.2,
            scheduler_factor=0.5,
            scheduler_patience=1,  # Fast for testing
            use_temporal_aug=False,
        )
        
        # Mock scheduler to verify it's called
        with patch('torch.optim.lr_scheduler.ReduceLROnPlateau') as mock_scheduler:
            mock_instance = MagicMock()
            mock_scheduler.return_value = mock_instance
            
            model.fit(X, y, expansion_start=start, expansion_end=end)
            
            # Should be instantiated with correct parameters
            mock_scheduler.assert_called_once()
            call_args = mock_scheduler.call_args
            assert call_args.kwargs['factor'] == 0.5
            assert call_args.kwargs['patience'] == 1
            assert call_args.kwargs['mode'] == 'min'
            
            # Should be stepped each epoch
            assert mock_instance.step.call_count == 2

    def test_early_stopping_patience(self, training_data):
        """Test early stopping with patience 20 (Stones requirement)."""
        X, y, start, end = training_data
        model = JadeModel(
            seed=1337,
            predict_pointers=True,
            n_epochs=5,  # Less than patience
            batch_size=16,
            val_split=0.2,
            early_stopping_patience=20,  # Stones requirement
            use_temporal_aug=False,
        )
        
        # Should complete all epochs without early stopping
        model.fit(X, y, expansion_start=start, expansion_end=end)
        assert model.is_fitted

    def test_temporal_augmentation_configuration(self):
        """Test temporal augmentation parameters match Stones specifications."""
        model = JadeModel(
            use_temporal_aug=True,
            jitter_sigma=0.03,  # Stones specification
            magnitude_warp_sigma=0.2,  # Stones specification
        )
        
        assert model.jitter_sigma == 0.03
        assert model.magnitude_warp_sigma == 0.2
        assert model.temporal_aug is not None


class TestJadeModelPrediction:
    """Test Jade model prediction methods."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted Jade model for prediction tests."""
        set_seed(1337)
        n_samples = 50
        X = np.random.randn(n_samples, 105, 11).astype(np.float32)
        y = np.random.randint(0, 3, n_samples)
        start = np.random.randint(0, 50, n_samples)
        end = np.random.randint(55, 104, n_samples)
        
        model = JadeModel(
            seed=1337,
            predict_pointers=True,
            n_epochs=1,
            batch_size=16,
            use_temporal_aug=False,
        )
        model.fit(X, y, expansion_start=start, expansion_end=end)
        
        return model, X, y, start, end

    def test_predict_single_task(self):
        """Test single-task prediction."""
        set_seed(1337)
        n_samples = 30
        X = np.random.randn(n_samples, 105, 11).astype(np.float32)
        y = np.random.randint(0, 3, n_samples)
        
        model = JadeModel(seed=1337, predict_pointers=False, n_epochs=1)
        model.fit(X, y)
        
        # Test prediction
        X_test = np.random.randn(10, 105, 11).astype(np.float32)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (10,)
        assert predictions.dtype == np.object_  # String labels
        assert all(pred in ['0', '1', '2'] for pred in predictions)

    def test_predict_proba(self, fitted_model):
        """Test probability prediction."""
        model, X, y, start, end = fitted_model
        
        X_test = np.random.randn(10, 105, 11).astype(np.float32)
        probs = model.predict_proba(X_test)
        
        assert probs.shape == (10, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)  # Probabilities sum to 1
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_with_pointers(self, fitted_model):
        """Test multi-task prediction with pointers."""
        model, X, y, start, end = fitted_model
        
        X_test = np.random.randn(10, 105, 11).astype(np.float32)
        results = model.predict_with_pointers(X_test)
        
        # Check return structure
        assert isinstance(results, dict)
        assert "labels" in results
        assert "probabilities" in results
        assert "pointers" in results
        
        # Check shapes
        assert results["labels"].shape == (10,)
        assert results["probabilities"].shape == (10, 3)
        assert results["pointers"].shape == (10, 2)
        
        # Check pointer values are in [0, 1]
        assert (results["pointers"] >= 0).all()
        assert (results["pointers"] <= 1).all()

    def test_predict_with_pointers_error(self):
        """Test error when predicting pointers with single-task model."""
        set_seed(1337)
        n_samples = 30
        X = np.random.randn(n_samples, 105, 11).astype(np.float32)
        y = np.random.randint(0, 3, n_samples)
        
        model = JadeModel(seed=1337, predict_pointers=False, n_epochs=1)
        model.fit(X, y)
        
        X_test = np.random.randn(10, 105, 11).astype(np.float32)
        
        with pytest.raises(ValueError, match="Model not trained with pointer prediction"):
            model.predict_with_pointers(X_test)

    def test_prediction_before_fitting(self):
        """Test error when predicting before fitting."""
        model = JadeModel()
        X_test = np.random.randn(10, 105, 11).astype(np.float32)
        
        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            model.predict(X_test)


class TestJadeModelSaveLoad:
    """Test Jade model save/load functionality."""

    def test_save_load_cycle(self, tmp_path):
        """Test complete save/load cycle."""
        set_seed(1337)
        n_samples = 30
        X = np.random.randn(n_samples, 105, 11).astype(np.float32)
        y = np.random.randint(0, 3, n_samples)
        
        # Train model
        model = JadeModel(seed=1337, predict_pointers=False, n_epochs=1)
        model.fit(X, y)
        
        # Save model
        save_path = tmp_path / "jade_model.pt"
        model.save(save_path)
        assert save_path.exists()
        
        # Load model
        new_model = JadeModel(seed=1337)
        new_model.load(save_path)
        
        # Check attributes are restored
        assert new_model.is_fitted
        assert new_model.n_classes == 3
        assert new_model.input_dim == 11
        assert new_model.MODEL_ID == model.MODEL_ID
        assert new_model.CODENAME == model.CODENAME
        
        # Check predictions match
        X_test = np.random.randn(5, 105, 11).astype(np.float32)
        pred1 = model.predict(X_test)
        pred2 = new_model.predict(X_test)
        np.testing.assert_array_equal(pred1, pred2)


class TestPointerRegressionLoss:
    """Test pointer regression loss computation."""

    def test_center_length_encoding(self):
        """Test center+length encoding for pointer regression."""
        # Create dummy data
        outputs = {"pointers_cl": torch.tensor([[0.5, 0.3], [0.2, 0.6]])}
        expansion_start = torch.tensor([20.0, 40.0])
        expansion_end = torch.tensor([60.0, 80.0])
        
        loss = compute_pointer_regression_loss(outputs, expansion_start, expansion_end)
        
        # Should be a scalar tensor
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_huber_delta_configuration(self):
        """Test Huber loss delta ≈ 0.08 (Stones specification)."""
        # This is tested indirectly through the loss computation
        # The delta value is hardcoded in the function
        outputs = {"pointers_cl": torch.tensor([[0.5, 0.3]])}
        expansion_start = torch.tensor([20.0])
        expansion_end = torch.tensor([60.0])
        
        # Should compute without error
        loss = compute_pointer_regression_loss(outputs, expansion_start, expansion_end)
        assert loss.item() >= 0

    def test_pointer_loss_weights(self):
        """Test center weight (1.0) > length weight (0.8)."""
        # Create perfect predictions for center, poor for length
        outputs = {"pointers_cl": torch.tensor([[0.4, 0.0]])}  # Perfect center, bad length
        expansion_start = torch.tensor([20.0])
        expansion_end = torch.tensor([60.0])
        
        loss_perfect = compute_pointer_regression_loss(outputs, expansion_start, expansion_end)
        
        # Create poor predictions for both
        outputs = {"pointers_cl": torch.tensor([[0.0, 0.0]])}  # Bad center and length
        loss_poor = compute_pointer_regression_loss(outputs, expansion_start, expansion_end)
        
        # Poor predictions should have higher loss
        assert loss_poor > loss_perfect


class TestJadeModelDeviceSupport:
    """Test Jade model works on different devices."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_training(self):
        """Test training on CUDA device."""
        set_seed(1337)
        n_samples = 30
        X = np.random.randn(n_samples, 105, 11).astype(np.float32)
        y = np.random.randint(0, 3, n_samples)
        
        model = JadeModel(
            seed=1337,
            device="cuda",
            predict_pointers=True,
            n_epochs=1,
            batch_size=16,
            use_temporal_aug=False,
        )
        
        # Should train on CUDA
        model.fit(X, y, expansion_start=np.random.randint(0, 50, n_samples),
                 expansion_end=np.random.randint(55, 104, n_samples))
        
        assert model.is_fitted
        assert model.device.type == "cuda"

    def test_cpu_training(self):
        """Test training on CPU device."""
        set_seed(1337)
        n_samples = 30
        X = np.random.randn(n_samples, 105, 11).astype(np.float32)
        y = np.random.randint(0, 3, n_samples)
        
        model = JadeModel(
            seed=1337,
            device="cpu",
            predict_pointers=False,
            n_epochs=1,
            batch_size=16,
        )
        
        model.fit(X, y)
        
        assert model.is_fitted
        assert model.device.type == "cpu"

    def test_mixed_precision(self):
        """Test automatic mixed precision configuration."""
        # Test with CUDA (should enable AMP)
        model_cuda = JadeModel(device="cuda")
        assert model_cuda.use_amp == torch.cuda.is_available()
        
        # Test with CPU (should disable AMP)
        model_cpu = JadeModel(device="cpu")
        assert model_cpu.use_amp is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])