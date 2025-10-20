"""Unit tests for bidirectional masked LSTM autoencoder.

Tests the complete pre-training pipeline:
1. Model architecture (bidirectional LSTM encoder + decoder)
2. Three masking strategies (random, block, patch)
3. Weight transfer to SimpleLSTM (bidirectional → unidirectional)
4. Pre-training infrastructure (MaskedLSTMPretrainer)
5. Integration with SimpleLSTM fine-tuning
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from moola.models.bilstm_masked_autoencoder import (
    BiLSTMMaskedAutoencoder,
    MaskingStrategy,
    apply_masking,
)
from moola.pretraining.data_augmentation import TimeSeriesAugmenter
from moola.pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer


class TestBiLSTMMaskedAutoencoder:
    """Test BiLSTMMaskedAutoencoder model architecture."""

    def test_model_initialization(self):
        """Test model builds with correct architecture."""
        model = BiLSTMMaskedAutoencoder(
            input_dim=4,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2
        )

        # Check encoder is bidirectional
        assert model.encoder_lstm.bidirectional is True
        assert model.encoder_lstm.hidden_size == 128
        assert model.encoder_lstm.num_layers == 2

        # Check mask token is learnable
        assert model.mask_token.requires_grad is True
        assert model.mask_token.shape == (1, 1, 4)

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = BiLSTMMaskedAutoencoder(input_dim=4, hidden_dim=128)

        # Create dummy masked input [batch=8, seq_len=105, features=4]
        x_masked = torch.randn(8, 105, 4)

        # Forward pass
        reconstruction = model(x_masked)

        # Check output shape
        assert reconstruction.shape == (8, 105, 4)

    def test_loss_computation(self):
        """Test masked reconstruction loss computation."""
        model = BiLSTMMaskedAutoencoder(input_dim=4, hidden_dim=128)

        # Create dummy data
        x_original = torch.randn(8, 105, 4)
        x_masked = x_original.clone()
        mask = torch.rand(8, 105) < 0.15  # 15% masked

        # Replace masked positions with mask token
        x_masked[mask] = model.mask_token.squeeze(0).expand(mask.sum(), 4)

        # Forward pass
        reconstruction = model(x_masked)

        # Compute loss
        total_loss, loss_dict = model.compute_loss(reconstruction, x_original, mask)

        # Verify loss components
        assert 'total' in loss_dict
        assert 'reconstruction' in loss_dict
        assert 'regularization' in loss_dict
        assert 'latent_std' in loss_dict
        assert 'num_masked' in loss_dict

        # Check loss is scalar tensor
        assert total_loss.ndim == 0
        assert total_loss.item() > 0


class TestMaskingStrategies:
    """Test all three masking strategies."""

    def test_random_masking(self):
        """Test BERT-style random masking."""
        x = torch.randn(8, 105, 4)
        mask_token = torch.randn(1, 1, 4)

        x_masked, mask = MaskingStrategy.mask_random(x, mask_token, mask_ratio=0.15)

        # Check shapes
        assert x_masked.shape == x.shape
        assert mask.shape == (8, 105)

        # Check mask ratio approximately 15%
        actual_ratio = mask.float().mean().item()
        assert 0.10 < actual_ratio < 0.20

        # Check masked positions replaced
        assert not torch.allclose(x_masked[mask], x[mask])

    def test_block_masking(self):
        """Test contiguous block masking."""
        x = torch.randn(8, 105, 4)
        mask_token = torch.randn(1, 1, 4)

        x_masked, mask = MaskingStrategy.mask_block(x, mask_token, mask_ratio=0.15)

        # Check shapes
        assert x_masked.shape == x.shape
        assert mask.shape == (8, 105)

        # For each sample, verify mask is contiguous
        for i in range(8):
            mask_indices = torch.where(mask[i])[0]
            if len(mask_indices) > 1:
                # Check indices are consecutive
                diffs = mask_indices[1:] - mask_indices[:-1]
                assert torch.all(diffs == 1)

    def test_patch_masking(self):
        """Test patch-level masking (PatchTST-inspired)."""
        x = torch.randn(8, 105, 4)
        mask_token = torch.randn(1, 1, 4)

        x_masked, mask = MaskingStrategy.mask_patch(
            x, mask_token, mask_ratio=0.15, patch_size=7
        )

        # Check shapes
        assert x_masked.shape == x.shape
        assert mask.shape == (8, 105)

        # Verify masking occurs in patch-sized blocks
        # 105 timesteps / 7 patch_size = 15 patches
        # 15% of 15 patches ≈ 2 patches masked = 14 timesteps
        num_patches = 105 // 7
        expected_masked_per_sample = int(num_patches * 0.15) * 7

        for i in range(8):
            num_masked = mask[i].sum().item()
            # Allow some tolerance (±1 patch)
            assert abs(num_masked - expected_masked_per_sample) <= 7

    def test_apply_masking_wrapper(self):
        """Test apply_masking wrapper function."""
        x = torch.randn(4, 105, 4)
        mask_token = torch.randn(1, 1, 4)

        # Test all strategies
        for strategy in ["random", "block", "patch"]:
            x_masked, mask = apply_masking(
                x, mask_token,
                mask_strategy=strategy,
                mask_ratio=0.15,
                patch_size=7
            )

            assert x_masked.shape == x.shape
            assert mask.shape == (4, 105)
            assert mask.any()


class TestPreTraining:
    """Test pre-training pipeline."""

    def test_pretrainer_initialization(self):
        """Test MaskedLSTMPretrainer initializes correctly."""
        pretrainer = MaskedLSTMPretrainer(
            input_dim=4,
            hidden_dim=128,
            num_layers=2,
            mask_ratio=0.15,
            mask_strategy="patch",
            device="cpu"
        )

        assert pretrainer.model is not None
        assert pretrainer.optimizer is not None
        assert pretrainer.mask_strategy == "patch"

    def test_pretrain_single_epoch(self):
        """Test pre-training runs for one epoch."""
        # Create small synthetic dataset
        X_unlabeled = np.random.randn(100, 105, 4).astype(np.float32)

        pretrainer = MaskedLSTMPretrainer(
            input_dim=4,
            hidden_dim=64,  # Smaller for faster test
            num_layers=1,
            mask_ratio=0.15,
            mask_strategy="random",
            batch_size=32,
            device="cpu"
        )

        # Train for 1 epoch
        history = pretrainer.pretrain(
            X_unlabeled,
            n_epochs=1,
            val_split=0.2,
            patience=10,
            verbose=False
        )

        # Check history has expected keys
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'train_recon' in history
        assert 'val_recon' in history

        # Check losses are reasonable
        assert len(history['train_loss']) == 1
        assert history['train_loss'][0] > 0
        assert history['val_loss'][0] > 0

    def test_encoder_save_load(self):
        """Test encoder saving and loading."""
        pretrainer = MaskedLSTMPretrainer(
            input_dim=4,
            hidden_dim=64,
            num_layers=1,
            device="cpu"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "encoder.pt"

            # Save encoder
            pretrainer.save_encoder(save_path)
            assert save_path.exists()

            # Load encoder in new instance
            pretrainer2 = MaskedLSTMPretrainer(
                input_dim=4,
                hidden_dim=64,
                num_layers=1,
                device="cpu"
            )
            pretrainer2.load_encoder(save_path)

            # Verify weights match
            state1 = pretrainer.model.encoder_lstm.state_dict()
            state2 = pretrainer2.model.encoder_lstm.state_dict()

            for key in state1:
                assert torch.allclose(state1[key], state2[key])


class TestWeightTransfer:
    """Test bidirectional → unidirectional weight transfer."""

    def test_weight_shape_compatibility(self):
        """Test bidirectional and unidirectional LSTM weight shapes."""
        # Bidirectional LSTM (encoder)
        bilstm = torch.nn.LSTM(
            input_size=4,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Unidirectional LSTM (SimpleLSTM)
        unilstm = torch.nn.LSTM(
            input_size=4,
            hidden_size=128,
            num_layers=2,
            bidirectional=False,
            batch_first=True
        )

        # Check layer 0 weight shapes
        # PyTorch stores bidirectional weights separately with "_reverse" suffix
        bi_weight_ih_fwd = bilstm.state_dict()['weight_ih_l0']  # Forward direction
        bi_weight_ih_bwd = bilstm.state_dict()['weight_ih_l0_reverse']  # Backward direction
        uni_weight_ih = unilstm.state_dict()['weight_ih_l0']

        # Both bidirectional and unidirectional have same shape: [hidden*4, input]
        assert bi_weight_ih_fwd.shape == (128 * 4, 4)
        assert bi_weight_ih_bwd.shape == (128 * 4, 4)
        assert uni_weight_ih.shape == (128 * 4, 4)

        # Forward weights are directly compatible
        assert bi_weight_ih_fwd.shape == uni_weight_ih.shape

    def test_bidirectional_to_unidirectional_transfer(self):
        """Test full weight transfer from bidirectional to unidirectional."""
        from moola.models.simple_lstm import SimpleLSTMModel

        # Create SimpleLSTM model
        model = SimpleLSTMModel(
            hidden_size=128,
            num_layers=2,
            device="cpu"
        )

        # Build model with dummy data
        X_dummy = np.random.randn(10, 105, 4).astype(np.float32)
        y_dummy = np.array(['A', 'B'] * 5)
        model.fit(X_dummy, y_dummy, unfreeze_encoder_after=0)

        # Create pre-trained encoder
        pretrainer = MaskedLSTMPretrainer(
            input_dim=4,
            hidden_dim=128,
            num_layers=2,
            device="cpu"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            encoder_path = Path(tmpdir) / "encoder.pt"
            pretrainer.save_encoder(encoder_path)

            # Load pre-trained weights
            model.load_pretrained_encoder(encoder_path, freeze_encoder=True)

            # Verify LSTM parameters are frozen
            for param in model.model.lstm.parameters():
                assert param.requires_grad is False

            # Verify classifier is still trainable
            for param in model.model.classifier.parameters():
                assert param.requires_grad is True


class TestDataAugmentation:
    """Test data augmentation for unlabeled data generation."""

    def test_augmenter_initialization(self):
        """Test TimeSeriesAugmenter initializes with correct parameters."""
        augmenter = TimeSeriesAugmenter(
            time_warp_prob=0.5,
            jitter_prob=0.5,
            volatility_scale_prob=0.3
        )

        assert augmenter.time_warp_prob == 0.5
        assert augmenter.jitter_prob == 0.5
        assert augmenter.volatility_scale_prob == 0.3

    def test_time_warp_augmentation(self):
        """Test time warping preserves shape."""
        augmenter = TimeSeriesAugmenter()
        X = np.random.randn(10, 105, 4).astype(np.float32)

        X_warped = augmenter.time_warp(X, sigma=0.12)

        assert X_warped.shape == X.shape
        assert not np.allclose(X_warped, X)

    def test_jitter_augmentation(self):
        """Test jittering adds noise."""
        augmenter = TimeSeriesAugmenter()
        X = np.random.randn(10, 105, 4).astype(np.float32)

        X_jittered = augmenter.jitter(X, sigma=0.05)

        assert X_jittered.shape == X.shape
        assert not np.allclose(X_jittered, X)

    def test_volatility_scaling(self):
        """Test volatility scaling preserves OHLC constraints."""
        augmenter = TimeSeriesAugmenter()

        # Create OHLC data with valid constraints
        N, T = 10, 105
        O = np.random.randn(N, T)
        H = O + np.abs(np.random.randn(N, T))
        L = O - np.abs(np.random.randn(N, T))
        C = np.random.uniform(L, H)
        X = np.stack([O, H, L, C], axis=-1).astype(np.float32)

        X_scaled = augmenter.volatility_scale(X, scale_range=(0.85, 1.15))

        # Verify OHLC constraints still hold
        for i in range(N):
            for t in range(T):
                o, h, l, c = X_scaled[i, t]
                assert h >= max(o, c), f"High < max(Open, Close) at ({i}, {t})"
                assert l <= min(o, c), f"Low > min(Open, Close) at ({i}, {t})"

    def test_augment_dataset(self):
        """Test dataset augmentation multiplies data."""
        augmenter = TimeSeriesAugmenter()
        X = np.random.randn(100, 105, 4).astype(np.float32)

        X_augmented = augmenter.augment_dataset(X, num_augmentations=4)

        # Should be original + 4 augmented = 5x size
        assert X_augmented.shape == (500, 105, 4)


class TestRTX4090Optimization:
    """Test RTX 4090 hardware optimizations."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_training(self):
        """Test FP16 mixed precision training works."""
        X_unlabeled = np.random.randn(64, 105, 4).astype(np.float32)

        pretrainer = MaskedLSTMPretrainer(
            input_dim=4,
            hidden_dim=128,
            num_layers=2,
            batch_size=512,  # Large batch for RTX 4090
            device="cuda"
        )

        # Train for 1 epoch
        history = pretrainer.pretrain(
            X_unlabeled,
            n_epochs=1,
            verbose=False
        )

        assert history['train_loss'][0] > 0

    def test_batch_size_memory_estimate(self):
        """Test batch size fits in 24GB VRAM (RTX 4090)."""
        # Estimate VRAM usage for batch_size=512
        # Model parameters: ~1.5M params × 4 bytes = ~6 MB
        # Activations per sample: 105 × 128 × 2 (bidirectional) × 4 bytes = ~100 KB
        # Batch activations: 512 samples × 100 KB = ~51 MB
        # Gradients: ~6 MB (same as parameters)
        # Optimizer states (AdamW): ~12 MB (2x parameters)
        # Total: ~75 MB per batch (well within 24 GB)

        model = BiLSTMMaskedAutoencoder(input_dim=4, hidden_dim=128, num_layers=2)
        total_params = sum(p.numel() for p in model.parameters())

        # Check parameter count is reasonable
        assert total_params < 2_000_000  # < 2M parameters

        # Estimate memory per batch (rough calculation)
        bytes_per_param = 4  # FP32
        param_memory_mb = (total_params * bytes_per_param) / (1024 ** 2)
        activation_memory_mb = (512 * 105 * 128 * 2 * 4) / (1024 ** 2)
        total_memory_mb = param_memory_mb * 3 + activation_memory_mb  # params + grads + optimizer

        # Should fit comfortably in 24GB
        assert total_memory_mb < 1000  # < 1 GB per batch


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pretrain_finetune_pipeline(self):
        """Test complete pipeline: pretrain → transfer → fine-tune."""
        from moola.models.simple_lstm import SimpleLSTMModel

        # Step 1: Create unlabeled data
        X_unlabeled = np.random.randn(100, 105, 4).astype(np.float32)

        # Step 2: Pre-train encoder
        pretrainer = MaskedLSTMPretrainer(
            input_dim=4,
            hidden_dim=64,
            num_layers=1,
            mask_strategy="patch",
            batch_size=32,
            device="cpu"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            encoder_path = Path(tmpdir) / "encoder.pt"

            # Pre-train for 2 epochs
            history = pretrainer.pretrain(
                X_unlabeled,
                n_epochs=2,
                save_path=encoder_path,
                verbose=False
            )

            assert encoder_path.exists()
            assert len(history['train_loss']) == 2

            # Step 3: Create labeled data
            X_labeled = np.random.randn(50, 105, 4).astype(np.float32)
            y_labeled = np.array(['class_A', 'class_B'] * 25)

            # Step 4: Fine-tune SimpleLSTM with pre-trained encoder
            model = SimpleLSTMModel(
                hidden_size=64,
                num_layers=1,
                n_epochs=2,
                device="cpu"
            )

            # Build model first
            model.fit(X_labeled[:10], y_labeled[:10], unfreeze_encoder_after=0)

            # Load pre-trained encoder
            model.load_pretrained_encoder(encoder_path, freeze_encoder=True)

            # Fine-tune with encoder frozen for 1 epoch, then unfreeze
            model.fit(X_labeled, y_labeled, unfreeze_encoder_after=1)

            # Verify model can predict
            predictions = model.predict(X_labeled[:5])
            assert len(predictions) == 5

    def test_cli_pretrain_command_simulation(self):
        """Test CLI command flow (without actually calling CLI)."""
        # This simulates what the CLI command does
        X_unlabeled = np.random.randn(100, 105, 4).astype(np.float32)

        # Initialize with CLI-like parameters
        pretrainer = MaskedLSTMPretrainer(
            input_dim=4,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            mask_ratio=0.15,
            mask_strategy="patch",
            patch_size=7,
            learning_rate=1e-3,
            batch_size=512,
            device="cpu",
            seed=1337
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "artifacts" / "pretrained" / "bilstm_encoder.pt"

            history = pretrainer.pretrain(
                X_unlabeled=X_unlabeled,
                n_epochs=2,
                val_split=0.1,
                patience=10,
                save_path=output_path,
                verbose=False
            )

            assert output_path.exists()
            assert 'train_loss' in history
            assert 'val_loss' in history


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
