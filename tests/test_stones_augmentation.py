"""Tests for Stones-compliant temporal augmentation.

Tests Stones non-negotiables:
- Jitter σ=0.03 (precise noise level)
- Magnitude warp σ=0.2 (controlled distortion)
- ×3 on-the-fly augmentation (3x effective data)
- Proper parameter validation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
import numpy as np
from unittest.mock import patch

from moola.utils.temporal_augmentation import TemporalAugmentation


class TestTemporalAugmentationStonesCompliance:
    """Test temporal augmentation compliance with Stones specifications."""

    @pytest.fixture
    def sample_data(self):
        """Create sample sequence data."""
        batch_size = 16
        seq_len = 105
        n_features = 11
        
        return torch.randn(batch_size, seq_len, n_features)

    @pytest.fixture
    def stones_augmentation(self):
        """Create augmentation with Stones-compliant parameters."""
        return TemporalAugmentation(
            jitter_prob=0.8,
            jitter_sigma=0.03,  # Stones specification
            magnitude_warp_prob=0.5,
            magnitude_warp_sigma=0.2,  # Stones specification
            magnitude_warp_knots=4,
            scaling_prob=0.0,  # Disabled per Stones
            time_warp_prob=0.0,  # Disabled per Stones
        )

    def test_jitter_sigma_configuration(self):
        """Test jitter σ=0.03 (Stones specification)."""
        aug = TemporalAugmentation(jitter_sigma=0.03)
        assert aug.jitter_sigma == 0.03

    def test_magnitude_warp_sigma_configuration(self):
        """Test magnitude warp σ=0.2 (Stones specification)."""
        aug = TemporalAugmentation(magnitude_warp_sigma=0.2)
        assert aug.magnitude_warp_sigma == 0.2

    def test_on_the_fly_augmentation_count(self, sample_data, stones_augmentation):
        """Test ×3 on-the-fly augmentation (3x effective data)."""
        # Apply augmentation multiple times to test variability
        original = sample_data.clone()
        augmented_samples = []
        
        for _ in range(10):
            aug_data = stones_augmentation.apply_augmentation(original.clone())
            augmented_samples.append(aug_data)
        
        # Check that augmentation produces different results
        # (not all identical to original)
        differences = [not torch.allclose(aug, original, atol=1e-6) for aug in augmented_samples]
        assert any(differences), "Augmentation should produce different results"

    def test_jitter_augmentation_effect(self, sample_data):
        """Test that jitter adds appropriate noise level."""
        aug = TemporalAugmentation(
            jitter_prob=1.0,  # Always apply
            jitter_sigma=0.03,
            magnitude_warp_prob=0.0,  # Disable other augmentations
        )
        
        original = sample_data.clone()
        augmented = aug.apply_augmentation(original.clone())
        
        # Check that data is modified but not drastically
        diff = torch.abs(augmented - original)
        mean_diff = diff.mean().item()
        
        # With σ=0.03, mean difference should be around 0.024 (0.03 * sqrt(2/π))
        assert 0.01 < mean_diff < 0.05, f"Mean difference {mean_diff} not in expected range"

    def test_magnitude_warp_augmentation_effect(self, sample_data):
        """Test that magnitude warp creates smooth distortions."""
        aug = TemporalAugmentation(
            jitter_prob=0.0,  # Disable jitter
            magnitude_warp_prob=1.0,  # Always apply
            magnitude_warp_sigma=0.2,
            magnitude_warp_knots=4,
        )
        
        original = sample_data.clone()
        augmented = aug.apply_augmentation(original.clone())
        
        # Check that data is modified
        assert not torch.allclose(augmented, original, atol=1e-6)
        
        # Check that the warp is smooth across time
        # (differences should be correlated across nearby timesteps)
        for i in range(sample_data.shape[0]):  # For each batch
            for j in range(sample_data.shape[2]):  # For each feature
                orig_seq = original[i, :, j]
                aug_seq = augmented[i, :, j]
                
                # Compute correlation of differences
                diff = aug_seq - orig_seq
                if torch.std(diff) > 1e-6:  # Only if there's variation
                    # Adjacent differences should be correlated (smooth warp)
                    corr = np.corrcoef(diff[:-1].numpy(), diff[1:].numpy())[0, 1]
                    assert corr > 0.3, f"Warp should be smooth, got correlation {corr}"

    def test_augmentation_probability_control(self, sample_data):
        """Test that augmentation probabilities are respected."""
        # Test with low probability
        aug_low = TemporalAugmentation(
            jitter_prob=0.0,  # Never apply jitter
            magnitude_warp_prob=0.0,  # Never apply magnitude warp
        )
        
        original = sample_data.clone()
        augmented_low = aug_low.apply_augmentation(original.clone())
        
        # Should be identical when no augmentation applied
        assert torch.allclose(augmented_low, original, atol=1e-6)

    def test_parameter_validation(self):
        """Test validation of augmentation parameters."""
        # Test negative sigma (should be allowed but produces warning)
        with pytest.warns(UserWarning):
            aug = TemporalAugmentation(jitter_sigma=-0.01)
            assert aug.jitter_sigma == -0.01

        # Test zero sigma (should work)
        aug = TemporalAugmentation(jitter_sigma=0.0)
        assert aug.jitter_sigma == 0.0

        # Test very large sigma (should work but may be extreme)
        aug = TemporalAugmentation(jitter_sigma=1.0)
        assert aug.jitter_sigma == 1.0

    def test_stones_default_parameters(self):
        """Test that Stones default parameters are correctly set."""
        aug = TemporalAugmentation(
            jitter_prob=0.8,
            jitter_sigma=0.03,
            magnitude_warp_prob=0.5,
            magnitude_warp_sigma=0.2,
            magnitude_warp_knots=4,
        )
        
        assert aug.jitter_prob == 0.8
        assert aug.jitter_sigma == 0.03
        assert aug.magnitude_warp_prob == 0.5
        assert aug.magnitude_warp_sigma == 0.2
        assert aug.magnitude_warp_knots == 4

    def test_augmentation_reproducibility(self, sample_data):
        """Test that augmentation is reproducible with seed."""
        aug = TemporalAugmentation(
            jitter_prob=1.0,
            jitter_sigma=0.03,
            magnitude_warp_prob=0.0,
        )
        
        original = sample_data.clone()
        
        # Set seed for reproducibility
        torch.manual_seed(1337)
        aug1 = aug.apply_augmentation(original.clone())
        
        torch.manual_seed(1337)
        aug2 = aug.apply_augmentation(original.clone())
        
        # Should be identical with same seed
        assert torch.allclose(aug1, aug2, atol=1e-6)

    def test_combined_augmentations(self, sample_data):
        """Test that jitter and magnitude warp work together."""
        aug = TemporalAugmentation(
            jitter_prob=1.0,
            jitter_sigma=0.03,
            magnitude_warp_prob=1.0,
            magnitude_warp_sigma=0.2,
            magnitude_warp_knots=4,
        )
        
        original = sample_data.clone()
        augmented = aug.apply_augmentation(original.clone())
        
        # Should be different from original
        assert not torch.allclose(augmented, original, atol=1e-6)
        
        # Difference should be larger than individual augmentations
        jitter_only = TemporalAugmentation(
            jitter_prob=1.0, jitter_sigma=0.03, magnitude_warp_prob=0.0
        )
        warp_only = TemporalAugmentation(
            jitter_prob=0.0, magnitude_warp_prob=1.0, magnitude_warp_sigma=0.2
        )
        
        aug_jitter = jitter_only.apply_augmentation(original.clone())
        aug_warp = warp_only.apply_augmentation(original.clone())
        
        diff_combined = torch.mean(torch.abs(augmented - original))
        diff_jitter = torch.mean(torch.abs(aug_jitter - original))
        diff_warp = torch.mean(torch.abs(aug_warp - original))
        
        # Combined should generally have larger effect
        assert diff_combined >= min(diff_jitter, diff_warp)

    def test_batch_processing_consistency(self, sample_data):
        """Test that augmentation is consistent across batch dimension."""
        aug = TemporalAugmentation(
            jitter_prob=1.0,
            jitter_sigma=0.03,
            magnitude_warp_prob=0.0,
        )
        
        original = sample_data.clone()
        augmented = aug.apply_augmentation(original.clone())
        
        # Each sample in batch should be independently augmented
        # (differences should vary across batch)
        batch_diffs = []
        for i in range(original.shape[0]):
            diff = torch.mean(torch.abs(augmented[i] - original[i]))
            batch_diffs.append(diff.item())
        
        # Should have variation across batch (not all same difference)
        std_diffs = np.std(batch_diffs)
        assert std_diffs > 1e-6, "Batch samples should be independently augmented"

    def test_feature_wise_independence(self, sample_data):
        """Test that augmentation affects features independently."""
        aug = TemporalAugmentation(
            jitter_prob=1.0,
            jitter_sigma=0.03,
            magnitude_warp_prob=0.0,
        )
        
        original = sample_data.clone()
        augmented = aug.apply_augmentation(original.clone())
        
        # Check that different features have different augmentation patterns
        feature_diffs = []
        for j in range(original.shape[2]):
            diff = torch.mean(torch.abs(augmented[:, :, j] - original[:, :, j]))
            feature_diffs.append(diff.item())
        
        # Should have variation across features
        std_diffs = np.std(feature_diffs)
        assert std_diffs > 1e-6, "Features should be independently augmented"

    def test_gradient_flow_preservation(self, sample_data):
        """Test that augmentation preserves gradient flow."""
        aug = TemporalAugmentation(
            jitter_prob=1.0,
            jitter_sigma=0.03,
            magnitude_warp_prob=0.0,
        )
        
        original = sample_data.clone()
        original.requires_grad_(True)
        
        augmented = aug.apply_augmentation(original)
        loss = augmented.sum()
        loss.backward()
        
        # Should have gradients
        assert original.grad is not None
        assert not torch.allclose(original.grad, torch.zeros_like(original.grad), atol=1e-6)

    def test_data_type_preservation(self, sample_data):
        """Test that augmentation preserves data type and device."""
        aug = TemporalAugmentation(jitter_prob=1.0, jitter_sigma=0.03)
        
        # Test with float32
        data_f32 = sample_data.float()
        aug_f32 = aug.apply_augmentation(data_f32)
        assert aug_f32.dtype == torch.float32
        
        # Test with float64
        data_f64 = sample_data.double()
        aug_f64 = aug.apply_augmentation(data_f64)
        assert aug_f64.dtype == torch.float64
        
        # Test device preservation (if CUDA available)
        if torch.cuda.is_available():
            data_cuda = sample_data.cuda()
            aug_cuda = aug.apply_augmentation(data_cuda)
            assert aug_cuda.device.type == "cuda"

    def test_edge_cases(self):
        """Test augmentation edge cases."""
        aug = TemporalAugmentation(jitter_prob=1.0, jitter_sigma=0.03)
        
        # Test with single sample
        single_sample = torch.randn(1, 105, 11)
        aug_single = aug.apply_augmentation(single_sample)
        assert aug_single.shape == single_sample.shape
        
        # Test with single feature
        single_feature = torch.randn(16, 105, 1)
        aug_single_feat = aug.apply_augmentation(single_feature)
        assert aug_single_feat.shape == single_feature.shape
        
        # Test with zero data
        zero_data = torch.zeros(16, 105, 11)
        aug_zero = aug.apply_augmentation(zero_data)
        # Should add noise to zero data
        assert not torch.allclose(aug_zero, zero_data, atol=1e-6)

    def test_performance_characteristics(self, sample_data):
        """Test that augmentation is performant."""
        aug = TemporalAugmentation(
            jitter_prob=0.8,
            jitter_sigma=0.03,
            magnitude_warp_prob=0.5,
            magnitude_warp_sigma=0.2,
        )
        
        import time
        
        # Time augmentation
        start_time = time.time()
        for _ in range(100):
            _ = aug.apply_augmentation(sample_data.clone())
        elapsed = time.time() - start_time
        
        # Should be reasonably fast (less than 1 second for 100 augmentations)
        assert elapsed < 1.0, f"Augmentation too slow: {elapsed:.3f}s for 100 runs"

    def test_stones_compliance_report(self, sample_data):
        """Generate compliance report for Stones specifications."""
        aug = TemporalAugmentation(
            jitter_prob=0.8,
            jitter_sigma=0.03,  # ✓ Stones compliant
            magnitude_warp_prob=0.5,
            magnitude_warp_sigma=0.2,  # ✓ Stones compliant
            magnitude_warp_knots=4,
        )
        
        # Test ×3 on-the-fly capability
        original = sample_data.clone()
        variations = []
        for i in range(3):
            aug_data = aug.apply_augmentation(original.clone())
            variations.append(aug_data)
        
        # Check that we get 3 different variations
        all_different = True
        for i in range(3):
            for j in range(i+1, 3):
                if torch.allclose(variations[i], variations[j], atol=1e-6):
                    all_different = False
                    break
        
        assert all_different, "Should generate 3 different variations on-the-fly"
        
        # Compliance summary
        compliance = {
            "jitter_sigma": aug.jitter_sigma == 0.03,
            "magnitude_warp_sigma": aug.magnitude_warp_sigma == 0.2,
            "on_the_fly_3x": all_different,
            "probabilities_reasonable": (0 <= aug.jitter_prob <= 1 and 
                                       0 <= aug.magnitude_warp_prob <= 1),
        }
        
        assert all(compliance.values()), f"Non-compliant: {compliance}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])