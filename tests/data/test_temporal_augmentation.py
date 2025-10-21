"""Unit tests for temporal augmentation (Phase 2 optimizations).

Tests temporal jittering and magnitude warping to ensure:
1. Pattern preservation (correlation > 0.95 for jitter, > 0.90 for warp)
2. Shape preservation
3. Augmentation actually modifies data
4. Combined augmentation pipeline works correctly
"""

import numpy as np
import pytest
import torch

from moola.utils.augmentation.temporal_augmentation import (
    augment_temporal_sequence,
    jitter,
    jitter_numpy,
    magnitude_warp,
    magnitude_warp_scipy,
    validate_jitter_preserves_patterns,
)

# Import standalone functions from data.temporal_augmentation module
from moola.data.temporal_augmentation import (
    add_jitter,
    augment_temporal,
    magnitude_warp as magnitude_warp_standalone,
    validate_augmentation,
)


class TestJitter:
    """Test temporal jittering augmentation."""

    def test_jitter_shape_preservation(self):
        """Test that jittering preserves tensor shape."""
        x = torch.randn(8, 105, 11)  # [B, T, F]
        x_jittered = jitter(x, sigma=0.03)
        assert x_jittered.shape == x.shape, f"Expected shape {x.shape}, got {x_jittered.shape}"

    def test_jitter_unbatched_shape(self):
        """Test jitter with unbatched input [T, F]."""
        x = torch.randn(105, 11)
        x_jittered = jitter(x, sigma=0.03)
        assert x_jittered.shape == x.shape

    def test_jitter_correlation_threshold(self):
        """Test that jittering preserves pattern correlation > 0.95."""
        torch.manual_seed(42)
        x = torch.randn(1, 105, 11)

        results = validate_jitter_preserves_patterns(x, sigma=0.03, n_samples=10)

        assert results['passes_threshold'], \
            f"Jitter failed correlation test: avg={results['avg_correlation']:.3f}, min={results['min_correlation']:.3f}"
        assert results['avg_correlation'] > 0.95, \
            f"Average correlation {results['avg_correlation']:.3f} below 0.95 threshold"

    def test_jitter_actually_modifies_data(self):
        """Test that jittering actually changes the data."""
        torch.manual_seed(42)
        x = torch.randn(8, 105, 11)
        x_jittered = jitter(x, sigma=0.03)

        # Should not be exactly equal
        assert not torch.allclose(x, x_jittered, rtol=0.01), \
            "Jittering did not modify data"

        # But should be close (within a few sigmas)
        diff = (x - x_jittered).abs().mean()
        assert diff < 0.1, f"Jittering modified data too much: mean diff = {diff:.4f}"

    def test_jitter_numpy_version(self):
        """Test NumPy version of jitter."""
        np.random.seed(42)
        x = np.random.randn(8, 105, 11).astype(np.float32)
        x_jittered = jitter_numpy(x, sigma=0.03)

        assert x_jittered.shape == x.shape
        assert not np.allclose(x, x_jittered, rtol=0.01)

    def test_jitter_different_sigma_values(self):
        """Test jittering with different sigma values."""
        torch.manual_seed(42)
        x = torch.randn(4, 105, 11)

        x_small = jitter(x, sigma=0.01)
        x_large = jitter(x, sigma=0.10)

        diff_small = (x - x_small).abs().mean()
        diff_large = (x - x_large).abs().mean()

        assert diff_small < diff_large, \
            f"Larger sigma should cause larger perturbations: {diff_small:.4f} vs {diff_large:.4f}"


class TestMagnitudeWarp:
    """Test magnitude warping augmentation."""

    def test_magnitude_warp_shape_preservation(self):
        """Test that warping preserves tensor shape."""
        torch.manual_seed(42)
        x = torch.randn(8, 105, 11)
        x_warped = magnitude_warp(x, sigma=0.2, n_knots=4, prob=1.0)
        assert x_warped.shape == x.shape, f"Expected shape {x.shape}, got {x_warped.shape}"

    def test_magnitude_warp_unbatched_shape(self):
        """Test magnitude warp with unbatched input [T, F]."""
        torch.manual_seed(42)
        x = torch.randn(105, 11)
        x.requires_grad = True  # Simulate training mode
        x_warped = magnitude_warp(x, sigma=0.2, n_knots=4, prob=1.0)
        assert x_warped.shape == x.shape

    def test_magnitude_warp_actually_modifies_data(self):
        """Test that warping actually changes the data."""
        torch.manual_seed(42)
        x = torch.randn(8, 105, 11)
        x.requires_grad = True  # Simulate training mode
        x_warped = magnitude_warp(x, sigma=0.2, n_knots=4, prob=1.0)

        # Should not be exactly equal
        assert not torch.allclose(x, x_warped, rtol=0.01), \
            "Magnitude warping did not modify data"

    def test_magnitude_warp_probability(self):
        """Test that prob=0.0 returns original data."""
        torch.manual_seed(42)
        x = torch.randn(8, 105, 11)
        x.requires_grad = True
        x_warped = magnitude_warp(x, sigma=0.2, n_knots=4, prob=0.0)

        assert torch.allclose(x, x_warped), \
            "magnitude_warp with prob=0.0 should return original data"

    def test_magnitude_warp_scipy_version(self):
        """Test SciPy version with true cubic spline."""
        np.random.seed(42)
        x = np.random.randn(8, 105, 11).astype(np.float32)
        x_warped = magnitude_warp_scipy(x, sigma=0.2, n_knots=4)

        assert x_warped.shape == x.shape
        assert not np.allclose(x, x_warped, rtol=0.01)

    def test_magnitude_warp_scipy_unbatched(self):
        """Test SciPy version with unbatched input."""
        np.random.seed(42)
        x = np.random.randn(105, 11).astype(np.float32)
        x_warped = magnitude_warp_scipy(x, sigma=0.2, n_knots=4)

        assert x_warped.shape == x.shape

    def test_magnitude_warp_smoothness(self):
        """Test that magnitude warp creates smooth scaling curves."""
        torch.manual_seed(42)
        x = torch.ones(1, 105, 1)  # Constant signal
        x.requires_grad = True

        x_warped = magnitude_warp(x, sigma=0.2, n_knots=4, prob=1.0)

        # Extract the warp curve (since original is 1.0)
        warp_curve = x_warped[0, :, 0]

        # Check that curve is smooth (no large jumps)
        diffs = torch.diff(warp_curve).abs()
        max_diff = diffs.max().item()

        # With 4 knots and linear interpolation, max jump should be reasonable
        assert max_diff < 0.5, f"Warp curve not smooth enough: max_diff = {max_diff:.4f}"


class TestCombinedAugmentation:
    """Test combined augmentation pipeline (jitter + magnitude warp)."""

    def test_augment_temporal_sequence_shape(self):
        """Test combined augmentation preserves shape."""
        torch.manual_seed(42)
        x = torch.randn(8, 105, 11)
        x_aug = augment_temporal_sequence(
            x,
            jitter_sigma=0.03,
            warp_sigma=0.2,
            warp_knots=4,
            jitter_prob=0.8,
            warp_prob=0.5
        )
        assert x_aug.shape == x.shape

    def test_augment_temporal_sequence_modifies_data(self):
        """Test that combined augmentation actually modifies data."""
        torch.manual_seed(42)
        x = torch.randn(8, 105, 11)
        x_aug = augment_temporal_sequence(x)

        # At least one augmentation should have been applied
        # (with prob=0.8 and prob=0.5, very unlikely to apply neither)
        assert not torch.allclose(x, x_aug, rtol=0.01), \
            "Combined augmentation did not modify data"

    def test_augment_temporal_sequence_phase2_params(self):
        """Test augmentation with Phase 2 optimized parameters."""
        torch.manual_seed(42)
        x = torch.randn(16, 105, 11)

        x_aug = augment_temporal_sequence(
            x,
            jitter_sigma=0.03,  # PHASE 2
            warp_sigma=0.2,     # PHASE 2
            warp_knots=4,       # PHASE 2
            jitter_prob=0.8,    # PHASE 2
            warp_prob=0.5       # PHASE 2
        )

        assert x_aug.shape == x.shape
        assert not torch.allclose(x, x_aug, rtol=0.01)

    def test_augmentation_diversity_across_samples(self):
        """Test that augmentation creates different perturbations for each sample."""
        torch.manual_seed(42)
        x = torch.randn(1, 105, 11)

        # Generate multiple augmented versions
        augmented_samples = []
        for _ in range(5):
            x_aug = augment_temporal_sequence(x)
            augmented_samples.append(x_aug)

        # Check that they're all different
        for i in range(len(augmented_samples)):
            for j in range(i + 1, len(augmented_samples)):
                assert not torch.allclose(augmented_samples[i], augmented_samples[j], rtol=0.01), \
                    f"Augmented samples {i} and {j} are too similar"


class TestTemporalAugmentationClass:
    """Test TemporalAugmentation class with Phase 2 parameters."""

    def test_temporal_augmentation_init(self):
        """Test TemporalAugmentation initialization with Phase 2 params."""
        from moola.utils.augmentation.temporal_augmentation import TemporalAugmentation

        aug = TemporalAugmentation(
            jitter_prob=0.8,
            jitter_sigma=0.03,
            magnitude_warp_prob=0.5,
            magnitude_warp_sigma=0.2,
            magnitude_warp_knots=4
        )

        assert aug.jitter_prob == 0.8
        assert aug.jitter_sigma == 0.03
        assert aug.magnitude_warp_prob == 0.5
        assert aug.magnitude_warp_sigma == 0.2
        assert aug.magnitude_warp_knots == 4

    def test_temporal_augmentation_apply(self):
        """Test TemporalAugmentation.apply_augmentation() method."""
        from moola.utils.augmentation.temporal_augmentation import TemporalAugmentation

        torch.manual_seed(42)
        aug = TemporalAugmentation(
            jitter_prob=0.8,
            jitter_sigma=0.03,
            magnitude_warp_prob=0.5,
            magnitude_warp_sigma=0.2
        )

        x = torch.randn(8, 105, 11)
        x_aug = aug.apply_augmentation(x)

        assert x_aug.shape == x.shape
        # With high probabilities, should modify data
        assert not torch.allclose(x, x_aug, rtol=0.01)

    def test_temporal_augmentation_dual_views(self):
        """Test TemporalAugmentation.__call__() generates two different views."""
        from moola.utils.augmentation.temporal_augmentation import TemporalAugmentation

        torch.manual_seed(42)
        aug = TemporalAugmentation(
            jitter_prob=0.8,
            jitter_sigma=0.03,
            magnitude_warp_prob=0.5
        )

        x = torch.randn(8, 105, 11)
        x_aug1, x_aug2 = aug(x)

        # Both views should have same shape
        assert x_aug1.shape == x.shape
        assert x_aug2.shape == x.shape

        # Two views should be different from each other
        assert not torch.allclose(x_aug1, x_aug2, rtol=0.01), \
            "Two augmented views should be different"


class TestPatternPreservation:
    """Test that augmentation preserves financial patterns."""

    def test_correlation_with_original(self):
        """Test correlation between original and augmented samples."""
        torch.manual_seed(42)
        x = torch.randn(1, 105, 11)

        # Apply augmentation multiple times
        correlations = []
        for _ in range(10):
            x_aug = augment_temporal_sequence(x)

            # Flatten and compute correlation
            orig_flat = x[0].flatten()
            aug_flat = x_aug[0].flatten()

            corr_matrix = torch.corrcoef(torch.stack([orig_flat, aug_flat]))
            corr = corr_matrix[0, 1].item()
            correlations.append(corr)

        avg_corr = np.mean(correlations)
        min_corr = np.min(correlations)

        # PHASE 2 requirement: correlation > 0.90 for combined augmentation
        assert avg_corr > 0.90, \
            f"Average correlation {avg_corr:.3f} below 0.90 threshold"
        assert min_corr > 0.80, \
            f"Minimum correlation {min_corr:.3f} too low (some patterns destroyed)"

    def test_augmentation_does_not_destroy_trends(self):
        """Test that augmentation preserves monotonic trends."""
        torch.manual_seed(42)

        # Create monotonically increasing signal
        x = torch.linspace(0, 1, 105).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 11)

        x_aug = augment_temporal_sequence(x)

        # Compute correlation of first feature (should still be high)
        orig = x[0, :, 0]
        aug = x_aug[0, :, 0]

        corr = torch.corrcoef(torch.stack([orig, aug]))[0, 1].item()

        assert corr > 0.85, \
            f"Augmentation destroyed monotonic trend: correlation = {corr:.3f}"


class TestStandaloneAddJitter:
    """Test standalone add_jitter function from data.temporal_augmentation."""

    def test_add_jitter_shape(self):
        """Test that jittering preserves tensor shape."""
        x = torch.randn(32, 105, 11)
        y = add_jitter(x, sigma=0.03, prob=1.0)
        assert y.shape == x.shape

    def test_add_jitter_preserves_correlation(self):
        """Test that jittering maintains high correlation with original."""
        torch.manual_seed(42)
        x = torch.randn(105, 11)
        y = add_jitter(x, sigma=0.03, prob=1.0)

        validation = validate_augmentation(x, y, min_correlation=0.95)
        assert validation["passes"], f"Correlation {validation['correlation']:.4f} below threshold"

    def test_add_jitter_probability(self):
        """Test that probability controls augmentation application."""
        torch.manual_seed(42)
        x = torch.randn(105, 11)

        # prob=0 should return unchanged
        y_no_aug = add_jitter(x.clone(), sigma=0.03, prob=0.0)
        assert torch.allclose(x, y_no_aug)


class TestStandaloneMagnitudeWarp:
    """Test standalone magnitude_warp function from data.temporal_augmentation."""

    def test_magnitude_warp_shape(self):
        """Test that warping preserves tensor shape."""
        torch.manual_seed(42)
        x = torch.randn(32, 105, 11)
        y = magnitude_warp_standalone(x, sigma=0.2, n_knots=4, prob=1.0)
        assert y.shape == x.shape

    def test_magnitude_warp_preserves_correlation(self):
        """Test that warping maintains high correlation with original."""
        torch.manual_seed(42)
        x = torch.randn(105, 11)
        y = magnitude_warp_standalone(x, sigma=0.2, n_knots=4, prob=1.0)

        validation = validate_augmentation(x, y, min_correlation=0.95)
        assert validation["passes"], f"Correlation {validation['correlation']:.4f} below threshold"

    def test_magnitude_warp_probability(self):
        """Test that prob=0 returns original data."""
        torch.manual_seed(42)
        x = torch.randn(105, 11)
        y = magnitude_warp_standalone(x, sigma=0.2, n_knots=4, prob=0.0)
        assert torch.allclose(x, y)


class TestStandaloneCombinedAugmentation:
    """Test standalone combined augmentation pipeline."""

    def test_augment_temporal_shape(self):
        """Test combined jitter + warp pipeline."""
        torch.manual_seed(42)
        x = torch.randn(105, 11)
        y = augment_temporal(x, jitter_prob=1.0, warp_prob=1.0)

        assert y.shape == x.shape

    def test_augment_temporal_preserves_correlation(self):
        """Test that combined augmentation maintains correlation."""
        torch.manual_seed(42)
        x = torch.randn(105, 11)
        y = augment_temporal(x, jitter_prob=1.0, warp_prob=1.0)

        validation = validate_augmentation(x, y, min_correlation=0.90)  # Slightly lower for combined
        assert validation["passes"], f"Correlation {validation['correlation']:.4f} below threshold"

    def test_augment_temporal_modifies_data(self):
        """Test that augmentation actually modifies data."""
        torch.manual_seed(42)
        x = torch.randn(105, 11)
        y = augment_temporal(x, jitter_prob=1.0, warp_prob=1.0)

        assert not torch.allclose(x, y)


class TestStandaloneValidateAugmentation:
    """Test validation function for augmented samples."""

    def test_validate_augmentation_metrics(self):
        """Test that validation computes correct metrics."""
        torch.manual_seed(42)
        x = torch.randn(105, 11)
        y = add_jitter(x, sigma=0.03, prob=1.0)

        metrics = validate_augmentation(x, y, min_correlation=0.95)

        assert "correlation" in metrics
        assert "passes" in metrics
        assert "mean_diff" in metrics
        assert "std_ratio" in metrics

        assert isinstance(metrics["correlation"], (float, np.floating))
        assert isinstance(metrics["passes"], (bool, np.bool_))
        assert metrics["correlation"] > 0.0 and metrics["correlation"] <= 1.0

    def test_validate_augmentation_passes_threshold(self):
        """Test that validation correctly identifies passing samples."""
        torch.manual_seed(42)
        x = torch.randn(105, 11)

        # Small augmentation should pass
        y_small = add_jitter(x, sigma=0.01, prob=1.0)
        metrics_small = validate_augmentation(x, y_small, min_correlation=0.95)
        assert metrics_small["passes"]

        # Large augmentation might not pass
        y_large = add_jitter(x, sigma=0.5, prob=1.0)
        metrics_large = validate_augmentation(x, y_large, min_correlation=0.95)
        # Don't assert on this one since it's random, just check metric is computed
        assert "passes" in metrics_large


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
