"""Tests for data augmentation module."""

import numpy as np
import pytest
import torch

from moola.aug import (
    Jitter,
    MagnitudeWarp,
    OnTheFlyAugmentation,
    create_augmentation_pipeline,
)


class TestJitter:
    """Test jitter augmentation."""

    def test_jitter_init(self):
        """Test jitter initialization."""
        jitter = Jitter(sigma=0.03, prob=0.8)
        assert jitter.sigma == 0.03
        assert jitter.prob == 0.8

    def test_jitter_forward_eval(self):
        """Test jitter forward pass in eval mode."""
        jitter = Jitter(sigma=0.03, prob=1.0)
        jitter.eval()

        x = torch.randn(4, 105, 11)
        output = jitter(x)

        # Should be identical in eval mode
        assert torch.equal(output, x)

    def test_jitter_forward_train(self):
        """Test jitter forward pass in train mode."""
        jitter = Jitter(sigma=0.03, prob=1.0)
        jitter.train()

        x = torch.randn(4, 105, 11)
        output = jitter(x)

        # Should be different in train mode
        assert not torch.equal(output, x)
        assert output.shape == x.shape

    def test_jitter_prob_zero(self):
        """Test jitter with zero probability."""
        jitter = Jitter(sigma=0.03, prob=0.0)
        jitter.train()

        x = torch.randn(4, 105, 11)
        output = jitter(x)

        # Should be identical with zero probability
        assert torch.equal(output, x)


class TestMagnitudeWarp:
    """Test magnitude warping augmentation."""

    def test_warp_init(self):
        """Test magnitude warp initialization."""
        warp = MagnitudeWarp(sigma=0.2, knots=4, prob=0.5)
        assert warp.sigma == 0.2
        assert warp.knots == 4
        assert warp.prob == 0.5

    def test_warp_forward_eval(self):
        """Test warp forward pass in eval mode."""
        warp = MagnitudeWarp(sigma=0.2, prob=1.0)
        warp.eval()

        x = torch.randn(4, 105, 11)
        output = warp(x)

        # Should be identical in eval mode
        assert torch.equal(output, x)

    def test_warp_forward_train(self):
        """Test warp forward pass in train mode."""
        warp = MagnitudeWarp(sigma=0.2, prob=1.0)
        warp.train()

        x = torch.randn(2, 50, 3)  # Smaller for faster testing
        output = warp(x)

        # Should be different in train mode
        assert not torch.equal(output, x)
        assert output.shape == x.shape

    def test_warp_prob_zero(self):
        """Test warp with zero probability."""
        warp = MagnitudeWarp(sigma=0.2, prob=0.0)
        warp.train()

        x = torch.randn(4, 105, 11)
        output = warp(x)

        # Should be identical with zero probability
        assert torch.equal(output, x)


class TestOnTheFlyAugmentation:
    """Test on-the-fly augmentation pipeline."""

    def test_init(self):
        """Test on-the-fly augmentation initialization."""
        aug = OnTheFlyAugmentation(jitter_sigma=0.03, warp_sigma=0.2, multiplier=3)
        assert aug.multiplier == 3

    def test_forward_eval(self):
        """Test on-the-fly augmentation in eval mode."""
        aug = OnTheFlyAugmentation(multiplier=3)
        aug.eval()

        x = torch.randn(4, 105, 11)
        output = aug(x)

        # Should be identical in eval mode
        assert torch.equal(output, x)
        assert output.shape == x.shape

    def test_forward_train(self):
        """Test on-the-fly augmentation in train mode."""
        aug = OnTheFlyAugmentation(multiplier=3)
        aug.train()

        x = torch.randn(2, 50, 3)  # Smaller for faster testing
        output = aug(x)

        # Should be 3x larger in train mode
        assert output.shape == (6, 50, 3)  # 2 * 3 = 6

    def test_multiplier_one(self):
        """Test on-the-fly augmentation with multiplier=1."""
        aug = OnTheFlyAugmentation(multiplier=1)
        aug.train()

        x = torch.randn(4, 105, 11)
        output = aug(x)

        # Should be identical with multiplier=1
        assert torch.equal(output, x)
        assert output.shape == x.shape


class TestAugmentationPipeline:
    """Test augmentation pipeline factory function."""

    def test_create_pipeline_default(self):
        """Test creating pipeline with default parameters."""
        pipeline = create_augmentation_pipeline()

        assert isinstance(pipeline, OnTheFlyAugmentation)
        assert pipeline.multiplier == 3

    def test_create_pipeline_custom(self):
        """Test creating pipeline with custom parameters."""
        pipeline = create_augmentation_pipeline(
            jitter_sigma=0.05, warp_sigma=0.3, multiplier=5, jitter_prob=0.9, warp_prob=0.7
        )

        assert pipeline.multiplier == 5
        assert pipeline.jitter.sigma == 0.05
        assert pipeline.warp.sigma == 0.3
        assert pipeline.jitter.prob == 0.9
        assert pipeline.warp.prob == 0.7


class TestStonesSpecifications:
    """Test that augmentations meet Stones specifications."""

    def test_jitter_sigma_spec(self):
        """Test jitter uses σ=0.03 as per Stones spec."""
        pipeline = create_augmentation_pipeline()
        assert pipeline.jitter.sigma == 0.03

    def test_warp_sigma_spec(self):
        """Test magnitude warp uses σ=0.2 as per Stones spec."""
        pipeline = create_augmentation_pipeline()
        assert pipeline.warp.sigma == 0.2

    def test_multiplier_spec(self):
        """Test multiplier is 3 as per Stones spec."""
        pipeline = create_augmentation_pipeline()
        assert pipeline.multiplier == 3

    def test_on_the_fly_behavior(self):
        """Test on-the-fly augmentation behavior."""
        pipeline = create_augmentation_pipeline()
        pipeline.train()

        x = torch.randn(2, 105, 11)
        output = pipeline(x)

        # Should create 3 augmented versions on-the-fly
        assert output.shape[0] == x.shape[0] * 3

        # Original should be included
        assert torch.equal(output[:2], x)


if __name__ == "__main__":
    pytest.main([__file__])
