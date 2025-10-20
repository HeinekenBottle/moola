"""Tests for RelativeFeatureTransform class."""

import numpy as np
import pytest

from moola.features.relative_transform import RelativeFeatureTransform


class TestRelativeFeatureTransform:
    """Test suite for RelativeFeatureTransform."""

    def test_init_valid_eps(self):
        """Test initialization with valid eps."""
        transform = RelativeFeatureTransform(eps=1e-8)
        assert transform.eps == 1e-8

    def test_init_invalid_eps(self):
        """Test initialization with invalid eps."""
        with pytest.raises(ValueError, match="eps must be positive"):
            RelativeFeatureTransform(eps=0.0)

        with pytest.raises(ValueError, match="eps must be positive"):
            RelativeFeatureTransform(eps=-1e-8)

    def test_transform_basic_shape(self):
        """Test basic transformation output shape."""
        transform = RelativeFeatureTransform()
        X = np.random.randn(10, 105, 4).astype(np.float32)
        X_rel = transform.transform(X)

        assert X_rel.shape == (10, 105, 11)
        assert X_rel.dtype == np.float32

    def test_transform_invalid_input_type(self):
        """Test transform with invalid input type."""
        transform = RelativeFeatureTransform()

        with pytest.raises(TypeError, match="X must be numpy array"):
            transform.transform([[1, 2, 3]])

    def test_transform_invalid_shape(self):
        """Test transform with invalid input shape."""
        transform = RelativeFeatureTransform()

        # Wrong number of dimensions
        with pytest.raises(ValueError, match="X must have shape"):
            transform.transform(np.random.randn(10, 105))

        # Wrong sequence length
        with pytest.raises(ValueError, match="X must have shape"):
            transform.transform(np.random.randn(10, 100, 4))

        # Wrong number of features
        with pytest.raises(ValueError, match="X must have shape"):
            transform.transform(np.random.randn(10, 105, 5))

    def test_transform_invalid_dtype(self):
        """Test transform with non-numeric dtype."""
        transform = RelativeFeatureTransform()
        # Create array with shape [1, 105, 4] but non-numeric dtype
        X = np.full((1, 105, 4), "a", dtype=object)

        with pytest.raises(TypeError, match="X must have numeric dtype"):
            transform.transform(X)

    def test_log_returns_first_bar_zero(self):
        """Test that first bar log returns are zero."""
        transform = RelativeFeatureTransform()
        X = np.random.randn(5, 105, 4).astype(np.float32) + 100  # Positive prices

        X_rel = transform.transform(X)

        # First 4 features are log returns
        log_returns_t0 = X_rel[:, 0, :4]

        assert np.allclose(log_returns_t0, 0.0), "First bar log returns should be zero"

    def test_log_returns_computation(self):
        """Test log return calculation accuracy."""
        transform = RelativeFeatureTransform()

        # Create test case with shape [1, 105, 4]
        X = np.ones((1, 105, 4), dtype=np.float32) * 100
        # Set first two bars: t=0 with O=100, t=1 with O=105
        X[0, 0, :] = [100, 110, 90, 105]
        X[0, 1, :] = [105, 115, 95, 110]

        X_rel = transform.transform(X)

        # Extract log returns for second bar
        log_return_open = X_rel[0, 1, 0]
        expected = np.log(105 / 100)

        assert np.isclose(log_return_open, expected, atol=1e-6)

    def test_candle_ratios_range(self):
        """Test that candle ratios are in [0, 1] range."""
        transform = RelativeFeatureTransform()
        X = np.random.randn(10, 105, 4).astype(np.float32) + 100

        X_rel = transform.transform(X)

        # Features 4-6 are candle ratios
        candle_ratios = X_rel[:, :, 4:7]

        assert np.all(candle_ratios >= 0.0), "Candle ratios should be >= 0"
        assert np.all(candle_ratios <= 1.0), "Candle ratios should be <= 1"

    def test_candle_ratios_computation(self):
        """Test candle ratio calculation accuracy."""
        transform = RelativeFeatureTransform()

        # Create test case with shape [1, 105, 4]
        X = np.ones((1, 105, 4), dtype=np.float32) * 100
        # Set first bar: O=100, H=110, L=90, C=105
        # Range = 110 - 90 = 20
        # Body = |105 - 100| / 20 = 0.25
        # Upper wick = (110 - 105) / 20 = 0.25
        # Lower wick = (100 - 90) / 20 = 0.5
        X[0, 0, :] = [100, 110, 90, 105]

        X_rel = transform.transform(X)

        body_ratio = X_rel[0, 0, 4]
        upper_wick_ratio = X_rel[0, 0, 5]
        lower_wick_ratio = X_rel[0, 0, 6]

        assert np.isclose(body_ratio, 0.25, atol=1e-6)
        assert np.isclose(upper_wick_ratio, 0.25, atol=1e-6)
        assert np.isclose(lower_wick_ratio, 0.5, atol=1e-6)

    def test_candle_ratios_doji(self):
        """Test candle ratios for doji (open == close)."""
        transform = RelativeFeatureTransform()

        # Create test case with shape [1, 105, 4]
        X = np.ones((1, 105, 4), dtype=np.float32) * 100
        # Doji: O=100, H=110, L=90, C=100
        X[0, 0, :] = [100, 110, 90, 100]

        X_rel = transform.transform(X)

        body_ratio = X_rel[0, 0, 4]
        upper_wick_ratio = X_rel[0, 0, 5]
        lower_wick_ratio = X_rel[0, 0, 6]

        assert np.isclose(body_ratio, 0.0, atol=1e-6)
        assert np.isclose(upper_wick_ratio, 0.5, atol=1e-6)
        assert np.isclose(lower_wick_ratio, 0.5, atol=1e-6)

    def test_zscores_clipping(self):
        """Test that z-scores are clipped to [-10, 10]."""
        transform = RelativeFeatureTransform()

        # Create data with extreme outliers
        X = np.random.randn(5, 105, 4).astype(np.float32)
        X[:, 50, :] = 1000  # Extreme outlier

        X_rel = transform.transform(X)

        # Features 7-10 are z-scores
        z_scores = X_rel[:, :, 7:11]

        assert np.all(z_scores >= -10.0), "Z-scores should be >= -10"
        assert np.all(z_scores <= 10.0), "Z-scores should be <= 10"

    def test_zscores_first_bar_zero(self):
        """Test that first bar z-scores are zero."""
        transform = RelativeFeatureTransform()
        X = np.random.randn(5, 105, 4).astype(np.float32)

        X_rel = transform.transform(X)

        # Features 7-10 are z-scores
        z_scores_t0 = X_rel[:, 0, 7:11]

        assert np.allclose(z_scores_t0, 0.0), "First bar z-scores should be zero"

    def test_zscores_computation(self):
        """Test z-score calculation accuracy."""
        transform = RelativeFeatureTransform()

        # Create simple test case with known statistics
        X = np.ones((1, 105, 4), dtype=np.float32) * 100
        X[0, 20, 0] = 120  # Outlier at t=20

        X_rel = transform.transform(X)

        # Z-score at t=20 should be positive (above mean)
        zscore_open_t20 = X_rel[0, 20, 7]

        assert zscore_open_t20 > 0, "Z-score should be positive for value above mean"

    def test_no_nan_values(self):
        """Test that output contains no NaN values."""
        transform = RelativeFeatureTransform()

        # Test with various edge cases
        X = np.random.randn(10, 105, 4).astype(np.float32)

        # Add some edge cases
        X[0, 0, :] = 0  # Zero prices
        X[1, 5, :] = 1e-10  # Very small prices

        X_rel = transform.transform(X)

        assert not np.any(np.isnan(X_rel)), "Output should not contain NaN values"
        assert not np.any(np.isinf(X_rel)), "Output should not contain inf values"

    def test_division_by_zero_handling(self):
        """Test handling of division by zero cases."""
        transform = RelativeFeatureTransform()

        # Create data with zero range (H == L)
        X = np.ones((1, 105, 4), dtype=np.float32) * 100
        X[0, 10, :] = [100, 100, 100, 100]  # Zero range

        X_rel = transform.transform(X)

        # Should not raise error and should not contain NaN/inf
        assert not np.any(np.isnan(X_rel))
        assert not np.any(np.isinf(X_rel))

    def test_get_feature_names(self):
        """Test feature names."""
        transform = RelativeFeatureTransform()
        names = transform.get_feature_names()

        assert len(names) == 11
        assert names[0] == "log_return_open"
        assert names[4] == "body_ratio"
        assert names[7] == "zscore_open"

    def test_feature_names_order(self):
        """Test that feature names match transformation order."""
        transform = RelativeFeatureTransform()
        names = transform.get_feature_names()

        expected = [
            "log_return_open",
            "log_return_high",
            "log_return_low",
            "log_return_close",
            "body_ratio",
            "upper_wick_ratio",
            "lower_wick_ratio",
            "zscore_open",
            "zscore_high",
            "zscore_low",
            "zscore_close",
        ]

        assert names == expected

    def test_batch_processing(self):
        """Test processing multiple samples."""
        transform = RelativeFeatureTransform()

        # Test with different batch sizes
        for batch_size in [1, 10, 100]:
            X = np.random.randn(batch_size, 105, 4).astype(np.float32) + 100
            X_rel = transform.transform(X)

            assert X_rel.shape == (batch_size, 105, 11)
            assert not np.any(np.isnan(X_rel))

    def test_deterministic_output(self):
        """Test that transformation is deterministic."""
        transform = RelativeFeatureTransform()

        X = np.random.randn(5, 105, 4).astype(np.float32)

        X_rel1 = transform.transform(X)
        X_rel2 = transform.transform(X)

        assert np.allclose(X_rel1, X_rel2), "Transformation should be deterministic"

    def test_realistic_price_data(self):
        """Test with realistic price ranges."""
        transform = RelativeFeatureTransform()

        # Simulate realistic stock prices
        X = np.random.randn(10, 105, 4).astype(np.float32) * 5 + 150

        # Ensure OHLC relationships: L <= O,C <= H
        for i in range(10):
            for t in range(105):
                low = X[i, t, 2]
                high = X[i, t, 1]
                X[i, t, 0] = np.random.uniform(low, high)  # Open
                X[i, t, 3] = np.random.uniform(low, high)  # Close

        X_rel = transform.transform(X)

        assert X_rel.shape == (10, 105, 11)
        assert not np.any(np.isnan(X_rel))
        assert not np.any(np.isinf(X_rel))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
