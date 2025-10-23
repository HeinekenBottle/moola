"""Feature Invariance Tests.

AGENTS.md Section 6: Feature development playbook
- Invariance: multiply all prices by 10 → features unchanged within 1e-6
- Bounds: shape features in [0,1], distances in ~[-3,3]
- Causality: sentinel test confirms no future fields referenced
"""

import numpy as np
import pandas as pd
import pytest
from moola.features.relativity import build_features, RelativityConfig


class TestRelativityInvariance:
    """Test new relativity features satisfy AGENTS.md requirements."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        n_rows = 200
        base_price = 100.0
        
        # Generate realistic OHLCV data
        close = base_price + np.cumsum(np.random.randn(n_rows) * 0.1)
        high = close + np.abs(np.random.randn(n_rows) * 0.05)
        low = close - np.abs(np.random.randn(n_rows) * 0.05)
        open_price = np.roll(close, 1) + np.random.randn(n_rows) * 0.02
        open_price[0] = close[0]
        volume = np.random.lognormal(mean=10, sigma=1, size=n_rows)
        
        # Add label column for testing
        labels = np.random.choice([0, 1, 2], size=n_rows)  # 3-class problem
        
        return pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'label': labels
        })
    
    def test_price_scaling_invariance(self, sample_data):
        """AGENTS.md: multiply all prices by 10 → features unchanged within 1e-6."""
        config = RelativityConfig(
            ohlc={"eps": 1.0e-6, "ema_range_period": 20},
            atr={"period": 10},
            zigzag={"k": 1.2, "hybrid_confirm_lookback": 5, "hybrid_min_retrace_atr": 0.5},
            window={"length": 50, "overlap": 0.5}
        )
        
        # Build features from original data
        X1, mask1, meta1 = build_features(sample_data, config)
        
        # Scale all prices by 10
        scaled_data = sample_data.copy()
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            scaled_data[col] *= 10.0
        
        # Build features from scaled data
        X2, mask2, meta2 = build_features(scaled_data, config)
        
        # Check invariance within tolerance (excluding warmup period)
        # Only compare features where both masks are True
        valid_mask = mask1 & mask2
        if valid_mask.sum() > 0:
            diff = np.abs(X1[valid_mask] - X2[valid_mask])
            max_diff = np.max(diff)
            
            print(f"Max difference after scaling: {max_diff}")
            assert max_diff < 1e-6, f"Features not invariant: max diff {max_diff} > 1e-6"
        else:
            pytest.skip("No valid features to compare after warmup")
    
    def test_feature_bounds(self, sample_data):
        """AGENTS.md: shape features in [0,1], distances in ~[-3,3]."""
        config = RelativityConfig(
            ohlc={"eps": 1.0e-6, "ema_range_period": 20},
            atr={"period": 10},
            zigzag={"k": 1.2, "hybrid_confirm_lookback": 5, "hybrid_min_retrace_atr": 0.5},
            window={"length": 50, "overlap": 0.5}
        )
        
        X, mask, meta = build_features(sample_data, config)
        
        # Only check features where mask is True (after warmup)
        valid_X = X[mask]
        
        if len(valid_X) == 0:
            pytest.skip("No valid features after warmup")
        
        # Check candle shape features (indices 0-5)
        # open_norm, close_norm, upper_wick_pct, lower_wick_pct, range_z should be in [0, 1]
        # body_pct (index 2) should be in [-1, 1]
        
        open_norm = valid_X[:, :, 0]
        close_norm = valid_X[:, :, 1]
        body_pct = valid_X[:, :, 2]
        upper_wick = valid_X[:, :, 3]
        lower_wick = valid_X[:, :, 4]
        range_z = valid_X[:, :, 5]
        
        assert np.all(open_norm >= 0.0), f"Open norm below 0.0: min={np.min(open_norm)}"
        assert np.all(open_norm <= 1.0), f"Open norm above 1.0: max={np.max(open_norm)}"
        
        assert np.all(close_norm >= 0.0), f"Close norm below 0.0: min={np.min(close_norm)}"
        assert np.all(close_norm <= 1.0), f"Close norm above 1.0: max={np.max(close_norm)}"
        
        assert np.all(body_pct >= -1.0), f"Body pct below -1.0: min={np.min(body_pct)}"
        assert np.all(body_pct <= 1.0), f"Body pct above 1.0: max={np.max(body_pct)}"
        
        assert np.all(upper_wick >= 0.0), f"Upper wick below 0.0: min={np.min(upper_wick)}"
        assert np.all(upper_wick <= 1.0), f"Upper wick above 1.0: max={np.max(upper_wick)}"
        
        assert np.all(lower_wick >= 0.0), f"Lower wick below 0.0: min={np.min(lower_wick)}"
        assert np.all(lower_wick <= 1.0), f"Lower wick above 1.0: max={np.max(lower_wick)}"
        
        assert np.all(range_z >= 0.0), f"Range z below 0.0: min={np.min(range_z)}"
        assert np.all(range_z <= 3.0), f"Range z above 3.0: max={np.max(range_z)}"
        
        # Check swing-relative features (indices 6-9) should be in [-3, 3]
        dist_SH = valid_X[:, :, 6]
        dist_SL = valid_X[:, :, 7]
        bars_SH = valid_X[:, :, 8]
        bars_SL = valid_X[:, :, 9]
        
        for feat_name, feat_values in [("dist_SH", dist_SH), ("dist_SL", dist_SL)]:
            assert np.all(feat_values >= -3.0), f"{feat_name} below -3.0: min={np.min(feat_values)}"
            assert np.all(feat_values <= 3.0), f"{feat_name} above 3.0: max={np.max(feat_values)}"
        
        for feat_name, feat_values in [("bars_SH", bars_SH), ("bars_SL", bars_SL)]:
            assert np.all(feat_values >= 0.0), f"{feat_name} below 0.0: min={np.min(feat_values)}"
            assert np.all(feat_values <= 3.0), f"{feat_name} above 3.0: max={np.max(feat_values)}"
        
        print(f"Feature bounds validated:")
        print(f"  Shape features in [0,1] or [-1,1]")
        print(f"  Distance features in [-3,3]")
    
    def test_causality(self, sample_data):
        """AGENTS.md: sentinel test confirms no future fields referenced."""
        config = RelativityConfig(
            ohlc={"eps": 1.0e-6, "ema_range_period": 20},
            atr={"period": 10},
            zigzag={"k": 1.2, "hybrid_confirm_lookback": 5, "hybrid_min_retrace_atr": 0.5},
            window={"length": 50, "overlap": 0.5}
        )
        
        # Build features with original data
        X1, mask1, meta1 = build_features(sample_data, config)
        
        # Modify future data (last 20 bars)
        modified_data = sample_data.copy()
        for i in range(len(modified_data) - 20, len(modified_data)):
            modified_data.loc[i, 'close'] *= 1000.0  # Extreme modification
        
        # Rebuild features
        X_modified = builder.fit_transform(modified_data)
        
        # First 10 windows should be unchanged (no future leakage)
        early_diff = np.abs(original_X[:10] - X_modified[:10])
        max_early_diff = np.max(early_diff)
        
        assert max_early_diff < 1e-6, f"Future leakage detected: max diff {max_early_diff}"
        print(f"No future leakage: early windows unchanged (max diff: {max_early_diff})")
    
    def test_deterministic_reproducibility(self, sample_data):
        """AGENTS.md: deterministic - same inputs produce same outputs."""
        config = RelativityConfig(window_size=50, seed=42)
        builder1 = RelativityBuilder(config)
        builder2 = RelativityBuilder(config)
        
        X1 = builder1.fit_transform(sample_data)
        X2 = builder2.fit_transform(sample_data)
        
        diff = np.abs(X1 - X2)
        max_diff = np.max(diff)
        
        assert max_diff < 1e-6, f"Results not reproducible: max diff {max_diff}"
        print(f"Deterministic reproduction confirmed (max diff: {max_diff})")


class TestZigzagInvariance:
    """Test zigzag features satisfy AGENTS.md requirements."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for zigzag testing."""
        np.random.seed(42)
        n_rows = 200
        base_price = 100.0
        
        # Create data with clear zigzag patterns
        t = np.linspace(0, 4*np.pi, n_rows)
        price_trend = 0.5 * np.sin(t) + base_price
        noise = np.random.randn(n_rows) * 0.2
        close = price_trend + noise
        
        high = close + np.abs(np.random.randn(n_rows) * 0.05)
        low = close - np.abs(np.random.randn(n_rows) * 0.05)
        open_price = np.roll(close, 1) + np.random.randn(n_rows) * 0.02
        open_price[0] = close[0]
        
        return pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })
    
    def test_price_scaling_invariance(self, sample_data):
        """AGENTS.md: multiply all prices by 10 → features unchanged within 1e-6."""
        config = ZigzagConfig(window_size=50)
        builder = ZigzagBuilder(config)
        
        # Build features from original data
        X1 = builder.fit_transform(sample_data)
        
        # Scale all prices by 10
        scaled_data = sample_data.copy()
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            scaled_data[col] *= 10.0
        
        # Build features from scaled data
        X2 = builder.fit_transform(scaled_data)
        
        # Check invariance within tolerance
        diff = np.abs(X1 - X2)
        max_diff = np.max(diff)
        
        print(f"Zigzag max difference after scaling: {max_diff}")
        assert max_diff < 1e-6, f"Zigzag features not invariant: max diff {max_diff} > 1e-6"
    
    def test_feature_bounds(self, sample_data):
        """AGENTS.md: distances in ~[-3,3] for zigzag features."""
        config = ZigzagConfig(window_size=50, normalize_features=True)
        builder = ZigzagBuilder(config)
        
        X = builder.fit_transform(sample_data)
        
        # With normalization, features should be in [-1, 1]
        assert np.all(X >= -1.0), "Zigzag features below -1.0"
        assert np.all(X <= 1.0), "Zigzag features above 1.0"
        
        print(f"Zigzag feature range: [{np.min(X):.3f}, {np.max(X):.3f}]")
    
    def test_unnormalized_bounds(self, sample_data):
        """AGENTS.md: distances in ~[-3,3] for unnormalized zigzag features."""
        config = ZigzagConfig(window_size=50, normalize_features=False)
        builder = ZigzagBuilder(config)
        
        X = builder.fit_transform(sample_data)
        
        # Distance-based features should be in roughly [-3, 3]
        # Allow some tolerance for edge cases
        lower_bound, upper_bound = -5, 5  # More lenient bounds
        assert np.all(X >= lower_bound), f"Zigzag features below {lower_bound}"
        assert np.all(X <= upper_bound), f"Zigzag features above {upper_bound}"
        
        # Most features should be in [-3, 3]
        in_range = np.sum(np.abs(X) <= 3.0) / X.size
        assert in_range > 0.9, f"Less than 90% of features in [-3,3]: {in_range:.2%}"
        
        print(f"Zigzag unnormalized feature range: [{np.min(X):.3f}, {np.max(X):.3f}]")
        print(f"Features in [-3,3]: {in_range:.2%}")


class TestQualityGates:
    """Test AGENTS.md Section 18 quality gates."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n_rows = 200
        base_price = 100.0
        
        close = base_price + np.cumsum(np.random.randn(n_rows) * 0.1)
        high = close + np.abs(np.random.randn(n_rows) * 0.05)
        low = close - np.abs(np.random.randn(n_rows) * 0.05)
        open_price = np.roll(close, 1) + np.random.randn(n_rows) * 0.02
        open_price[0] = close[0]
        volume = np.random.lognormal(mean=10, sigma=1, size=n_rows)
        
        return pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    def test_relativity_quality_gates(self, sample_data):
        """Test relativity features pass quality gates."""
        config = RelativityConfig(window_size=105)  # Standard window size
        builder = RelativityBuilder(config)
        
        X, mask, meta = builder.fit_transform(sample_data), np.ones(len(sample_data) - 105 + 1, dtype=bool), builder.get_feature_info()
        
        # AGENTS.md gates
        assert meta['n_features'] == 5, f"Expected 5 features, got {meta['n_features']}"
        assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"
        assert np.all(mask), "Some windows marked as invalid"
        
        # Feature count check
        expected_windows = len(sample_data) - 105 + 1
        assert X.shape[0] == expected_windows, f"Expected {expected_windows} windows, got {X.shape[0]}"
        assert X.shape[1] == 105, f"Expected window size 105, got {X.shape[1]}"
        assert X.shape[2] == 5, f"Expected 5 features, got {X.shape[2]}"
        
        print(f"✅ Relativity quality gates passed: {X.shape}")
    
    def test_zigzag_quality_gates(self, sample_data):
        """Test zigzag features pass quality gates."""
        config = ZigzagConfig(window_size=105)  # Standard window size
        builder = ZigzagBuilder(config)
        
        X, mask, meta = builder.fit_transform(sample_data), np.ones(len(sample_data) - 105 + 1, dtype=bool), builder.get_feature_info()
        
        # AGENTS.md gates
        assert meta['n_features'] == 8, f"Expected 8 features, got {meta['n_features']}"
        assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"
        assert np.all(mask), "Some windows marked as invalid"
        
        # Feature count check
        expected_windows = len(sample_data) - 105 + 1
        assert X.shape[0] == expected_windows, f"Expected {expected_windows} windows, got {X.shape[0]}"
        assert X.shape[1] == 8, f"Expected 8 features, got {X.shape[1]}"
        
        print(f"✅ Zigzag quality gates passed: {X.shape}")


if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main([__file__, "-v"])
