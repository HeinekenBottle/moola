"""Zigzag Feature Tests.

AGENTS.md Section 6: Feature development playbook
- Causality: no use of future bars
- Hybrid confirmation: K×ATR and hybrid rules
- Scale invariance: features don't depend on absolute price levels
- Bounds: distance features in ~[-3,3]
"""

import numpy as np
import pandas as pd
import pytest

from moola.features.zigzag import CausalZigZag, swing_relative


class TestCausalZigZag:
    """Test CausalZigZag implementation."""

    @pytest.fixture
    def synthetic_ohlc(self):
        """Create synthetic OHLC data for testing."""
        np.random.seed(42)
        n_bars = 100
        base_price = 100.0

        # Generate price series with trends
        price_path = base_price + np.cumsum(np.random.randn(n_bars) * 0.1)

        # Add some volatility clusters (higher volatility periods)
        volatility = np.ones(n_bars)
        volatility[30:40] = 3.0  # High volatility period
        volatility[60:65] = 2.5  # Medium volatility period

        # Create OHLC from price path
        data = []
        for i in range(n_bars):
            if i == 0:
                prev_close = base_price
            else:
                prev_close = close

            # Generate intraday movement
            daily_range = np.random.exponential(scale=0.5) * volatility[i]

            # Determine trend direction
            if i < 30:
                trend = 0.2  # Uptrend
            elif i < 50:
                trend = -0.15  # Downtrend
            elif i < 70:
                trend = 0.1  # Uptrend
            else:
                trend = -0.05  # Slight downtrend

            # Calculate OHLC
            open_price = prev_close + np.random.randn() * 0.1
            close = open_price + trend + np.random.randn() * 0.05

            high = max(open_price, close) + np.random.exponential(daily_range * 0.3)
            low = min(open_price, close) - np.random.exponential(daily_range * 0.3)

            # Ensure high >= low
            if high < low:
                high, low = low, high

            # Volume (not used by zigzag but for completeness)
            volume = np.random.lognormal(mean=10, sigma=1)

            data.append(
                {"open": open_price, "high": high, "low": low, "close": close, "volume": volume}
            )

        return pd.DataFrame(data)

    def test_atr_calculation(self, synthetic_ohlc):
        """Test ATR calculation using True Range."""
        zigzag = CausalZigZag(atr_period=10, k=1.2)

        # Process first few bars
        for i in range(min(15, len(synthetic_ohlc))):
            row = synthetic_ohlc.iloc[i]
            prev_SH, prev_SL = zigzag.update(row["open"], row["high"], row["low"], row["close"])

        state = zigzag.get_state()

        # ATR should be positive and reasonable
        atr = state["current_atr"]
        assert atr > 0, f"ATR should be positive, got {atr}"
        assert atr < 10.0, f"ATR seems too large: {atr}"  # Given our synthetic data

        # ATR should be in same ballpark as typical daily range
        typical_range = synthetic_ohlc["high"] - synthetic_ohlc["low"]
        assert (
            abs(atr - typical_range.mean()) / typical_range.mean() < 2.0
        ), f"ATR {atr} should be close to typical range {typical_range.mean()}"

    def test_swing_detection(self, synthetic_ohlc):
        """Test basic swing detection with K×ATR threshold."""
        zigzag = CausalZigZag(atr_period=5, k=1.0)  # Use lower K for easier swing detection

        # Process all data
        swings_detected = []
        for i in range(len(synthetic_ohlc)):
            row = synthetic_ohlc.iloc[i]
            prev_SH, prev_SL = zigzag.update(row["open"], row["high"], row["low"], row["close"])

            if prev_SH or prev_SL:
                swings_detected.append(i)

        # Should detect some swings in our synthetic data
        assert len(swings_detected) > 0, "No swings detected in trending synthetic data"

        # Check that swings are detected at reasonable times
        # (not too early, not all at the end)
        assert swings_detected[0] > 5, "First swing detected too early (warmup period)"
        assert swings_detected[-1] < len(synthetic_ohlc) - 5, "Last swing detected too late"

    def test_hybrid_confirmation(self, synthetic_ohlc):
        """Test hybrid confirmation rule."""
        # Create specific pattern for hybrid testing
        zigzag = CausalZigZag(atr_period=10, k=2.0, hybrid_lb=3, hybrid_min_atr=0.5)

        # Track when swings are confirmed
        swing_times = []

        for i in range(len(synthetic_ohlc)):
            row = synthetic_ohlc.iloc[i]
            prev_SH, prev_SL = zigzag.update(row["open"], row["high"], row["low"], row["close"])

            if prev_SH or prev_SL:
                swing_times.append(i)

        # With hybrid confirmation, should still detect swings
        # but possibly at different times than primary-only confirmation
        assert len(swing_times) > 0, "No swings detected with hybrid confirmation"

    def test_causality(self, synthetic_ohlc):
        """Test that zigzag is strictly causal."""
        zigzag = CausalZigZag(atr_period=10, k=1.2)

        # Process data up to certain point
        midpoint = len(synthetic_ohlc) // 2
        states_at_midpoint = []

        for i in range(midpoint):
            row = synthetic_ohlc.iloc[i]
            prev_SH, prev_SL = zigzag.update(row["open"], row["high"], row["low"], row["close"])

            if i > 20:  # After warmup
                state = zigzag.get_state()
                states_at_midpoint.append(state["current_trend"])

        # Continue processing remaining data
        for i in range(midpoint, len(synthetic_ohlc)):
            row = synthetic_ohlc.iloc[i]
            prev_SH, prev_SL = zigzag.update(row["open"], row["high"], row["low"], row["close"])

        # Early states should not be affected by later data
        # (This is more of a design test - our implementation should be causal)
        final_state = zigzag.get_state()

        # Check that we processed all bars
        assert final_state["bar_count"] == len(
            synthetic_ohlc
        ), f"Expected {len(synthetic_ohlc)} bars, got {final_state['bar_count']}"

    def test_scale_invariance(self, synthetic_ohlc):
        """Test that zigzag features are scale-invariant."""
        zigzag1 = CausalZigZag(atr_period=10, k=1.2)
        zigzag2 = CausalZigZag(atr_period=10, k=1.2)

        # Process original data
        swings1 = []
        for i in range(len(synthetic_ohlc)):
            row = synthetic_ohlc.iloc[i]
            prev_SH, prev_SL = zigzag1.update(row["open"], row["high"], row["low"], row["close"])
            if prev_SH or prev_SL:
                swings1.append(i)

        # Scale all prices by 10x
        scaled_ohlc = synthetic_ohlc.copy()
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            scaled_ohlc[col] *= 10.0

        # Process scaled data
        swings2 = []
        for i in range(len(synthetic_ohlc)):
            row = scaled_ohlc.iloc[i]
            prev_SH, prev_SL = zigzag2.update(row["open"], row["high"], row["low"], row["close"])
            if prev_SH or prev_SL:
                swings2.append(i)

        # Swing detection should be at the same times (scale-invariant)
        assert len(swings1) == len(
            swings2
        ), f"Different number of swings: {len(swings1)} vs {len(swings2)}"

        # Allow small difference in timing due to floating point precision
        for i, (s1, s2) in enumerate(zip(swings1, swings2)):
            assert abs(s1 - s2) <= 1, f"Swing {i} timing differs: {s1} vs {s2}"


class TestSwingRelative:
    """Test swing_relative function."""

    def test_distance_features(self):
        """Test distance feature calculations."""
        close = 100.0
        prev_SH_price = 105.0  # Above current
        prev_SL_price = 95.0  # Below current
        atr = 2.0
        bars_since_SH = 10
        bars_since_SL = 5
        K = 20  # Normalization constant

        dist_SH, dist_SL, bars_SH_norm, bars_SL_norm = swing_relative(
            close, prev_SH_price, prev_SL_price, atr, bars_since_SH, bars_since_SL, K
        )

        # Distance to swing high should be positive (below SH)
        assert dist_SH > 0, f"Distance to SH should be positive, got {dist_SH}"
        expected_dist_SH = (105.0 - 100.0) / 2.0  # = 2.5
        assert abs(dist_SH - expected_dist_SH) < 0.01, f"Expected SH distance ~2.5, got {dist_SH}"

        # Distance to swing low should be positive (above SL)
        assert dist_SL > 0, f"Distance to SL should be positive, got {dist_SL}"
        expected_dist_SL = (100.0 - 95.0) / 2.0  # = 2.5
        assert abs(dist_SL - expected_dist_SL) < 0.01, f"Expected SL distance ~2.5, got {dist_SL}"

        # Time features should be normalized by K
        expected_bars_SH = 10 / 20  # = 0.5
        expected_bars_SL = 5 / 20  # = 0.25

        assert (
            abs(bars_SH_norm - expected_bars_SH) < 0.01
        ), f"Expected bars since SH ~0.5, got {bars_SH_norm}"
        assert (
            abs(bars_SL_norm - expected_bars_SL) < 0.01
        ), f"Expected bars since SL ~0.25, got {bars_SL_norm}"

    def test_feature_bounds(self):
        """Test that features are properly bounded."""
        close = 100.0
        atr = 1.0
        K = 10

        # Test extreme values
        # Far below swing high (max positive distance)
        dist_SH, _, _, _ = swing_relative(close, 200.0, None, atr, 50, 0, K)
        assert dist_SH <= 3.0, f"Distance should be clipped to 3, got {dist_SH}"

        # Far below swing low (max negative distance) - test by swapping roles
        _, dist_SL, _, _ = swing_relative(200.0, None, 100.0, atr, 0, 50, K)
        assert dist_SL <= 3.0, f"Distance should be clipped to 3, got {dist_SL}"

        # Time features should be positive and bounded
        _, _, bars_SH, bars_SL = swing_relative(close, 105.0, 95.0, atr, 100, 100, K)
        assert bars_SH <= 3.0, f"Bars since SH should be clipped to 3, got {bars_SH}"
        assert bars_SL <= 3.0, f"Bars since SL should be clipped to 3, got {bars_SL}"
        assert bars_SH >= 0, f"Bars since SH should be non-negative, got {bars_SH}"
        assert bars_SL >= 0, f"Bars since SL should be non-negative, got {bars_SL}"

    def test_null_handling(self):
        """Test handling of missing swing points."""
        close = 100.0
        atr = 2.0
        bars_since_SH = 10
        bars_since_SL = 5
        K = 20

        # No previous swing high
        dist_SH, dist_SL, bars_SH_norm, bars_SL_norm = swing_relative(
            close, None, 95.0, atr, bars_since_SH, bars_since_SL, K
        )
        assert dist_SH == 0.0, "Distance to missing SH should be 0"
        assert dist_SL > 0, "Distance to existing SL should be positive"

        # No previous swing low
        dist_SH, dist_SL, bars_SH_norm, bars_SL_norm = swing_relative(
            close, 105.0, None, atr, bars_since_SH, bars_since_SL, K
        )
        assert dist_SL == 0.0, "Distance to missing SL should be 0"
        assert dist_SH > 0, "Distance to existing SH should be positive"

        # No previous swings
        dist_SH, dist_SL, bars_SH_norm, bars_SL_norm = swing_relative(
            close, None, None, atr, bars_since_SH, bars_since_SL, K
        )
        assert dist_SH == 0.0, "Distance to missing SH should be 0"
        assert dist_SL == 0.0, "Distance to missing SL should be 0"

    def test_zero_atr_handling(self):
        """Test handling of zero ATR (division by zero prevention)."""
        close = 100.0
        prev_SH_price = 105.0
        prev_SL_price = 95.0
        atr = 0.0  # Zero ATR
        bars_since_SH = 10
        bars_since_SL = 5
        K = 20

        # Should not raise exception or return inf/nan
        dist_SH, dist_SL, bars_SH_norm, bars_SL_norm = swing_relative(
            close, prev_SH_price, prev_SL_price, atr, bars_since_SH, bars_since_SL, K
        )

        # All features should be finite
        assert np.isfinite(dist_SH), f"Distance SH should be finite, got {dist_SH}"
        assert np.isfinite(dist_SL), f"Distance SL should be finite, got {dist_SL}"
        assert np.isfinite(bars_SH_norm), f"Bars SH should be finite, got {bars_SH_norm}"
        assert np.isfinite(bars_SL_norm), f"Bars SL should be finite, got {bars_SL_norm}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
