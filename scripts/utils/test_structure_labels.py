"""Test script for structure label generation validation.

This script performs unit tests and integration tests on the label generation
algorithms to ensure correctness and consistency.

Usage:
    python3 scripts/test_structure_labels.py
"""

import numpy as np
from loguru import logger

# Import the functions we want to test
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_structure_labels import (
    compute_atr,
    label_candlestick,
    label_expansion,
    label_swing_points,
)


def test_compute_atr():
    """Test ATR computation with known values."""
    logger.info("Testing compute_atr()...")

    # Create simple test data: flat prices (no volatility)
    ohlc_flat = np.ones((2, 20, 4), dtype=np.float32)
    ohlc_flat[:, :, 0] = 100  # open
    ohlc_flat[:, :, 1] = 100  # high
    ohlc_flat[:, :, 2] = 100  # low
    ohlc_flat[:, :, 3] = 100  # close

    atr_flat = compute_atr(ohlc_flat, window=14)
    assert atr_flat.shape == (2, 20), f"Expected shape (2, 20), got {atr_flat.shape}"
    assert np.allclose(atr_flat, 0.0), "ATR should be 0 for flat prices"

    # Create data with increasing volatility
    ohlc_volatile = np.zeros((1, 20, 4), dtype=np.float32)
    for t in range(20):
        ohlc_volatile[0, t] = [100 + t, 105 + t, 95 + t, 100 + t]

    atr_volatile = compute_atr(ohlc_volatile, window=5)
    assert atr_volatile.shape == (1, 20), f"Expected shape (1, 20), got {atr_volatile.shape}"
    # ATR should be approximately 10 (range of high-low)
    assert np.all(atr_volatile[0, 5:] >= 9.0), "ATR should be ~10 for volatile data"
    assert np.all(atr_volatile[0, 5:] <= 11.0), "ATR should be ~10 for volatile data"

    logger.success("compute_atr() passed all tests")


def test_label_expansion():
    """Test expansion detection with controlled scenarios."""
    logger.info("Testing label_expansion()...")

    # Test 1: No expansion (consolidating prices)
    ohlc_consolidation = np.ones((1, 50, 4), dtype=np.float32)
    for t in range(50):
        # Small random walk
        base = 100 + np.sin(t * 0.1) * 2
        ohlc_consolidation[0, t] = [base, base + 0.5, base - 0.5, base + 0.25]

    exp_labels = label_expansion(ohlc_consolidation, epsilon_factor=2.0)
    expansion_rate = exp_labels.sum() / exp_labels.size
    assert expansion_rate < 0.1, f"Expected <10% expansion in consolidation, got {expansion_rate:.1%}"

    # Test 2: Clear expansion (big breakout)
    ohlc_breakout = np.zeros((1, 50, 4), dtype=np.float32)
    for t in range(30):
        ohlc_breakout[0, t] = [100, 101, 99, 100]  # Consolidation
    for t in range(30, 50):
        price = 100 + (t - 30) * 5  # Strong uptrend
        ohlc_breakout[0, t] = [price, price + 2, price - 1, price + 1]

    exp_labels = label_expansion(ohlc_breakout, epsilon_factor=0.5)  # Lower threshold for test
    expansion_rate_breakout = exp_labels[0, 30:].sum() / exp_labels[0, 30:].size
    assert expansion_rate_breakout > 0.3, f"Expected >30% expansion in breakout, got {expansion_rate_breakout:.1%}"

    logger.success("label_expansion() passed all tests")


def test_label_swing_points():
    """Test swing point detection with known patterns."""
    logger.info("Testing label_swing_points()...")

    # Create a simple pattern with clear swing points
    ohlc_pattern = np.zeros((1, 20, 4), dtype=np.float32)

    # Pattern: low -> high -> low -> high
    prices = [100, 102, 104, 106, 108, 106, 104, 102, 100, 102, 104, 106, 108, 110, 108, 106, 104, 102, 100, 98]

    for t, price in enumerate(prices):
        ohlc_pattern[0, t] = [price, price + 1, price - 1, price]

    swing_labels = label_swing_points(ohlc_pattern, window=5)

    # Check that we detected some swing highs and lows
    num_swing_highs = (swing_labels == 1).sum()
    num_swing_lows = (swing_labels == 2).sum()

    assert num_swing_highs > 0, "Should detect at least one swing high"
    assert num_swing_lows > 0, "Should detect at least one swing low"

    # Verify that swing high is at local maximum
    swing_high_indices = np.where(swing_labels[0] == 1)[0]
    for idx in swing_high_indices:
        if idx >= 2 and idx < len(prices) - 2:
            # Check it's higher than neighbors
            assert prices[idx] >= prices[idx - 1], f"Swing high at {idx} should be >= prev"
            assert prices[idx] >= prices[idx + 1], f"Swing high at {idx} should be >= next"

    logger.success("label_swing_points() passed all tests")


def test_label_candlestick():
    """Test candlestick pattern classification."""
    logger.info("Testing label_candlestick()...")

    # Create test cases for each pattern type
    ohlc_patterns = np.zeros((4, 10, 4), dtype=np.float32)

    # Pattern 0: Bullish candles (close > open)
    for t in range(10):
        ohlc_patterns[0, t] = [100, 105, 99, 104]  # Bullish

    # Pattern 1: Bearish candles (close < open)
    for t in range(10):
        ohlc_patterns[1, t] = [104, 105, 99, 100]  # Bearish

    # Pattern 2: Neutral candles (small body)
    for t in range(10):
        ohlc_patterns[2, t] = [100, 105, 99, 101]  # Small body

    # Pattern 3: Doji candles (open ≈ close)
    for t in range(10):
        ohlc_patterns[3, t] = [100, 102, 98, 100]  # Doji

    candle_labels = label_candlestick(ohlc_patterns, doji_threshold=0.1)

    # Check that most candles are classified correctly
    bullish_pct = (candle_labels[0] == 0).sum() / candle_labels[0].size
    bearish_pct = (candle_labels[1] == 1).sum() / candle_labels[1].size
    doji_pct = (candle_labels[3] == 3).sum() / candle_labels[3].size

    assert bullish_pct >= 0.5, f"Expected >=50% bullish in row 0, got {bullish_pct:.1%}"
    assert bearish_pct >= 0.5, f"Expected >=50% bearish in row 1, got {bearish_pct:.1%}"
    assert doji_pct >= 0.5, f"Expected >=50% doji in row 3, got {doji_pct:.1%}"

    logger.success("label_candlestick() passed all tests")


def test_determinism():
    """Test that algorithms are deterministic (same input → same output)."""
    logger.info("Testing determinism...")

    # Create random test data
    np.random.seed(42)
    ohlc = np.random.randn(10, 50, 4).astype(np.float32)
    ohlc = np.abs(ohlc) * 100 + 100  # Make positive prices

    # Ensure OHLC constraints
    for i in range(10):
        for t in range(50):
            o, h, l, c = ohlc[i, t]
            ohlc[i, t, 1] = max(o, h, l, c)  # high
            ohlc[i, t, 2] = min(o, h, l, c)  # low

    # Run twice
    exp1 = label_expansion(ohlc, epsilon_factor=2.0)
    exp2 = label_expansion(ohlc, epsilon_factor=2.0)

    swing1 = label_swing_points(ohlc, window=5)
    swing2 = label_swing_points(ohlc, window=5)

    candle1 = label_candlestick(ohlc, doji_threshold=0.1)
    candle2 = label_candlestick(ohlc, doji_threshold=0.1)

    # Check they're identical
    assert np.array_equal(exp1, exp2), "Expansion labels not deterministic"
    assert np.array_equal(swing1, swing2), "Swing labels not deterministic"
    assert np.array_equal(candle1, candle2), "Candle labels not deterministic"

    logger.success("All algorithms are deterministic")


def test_label_ranges():
    """Test that all labels are in valid ranges."""
    logger.info("Testing label value ranges...")

    # Create random test data
    np.random.seed(123)
    ohlc = np.random.randn(5, 30, 4).astype(np.float32)
    ohlc = np.abs(ohlc) * 100 + 100

    # Ensure OHLC constraints
    for i in range(5):
        for t in range(30):
            o, h, l, c = ohlc[i, t]
            ohlc[i, t, 1] = max(o, h, l, c)
            ohlc[i, t, 2] = min(o, h, l, c)

    exp_labels = label_expansion(ohlc)
    swing_labels = label_swing_points(ohlc)
    candle_labels = label_candlestick(ohlc)

    # Check ranges
    assert np.all((exp_labels >= 0) & (exp_labels <= 1)), "Expansion labels out of range [0, 1]"
    assert np.all((swing_labels >= 0) & (swing_labels <= 2)), "Swing labels out of range [0, 2]"
    assert np.all((candle_labels >= 0) & (candle_labels <= 3)), "Candle labels out of range [0, 3]"

    # Check they're integers
    assert exp_labels.dtype == np.int32, "Expansion labels should be int32"
    assert swing_labels.dtype == np.int32, "Swing labels should be int32"
    assert candle_labels.dtype == np.int32, "Candle labels should be int32"

    logger.success("All label ranges are valid")


def run_all_tests():
    """Run all test functions."""
    logger.info("=" * 80)
    logger.info("STRUCTURE LABEL GENERATION TESTS")
    logger.info("=" * 80)
    logger.info("")

    tests = [
        test_compute_atr,
        test_label_expansion,
        test_label_swing_points,
        test_label_candlestick,
        test_determinism,
        test_label_ranges,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
            logger.info("")
        except AssertionError as e:
            logger.error(f"Test failed: {test_func.__name__}")
            logger.error(f"  {e}")
            failed += 1
            logger.info("")
        except Exception as e:
            logger.error(f"Test error: {test_func.__name__}")
            logger.exception(e)
            failed += 1
            logger.info("")

    logger.info("=" * 80)
    logger.info(f"TEST RESULTS: {passed} passed, {failed} failed")
    logger.info("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
