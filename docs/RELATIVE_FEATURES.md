# Relative Feature Transformation

## Overview

The `RelativeFeatureTransform` class converts absolute OHLC (Open-High-Low-Close) price data into scale-invariant relative features. This transformation enables machine learning models to generalize better across different price ranges and market conditions.

## Transformation Pipeline

Input: `[N, 105, 4]` OHLC data
Output: `[N, 105, 11]` relative features

### Feature Groups

#### 1. Log Returns (4 features)
- `log_return_open`: log(open_t / open_t-1)
- `log_return_high`: log(high_t / high_t-1)
- `log_return_low`: log(low_t / low_t-1)
- `log_return_close`: log(close_t / close_t-1)

**Properties:**
- Scale-invariant: captures percentage changes rather than absolute changes
- First bar (t=0) is zero (no previous price available)
- Typically ranges from -0.1 to +0.1 for normal market moves

#### 2. Candle Ratios (3 features)
- `body_ratio`: abs(close - open) / range
- `upper_wick_ratio`: (high - max(open, close)) / range
- `lower_wick_ratio`: (min(open, close) - low) / range

**Properties:**
- All values in [0, 1] range
- Sum of all three ratios approximately equals 1
- Captures candle shape independent of price level
- Doji candles have body_ratio ≈ 0

#### 3. Rolling Z-Scores (4 features)
- `zscore_open`: standardized open price over 20-bar window
- `zscore_high`: standardized high price over 20-bar window
- `zscore_low`: standardized low price over 20-bar window
- `zscore_close`: standardized close price over 20-bar window

**Properties:**
- Captures deviation from recent mean
- Clipped to [-10, 10] to handle extreme outliers
- First bar (t=0) is zero (no history available)
- For t < 20, uses all available history

## Usage

```python
from moola.features.relative_transform import RelativeFeatureTransform

# Initialize transformer
transform = RelativeFeatureTransform(eps=1e-8)

# Transform OHLC data
X_ohlc = load_ohlc_data()  # Shape: [N, 105, 4]
X_relative = transform.transform(X_ohlc)  # Shape: [N, 105, 11]

# Get feature names
feature_names = transform.get_feature_names()
```

## Edge Cases Handled

1. **First bar (no previous price):** Log returns and z-scores are set to zero
2. **Division by zero (zero range):** Epsilon (1e-8) added to denominators
3. **NaN values:** Replaced with zeros after computation
4. **Extreme values:** Z-scores clipped to [-10, 10]
5. **Negative prices:** Epsilon added before log to prevent log(0)

## Benefits for ML Models

1. **Scale invariance:** Model trained on $50 stocks works on $500 stocks
2. **Improved generalization:** Reduces overfitting to specific price ranges
3. **Numerical stability:** All features have reasonable ranges
4. **Captures patterns:** Focuses on relative relationships rather than absolute values
5. **Market regime agnostic:** Works across different volatility regimes

## Implementation Details

- **Vectorized operations:** Uses NumPy broadcasting for efficiency
- **Memory efficient:** In-place operations where possible
- **Type safe:** Comprehensive input validation
- **Deterministic:** Same input always produces same output
- **Production ready:** Extensive error handling and logging

## Testing

Comprehensive test suite covers:
- Input validation (shape, dtype, type)
- Edge cases (zero range, first bar, NaN values)
- Numerical accuracy (log returns, ratios, z-scores)
- Data quality (no NaN/inf in output)
- Batch processing (various batch sizes)
- Determinism (reproducible results)

Run tests:
```bash
python3 -m pytest tests/test_relative_transform.py -v
```

## Example Output

For typical stock price data ($150 ± $5):

| Feature | Min | Max | Mean | Std |
|---------|-----|-----|------|-----|
| log_return_open | -0.11 | +0.10 | 0.00 | 0.04 |
| body_ratio | 0.00 | 0.95 | 0.15 | 0.24 |
| upper_wick_ratio | 0.00 | 0.98 | 0.51 | 0.29 |
| lower_wick_ratio | 0.00 | 1.00 | 0.53 | 0.30 |
| zscore_open | -2.39 | +2.49 | 0.05 | 0.98 |

## See Also

- `examples/demo_relative_transform.py` - Interactive demonstration
- `tests/test_relative_transform.py` - Test suite
- `src/moola/features/relative_transform.py` - Implementation
