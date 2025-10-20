# Automatic Structure Label Generation

## Overview

The `generate_structure_labels.py` script automatically generates three types of structural labels for unlabeled OHLC time-series data. These labels capture market microstructure patterns that can be used for:

- **Pre-training tasks** (e.g., predict next swing point, detect expansion)
- **Feature engineering** (encode structural information as input features)
- **Data analysis** (understand market behavior patterns)
- **Validation** (sanity-check data quality)

## Label Types

### 1. Expansion Detection (Binary)

**Purpose:** Identify bars where price breaks out significantly beyond recent range.

**Algorithm:**
- Computes Adaptive ATR (Average True Range) with 14-bar window
- `epsilon = epsilon_factor × ATR` (default: 2.0 × ATR)
- For each bar, checks if high/low exceeds recent 5-bar range by more than epsilon
- Labels: 0 = no expansion, 1 = expansion

**Characteristics:**
- Adaptive to current volatility (ATR-based threshold)
- Sensitive to breakouts (uses high/low, not just close)
- Rare events (~0.33% of bars on real data)

**Use cases:**
- Detect significant price movements
- Identify trend initiation points
- Filter consolidation vs trending periods

### 2. Swing Point Classification (3-class)

**Purpose:** Identify local price extrema (support/resistance levels).

**Algorithm:**
- Uses sliding window (default: 5 bars = 2 before + current + 2 after)
- Swing high: bar's high is maximum in window
- Swing low: bar's low is minimum in window
- Labels: 0 = neither, 1 = swing high, 2 = swing low

**Characteristics:**
- Window-based local extrema detection
- ~14% swing highs, ~14% swing lows on real data
- Depends on window size (larger = fewer, stronger swings)

**Use cases:**
- Detect support/resistance levels
- Identify reversal points
- Structural features for pattern recognition

### 3. Candlestick Pattern (4-class)

**Purpose:** Classify candles by body size and direction.

**Algorithm:**
- `body_size = |close - open|`
- `total_range = high - low`
- `body_ratio = body_size / total_range`
- Classification:
  - Doji: `body_ratio < doji_threshold` (default: 0.1)
  - Neutral: `body_ratio < 0.3` and not doji
  - Bullish: `close > open` and not neutral/doji
  - Bearish: `close < open` and not neutral/doji
- Labels: 0 = bullish, 1 = bearish, 2 = neutral, 3 = doji

**Characteristics:**
- Based on traditional candlestick analysis
- ~35% bullish, ~34% bearish, ~20% neutral, ~10% doji on real data
- Captures indecision (doji/neutral) vs directional moves

**Use cases:**
- Encode price action patterns
- Detect indecision zones
- Simple directional features

## Installation

No additional dependencies required beyond the moola project base:

```bash
# Verify dependencies
python3 -c "import numpy, pandas, loguru"
```

## Usage

### Basic Usage

```bash
python3 scripts/generate_structure_labels.py \
    --input data/raw/unlabeled_windows.parquet \
    --output data/processed/unlabeled_with_labels.parquet
```

### Custom Parameters

```bash
# More sensitive expansion detection (lower epsilon)
python3 scripts/generate_structure_labels.py \
    --input data/raw/unlabeled_windows.parquet \
    --output data/processed/unlabeled_sensitive.parquet \
    --epsilon-factor 1.5 \
    --swing-window 7 \
    --doji-threshold 0.05
```

```bash
# Conservative (fewer detections)
python3 scripts/generate_structure_labels.py \
    --input data/raw/unlabeled_windows.parquet \
    --output data/processed/unlabeled_conservative.parquet \
    --epsilon-factor 3.0 \
    --swing-window 3 \
    --doji-threshold 0.15
```

### With Validation

```bash
python3 scripts/generate_structure_labels.py \
    --input data/raw/unlabeled_windows.parquet \
    --output data/processed/unlabeled_with_labels.parquet \
    --validate
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input` | Path | Required | Input parquet file with unlabeled windows |
| `--output` | Path | Required | Output parquet file for labeled data |
| `--epsilon-factor` | float | 2.0 | ATR multiplier for expansion detection. Lower = more sensitive |
| `--swing-window` | int | 5 | Window size for swing point detection. Must be odd and ≥ 3 |
| `--doji-threshold` | float | 0.1 | Body ratio threshold for doji (0.0-1.0). Lower = stricter |
| `--validate` | flag | False | Run validation checks after generation |

## Input Format

Parquet file with columns:
- `window_id`: str, unique identifier
- `features`: numpy array of shape (105, 4) containing OHLC data

Example:
```python
import pandas as pd
df = pd.read_parquet('data/raw/unlabeled_windows.parquet')
# df columns: ['window_id', 'features']
# df['features'][0].shape = (105, 4)  # 105 bars × 4 OHLC values
```

## Output Format

Parquet file with columns:
- `window_id`: str, unique identifier
- `features`: numpy array (105, 4) - original OHLC data
- `expansion_labels`: numpy array (105,) - binary expansion labels
- `swing_labels`: numpy array (105,) - 3-class swing point labels
- `candle_labels`: numpy array (105,) - 4-class candlestick labels

Example:
```python
import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/unlabeled_with_labels.parquet')
sample = df.iloc[0]

print(sample['window_id'])            # "window_0"
print(sample['expansion_labels'])     # array([0, 0, 1, 0, ...])
print(sample['swing_labels'])         # array([0, 1, 0, 2, ...])
print(sample['candle_labels'])        # array([0, 1, 3, 0, ...])
```

## Performance

- **Dataset:** 11,873 windows × 105 bars = 1,246,665 bars
- **Processing time:** ~5 seconds on MacBook Pro M1
- **Memory usage:** ~6 MB parquet file (compressed)
- **Vectorized:** All operations use NumPy vectorization
- **Deterministic:** Same input always produces same output

## Testing

Run the test suite to verify correctness:

```bash
python3 scripts/test_structure_labels.py
```

Tests cover:
- ATR computation accuracy
- Expansion detection sensitivity
- Swing point detection correctness
- Candlestick pattern classification
- Determinism (reproducibility)
- Label value ranges

Expected output:
```
================================================================================
STRUCTURE LABEL GENERATION TESTS
================================================================================

Testing compute_atr()...
✓ compute_atr() passed all tests

Testing label_expansion()...
✓ label_expansion() passed all tests

Testing label_swing_points()...
✓ label_swing_points() passed all tests

Testing label_candlestick()...
✓ label_candlestick() passed all tests

Testing determinism...
✓ All algorithms are deterministic

Testing label value ranges...
✓ All label ranges are valid

================================================================================
TEST RESULTS: 6 passed, 0 failed
================================================================================
```

## Integration with Moola Pipeline

### Pre-training Tasks

Use labels as supervision for pre-training:

```python
from moola.models.masked_bilstm import MaskedBiLSTM

# Load labeled data
df = pd.read_parquet('data/processed/unlabeled_with_labels.parquet')

# Task 1: Predict next swing point
X = features  # (N, 105, 4)
y = swing_labels  # (N, 105)

# Task 2: Detect expansion bars
X = features
y = expansion_labels

# Task 3: Predict candlestick pattern
X = features
y = candle_labels
```

### Feature Engineering

Add structural features to SimpleLSTM input:

```python
import numpy as np

# Load data
df = pd.read_parquet('data/processed/unlabeled_with_labels.parquet')

# Combine OHLC + structural labels as features
features = np.concatenate([
    df['features'],           # (N, 105, 4) OHLC
    df['expansion_labels'],   # (N, 105, 1) binary
    df['swing_labels'],       # (N, 105, 1) 3-class
    df['candle_labels'],      # (N, 105, 1) 4-class
], axis=-1)  # Result: (N, 105, 7)

# Train SimpleLSTM with enriched features
model.fit(features, labels)
```

### Data Analysis

Understand market microstructure:

```python
import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/unlabeled_with_labels.parquet')

# Find windows with expansions
exp_windows = df[df['expansion_labels'].apply(lambda x: np.array(x).sum() > 0)]
print(f"Expansion windows: {len(exp_windows)} / {len(df)} ({len(exp_windows)/len(df):.1%})")

# Analyze swing point frequency
swing_labels = np.concatenate([np.array(x) for x in df['swing_labels']])
high_rate = (swing_labels == 1).sum() / len(swing_labels)
low_rate = (swing_labels == 2).sum() / len(swing_labels)
print(f"Swing high rate: {high_rate:.2%}")
print(f"Swing low rate: {low_rate:.2%}")

# Candlestick distribution
candle_labels = np.concatenate([np.array(x) for x in df['candle_labels']])
for i, name in enumerate(['bullish', 'bearish', 'neutral', 'doji']):
    rate = (candle_labels == i).sum() / len(candle_labels)
    print(f"{name.capitalize()}: {rate:.2%}")
```

## Troubleshooting

### Issue: Expansion rate too low (< 0.1%)

**Solution:** Lower `--epsilon-factor` (e.g., 1.5 or 1.0)

```bash
python3 scripts/generate_structure_labels.py \
    --epsilon-factor 1.5 \
    --input ... --output ...
```

### Issue: Too many swing points detected

**Solution:** Increase `--swing-window` (e.g., 7 or 9)

```bash
python3 scripts/generate_structure_labels.py \
    --swing-window 7 \
    --input ... --output ...
```

### Issue: Not enough dojis detected

**Solution:** Increase `--doji-threshold` (e.g., 0.15)

```bash
python3 scripts/generate_structure_labels.py \
    --doji-threshold 0.15 \
    --input ... --output ...
```

### Issue: Label length mismatch error

**Cause:** Input data has wrong shape (not 105 bars per window)

**Solution:** Verify input data format:

```python
import pandas as pd
df = pd.read_parquet('data/raw/unlabeled_windows.parquet')
print(df['features'][0].shape)  # Should be (105, 4)
```

## Implementation Details

### Vectorization Strategy

All algorithms use NumPy vectorization for efficiency:

```python
# Example: Swing point detection (vectorized)
for t in range(half_window, T - half_window):
    window_high = high[:, t - half_window : t + half_window + 1]
    is_swing_high = high[:, t] == window_high.max(axis=1)  # Vectorized across all windows
    swing_labels[:, t] = np.where(is_swing_high, 1, 0)
```

### Memory Efficiency

- **Streaming:** Processes all windows in single pass
- **Dtype optimization:** Uses `int32` for labels (not `int64`)
- **Compression:** Saves output with Snappy compression

### Determinism

No randomness used anywhere:
- No random seeds
- No stochastic algorithms
- Same input → same output (guaranteed)

## References

- **ATR (Average True Range):** Wilder, J. W. (1978). New Concepts in Technical Trading Systems.
- **Swing Points:** Elder, A. (1993). Trading for a Living.
- **Candlestick Patterns:** Nison, S. (1991). Japanese Candlestick Charting Techniques.

## License

Part of the Moola project. See project LICENSE for details.

## Support

Questions? Issues?

1. Check this README first
2. Run `python3 scripts/test_structure_labels.py` to verify setup
3. Check logs in `logs/generate_structure_labels_*.log`
4. Review code comments in `scripts/generate_structure_labels.py`

## Changelog

### v1.0.0 (2025-10-17)

- Initial release
- Three label types: expansion, swing points, candlesticks
- Adaptive ATR-based expansion detection
- Vectorized NumPy implementation
- Comprehensive test suite
- CLI interface with validation
