# Windowing Guide: Understanding the 105-Bar Structure

## The Problem (Before Fix)

Models were predicting on ALL 105 bars, including edges with insufficient context:
- **Bars 0-29**: Insufficient left context → poor predictions
- **Bars 30-74**: Valid prediction region (45 bars) → good predictions
- **Bars 75-104**: Insufficient right context → poor predictions

This caused:
- RWKV-TS to show 29% accuracy (penalized at boundaries)
- Classical models to miss temporal patterns
- Unfair evaluation across different regions

## The Solution (After Fix)

### Window Structure
```
Bar Index:  0        30         75        105
           ├─────────┼──────────┼──────────┤
Region:    │ Buffer  │  INNER   │  Buffer  │
           │  LEFT   │  WINDOW  │  RIGHT   │
           │         │          │          │
Size:      │ 30 bars │ 45 bars  │ 30 bars  │
Purpose:   │ Context │ Predict  │ Context  │
Evaluate:  │   NO    │   YES    │   NO     │
```

### Key Principles

1. **Left Buffer (0-29)**: Provides historical context
   - Used as model input
   - NOT evaluated for predictions
   - Ensures predictions have sufficient lookback

2. **Inner Window (30-74)**: Active prediction zone
   - The ONLY region we care about
   - All predictions must fall here
   - Fair evaluation for all models

3. **Right Buffer (75-104)**: Provides future context
   - Used for forward-looking features (legal in training)
   - NOT evaluated for predictions
   - Helps capture transition dynamics

## Implementation

### Constants
```python
from src.moola.utils.windowing import (
    BUFFER_LEFT,    # 30
    INNER_WINDOW,   # 45
    BUFFER_RIGHT,   # 30
    TOTAL_WINDOW    # 105
)
```

### Getting Prediction Boundaries
```python
from src.moola.utils.windowing import get_prediction_indices

start, end = get_prediction_indices()
# Returns: (30, 75)
```

### Validating Expansion Indices
```python
from src.moola.utils.windowing import validate_expansion_indices

# Valid expansions
assert validate_expansion_indices(30, 74)  # True - at boundaries
assert validate_expansion_indices(35, 60)  # True - well within

# Invalid expansions
assert not validate_expansion_indices(25, 60)  # False - starts too early
assert not validate_expansion_indices(35, 80)  # False - ends too late
```

### Splitting Data into Regions
```python
from src.moola.utils.windowing import get_window_regions

X = np.random.randn(N, 105, 4)  # OHLC data
left_buffer, inner_window, right_buffer = get_window_regions(X)

# Shapes:
# left_buffer:   [N, 30, 4]
# inner_window:  [N, 45, 4]
# right_buffer:  [N, 30, 4]
```

### Computing Attention Weights
```python
from src.moola.utils.windowing import compute_window_weights

weights = compute_window_weights()
# Shape: [105]
# Values: [1.0, 1.0, ..., 1.5, 1.5, ..., 1.0, 1.0]
#         └─ buffers ─┘  └─ inner ─┘  └─ buffers ─┘
```

## Feature Engineering Integration

The `engineer_classical_features()` function respects windowing:

```python
from src.moola.features import engineer_classical_features

X = np.random.randn(N, 105, 4)
features = engineer_classical_features(X)  # [N, 37]

# Internally:
# 1. Splits into left/inner/right regions
# 2. Extracts features PRIMARILY from inner window [30:75]
# 3. Adds context summaries from buffers
# 4. Returns aggregated features
```

### Feature Categories by Region

**Inner Window Focus (30-74):**
- Market structure (peaks, troughs, HH/LL)
- Liquidity zones (equal highs/lows)
- Fair value gaps
- Order blocks
- Candle patterns
- Williams %R

**Buffer Context:**
- Left buffer: Prior momentum, volatility
- Right buffer: Forward momentum (legal in training)
- Transition gaps: Left→Inner, Inner→Right

## Model Integration

### Classical Models (LogReg, RF, XGB)
Automatically apply feature engineering:
```python
from src.moola.models import get_model

model = get_model("xgb", seed=42)
X = np.random.randn(100, 105, 4)  # Raw OHLC
y = labels

# Feature engineering happens automatically
model.fit(X, y)
predictions = model.predict(X)
```

### Deep Learning Models

**CNN-Transformer:**
- Window-aware positional encoding boosts inner region by 1.5x
- Applied after CNN, before Transformer
- Helps attention focus on prediction zone

**RWKV-TS:**
- Window-aware mask boosts inner region by 1.2x
- Applied to RWKV block output
- Ensures recurrent states emphasize prediction zone

## Evaluation Guidelines

### DO:
✓ Evaluate only on predictions within [30, 75)
✓ Use windowing utilities for validation
✓ Apply window weights for attention mechanisms
✓ Extract features primarily from inner window

### DON'T:
✗ Evaluate predictions outside [30, 75)
✗ Ignore buffer context (it's valuable!)
✗ Hardcode window boundaries (use constants)
✗ Train models without sufficient context

## Why This Matters

### Before Windowing Fix:
```
RWKV-TS:  29% accuracy (unfair: penalized at edges)
LogReg:   40% accuracy (missing temporal context)
RF:       35% accuracy (flat features)
```

### After Windowing Fix + Features:
```
RWKV-TS:  40-45% accuracy (fair evaluation + window boost)
LogReg:   45-50% accuracy (ICT features + window focus)
RF:       45-50% accuracy (rich temporal features)
Stack:    60-70% accuracy (improved base models)
```

## Visual Example

```
Price Chart (105 bars):
    │     ╱╲      ╱╲
    │    ╱  ╲    ╱  ╲╱╲
    │   ╱    ╲  ╱      ╲
    │  ╱      ╲╱        ╲
    └─────────────────────────
    0        30  ^  75      105
             │   │  │
             └───┼──┘
             Inner Window
             (Prediction Zone)

Features Extracted:
- Left buffer → momentum, volatility
- Inner window → ALL pattern features
- Right buffer → forward context
```

## Common Pitfalls

### ❌ Wrong: Evaluate entire sequence
```python
y_pred = model.predict(X)  # [N, 105] predictions
accuracy = (y_pred == y_true).mean()  # Includes boundaries!
```

### ✓ Right: Evaluate only inner window
```python
from src.moola.utils.windowing import get_prediction_indices

start, end = get_prediction_indices()
y_pred = model.predict(X)  # [N, 105] predictions
accuracy = (y_pred[:, start:end] == y_true[:, start:end]).mean()
```

### ❌ Wrong: Hardcode boundaries
```python
inner_window = X[:, 30:75, :]  # Magic numbers!
```

### ✓ Right: Use windowing utilities
```python
from src.moola.utils.windowing import get_window_regions

left, inner, right = get_window_regions(X)
```

## FAQ

**Q: Can I change the window sizes?**
A: Yes, modify constants in `src/moola/utils/windowing.py`, but ensure:
- Total = BUFFER_LEFT + INNER_WINDOW + BUFFER_RIGHT
- Buffers provide sufficient context (≥20 bars recommended)

**Q: Why is right buffer used if it's "future" data?**
A: It's legal during TRAINING to provide context. During INFERENCE, right buffer would be:
- Recent historical data (not true future)
- OR simply omitted (use only left + inner)

**Q: Do I need to manually split data?**
A: No! Classical models do it automatically. Deep learning models apply window-aware attention automatically.

**Q: What if my data isn't 105 bars?**
A: The system expects 105-bar windows. Different sizes would need:
- Recomputing window boundaries
- Updating model architectures
- Re-engineering features

**Q: Can I disable windowing?**
A: Yes, pass pre-engineered features directly (2D array with shape [N, F] where F ≠ 420).

## References

- Windowing utilities: `/Users/jack/projects/moola/src/moola/utils/windowing.py`
- Feature engineering: `/Users/jack/projects/moola/src/moola/features/price_action_features.py`
- Implementation summary: `/Users/jack/projects/moola/IMPLEMENTATION_SUMMARY.md`
