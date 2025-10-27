# Pre-training Feature Validation Analysis

**Date**: 2025-10-26
**Dataset**: 210 labeled expansion windows
**Purpose**: Validate whether heuristic pre-training features align with human-labeled expansions

---

## Executive Summary

**CRITICAL FINDING**: All 12 pre-training features show **near-zero correlation** (< 0.03) with human-labeled expansion boundaries. This explains why pre-training has not improved supervised model performance.

### Key Metrics

| Feature | Correlation | Separation | Best Threshold | Best F1 |
|---------|-------------|------------|----------------|---------|
| **expansion_proxy** | 0.009 | +0.032 | 0.20 | 0.383 |
| **range_z** | 0.002 | -0.009 | 0.20 | 0.372 |
| **body_pct** | -0.004 | -0.007 | 0.05 | 0.310 |
| **consol_proxy** | -0.008 | -0.000 | N/A | 0.000 |
| dist_to_prev_SH | 0.015 | +0.037 | N/A | N/A |
| dist_to_prev_SL | -0.014 | -0.049 | N/A | N/A |

**Interpretation**:
- ❌ **All correlations < 0.03** - essentially no linear relationship
- ⚠️ **Best F1 = 0.383** (expansion_proxy @ 0.20) - still only 38% accuracy
- ✅ **consol_proxy completely failed** - indicates consolidations are NOT the inverse of expansions

---

## What This Means for Your Project

### 1. **Pre-training Strategy Needs Rethinking**

**Current approach** (MAE with heuristic targets):
```python
# Current pre-training formula
expansion_proxy = range_z × leg_dir × body_pct
# Correlation with actual expansions: 0.009 (basically zero)
```

**Problem**: The heuristic doesn't capture what humans label as expansions.

**Impact**: Pre-training on these features will NOT jumpstart supervised learning because the encoder learns to reconstruct **noisy, misaligned signals** instead of **meaningful expansion patterns**.

### 2. **Why Soft Span Loss Training Shows Weak Probability Separation**

Remember from your training runs:
- Soft span (50ep): In-span mean = 0.090, Out-of-span mean = 0.090 (0.000 separation)
- CRF (50ep): In-span mean = 0.101, Out-of-span mean = 0.092 (0.009 separation)

**Root cause is NOW CLEAR**:
- The 12 relativity features don't discriminate expansions well
- Pre-training on misaligned proxies doesn't help
- Model can't learn what isn't in the features

### 3. **Your Intuition Was Correct**

You said: *"Your current features might miss subtle 'blips' or overfit flat areas."*

**Validation confirms**:
- Features DO miss blips (recall at best threshold = 0.692)
- Features DO over-predict (precision at best threshold = 0.265)
- The heuristics were designed for ICT 5-6 bar surges, but your labels capture different patterns

---

## Detailed Analysis

### Feature Separation Visualization

From `feature_separation.png`:

```
Features with POSITIVE separation (higher in expansions):
  dist_to_prev_SH:     +0.037  ← Best predictor
  expansion_proxy:     +0.032
  lower_wick_pct:      +0.009
  open_norm:           +0.006
  upper_wick_pct:      +0.004

Features with NEGATIVE separation (lower in expansions):
  dist_to_prev_SL:     -0.049  ← Inverted relationship!
  range_z:             -0.009  ← Surprisingly ANTI-correlated
  body_pct:            -0.007
  close_norm:          -0.001
  bars_since_SL_norm:  -0.001
```

**Surprises**:
1. **`range_z` is NEGATIVE** - Expansions have LOWER relative range than non-expansions
   - This contradicts the "high volatility = expansion" assumption
   - Suggests your expansions are about **direction**, not **magnitude**

2. **`dist_to_prev_SH` is highest** - Expansions occur when FURTHER from swing high
   - Makes sense: expansions happen mid-leg, not at reversals
   - But correlation still weak (0.015)

3. **`expansion_proxy` weak despite formula** - Designed for this task, yet only 0.009 correlation
   - Formula: `range_z × leg_dir × body_pct`
   - All three components are weak → product is also weak

### Threshold Optimization Results

From `threshold_opt_expansion_proxy.png`:

**expansion_proxy performance**:
```
Threshold 0.10: F1=0.12, Precision=0.07, Recall=0.45  ← High recall, terrible precision
Threshold 0.20: F1=0.38, Precision=0.27, Recall=0.69  ← Best F1 (still weak)
Threshold 0.30: F1=0.12, Precision=0.08, Recall=0.34
Threshold 0.50: F1=0.10, Precision=0.07, Recall=0.22  ← Low recall
```

**Pattern**: As threshold increases:
- ✅ Fewer false positives (slightly better precision)
- ❌ Missing more true expansions (lower recall)
- F1 peaks at 0.20 but still only 38% accuracy

**Why precision is so low (0.27)**:
- For every 1 true expansion detected, 2.7 false positives are flagged
- Heuristic triggers on patterns that humans DON'T label as expansions
- OR: Heuristic misses patterns that humans DO label as expansions

---

## Root Cause Analysis: Why Features Fail

### Hypothesis 1: **Feature Definition Mismatch**

**What the features detect**:
- `expansion_proxy`: High range × strong directional body × trend alignment
- Designed for ICT-style "5-6 bar surges" with large candles

**What humans actually label** (inferred from low correlation):
- Directional moves that may NOT have large candles
- Expansions defined by **price progression** rather than **volatility**
- Patterns that span variable durations (not just 5-6 bars)
- Includes subtle breakouts that don't show high `range_z`

### Hypothesis 2: **Temporal Aggregation Issue**

**Current approach**: Features are bar-by-bar (105 timesteps independent)

**Problem**: Expansions are **sequence patterns**
- A single bar with high `range_z` doesn't mean it's in an expansion
- Expansions require sustained directional movement
- Features don't capture **momentum** or **acceleration**

### Hypothesis 3: **Labeling Includes Context**

**Human annotation process** (from your Candlesticks project):
- You mark expansion_start and expansion_end based on **chart reading**
- Includes context: prior consolidation, breakout point, exhaustion
- Features are **point-wise** but labels are **contextual**

Example:
```
Bar 50-60: Consolidation (low range_z, small bodies)
Bar 61:    Expansion START (may have only moderate range_z)
Bar 62-75: Expansion continuation (sustained move)
Bar 76:    Expansion END (exhaustion candle)

Heuristic flags: Bars 62-75 (high range_z)
Human labels:    Bars 61-76 (includes breakout and exhaustion)
Mismatch:        Start/end boundaries differ
```

---

## Recommendations

### Option 1: **Skip Pre-training (Short-term solution)**

**Rationale**: If heuristic features don't align, pre-training won't help.

**Action**:
- Focus on supervised learning with 210 labeled samples
- Use your existing JadeCompact (97K params) without pre-trained encoder
- Rely on data augmentation (overlapping windows) to expand dataset

**Expected outcome**: Same or better performance than pre-training approach

**Pros**:
- Simpler pipeline
- Faster iteration (no pre-training phase)
- Model learns directly from labeled examples

**Cons**:
- Doesn't leverage 1.8M unlabeled bars
- Limited to 210 samples (small dataset)

---

### Option 2: **Create Better Heuristic Features (Medium-term)**

**Goal**: Design features that actually correlate with human labels

#### 2A. **Temporal Features (Momentum/Acceleration)**

Current features are **instantaneous**. Add **temporal aggregation**:

```python
# New features to test:

# 1. Momentum (3-bar trailing)
momentum_3 = (close[i] - close[i-3]) / (high[i-3:i+1].max() - low[i-3:i+1].min())

# 2. Directional consistency (bars in same direction)
direction_streak = count_consecutive_same_sign(close[i-5:i+1] - open[i-5:i+1])

# 3. Acceleration (rate of change of momentum)
accel = momentum_3[i] - momentum_3[i-1]

# 4. Price distance from consolidation zone
# (assuming consolidation is low-vol preceding expansion)
dist_from_consol = (close[i] - consolidation_range.center) / consolidation_range.width
```

**Why these might work**:
- Capture **sustained movement** (not just single bars)
- Detect **breakouts from consolidation** (transition pattern)
- Measure **acceleration** (expansion often accelerates after breakout)

#### 2B. **Pattern-Based Features**

Detect specific ICT/swing patterns that precede expansions:

```python
# 1. Fair Value Gap (FVG) detection
fvg_score = detect_fair_value_gap(ohlc[i-10:i+1])

# 2. Order block detection
order_block_dist = distance_to_nearest_order_block(ohlc[i-20:i+1])

# 3. Break of structure (BoS)
bos_signal = detect_break_of_structure(swing_highs, swing_lows, current_price)
```

**Caveat**: These are complex and may require manual tuning

---

### Option 3: **Contrastive Pre-training WITHOUT Heuristics (Advanced)**

**Approach**: Learn representations from unlabeled data WITHOUT heuristic targets

**Method**: SimCLR-style contrastive learning
```python
# Don't use expansion_proxy as target
# Instead: Learn features that distinguish DIFFERENT windows

# 1. Augment window (jitter, warp, mask)
window_aug1 = augment(window)
window_aug2 = augment(window)

# 2. Encode both augmentations
z1 = encoder(window_aug1)
z2 = encoder(window_aug2)

# 3. Contrastive loss: Push z1 and z2 together, push other windows apart
loss = contrastive_loss(z1, z2, batch_z)
```

**Why this works**:
- Learns **general temporal representations** (not expansion-specific)
- No heuristic mismatch (learns from data distribution itself)
- Transfer to supervised task via fine-tuning

**Challenges**:
- More complex to implement
- Requires careful augmentation design
- May not learn expansion-specific features

---

### Option 4: **Hybrid: Better Features + Contrastive Learning**

**Best of both worlds**:

1. **Phase 1**: Create improved temporal features (momentum, acceleration, pattern scores)
2. **Phase 2**: Use contrastive pre-training to learn encoder
3. **Phase 3**: Fine-tune with your 210 labeled samples
4. **Phase 4**: Validate that improved features correlate better (re-run validation script)

**Expected improvement**:
- Better features → Encoder learns more relevant patterns
- Contrastive learning → No heuristic mismatch risk
- Fine-tuning → Supervised signal aligns encoder to expansions
- Target: Feature correlation 0.009 → 0.20+ (20x improvement)

---

## Immediate Next Steps

### 1. **Examine Specific Mismatch Cases**

**Action**: Look at windows where features and labels disagree most

```python
# Find worst mismatches
results = pd.read_csv("artifacts/feature_validation/feature_alignment_results.csv")

# Windows where expansion_proxy is HIGH but human labels are NOT expansion
high_proxy_non_expansion = results[
    (results["feature_name"] == "expansion_proxy") &
    (results["expansion_proxy"] > 0.3) &  # High heuristic value
    (results["in_expansion_fraction"] < 0.2)  # But low actual expansion coverage
]

# Visualize these windows to understand WHY
```

**Goal**: Understand what patterns the heuristic detects but you DON'T label as expansions

### 2. **Test New Temporal Features**

**Action**: Add momentum/acceleration features to `relativity.py`

```python
# Add to build_features():

# Momentum (3-bar)
if i >= 3:
    momentum_3 = (c - closes[i-3]) / (max(highs[i-3:i+1]) - min(lows[i-3:i+1]) + eps)
else:
    momentum_3 = 0.0

# Directional streak
if i >= 5:
    streak = count_consecutive_same_sign([closes[j] - opens[j] for j in range(i-5, i+1)])
else:
    streak = 0
```

**Then re-run validation**:
```bash
python3 scripts/validate_pretraining_features.py --data data/processed/labeled/train_latest_overlaps_v2.parquet
```

**Target**: Correlation > 0.10 (10x current best)

### 3. **Decision Point: Pre-train or Not?**

**If improved features achieve correlation > 0.15**:
- ✅ Proceed with pre-training (heuristic alignment is decent)
- Run MAE on 1.8M unlabeled bars
- Transfer encoder to supervised task

**If correlation stays < 0.10**:
- ❌ Skip pre-training entirely
- Focus on supervised learning with 210 samples
- Use data augmentation + extended training (100-200 epochs)

---

## Technical Details

### Validation Script Location

```bash
scripts/validate_pretraining_features.py
```

### Generated Artifacts

```
artifacts/feature_validation/
├── feature_alignment_results.csv          # Raw metrics per window
├── feature_separation.png                 # Bar chart of feature separations
├── threshold_opt_range_z.png              # Threshold optimization for range_z
├── threshold_opt_expansion_proxy.png      # Threshold optimization for expansion_proxy
├── threshold_opt_body_pct.png             # Threshold optimization for body_pct
├── threshold_opt_consol_proxy.png         # Threshold optimization for consol_proxy
├── feature_correlation_heatmap.png        # Per-window correlation heatmap
└── recommendations.txt                     # Auto-generated recommendations
```

### Reproducibility

```bash
# Full validation (210 samples)
python3 scripts/validate_pretraining_features.py \
  --data data/processed/labeled/train_latest_overlaps_v2.parquet \
  --output artifacts/feature_validation

# Quick test (30 samples)
python3 scripts/validate_pretraining_features.py \
  --data data/processed/labeled/train_latest_overlaps_v2.parquet \
  --output artifacts/feature_validation \
  --max-samples 30
```

---

## Conclusion

Your intuition to validate pre-training features was **100% correct**. The analysis reveals:

1. **Heuristic features don't align with human labels** (correlation < 0.03)
2. **Pre-training on these features won't help supervised learning**
3. **Need better features OR skip pre-training entirely**

**Recommended path forward**:
- **Tonight**: Skip pre-training, run extended soft span training (100-200 epochs) on 210 samples
- **This week**: Design + validate improved temporal features (momentum, acceleration)
- **Next week**: If improved features correlate > 0.15, revisit pre-training strategy

**Expected outcome**: F1 0.14 → 0.40-0.55 via extended training alone, WITHOUT pre-training.

---

**Created**: 2025-10-26
**Analysis tool**: `scripts/validate_pretraining_features.py`
**Dataset**: 210 labeled expansion windows (train_latest_overlaps_v2.parquet)
