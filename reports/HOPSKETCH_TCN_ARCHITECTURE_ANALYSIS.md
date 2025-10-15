# HopSketch TCN Architecture Analysis

**Date**: 2025-10-14
**Source**: `/Users/jack/pivot/archive/hopsketch_spec1/`
**Purpose**: Document the successful TCN+XGBoost architecture that worked in HopSketch

---

## Executive Summary

The HopSketch architecture used **3 Bidirectional TCN variants** trained on **15 simple features** (NOT 37 complex features), combined with an **XGBoost meta-learner** for span refinement. This is fundamentally different from the current moola architecture.

**Key Differences from Current Moola:**

| Aspect | HopSketch (Old/Working) | Moola (Current/56.5%) |
|--------|------------------------|----------------------|
| Input Features | **15 simple features** | 37 complex ICT features |
| Feature Type | Normalized OHLC + geometry + context | Handcrafted market structure |
| Model Type | Bidirectional TCN (3 variants) | CNN-Transformer |
| Input Region | Full 105 bars (all features) | Full 105 bars (OHLC) |
| Architecture | Forward + Backward TCN paths | CNN → Transformer → Pooling |
| Output | Multi-task (type + pointers) | Single classification |
| Pooling | Bidirectional concat | Global average (signal dilution!) |
| Ensemble | 3 TCN variants + XGBoost meta | Stacking (LR+RF+XGB) |
| XGBoost Role | Meta-learner on TCN outputs | Direct classifier on features |
| Parameters | ~1.2M per TCN | ~200K (CNN-Transformer) |

---

## 1. The 15 Features

**Source**: `features/geometry_features_15feat.py`

### Feature Breakdown:

#### Group 1: OHLC Normalized (4 features)
```python
ohlc_mean = ohlc.mean(axis=0)  # [4]
ohlc_std = ohlc.std(axis=0) + 1e-8
ohlc_norm = (ohlc - ohlc_mean) / ohlc_std  # [seq_len, 4]
```
- Normalized open, high, low, close (by window statistics)

#### Group 2: Geometry Features (7 features)
```python
1. body = close - open
2. range = high - low
3. body_frac = body / range
4. upper_wick = high - max(open, close)
5. lower_wick = min(open, close) - low
6. close_pos = (close - low) / range
7. open_pos = (open - low) / range
```

#### Group 3: Context Features (4 features)
```python
1. direction = sign(close - open)  # {-1, 0, 1}
2. gap_prev = open[t] - close[t-1]
3. pct_up = (close - open) / open
4. run_length = consecutive bars in same direction
```

### Critical Insight: Simple Per-Bar Features

**Why this works better than 37 complex features:**
1. **No smoothing operations** (no `.mean()`, `.std()` across pattern region)
2. **Per-bar granularity** (TCN learns aggregation, not handcrafted)
3. **Normalized by window** (removes absolute price dependence)
4. **Geometric focus** (body/wick ratios, not ICT concepts like OB/FVG)
5. **Temporal context preserved** (run_length, gap_prev show momentum)

---

## 2. The 3 TCN Variants

**Source**: `scripts/02_train_pointer_tcn_ensemble.py`, `models/tcn_bidirectional.py`

### BidirectionalTCN Architecture

```python
class BidirectionalTCN:
    input_channels: 15          # NOT 37 or 4 OHLC!
    tcn_channels: [128, 128, 128, 128]  # Hidden dim per direction
    num_tcn_layers: 4
    kernel_size: 5
    dilations: [1, 2, 4, 8]     # Receptive field ~= 15 bars
    dropout: 0.2

    # Architecture:
    forward_tcn:  [15 → 128] × 4 layers → [batch, 105, 128]
    backward_tcn: [15 → 128] × 4 layers → [batch, 105, 128]
    concat:       [batch, 105, 256]  # Forward + Backward

    # Multi-task heads:
    type_head:  [256 → 3]        # Classification (3 types)
    start_head: [256 → 105]      # Pointer to start bar
    end_head:   [256 → 105]      # Pointer to end bar

    # Parameters: ~1,200,000
```

### Why Bidirectional?

**Forward Path (Past Context)**:
- Processes bars [0 → 104]
- Captures "what happened before"
- Helps identify pattern start

**Backward Path (Future Context)**:
- Processes bars [104 → 0]
- Captures "what happens after"
- Helps identify pattern end

**Concatenation**:
- Combines both directions → 256-dim representation
- Each position has access to both past AND future context
- Critical for pointer prediction (need both sides to localize pattern)

### The 3 Variants (a, b, c)

**Differences between variants** (from `get_variant_config()`):
- **Variant A**: Baseline configuration
- **Variant B**: Different augmentation strategy
- **Variant C**: Different hyperparameters (filters, dropout, label_sigma)

Each variant trains independently with 5-fold cross-validation, producing diverse predictions for ensembling.

---

## 3. Multi-Task Learning

### Loss Function: Fixed-Weight Multi-Task

**Source**: `scripts/02_train_pointer_tcn_ensemble.py:136-209`

```python
class FixedWeightLoss:
    # Fixed weights (NOT learnable)
    type_weight = 2.0      # Classification
    start_weight = 3.0     # Start pointer
    end_weight = 3.0       # End pointer

    def forward(outputs, batch):
        # Type loss (cross-entropy)
        type_loss = F.cross_entropy(type_logits, type_labels)

        # Pointer losses (KL divergence with Gaussian targets)
        start_loss = F.kl_div(log_softmax(start_logits), gaussian_target_start)
        end_loss = F.kl_div(log_softmax(end_logits), gaussian_target_end)

        # Combined loss
        total_loss = 2.0*type_loss + 3.0*start_loss + 3.0*end_loss
        return total_loss
```

### Gaussian Label Smoothing for Pointers

**Instead of one-hot encoding:**
```python
# Ground truth: start_idx=65
# Traditional: [0, 0, ..., 1 (at 65), ..., 0]  ← Sharp peak

# Gaussian smoothing (sigma=2.0):
# [0, 0, ..., 0.1, 0.6, 1.0, 0.6, 0.1, ..., 0]  ← Soft peak
gaussian_target = exp(-0.5 * ((i - gt_idx) / sigma)^2)
```

**Why this works:**
- Handles annotation uncertainty (pattern boundaries are fuzzy)
- Smoother gradients → better training stability
- Allows ±2 bar tolerance in evaluation

---

## 4. XGBoost Meta-Learner

**Source**: `scripts/03_train_meta_ensemble.py`

### Architecture: XGBoost as Meta-Learner (NOT Base Classifier)

**CRITICAL DIFFERENCE**: XGBoost does NOT operate on raw 15 features!

**Data Flow:**
```
Raw OHLC [105 bars]
    ↓
Extract 15 features (geometry_features_15feat.py)
    ↓
3 TCN Variants (a, b, c) → Each predicts:
    - Type probabilities [3]
    - Start pointer distribution [105]
    - End pointer distribution [105]
    ↓
Ensemble Inference (weighted fusion of 3 TCNs)
    - Fuses predictions across variants
    - Generates "fused spans" (candidate pattern regions)
    ↓
compute_meta_features() → Extracts features from fused spans:
    - TCN confidence scores from each variant
    - Agreement/disagreement metrics across variants
    - Span geometry (length, position in window)
    - Pattern characteristics within span
    ↓
XGBoost Meta-Learner:
    - Classifier: Is this fused span correct? (±2 bars)
    - Regressors: How to adjust start/end pointers?
    ↓
Final Prediction: Refined span after XGBoost adjustment
```

### XGBoost Configuration

```python
# Classifier (span validation)
XGBClassifier(
    max_depth=4,
    n_estimators=300,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
)

# Regressors (pointer refinement)
XGBRegressor(
    max_depth=4,
    n_estimators=400,  # More trees for regression
    learning_rate=0.05,
    objective='reg:squarederror',
)
```

### XGBoost Training Data

**From `build_training_matrices()`:**

For each sample:
1. Get TCN ensemble predictions → fused spans
2. For each fused span:
   - Extract meta-features (span geometry, TCN scores, agreement)
   - Compare to ground truth (start_idx, end_idx)
   - Label: 1 if within ±2 bars, 0 otherwise
   - Offset: [start_delta, end_delta] for regression

**Result**:
- X: Meta-features [N_spans, feature_dim]
- y: Binary labels (span correct or not)
- offsets: [N_spans, 2] (for pointer refinement)
- groups: Day IDs (for GroupKFold CV)

### Why Meta-Learning?

**Advantages over direct classification:**
1. **Leverages TCN uncertainty**: Knows when TCNs agree/disagree
2. **Span-level reasoning**: Validates full pattern, not just bars
3. **Refinement capability**: Adjusts pointers based on TCN confidence
4. **Ensemble diversity**: Combines 3 TCN variants' perspectives
5. **Better calibration**: Isotonic regression on XGBoost probabilities

---

## 5. Training Pipeline

### Step 1: Train 3 TCN Variants

```bash
# Variant A
python scripts/02_train_pointer_tcn_ensemble.py --variant=a --epochs=30 --folds=5

# Variant B
python scripts/02_train_pointer_tcn_ensemble.py --variant=b --epochs=30 --folds=5

# Variant C
python scripts/02_train_pointer_tcn_ensemble.py --variant=c --epochs=30 --folds=5
```

**Each variant produces:**
- 5 fold checkpoints (for cross-validation)
- OOF (out-of-fold) predictions
- Holdout set predictions

### Step 2: Train XGBoost Meta-Learner

```bash
python scripts/03_train_meta_ensemble.py \
    --variant-a=models/tcn_a_best.pt \
    --variant-b=models/tcn_b_best.pt \
    --variant-c=models/tcn_c_best.pt
```

**XGBoost training:**
1. Load 3 TCN checkpoints
2. Run inference on full dataset → generate fused spans
3. Extract meta-features from fused spans
4. Train XGBoost classifier + regressors with 5-fold GroupKFold CV
5. Apply isotonic calibration to probabilities
6. Save: `xgb_classifier.joblib`, `xgb_reg_start.joblib`, `xgb_reg_end.joblib`

---

## 6. Comparison with Moola's Current Architecture

### What HopSketch Did Right

✅ **Simple features** (15 normalized/geometric) vs complex ICT features (37)
✅ **Per-bar granularity** (let model learn aggregation) vs handcrafted aggregations
✅ **Multi-task learning** (type + pointers) vs single classification
✅ **Bidirectional context** (past + future) vs global attention with dilution
✅ **Pointer prediction** (direct localization) vs implicit pattern detection
✅ **Meta-learning** (XGBoost on TCN outputs) vs direct XGBoost on features
✅ **Ensemble diversity** (3 variants with augmentation) vs stacking (LR+RF+XGB)

### What Moola Does Wrong

❌ **Signal dilution**: CNN-Transformer averages 5-bar pattern with 100 noise bars (4.8% effective signal)
❌ **No attention masking**: Pattern attends to all 105 bars including buffers
❌ **Global pooling**: Destroys spatial localization before classification
❌ **Complex features**: 37 features with smoothing operations (18 near-zero correlation)
❌ **Single-task**: Only classification, no pointer prediction
❌ **No spatial reasoning**: Can't point to where pattern is, only classify

---

## 7. Key Insights for Improving Moola

### Insight 1: Feature Simplicity Wins

**HopSketch's 15 features** are simpler than moola's 37 complex features, yet likely more effective:
- No smoothing → preserves signal
- Per-bar granularity → model learns patterns
- Normalized → removes price dependence
- Geometric → universal across timeframes

**Recommendation**: Test moola with HopSketch's 15 features:
```python
from pivot.archive.hopsketch_spec1.features.geometry_features_15feat import extract_features

# Replace engineer_classical_features() with extract_features()
X_simple = np.stack([extract_features(sample) for sample in X_raw])
# Train XGBoost on [N, 15] instead of [N, 37]
```

### Insight 2: Multi-Task Learning is Powerful

**HopSketch trains on 3 objectives simultaneously:**
1. Type classification (consolidation/retracement/reversal)
2. Start pointer (where pattern begins)
3. End pointer (where pattern ends)

**Why this works:**
- Pattern localization forces spatial understanding
- Multi-task regularization prevents overfitting
- Pointer prediction provides interpretable output

**Recommendation**: Add pointer prediction to CNN-Transformer:
```python
# New output heads:
type_head: [hidden → 3]        # Classification
start_head: [hidden → 105]     # Where is pattern start?
end_head: [hidden → 105]       # Where is pattern end?

# Loss:
loss = 2.0*type_loss + 3.0*start_loss + 3.0*end_loss
```

### Insight 3: Bidirectional Context Matters

**HopSketch's forward + backward TCN paths provide:**
- Past context (what led to pattern)
- Future context (what happens after pattern)
- Both are critical for localization

**Current moola CNN-Transformer:**
- Only has global attention (all-to-all)
- No explicit past/future separation
- Contaminated by buffer regions

**Recommendation**: Add bidirectional design:
```python
# Option A: Bidirectional Transformer encoder
transformer = nn.TransformerEncoder(
    encoder_layer,
    num_layers=3,
    bidirectional=True  # If supported
)

# Option B: Causal masking with forward/backward branches
forward_mask = torch.tril(torch.ones(105, 105))   # Can only see past
backward_mask = torch.triu(torch.ones(105, 105))  # Can only see future
```

### Insight 4: Meta-Learning on Deep Model Outputs

**HopSketch's XGBoost meta-learner operates on:**
- TCN predictions (probabilities, pointers)
- Ensemble agreement metrics
- Span geometry features

**Current moola stacking ensemble:**
- Combines LogReg + RF + XGBoost base models
- Uses OOF predictions as features
- Meta-learner = LogisticRegression

**Recommendation**: Use XGBoost as meta-learner:
```python
# 1. Get CNN-Transformer predictions:
#    - Type probabilities [N, 3]
#    - Attention weights [N, 105, 105]
#    - Pooled features [N, hidden_dim]

# 2. Extract meta-features:
#    - Prediction confidence (max probability)
#    - Attention concentration (entropy of attention)
#    - Pattern characteristics from attention-weighted bars

# 3. Train XGBoost meta-learner on meta-features:
X_meta = compute_meta_features_from_cnn_transformer(predictions, attention)
xgb_meta = XGBClassifier(max_depth=4, n_estimators=300)
xgb_meta.fit(X_meta, y)
```

### Insight 5: Why Moola's "Option 1" Failed

**Hypothesis**: 5 simple features from expansion region → 63-66% accuracy
**Result**: 48.7% accuracy (FAILED)

**Why HopSketch's 15 features might work where 5 failed:**

| Aspect | Moola's 5 Features | HopSketch's 15 Features |
|--------|-------------------|------------------------|
| Input scope | Expansion region ONLY (6 bars) | Full window (105 bars) |
| Context | No context (isolated pattern) | Context (before/after pattern) |
| Normalization | None | Window-normalized OHLC |
| Temporal info | None | run_length, gap_prev |
| Granularity | Aggregated (mean, ratios) | Per-bar (let model aggregate) |
| Model | XGBoost directly on features | TCN on features → XGBoost meta |

**Critical difference**: HopSketch uses 15 features across **entire 105-bar window**, not just expansion region. The model learns to localize the pattern via pointer prediction.

**Moola's mistake**: Extracted 5 features from **expansion region only** (6 bars), losing all context.

---

## 8. Recommended Next Steps for Moola

### Option 1: Test HopSketch's 15 Features (Fastest, 1-2 hours)

```bash
# Copy feature extraction from HopSketch
cp pivot/archive/hopsketch_spec1/features/geometry_features_15feat.py \
   src/moola/features/hopsketch_features.py

# Create test script
cat > scripts/test_hopsketch_features.py << 'EOF'
from moola.features.hopsketch_features import extract_features
X_15feat = np.stack([extract_features(sample) for sample in X_raw])
# Train XGBoost on [N, 15] from FULL 105-bar window
# Expected: 52-56% (better than 48% from 5 features)
EOF
```

### Option 2: Add Multi-Task Pointer Prediction (Medium, 1-2 days)

```python
# Modify CNN-Transformer to output:
# 1. Type classification [N, 3]
# 2. Start pointer [N, 105]
# 3. End pointer [N, 105]

# Use same loss as HopSketch:
loss = 2.0*type_loss + 3.0*start_loss + 3.0*end_loss
```

### Option 3: Implement Bidirectional TCN (High effort, 2-3 days)

```python
# Port HopSketch's BidirectionalTCN to moola
# Train on 15 simple features
# Expected: 60-65% (closer to HopSketch performance)
```

### Option 4: Fix CNN-Transformer Architecture (Medium, 1 day)

```python
# 1. Add attention masking for [30:75] region
# 2. Region-specific pooling (not global)
# 3. Add pointer prediction heads
# Expected: 58-62%
```

---

## 9. Conclusion

**Why HopSketch Worked:**
1. **Simple features** (15 normalized/geometric) over complex ICT features
2. **Full window context** (105 bars) not just expansion region
3. **Multi-task learning** (classification + pointer prediction)
4. **Bidirectional architecture** (past + future context)
5. **Meta-learning** (XGBoost on TCN outputs, not raw features)
6. **Ensemble diversity** (3 variants with different augmentations)

**Why Moola Struggles:**
1. Signal dilution (4.8% effective signal from global pooling)
2. Complex features with smoothing (37 → 18 near-zero)
3. No spatial localization (can't point to pattern)
4. Single-task learning (classification only)
5. Wrong baseline assumption (thought XGBoost alone got 56.5%)

**Fastest Path Forward:**
Test HopSketch's 15 features on moola's data:
- Extract from full 105-bar window (not just expansion)
- Train XGBoost directly (quick validation)
- If successful (>52%), implement multi-task TCN

**Long-term Solution:**
Implement BidirectionalTCN with multi-task learning:
- Port HopSketch architecture to moola
- Train with pointer prediction
- Add XGBoost meta-learner on TCN outputs
- Target: 60-65% accuracy

---

**Next Action**: Run test_hopsketch_features.py to validate if 15-feature approach works better than 5-feature approach.
