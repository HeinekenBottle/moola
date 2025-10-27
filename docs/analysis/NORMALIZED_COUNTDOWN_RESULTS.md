# Normalized Countdown Training Results

**Date:** 2025-10-27
**Experiment:** Countdown normalization [0,1] + Reverse engineering proxies
**Duration:** 100 epochs, 2.2 minutes on RTX 4090

---

## Executive Summary

### ✅ Countdown Normalization: SUCCESS
- **Goal:** Reduce countdown loss from 10.08 → ~0.5 for gradient balance
- **Result:** Countdown loss reduced 50-100x (10.08 → 0.10-0.22)
- **Impact:** Gradient domination eliminated (91% → comparable to other tasks)

### ❌ Reverse Engineering Proxies: FAILURE
- **Goal:** Extract mathematical rules with correlation >0.15
- **Result:** Correlation = 0.017 (insufficient for pseudo-labels)
- **Reason:** Linear formulas cannot capture complex expansion patterns

### ❌ Span Detection: FAILURE (Unchanged from Baseline)
- **Span F1:** 0.000 (all 100 epochs)
- **Probability Separation:** 0.002 (target: >0.05)
- **Conclusion:** Countdown normalization alone does not enable span learning

---

## Detailed Results

### 1. Countdown Loss Normalization

#### Before Normalization (Baseline)
```
Epoch 1:
- loss_type: 0.62
- loss_ptr: 0.02
- loss_span: 0.68
- loss_countdown: 10.08  ← 91.2% of total loss
```

#### After Normalization
```
Epoch 1:
- loss_type: 1.14
- loss_ptr: 0.02
- loss_span: 0.67
- loss_countdown: 0.17  ← Now only 13% of total loss

Epoch 10:
- loss_type: 0.75
- loss_ptr: 0.02
- loss_span: 0.26
- loss_countdown: 0.10  ← Reduced 100x!

Epoch 100:
- loss_type: 0.20
- loss_ptr: 0.03
- loss_span: 0.22
- loss_countdown: 0.05  ← Stable at ~0.05
```

**Analysis:**
- Countdown loss reduced **50-100x** (10.08 → 0.10-0.22)
- Now comparable to other loss components (0.02-0.70 range)
- Allows uncertainty weighting to properly balance tasks
- **Normalization implementation was correct and effective**

---

### 2. Reverse Engineering Proxy Extraction

#### Configuration
- **Data:** 210 labeled samples (168 train, 42 val)
- **Features:** momentum_3, direction_streak, cum_return_5, volatility
- **Formula:** prob = clip((return × w_r + streak × w_s) / (vol + ε), 0, 1)
- **Grid Search:** 27 parameter combinations

#### Results
```
Summary Statistics:
- Avg momentum:      -0.0328
- Avg streak:        4.37
- Avg cum return:    -0.0001
- Avg volatility:    2.55
- Avg length:        7.4 bars

Best Parameters:
- return_weight:     2.0
- streak_weight:     0.05
- vol_epsilon:       0.001

Correlation: 0.0170 (target: >0.15)
```

**Analysis:**
- Correlation **9x below target** (0.017 vs 0.15)
- Linear formulas cannot capture expansion patterns
- Temporal features (momentum, streak, cumulative return) weakly correlated
- **Too weak for soft pseudo-labels in pre-training**

---

### 3. Span Detection Performance

#### Metrics (Epoch 100, Validation Set)
```
Span F1:         0.000
Span Precision:  0.000
Span Recall:     0.000

Probability Distribution:
- In-span mean:     0.063
- Out-span mean:    0.061
- Separation:       0.002  ← Target: >0.05 (25x below!)
```

#### Comparison to Baseline
| Metric | Baseline (unnormalized) | Normalized | Change |
|--------|------------------------|------------|--------|
| Countdown loss (epoch 1) | 10.08 | 0.17 | **-98.3%** ✅ |
| Span F1 | 0.000 | 0.000 | **No change** ❌ |
| Probability separation | 0.004 | 0.002 | **Worse** ❌ |

**Analysis:**
- Countdown normalization **did not translate to span learning**
- Model still cannot differentiate expansion vs non-expansion regions
- Probability separation even slightly worse than baseline
- **The root problem is deeper than gradient imbalance**

---

## Root Cause Analysis

### Why Countdown Normalization Didn't Help Span Detection

1. **Gradient balance is necessary but not sufficient**
   - Fixed: Countdown loss no longer dominates (91% → 13%)
   - Unfixed: Model still predicts uniform probabilities (~0.06 everywhere)

2. **Weak supervision signal**
   - Binary mask supervision may be too sparse (7.4 bars in-span vs 97.6 bars out-of-span)
   - Model learns to predict low probabilities everywhere (safe bet)

3. **Architecture limitations**
   - JadeCompact (97K params) may lack capacity for this multi-task problem
   - BiLSTM global average pooling may lose temporal localization

4. **Multi-task interference**
   - Even with uncertainty weighting, four tasks may be too many
   - Pointer prediction, type classification, span detection, countdown regression competing

---

## Artifacts Generated

### On RunPod (artifacts/baseline_normalized/)
- `loss_components.csv` - Detailed loss breakdown per epoch
- `uncertainty_params.csv` - Learned σ parameters for task weighting
- `probability_stats.csv` - In-span vs out-of-span probability distributions
- `feature_stats.csv` - Per-feature statistics during training
- `gradient_stats.csv` - Gradient norms per layer
- `epoch_metrics.csv` - High-level metrics (loss, F1, P, R)
- `best_model.pt` - Best checkpoint (epoch 67, val_loss=1.0281)
- `metadata.json` - Hyperparameters and dataset info

### Local (artifacts/reverse_proxies/)
- `proxy_formula.json` - Best formula parameters
- `expansion_stats.csv` - Statistics from 168 expansion windows
- `proxy_validation.png` - Scatter plot and histogram of proxy vs labels

---

## Next Steps & Recommendations

### Option 1: Simplified Single-Task Baseline (Recommended)
**Rationale:** Multi-task learning may be too ambitious for 210 samples

**Actions:**
1. Train **span detection only** (remove pointer, type, countdown tasks)
2. Use binary cross-entropy on soft span mask
3. Increase model capacity (128 → 256 hidden units)
4. Try class weighting for in-span timesteps (weight=10-20)

**Expected outcome:** Probability separation >0.05, span F1 >0.10

---

### Option 2: Stones Architecture (Binary + Countdown Only)
**Rationale:** Reduce from 4 tasks to 2 tasks, focus on expansion detection

**Actions:**
1. Remove pointer prediction and type classification
2. Keep only binary mask + normalized countdown
3. Use focal loss or weighted BCE for binary mask
4. Train for 200 epochs with early stopping

**Expected outcome:** Better gradient flow to expansion tasks only

---

### Option 3: Data Augmentation via Sliding Windows
**Rationale:** 210 samples may be insufficient, but we have 105 timesteps per sample

**Actions:**
1. Generate overlapping windows (stride=5) from existing 210 samples
2. Creates ~20x more training examples
3. Risk: May introduce label leakage, but worth testing

**Expected outcome:** More gradient updates, better generalization

---

### Option 4: Pre-training with Unlabeled Data (Deferred)
**Rationale:** Reverse engineering proxies failed (corr=0.017)

**Why not now:**
- Need better supervision signal first
- Proxy correlation too weak (0.017 vs target 0.15)
- Should establish supervised baseline before attempting semi-supervised

**When to revisit:**
- After achieving span F1 >0.20 in supervised setting
- Could try SimCLR or masked autoencoder instead of proxies

---

## Key Learnings

1. **Countdown normalization works as intended**
   - Successfully reduced gradient magnitude 50-100x
   - Confirmed via loss_components.csv breakdown
   - Implementation was correct

2. **Reverse engineering is harder than expected**
   - Linear formulas insufficient for expansion patterns
   - Temporal features (momentum, streak) weakly predictive
   - Would need non-linear methods (random forest, gradient boosting)

3. **Multi-task learning very challenging at 210 samples**
   - Four tasks may cause interference
   - Uncertainty weighting alone not enough
   - Simpler baselines needed first

4. **The fundamental problem remains**
   - Model predicts uniform probabilities (~0.06)
   - Cannot distinguish in-span vs out-of-span regions
   - This is a **capacity, architecture, or supervision** issue, not just gradient balance

---

## Conclusion

**Countdown normalization succeeded** in its goal (gradient balance), but **did not solve the span detection problem**. The model still cannot differentiate expansion regions from non-expansion regions (separation=0.002, F1=0.000).

**Recommendation:** Start with Option 1 (simplified single-task baseline) to establish whether this is a multi-task interference issue or a deeper architectural/data limitation.

All artifacts and detailed logs are available in `artifacts/baseline_normalized/` for further analysis.
