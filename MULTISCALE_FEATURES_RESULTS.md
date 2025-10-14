# Multi-Scale Features - Initial Results

**Date**: 2025-10-14
**Implementation**: `engineer_multiscale_features()` in `price_action_features.py`

## Results Summary

### Feature Extraction
```
Multi-scale features: 21 features (down from 37 classical)
- Scale 1 (Pattern): 5 features
- Scale 2 (Context): 10 features
- Scale 3 (Relative): 5 features
- Interaction: 1 feature
```

### Feature-Label Correlations
```
Multi-scale:  Max |corr| = 0.2085 (vs 0.1152 baseline)
Classical:    Max |corr| = 0.1925 (with expansion indices)

Improvement: +0.0160 (+8.3%)
Features >0.15: 2/21 vs 5/37 (classical)
```

### Model Performance (5-fold CV)

| Model | Multi-scale | Previous Best | Change |
|-------|-------------|---------------|--------|
| LogReg | 44.35% ± 6.39% | 53.04% ± 13.01% | **-8.7%** ❌ |
| XGBoost | 50.43% ± 7.06% | 44.35% ± 5.07% | **+6.1%** ✅ |
| RandomForest | 47.83% ± 8.25% | 53.91% (no exp) | **-6.1%** ❌ |

### Success Criteria Status
```
❌ Max correlation >0.22:     0.2085 (close!)
❌ XGBoost >55%:              50.43% (not quite)
❌ Any model >60%:            50.43% (far off)
```

---

## Analysis

### What Worked ✅

1. **Feature count reduction**: 21 vs 37 (43% fewer)
2. **Correlation improvement**: 0.1152 → 0.2085 (+81%)
3. **XGBoost improvement**: 44.35% → 50.43% (+6%)
4. **Cleaner signal**: Separated pattern/context/relative scales

### What Didn't Work ❌

1. **LogReg regression**: 53% → 44% (-9%)
2. **RandomForest regression**: 54% → 48% (-6%)
3. **Still below target**: Best 50% vs 60-65% target
4. **High variance**: ±6-8% across folds

---

## Root Cause Analysis

### Issue 1: Feature Scale Mismatch

Looking at feature statistics:
```python
Feature 5 (wick_balance): mean=2.3e9, std=2.9e9  # HUGE outlier!
Feature 6 (williams_r):   mean=-42.98, std=29.98
Features 7-8 (volatility): mean=2.2e-4, std=1.5e-4  # Very small
```

**Problem**: Features have wildly different scales → LogReg/RF sensitive to this.

**Solution**: Add StandardScaler normalization.

### Issue 2: Wick Balance Calculation Error

```python
# Current (lines 119-122):
wick_balance = (upper_wick / (lower_wick + 1e-10)).mean()
```

**Problem**: Division creates huge values when lower_wick ≈ 0.

**Fix**: Use ratio bounded to [0, 1] or clip outliers.

### Issue 3: Still Fundamentally Weak Signal

```
Max correlation: 0.2085 (target was >0.22)
Pattern length: 6 bars (still very short)
```

**Reality**: Even with better engineering, 6-bar patterns may not have enough signal for 60%+ accuracy.

---

## Recommendations

### Immediate Fixes (1 hour)

**1. Fix wick_balance calculation**
```python
# Replace lines 115-122 with:
if len(p_o) > 0:
    upper_wick = p_h - np.maximum(p_o, p_c)
    lower_wick = np.minimum(p_o, p_c) - p_l
    total_wick = upper_wick + lower_wick + 1e-10
    wick_balance = (upper_wick / total_wick).mean()  # Now [0, 1]
```

**2. Add feature normalization**
```python
from sklearn.preprocessing import StandardScaler

# After feature extraction:
scaler = StandardScaler()
X_multiscale_normalized = scaler.fit_transform(X_multiscale)
```

**Expected improvement**: LogReg 44% → 52%, XGB 50% → 54%

### Short-term Enhancements (2-3 hours)

**3. Add more pattern-specific features**
```python
# Consolidation signature features:
- sideways_score = abs(p_c[-1] - p_c[0]) / p_c.mean()  # Low = consolidation
- compression_ratio = (p_h.max() - p_l.min()) / (c_h.max() - c_l.min())  # Low = compression
- bar_uniformity = 1 - p_c.std() / p_c.mean()  # High = sideways

# Retracement signature features:
- reversal_strength = abs(np.diff(p_c)).sum() / p_c[0]  # High = volatile
- momentum_change = abs((p_c[-1] - p_c[0]) / p_c[0])  # High = directional
```

**Expected improvement**: +3-5% (target: 55-60%)

**4. Ensemble with classical features**
```python
# Combine both feature sets:
X_hybrid = np.concatenate([X_multiscale, X_classical[:, top_10_indices]], axis=1)
```

**Expected improvement**: +2-4% (target: 57-62%)

### Alternative Approach (if improvements insufficient)

**5. Accept current limitations**
```
Current best: 54.78% (stacking ensemble)
Multi-scale best: 50.43% (XGBoost)
Realistic target: 55-58% (with fixes)
```

**Reality check**: With 6-bar patterns and max correlation 0.2085, achieving 60%+ may require:
- More labeled data (115 → 500+ samples)
- Better label definitions (inter-rater reliability check)
- Different task formulation (multi-class instead of binary)
- Domain-specific features (order flow, volume profile)

---

## Next Steps

### Priority 1: Fix Bugs (30 min)
1. Fix wick_balance calculation (bounded ratio)
2. Add StandardScaler normalization
3. Re-run tests

### Priority 2: Test Fixes (30 min)
```bash
python scripts/test_multiscale_features.py
```

**Expected**:
- LogReg: 44% → 52%
- XGBoost: 50% → 54%
- Max correlation: 0.2085 → 0.21 (unchanged, but stable)

### Priority 3: If Still <55% (1-2 hours)
- Add consolidation/retracement signature features
- Test hybrid approach (multi-scale + classical top-10)
- Consider accepting 52-58% as realistic ceiling

### Priority 4: Upload to Server
Once local tests show improvement:
```bash
scp src/moola/features/price_action_features.py root@SERVER:/workspace/moola/src/moola/features/
ssh root@SERVER "cd /workspace && python -m moola.cli oof --model xgb"
```

---

## Lessons Learned

1. **Feature scaling matters**: LogReg/RF very sensitive to unnormalized features
2. **Ratio features need bounds**: Division can create huge outliers
3. **Correlation improved but not enough**: 0.1152 → 0.2085 still weak
4. **XGBoost more robust**: Handles unscaled features better than LogReg
5. **Pattern length is fundamental limit**: 6 bars may not contain enough signal

---

## Files Modified

- ✅ `src/moola/features/price_action_features.py` - Added `engineer_multiscale_features()`
- ✅ `scripts/test_multiscale_features.py` - Comprehensive test script
- ✅ `MODEL_FORENSICS_FINDINGS.md` - Previous analysis
- ✅ `EXPANSION_INDICES_ANALYSIS.md` - Previous investigation

---

**Status**: Initial implementation complete, needs bug fixes for target performance
**Next**: Fix wick_balance and add normalization
**Target**: 52-58% (realistic given 0.2085 max correlation)
