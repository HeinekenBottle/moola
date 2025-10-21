# Bootstrap Confidence Intervals - Quick Reference

**Phase 4: Robust performance estimation on small datasets (34 validation samples)**

---

## Quick Start

### 1. Enable Bootstrap CIs (Default Settings)
```bash
python3 -m moola.cli train \
    --model enhanced_simple_lstm \
    --split data/artifacts/splits/v1/fold_0.json \
    --bootstrap-ci \
    --device cuda
```

**Output:**
```
============================================================
BOOTSTRAP CONFIDENCE INTERVALS
============================================================
Resamples: 1000, Confidence: 95%

Classification Accuracy:
  Accuracy: 0.8529 [95% CI: 0.7647 - 0.9118]
  CI width: 0.1471 (14.7 percentage points)
```

---

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--bootstrap-ci` | False | Enable bootstrap confidence intervals |
| `--bootstrap-resamples` | 1000 | Number of bootstrap resamples |
| `--bootstrap-confidence` | 0.95 | Confidence level (0.95 = 95%) |

---

## Interpreting Results

### 1. CI Width (Uncertainty)
```
Accuracy: 0.85 [0.78 - 0.92]
          └────────────┘
           14pp wide = moderate uncertainty
```

**Guidelines:**
- **< 10pp:** Low uncertainty (large validation set)
- **10-20pp:** Moderate uncertainty (typical for 34 samples)
- **> 20pp:** High uncertainty (very small validation set)

### 2. Model Comparison
```
Model A: 0.85 [0.78 - 0.92]
Model B: 0.92 [0.87 - 0.97]
```

**Decision:**
- **CIs overlap:** Cannot confidently say B > A (may be random variation)
- **CIs don't overlap:** Strong evidence B > A (statistically significant)

### 3. Small Sample Awareness
```
Validation set size: 34 samples (small sample - CIs will be wide)
```

**Expected CI Widths (34 samples):**
- Accuracy: ~15-20 percentage points
- Pointer MAE: ~1-2 bars
- Hit@±3: ~15-20 percentage points

---

## Common Use Cases

### 1. Compare Two Models
```bash
# Model A (baseline)
python3 -m moola.cli train --model simple_lstm --bootstrap-ci --device cuda

# Model B (with pre-training)
python3 -m moola.cli train --model simple_lstm --load-pretrained artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt --bootstrap-ci --device cuda
```

**Check:** Do 95% CIs overlap?
- **No overlap:** Pre-training significantly improves performance
- **Overlap:** Cannot confidently say pre-training helps

### 2. More Stable CIs (2000 Resamples)
```bash
python3 -m moola.cli train \
    --model enhanced_simple_lstm \
    --bootstrap-ci \
    --bootstrap-resamples 2000 \
    --device cuda
```

**Trade-off:** More resamples = more stable CIs, but slower (~15s instead of ~8s)

### 3. Wider CIs (99% Confidence)
```bash
python3 -m moola.cli train \
    --model enhanced_simple_lstm \
    --bootstrap-ci \
    --bootstrap-confidence 0.99 \
    --device cuda
```

**Trade-off:** Higher confidence = wider CIs (more conservative)

### 4. Full Uncertainty Stack
```bash
python3 -m moola.cli train \
    --model enhanced_simple_lstm \
    --bootstrap-ci \
    --mc-dropout \
    --compute-calibration \
    --device cuda
```

**Combines:**
- Bootstrap CIs (performance uncertainty)
- MC Dropout (prediction uncertainty)
- Calibration (confidence calibration)

---

## Python API (Programmatic Usage)

### 1. Classification Accuracy
```python
from moola.utils.metrics.bootstrap import bootstrap_accuracy

result = bootstrap_accuracy(
    y_true=y_test,
    y_pred=y_test_pred,
    n_resamples=1000,
    confidence_level=0.95
)

print(f"Accuracy: {result['mean']:.3f} [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
# Output: Accuracy: 0.853 [0.765, 0.912]
```

### 2. Pointer Regression
```python
from moola.utils.metrics.bootstrap import bootstrap_pointer_metrics

ptr_ci = bootstrap_pointer_metrics(
    pred_start=pred_start,
    pred_end=pred_end,
    true_start=true_start,
    true_end=true_end,
    tolerance=3,
    n_resamples=1000
)

print(f"Hit@±3: {ptr_ci['hit_at_pm3']['mean']:.1%} "
      f"[{ptr_ci['hit_at_pm3']['ci_lower']:.1%}, "
      f"{ptr_ci['hit_at_pm3']['ci_upper']:.1%}]")
# Output: Hit@±3: 73.5% [61.8%, 83.8%]
```

### 3. Calibration
```python
from moola.utils.metrics.bootstrap import bootstrap_calibration_metrics

cal_ci = bootstrap_calibration_metrics(
    probs=y_probs,
    labels=y_true,
    n_resamples=1000
)

print(f"ECE: {cal_ci['ece']['mean']:.4f} "
      f"[{cal_ci['ece']['ci_lower']:.4f}, {cal_ci['ece']['ci_upper']:.4f}]")
# Output: ECE: 0.0234 [0.0156, 0.0318]
```

---

## When to Use Bootstrap CIs

### ✅ Use When:
- Small validation set (< 100 samples)
- Comparing multiple models
- Reporting results for publication
- Need honest uncertainty estimates
- Stakeholders need confidence ranges

### ❌ Don't Use When:
- Large validation set (> 1000 samples) - point estimates sufficient
- Extremely small samples (< 10) - bootstrap unreliable
- Need prediction-level uncertainty - use MC Dropout instead

---

## Troubleshooting

### CI Too Wide (> 20 percentage points)
**Problem:** High uncertainty, hard to compare models

**Solutions:**
1. Collect more validation data (via Candlesticks annotation)
2. Use cross-validation instead of single split
3. Combine with MC Dropout for prediction-level uncertainty
4. Increase resamples to 2000 for more stable estimates

### CIs Always Overlap
**Problem:** Cannot distinguish between models

**Solutions:**
1. Models may genuinely be similar (not a problem!)
2. Collect more validation data to narrow CIs
3. Look for larger performance gaps (5-10% accuracy difference)
4. Consider ensemble methods to boost performance

### Bootstrap Too Slow
**Problem:** 1000 resamples takes too long

**Solutions:**
1. Reduce to 500 resamples for debugging (`--bootstrap-resamples 500`)
2. Use fewer resamples during development (500-1000)
3. Use full 1000-2000 for final experiments
4. Bootstrap is CPU-only, ~8s overhead is expected

---

## Example Output

```
============================================================
BOOTSTRAP CONFIDENCE INTERVALS
============================================================
Resamples: 1000, Confidence: 95%
Validation set size: 34 samples (small sample - CIs will be wide)

Classification Accuracy:
  Accuracy: 0.8529 [95% CI: 0.7647 - 0.9118]
  Standard deviation: 0.0374
  CI width: 0.1471 (14.7 percentage points)

Pointer Regression Metrics:
  start_mae: 2.34 [95% CI: 1.82 - 2.91]
  end_mae: 2.45 [95% CI: 1.93 - 3.01]
  hit_at_pm3: 0.7353 [95% CI: 0.6176 - 0.8382]
  center_mae: 1.78 [95% CI: 1.35 - 2.23]

Calibration Metrics:
  ece: 0.0234 [95% CI: 0.0156 - 0.0318]
  brier: 0.1245 [95% CI: 0.0987 - 0.1521]

Interpretation Guide:
  - Non-overlapping CIs → models are significantly different
  - Overlapping CIs → difference may be due to random variation
  - Wide CIs → high uncertainty (expected with 34 validation samples)
============================================================
```

---

## Key Takeaways

1. **Enable with:** `--bootstrap-ci` flag
2. **Default:** 1000 resamples, 95% confidence
3. **Expect:** ~15-20pp CI width for accuracy (34 samples)
4. **Compare models:** Check for non-overlapping CIs
5. **Overhead:** ~8 seconds (CPU-only, after training)
6. **Honest:** Wide CIs reflect true uncertainty in small samples

---

## Demo Script

```bash
# See realistic example with 34-sample validation set
python3 scripts/demo_bootstrap_ci.py
```

---

## References

- **Implementation:** `src/moola/utils/metrics/bootstrap.py`
- **Tests:** `tests/utils/test_bootstrap.py`
- **Config:** `configs/phase4_bootstrap.json`
- **Full Summary:** `PHASE4_BOOTSTRAP_SUMMARY.md`
