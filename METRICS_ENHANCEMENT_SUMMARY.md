# Comprehensive Metrics Enhancement for Jade Fine-Tuning

## Summary

Enhanced `scripts/finetune_jade.py` with comprehensive multi-task performance tracking, including pointer regression metrics, calibration metrics, per-class performance, and joint success metrics.

## What Was Added

### 1. New Metrics Functions

#### `compute_pointer_metrics()`
Tracks pointer regression performance:
- **Center MAE**: Mean absolute error for expansion center prediction (in bars)
- **Length MAE**: Mean absolute error for expansion length prediction (in bars)
- **Hit@±3**: Percentage of predictions where center is within ±3 bars of ground truth
- **Hit@±5**: Percentage of predictions where center is within ±5 bars of ground truth

#### `compute_calibration_metrics()`
Measures prediction confidence calibration:
- **ECE (Expected Calibration Error)**: Average calibration error across 10 probability bins
- **MCE (Maximum Calibration Error)**: Worst-case calibration error
- **Brier Score**: Mean squared error of probability predictions

#### `compute_joint_success()`
Evaluates true multi-task performance:
- **Joint Success@3**: Percentage of samples with BOTH correct classification AND pointer within ±3 bars
- **Joint Success@5**: Percentage of samples with BOTH correct classification AND pointer within ±5 bars

#### `create_metrics_table()`
Rich table formatter for organized metrics display with categories:
- Classification metrics
- Pointer regression metrics
- Calibration metrics
- Multi-task metrics
- Loss components

### 2. Enhanced Per-Class Metrics

Now tracks for each class (consolidation, retracement):
- Precision
- Recall
- F1-score
- Support
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)

### 3. Enhanced Validation Loop

Updated `evaluate()` function to:
- Collect predictions, probabilities, and pointer outputs
- Compute all comprehensive metrics after validation loop
- Display metrics in organized table format
- Show confusion matrix
- Track per-task loss components (loss_type, loss_ptr)

### 4. Cross-Validation Summary

Enhanced CV summary to:
- Track comprehensive metrics for each fold (not just F1)
- Compute mean ± std for ALL metrics across folds
- Display comprehensive CV summary table
- Highlight key metrics with statistics
- Save all metrics to JSON for later analysis

### 5. Training Output Enhancements

Improved training loop output:
- Shows `joint@3` metric during each epoch
- Tracks best_metrics (full dict) instead of just best_f1
- Comprehensive fold summary with F1 and Joint@3

## New Imports Added

```python
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    confusion_matrix,
)
```

## Output Format

### Per-Epoch Output
```
Epoch 15/20: train_loss=0.3245, val_loss=0.4123, val_acc=0.857, val_f1=0.834, joint@3=0.741
```

### Per-Fold Validation Output
```
┌─────────────────────────────────────────────┐
│         Validation Metrics                  │
├───────────────────────────┬─────────────────┤
│ Classification            │                 │
│   accuracy                │   0.857 (85.7%) │
│   f1_macro                │   0.834 (83.4%) │
│   precision_cons          │   0.880 (88.0%) │
│   recall_cons             │   0.900 (90.0%) │
│   ...                     │                 │
│ Pointer Regression        │                 │
│   center_mae              │          2.3400 │
│   length_mae              │          3.1200 │
│   hit_at_3                │   0.714 (71.4%) │
│   hit_at_5                │   0.886 (88.6%) │
│ Calibration               │                 │
│   ece                     │          0.0523 │
│   mce                     │          0.1234 │
│   brier                   │          0.0812 │
│ Multi-Task                │                 │
│   joint_success_at_3      │   0.629 (62.9%) │
│   joint_success_at_5      │   0.771 (77.1%) │
└───────────────────────────┴─────────────────┘

Confusion Matrix:
  True Cons / Pred Cons: 24
  True Cons / Pred Retr: 3
  True Retr / Pred Cons: 2
  True Retr / Pred Retr: 6
```

### Cross-Validation Summary
```
============================================================
Cross-Validation Summary (5 folds)
============================================================

[Comprehensive metrics table with all averages]

Key Metrics (mean ± std):
  F1 Macro: 0.834 ± 0.021
  Accuracy: 0.857 ± 0.018
  Joint Success@3: 0.741 ± 0.034
  Center MAE: 2.34 ± 0.42 bars
  Hit@±3: 0.714 ± 0.038
  AUROC: 0.892 ± 0.015
  ECE: 0.0523 ± 0.0089
```

## Saved to JSON

`cv_summary.json` now contains:
- `avg_metrics`: Dictionary with mean and std for ALL metrics
- `fold_results`: List of per-fold comprehensive metrics
- `config`: Training configuration

## Usage

No CLI changes needed. Simply run:

```bash
python3 scripts/finetune_jade.py \
  --data data/processed/labeled/train_latest.parquet \
  --pretrained-encoder artifacts/jade_pretrain_20ep/checkpoint_best.pt \
  --freeze-encoder \
  --epochs 20 \
  --device cuda
```

All comprehensive metrics will be computed and displayed automatically.

## Benefits

1. **Holistic Performance View**: No longer just optimizing for classification F1 - track pointer accuracy and joint success
2. **Calibration Tracking**: Understand if model is overconfident or underconfident
3. **Per-Class Analysis**: Identify which class (consolidation vs retracement) needs improvement
4. **Multi-Task Success**: Understand true utility - both tasks must succeed for production use
5. **Statistical Rigor**: Track mean ± std across folds for confidence in metrics
6. **Easy Comparison**: Standardized output format for comparing experiments

## Key Insights Enabled

- **Is the model learning both tasks equally?** Check loss_type vs loss_ptr
- **Is pointer task useful?** Compare joint_success@3 to simple accuracy
- **Is the model calibrated?** Low ECE means reliable confidence scores
- **Which class is harder?** Compare f1_cons vs f1_retr
- **Is ±3 bars sufficient?** Compare hit_at_3 vs hit_at_5 and joint metrics

## Files Modified

- `scripts/finetune_jade.py`: Enhanced with comprehensive metrics tracking

## Testing

All metrics functions tested with synthetic data:
- Pointer metrics: ✓ Correct MAE and Hit@N calculation
- Calibration metrics: ✓ ECE/MCE/Brier computation
- Joint success: ✓ Correct intersection of tasks
- Table formatting: ✓ Clean organized output

## Next Steps

Consider adding:
1. Per-epoch metrics tracking to CSV for plotting learning curves
2. Correlation analysis between pointer error and classification error
3. Breakdown of joint success by predicted class
4. Visualization of calibration curves
