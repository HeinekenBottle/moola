#!/usr/bin/env python3
"""Demo script showing bootstrap confidence interval computation.

Simulates small validation set (34 samples) and computes bootstrap CIs
to demonstrate uncertainty quantification for robust performance estimation.

Usage:
    python3 scripts/demo_bootstrap_ci.py
"""

import numpy as np
from moola.utils.metrics.bootstrap import (
    bootstrap_accuracy,
    bootstrap_pointer_metrics,
    bootstrap_calibration_metrics,
    format_bootstrap_result,
)


def main():
    """Demonstrate bootstrap CI computation on simulated data."""
    print("=" * 80)
    print("BOOTSTRAP CONFIDENCE INTERVALS DEMO")
    print("=" * 80)
    print()

    # Simulate small validation set (34 samples - realistic for Moola)
    np.random.seed(42)
    n_samples = 34
    print(f"Validation set size: {n_samples} samples (small sample regime)")
    print(f"Bootstrap resamples: 1000")
    print(f"Confidence level: 95%")
    print()

    # Simulate classification predictions (~75% accuracy)
    y_true = np.random.binomial(1, 0.5, size=n_samples)  # Balanced classes
    # Create predictions with ~75% accuracy
    y_pred = y_true.copy()
    flip_indices = np.random.choice(n_samples, size=int(0.25 * n_samples), replace=False)
    y_pred[flip_indices] = 1 - y_pred[flip_indices]

    point_accuracy = (y_true == y_pred).mean()
    print(f"Point estimate accuracy: {point_accuracy:.4f}")
    print()

    # Bootstrap CI for accuracy
    print("-" * 80)
    print("1. CLASSIFICATION ACCURACY")
    print("-" * 80)

    acc_ci = bootstrap_accuracy(y_true, y_pred, n_resamples=1000, confidence_level=0.95)

    print(f"  {format_bootstrap_result('Accuracy', acc_ci)}")
    print(f"  Standard deviation: {acc_ci['std']:.4f}")
    ci_width = acc_ci["ci_upper"] - acc_ci["ci_lower"]
    print(f"  CI width: {ci_width:.4f} ({ci_width*100:.1f} percentage points)")
    print()
    print("  Interpretation:")
    print(
        f"    - True performance likely between {acc_ci['ci_lower']:.1%} and {acc_ci['ci_upper']:.1%}"
    )
    print(f"    - Wide CI ({ci_width*100:.1f}pp) reflects small sample uncertainty")
    print()

    # Simulate pointer regression predictions
    print("-" * 80)
    print("2. POINTER REGRESSION METRICS")
    print("-" * 80)

    true_start = np.random.randint(10, 40, size=n_samples)
    true_end = true_start + np.random.randint(20, 50, size=n_samples)

    # Add noise to predictions (~3 bars MAE)
    pred_start = true_start + np.random.randint(-5, 6, size=n_samples)
    pred_end = true_end + np.random.randint(-5, 6, size=n_samples)

    ptr_ci = bootstrap_pointer_metrics(
        pred_start=pred_start,
        pred_end=pred_end,
        true_start=true_start,
        true_end=true_end,
        tolerance=3,
        n_resamples=1000,
        confidence_level=0.95,
    )

    for metric_name in ["start_mae", "end_mae", "hit_at_pm3", "center_mae"]:
        print(f"  {format_bootstrap_result(metric_name, ptr_ci[metric_name])}")

    print()
    print("  Interpretation:")
    print(
        f"    - Hit@±3: {ptr_ci['hit_at_pm3']['mean']:.1%} with 95% CI [{ptr_ci['hit_at_pm3']['ci_lower']:.1%}, {ptr_ci['hit_at_pm3']['ci_upper']:.1%}]"
    )
    print("    - Non-overlapping CIs between models indicate significant difference")
    print()

    # Simulate calibration data
    print("-" * 80)
    print("3. CALIBRATION METRICS")
    print("-" * 80)

    # Create somewhat calibrated predictions
    probs = np.random.uniform(0.2, 0.8, size=n_samples)
    labels = (probs + np.random.normal(0, 0.15, size=n_samples) > 0.5).astype(int)

    cal_ci = bootstrap_calibration_metrics(
        probs=probs, labels=labels, n_resamples=1000, confidence_level=0.95
    )

    for metric_name in ["ece", "brier"]:
        print(f"  {format_bootstrap_result(metric_name, cal_ci[metric_name])}")

    print()
    print("  Interpretation:")
    print(f"    - ECE: {cal_ci['ece']['mean']:.4f} ± {cal_ci['ece']['std']:.4f} (target: <0.08)")
    if cal_ci["ece"]["ci_upper"] < 0.08:
        print("    - Model is well-calibrated (upper CI < 0.08)")
    elif cal_ci["ece"]["ci_lower"] > 0.15:
        print("    - Model is poorly calibrated (lower CI > 0.15)")
    else:
        print("    - Calibration uncertain - CI spans acceptable and poor ranges")
    print()

    # Comparison example
    print("=" * 80)
    print("MODEL COMPARISON EXAMPLE")
    print("=" * 80)
    print()
    print("Scenario: Compare two models on small validation set")
    print()

    # Model A: 75% accuracy
    y_pred_a = y_pred.copy()
    acc_a = bootstrap_accuracy(y_true, y_pred_a, n_resamples=1000, confidence_level=0.95)

    # Model B: 82% accuracy (flip fewer predictions)
    y_pred_b = y_true.copy()
    flip_b = np.random.choice(n_samples, size=int(0.18 * n_samples), replace=False)
    y_pred_b[flip_b] = 1 - y_pred_b[flip_b]
    acc_b = bootstrap_accuracy(y_true, y_pred_b, n_resamples=1000, confidence_level=0.95)

    print(f"Model A: {format_bootstrap_result('Accuracy', acc_a)}")
    print(f"Model B: {format_bootstrap_result('Accuracy', acc_b)}")
    print()

    # Check CI overlap
    overlap = not (acc_a["ci_upper"] < acc_b["ci_lower"] or acc_b["ci_upper"] < acc_a["ci_lower"])

    if overlap:
        print("CIs OVERLAP:")
        print("  - Cannot confidently say Model B is better than Model A")
        print("  - Difference may be due to random variation")
        print("  - Need more validation data or smaller performance gap to distinguish")
    else:
        print("CIs DO NOT OVERLAP:")
        print("  - Strong evidence Model B outperforms Model A")
        print("  - Difference is statistically significant at 95% confidence")

    print()
    print("=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print()
    print("1. Small samples (34) produce wide CIs - expected and honest uncertainty")
    print("2. Use bootstrap CIs to compare models - check for non-overlapping CIs")
    print("3. Wide CIs indicate need for:")
    print("   - More validation data (via annotation)")
    print("   - Cross-validation")
    print("   - MC Dropout uncertainty estimation")
    print("4. Bootstrap is CPU-only, adds ~8 seconds for 1000 resamples")
    print("5. Report both point estimate AND CI in experiment logs")
    print()


if __name__ == "__main__":
    main()
