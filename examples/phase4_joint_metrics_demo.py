"""Demo of Phase 4 joint metrics for dual-task model evaluation.

This script demonstrates how to use joint metrics to evaluate dual-task models
where BOTH pointer localization AND type classification must be correct.
"""

import numpy as np

from moola.utils.metrics.joint_metrics import (
    compute_joint_hit_accuracy,
    compute_joint_hit_f1,
    compute_task_contribution_analysis,
    select_best_model_by_joint_metric,
)


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def demo_perfect_predictions():
    """Demo: Perfect predictions on both tasks."""
    print_section("Demo 1: Perfect Predictions")

    n = 50
    pred_start = np.arange(n)
    pred_end = np.arange(n) + 10
    true_start = np.arange(n)
    true_end = np.arange(n) + 10
    pred_type = np.random.randint(0, 2, n)
    true_type = pred_type.copy()  # Perfect type predictions

    joint_acc = compute_joint_hit_accuracy(
        pred_start, pred_end, true_start, true_end, pred_type, true_type, tolerance=3
    )

    joint_f1 = compute_joint_hit_f1(
        pred_start, pred_end, true_start, true_end, pred_type, true_type, tolerance=3
    )

    print(f"Joint Accuracy:     {joint_acc['joint_accuracy']:.1%}")
    print(f"Pointer Hit Rate:   {joint_acc['pointer_hit_rate']:.1%}")
    print(f"Type Accuracy:      {joint_acc['type_accuracy']:.1%}")
    print(f"Joint F1:           {joint_f1['joint_f1']:.3f}")


def demo_pointer_bottleneck():
    """Demo: Good type classification, poor pointer regression."""
    print_section("Demo 2: Pointer Bottleneck")

    n = 100
    # Type predictions are good (80% accuracy)
    true_type = np.random.randint(0, 2, n)
    pred_type = true_type.copy()
    flip_mask = np.random.random(n) < 0.2
    pred_type[flip_mask] = 1 - pred_type[flip_mask]

    # Pointer predictions are poor (40% hit rate)
    true_start = np.random.randint(10, 40, n)
    true_end = true_start + np.random.randint(20, 40, n)
    pred_start = true_start.copy()
    pred_end = true_end.copy()

    # Add large errors to 60% of samples
    error_mask = np.random.random(n) < 0.6
    pred_start[error_mask] += np.random.randint(-10, 10, error_mask.sum())
    pred_end[error_mask] += np.random.randint(-10, 10, error_mask.sum())

    joint_acc = compute_joint_hit_accuracy(
        pred_start, pred_end, true_start, true_end, pred_type, true_type, tolerance=3
    )

    contribution = compute_task_contribution_analysis(
        pred_start, pred_end, true_start, true_end, pred_type, true_type, tolerance=3
    )

    print(f"Joint Accuracy:     {joint_acc['joint_accuracy']:.1%}")
    print(f"Pointer Hit Rate:   {joint_acc['pointer_hit_rate']:.1%} ← Bottleneck!")
    print(f"Type Accuracy:      {joint_acc['type_accuracy']:.1%}")
    print("\nTask Contribution:")
    print(f"  Both Correct:     {contribution['both_correct']['fraction']:.1%}")
    print(f"  Pointer Only:     {contribution['pointer_only_correct']['fraction']:.1%}")
    print(
        f"  Type Only:        {contribution['type_only_correct']['fraction']:.1%} ← High! Pointer limiting performance"
    )
    print(f"  Both Wrong:       {contribution['both_wrong']['fraction']:.1%}")
    print("\n💡 Recommendation: Increase loss_beta to give more weight to pointer task")


def demo_type_bottleneck():
    """Demo: Good pointer regression, poor type classification."""
    print_section("Demo 3: Type Classification Bottleneck")

    n = 100
    # Pointer predictions are good (75% hit rate)
    true_start = np.random.randint(10, 40, n)
    true_end = true_start + np.random.randint(20, 40, n)
    pred_start = true_start + np.random.randint(-2, 3, n)  # Small errors
    pred_end = true_end + np.random.randint(-2, 3, n)

    # Type predictions are poor (55% accuracy)
    true_type = np.random.randint(0, 2, n)
    pred_type = true_type.copy()
    flip_mask = np.random.random(n) < 0.45
    pred_type[flip_mask] = 1 - pred_type[flip_mask]

    joint_acc = compute_joint_hit_accuracy(
        pred_start, pred_end, true_start, true_end, pred_type, true_type, tolerance=3
    )

    contribution = compute_task_contribution_analysis(
        pred_start, pred_end, true_start, true_end, pred_type, true_type, tolerance=3
    )

    print(f"Joint Accuracy:     {joint_acc['joint_accuracy']:.1%}")
    print(f"Pointer Hit Rate:   {joint_acc['pointer_hit_rate']:.1%}")
    print(f"Type Accuracy:      {joint_acc['type_accuracy']:.1%} ← Bottleneck!")
    print("\nTask Contribution:")
    print(f"  Both Correct:     {contribution['both_correct']['fraction']:.1%}")
    print(
        f"  Pointer Only:     {contribution['pointer_only_correct']['fraction']:.1%} ← High! Type limiting performance"
    )
    print(f"  Type Only:        {contribution['type_only_correct']['fraction']:.1%}")
    print(f"  Both Wrong:       {contribution['both_wrong']['fraction']:.1%}")
    print("\n💡 Recommendation: Decrease loss_beta to give more weight to type classification")


def demo_model_selection():
    """Demo: Selecting best model based on joint metrics."""
    print_section("Demo 4: Model Selection")

    # Simulate 5 models with different performance characteristics
    models = [
        {
            "name": "Model A: High pointer, low type",
            "joint_f1": 0.45,
            "joint_accuracy": 0.42,
            "pointer_hit_rate": 0.78,
            "type_accuracy": 0.51,
        },
        {
            "name": "Model B: Balanced performance",
            "joint_f1": 0.62,
            "joint_accuracy": 0.58,
            "pointer_hit_rate": 0.68,
            "type_accuracy": 0.72,
        },
        {
            "name": "Model C: High type, low pointer",
            "joint_f1": 0.48,
            "joint_accuracy": 0.44,
            "pointer_hit_rate": 0.52,
            "type_accuracy": 0.81,
        },
        {
            "name": "Model D: Low pointer (filtered out)",
            "joint_f1": 0.55,
            "joint_accuracy": 0.50,
            "pointer_hit_rate": 0.35,  # Below threshold!
            "type_accuracy": 0.75,
        },
        {
            "name": "Model E: Low type (filtered out)",
            "joint_f1": 0.52,
            "joint_accuracy": 0.48,
            "pointer_hit_rate": 0.72,
            "type_accuracy": 0.45,  # Below threshold!
        },
    ]

    print("Candidate Models:")
    for i, model in enumerate(models):
        print(f"\n{model['name']}:")
        print(f"  Joint F1:         {model['joint_f1']:.3f}")
        print(f"  Joint Accuracy:   {model['joint_accuracy']:.1%}")
        print(f"  Pointer Hit:      {model['pointer_hit_rate']:.1%}")
        print(f"  Type Accuracy:    {model['type_accuracy']:.1%}")

    # Select best model
    best_result, best_idx = select_best_model_by_joint_metric(
        results=models,
        metric_name="joint_f1",
        min_pointer_hit=0.4,
        min_type_acc=0.5,
    )

    print("\n" + "-" * 60)
    print(f"✅ Selected: {best_result['name']}")
    print(f"   Joint F1:         {best_result['joint_f1']:.3f}")
    print(f"   Joint Accuracy:   {best_result['joint_accuracy']:.1%}")
    print(f"   Pointer Hit:      {best_result['pointer_hit_rate']:.1%}")
    print(f"   Type Accuracy:    {best_result['type_accuracy']:.1%}")
    print("\n💡 Model B selected: Highest joint_f1 among models meeting minimum thresholds")
    print("   Models D and E filtered out for not meeting min_pointer_hit=0.4 or min_type_acc=0.5")


def main():
    """Run all demos."""
    print("=" * 60)
    print("PHASE 4: JOINT METRICS DEMONSTRATION")
    print("=" * 60)
    print("\nDemonstrating joint metrics for dual-task model evaluation")
    print("where BOTH pointer localization AND type classification")
    print("must be correct for a prediction to count as successful.")

    np.random.seed(42)

    demo_perfect_predictions()
    demo_pointer_bottleneck()
    demo_type_bottleneck()
    demo_model_selection()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Joint metrics require BOTH tasks to be correct")
    print("2. Task contribution breakdown diagnoses bottleneck")
    print("3. Use joint_f1 for model selection with minimum thresholds")
    print("4. Adjust loss_beta based on which task is limiting performance")


if __name__ == "__main__":
    main()
