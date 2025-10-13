"""Example: Multi-Task Pointer Prediction Training

This script demonstrates how to train the CNN-Transformer model with
multi-task learning for both classification and pointer prediction.

Phase 3 - Multi-Task Pointer Prediction System
"""

import numpy as np
from pathlib import Path
import sys

# Add moola to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moola.models.cnn_transformer import CnnTransformerModel
from moola.utils.metrics import compute_pointer_metrics


def generate_mock_data(n_samples=100):
    """Generate mock OHLC data with pointer labels for demonstration."""
    np.random.seed(42)

    # Generate OHLC data: [N, 105, 4]
    X = np.random.randn(n_samples, 105, 4).astype(np.float32)

    # Generate classification labels
    y = np.random.choice(['consolidation', 'retracement', 'reversal'], size=n_samples)

    # Generate pointer labels (relative to inner window [0, 45))
    pointer_starts = np.random.randint(0, 45, size=n_samples)
    pointer_ends = np.random.randint(0, 45, size=n_samples)

    # Ensure start < end for most samples
    for i in range(n_samples):
        if pointer_starts[i] >= pointer_ends[i]:
            pointer_ends[i] = min(pointer_starts[i] + np.random.randint(1, 10), 44)

    return X, y, pointer_starts, pointer_ends


def example_single_task_training():
    """Example 1: Single-task classification (backward compatible)."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single-Task Classification (Backward Compatible)")
    print("="*80)

    # Generate data (without pointers)
    X, y, _, _ = generate_mock_data(n_samples=100)

    # Initialize model WITHOUT pointer prediction
    model = CnnTransformerModel(
        seed=1337,
        n_epochs=5,
        batch_size=16,
        predict_pointers=False,  # Single-task mode
        device='cpu'
    )

    print(f"\nTraining single-task model on {len(X)} samples...")
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X[:10])
    y_proba = model.predict_proba(X[:10])

    print(f"\nPredictions (first 10):")
    print(f"Labels: {y_pred}")
    print(f"Probabilities shape: {y_proba.shape}")

    return model


def example_multitask_training():
    """Example 2: Multi-task classification + pointer prediction."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Multi-Task Classification + Pointer Prediction")
    print("="*80)

    # Generate data WITH pointers
    X, y, pointer_starts, pointer_ends = generate_mock_data(n_samples=100)

    # Initialize model WITH pointer prediction
    model = CnnTransformerModel(
        seed=1337,
        n_epochs=5,
        batch_size=16,
        predict_pointers=True,  # Multi-task mode
        loss_alpha=0.5,  # Weight for classification loss
        loss_beta=0.25,  # Weight for each pointer loss
        device='cpu'
    )

    print(f"\nTraining multi-task model on {len(X)} samples...")
    print(f"Pointer start range: [{pointer_starts.min()}, {pointer_starts.max()}]")
    print(f"Pointer end range: [{pointer_ends.min()}, {pointer_ends.max()}]")

    # Train with pointer labels
    model.fit(X, y, pointer_starts=pointer_starts, pointer_ends=pointer_ends)

    # Multi-task predictions
    results = model.predict_with_pointers(X[:10])

    print(f"\nMulti-task predictions (first 10 samples):")
    print(f"Classification labels: {results['labels']}")
    print(f"Classification probs shape: {results['probabilities'].shape}")
    print(f"Start predictions: {results['start_predictions']}")
    print(f"End predictions: {results['end_predictions']}")
    print(f"Start probabilities shape: {results['start_probabilities'].shape}")
    print(f"End probabilities shape: {results['end_probabilities'].shape}")

    return model, results


def example_pointer_evaluation():
    """Example 3: Evaluate pointer prediction metrics."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Pointer Prediction Evaluation")
    print("="*80)

    # Generate test data
    X, y, pointer_starts, pointer_ends = generate_mock_data(n_samples=50)

    # Train model
    model = CnnTransformerModel(
        seed=1337,
        n_epochs=5,
        batch_size=16,
        predict_pointers=True,
        device='cpu'
    )

    print(f"\nTraining model on {len(X)} samples...")
    model.fit(X, y, pointer_starts=pointer_starts, pointer_ends=pointer_ends)

    # Get predictions
    results = model.predict_with_pointers(X)

    # Compute pointer metrics
    metrics = compute_pointer_metrics(
        start_preds=results['start_probabilities'],
        end_preds=results['end_probabilities'],
        start_true=pointer_starts,
        end_true=pointer_ends,
        k=3
    )

    print("\nPointer Prediction Metrics:")
    print("-" * 60)
    print(f"Start AUC:              {metrics['start_auc']:.4f}")
    print(f"End AUC:                {metrics['end_auc']:.4f}")
    print(f"Start Precision@3:      {metrics['start_precision_at_k']:.4f} ({metrics['start_precision_at_k']*100:.1f}%)")
    print(f"End Precision@3:        {metrics['end_precision_at_k']:.4f} ({metrics['end_precision_at_k']*100:.1f}%)")
    print(f"Start Exact Accuracy:   {metrics['start_exact_accuracy']:.4f} ({metrics['start_exact_accuracy']*100:.1f}%)")
    print(f"End Exact Accuracy:     {metrics['end_exact_accuracy']:.4f} ({metrics['end_exact_accuracy']*100:.1f}%)")
    print(f"Avg Start Error:        {metrics['avg_start_error']:.2f} timesteps")
    print(f"Avg End Error:          {metrics['avg_end_error']:.2f} timesteps")

    return metrics


def example_save_load():
    """Example 4: Save and load multi-task model."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Save and Load Multi-Task Model")
    print("="*80)

    # Generate data
    X, y, pointer_starts, pointer_ends = generate_mock_data(n_samples=50)

    # Train model
    model = CnnTransformerModel(
        seed=1337,
        n_epochs=3,
        batch_size=16,
        predict_pointers=True,
        device='cpu'
    )

    print(f"\nTraining model...")
    model.fit(X, y, pointer_starts=pointer_starts, pointer_ends=pointer_ends)

    # Get predictions before save
    results_before = model.predict_with_pointers(X[:5])

    # Save model
    save_path = Path("/tmp/multitask_model.pt")
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")

    # Load model
    loaded_model = CnnTransformerModel(device='cpu')
    loaded_model.load(save_path)
    print(f"Model loaded from: {save_path}")

    # Get predictions after load
    results_after = loaded_model.predict_with_pointers(X[:5])

    # Verify predictions match
    labels_match = np.array_equal(results_before['labels'], results_after['labels'])
    start_match = np.array_equal(results_before['start_predictions'], results_after['start_predictions'])
    end_match = np.array_equal(results_before['end_predictions'], results_after['end_predictions'])

    print(f"\nVerification:")
    print(f"  Classification labels match: {labels_match}")
    print(f"  Start predictions match:     {start_match}")
    print(f"  End predictions match:       {end_match}")
    print(f"  Model state preserved:       {labels_match and start_match and end_match}")

    return loaded_model


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("MULTI-TASK POINTER PREDICTION SYSTEM - EXAMPLE USAGE")
    print("="*80)

    # Example 1: Single-task (backward compatible)
    model_single = example_single_task_training()

    # Example 2: Multi-task training
    model_multi, results = example_multitask_training()

    # Example 3: Pointer metrics
    metrics = example_pointer_evaluation()

    # Example 4: Save/load
    loaded_model = example_save_load()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Single-task mode (predict_pointers=False) remains fully backward compatible")
    print("2. Multi-task mode (predict_pointers=True) requires pointer_starts and pointer_ends")
    print("3. Use predict_with_pointers() to get all task predictions")
    print("4. Use compute_pointer_metrics() for comprehensive evaluation")
    print("5. Save/load preserves multi-task state correctly")
    print("\nSee code for implementation details.")


if __name__ == "__main__":
    main()
