"""Example: Phase 3 Uncertainty Quantification with MC Dropout and Temperature Scaling.

This example demonstrates how to use MC Dropout and Temperature Scaling for uncertainty
estimation and probability calibration in the BiLSTM dual-task model.

Usage:
    python3 examples/phase3_uncertainty_example.py
"""

import torch
import torch.nn as nn
import numpy as np
from moola.utils.uncertainty.mc_dropout import (
    mc_dropout_predict,
    get_uncertainty_threshold,
    TemperatureScaling,
    apply_temperature_scaling,
)


# Create a dummy model for demonstration
class DummyDualTaskModel(nn.Module):
    """Simplified version of EnhancedSimpleLSTM for demonstration."""

    def __init__(self, input_dim=11, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim * 2, 2)
        self.pointer_head = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)

        type_logits = self.classifier(last_hidden)
        pointers = torch.sigmoid(self.pointer_head(last_hidden))

        return {"type_logits": type_logits, "pointers": pointers}


def main():
    print("=" * 80)
    print("Phase 3: Uncertainty Quantification Example")
    print("=" * 80)

    # Create dummy model
    model = DummyDualTaskModel()
    model.eval()

    # Create dummy test data (N=10 samples, T=105 timesteps, D=11 features)
    batch_size = 10
    seq_len = 105
    input_dim = 11
    X_test = torch.randn(batch_size, seq_len, input_dim)

    print(f"\nTest data shape: {X_test.shape}")
    print(f"Model: DummyDualTaskModel (BiLSTM + dual-task)")

    # ========================================================================
    # MC Dropout Uncertainty Estimation
    # ========================================================================
    print("\n" + "=" * 80)
    print("MC DROPOUT UNCERTAINTY ESTIMATION")
    print("=" * 80)

    print(f"\nRunning 50 forward passes with dropout enabled...")
    mc_results = mc_dropout_predict(
        model=model, x=X_test, n_passes=50, dropout_rate=0.15
    )

    print(f"\nType classification uncertainty:")
    print(f"  Mean predictive entropy: {mc_results['type_entropy'].mean():.4f}")
    print(
        f"  Entropy range: [{mc_results['type_entropy'].min():.4f}, {mc_results['type_entropy'].max():.4f}]"
    )

    print(f"\nPointer regression uncertainty:")
    mean_ptr_std = mc_results["pointer_std"].mean(axis=0)
    print(f"  Mean center std: {mean_ptr_std[0]:.4f}")
    print(f"  Mean length std: {mean_ptr_std[1]:.4f}")

    # Flag high-uncertainty samples
    threshold = get_uncertainty_threshold(mc_results["type_entropy"], percentile=90)
    high_uncertainty = mc_results["type_entropy"] > threshold
    print(
        f"\nHigh uncertainty samples (top 10%): {high_uncertainty.sum()} / {len(high_uncertainty)}"
    )
    print(f"  Uncertainty threshold: {threshold:.4f}")

    # Show per-sample breakdown
    print(f"\nPer-sample uncertainty breakdown:")
    print("  Sample | Type Entropy | Center Std | Length Std | High Unc?")
    print("  " + "-" * 62)
    for i in range(batch_size):
        is_uncertain = "YES" if high_uncertainty[i] else "NO"
        print(
            f"  {i:6d} | {mc_results['type_entropy'][i]:12.4f} | "
            f"{mc_results['pointer_std'][i, 0]:10.4f} | "
            f"{mc_results['pointer_std'][i, 1]:10.4f} | {is_uncertain:>9s}"
        )

    # ========================================================================
    # Temperature Scaling Calibration
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEMPERATURE SCALING CALIBRATION")
    print("=" * 80)

    # Create dummy validation dataloader
    val_size = 32
    X_val = torch.randn(val_size, seq_len, input_dim)
    y_val = torch.randint(0, 2, (val_size,))
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, shuffle=False
    )

    print(f"\nFitting temperature on validation set ({val_size} samples)...")
    temp_scaler, optimal_temp = apply_temperature_scaling(
        model, val_dataloader, device="cpu"
    )

    print(f"\nOptimal temperature: {optimal_temp:.4f}")
    if optimal_temp > 1.5:
        print("  Model is OVERCONFIDENT (temperature > 1.5)")
    elif optimal_temp < 0.7:
        print("  Model is UNDERCONFIDENT (temperature < 0.7)")
    else:
        print("  Model is well-calibrated (temperature ≈ 1.0)")

    # Show effect of temperature scaling
    print(f"\nEffect of temperature scaling on probabilities:")
    sample_logits = model(X_test[:3])["type_logits"]  # First 3 samples
    uncalibrated_probs = torch.softmax(sample_logits, dim=-1)
    calibrated_logits = temp_scaler(sample_logits)
    calibrated_probs = torch.softmax(calibrated_logits, dim=-1)

    print("  Sample | Uncalibrated P(class=1) | Calibrated P(class=1)")
    print("  " + "-" * 58)
    for i in range(3):
        print(
            f"  {i:6d} | {uncalibrated_probs[i, 1]:23.4f} | {calibrated_probs[i, 1]:21.4f}"
        )

    # ========================================================================
    # Combined Example: Production Inference
    # ========================================================================
    print("\n" + "=" * 80)
    print("PRODUCTION INFERENCE WITH UNCERTAINTY")
    print("=" * 80)

    print(f"\nProcessing {batch_size} samples with uncertainty estimates...")

    # Get predictions with uncertainty
    type_probs = mc_results["type_probs_mean"]  # [N, 2]
    type_uncertainty = mc_results["type_entropy"]  # [N]
    pointer_preds = mc_results["pointer_mean"]  # [N, 2]
    pointer_uncertainty = mc_results["pointer_std"]  # [N, 2]

    # Apply temperature scaling to logits for calibrated probabilities
    test_logits = model(X_test)["type_logits"]
    calibrated_logits = temp_scaler(test_logits)
    calibrated_probs = torch.softmax(calibrated_logits, dim=-1).detach().numpy()

    print(f"\nPrediction summary:")
    print(
        "  Sample | Pred Class | Calibrated P | Uncertainty | Pointer (C,L) | Ptr Unc (C,L)"
    )
    print("  " + "-" * 84)
    for i in range(batch_size):
        pred_class = calibrated_probs[i].argmax()
        pred_prob = calibrated_probs[i, pred_class]
        unc = type_uncertainty[i]
        ptr = pointer_preds[i]
        ptr_unc = pointer_uncertainty[i]
        print(
            f"  {i:6d} | {pred_class:10d} | {pred_prob:12.4f} | "
            f"{unc:11.4f} | ({ptr[0]:.2f}, {ptr[1]:.2f}) | "
            f"({ptr_unc[0]:.3f}, {ptr_unc[1]:.3f})"
        )

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)

    # Key takeaways
    print("\nKey Takeaways:")
    print("  1. MC Dropout provides uncertainty estimates for both tasks")
    print("  2. Higher entropy = more uncertain classification")
    print("  3. Higher std dev = more uncertain pointer regression")
    print("  4. Temperature scaling improves probability calibration")
    print("  5. Combine both for production-ready uncertainty-aware predictions")
    print("\nFor real training, use:")
    print("  python3 -m moola.cli train \\")
    print("    --model enhanced_simple_lstm \\")
    print("    --predict-pointers \\")
    print("    --mc-dropout \\")
    print("    --mc-passes 50 \\")
    print("    --temperature-scaling \\")
    print("    --device cuda")


if __name__ == "__main__":
    main()
