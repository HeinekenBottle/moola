"""Temperature scaling for model calibration.

Calibrates model confidence scores using temperature scaling.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


class TemperatureScaling(nn.Module):
    """Temperature scaling for calibration.

    Learns a single temperature parameter to scale logits.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling.

        Args:
            logits: Model logits [batch, n_classes]

        Returns:
            Calibrated logits
        """
        return logits / self.temperature

    def fit(
        self, logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 1000, lr: float = 0.01
    ) -> dict:
        """Learn temperature parameter on validation set.

        Args:
            logits: Validation logits [batch, n_classes]
            targets: Validation targets [batch]
            max_iter: Maximum optimization iterations
            lr: Learning rate

        Returns:
            Training metrics
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def loss_fn():
            scaled_logits = self(logits)
            loss = F.cross_entropy(scaled_logits, targets)
            return loss.item()

        # Optimize temperature
        optimizer.step(loss_fn)

        # Compute metrics
        with torch.no_grad():
            scaled_logits = self(logits)
            calibrated_probs = F.softmax(scaled_logits, dim=1)
            original_probs = F.softmax(logits, dim=1)

            # Expected Calibration Error (ECE)
            ece_original = self._compute_ece(original_probs, targets)
            ece_calibrated = self._compute_ece(calibrated_probs, targets)

            # Accuracy
            pred_original = original_probs.argmax(dim=1)
            pred_calibrated = calibrated_probs.argmax(dim=1)
            acc_original = accuracy_score(targets.cpu(), pred_original.cpu())
            acc_calibrated = accuracy_score(targets.cpu(), pred_calibrated.cpu())

        return {
            "temperature": float(self.temperature),
            "ece_original": ece_original,
            "ece_calibrated": ece_calibrated,
            "accuracy_original": acc_original,
            "accuracy_calibrated": acc_calibrated,
            "ece_improvement": ece_original - ece_calibrated,
        }

    def _compute_ece(self, probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error.

        Args:
            probs: Predicted probabilities [batch, n_classes]
            targets: True targets [batch]
            n_bins: Number of confidence bins

        Returns:
            ECE score
        """
        confidence, pred_class = probs.max(dim=1)
        accuracy = (pred_class == targets).float()

        # Create bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin > 0:
                # Compute accuracy and confidence in this bin
                accuracy_in_bin = accuracy[in_bin].mean()
                avg_confidence_in_bin = confidence[in_bin].mean()

                # Add to ECE
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(ece)


def apply_temperature_scaling(
    model: nn.Module, val_loader, device: str = "cuda"
) -> TemperatureScaling:
    """Apply temperature scaling to a trained model.

    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to run on

    Returns:
        Fitted temperature scaler
    """
    model.eval()
    scaler = TemperatureScaling().to(device)

    # Collect all predictions and targets
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, batch.y  # Adjust for your data format

            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            if hasattr(logits, "logits"):  # Handle models with logits attribute
                logits = logits.logits

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Fit temperature scaler
    metrics = scaler.fit(all_logits, all_targets)

    print(f"Temperature: {metrics['temperature']:.4f}")
    print(f"ECE: {metrics['ece_original']:.4f} → {metrics['ece_calibrated']:.4f}")
    print(f"Accuracy: {metrics['accuracy_original']:.4f} → {metrics['accuracy_calibrated']:.4f}")

    return scaler
