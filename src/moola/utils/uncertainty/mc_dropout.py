"""Monte Carlo Dropout and Temperature Scaling for Uncertainty Quantification.

Implements Phase 3 uncertainty estimation for the BiLSTM dual-task model:
- Monte Carlo Dropout: Multiple forward passes with dropout enabled for uncertainty estimation
- Temperature Scaling: Probability calibration using validation set optimization

References:
    - MC Dropout: Gal & Ghahramani (2016) - "Dropout as a Bayesian Approximation"
    - Temperature Scaling: Guo et al. (2017) - "On Calibration of Modern Neural Networks"

Usage:
    >>> from moola.utils.uncertainty.mc_dropout import mc_dropout_predict, apply_temperature_scaling
    >>>
    >>> # MC Dropout inference
    >>> results = mc_dropout_predict(model, x, n_passes=50, dropout_rate=0.15)
    >>> print(f"Mean entropy: {results['type_entropy'].mean():.4f}")
    >>>
    >>> # Temperature scaling
    >>> temp_scaler, optimal_temp = apply_temperature_scaling(model, val_loader, device='cuda')
    >>> print(f"Optimal temperature: {optimal_temp:.4f}")
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers for MC Dropout inference.

    Switches all dropout layers to training mode while keeping batch norm and other
    layers in eval mode. This allows stochastic forward passes for uncertainty estimation.

    Args:
        model: PyTorch model with dropout layers
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module, x: torch.Tensor, n_passes: int = 50, dropout_rate: float = 0.15
) -> Dict[str, np.ndarray]:
    """Perform MC Dropout inference with multiple forward passes.

    Runs the model multiple times with dropout enabled to estimate predictive uncertainty.
    For classification tasks, uses predictive entropy as the uncertainty measure.
    For regression tasks (pointers), uses standard deviation across predictions.

    Args:
        model: Trained model with dropout layers
        x: Input tensor (batch, seq_len, features)
        n_passes: Number of forward passes (default: 50)
        dropout_rate: Dropout rate for uncertainty estimation (default: 0.15, recommended)

    Returns:
        Dictionary with:
            - type_probs_mean: Mean type probabilities (batch, n_classes)
            - type_probs_std: Std dev of type probabilities (batch, n_classes)
            - type_entropy: Predictive entropy (batch,)
            - pointer_mean: Mean pointer predictions (batch, 2)
            - pointer_std: Std dev of pointer predictions (batch, 2)
            - all_type_probs: All type predictions (n_passes, batch, n_classes)
            - all_pointers: All pointer predictions (n_passes, batch, 2)

    Notes:
        - Higher entropy indicates higher uncertainty for classification
        - Higher std dev indicates higher uncertainty for regression (pointers)
        - Recommended n_passes: 50-100 for production (trade-off speed vs accuracy)
        - Recommended dropout_rate: 0.10-0.20 (lower = less uncertainty, higher = more)
    """
    # Save original training mode
    original_training = model.training

    # Set model to eval mode (disable batch norm updates)
    model.eval()

    # Enable dropout layers only
    enable_dropout(model)

    all_type_probs = []
    all_pointers = []

    with torch.no_grad():
        for _ in range(n_passes):
            outputs = model(x)

            # Type classification probabilities
            type_logits = outputs["type_logits"]
            type_probs = torch.softmax(type_logits, dim=-1)
            all_type_probs.append(type_probs.cpu().numpy())

            # Pointer regression predictions
            pointers = outputs["pointers"]
            all_pointers.append(pointers.cpu().numpy())

    # Restore original training mode
    model.train(original_training)

    # Stack predictions
    all_type_probs = np.stack(all_type_probs, axis=0)  # (n_passes, batch, n_classes)
    all_pointers = np.stack(all_pointers, axis=0)  # (n_passes, batch, 2)

    # Compute statistics
    type_probs_mean = all_type_probs.mean(axis=0)  # (batch, n_classes)
    type_probs_std = all_type_probs.std(axis=0)  # (batch, n_classes)

    # Predictive entropy: -sum(p * log(p)) where p is mean probability
    # Higher entropy = higher uncertainty
    type_entropy = -np.sum(type_probs_mean * np.log(type_probs_mean + 1e-10), axis=1)

    pointer_mean = all_pointers.mean(axis=0)  # (batch, 2)
    pointer_std = all_pointers.std(axis=0)  # (batch, 2)

    return {
        "type_probs_mean": type_probs_mean,
        "type_probs_std": type_probs_std,
        "type_entropy": type_entropy,
        "pointer_mean": pointer_mean,
        "pointer_std": pointer_std,
        "all_type_probs": all_type_probs,
        "all_pointers": all_pointers,
    }


def get_uncertainty_threshold(entropy: np.ndarray, percentile: float = 90) -> float:
    """Compute uncertainty threshold for flagging uncertain predictions.

    Determines a threshold above which predictions are considered "high uncertainty"
    based on the specified percentile of the entropy distribution.

    Args:
        entropy: Predictive entropy values (n_samples,)
        percentile: Percentile for threshold (default: 90 = flag top 10% uncertain)

    Returns:
        Entropy threshold value

    Examples:
        >>> entropy = mc_results['type_entropy']
        >>> threshold = get_uncertainty_threshold(entropy, percentile=90)
        >>> high_uncertainty = entropy > threshold
        >>> print(f"High uncertainty samples: {high_uncertainty.sum()} / {len(entropy)}")
    """
    return np.percentile(entropy, percentile)


class TemperatureScaling(nn.Module):
    """Temperature scaling for probability calibration.

    Learns a single temperature parameter to scale logits for better calibration.
    Temperature scaling does not change model predictions (argmax is invariant),
    but improves the reliability of predicted probabilities.

    Attributes:
        temperature: Learnable temperature parameter (initialized to 1.0)

    Reference:
        Guo et al. (2017) - "On Calibration of Modern Neural Networks"

    Example:
        >>> temp_scaler = TemperatureScaling()
        >>> optimal_temp = temp_scaler.fit(val_logits, val_labels)
        >>> calibrated_logits = temp_scaler(test_logits)
        >>> calibrated_probs = torch.softmax(calibrated_logits, dim=-1)
    """

    def __init__(self):
        super().__init__()
        # Initialize to 1.0 (no scaling)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature.

        Args:
            logits: Model logits (batch, n_classes)

        Returns:
            Temperature-scaled logits (batch, n_classes)

        Notes:
            - temperature > 1: Softer probabilities (less confident)
            - temperature < 1: Sharper probabilities (more confident)
            - temperature = 1: No scaling (identity)
        """
        return logits / self.temperature

    def fit(
        self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50, lr: float = 0.01
    ) -> float:
        """Fit temperature parameter using validation set.

        Optimizes temperature to minimize negative log-likelihood on validation data.

        Args:
            logits: Validation set logits (n_samples, n_classes)
            labels: True labels (n_samples,)
            max_iter: Maximum optimization iterations (default: 50)
            lr: Learning rate (default: 0.01)

        Returns:
            Optimal temperature value

        Notes:
            - Uses L-BFGS optimizer for fast convergence
            - Typical optimal temperature range: 0.5 to 3.0
            - Higher temperature indicates model is overconfident
        """
        # Move to same device as logits
        self.temperature = nn.Parameter(torch.ones(1, device=logits.device))

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = nn.CrossEntropyLoss()(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        return self.temperature.item()


def apply_temperature_scaling(
    model: nn.Module, val_loader: torch.utils.data.DataLoader, device: str = "cuda"
) -> Tuple[TemperatureScaling, float]:
    """Learn and apply temperature scaling on validation set.

    Extracts logits from the validation set and fits a temperature parameter
    to improve probability calibration.

    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device for computation ('cuda' or 'cpu')

    Returns:
        Tuple of:
            - Fitted TemperatureScaling module
            - Optimal temperature value

    Example:
        >>> temp_scaler, optimal_temp = apply_temperature_scaling(model, val_loader, device='cuda')
        >>> print(f"Optimal temperature: {optimal_temp:.4f}")
        >>>
        >>> # Use for calibrated inference
        >>> test_logits = model(test_data)['type_logits']
        >>> calibrated_logits = temp_scaler(test_logits)
        >>> calibrated_probs = torch.softmax(calibrated_logits, dim=-1)

    Notes:
        - Validation set should NOT be used for training the main model
        - Temperature scaling is fast (typically <1 second)
        - Improves Expected Calibration Error (ECE) without changing predictions
    """
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            # Unpack batch (handle different dataloader formats)
            if len(batch) == 2:
                # Single-task: (windows, labels)
                windows, pattern_type = batch
            elif len(batch) == 4:
                # Multi-task: (windows, labels, ptr_start, ptr_end)
                windows, pattern_type, _, _ = batch
            else:
                # Dict format: {'window': ..., 'pattern_type': ...}
                windows = batch["window"]
                pattern_type = batch["pattern_type"]

            windows = windows.to(device)
            pattern_type = pattern_type.to(device)

            outputs = model(windows)

            # Extract type logits (handle dict or tensor output)
            if isinstance(outputs, dict):
                logits = outputs["type_logits"]
            else:
                logits = outputs

            all_logits.append(logits.cpu())
            all_labels.append(pattern_type.cpu())

    logits = torch.cat(all_logits, dim=0).to(device)
    labels = torch.cat(all_labels, dim=0).to(device)

    # Fit temperature scaling
    temp_scaler = TemperatureScaling()
    optimal_temp = temp_scaler.fit(logits, labels)

    return temp_scaler, optimal_temp
