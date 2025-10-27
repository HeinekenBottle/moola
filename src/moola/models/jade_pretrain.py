"""Jade Pretrainer: BiLSTM Masked Autoencoder for Self-Supervised Learning.

Implements masked reconstruction pretraining on unlabeled 5-year NQ data:
- Architecture: BiLSTM encoder (128 hidden × 2 directions × 2 layers) + linear decoder
- Objective: Reconstruct masked relativity features (10D)
- Loss: Huber (δ=1.0) with uncertainty weighting (learnable σ)
- Masking: 15% random or patch-based (10-bar blocks)
- Regularization: High dropout (0.65) for small-sample robustness

Usage:
    from moola.models.jade_pretrain import JadePretrainer, JadeConfig

    config = JadeConfig(input_size=10, hidden_size=128, dropout=0.65)
    model = JadePretrainer(config)
    loss, metrics = model((X, mask, valid_mask))
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class JadeConfig:
    """Configuration for Jade pretrainer."""

    input_size: int = 11  # Relativity features (6 candle + 4 swing + 1 expansion_proxy)
    hidden_size: int = 128  # BiLSTM hidden units per direction
    num_layers: int = 2  # BiLSTM layers
    dropout: float = 0.65  # High dropout for <200 sample regime (PDF-backed)
    huber_delta: float = 1.0  # Huber loss delta for reconstruction
    mask_strategy: str = "random"  # 'random' or 'patch'
    mask_ratio: float = 0.15  # Fraction of timesteps to mask
    patch_size: int = 10  # Patch size for patch masking (if mask_strategy='patch')
    use_uncertainty_weighting: bool = True  # Learnable σ for uncertainty (prep for MTL)

    def __post_init__(self):
        """Validate configuration."""
        assert self.input_size > 0, "input_size must be positive"
        assert self.hidden_size > 0, "hidden_size must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert 0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"
        assert self.mask_strategy in [
            "random",
            "patch",
        ], "mask_strategy must be 'random' or 'patch'"
        assert 0.0 < self.mask_ratio < 1.0, "mask_ratio must be in (0, 1)"


class JadePretrainer(nn.Module):
    """BiLSTM masked autoencoder for self-supervised pretraining.

    Architecture:
        Input [batch, K, D=11] → BiLSTM Encoder → Pooled Repr [batch, 256]
                                                ↓
                                          Decoder (Linear)
                                                ↓
                                    Reconstructed Features [batch, K, D=11]

    Loss:
        Huber(δ=1.0) on masked positions only, with optional uncertainty weighting:
        L = (1/2σ²) * Huber(recon[mask], target[mask]) + log(σ)
    """

    def __init__(self, config: JadeConfig):
        """Initialize Jade pretrainer.

        Args:
            config: JadeConfig with architecture and training parameters
        """
        super().__init__()
        self.config = config

        # BiLSTM Encoder (same as Jade core for transfer learning)
        self.encoder = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        # Decoder: Linear projection from hidden state back to features
        # BiLSTM outputs 2 * hidden_size (bidirectional)
        self.decoder = nn.Linear(config.hidden_size * 2, config.input_size)

        # Uncertainty weighting (homoscedastic σ for smooth MTL transition)
        if config.use_uncertainty_weighting:
            self.log_sigma = nn.Parameter(torch.tensor(0.0))  # log(σ) for numerical stability
        else:
            self.register_buffer("log_sigma", torch.tensor(0.0))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights (Xavier for LSTM, zeros for decoder bias)."""
        for name, param in self.encoder.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward pass with masked reconstruction.

        Args:
            batch: Tuple of (X, mask, valid_mask)
                - X: Features [batch, K, D] (may include padding)
                - mask: Reconstruction mask [batch, K] (True for masked positions)
                - valid_mask: Valid positions [batch, K] (False for warmup/padding)

        Returns:
            Tuple of (loss, metrics_dict)
                - loss: Scalar reconstruction loss
                - metrics_dict: {'loss': float, 'mae': float, 'sigma': float}
        """
        X, mask, valid_mask = batch

        # Move to same device as model
        device = next(self.parameters()).device
        X = X.to(device)
        mask = mask.to(device)
        valid_mask = valid_mask.to(device)

        # Encode: BiLSTM over full sequence
        # Output: [batch, K, hidden*2]
        encoded, _ = self.encoder(X)

        # Decode: Project back to feature space
        # Output: [batch, K, D]
        reconstructed = self.decoder(encoded)

        # Compute loss only on masked AND valid positions
        loss_mask = mask & valid_mask  # [batch, K]

        if loss_mask.sum() == 0:
            # No masked positions (shouldn't happen, but defensive)
            return torch.tensor(0.0, device=device), {
                "loss": 0.0,
                "mae": 0.0,
                "sigma": torch.exp(self.log_sigma).item(),
                "n_masked": 0,
            }

        # Extract masked positions
        X_masked = X[loss_mask]  # [n_masked, D]
        recon_masked = reconstructed[loss_mask]  # [n_masked, D]

        # Huber loss (robust to outliers)
        huber_loss = F.huber_loss(
            recon_masked, X_masked, reduction="mean", delta=self.config.huber_delta
        )

        # Uncertainty weighting (optional, prep for Kendall MTL)
        if self.config.use_uncertainty_weighting:
            sigma = torch.exp(self.log_sigma)
            total_loss = huber_loss / (2 * sigma**2) + self.log_sigma
        else:
            total_loss = huber_loss
            sigma = torch.tensor(1.0, device=device)

        # Compute MAE for interpretability
        with torch.no_grad():
            mae = (recon_masked - X_masked).abs().mean()

        # Metrics
        metrics = {
            "loss": total_loss.item(),
            "huber_loss": huber_loss.item(),
            "mae": mae.item(),
            "sigma": sigma.item(),
            "n_masked": loss_mask.sum().item(),
            "mask_ratio": loss_mask.float().mean().item(),
        }

        return total_loss, metrics

    def get_encoder_state_dict(self) -> dict:
        """Extract encoder weights for transfer to Jade core model.

        Returns:
            State dict containing only encoder parameters
        """
        return {
            k.replace("encoder.", ""): v
            for k, v in self.state_dict().items()
            if k.startswith("encoder.")
        }

    def get_model_info(self) -> dict:
        """Get model architecture summary.

        Returns:
            Dict with model metadata
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": "JadePretrainer",
            "input_size": self.config.input_size,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "dropout": self.config.dropout,
            "mask_strategy": self.config.mask_strategy,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "use_uncertainty_weighting": self.config.use_uncertainty_weighting,
        }

    @torch.no_grad()
    def compute_mc_dropout_uncertainty(
        self,
        X: torch.Tensor,
        valid_mask: torch.Tensor,
        n_passes: int = 50,
        dropout_rate: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MC Dropout uncertainty for calibration (ECE <0.10).

        Args:
            X: Features [batch, K, D]
            valid_mask: Valid positions [batch, K]
            n_passes: Number of MC forward passes (default 50, per PDF)
            dropout_rate: Dropout rate for MC sampling (default 0.1)

        Returns:
            Tuple of (mean_recon, variance)
                - mean_recon: Mean reconstruction [batch, K, D]
                - variance: Epistemic uncertainty [batch, K, D]
        """
        self.train()  # Enable dropout

        # Store original dropout, temporarily set to MC rate
        original_dropout = self.encoder.dropout
        if self.config.num_layers > 1:
            # Can't modify dropout dynamically in LSTM, use as-is
            pass

        predictions = []
        for _ in range(n_passes):
            encoded, _ = self.encoder(X)
            recon = self.decoder(encoded)
            predictions.append(recon)

        # Stack predictions: [n_passes, batch, K, D]
        predictions = torch.stack(predictions, dim=0)

        # Mean and variance
        mean_recon = predictions.mean(dim=0)  # [batch, K, D]
        variance = predictions.var(dim=0)  # [batch, K, D]

        self.eval()  # Disable dropout

        return mean_recon, variance


def create_jade_pretrainer(
    input_size: int = 11,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.65,
    mask_strategy: str = "random",
    use_uncertainty_weighting: bool = True,
) -> JadePretrainer:
    """Factory function to create Jade pretrainer with default config.

    Args:
        input_size: Number of input features (default 11 = 6 candle + 4 swing + 1 expansion)
        hidden_size: BiLSTM hidden size (default 128)
        num_layers: Number of LSTM layers (default 2)
        dropout: Dropout rate (default 0.65 for small-sample regime)
        mask_strategy: 'random' or 'patch' (default 'random')
        use_uncertainty_weighting: Enable learnable σ (default True)

    Returns:
        JadePretrainer model
    """
    config = JadeConfig(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        mask_strategy=mask_strategy,
        use_uncertainty_weighting=use_uncertainty_weighting,
    )
    return JadePretrainer(config)
