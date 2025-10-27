"""Bidirectional Masked LSTM Autoencoder for Self-Supervised Pre-training.

Inspired by BERT masked language modeling and PatchTST masked prediction.
Pre-trains a bidirectional LSTM encoder on unlabeled time series data using
masked reconstruction as the self-supervised objective.

Architecture:
    Input: [Batch, 105, 4] OHLC sequences
        ↓
    Random Masking: 15% of timesteps → MASK_TOKEN
        ↓
    Bidirectional LSTM Encoder: [Batch, 105, 256] (128*2 from bidirectional)
        ↓
    Decoder MLP: [Batch, 105, 4] reconstruction
        ↓
    Loss: MSE on MASKED positions only

Key Features:
    - Three masking strategies: random, block, patch
    - Learnable mask token (optimized during training)
    - Latent regularization to prevent representation collapse
    - Bidirectional context for better temporal modeling
    - Compatible with SimpleLSTM for transfer learning

Expected Performance:
    - Pre-training: ~20 minutes on H100 GPU (11K samples, 50 epochs)
    - Fine-tuning: +8-12% accuracy improvement over baseline
    - Breaks class collapse (Class 1 accuracy: 0% → 45-55%)

Reference:
    - Implementation roadmap: MASKED_LSTM_IMPLEMENTATION_ROADMAP.md
    - Architecture analysis: LSTM_CHART_INTERACTION_ANALYSIS.md
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMMaskedAutoencoder(nn.Module):
    """Bidirectional LSTM autoencoder with masked reconstruction objective.

    This model learns temporal representations by reconstructing masked
    timesteps in OHLC sequences. The bidirectional encoder sees both
    past and future context, enabling better pattern recognition.

    Args:
        input_dim: Input feature dimension (4 for OHLC)
        hidden_dim: LSTM hidden dimension per direction (total: 2*hidden_dim)
        num_layers: Number of stacked LSTM layers
        dropout: Dropout rate for regularization
    """

    def __init__(
        self, input_dim: int = 4, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Learnable mask token (will be optimized during training)
        # Shape: [1, 1, input_dim] for broadcasting
        self.mask_token = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)

        # Encoder: Bidirectional LSTM
        # Critical: bidirectional=True sees both past and future context
        # Output dimension: hidden_dim * 2 (forward + backward)
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,  # CRITICAL: User's explicit requirement
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder: Projects encoded features back to input space
        # Input: [batch, seq_len, hidden_dim*2] (bidirectional)
        # Output: [batch, seq_len, input_dim] (OHLC reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        # Layer norm for training stability
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode masked sequence and reconstruct.

        Args:
            x_masked: [batch, seq_len, input_dim] with masked positions
                     Masked positions replaced with self.mask_token

        Returns:
            reconstruction: [batch, seq_len, input_dim] reconstructed OHLC
        """
        # Encode with bidirectional LSTM
        # encoded shape: [batch, seq_len, hidden_dim*2]
        encoded, _ = self.encoder_lstm(x_masked)

        # Layer norm for training stability
        encoded = self.layer_norm(encoded)

        # Decode to reconstruction
        reconstruction = self.decoder(encoded)

        return reconstruction

    def get_encoder_state_dict(self) -> dict:
        """Extract encoder weights for transfer learning to SimpleLSTM.

        Returns:
            Dictionary with encoder LSTM state dict and hyperparameters
        """
        return {
            "encoder_lstm": self.encoder_lstm.state_dict(),
            "hyperparams": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
            },
        }

    def compute_loss(
        self, reconstruction: torch.Tensor, x_original: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Compute masked reconstruction loss with regularization.

        Critical: Loss ONLY on masked positions (not visible positions).
        This forces encoder to learn from context, not just copy input.

        Args:
            reconstruction: [batch, seq_len, features] reconstructed OHLC
            x_original: [batch, seq_len, features] original OHLC
            mask: [batch, seq_len] boolean mask (True = masked)

        Returns:
            total_loss: Scalar loss for backpropagation
            loss_dict: Dictionary with loss components for logging
        """
        # Reconstruction loss on MASKED positions only
        # This is the key difference from standard autoencoders
        num_masked = mask.sum()
        if num_masked == 0:
            # Edge case: no masked positions (shouldn't happen in practice)
            reconstruction_loss = torch.tensor(0.0, device=reconstruction.device)
        else:
            reconstruction_loss = F.mse_loss(
                reconstruction[mask], x_original[mask], reduction="mean"
            )

        # Optional: Latent regularization to prevent representation collapse
        # Encourage diversity in encoded representations
        # This prevents all encoded vectors from collapsing to the same value
        encoded = self.encoder_lstm(x_original)[0]  # [B, T, D]
        latent_std = torch.std(encoded, dim=(0, 1)).mean()
        reg_loss = torch.relu(1.0 - latent_std)  # Penalize std < 1.0

        # Total loss: reconstruction + small regularization term
        total_loss = reconstruction_loss + 0.1 * reg_loss

        # Logging dictionary for monitoring
        loss_dict = {
            "total": total_loss.item(),
            "reconstruction": reconstruction_loss.item(),
            "regularization": reg_loss.item(),
            "latent_std": latent_std.item(),
            "num_masked": num_masked.item(),
        }

        return total_loss, loss_dict


class MaskingStrategy:
    """Masking strategies for masked autoencoding.

    Implements three approaches:
    1. Random: BERT-style random masking (15% of timesteps)
    2. Block: Contiguous blocks of masked timesteps
    3. Patch: PatchTST-inspired patch-level masking
    """

    @staticmethod
    def mask_random(
        x: torch.Tensor, mask_token: torch.Tensor, mask_ratio: float = 0.15
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly mask 15% of timesteps (BERT-style).

        Args:
            x: [batch, seq_len, features] input tensor
            mask_token: [1, 1, features] learnable mask token
            mask_ratio: Proportion of timesteps to mask (default: 0.15)

        Returns:
            x_masked: Input with masked timesteps replaced by mask_token
            mask: Boolean mask [batch, seq_len] (True = masked)
        """
        B, T, D = x.shape

        # Generate random mask (mask_ratio of positions set to True)
        mask = torch.rand(B, T, device=x.device) < mask_ratio

        # Replace masked positions with learnable mask token
        x_masked = x.clone()
        # Expand mask_token to match number of masked positions
        # mask_token shape: [1, 1, D] → [num_masked, D]
        num_masked = mask.sum()
        if num_masked > 0:
            x_masked[mask] = mask_token.squeeze(0).expand(num_masked, D)

        return x_masked, mask

    @staticmethod
    def mask_block(
        x: torch.Tensor, mask_token: torch.Tensor, mask_ratio: float = 0.15
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask contiguous blocks of timesteps.

        More challenging than random masking - forces learning of
        long-range dependencies to reconstruct missing blocks.

        Example:
            bars 1-20: visible
            bars 21-35: MASKED (block 1)
            bars 36-60: visible
            bars 61-75: MASKED (block 2)
            bars 76-105: visible

        Args:
            x: [batch, seq_len, features] input tensor
            mask_token: [1, 1, features] learnable mask token
            mask_ratio: Proportion of timesteps to mask

        Returns:
            x_masked: Input with masked blocks
            mask: Boolean mask [batch, seq_len]
        """
        B, T, D = x.shape
        block_size = int(T * mask_ratio)

        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)

        # Random start position for each sample in batch
        for i in range(B):
            start_idx = torch.randint(0, T - block_size + 1, (1,), device=x.device).item()
            mask[i, start_idx : start_idx + block_size] = True

        # Apply masking
        x_masked = x.clone()
        num_masked = mask.sum()
        if num_masked > 0:
            x_masked[mask] = mask_token.squeeze(0).expand(num_masked, D)

        return x_masked, mask

    @staticmethod
    def mask_patch(
        x: torch.Tensor, mask_token: torch.Tensor, mask_ratio: float = 0.15, patch_size: int = 7
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask entire patches (subseries) of timesteps (PatchTST-inspired).

        Divides sequence into patches and masks complete patches.
        This approach is effective for time series as it preserves
        local temporal structure within patches.

        Example (105 bars, patch_size=7):
            15 patches total (105/7)
            Mask ~2 patches (15% of 15)
            Patches 3 and 8 masked completely

        Args:
            x: [batch, seq_len, features] input tensor
            mask_token: [1, 1, features] learnable mask token
            mask_ratio: Proportion of patches to mask
            patch_size: Size of each patch in timesteps

        Returns:
            x_masked: Input with masked patches
            mask: Boolean mask [batch, seq_len]
        """
        B, T, D = x.shape
        num_patches = T // patch_size
        num_masked = max(1, int(num_patches * mask_ratio))

        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)

        for i in range(B):
            # Randomly select patches to mask
            masked_patches = torch.randperm(num_patches, device=x.device)[:num_masked]

            # Expand patch mask to full sequence
            for patch_idx in masked_patches:
                start = patch_idx * patch_size
                end = min(start + patch_size, T)
                mask[i, start:end] = True

        # Apply masking
        x_masked = x.clone()
        num_masked = mask.sum()
        if num_masked > 0:
            x_masked[mask] = mask_token.squeeze(0).expand(num_masked, D)

        return x_masked, mask


def apply_masking(
    x: torch.Tensor,
    mask_token: torch.Tensor,
    mask_strategy: Literal["random", "block", "patch"] = "random",
    mask_ratio: float = 0.15,
    patch_size: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply masking strategy to input tensor.

    Args:
        x: [batch, seq_len, features] input tensor
        mask_token: [1, 1, features] learnable mask token
        mask_strategy: Masking approach ("random", "block", "patch")
        mask_ratio: Proportion of timesteps/patches to mask
        patch_size: Patch size for patch masking strategy

    Returns:
        x_masked: Input with masked positions
        mask: Boolean mask indicating masked positions
    """
    if mask_strategy == "random":
        return MaskingStrategy.mask_random(x, mask_token, mask_ratio)
    elif mask_strategy == "block":
        return MaskingStrategy.mask_block(x, mask_token, mask_ratio)
    elif mask_strategy == "patch":
        return MaskingStrategy.mask_patch(x, mask_token, mask_ratio, patch_size)
    else:
        raise ValueError(f"Unknown mask_strategy: {mask_strategy}")
