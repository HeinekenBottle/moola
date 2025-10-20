"""Bidirectional LSTM Autoencoder for time series representation learning.

This module implements a Bi-LSTM autoencoder for self-supervised pre-training on unlabeled time series data.
The encoder can be used to initialize downstream models (SimpleLSTM, etc.) via transfer learning.

Pre-training objective: Reconstruct input time series from compressed latent representation.
Fine-tuning: Use encoder as feature extractor for classification tasks.

Example:
    # Pre-training
    >>> pretrainer = BiLSTMPretrainer(device="cuda")
    >>> X_unlabeled = load_unlabeled_data()  # [11873, 105, 4]
    >>> history = pretrainer.pretrain(X_unlabeled, n_epochs=50)
    >>> pretrainer.save_encoder("artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt")

    # Fine-tuning
    >>> model = SimpleLSTMModel()
    >>> model.load_pretrained_encoder("artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt")
    >>> model.fit(X_train, y_train)
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from ..utils.seeds import get_device, set_seed


class BiLSTMAutoencoder(nn.Module):
    """Bidirectional LSTM Autoencoder architecture.

    Architecture:
        Input [B, T, F] → Bi-LSTM Encoder → Latent [B, Z]
                                                ↓
        Output [B, T, F] ← LSTM Decoder ← Expanded Latent [B, T, H]

    Args:
        input_dim: Input feature dimension (default: 4 for OHLC)
        hidden_dim: LSTM hidden dimension (default: 128)
        latent_dim: Latent representation dimension (default: 64)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout rate (default: 0.2)
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder: Bi-LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Encoder projection: Bi-LSTM hidden → Latent
        self.encoder_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder projection: Latent → LSTM hidden
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        # Decoder: LSTM (unidirectional)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder output: LSTM hidden → Input reconstruction
        self.decoder_output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode time series to latent representation.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]

        Returns:
            Latent representation [batch_size, latent_dim]
        """
        # Bi-LSTM encoding
        lstm_out, (h_n, c_n) = self.encoder_lstm(x)
        # h_n shape: [num_layers * 2, batch_size, hidden_dim]

        # Concatenate final forward and backward hidden states
        h_forward = h_n[-2, :, :]  # [batch_size, hidden_dim]
        h_backward = h_n[-1, :, :]  # [batch_size, hidden_dim]
        h_concat = torch.cat(
            [h_forward, h_backward], dim=1
        )  # [batch_size, hidden_dim * 2]

        # Project to latent space
        latent = self.encoder_fc(h_concat)  # [batch_size, latent_dim]
        return latent

    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode latent representation to time series reconstruction.

        Args:
            latent: Latent tensor [batch_size, latent_dim]
            seq_len: Output sequence length

        Returns:
            Reconstructed time series [batch_size, seq_len, input_dim]
        """
        batch_size = latent.size(0)

        # Project latent to hidden space
        h_0 = self.decoder_fc(latent)  # [batch_size, hidden_dim]

        # Initialize LSTM hidden states
        h_0 = h_0.unsqueeze(0).repeat(
            self.num_layers, 1, 1
        )  # [num_layers, batch_size, hidden_dim]
        c_0 = torch.zeros_like(h_0)

        # Generate sequence input (broadcast latent across all timesteps)
        decoder_input = latent.unsqueeze(1).repeat(
            1, seq_len, 1
        )  # [batch_size, seq_len, latent_dim]
        decoder_input = self.decoder_fc(
            decoder_input
        )  # [batch_size, seq_len, hidden_dim]

        # LSTM decoding
        lstm_out, _ = self.decoder_lstm(
            decoder_input, (h_0, c_0)
        )  # [batch_size, seq_len, hidden_dim]

        # Project to output space
        reconstruction = self.decoder_output(
            lstm_out
        )  # [batch_size, seq_len, input_dim]
        return reconstruction

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass: Encode → Decode.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]

        Returns:
            Tuple of (reconstruction, latent)
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent, seq_len=x.size(1))
        return reconstruction, latent


class BiLSTMPretrainer:
    """Pre-trainer for Bi-LSTM Autoencoder.

    Handles pre-training on unlabeled data and encoder weight serialization.

    Args:
        input_dim: Input feature dimension
        hidden_dim: LSTM hidden dimension
        latent_dim: Latent representation dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        learning_rate: AdamW learning rate
        batch_size: Training batch size
        device: Device for training ('cpu' or 'cuda')
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        device: str = "cuda",
        seed: int = 1337,
    ):
        self.device = get_device(device)
        set_seed(seed)

        # Build model
        self.model = BiLSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )

        # Training config
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

    def pretrain(
        self, X: np.ndarray, n_epochs: int = 50, patience: int = 10, val_split: float = 0.1
    ) -> dict:
        """Pre-train autoencoder on unlabeled time series data.

        Training uses MSE reconstruction loss with latent regularization to prevent collapse.

        Args:
            X: Unlabeled time series data [N, seq_len, input_dim]
            n_epochs: Number of training epochs
            patience: Early stopping patience
            val_split: Validation split ratio

        Returns:
            Training history with keys:
                - train_loss: List of training losses
                - val_loss: List of validation losses
                - best_epoch: Best epoch index
                - best_val_loss: Best validation loss
        """
        logger.info(
            f"[PRETRAINING] Starting Bi-LSTM AE pre-training on {len(X)} samples"
        )

        # Split train/val
        N = len(X)
        val_size = int(N * val_split)
        indices = np.random.permutation(N)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        X_train = torch.FloatTensor(X[train_indices]).to(self.device)
        X_val = torch.FloatTensor(X[val_indices]).to(self.device)

        logger.info(
            f"[PRETRAINING] Train: {len(X_train)}, Val: {len(X_val)} | Input shape: {X.shape}"
        )

        # Dataloaders
        train_dataset = torch.utils.data.TensorDataset(X_train)
        num_workers = 4 if self.device.type == "cuda" else 0
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # Training loop
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        for epoch in range(n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            recon_loss_accum = 0.0
            reg_loss_accum = 0.0

            for (batch_X,) in train_loader:
                self.optimizer.zero_grad()

                # Forward
                x_recon, latent = self.model(batch_X)

                # Reconstruction loss (MSE)
                recon_loss = F.mse_loss(x_recon, batch_X)

                # Latent regularization (prevent collapse)
                # Encourage latent std >= 1.0 to maintain diverse representations
                latent_std = torch.std(latent, dim=0).mean()
                reg_loss = torch.relu(1.0 - latent_std)

                # Total loss
                loss = recon_loss + 0.1 * reg_loss

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                self.optimizer.step()

                train_loss += loss.item()
                recon_loss_accum += recon_loss.item()
                reg_loss_accum += reg_loss.item()

            train_loss /= len(train_loader)
            recon_loss_accum /= len(train_loader)
            reg_loss_accum /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                x_recon_val, latent_val = self.model(X_val)
                val_recon_loss = F.mse_loss(x_recon_val, X_val).item()
                val_latent_std = torch.std(latent_val, dim=0).mean().item()
                val_loss = val_recon_loss + 0.1 * torch.relu(
                    torch.tensor(1.0 - val_latent_std)
                ).item()

            history["val_loss"].append(val_loss)

            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{n_epochs}] "
                    f"Train Loss: {train_loss:.6f} (recon: {recon_loss_accum:.6f}, reg: {reg_loss_accum:.6f}) | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Val Latent Std: {val_latent_std:.4f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                logger.debug(
                    f"[PRETRAINING] New best validation loss: {best_val_loss:.6f}"
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        f"[PRETRAINING] Early stopping at epoch {epoch+1} (best epoch: {best_epoch+1})"
                    )
                    break

        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val_loss

        logger.success(
            f"[PRETRAINING] Complete! Best epoch: {best_epoch+1}, Best val loss: {best_val_loss:.6f}"
        )

        return history

    def save_encoder(self, path: Path):
        """Save encoder weights for downstream fine-tuning.

        Saves encoder LSTM and projection layers only (decoder is discarded).

        Args:
            path: Output path for encoder weights (.pt file)
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Extract encoder state dict
        encoder_state_dict = {
            k: v
            for k, v in self.model.state_dict().items()
            if k.startswith("encoder_")
        }

        # Save checkpoint
        checkpoint = {
            "encoder_state_dict": encoder_state_dict,
            "hyperparams": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
                "num_layers": self.num_layers,
            },
        }
        torch.save(checkpoint, path)

        file_size_mb = path.stat().st_size / (1024 * 1024)
        logger.success(
            f"[PRETRAINING] Saved encoder weights to {path} ({file_size_mb:.2f} MB)"
        )
