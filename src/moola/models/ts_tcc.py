"""TS-TCC: Time-Series representation learning via Temporal and Contextual Contrasting.

This module implements semi-supervised learning for time series using contrastive pre-training.
The approach is based on TS-TCC (https://arxiv.org/abs/2106.14112) adapted for OHLC financial data.

Architecture:
- Shared Encoder: CNN-Transformer backbone (same as CnnTransformerModel)
- Projection Head: Maps encoder outputs to contrastive embedding space
- InfoNCE Loss: Contrastive loss to learn representations

Training Pipeline:
1. Pre-training Phase: Train encoder on unlabeled data with contrastive loss
2. Fine-tuning Phase: Replace projection head with classifier, fine-tune on labeled data
3. The pre-trained encoder provides better initialization than random weights

Expected Improvement:
- Pre-training on 118k unlabeled windows → +3-5% accuracy vs random init
- Fine-tuning on 115 labeled samples → Target 64-68% accuracy

Reference:
    - Eldele et al. "Time-Series Representation Learning via Temporal and Contextual Contrasting"
      https://arxiv.org/abs/2106.14112
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from ..utils.seeds import get_device, set_seed
from ..utils.temporal_augmentation import TemporalAugmentation
from .cnn_transformer import CNNBlock, RelativePositionalEncoding


class TSTCCEncoder(nn.Module):
    """TS-TCC encoder: CNN + Transformer backbone for time series.

    Same architecture as CnnTransformerModel but without classification head.
    This will be used for both pre-training (contrastive) and fine-tuning (classifier).
    """

    def __init__(
        self,
        input_dim: int,
        cnn_channels: list[int],
        cnn_kernels: list[int],
        transformer_layers: int,
        transformer_heads: int,
        dropout: float,
    ):
        """Initialize TS-TCC encoder.

        Args:
            input_dim: Input feature dimension (4 for OHLC)
            cnn_channels: CNN channel sizes (default: [64, 128, 128])
            cnn_kernels: CNN kernel sizes (default: [3, 5, 9])
            transformer_layers: Number of Transformer encoder layers
            transformer_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.cnn_channels = cnn_channels
        self.d_model = cnn_channels[-1]  # Final CNN output becomes Transformer input

        # CNN blocks (local pattern detection)
        in_channels = [input_dim] + cnn_channels[:-1]
        self.cnn_blocks = nn.ModuleList([
            CNNBlock(in_ch, out_ch, cnn_kernels, dropout)
            for in_ch, out_ch in zip(in_channels, cnn_channels)
        ])

        # Transformer encoder (global context modeling)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=transformer_heads,
            dim_feedforward=4 * self.d_model,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

        # Relative positional encoding
        self.rel_pos_enc = RelativePositionalEncoding(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder.

        Args:
            x: Input tensor [batch, seq_len, input_dim]

        Returns:
            Encoded features [batch, seq_len, d_model]
        """
        # Transpose to [batch, input_dim, seq_len] for Conv1d
        x = x.transpose(1, 2)

        # Apply CNN blocks
        for cnn_block in self.cnn_blocks:
            x = cnn_block(x)

        # Transpose back to [batch, seq_len, d_model] for Transformer
        x = x.transpose(1, 2)

        # Add relative positional encoding
        B, T, C = x.shape
        rel_pos_encoding = self.rel_pos_enc(T)  # [T, T, C]
        pos_bias = rel_pos_encoding.mean(dim=1)  # [T, C]
        x = x + pos_bias.unsqueeze(0)  # Add positional bias [1, T, C]

        # Apply Transformer encoder
        x = self.transformer(x)  # [B, seq_len, d_model]

        return x


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning.

    Maps encoder output to low-dimensional embedding space for contrastive loss.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        """Initialize projection head.

        Args:
            input_dim: Encoder output dimension (d_model)
            hidden_dim: Hidden layer dimension
            output_dim: Final embedding dimension
        """
        super().__init__()
        # Use LayerNorm instead of BatchNorm for stability in contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project encoder features to embedding space.

        Args:
            x: Encoder output [batch, d_model] (pooled features)

        Returns:
            Projected embeddings [batch, output_dim]
        """
        return self.projection(x)


class TSTCC(nn.Module):
    """TS-TCC model for contrastive pre-training.

    Combines encoder and projection head for contrastive learning.
    """

    def __init__(
        self,
        input_dim: int,
        cnn_channels: list[int] = None,
        cnn_kernels: list[int] = None,
        transformer_layers: int = 3,
        transformer_heads: int = 4,
        dropout: float = 0.25,
        projection_hidden_dim: int = 128,
        projection_output_dim: int = 64,
    ):
        """Initialize TS-TCC model.

        Args:
            input_dim: Input feature dimension (4 for OHLC)
            cnn_channels: CNN channel sizes (default: [64, 128, 128])
            cnn_kernels: CNN kernel sizes (default: [3, 5, 9])
            transformer_layers: Number of Transformer encoder layers
            transformer_heads: Number of attention heads
            dropout: Dropout rate
            projection_hidden_dim: Projection head hidden dimension
            projection_output_dim: Projection head output dimension
        """
        super().__init__()

        self.cnn_channels = cnn_channels or [64, 128, 128]
        self.cnn_kernels = cnn_kernels or [3, 5, 9]

        # Encoder (shared backbone)
        self.encoder = TSTCCEncoder(
            input_dim=input_dim,
            cnn_channels=self.cnn_channels,
            cnn_kernels=self.cnn_kernels,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            dropout=dropout,
        )

        # Projection head (for contrastive learning)
        self.projection_head = ProjectionHead(
            input_dim=self.encoder.d_model,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_output_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for contrastive learning.

        Args:
            x: Input tensor [batch, seq_len, input_dim]

        Returns:
            Projected embeddings [batch, projection_output_dim]
        """
        # Encode
        features = self.encoder(x)  # [batch, seq_len, d_model]

        # Global average pooling over sequence
        pooled = features.mean(dim=1)  # [batch, d_model]

        # Project to embedding space
        embeddings = self.projection_head(pooled)  # [batch, output_dim]

        return embeddings


def info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5
) -> torch.Tensor:
    """InfoNCE (Normalized Temperature-scaled Cross Entropy) loss.

    Contrastive loss that pulls positive pairs together and pushes negative pairs apart.
    Uses numerically stable implementation for FP16 training.

    Args:
        z1: Embeddings from augmented view 1 [batch, embedding_dim]
        z2: Embeddings from augmented view 2 [batch, embedding_dim]
        temperature: Temperature scaling parameter (default: 0.5, higher for stability)

    Returns:
        Scalar loss value
    """
    batch_size = z1.shape[0]
    device = z1.device

    # Normalize embeddings (L2 normalization)
    z1 = F.normalize(z1, dim=1, p=2)
    z2 = F.normalize(z2, dim=1, p=2)

    # Concatenate both views
    z = torch.cat([z1, z2], dim=0)  # [2*batch, embedding_dim]

    # Compute similarity matrix
    similarity_matrix = torch.mm(z, z.t()) / temperature  # [2*batch, 2*batch]

    # Create labels: positive pair for each sample
    # For sample i: positive is i + batch_size (or i - batch_size if i >= batch_size)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=device),
        torch.arange(0, batch_size, device=device)
    ])

    # Create mask to exclude self-similarity from denominator
    # We'll use cross_entropy which handles this more stably
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)

    # For numerical stability, subtract max before exp (done internally by cross_entropy)
    # We'll manually implement to exclude diagonal
    similarity_matrix_masked = similarity_matrix.clone()

    # Set diagonal to very negative value (but safe for FP16)
    # -100 is safe: exp(-100) ≈ 0, avoids overflow in FP16 (max ~65k)
    similarity_matrix_masked = similarity_matrix_masked.masked_fill(mask, -100.0)

    # Use cross entropy loss (more numerically stable than manual log_softmax)
    # CrossEntropyLoss expects [batch, num_classes] logits and [batch] labels
    loss = F.cross_entropy(similarity_matrix_masked, labels, reduction='mean')

    return loss


class TSTCCPretrainer:
    """Pre-training manager for TS-TCC contrastive learning.

    Handles:
    1. Pre-training on unlabeled data with contrastive loss
    2. Saving pre-trained encoder weights
    3. Loading pre-trained weights for fine-tuning
    """

    def __init__(
        self,
        input_dim: int = 4,
        cnn_channels: list[int] = None,
        cnn_kernels: list[int] = None,
        transformer_layers: int = 3,
        transformer_heads: int = 4,
        dropout: float = 0.25,
        projection_hidden_dim: int = 128,
        projection_output_dim: int = 64,
        temperature: float = 0.5,
        n_epochs: int = 100,
        batch_size: int = 512,
        learning_rate: float = 1e-4,
        device: str = "cpu",
        use_amp: bool = True,
        num_workers: int = 16,
        val_split: float = 0.1,
        early_stopping_patience: int = 15,
        seed: int = 1337,
    ):
        """Initialize TS-TCC pre-trainer.

        Args:
            input_dim: Input feature dimension (4 for OHLC)
            cnn_channels: CNN channel sizes
            cnn_kernels: CNN kernel sizes
            transformer_layers: Number of Transformer layers
            transformer_heads: Number of attention heads
            dropout: Dropout rate
            projection_hidden_dim: Projection head hidden dimension
            projection_output_dim: Projection head output dimension
            temperature: Temperature for InfoNCE loss
            n_epochs: Number of pre-training epochs
            batch_size: Batch size for pre-training (512 for RTX 4090)
            learning_rate: Learning rate
            device: Device to train on ('cpu' or 'cuda')
            use_amp: Use automatic mixed precision (FP16)
            num_workers: Number of DataLoader workers (16 for high-end GPUs)
            val_split: Validation split ratio
            early_stopping_patience: Early stopping patience
            seed: Random seed
        """
        self.seed = seed
        set_seed(seed)

        self.device = get_device(device)
        self.use_amp = use_amp and (device == "cuda") and torch.cuda.is_available()

        # Model
        self.model = TSTCC(
            input_dim=input_dim,
            cnn_channels=cnn_channels,
            cnn_kernels=cnn_kernels,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            dropout=dropout,
            projection_hidden_dim=projection_hidden_dim,
            projection_output_dim=projection_output_dim,
        ).to(self.device)

        # Training hyperparameters
        self.temperature = temperature
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.val_split = val_split
        self.early_stopping_patience = early_stopping_patience

        # Temporal augmentation
        self.augmentation = TemporalAugmentation(
            jitter_prob=0.8,
            jitter_sigma=0.03,
            scaling_prob=0.5,
            scaling_sigma=0.1,
            time_warp_prob=0.5,
            time_warp_sigma=0.2,
        )

    def pretrain(self, X_unlabeled: np.ndarray) -> dict:
        """Pre-train encoder on unlabeled data with contrastive learning.

        Args:
            X_unlabeled: Unlabeled time series [N, seq_len, features]
                         For OHLC: [N, 105, 4]

        Returns:
            Training history dictionary with keys:
            - train_loss: List of training losses per epoch
            - val_loss: List of validation losses per epoch
            - best_epoch: Epoch with lowest validation loss
        """
        set_seed(self.seed)

        # Reshape if needed
        if X_unlabeled.ndim == 2:
            N, D = X_unlabeled.shape
            if D % 4 == 0:
                T = D // 4
                X_unlabeled = X_unlabeled.reshape(N, T, 4)

        print(f"[PRETRAINING] Dataset: {X_unlabeled.shape[0]} unlabeled samples")
        print(f"[PRETRAINING] Input shape: {X_unlabeled.shape}")

        # Split train/val
        if self.val_split > 0:
            X_train, X_val = train_test_split(
                X_unlabeled,
                test_size=self.val_split,
                random_state=self.seed
            )
        else:
            X_train = X_unlabeled
            X_val = None

        print(f"[PRETRAINING] Train: {len(X_train)}, Val: {len(X_val) if X_val is not None else 0}")

        # Create dataloaders
        X_train_tensor = torch.FloatTensor(X_train)
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers if self.device.type == "cuda" else 0,
            pin_memory=True if self.device.type == "cuda" else False,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            val_dataset = torch.utils.data.TensorDataset(X_val_tensor)
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.device.type == "cuda" else False,
            )
        else:
            val_loader = None

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

        # Mixed precision scaler
        scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_epoch': 0,
        }

        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(self.n_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_X, in train_loader:
                batch_X = batch_X.to(self.device, non_blocking=True)

                # Generate two augmented views
                x_aug1, x_aug2 = self.augmentation(batch_X)

                optimizer.zero_grad()

                # Forward pass with mixed precision
                if self.use_amp:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        z1 = self.model(x_aug1)
                        z2 = self.model(x_aug2)
                        loss = info_nce_loss(z1, z2, temperature=self.temperature)

                    scaler.scale(loss).backward()
                    # Gradient clipping to prevent explosion
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    z1 = self.model(x_aug1)
                    z2 = self.model(x_aug2)
                    loss = info_nce_loss(z1, z2, temperature=self.temperature)

                    loss.backward()
                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_X, in val_loader:
                        batch_X = batch_X.to(self.device, non_blocking=True)

                        # Generate two augmented views
                        x_aug1, x_aug2 = self.augmentation(batch_X)

                        if self.use_amp:
                            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                                z1 = self.model(x_aug1)
                                z2 = self.model(x_aug2)
                                loss = info_nce_loss(z1, z2, temperature=self.temperature)
                        else:
                            z1 = self.model(x_aug1)
                            z2 = self.model(x_aug2)
                            loss = info_nce_loss(z1, z2, temperature=self.temperature)

                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                history['val_loss'].append(avg_val_loss)

                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    history['best_epoch'] = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    print(f"[PRETRAINING] Early stopping at epoch {epoch+1}")
                    break

                # Log progress
                if (epoch + 1) % max(1, self.n_epochs // 10) == 0:
                    print(f"Epoch [{epoch+1}/{self.n_epochs}] "
                          f"Train Loss: {avg_train_loss:.4f} | "
                          f"Val Loss: {avg_val_loss:.4f}")
            else:
                if (epoch + 1) % max(1, self.n_epochs // 10) == 0:
                    print(f"Epoch [{epoch+1}/{self.n_epochs}] "
                          f"Train Loss: {avg_train_loss:.4f}")

        print(f"\n[PRETRAINING] Complete! Best epoch: {history['best_epoch']+1}")
        print(f"[PRETRAINING] Best validation loss: {best_val_loss:.4f}")

        return history

    def save_encoder(self, path: Path) -> None:
        """Save pre-trained encoder weights.

        Args:
            path: Path to save encoder weights
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save only encoder weights (not projection head)
        save_dict = {
            'encoder_state_dict': self.model.encoder.state_dict(),
            'hyperparams': {
                'cnn_channels': self.model.cnn_channels,
                'cnn_kernels': self.model.cnn_kernels,
            }
        }

        torch.save(save_dict, path)
        print(f"[PRETRAINING] Saved encoder weights to {path}")

    def load_encoder(self, path: Path) -> dict:
        """Load pre-trained encoder weights.

        Args:
            path: Path to encoder weights file

        Returns:
            Dictionary with encoder_state_dict and hyperparams
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print(f"[PRETRAINING] Loaded encoder weights from {path}")

        return checkpoint
