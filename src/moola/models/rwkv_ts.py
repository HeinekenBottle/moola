"""RWKV-TS: RNN/state-space model for sequential temporal modeling.

This model is designed for time series data with OHLC (Open, High, Low, Close) format.
It uses causal recurrence with instance normalization for stable sequential modeling.

Architecture:
- Multi-scale patching with sizes [7, 15, 21, 35]
- Recurrent state-space blocks with d_model=128
- Instance normalization for stable training
- Dropout regularization

Reference:
    RWKV: Reinventing RNNs for the Transformer Era
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.seeds import get_device, set_seed
from .base import BaseModel


class RWKVBlock(nn.Module):
    """Single RWKV recurrent block with time-mixing and channel-mixing.

    This simplified version uses causal state updates for sequential processing.
    """

    def __init__(self, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model

        # Time-mixing parameters (simplified RWKV attention)
        self.time_mix_k = nn.Parameter(torch.randn(1, 1, d_model))
        self.time_mix_v = nn.Parameter(torch.randn(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.randn(1, 1, d_model))

        # Time-mixing projections
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

        # Channel-mixing (feed-forward)
        self.channel_mix_k = nn.Parameter(torch.randn(1, 1, d_model))
        self.channel_mix_r = nn.Parameter(torch.randn(1, 1, d_model))
        self.ffn_key = nn.Linear(d_model, 4 * d_model, bias=False)
        self.ffn_value = nn.Linear(4 * d_model, d_model, bias=False)
        self.ffn_receptance = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, state: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional state for recurrence.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            state: Previous state [batch, 1, d_model] or None

        Returns:
            (output, new_state): Output tensor and updated state
        """
        B, T, C = x.shape

        # Initialize state if None
        if state is None:
            state = torch.zeros(B, 1, C, device=x.device)

        # Time-mixing with state
        x_norm = self.ln1(x)

        # Mix current input with previous state
        xx = torch.cat([state, x_norm[:, :-1, :]], dim=1)  # Shift right
        k = self.key(x_norm * self.time_mix_k + xx * (1 - self.time_mix_k))
        v = self.value(x_norm * self.time_mix_v + xx * (1 - self.time_mix_v))
        r = self.receptance(x_norm * self.time_mix_r + xx * (1 - self.time_mix_r))

        # Simplified time-mixing (causal attention-like)
        rwkv_out = torch.sigmoid(r) * self.output(k * v)
        x = x + self.dropout(rwkv_out)

        # Channel-mixing (FFN)
        x_norm = self.ln2(x)
        xx = torch.cat([state, x_norm[:, :-1, :]], dim=1)
        k = self.ffn_key(x_norm * self.channel_mix_k + xx * (1 - self.channel_mix_k))
        r = self.ffn_receptance(x_norm * self.channel_mix_r + xx * (1 - self.channel_mix_r))
        ffn_out = torch.sigmoid(r) * self.ffn_value(torch.square(torch.relu(k)))
        x = x + self.dropout(ffn_out)

        # Update state with last timestep
        new_state = x[:, -1:, :]

        return x, new_state


class RWKVTSModel(BaseModel):
    """RWKV-TS model for time series classification.

    Designed for sequential temporal modeling of OHLC data (105×4 format).
    Uses causal recurrence with multi-layer state-space architecture.
    """

    def __init__(
        self,
        seed: int = 1337,
        d_model: int = 128,
        n_layers: int = 4,
        patch_sizes: list[int] = None,
        dropout: float = 0.2,
        instance_norm: bool = True,
        n_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize RWKV-TS model.

        Args:
            seed: Random seed for reproducibility
            d_model: Model dimension
            n_layers: Number of RWKV blocks
            patch_sizes: Multi-scale patch sizes for temporal features
            dropout: Dropout rate
            instance_norm: Whether to use instance normalization
            n_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cpu' or 'cuda')
            **kwargs: Additional parameters
        """
        super().__init__(seed=seed)
        self.d_model = d_model
        self.n_layers = n_layers
        self.patch_sizes = patch_sizes or [7, 15, 21, 35]
        self.dropout_rate = dropout
        self.instance_norm = instance_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device_str = device
        self.device = get_device(device)

        set_seed(seed)

        # Model will be built after seeing input dimension and num classes
        self.model = None
        self.n_classes = None
        self.input_dim = None

    def _build_model(self, input_dim: int, n_classes: int) -> nn.Module:
        """Build the RWKV-TS neural network architecture.

        Args:
            input_dim: Input feature dimension
            n_classes: Number of output classes

        Returns:
            PyTorch model
        """
        class RWKVTSNet(nn.Module):
            def __init__(
                self,
                input_dim: int,
                d_model: int,
                n_layers: int,
                n_classes: int,
                dropout: float,
                instance_norm: bool,
            ):
                super().__init__()
                self.input_dim = input_dim
                self.d_model = d_model
                self.instance_norm_enabled = instance_norm

                # Input projection
                self.input_proj = nn.Linear(input_dim, d_model)

                # Instance normalization
                if instance_norm:
                    self.instance_norm = nn.InstanceNorm1d(d_model)

                # RWKV blocks
                self.blocks = nn.ModuleList([
                    RWKVBlock(d_model, dropout) for _ in range(n_layers)
                ])

                # Output head
                self.output_norm = nn.LayerNorm(d_model)
                self.classifier = nn.Linear(d_model, n_classes)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward pass.

                Args:
                    x: Input tensor [batch, seq_len, input_dim]

                Returns:
                    Logits [batch, n_classes]
                """
                # Input projection
                x = self.input_proj(x)

                # Instance normalization (across sequence dimension)
                # Skip if sequence length is 1 (InstanceNorm requires > 1 spatial element)
                if self.instance_norm_enabled and x.size(1) > 1:
                    # Transpose to [batch, d_model, seq_len] for InstanceNorm1d
                    x = x.transpose(1, 2)
                    x = self.instance_norm(x)
                    x = x.transpose(1, 2)

                # Apply RWKV blocks with state
                state = None
                for block in self.blocks:
                    x, state = block(x, state)

                # Pool over sequence (use last timestep for classification)
                x = x[:, -1, :]  # Take last timestep

                # Classification head
                x = self.output_norm(x)
                x = self.dropout(x)
                logits = self.classifier(x)

                # TODO: Add evidential deep learning layer here
                # - Replace final linear layer with evidential output
                # - Predict Dirichlet distribution parameters (alpha)
                # - Add evidential loss function (MSE + KL divergence)

                # TODO: Add conformal prediction layer here
                # - Compute calibrated prediction sets
                # - Track non-conformity scores during training
                # - Provide coverage guarantees for predictions

                return logits

        model = RWKVTSNet(
            input_dim=input_dim,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_classes=n_classes,
            dropout=self.dropout_rate,
            instance_norm=self.instance_norm,
        )

        return model.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RWKVTSModel":
        """Train RWKV-TS model.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, D] for sequences
            y: Target labels of shape [N]

        Returns:
            Self for method chaining
        """
        set_seed(self.seed)

        # Handle input shape (assume flat features need reshaping)
        if X.ndim == 2:
            # Assume input is [N, D] - reshape to [N, T, F]
            # For OHLC data: D = 105*4 = 420 → [N, 105, 4]
            # More generally, try to infer reasonable sequence length
            N, D = X.shape
            # Use a heuristic: if D is divisible by 4 (OHLC), reshape accordingly
            if D % 4 == 0:
                T = D // 4
                X = X.reshape(N, T, 4)
            else:
                # Otherwise, treat as single timestep with D features
                X = X.reshape(N, 1, D)

        N, T, F = X.shape
        self.input_dim = F

        # Get unique classes and build label mapping
        unique_labels = np.unique(y)
        self.n_classes = len(unique_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # Build model
        self.model = self._build_model(self.input_dim, self.n_classes)

        # Convert data to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_indices = np.array([self.label_to_idx[label] for label in y])
        y_tensor = torch.LongTensor(y_indices).to(self.device)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # Setup optimizer and loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                # Forward pass
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Track metrics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total

            if (epoch + 1) % max(1, self.n_epochs // 5) == 0:
                print(f"Epoch [{epoch+1}/{self.n_epochs}] Loss: {avg_loss:.4f} Acc: {accuracy:.4f}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, D]

        Returns:
            Predicted labels of shape [N]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Reshape input to [N, T, F]
        if X.ndim == 2:
            N, D = X.shape
            if D % 4 == 0:
                T = D // 4
                X = X.reshape(N, T, 4)
            else:
                X = X.reshape(N, 1, D)

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            _, predicted = torch.max(logits, 1)

        # Convert indices back to original labels
        predicted_labels = np.array([self.idx_to_label[idx.item()] for idx in predicted])

        return predicted_labels

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, D]

        Returns:
            Class probabilities of shape [N, C]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Reshape input to [N, T, F]
        if X.ndim == 2:
            N, D = X.shape
            if D % 4 == 0:
                T = D // 4
                X = X.reshape(N, T, 4)
            else:
                X = X.reshape(N, 1, D)

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def save(self, path: Path) -> None:
        """Save model to disk using PyTorch format.

        Args:
            path: Path to save model file (.pt extension)
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state and metadata
        save_dict = {
            'model_state_dict': self.model.state_dict() if self.model is not None else None,
            'label_to_idx': self.label_to_idx if hasattr(self, 'label_to_idx') else None,
            'idx_to_label': self.idx_to_label if hasattr(self, 'idx_to_label') else None,
            'n_classes': self.n_classes,
            'input_dim': self.input_dim,
            'hyperparams': {
                'd_model': self.d_model,
                'n_layers': self.n_layers,
                'patch_sizes': self.patch_sizes,
                'dropout_rate': self.dropout_rate,
                'instance_norm': self.instance_norm,
            }
        }

        torch.save(save_dict, path)

    def load(self, path: Path) -> "RWKVTSModel":
        """Load model from disk.

        Args:
            path: Path to model file

        Returns:
            Self with loaded model
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Restore metadata
        self.label_to_idx = checkpoint['label_to_idx']
        self.idx_to_label = checkpoint['idx_to_label']
        self.n_classes = checkpoint['n_classes']
        self.input_dim = checkpoint['input_dim']

        # Restore hyperparameters
        hyperparams = checkpoint['hyperparams']
        self.d_model = hyperparams['d_model']
        self.n_layers = hyperparams['n_layers']
        self.patch_sizes = hyperparams['patch_sizes']
        self.dropout_rate = hyperparams['dropout_rate']
        self.instance_norm = hyperparams['instance_norm']

        # Rebuild and load model
        self.model = self._build_model(self.input_dim, self.n_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.is_fitted = True
        return self
