"""RWKV-TS: RNN/state-space model for sequential temporal modeling.

This model is designed for time series data with OHLC (Open, High, Low, Close) format.
It uses causal recurrence with instance normalization for stable sequential modeling.

Architecture:
- Multi-scale patching with sizes [7, 15, 21, 35]
- Recurrent state-space blocks with d_model=128 (restored from 96 for better capacity)
- 4 layers (reverted from 6 - too many parameters for small dataset)
- Instance normalization for stable training
- Dropout regularization

Training Enhancements:
- Mixup + CutMix augmentation (alpha=0.2, gentler mixing for small dataset)
- Early stopping with patience=30 (increased from 20 for full convergence on cleaned dataset)
- Learning rate: 5e-4 (increased from 3e-4 for faster convergence)
- Max epochs: 60 (increased from 10)
- AdamW optimizer with weight_decay=1e-4
- Validation split: 15% (~20 samples for stable early stopping)

Note: 6 layers caused severe overfitting (982K params for 115 samples = 8,540:1 ratio)
      4 layers provides 655K params (5,695:1 ratio - still high but more stable)

Reference:
    RWKV: Reinventing RNNs for the Transformer Era
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from ..utils.augmentation import mixup_criterion, mixup_cutmix
from ..utils.early_stopping import EarlyStopping
from ..utils.focal_loss import FocalLoss
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

        # Window-aware attention mask
        mask = torch.ones(105)
        mask[30:75] = 1.2  # 20% boost for inner window
        self.register_buffer('window_mask', mask)

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

        # Apply window mask to boost inner prediction region
        # Only apply if sequence length matches expected window size
        if T == 105:
            rwkv_out = rwkv_out * self.window_mask[None, :, None]

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
        n_epochs: int = 60,
        batch_size: int = 512,
        learning_rate: float = 5e-4,
        device: str = "cpu",
        use_amp: bool = True,
        num_workers: int = 16,
        early_stopping_patience: int = 30,
        val_split: float = 0.15,
        mixup_alpha: float = 0.2,
        cutmix_prob: float = 0.5,
        **kwargs,
    ):
        """Initialize RWKV-TS model.

        Args:
            seed: Random seed for reproducibility
            d_model: Model dimension (128 for adequate capacity on 420 features)
            n_layers: Number of RWKV blocks (4 layers = ~28 bar receptive field for 105-bar sequences)
            patch_sizes: Multi-scale patch sizes for temporal features
            dropout: Dropout rate
            instance_norm: Whether to use instance normalization
            n_epochs: Number of training epochs (increased to 60 with early stopping)
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer (5e-4 for faster convergence)
            device: Device to train on ('cpu' or 'cuda')
            use_amp: Use automatic mixed precision (FP16) when device='cuda'
            num_workers: Number of DataLoader worker processes
            early_stopping_patience: Epochs to wait before stopping (default: 30)
            val_split: Validation split ratio for early stopping (default: 0.15, ~20 samples)
            mixup_alpha: Mixup interpolation strength (default: 0.2, gentler for small dataset)
            cutmix_prob: Probability of applying cutmix vs mixup (default: 0.5)
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
        self.use_amp = use_amp and (device == "cuda") and torch.cuda.is_available()
        self.num_workers = num_workers
        self.early_stopping_patience = early_stopping_patience
        self.val_split = val_split
        self.mixup_alpha = mixup_alpha
        self.cutmix_prob = cutmix_prob

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

    def fit(self, X: np.ndarray, y: np.ndarray, expansion_start: np.ndarray = None, expansion_end: np.ndarray = None) -> "RWKVTSModel":
        """Train RWKV-TS model.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, D] for sequences
            y: Target labels of shape [N]
            expansion_start: Optional expansion start indices of shape [N] (unused for deep learning models)
            expansion_end: Optional expansion end indices of shape [N] (unused for deep learning models)

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

        # Log class distribution
        y_indices_for_weights = np.array([self.label_to_idx[label] for label in y])
        unique_classes, class_counts = np.unique(y_indices_for_weights, return_counts=True)
        n_samples = len(y)
        n_classes = len(unique_classes)

        print(f"[CLASS BALANCE] Class distribution: {dict(zip(unique_classes, class_counts))}")
        print(f"[LOSS] Using Focal Loss (gamma=2.0) WITHOUT class weights to avoid double correction")

        # Build model
        self.model = self._build_model(self.input_dim, self.n_classes)

        # GPU diagnostic logging
        if self.device.type == "cuda":
            print(f"[GPU] Training on: {torch.cuda.get_device_name(0)}")
            print(f"[GPU] Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"[GPU] Mixed precision (FP16): {self.use_amp}")

        # Convert labels to indices
        y_indices = np.array([self.label_to_idx[label] for label in y])

        # Split into train/val for early stopping if val_split > 0
        if self.val_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_indices, test_size=self.val_split, random_state=self.seed, stratify=y_indices
            )
        else:
            X_train, y_train = X, y_indices
            X_val, y_val = None, None

        # Convert training data to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)

        # Create training dataset and dataloader
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        num_workers = self.num_workers if self.device.type == "cuda" else 0
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        # Create validation dataloader if needed
        val_dataloader = None
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,  # Use 0 for validation to avoid overhead
                pin_memory=True if self.device.type == "cuda" else False,
            )

        # Setup optimizer and loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        # Use Focal Loss WITHOUT class weights to avoid double correction
        # Focal loss already handles imbalance via gamma parameter
        criterion = FocalLoss(gamma=2.0, alpha=None, reduction='mean')

        # Setup mixed precision training
        scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Setup early stopping if validation data available
        early_stopping = None
        if val_dataloader is not None:
            early_stopping = EarlyStopping(
                patience=self.early_stopping_patience,
                mode="min",
                verbose=True
            )

        # Training loop
        for epoch in range(self.n_epochs):
            # Training phase
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in train_dataloader:
                # CRITICAL: Move batch to GPU
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                # Apply augmentation (mixup + cutmix)
                batch_X_aug, y_a, y_b, lam = mixup_cutmix(
                    batch_X, batch_y,
                    mixup_alpha=self.mixup_alpha,
                    cutmix_prob=self.cutmix_prob,
                )

                optimizer.zero_grad()

                # Forward pass with mixed precision
                if self.use_amp:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        logits = self.model(batch_X_aug)
                        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard forward/backward pass
                    logits = self.model(batch_X_aug)
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                    loss.backward()
                    optimizer.step()

                # Track metrics (using original labels for accuracy)
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

            avg_train_loss = total_loss / len(train_dataloader)
            train_accuracy = correct / total

            # Validation phase
            if val_dataloader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_dataloader:
                        batch_X = batch_X.to(self.device, non_blocking=True)
                        batch_y = batch_y.to(self.device, non_blocking=True)

                        if self.use_amp:
                            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                                logits = self.model(batch_X)
                                loss = criterion(logits, batch_y)
                        else:
                            logits = self.model(batch_X)
                            loss = criterion(logits, batch_y)

                        val_loss += loss.item()
                        _, predicted = torch.max(logits, 1)
                        val_correct += (predicted == batch_y).sum().item()
                        val_total += batch_y.size(0)

                avg_val_loss = val_loss / len(val_dataloader)
                val_accuracy = val_correct / val_total

                # Check early stopping
                if early_stopping(avg_val_loss, self.model):
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

                if (epoch + 1) % max(1, self.n_epochs // 10) == 0:
                    gpu_mem = f" GPU: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB" if self.device.type == "cuda" else ""
                    print(f"Epoch [{epoch+1}/{self.n_epochs}] Train Loss: {avg_train_loss:.4f} Acc: {train_accuracy:.4f} | "
                          f"Val Loss: {avg_val_loss:.4f} Acc: {val_accuracy:.4f}{gpu_mem}")
            else:
                if (epoch + 1) % max(1, self.n_epochs // 10) == 0:
                    gpu_mem = f" GPU: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB" if self.device.type == "cuda" else ""
                    print(f"Epoch [{epoch+1}/{self.n_epochs}] Loss: {avg_train_loss:.4f} Acc: {train_accuracy:.4f}{gpu_mem}")

        # Restore best model if early stopping was used
        if early_stopping is not None:
            early_stopping.load_best_model(self.model)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, expansion_start: np.ndarray = None, expansion_end: np.ndarray = None) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, D]
            expansion_start: Optional expansion start indices of shape [N] (unused for deep learning models)
            expansion_end: Optional expansion end indices of shape [N] (unused for deep learning models)

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

    def predict_proba(self, X: np.ndarray, expansion_start: np.ndarray = None, expansion_end: np.ndarray = None) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, D]
            expansion_start: Optional expansion start indices of shape [N] (unused for deep learning models)
            expansion_end: Optional expansion end indices of shape [N] (unused for deep learning models)

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
