"""SimpleLSTM: Lightweight LSTM model for time series classification.

Designed as a replacement for RWKV-TS with significantly fewer parameters (~70K vs 655K).
Uses LSTM with multi-head attention for temporal modeling of OHLC data.

Architecture:
- Single LSTM layer (hidden_size=64)
- Multi-head self-attention (4 heads)
- Dropout regularization (0.4)
- Small FC head (64 -> 32 -> num_classes)

Target: ~70K parameters for 98-sample dataset (700:1 ratio vs 6700:1 for RWKV-TS)

Training Configuration (Phase 2):
- Mixup + CutMix augmentation (alpha=0.4, increased for better generalization)
- Temporal augmentation: jitter (50%), scaling (30%), time_warp (30%)
- Early stopping with patience=20 (optimized for Phase 2 with augmentation)
- Learning rate: 5e-4
- Max epochs: 60
- AdamW optimizer with weight_decay=1e-4
- Validation split: 15%
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
from ..utils.temporal_augmentation import TemporalAugmentation
from .base import BaseModel


class SimpleLSTMModel(BaseModel):
    """Simple LSTM with attention for time series classification.

    Lightweight alternative to RWKV-TS with ~10x fewer parameters.
    """

    def __init__(
        self,
        seed: int = 1337,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.4,
        n_epochs: int = 60,
        batch_size: int = 512,
        learning_rate: float = 5e-4,
        device: str = "cpu",
        use_amp: bool = True,
        num_workers: int = 16,
        early_stopping_patience: int = 20,
        val_split: float = 0.15,
        mixup_alpha: float = 0.4,
        cutmix_prob: float = 0.5,
        use_temporal_aug: bool = True,
        jitter_prob: float = 0.5,
        scaling_prob: float = 0.3,
        time_warp_prob: float = 0.3,
        **kwargs,
    ):
        """Initialize SimpleLSTM model.

        Args:
            seed: Random seed for reproducibility
            hidden_size: LSTM hidden dimension (default: 64)
            num_layers: Number of LSTM layers (default: 1)
            num_heads: Number of attention heads (default: 4)
            dropout: Dropout rate (default: 0.4 for strong regularization)
            n_epochs: Number of training epochs (default: 60)
            batch_size: Training batch size (default: 512)
            learning_rate: Learning rate for optimizer (default: 5e-4)
            device: Device to train on ('cpu' or 'cuda')
            use_amp: Use automatic mixed precision (FP16) when device='cuda'
            num_workers: Number of DataLoader worker processes
            early_stopping_patience: Epochs to wait before stopping (default: 20, Phase 2 optimized)
            val_split: Validation split ratio (default: 0.15)
            mixup_alpha: Mixup interpolation strength (default: 0.4, Phase 2 augmentation)
            cutmix_prob: Probability of applying cutmix vs mixup (default: 0.5)
            use_temporal_aug: Enable temporal augmentation (jitter, scaling, time_warp)
            jitter_prob: Probability of applying jitter (default: 0.5)
            scaling_prob: Probability of applying scaling (default: 0.3)
            time_warp_prob: Probability of applying time warping (default: 0.3)
            **kwargs: Additional parameters
        """
        super().__init__(seed=seed)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout
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
        self.use_temporal_aug = use_temporal_aug
        self.jitter_prob = jitter_prob
        self.scaling_prob = scaling_prob
        self.time_warp_prob = time_warp_prob

        set_seed(seed)

        # Initialize temporal augmentation pipeline
        if self.use_temporal_aug:
            self.temporal_aug = TemporalAugmentation(
                jitter_prob=jitter_prob,
                jitter_sigma=0.05,  # 5% noise
                scaling_prob=scaling_prob,
                scaling_sigma=0.1,  # 10% magnitude variation
                permutation_prob=0.0,  # Disabled (breaks temporal order)
                time_warp_prob=time_warp_prob,
                time_warp_sigma=0.2,
                rotation_prob=0.0,  # Disabled (OHLC order matters)
            )

        # Model will be built after seeing input dimension and num classes
        self.model = None
        self.n_classes = None
        self.input_dim = None

    def _build_model(self, input_dim: int, n_classes: int) -> nn.Module:
        """Build SimpleLSTM neural network architecture.

        Args:
            input_dim: Input feature dimension (4 for OHLC)
            n_classes: Number of output classes

        Returns:
            PyTorch model
        """
        class SimpleLSTMNet(nn.Module):
            def __init__(
                self,
                input_dim: int,
                hidden_size: int,
                num_layers: int,
                num_heads: int,
                n_classes: int,
                dropout: float,
            ):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_size = hidden_size

                # LSTM layer
                self.lstm = nn.LSTM(
                    input_dim,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=False  # Unidirectional for causal modeling
                )

                # Multi-head self-attention
                self.attention = nn.MultiheadAttention(
                    hidden_size,
                    num_heads,
                    dropout=dropout,
                    batch_first=True
                )

                # Layer normalization
                self.ln = nn.LayerNorm(hidden_size)

                # Classification head
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, n_classes)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward pass.

                Args:
                    x: Input tensor [batch, seq_len, input_dim]
                       Expected: [B, 105, 4] for OHLC data

                Returns:
                    Logits [batch, n_classes]
                """
                # LSTM processing
                lstm_out, _ = self.lstm(x)  # [B, 105, 64]

                # Self-attention over sequence
                attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # [B, 105, 64]

                # Residual connection + layer norm
                x = self.ln(lstm_out + attn_out)  # [B, 105, 64]

                # Use last timestep for classification
                last_hidden = x[:, -1, :]  # [B, 64]

                # Classification
                logits = self.classifier(last_hidden)  # [B, n_classes]

                return logits

        model = SimpleLSTMNet(
            input_dim=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            n_classes=n_classes,
            dropout=self.dropout_rate,
        )

        return model.to(self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        expansion_start: np.ndarray = None,
        expansion_end: np.ndarray = None,
        unfreeze_encoder_after: int = 0,
    ) -> "SimpleLSTMModel":
        """Train SimpleLSTM model.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, D]
            y: Target labels of shape [N]
            expansion_start: Optional expansion start indices (unused)
            expansion_end: Optional expansion end indices (unused)
            unfreeze_encoder_after: Epoch to unfreeze encoder (0 = never unfreeze,
                                   >0 = unfreeze after N epochs). Used with pre-trained encoder.

        Returns:
            Self for method chaining
        """
        set_seed(self.seed)

        # Handle input shape
        if X.ndim == 2:
            N, D = X.shape
            if D % 4 == 0:
                T = D // 4
                X = X.reshape(N, T, 4)
            else:
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

        print(f"[CLASS BALANCE] Class distribution: {dict(zip(unique_classes, class_counts))}")
        print(f"[LOSS] Using Focal Loss (gamma=2.0) WITHOUT class weights to avoid double correction")

        # Build model
        self.model = self._build_model(self.input_dim, self.n_classes)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[MODEL] SimpleLSTM parameters: {trainable_params:,} (target: ~70K)")
        print(f"[MODEL] Parameter-to-sample ratio: {trainable_params/N:.1f}:1")

        # GPU diagnostic logging
        if self.device.type == "cuda":
            print(f"[GPU] Training on: {torch.cuda.get_device_name(0)}")
            print(f"[GPU] Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"[GPU] Mixed precision (FP16): {self.use_amp}")

        # Convert labels to indices
        y_indices = np.array([self.label_to_idx[label] for label in y])

        # Split into train/val for early stopping
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
                num_workers=0,
                pin_memory=True if self.device.type == "cuda" else False,
            )

        # Setup optimizer and loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        # Use Focal Loss WITHOUT class weights to avoid double correction
        criterion = FocalLoss(gamma=2.0, alpha=None, reduction='mean')

        # Setup mixed precision training
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

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
            # Unfreeze encoder if scheduled (for pre-trained models)
            if unfreeze_encoder_after > 0 and epoch == unfreeze_encoder_after:
                print(f"\n[SSL PRE-TRAINING] Unfreezing LSTM encoder at epoch {epoch + 1}")
                for param in self.model.lstm.parameters():
                    param.requires_grad = True

                # Reduce learning rate after unfreezing (from config)
                from ..config.training_config import MASKED_LSTM_UNFREEZE_LR_REDUCTION
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= MASKED_LSTM_UNFREEZE_LR_REDUCTION
                print(f"[SSL PRE-TRAINING] Reduced LR to {optimizer.param_groups[0]['lr']:.6f}\n")

            # Training phase
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in train_dataloader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                # Apply temporal augmentation first (jitter, scaling, time_warp)
                if self.use_temporal_aug:
                    batch_X = self.temporal_aug.apply_augmentation(batch_X)

                # Then apply mixup/cutmix
                batch_X_aug, y_a, y_b, lam = mixup_cutmix(
                    batch_X, batch_y,
                    mixup_alpha=self.mixup_alpha,
                    cutmix_prob=self.cutmix_prob,
                )

                optimizer.zero_grad()

                # Forward pass with mixed precision
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model(batch_X_aug)
                        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = self.model(batch_X_aug)
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                    loss.backward()
                    optimizer.step()

                # Track metrics
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
                            with torch.cuda.amp.autocast():
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
            expansion_start: Optional (unused)
            expansion_end: Optional (unused)

        Returns:
            Predicted labels of shape [N]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Reshape input
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
            expansion_start: Optional (unused)
            expansion_end: Optional (unused)

        Returns:
            Class probabilities of shape [N, C]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Reshape input
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

        save_dict = {
            'model_state_dict': self.model.state_dict() if self.model is not None else None,
            'label_to_idx': self.label_to_idx if hasattr(self, 'label_to_idx') else None,
            'idx_to_label': self.idx_to_label if hasattr(self, 'idx_to_label') else None,
            'n_classes': self.n_classes,
            'input_dim': self.input_dim,
            'hyperparams': {
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'dropout_rate': self.dropout_rate,
            }
        }

        torch.save(save_dict, path)

    def load(self, path: Path) -> "SimpleLSTMModel":
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
        self.hidden_size = hyperparams['hidden_size']
        self.num_layers = hyperparams['num_layers']
        self.num_heads = hyperparams['num_heads']
        self.dropout_rate = hyperparams['dropout_rate']

        # Rebuild and load model
        self.model = self._build_model(self.input_dim, self.n_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.is_fitted = True
        return self

    def load_pretrained_encoder(
        self,
        encoder_path: Path,
        freeze_encoder: bool = True
    ) -> "SimpleLSTMModel":
        """Load pre-trained bidirectional LSTM encoder weights.

        Maps bidirectional encoder weights to unidirectional LSTM.
        The bidirectional encoder has 2x parameters (forward + backward),
        so we extract only the forward direction weights.

        Args:
            encoder_path: Path to pre-trained encoder checkpoint (.pt file)
            freeze_encoder: If True, freeze LSTM weights during initial training

        Returns:
            Self with pre-trained encoder loaded

        Raises:
            ValueError: If model not built yet or hidden size mismatch
        """
        if self.model is None:
            raise ValueError(
                "Model must be built first. Call fit() or _build_model() before loading encoder."
            )

        print(f"[SSL PRE-TRAINING] Loading pre-trained encoder from: {encoder_path}")

        # Load checkpoint
        checkpoint = torch.load(encoder_path, map_location=self.device)
        encoder_state_dict = checkpoint['encoder_state_dict']
        hyperparams = checkpoint['hyperparams']

        # Verify architecture compatibility
        pretrained_hidden = hyperparams['hidden_dim']
        if self.hidden_size != pretrained_hidden:
            raise ValueError(
                f"Hidden size mismatch: SimpleLSTM={self.hidden_size}, "
                f"Pre-trained encoder={pretrained_hidden}"
            )

        print(f"[SSL PRE-TRAINING] Architecture verified (hidden_dim={pretrained_hidden})")

        # Map bidirectional LSTM weights to unidirectional LSTM
        # PyTorch bidirectional LSTM structure:
        #   - weight_ih_l0: Forward input-hidden weights [hidden*4, input]
        #   - weight_ih_l0_reverse: Backward input-hidden weights [hidden*4, input]
        #   - weight_hh_l0: Forward hidden-hidden weights [hidden*4, hidden]
        #   - weight_hh_l0_reverse: Backward hidden-hidden weights [hidden*4, hidden]
        #   - bias_ih_l0, bias_hh_l0: Forward biases [hidden*4]
        #   - bias_ih_l0_reverse, bias_hh_l0_reverse: Backward biases [hidden*4]
        #
        # Strategy: Copy ONLY forward direction weights (ignore _reverse parameters)

        model_state_dict = self.model.state_dict()
        loaded_keys = []

        for key in encoder_state_dict:
            # Skip backward (reverse) direction weights
            if '_reverse' in key:
                continue

            # Map encoder_lstm.weight_XX_lY → lstm.weight_XX_lY
            model_key = key.replace('encoder_lstm.', 'lstm.')

            if model_key in model_state_dict:
                # Verify shapes match
                encoder_shape = encoder_state_dict[key].shape
                model_shape = model_state_dict[model_key].shape

                if encoder_shape == model_shape:
                    model_state_dict[model_key] = encoder_state_dict[key]
                    loaded_keys.append(model_key)
                else:
                    print(f"[SSL PRE-TRAINING] WARNING: Shape mismatch for {model_key}:")
                    print(f"  Expected: {model_shape}, Got: {encoder_shape}")
            else:
                print(f"[SSL PRE-TRAINING] WARNING: Key not found in model: {model_key}")

        # Load mapped weights into model
        self.model.load_state_dict(model_state_dict)

        print(f"[SSL PRE-TRAINING] Loaded {len(loaded_keys)} parameter tensors:")
        for key in loaded_keys:
            print(f"  ✓ {key}")

        # Freeze encoder if requested
        if freeze_encoder:
            print(f"[SSL PRE-TRAINING] Freezing LSTM encoder weights")
            for param in self.model.lstm.parameters():
                param.requires_grad = False
            print(f"  → Encoder frozen. Only classifier will be trained initially.")
        else:
            print(f"[SSL PRE-TRAINING] Encoder unfrozen. All parameters trainable.")

        return self
