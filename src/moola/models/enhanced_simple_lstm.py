"""Enhanced SimpleLSTM with Feature-Aware Transfer Learning Support.

Enhanced version of SimpleLSTM that can handle both OHLC-only input (4-dim) and
OHLC+features input (25-30+ dim) for seamless transfer learning from feature-aware
pre-trained encoders.

Key Features:
    - Dual input support: OHLC-only or OHLC+features
    - Feature-aware encoder transfer from pre-trained models
    - Backward compatibility with original SimpleLSTM
    - Adaptive architecture based on input dimensionality
    - Maintains all original SimpleLSTM hyperparameters and training logic

Architecture Variants:
    1. OHLC-only mode: [B, 105, 4] → Same as original SimpleLSTM
    2. Feature-aware mode: [B, 105, 4+features] → Enhanced with feature fusion

Transfer Learning:
    - Compatible with FeatureAwareBiLSTMMaskedAutoencoder encoder weights
    - Supports both frozen and fine-tuning transfer strategies
    - Maintains original SimpleLSTM training pipeline

Usage:
    >>> from moola.models import EnhancedSimpleLSTMModel
    >>>
    >>> # OHLC-only mode (backward compatible)
    >>> model = EnhancedSimpleLSTMModel()
    >>> model.fit(X_ohlc, y)
    >>>
    >>> # Feature-aware mode with transfer learning
    >>> model = EnhancedSimpleLSTMModel()
    >>> model.fit(X_combined, y, pretrained_encoder_path="path/to/encoder.pt")
    >>>
    >>> # Automatic mode detection based on input dimensionality
    >>> model.fit(X, y)  # X.shape[-1] determines mode
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.model_selection import train_test_split

from ..utils.augmentation import mixup_criterion, mixup_cutmix
from ..utils.data_validation import DataValidator
from ..utils.early_stopping import EarlyStopping
from ..utils.focal_loss import FocalLoss
from ..utils.model_diagnostics import ModelDiagnostics
from ..utils.seeds import get_device, set_seed
from ..utils.temporal_augmentation import TemporalAugmentation
from ..utils.training_utils import TrainingSetup
from .base import BaseModel


def compute_pointer_regression_loss(
    outputs: dict,
    expansion_start: torch.Tensor,
    expansion_end: torch.Tensor,
) -> torch.Tensor:
    """Compute regression loss for pointer prediction.

    Args:
        outputs: Model outputs dict with 'pointers' key [B, 2]
        expansion_start: Target start indices [B]
        expansion_end: Target end indices [B]

    Returns:
        Smooth L1 loss for pointer regression
    """
    pointers = outputs['pointers']  # [B, 2], already scaled to [0, 104]
    targets = torch.stack([expansion_start, expansion_end], dim=1).float()  # [B, 2]
    return F.smooth_l1_loss(pointers, targets)


class EnhancedSimpleLSTMModel(BaseModel):
    """Enhanced Simple LSTM with dual input support for feature-aware transfer learning.

    Automatically detects input mode (OHLC-only vs OHLC+features) and adapts
    architecture accordingly while maintaining backward compatibility.
    """

    def __init__(
        self,
        seed: int = 1337,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_heads: int = 2,
        dropout: float = 0.1,
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
        time_warp_prob: float = 0.0,
        feature_fusion: str = "concat",  # For feature-aware mode
        predict_pointers: bool = False,  # Multi-task: predict expansion start/end
        loss_alpha: float = 0.5,  # Weight for classification loss
        loss_beta: float = 0.25,  # Weight for each pointer loss
        **kwargs,
    ):
        """Initialize Enhanced SimpleLSTM model.

        Args:
            seed: Random seed for reproducibility
            hidden_size: LSTM hidden dimension (default: 128, matches BiLSTM encoder)
            num_layers: Number of LSTM layers (default: 1)
            num_heads: Number of attention heads (default: 2)
            dropout: Dropout rate (default: 0.1)
            n_epochs: Number of training epochs (default: 60)
            batch_size: Training batch size (default: 512)
            learning_rate: Learning rate for optimizer (default: 5e-4)
            device: Device to train on ('cpu' or 'cuda')
            use_amp: Use automatic mixed precision (FP16) when device='cuda'
            num_workers: Number of DataLoader worker processes
            early_stopping_patience: Epochs to wait before stopping (default: 20)
            val_split: Validation split ratio (default: 0.15)
            mixup_alpha: Mixup interpolation strength (default: 0.4)
            cutmix_prob: Probability of applying cutmix vs mixup (default: 0.5)
            use_temporal_aug: Enable temporal augmentation
            jitter_prob: Probability of applying jitter (default: 0.5)
            scaling_prob: Probability of applying scaling (default: 0.3)
            time_warp_prob: Probability of applying time warping (default: 0.0)
            feature_fusion: Fusion strategy for feature-aware mode ('concat', 'add', 'gate')
            predict_pointers: Enable multi-task pointer prediction (expansion_start, expansion_end)
            loss_alpha: Weight for classification loss in multi-task mode (default: 0.5)
            loss_beta: Weight for each pointer loss in multi-task mode (default: 0.25)
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
        self.feature_fusion = feature_fusion
        self.predict_pointers = predict_pointers
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta

        set_seed(seed)

        # Initialize temporal augmentation pipeline
        if self.use_temporal_aug:
            self.temporal_aug = TemporalAugmentation(
                jitter_prob=jitter_prob,
                jitter_sigma=0.05,
                scaling_prob=scaling_prob,
                scaling_sigma=0.1,
                permutation_prob=0.0,
                time_warp_prob=time_warp_prob,
                time_warp_sigma=0.12,
                rotation_prob=0.0,
            )

        # Model will be built after seeing input dimension and num classes
        self.model = None
        self.n_classes = None
        self.input_dim = None
        self.ohlc_dim = None
        self.feature_dim = None
        self.is_feature_aware = False

    def _build_model(self, input_dim: int, n_classes: int) -> nn.Module:
        """Build Enhanced SimpleLSTM neural network architecture.

        Automatically detects mode and builds appropriate architecture:
        - OHLC-only mode: Same as original SimpleLSTM
        - Feature-aware mode: Enhanced with feature fusion

        Args:
            input_dim: Input feature dimension
            n_classes: Number of output classes

        Returns:
            PyTorch model
        """
        # Detect mode based on input dimension
        if input_dim == 4:
            # OHLC-only mode (backward compatible)
            self.ohlc_dim = 4
            self.feature_dim = 0
            self.is_feature_aware = False
            logger.info(f"Building OHLC-only SimpleLSTM (input_dim={input_dim})")
        elif input_dim == 11:
            # RelativeTransform mode (all features replace OHLC)
            self.ohlc_dim = 11
            self.feature_dim = 0
            self.is_feature_aware = False
            logger.info(f"Building RelativeTransform SimpleLSTM (input_dim={input_dim})")
        elif input_dim > 4:
            # Feature-aware mode (OHLC + additional features)
            self.ohlc_dim = 4
            self.feature_dim = input_dim - 4
            self.is_feature_aware = True
            logger.info(f"Building Feature-aware SimpleLSTM (ohlc_dim=4, feature_dim={self.feature_dim})")
        else:
            raise ValueError(f"Invalid input_dim: {input_dim}. Expected >= 4 for OHLC data.")

        class EnhancedSimpleLSTMNet(nn.Module):
            def __init__(
                self,
                ohlc_dim: int,
                feature_dim: int,
                hidden_size: int,
                num_layers: int,
                num_heads: int,
                n_classes: int,
                dropout: float,
                feature_fusion: str,
                predict_pointers: bool = False,
            ):
                super().__init__()
                self.ohlc_dim = ohlc_dim
                self.feature_dim = feature_dim
                self.hidden_size = hidden_size
                self.feature_fusion = feature_fusion
                self.predict_pointers = predict_pointers

                # Input processing based on mode
                if feature_dim == 0:
                    # OHLC-only mode
                    encoder_input_dim = ohlc_dim
                    self.ohlc_proj = None
                    self.feature_proj = None
                    self.gate_proj = None
                else:
                    # Feature-aware mode
                    if feature_fusion == "concat":
                        encoder_input_dim = ohlc_dim + feature_dim
                        self.ohlc_proj = None
                        self.feature_proj = None
                        self.gate_proj = None
                    elif feature_fusion == "add":
                        encoder_input_dim = max(ohlc_dim, feature_dim)
                        self.ohlc_proj = nn.Linear(ohlc_dim, encoder_input_dim)
                        self.feature_proj = nn.Linear(feature_dim, encoder_input_dim)
                        self.gate_proj = None
                    elif feature_fusion == "gate":
                        encoder_input_dim = max(ohlc_dim, feature_dim)
                        self.ohlc_proj = nn.Linear(ohlc_dim, encoder_input_dim)
                        self.feature_proj = nn.Linear(feature_dim, encoder_input_dim)
                        self.gate_proj = nn.Linear(encoder_input_dim * 2, encoder_input_dim)
                    else:
                        raise ValueError(f"Unknown feature_fusion: {feature_fusion}")

                # LSTM layer
                self.lstm = nn.LSTM(
                    encoder_input_dim,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=True,
                )

                # Multi-head self-attention
                self.attention = nn.MultiheadAttention(
                    hidden_size * 2, num_heads, dropout=dropout, batch_first=True
                )

                # Layer normalization
                self.ln = nn.LayerNorm(hidden_size * 2)

                # Classification head
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size * 2, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, n_classes),
                )

                # Pointer head (multi-task: predict expansion_start and expansion_end)
                if predict_pointers:
                    self.pointer_head = nn.Sequential(
                        nn.Linear(hidden_size * 2, 128),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(128, 2),  # 2 outputs: start, end
                    )

            def _fuse_inputs(self, x: torch.Tensor) -> torch.Tensor:
                """Fuse inputs based on mode and fusion strategy.

                Args:
                    x: Input tensor [batch, seq_len, input_dim]

                Returns:
                    Fused input [batch, seq_len, encoder_input_dim]
                """
                if self.feature_dim == 0:
                    # OHLC-only mode
                    return x

                # Feature-aware mode: split inputs
                ohlc = x[..., :self.ohlc_dim]
                features = x[..., self.ohlc_dim:]

                if self.feature_fusion == "concat":
                    # Simple concatenation
                    return torch.cat([ohlc, features], dim=-1)

                elif self.feature_fusion == "add":
                    # Project and add
                    ohlc_proj = self.ohlc_proj(ohlc)
                    feature_proj = self.feature_proj(features)
                    return ohlc_proj + feature_proj

                elif self.feature_fusion == "gate":
                    # Gated fusion
                    ohlc_proj = self.ohlc_proj(ohlc)
                    feature_proj = self.feature_proj(features)
                    combined = torch.cat([ohlc_proj, feature_proj], dim=-1)
                    gate = torch.sigmoid(self.gate_proj(combined))
                    return gate * ohlc_proj + (1 - gate) * feature_proj

            def forward(self, x: torch.Tensor) -> torch.Tensor | dict:
                """Forward pass with optional multi-task output.

                Args:
                    x: Input tensor [batch, seq_len, input_dim]
                       - OHLC-only: [B, 105, 4]
                       - Feature-aware: [B, 105, 4+feature_dim]

                Returns:
                    If predict_pointers=False:
                        Logits [batch, n_classes]
                    If predict_pointers=True:
                        Dictionary with keys:
                        - 'type_logits': [B, n_classes] class logits
                        - 'pointers': [B, 2] expansion [start, end] in range [0, 104]
                """
                # Fuse inputs
                x_fused = self._fuse_inputs(x)

                # LSTM processing
                lstm_out, _ = self.lstm(x_fused)  # [B, 105, 256]

                # Self-attention over sequence
                attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # [B, 105, 256]

                # Residual connection + layer norm
                x = self.ln(lstm_out + attn_out)  # [B, 105, 256]

                # Use last timestep for classification
                last_hidden = x[:, -1, :]  # [B, 256]

                # Classification
                logits = self.classifier(last_hidden)  # [B, n_classes]

                # Return early if single-task mode
                if not self.predict_pointers:
                    return logits

                # Multi-task: also predict pointers
                pointers = self.pointer_head(last_hidden)  # [B, 2]
                pointers = torch.sigmoid(pointers) * 104  # Scale to [0, 104]

                return {'type_logits': logits, 'pointers': pointers}

        model = EnhancedSimpleLSTMNet(
            ohlc_dim=self.ohlc_dim,
            feature_dim=self.feature_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            n_classes=n_classes,
            dropout=self.dropout_rate,
            feature_fusion=self.feature_fusion,
            predict_pointers=self.predict_pointers,
        )

        return model.to(self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        expansion_start: np.ndarray = None,
        expansion_end: np.ndarray = None,
        unfreeze_encoder_after: int = 0,
        pretrained_encoder_path: Path = None,
        freeze_encoder: bool = True,
    ) -> "EnhancedSimpleLSTMModel":
        """Train Enhanced SimpleLSTM model.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, D]
            y: Target labels of shape [N]
            expansion_start: Optional expansion start indices (unused)
            expansion_end: Optional expansion end indices (unused)
            unfreeze_encoder_after: Epoch to unfreeze encoder (0 = never unfreeze,
                                   >0 = unfreeze after N epochs, -1 = start unfrozen).
            pretrained_encoder_path: Optional path to pre-trained encoder checkpoint.
                                    Supports both original and feature-aware encoders.
            freeze_encoder: If True and pretrained_encoder_path is provided, freeze encoder
                           weights during initial training.

        Returns:
            Self for method chaining
        """
        set_seed(self.seed)

        # Prepare and validate data
        X, y_indices, self.label_to_idx, self.idx_to_label, self.n_classes = (
            DataValidator.prepare_training_data(X, y, expected_features=4)  # Min 4 for OHLC
        )

        N, T, F = X.shape
        self.input_dim = F

        logger.info(
            f"Enhanced SimpleLSTM training: {X.shape} samples, "
            f"mode={'feature-aware' if self.is_feature_aware else 'OHLC-only'}"
        )

        # Build model (this will detect mode)
        self.model = self._build_model(self.input_dim, self.n_classes)

        # Load pre-trained encoder if provided
        if pretrained_encoder_path is not None:
            logger.info(
                f"Loading pre-trained encoder from: {pretrained_encoder_path} "
                f"(freeze={freeze_encoder})"
            )
            self.load_pretrained_encoder(
                encoder_path=pretrained_encoder_path, freeze_encoder=freeze_encoder
            )

        # Log model diagnostics
        ModelDiagnostics.log_model_info(self.model, N)
        ModelDiagnostics.log_gpu_info(self.device, self.use_amp)

        # Split into train/val for early stopping
        if self.val_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_indices, test_size=self.val_split, random_state=self.seed, stratify=y_indices
            )
        else:
            X_train, y_train = X, y_indices
            X_val, y_val = None, None

        # Create training dataloader
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        train_dataloader = TrainingSetup.create_dataloader(
            X_train_tensor,
            y_train_tensor,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            device=self.device,
        )

        # Create validation dataloader if needed
        val_dataloader = None
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            val_dataloader = TrainingSetup.create_dataloader(
                X_val_tensor,
                y_val_tensor,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                device=self.device,
            )

        # Setup optimizer and loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        # Use Focal Loss WITH class weights for class imbalance
        class_weights = torch.tensor([1.0, 1.17], dtype=torch.float32, device=self.device)
        criterion = FocalLoss(gamma=2.0, alpha=class_weights, reduction="mean")

        # Setup mixed precision training
        scaler = TrainingSetup.setup_mixed_precision(self.use_amp, self.device)

        # Setup early stopping if validation data available
        early_stopping = None
        if val_dataloader is not None:
            early_stopping = EarlyStopping(
                patience=self.early_stopping_patience, mode="min", verbose=True
            )

        # Training loop (same as original SimpleLSTM)
        for epoch in range(self.n_epochs):
            # Unfreeze encoder if scheduled
            if unfreeze_encoder_after == -1 and epoch == 0:
                logger.info("Starting with UNFROZEN LSTM encoder (unfreeze_encoder_after=-1)")
                for param in self.model.lstm.parameters():
                    param.requires_grad = True
            elif unfreeze_encoder_after > 0 and epoch == unfreeze_encoder_after:
                logger.info(f"[TWO-PHASE] Unfreezing LSTM encoder at epoch {epoch + 1}")
                for param in self.model.lstm.parameters():
                    param.requires_grad = True

                # Reduce learning rate after unfreezing
                from ..config.training_config import MASKED_LSTM_UNFREEZE_LR_REDUCTION

                for param_group in optimizer.param_groups:
                    param_group["lr"] *= MASKED_LSTM_UNFREEZE_LR_REDUCTION
                logger.info(f"[TWO-PHASE] Reduced LR to {optimizer.param_groups[0]['lr']:.6f} "
                           f"(multiplier: {MASKED_LSTM_UNFREEZE_LR_REDUCTION})")

            # Training phase
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in train_dataloader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                # Apply temporal augmentation
                if self.use_temporal_aug:
                    batch_X = self.temporal_aug.apply_augmentation(batch_X)

                # Apply mixup/cutmix
                batch_X_aug, y_a, y_b, lam = mixup_cutmix(
                    batch_X,
                    batch_y,
                    mixup_alpha=self.mixup_alpha,
                    cutmix_prob=self.cutmix_prob,
                )

                optimizer.zero_grad()

                # Forward pass with mixed precision
                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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
                            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

                if (epoch + 1) % max(1, self.n_epochs // 10) == 0:
                    gpu_mem = (
                        f" GPU: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB"
                        if self.device.type == "cuda"
                        else ""
                    )
                    mode_str = "Feature-aware" if self.is_feature_aware else "OHLC-only"
                    logger.info(
                        f"Epoch [{epoch+1}/{self.n_epochs}] [{mode_str}] "
                        f"Train Loss: {avg_train_loss:.4f} Acc: {train_accuracy:.4f} | "
                        f"Val Loss: {avg_val_loss:.4f} Acc: {val_accuracy:.4f}{gpu_mem}"
                    )
            else:
                if (epoch + 1) % max(1, self.n_epochs // 10) == 0:
                    gpu_mem = (
                        f" GPU: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB"
                        if self.device.type == "cuda"
                        else ""
                    )
                    mode_str = "Feature-aware" if self.is_feature_aware else "OHLC-only"
                    logger.info(
                        f"Epoch [{epoch+1}/{self.n_epochs}] [{mode_str}] "
                        f"Loss: {avg_train_loss:.4f} Acc: {train_accuracy:.4f}{gpu_mem}"
                    )

        # Restore best model if early stopping was used
        if early_stopping is not None:
            early_stopping.load_best_model(self.model)

        self.is_fitted = True
        return self

    def predict(
        self, X: np.ndarray, expansion_start: np.ndarray = None, expansion_end: np.ndarray = None
    ) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Reshape input
        X = DataValidator.reshape_input(X, expected_features=4)  # Min 4 for OHLC

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            _, predicted = torch.max(logits, 1)

        # Convert indices back to original labels
        predicted_labels = np.array([self.idx_to_label[idx.item()] for idx in predicted])

        return predicted_labels

    def predict_proba(
        self, X: np.ndarray, expansion_start: np.ndarray = None, expansion_end: np.ndarray = None
    ) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Reshape input
        X = DataValidator.reshape_input(X, expected_features=4)

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def save(self, path: Path) -> None:
        """Save model to disk using PyTorch format."""
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "model_state_dict": self.model.state_dict() if self.model is not None else None,
            "label_to_idx": self.label_to_idx if hasattr(self, "label_to_idx") else None,
            "idx_to_label": self.idx_to_label if hasattr(self, "idx_to_label") else None,
            "n_classes": self.n_classes,
            "input_dim": self.input_dim,
            "ohlc_dim": self.ohlc_dim,
            "feature_dim": self.feature_dim,
            "is_feature_aware": self.is_feature_aware,
            "feature_fusion": self.feature_fusion,
            "hyperparams": {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
            },
        }

        torch.save(save_dict, path)

    def load(self, path: Path) -> "EnhancedSimpleLSTMModel":
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)

        # Restore metadata
        self.label_to_idx = checkpoint["label_to_idx"]
        self.idx_to_label = checkpoint["idx_to_label"]
        self.n_classes = checkpoint["n_classes"]
        self.input_dim = checkpoint["input_dim"]
        self.ohlc_dim = checkpoint.get("ohlc_dim", 4)
        self.feature_dim = checkpoint.get("feature_dim", 0)
        self.is_feature_aware = checkpoint.get("is_feature_aware", False)
        self.feature_fusion = checkpoint.get("feature_fusion", "concat")

        # Restore hyperparameters
        hyperparams = checkpoint["hyperparams"]
        self.hidden_size = hyperparams["hidden_size"]
        self.num_layers = hyperparams["num_layers"]
        self.num_heads = hyperparams["num_heads"]
        self.dropout_rate = hyperparams["dropout_rate"]

        # Rebuild and load model
        self.model = self._build_model(self.input_dim, self.n_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.is_fitted = True
        return self

    def load_pretrained_encoder(
        self, encoder_path: Path, freeze_encoder: bool = True
    ) -> "EnhancedSimpleLSTMModel":
        """Load pre-trained encoder weights with strict validation.

        Args:
            encoder_path: Path to pre-trained encoder checkpoint (.pt file)
            freeze_encoder: If True, freeze LSTM weights during initial training

        Returns:
            Self with pre-trained encoder loaded

        Raises:
            ValueError: If model not built yet
            AssertionError: If pretrained loading validation fails
        """
        if self.model is None:
            raise ValueError(
                "Model must be built first. Call fit() or _build_model() before loading encoder."
            )

        # Use strict pretrained loader
        from .pretrained_utils import load_pretrained_strict

        logger.info(f"Loading pretrained encoder with STRICT VALIDATION from: {encoder_path}")

        # Load with encoder-only validation (≥40% match, 0 shape mismatches)
        # With current architecture (num_layers=2), expect ~44.4% match
        # (encoder tensors only: 8/18, missing attention + classifier = expected)
        pretrained_stats = load_pretrained_strict(
            model=self.model,
            checkpoint_path=str(encoder_path),
            freeze_encoder=freeze_encoder,
            min_match_ratio=0.40,  # Encoder-only: 8/18 tensors = 44.4%
            allow_shape_mismatch=False,
        )

        # Store stats for inspection
        self.pretrained_stats = pretrained_stats

        logger.success(
            f"✓ Pretrained encoder loaded successfully\n"
            f"  Matched: {pretrained_stats['n_matched']} tensors ({pretrained_stats['match_ratio']:.1%})\n"
            f"  Missing: {pretrained_stats['n_missing']} tensors\n"
            f"  Frozen: {pretrained_stats['n_frozen']} parameters"
        )

        return self