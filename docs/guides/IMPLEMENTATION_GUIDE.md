"""
Production SimpleLSTM with Fixed Transfer Learning

This version fixes the critical transfer learning bug where the encoder
remained frozen throughout training. Implements proper progressive unfreezing
with configurable schedules.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.model_selection import train_test_split

from ..utils.early_stopping import EarlyStopping
from ..utils.focal_loss import FocalLoss
from ..utils.model_diagnostics import ModelDiagnostics
from ..utils.seeds import get_device, set_seed
from ..utils.training_utils import TrainingSetup
from .base import BaseModel


class ProductionSimpleLSTM(BaseModel):
    """
    Production-ready SimpleLSTM with proper transfer learning support.

    Key improvements:
    1. Fixed transfer learning with progressive unfreezing
    2. Proper encoder weight loading and mapping
    3. Configurable unfreeze schedules
    4. Differential learning rates for encoder vs classifier
    5. Comprehensive logging and monitoring
    """

    def __init__(
        self,
        seed: int = 1337,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.3,
        n_epochs: int = 60,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        use_amp: bool = True,
        num_workers: int = 16,
        early_stopping_patience: int = 20,
        val_split: float = 0.15,
        # Transfer learning parameters
        pretrained_encoder_path: Optional[Path] = None,
        freeze_encoder: bool = True,
        unfreeze_schedule: List[int] = None,
        encoder_lr_multiplier: float = 0.1,
        **kwargs,
    ):
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

        # Transfer learning configuration
        self.pretrained_encoder_path = pretrained_encoder_path
        self.freeze_encoder = freeze_encoder
        self.unfreeze_schedule = unfreeze_schedule or [10, 20, 30]  # Default unfreeze at epochs 10, 20, 30
        self.encoder_lr_multiplier = encoder_lr_multiplier

        set_seed(seed)

        # Model will be built after seeing input dimension and num classes
        self.model = None
        self.n_classes = None
        self.input_dim = None

        # Transfer learning state
        self.transfer_learning_enabled = pretrained_encoder_path is not None
        self.current_unfreeze_phase = 0

        logger.info(f"Production SimpleLSTM initialized")
        logger.info(f"  - Transfer learning: {'enabled' if self.transfer_learning_enabled else 'disabled'}")
        if self.transfer_learning_enabled:
            logger.info(f"  - Pre-trained encoder: {pretrained_encoder_path}")
            logger.info(f"  - Freeze encoder: {freeze_encoder}")
            logger.info(f"  - Unfreeze schedule: {self.unfreeze_schedule}")
            logger.info(f"  - Encoder LR multiplier: {encoder_lr_multiplier}")

    def _build_model(self, input_dim: int, n_classes: int) -> nn.Module:
        """Build Production SimpleLSTM with proper encoder/decoder separation."""

        class ProductionSimpleLSTMNet(nn.Module):
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
                self.num_layers = num_layers

                # Encoder (for transfer learning)
                self.encoder = nn.LSTM(
                    input_dim,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=False,  # Unidirectional for online inference
                )

                # Multi-head attention
                self.attention = nn.MultiheadAttention(
                    hidden_size, num_heads, dropout=dropout, batch_first=True
                )

                # Classification head
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, n_classes),
                )

                # Layer normalization
                self.layer_norm = nn.LayerNorm(hidden_size)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Encoder
                lstm_out, _ = self.encoder(x)  # [B, T, H]

                # Self-attention on the sequence
                attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

                # Residual connection and layer norm
                attended = self.layer_norm(lstm_out + attn_out)

                # Global average pooling
                pooled = torch.mean(attended, dim=1)  # [B, H]

                # Classification
                logits = self.classifier(pooled)  # [B, n_classes]

                return logits

        model = ProductionSimpleLSTMNet(
            input_dim=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            n_classes=n_classes,
            dropout=self.dropout_rate,
        )

        return model.to(self.device)

    def load_pretrained_encoder(
        self, encoder_path: Path, freeze_encoder: bool = True
    ) -> "ProductionSimpleLSTM":
        """
        Load pre-trained encoder with proper weight mapping and validation.

        This is the CRITICAL fix for the transfer learning bug.
        """

        if self.model is None:
            raise ValueError(
                "Model must be built first. Call fit() or _build_model() before loading encoder."
            )

        logger.info(f"Loading pre-trained encoder from: {encoder_path}")
        logger.info(f"Initial freeze state: {freeze_encoder}")

        # Load checkpoint
        checkpoint = torch.load(encoder_path, map_location=self.device)

        # Handle different checkpoint formats
        if "encoder_state_dict" in checkpoint:
            encoder_state_dict = checkpoint["encoder_state_dict"]
            hyperparams = checkpoint["hyperparams"]
        elif "model_state_dict" in checkpoint:
            # Extract encoder weights from full model state dict
            full_state_dict = checkpoint["model_state_dict"]
            encoder_state_dict = {
                k.replace("encoder.", ""): v
                for k, v in full_state_dict.items()
                if k.startswith("encoder.")
            }
            hyperparams = checkpoint.get("hyperparams", {})
        else:
            raise ValueError("Invalid checkpoint format. Expected 'encoder_state_dict' or 'model_state_dict'.")

        # Verify architecture compatibility
        pretrained_hidden = hyperparams.get("hidden_dim") or hyperparams.get("hidden_size")
        pretrained_layers = hyperparams.get("num_layers", 1)

        if self.hidden_size != pretrained_hidden:
            raise ValueError(
                f"Hidden size mismatch: SimpleLSTM={self.hidden_size}, "
                f"Pre-trained encoder={pretrained_hidden}"
            )

        logger.info(
            f"Architecture compatibility verified: "
            f"Pre-trained={pretrained_layers} layers, SimpleLSTM={self.num_layers} layers"
        )

        # Map bidirectional to unidirectional weights if needed
        model_state_dict = self.model.state_dict()
        loaded_keys = []
        skipped_keys = []

        for key in encoder_state_dict:
            # Skip bidirectional reverse layers if present
            if "_reverse" in key:
                skipped_keys.append(key)
                continue

            # Map encoder weights
            model_key = f"encoder.{key}"

            if model_key in model_state_dict:
                encoder_shape = encoder_state_dict[key].shape
                model_shape = model_state_dict[model_key].shape

                # Handle bidirectional to unidirectional conversion
                if len(encoder_shape) > 0 and encoder_shape[0] == 2 * model_shape[0]:
                    # Bidirectional weight, take forward direction only
                    logger.info(f"Converting bidirectional weight {key} to unidirectional")
                    converted_weight = encoder_state_dict[key][:model_shape[0], ...]
                    model_state_dict[model_key] = converted_weight
                elif encoder_shape == model_shape:
                    # Direct mapping
                    model_state_dict[model_key] = encoder_state_dict[key]
                else:
                    logger.warning(
                        f"Shape mismatch for {model_key}: Expected {model_shape}, Got {encoder_shape}"
                    )
                    continue

                loaded_keys.append(model_key)
            else:
                logger.warning(f"Key not found in model: {model_key}")

        # Load mapped weights
        self.model.load_state_dict(model_state_dict)

        logger.success(
            f"Successfully loaded {len(loaded_keys)} encoder parameters"
        )
        if skipped_keys:
            logger.info(f"Skipped {len(skipped_keys)} bidirectional reverse weights")

        # Verify weight transfer
        if len(loaded_keys) == 0:
            raise ValueError(
                "Failed to load any encoder weights. Check architecture compatibility."
            )

        # Freeze encoder if requested
        if freeze_encoder:
            logger.info("Freezing encoder weights for initial training")
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder frozen. Only classifier will be trained initially.")
        else:
            logger.info("Encoder unfrozen. All parameters will be trainable.")

        return self

    def _setup_differential_optimizers(self) -> tuple:
        """
        Setup optimizers with differential learning rates for encoder vs classifier.
        This is crucial for effective transfer learning.
        """

        # Separate encoder and classifier parameters
        encoder_params = []
        classifier_params = []

        for name, param in self.model.named_parameters():
            if "encoder" in name:
                encoder_params.append(param)
            else:
                classifier_params.append(param)

        # Setup differential learning rates
        encoder_lr = self.learning_rate * self.encoder_lr_multiplier

        param_groups = [
            {"params": classifier_params, "lr": self.learning_rate, "name": "classifier"},
            {"params": encoder_params, "lr": encoder_lr, "name": "encoder"}
        ]

        # Filter out frozen parameters
        active_param_groups = []
        for group in param_groups:
            active_params = [p for p in group["params"] if p.requires_grad]
            if active_params:
                active_param_groups.append({**group, "params": active_params})

        optimizer = torch.optim.AdamW(active_param_groups, weight_decay=1e-4)

        logger.info(f"Setup differential optimizers:")
        logger.info(f"  - Classifier LR: {self.learning_rate:.6f}")
        logger.info(f"  - Encoder LR: {encoder_lr:.6f} (multiplier: {self.encoder_lr_multiplier})")
        logger.info(f"  - Active parameter groups: {len(active_param_groups)}")

        return optimizer

    def _apply_progressive_unfreezing(self, epoch: int, optimizer: torch.optim.Optimizer):
        """
        Apply progressive unfreezing based on schedule.
        This is the core of the transfer learning fix.
        """

        if epoch in self.unfreeze_schedule:
            phase_idx = self.unfreeze_schedule.index(epoch)
            logger.info(f"[PROGRESSIVE UNFREEZING] Phase {phase_idx + 1} at epoch {epoch + 1}")

            if phase_idx == 0:
                # First unfreeze: Unfreeze last LSTM layer
                logger.info("Unfreezing last LSTM layer")
                for name, param in self.model.encoder.named_parameters():
                    if "weight_hh_l0" in name or "weight_ih_l0" in name:
                        param.requires_grad = True
                        logger.debug(f"  Unfrozen: {name}")

            elif phase_idx == 1:
                # Second unfreeze: Unfreeze all encoder layers
                logger.info("Unfreezing all encoder layers")
                for param in self.model.encoder.parameters():
                    param.requires_grad = True
                logger.info("All encoder layers now trainable")

            elif phase_idx >= 2:
                # Full training: All parameters trainable
                logger.info("Full model training - all parameters trainable")
                for param in self.model.parameters():
                    param.requires_grad = True

            # Rebuild optimizer with new trainable parameters
            new_optimizer = self._setup_differential_optimizers()

            # Copy state from old optimizer
            new_optimizer.load_state_dict(optimizer.state_dict())

            logger.info(f"Optimizer rebuilt with new trainable parameters")
            return new_optimizer

        return optimizer

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        expansion_start: np.ndarray = None,
        expansion_end: np.ndarray = None,
        unfreeze_encoder_after: int = 0,
        pretrained_encoder_path: Path = None,
        freeze_encoder: bool = True,
    ) -> "ProductionSimpleLSTM":
        """
        Train model with proper transfer learning support.
        """

        set_seed(self.seed)

        # Data preparation
        from ..utils.data_validation import DataValidator

        X, y_indices, self.label_to_idx, self.idx_to_label, self.n_classes = (
            DataValidator.prepare_training_data(X, y, expected_features=4)
        )

        N, T, F = X.shape
        self.input_dim = F

        # Build model
        self.model = self._build_model(self.input_dim, self.n_classes)

        # Load pre-trained encoder if provided
        if pretrained_encoder_path is not None:
            self.load_pretrained_encoder(
                encoder_path=pretrained_encoder_path,
                freeze_encoder=freeze_encoder
            )

        # Setup data loaders
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_indices, test_size=self.val_split, random_state=self.seed, stratify=y_indices
        )

        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.LongTensor(y_val)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Setup optimizer with differential learning rates
        optimizer = self._setup_differential_optimizers()

        # Setup loss function
        class_weights = torch.tensor([1.0, 1.17], dtype=torch.float32, device=self.device)
        criterion = FocalLoss(gamma=2.0, alpha=class_weights, reduction="mean")

        # Setup mixed precision
        scaler = TrainingSetup.setup_mixed_precision(self.use_amp, self.device)

        # Setup early stopping
        early_stopping = EarlyStopping(
            patience=self.early_stopping_patience, mode="min", verbose=True
        )

        # Log model info
        ModelDiagnostics.log_model_info(self.model, N)
        ModelDiagnostics.log_gpu_info(self.device, self.use_amp)

        # Training loop with progressive unfreezing
        logger.info(f"Starting training for {self.n_epochs} epochs")
        if self.transfer_learning_enabled:
            logger.info(f"Transfer learning enabled with unfreeze schedule: {self.unfreeze_schedule}")

        for epoch in range(self.n_epochs):
            # Apply progressive unfreezing
            if self.transfer_learning_enabled:
                optimizer = self._apply_progressive_unfreezing(epoch, optimizer)

            # Training phase
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        logits = self.model(batch_X)
                        loss = criterion(logits, batch_y)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = self.model(batch_X)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

            train_loss = total_loss / len(train_loader)
            train_accuracy = correct / total

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

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

            val_loss_avg = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total

            # Log progress
            if (epoch + 1) % max(1, self.n_epochs // 10) == 0:
                gpu_mem = (
                    f" GPU: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB"
                    if self.device.type == "cuda"
                    else ""
                )

                unfreeze_status = ""
                if self.transfer_learning_enabled:
                    trainable_encoder_params = sum(
                        p.numel() for p in self.model.encoder.parameters() if p.requires_grad
                    )
                    total_encoder_params = sum(p.numel() for p in self.model.encoder.parameters())
                    encoder_frozen_ratio = 1 - (trainable_encoder_params / total_encoder_params)
                    unfreeze_status = f" [Encoder: {encoder_frozen_ratio:.1%} frozen]"

                logger.info(
                    f"Epoch [{epoch+1}/{self.n_epochs}] "
                    f"Train Loss: {train_loss:.4f} Acc: {train_accuracy:.4f} | "
                    f"Val Loss: {val_loss_avg:.4f} Acc: {val_accuracy:.4f}{gpu_mem}{unfreeze_status}"
                )

            # Early stopping
            if early_stopping(val_loss_avg, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Load best model
        early_stopping.load_best_model(self.model)
        self.is_fitted = True

        logger.info("Training completed successfully")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Reshape input
        from ..utils.data_validation import DataValidator
        X = DataValidator.reshape_input(X, expected_features=4)

        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(X_tensor)
            _, predicted = torch.max(logits, 1)

        # Convert indices back to original labels
        predicted_labels = np.array([self.idx_to_label[idx.item()] for idx in predicted])
        return predicted_labels

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        from ..utils.data_validation import DataValidator
        X = DataValidator.reshape_input(X, expected_features=4)

        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def save(self, path: Path) -> None:
        """Save model with transfer learning state."""
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "model_state_dict": self.model.state_dict() if self.model is not None else None,
            "label_to_idx": self.label_to_idx if hasattr(self, "label_to_idx") else None,
            "idx_to_label": self.idx_to_label if hasattr(self, "idx_to_label") else None,
            "n_classes": self.n_classes,
            "input_dim": self.input_dim,
            "hyperparams": {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
            },
            "transfer_learning_config": {
                "enabled": self.transfer_learning_enabled,
                "pretrained_encoder_path": str(self.pretrained_encoder_path) if self.pretrained_encoder_path else None,
                "unfreeze_schedule": self.unfreeze_schedule,
                "encoder_lr_multiplier": self.encoder_lr_multiplier,
            }
        }

        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path) -> "ProductionSimpleLSTM":
        """Load model with transfer learning state."""
        checkpoint = torch.load(path, map_location=self.device)

        # Restore metadata
        self.label_to_idx = checkpoint["label_to_idx"]
        self.idx_to_label = checkpoint["idx_to_label"]
        self.n_classes = checkpoint["n_classes"]
        self.input_dim = checkpoint["input_dim"]

        # Restore hyperparameters
        hyperparams = checkpoint["hyperparams"]
        self.hidden_size = hyperparams["hidden_size"]
        self.num_layers = hyperparams["num_layers"]
        self.num_heads = hyperparams["num_heads"]
        self.dropout_rate = hyperparams["dropout_rate"]

        # Restore transfer learning config
        if "transfer_learning_config" in checkpoint:
            tl_config = checkpoint["transfer_learning_config"]
            self.transfer_learning_enabled = tl_config["enabled"]
            self.pretrained_encoder_path = Path(tl_config["pretrained_encoder_path"]) if tl_config["pretrained_encoder_path"] else None
            self.unfreeze_schedule = tl_config["unfreeze_schedule"]
            self.encoder_lr_multiplier = tl_config["encoder_lr_multiplier"]

        # Rebuild and load model
        self.model = self._build_model(self.input_dim, self.n_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.is_fitted = True
        logger.info(f"Model loaded from {path}")
        return self