"""Enhanced SimpleLSTM: Dual-input architecture combining raw OHLC and engineered features.

Designed as an optimized replacement for RWKV-TS with dramatically fewer parameters (~17K vs 409K).
Uses bidirectional LSTM for temporal processing and engineered feature encoder for pattern recognition.

Enhanced Architecture:
- OHLC Encoder: 4 -> BiLSTM(128) -> 256
- Feature Encoder: 25-30 -> Linear(32) -> 32
- Feature Fusion: Concatenate(256 + 32) -> 288
- Multi-head attention (2 heads, efficient parameter usage)
- Classification head: 288 -> 32 -> num_classes

Target: ~17K parameters for 98-sample dataset (95.8% reduction from 409K)
Parameter-to-sample ratio: 174:1 (optimized for small dataset training)

Training Configuration (Phase 2 + Pre-training):
- Mixup + CutMix augmentation (alpha=0.4, increased for better generalization)
- Temporal augmentation: jitter (50%), scaling (30%), time_warp (30%, sigma=0.12)
- Early stopping with patience=20 (optimized for Phase 2 with augmentation)
- Pre-trained encoder support: BiLSTM masked autoencoder transfer learning
- Learning rate: 5e-4
- Max epochs: 60
- AdamW optimizer with weight_decay=1e-4
- Validation split: 15%

Key Features:
- Dual-input processing (raw OHLC temporal + engineered features)
- Efficient parameter usage for small datasets
- Bidirectional temporal processing with attention
- Backward compatibility with existing pipeline
- Enhanced performance through feature fusion
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.model_selection import train_test_split

from ..features.small_dataset_features import create_small_dataset_feature_engineer
from ..utils.augmentation import mixup_criterion, mixup_cutmix
from ..utils.data_validation import DataValidator
from ..utils.early_stopping import EarlyStopping
from ..utils.focal_loss import FocalLoss
from ..utils.model_diagnostics import ModelDiagnostics
from ..utils.seeds import get_device, set_seed
from ..utils.temporal_augmentation import TemporalAugmentation
from ..utils.training_utils import TrainingSetup
from .base import BaseModel


class EnhancedDataset(torch.utils.data.Dataset):
    """Custom dataset for dual-input processing with OHLC and engineered features."""

    def __init__(self, ohlc_data, labels, engineered_data=None):
        self.ohlc_data = torch.FloatTensor(ohlc_data)
        self.labels = torch.LongTensor(labels)
        self.engineered_data = (
            torch.FloatTensor(engineered_data) if engineered_data is not None else None
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.engineered_data is not None:
            return self.ohlc_data[idx], self.engineered_data[idx], self.labels[idx]
        else:
            return self.ohlc_data[idx], self.labels[idx]


def enhanced_collate_fn(batch):
    """Custom collate function to handle dual inputs."""
    if len(batch[0]) == 3:  # Dual-input mode
        ohlc_batch, features_batch, labels_batch = zip(*batch)
        return (torch.stack(ohlc_batch), torch.stack(features_batch), torch.tensor(labels_batch))
    else:  # OHLC-only mode
        ohlc_batch, labels_batch = zip(*batch)
        return (torch.stack(ohlc_batch), torch.tensor(labels_batch))


class SimpleLSTMModel(BaseModel):
    """Enhanced SimpleLSTM with dual-input architecture combining raw OHLC and engineered features.

    Optimized for small datasets with efficient parameter usage and feature fusion.
    """

    def __init__(
        self,
        seed: int = 1337,
        hidden_size: int = 32,  # Reduced from 128 to achieve target parameter count (~17K)
        num_layers: int = 1,
        num_heads: int = 2,  # Optimized for efficiency while maintaining performance
        dropout: float = 0.5,  # Stones compliance: dense dropout 0.4-0.5
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
        time_warp_prob: float = 0.0,  # DISABLED for small dataset fine-tuning
        # Enhanced dual-input architecture parameters
        feature_encoder_hidden: int = 16,  # Hidden dimension for feature encoder (reduced)
        classifier_hidden: int = 16,  # Hidden dimension for classification head (reduced)
        use_engineered_features: bool = True,  # Enable dual-input processing
        max_engineered_features: int = 25,  # Maximum engineered features to use
        **kwargs,
    ):
        """Initialize Enhanced SimpleLSTM model with dual-input architecture.

        Args:
            seed: Random seed for reproducibility
            hidden_size: LSTM hidden dimension for OHLC encoder (default: 128, matches BiLSTM encoder)
            num_layers: Number of LSTM layers (default: 1)
            num_heads: Number of attention heads (default: 2, optimized for efficiency)
            dropout: Dropout rate (default: 0.1 for small dataset)
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
            time_warp_prob: Probability of applying time warping (default: 0.0)
            feature_encoder_hidden: Hidden dimension for engineered feature encoder (default: 32)
            classifier_hidden: Hidden dimension for classification head (default: 32)
            use_engineered_features: Enable dual-input processing with engineered features (default: True)
            max_engineered_features: Maximum engineered features to extract (default: 25)
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

        # Enhanced dual-input architecture parameters
        self.feature_encoder_hidden = feature_encoder_hidden
        self.classifier_hidden = classifier_hidden
        self.use_engineered_features = use_engineered_features
        self.max_engineered_features = max_engineered_features

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
                time_warp_sigma=0.12,  # 12% - Goldilocks zone for masked reconstruction
                rotation_prob=0.0,  # Disabled (OHLC order matters)
            )

        # Initialize feature engineer for dual-input processing
        if self.use_engineered_features:
            self.feature_engineer = create_small_dataset_feature_engineer(
                max_features_per_category=max_engineered_features
                // 5,  # Distribute across 5 categories
                robust_scaling=True,
                feature_selection=True,
            )
            logger.info(f"Enhanced SimpleLSTM initialized with engineered features support")
            logger.info(f"  - OHLC encoder: BiLSTM({hidden_size}) -> {hidden_size * 2}")
            logger.info(
                f"  - Feature encoder: up to {max_engineered_features} features -> {feature_encoder_hidden}"
            )
            logger.info(f"  - Target parameters: ~17K (95.8% reduction from 409K)")
        else:
            self.feature_engineer = None
            logger.info(f"Enhanced SimpleLSTM initialized with OHLC-only mode")
            logger.info(f"  - OHLC encoder: BiLSTM({hidden_size}) -> {hidden_size * 2}")
            logger.info(f"  - Target parameters: ~17K (95.9% reduction from 409K)")

        # Model will be built after seeing input dimension and num classes
        self.model = None
        self.n_classes = None
        self.input_dim = None

    def _build_model(self, input_dim: int, n_classes: int) -> nn.Module:
        """Build Enhanced SimpleLSTM neural network architecture with dual-input processing.

        Args:
            input_dim: Input feature dimension (4 for OHLC, 11 for RelativeTransform)
            n_classes: Number of output classes

        Returns:
            PyTorch model with enhanced dual-input architecture
        """

        class EnhancedSimpleLSTMNet(nn.Module):
            def __init__(
                self,
                input_dim: int,
                hidden_size: int,
                num_layers: int,
                num_heads: int,
                n_classes: int,
                dropout: float,
                feature_encoder_hidden: int,
                classifier_hidden: int,
                use_engineered_features: bool,
            ):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_size = hidden_size
                self.use_engineered_features = use_engineered_features

                # 1. OHLC Encoder: Bidirectional LSTM for temporal processing
                self.ohlc_encoder = nn.LSTM(
                    input_dim,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=True,
                )

                # 2. Feature Encoder: Process engineered features if enabled
                if use_engineered_features:
                    # Adaptive input dimension (determined after feature extraction)
                    self.feature_encoder = nn.Sequential(
                        nn.Linear(1, feature_encoder_hidden),  # Will be resized dynamically
                        nn.ReLU(),
                        nn.Dropout(dropout * 0.5),  # Less dropout for feature encoder
                    )
                else:
                    self.feature_encoder = None

                # 3. Lightweight attention mechanism (operate on last timestep only)
                # Use attention over compressed temporal representation instead of full sequence
                attention_dim = min(64, hidden_size)  # Limit attention dimension for efficiency
                self.temporal_projection = nn.Linear(hidden_size * 2, attention_dim)
                self.attention = nn.MultiheadAttention(
                    attention_dim, num_heads, dropout=dropout, batch_first=True
                )

                # 4. Layer normalization
                self.layer_norm = nn.LayerNorm(attention_dim)

                # 5. Enhanced Classification Head with fusion support
                fusion_dim = attention_dim + (
                    feature_encoder_hidden if use_engineered_features else 0
                )
                self.classifier = nn.Sequential(
                    nn.Linear(fusion_dim, classifier_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(classifier_hidden, n_classes),
                )

            def _resize_feature_encoder(self, feature_dim: int) -> None:
                """Dynamically resize feature encoder based on extracted feature dimension."""
                if (
                    self.use_engineered_features
                    and hasattr(self, "feature_encoder")
                    and self.feature_encoder is not None
                ):
                    # Replace the first linear layer with correct dimensions
                    old_layer = self.feature_encoder[0]
                    # Create new layer on same device as old layer
                    new_layer = nn.Linear(feature_dim, old_layer.out_features).to(
                        old_layer.weight.device
                    )

                    # Initialize with similar weights/biases if dimensions match
                    if old_layer.in_features == feature_dim:
                        new_layer.weight.data.copy_(old_layer.weight.data)
                        if old_layer.bias is not None:
                            new_layer.bias.data.copy_(old_layer.bias.data)
                    else:
                        # Kaiming initialization for new dimensions
                        nn.init.kaiming_uniform_(new_layer.weight, nonlinearity="relu")
                        if new_layer.bias is not None:
                            nn.init.constant_(new_layer.bias, 0)

                    self.feature_encoder[0] = new_layer

            def forward(
                self, x_ohlc: torch.Tensor, x_features: torch.Tensor = None
            ) -> torch.Tensor:
                """Forward pass with dual-input support.

                Args:
                    x_ohlc: OHLC input tensor [batch, seq_len, input_dim]
                            Expected: [B, 105, 4] for OHLC data
                    x_features: Engineered features [batch, feature_dim] (optional)

                Returns:
                    Logits [batch, n_classes]
                """
                # 1. OHLC Temporal Processing
                lstm_out, _ = self.ohlc_encoder(x_ohlc)  # [B, 105, hidden_size * 2]

                # 2. Extract last timestep for attention
                last_timestep = lstm_out[:, -1:, :]  # [B, 1, hidden_size * 2]

                # 3. Project to attention dimension
                projected = self.temporal_projection(last_timestep)  # [B, 1, attention_dim]

                # 4. Self-attention (single timestep self-attention for efficiency)
                attn_out, _ = self.attention(
                    projected, projected, projected
                )  # [B, 1, attention_dim]

                # 5. Residual connection + layer normalization
                temporal_repr = self.layer_norm(projected + attn_out)  # [B, 1, attention_dim]

                # 6. Extract final temporal representation
                temporal_features = temporal_repr[:, 0, :]  # [B, attention_dim]

                # 6. Feature Processing (if enabled and available)
                if self.use_engineered_features and x_features is not None:
                    # Dynamically resize feature encoder if needed
                    self._resize_feature_encoder(x_features.shape[-1])

                    # Process engineered features
                    encoded_features = self.feature_encoder(
                        x_features
                    )  # [B, feature_encoder_hidden]

                    # 7. Feature Fusion: Concatenate temporal and feature representations
                    fused_features = torch.cat(
                        [temporal_features, encoded_features], dim=-1
                    )  # [B, fusion_dim]
                else:
                    # OHLC-only mode
                    fused_features = temporal_features

                # 8. Classification
                logits = self.classifier(fused_features)  # [B, n_classes]

                return logits

        model = EnhancedSimpleLSTMNet(
            input_dim=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            n_classes=n_classes,
            dropout=self.dropout_rate,
            feature_encoder_hidden=self.feature_encoder_hidden,
            classifier_hidden=self.classifier_hidden,
            use_engineered_features=self.use_engineered_features,
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
    ) -> "SimpleLSTMModel":
        """Train Enhanced SimpleLSTM model with dual-input support.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, D] (OHLC data)
            y: Target labels of shape [N]
            expansion_start: Optional expansion start indices (used for engineered feature extraction)
            expansion_end: Optional expansion end indices (used for engineered feature extraction)
            unfreeze_encoder_after: Epoch to unfreeze encoder (0 = never unfreeze,
                                   >0 = unfreeze after N epochs, -1 = start unfrozen).
                                   Used with pre-trained encoder. Recommended: 10 for two-phase training.
            pretrained_encoder_path: Optional path to pre-trained encoder checkpoint (.pt file).
                                    If provided, loads encoder weights before training.
            freeze_encoder: If True and pretrained_encoder_path is provided, freeze encoder
                           weights during initial training (default: True).

        Returns:
            Self for method chaining
        """
        set_seed(self.seed)

        # Prepare and validate OHLC data
        X, y_indices, self.label_to_idx, self.idx_to_label, self.n_classes = (
            DataValidator.prepare_training_data(X, y, expected_features=4)
        )

        N, T, F = X.shape
        self.input_dim = F

        # Extract engineered features if enabled
        X_engineered = None
        if self.use_engineered_features and self.feature_engineer is not None:
            logger.info("Extracting engineered features for dual-input processing...")
            X_engineered = self.feature_engineer.extract_features(
                X, expansion_start, expansion_end, y=y_indices
            )
            logger.info(f"Extracted engineered features: {X_engineered.shape}")
            logger.info(
                f"Enhanced architecture will process OHLC: {X.shape} + Features: {X_engineered.shape}"
            )

        logger.info(
            "Using Focal Loss (gamma=2.0) WITH class weights [1.0, 1.17] for class imbalance"
        )

        # Build enhanced model
        self.model = self._build_model(self.input_dim, self.n_classes)

        # Store engineered features for training
        self._X_engineered = X_engineered

        # Load pre-trained encoder if provided (BEFORE training starts)
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

        # Split into train/val for early stopping (for both OHLC and engineered features)
        if self.val_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_indices, test_size=self.val_split, random_state=self.seed, stratify=y_indices
            )

            # Split engineered features if available
            if X_engineered is not None:
                X_engineered_train, X_engineered_val = train_test_split(
                    X_engineered,
                    test_size=self.val_split,
                    random_state=self.seed,
                    stratify=y_indices,
                )
            else:
                X_engineered_train, X_engineered_val = None, None
        else:
            X_train, y_train = X, y_indices
            X_val, y_val = None, None
            X_engineered_train = X_engineered
            X_engineered_val = None

        # Create training dataset and dataloader
        train_dataset = EnhancedDataset(X_train, y_train, X_engineered_train)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=enhanced_collate_fn,
        )

        # Create validation dataloader if needed
        val_dataloader = None
        if X_val is not None:
            val_dataset = EnhancedDataset(X_val, y_val, X_engineered_val)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,  # Validation doesn't need workers
                pin_memory=self.device.type == "cuda",
                collate_fn=enhanced_collate_fn,
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

        # Training loop
        for epoch in range(self.n_epochs):
            # Unfreeze encoder if scheduled (for pre-trained models)
            # unfreeze_encoder_after semantics:
            #   -1 = start unfrozen (immediately at epoch 0)
            #    0 = never unfreeze (keep frozen entire training)
            #   >0 = unfreeze after N epochs (two-phase training)
            if unfreeze_encoder_after == -1 and epoch == 0:
                logger.info("Starting with UNFROZEN LSTM encoder (unfreeze_encoder_after=-1)")
                for param in self.model.ohlc_encoder.parameters():
                    param.requires_grad = True
            elif unfreeze_encoder_after > 0 and epoch == unfreeze_encoder_after:
                logger.info(f"[TWO-PHASE] Unfreezing LSTM encoder at epoch {epoch + 1}")
                for param in self.model.ohlc_encoder.parameters():
                    param.requires_grad = True

                # Reduce learning rate after unfreezing (from config)
                from ..config.training_config import MASKED_LSTM_UNFREEZE_LR_REDUCTION

                for param_group in optimizer.param_groups:
                    param_group["lr"] *= MASKED_LSTM_UNFREEZE_LR_REDUCTION
                logger.info(
                    f"[TWO-PHASE] Reduced LR to {optimizer.param_groups[0]['lr']:.6f} "
                    f"(multiplier: {MASKED_LSTM_UNFREEZE_LR_REDUCTION})"
                )

            # Training phase
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_data in train_dataloader:
                # Handle both dual-input and OHLC-only modes
                if len(batch_data) == 3:  # Dual-input mode
                    batch_X_ohlc, batch_X_features, batch_y = batch_data
                    batch_X_ohlc = batch_X_ohlc.to(self.device, non_blocking=True)
                    batch_X_features = batch_X_features.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                else:  # OHLC-only mode
                    batch_X_ohlc, batch_y = batch_data
                    batch_X_ohlc = batch_X_ohlc.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    batch_X_features = None

                # Apply temporal augmentation to OHLC data only
                if self.use_temporal_aug:
                    batch_X_ohlc = self.temporal_aug.apply_augmentation(batch_X_ohlc)

                # Apply mixup/cutmix to OHLC data
                batch_X_ohlc_aug, y_a, y_b, lam = mixup_cutmix(
                    batch_X_ohlc,
                    batch_y,
                    mixup_alpha=self.mixup_alpha,
                    cutmix_prob=self.cutmix_prob,
                )

                # Apply same mixup to engineered features if available and non-empty
                if batch_X_features is not None and batch_X_features.shape[-1] > 0:
                    batch_X_features_aug, _, _, _ = mixup_cutmix(
                        batch_X_features,
                        batch_y,
                        mixup_alpha=self.mixup_alpha,
                        cutmix_prob=self.cutmix_prob,
                    )
                else:
                    batch_X_features_aug = batch_X_features  # Use as-is if empty

                optimizer.zero_grad()

                # Forward pass with mixed precision
                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        logits = self.model(batch_X_ohlc_aug, batch_X_features_aug)
                        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = self.model(batch_X_ohlc_aug, batch_X_features_aug)
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
                    for batch_data in val_dataloader:
                        # Handle both dual-input and OHLC-only modes
                        if len(batch_data) == 3:  # Dual-input mode
                            batch_X_ohlc, batch_X_features, batch_y = batch_data
                            batch_X_ohlc = batch_X_ohlc.to(self.device, non_blocking=True)
                            batch_X_features = batch_X_features.to(self.device, non_blocking=True)
                            batch_y = batch_y.to(self.device, non_blocking=True)
                        else:  # OHLC-only mode
                            batch_X_ohlc, batch_y = batch_data
                            batch_X_ohlc = batch_X_ohlc.to(self.device, non_blocking=True)
                            batch_y = batch_y.to(self.device, non_blocking=True)
                            batch_X_features = None

                        if self.use_amp:
                            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                                logits = self.model(batch_X_ohlc, batch_X_features)
                                loss = criterion(logits, batch_y)
                        else:
                            logits = self.model(batch_X_ohlc, batch_X_features)
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
                    logger.info(
                        f"Epoch [{epoch+1}/{self.n_epochs}] Train Loss: {avg_train_loss:.4f} Acc: {train_accuracy:.4f} | "
                        f"Val Loss: {avg_val_loss:.4f} Acc: {val_accuracy:.4f}{gpu_mem}"
                    )
            else:
                if (epoch + 1) % max(1, self.n_epochs // 10) == 0:
                    gpu_mem = (
                        f" GPU: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB"
                        if self.device.type == "cuda"
                        else ""
                    )
                    logger.info(
                        f"Epoch [{epoch+1}/{self.n_epochs}] Loss: {avg_train_loss:.4f} Acc: {train_accuracy:.4f}{gpu_mem}"
                    )

        # Restore best model if early stopping was used
        if early_stopping is not None:
            early_stopping.load_best_model(self.model)

        self.is_fitted = True
        return self

    def predict(
        self, X: np.ndarray, expansion_start: np.ndarray = None, expansion_end: np.ndarray = None
    ) -> np.ndarray:
        """Predict class labels with dual-input support.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, D] (OHLC data)
            expansion_start: Optional expansion start indices (used for engineered feature extraction)
            expansion_end: Optional expansion end indices (used for engineered feature extraction)

        Returns:
            Predicted labels of shape [N]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Reshape OHLC input
        X = DataValidator.reshape_input(X, expected_features=4)

        # Extract engineered features if enabled
        X_engineered = None
        if self.use_engineered_features and self.feature_engineer is not None:
            X_engineered = self.feature_engineer.extract_features(X, expansion_start, expansion_end)

        # Prepare tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        X_engineered_tensor = (
            torch.FloatTensor(X_engineered).to(self.device) if X_engineered is not None else None
        )

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor, X_engineered_tensor)
            _, predicted = torch.max(logits, 1)

        # Convert indices back to original labels
        predicted_labels = np.array([self.idx_to_label[idx.item()] for idx in predicted])

        return predicted_labels

    def predict_proba(
        self, X: np.ndarray, expansion_start: np.ndarray = None, expansion_end: np.ndarray = None
    ) -> np.ndarray:
        """Predict class probabilities with dual-input support.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, D] (OHLC data)
            expansion_start: Optional expansion start indices (used for engineered feature extraction)
            expansion_end: Optional expansion end indices (used for engineered feature extraction)

        Returns:
            Class probabilities of shape [N, C]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Reshape OHLC input
        X = DataValidator.reshape_input(X, expected_features=4)

        # Extract engineered features if enabled
        X_engineered = None
        if self.use_engineered_features and self.feature_engineer is not None:
            X_engineered = self.feature_engineer.extract_features(X, expansion_start, expansion_end)

        # Prepare tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        X_engineered_tensor = (
            torch.FloatTensor(X_engineered).to(self.device) if X_engineered is not None else None
        )

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor, X_engineered_tensor)
            probs = F.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def save(self, path: Path) -> None:
        """Save model to disk using PyTorch format.

        Args:
            path: Path to save model file (.pt extension)
        """
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
                # Enhanced dual-input architecture parameters
                "feature_encoder_hidden": self.feature_encoder_hidden,
                "classifier_hidden": self.classifier_hidden,
                "use_engineered_features": self.use_engineered_features,
                "max_engineered_features": self.max_engineered_features,
            },
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
        self.label_to_idx = checkpoint["label_to_idx"]
        self.idx_to_label = checkpoint["idx_to_label"]
        self.n_classes = checkpoint["n_classes"]
        self.input_dim = checkpoint["input_dim"]

        # Restore hyperparameters with backward compatibility
        hyperparams = checkpoint["hyperparams"]
        self.hidden_size = hyperparams["hidden_size"]
        self.num_layers = hyperparams["num_layers"]
        self.num_heads = hyperparams["num_heads"]
        self.dropout_rate = hyperparams["dropout_rate"]

        # Enhanced dual-input architecture parameters (with defaults for backward compatibility)
        self.feature_encoder_hidden = hyperparams.get("feature_encoder_hidden", 32)
        self.classifier_hidden = hyperparams.get("classifier_hidden", 32)
        self.use_engineered_features = hyperparams.get("use_engineered_features", False)
        self.max_engineered_features = hyperparams.get("max_engineered_features", 25)

        # Rebuild and load model
        self.model = self._build_model(self.input_dim, self.n_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.is_fitted = True
        return self

    def load_pretrained_encoder(
        self, encoder_path: Path, freeze_encoder: bool = True
    ) -> "SimpleLSTMModel":
        """Load pre-trained bidirectional LSTM encoder weights.

        Handles layer count mismatch: If pre-trained encoder has more layers,
        only load the first layer. Both encoders must be bidirectional.

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

        logger.info(f"Loading pre-trained encoder from: {encoder_path}")

        # Load checkpoint
        checkpoint = torch.load(encoder_path, map_location=self.device)
        encoder_state_dict = checkpoint["encoder_state_dict"]
        hyperparams = checkpoint["hyperparams"]

        # Verify architecture compatibility
        pretrained_hidden = hyperparams["hidden_dim"]
        pretrained_layers = hyperparams.get("num_layers", 1)

        if self.hidden_size != pretrained_hidden:
            raise ValueError(
                f"Hidden size mismatch: SimpleLSTM={self.hidden_size}, "
                f"Pre-trained encoder={pretrained_hidden}"
            )

        logger.info(
            f"Architecture: Pre-trained={pretrained_layers} layers, "
            f"SimpleLSTM={self.num_layers} layers, hidden_dim={pretrained_hidden}"
        )

        # Handle layer count mismatch: only load layer 0 (first layer)
        if pretrained_layers > self.num_layers:
            logger.warning(
                f"Pre-trained encoder has {pretrained_layers} layers but Enhanced SimpleLSTM has "
                f"{self.num_layers} layer(s). Loading only the first layer weights."
            )

        # Map bidirectional LSTM weights (load only matching layers)
        model_state_dict = self.model.state_dict()
        loaded_keys = []
        skipped_keys = []

        for key in encoder_state_dict:
            # Only load layer 0 weights if there's a layer count mismatch
            # Key format: weight_ih_l0, weight_hh_l0, weight_ih_l0_reverse, etc.
            # For multi-layer: weight_ih_l1, weight_ih_l1_reverse, etc.
            if pretrained_layers > self.num_layers:
                # Check if this is a layer 1+ weight (skip it)
                if "_l1" in key or "_l2" in key or "_l3" in key:
                    skipped_keys.append(key)
                    continue

            # Map encoder keys to Enhanced SimpleLSTM's OHLC encoder keys
            # Encoder keys: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0, *_reverse
            # Enhanced SimpleLSTM keys: ohlc_encoder.weight_ih_l0, ohlc_encoder.weight_hh_l0, etc.
            model_key = f"ohlc_encoder.{key}"

            if model_key in model_state_dict:
                # Verify shapes match
                encoder_shape = encoder_state_dict[key].shape
                model_shape = model_state_dict[model_key].shape

                if encoder_shape == model_shape:
                    model_state_dict[model_key] = encoder_state_dict[key]
                    loaded_keys.append(model_key)
                else:
                    logger.warning(
                        f"Shape mismatch for {model_key}: Expected {model_shape}, Got {encoder_shape}"
                    )
            else:
                logger.warning(f"Key not found in model: {model_key}")

        # Load mapped weights into model
        self.model.load_state_dict(model_state_dict)

        logger.success(
            f"Loaded {len(loaded_keys)} parameter tensors from pre-trained encoder "
            f"(skipped {len(skipped_keys)} higher-layer tensors)"
        )
        for key in loaded_keys:
            logger.debug(f"  ✓ {key}")
        if skipped_keys:
            logger.info(f"Skipped {len(skipped_keys)} layer 1+ weights (layer count mismatch)")

        # Verify weight transfer actually happened
        if len(loaded_keys) == 0:
            raise ValueError(
                "Failed to load any weights from pre-trained encoder. "
                "Check architecture compatibility."
            )

        # Freeze encoder if requested
        if freeze_encoder:
            logger.info("Freezing OHLC encoder weights")
            for param in self.model.ohlc_encoder.parameters():
                param.requires_grad = False
            logger.info(
                "OHLC encoder frozen. Only feature encoder and classifier will be trained initially."
            )
        else:
            logger.info("OHLC encoder unfrozen. All parameters trainable.")

        return self
