"""Jade Architecture - Production BiLSTM with Multi-task Learning.

Jade is a production-ready BiLSTM architecture implementing the Stones non-negotiables
for robust multi-task learning with uncertainty-weighted loss.

Architecture:
    - BiLSTM(11→128×2, 2 layers) → global average pool → two heads
    - Pointer head: center(sigmoid), length(sigmoid)  
    - Type head: 3-way logits
    - Gradient clip 1.5–2.0
    - ReduceLROnPlateau
    - Early stop patience 20
    - Dropout: recurrent 0.6–0.7, dense 0.4–0.5, input 0.2–0.3
    - Uncertainty-weighted loss as DEFAULT (no manual λ)
    - Registry format: moola-lstm-m-v1.0 // codename: Jade

Key Features:
    - Uncertainty-weighted multi-task loss (default, no manual λ)
    - Center+length pointer encoding only
    - Huber δ≈0.08 for pointer regression
    - Proper dropout configuration per Stones guide
    - Gradient clipping 1.5-2.0
    - ReduceLROnPlateau scheduler
    - Early stopping with patience 20
    - Production-ready with comprehensive monitoring

Usage:
    >>> from moola.models import JadeModel
    >>>
    >>> # Standard usage with uncertainty-weighted loss (default)
    >>> model = JadeModel(predict_pointers=True)
    >>> model.fit(X, y, expansion_start=starts, expansion_end=ends)
    >>>
    >>> # Single-task mode
    >>> model = JadeModel(predict_pointers=False)
    >>> model.fit(X, y)
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from ..utils.augmentation import mixup_criterion, mixup_cutmix
from ..utils.data_validation import DataValidator
from ..utils.early_stopping import EarlyStopping
from ..utils.focal_loss import FocalLoss
from ..utils.model_diagnostics import ModelDiagnostics
from ..utils.seeds import get_device, set_seed
from ..utils.temporal_augmentation import TemporalAugmentation
from .base import BaseModel


class UncertaintyWeightedLoss(nn.Module):
    """Multi-task loss with learnable uncertainty weighting.

    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene
    Geometry and Semantics" (Kendall et al., CVPR 2018).

    Learns optimal task weights by modeling heteroscedastic uncertainty.
    This is the DEFAULT and ONLY loss weighting method for Jade architecture.

    For regression: (1/2σ²)L + log(σ)
    For classification: (1/σ²)L + log(σ)

    Attributes:
        log_var_ptr: Log variance for pointer regression task
        log_var_type: Log variance for type classification task
    """

    def __init__(self):
        super().__init__()
        # Initialize log variances to 0 (σ = 1.0, equal weighting)
        self.log_var_ptr = nn.Parameter(torch.zeros(1))
        self.log_var_type = nn.Parameter(torch.zeros(1))

    def forward(self, ptr_loss: torch.Tensor, type_loss: torch.Tensor) -> torch.Tensor:
        """Compute weighted multi-task loss.

        Args:
            ptr_loss: Pointer regression loss (scalar)
            type_loss: Type classification loss (scalar)

        Returns:
            Weighted total loss with uncertainty regularization
        """
        # Regression: (1/2σ²)L + log(σ)
        precision_ptr = torch.exp(-self.log_var_ptr)
        weighted_ptr = 0.5 * precision_ptr * ptr_loss + self.log_var_ptr

        # Classification: (1/σ²)L + log(σ)
        precision_type = torch.exp(-self.log_var_type)
        weighted_type = precision_type * type_loss + self.log_var_type

        return weighted_ptr + weighted_type

    def get_uncertainties(self) -> dict:
        """Return current σ values for monitoring.

        Returns:
            dict with sigma_ptr and sigma_type (higher = more uncertain)
        """
        return {
            "sigma_ptr": torch.exp(0.5 * self.log_var_ptr).item(),
            "sigma_type": torch.exp(0.5 * self.log_var_type).item(),
        }


def compute_pointer_regression_loss(
    outputs: dict,
    expansion_start: torch.Tensor,
    expansion_end: torch.Tensor,
) -> torch.Tensor:
    """Compute regression loss for pointer prediction using center-length encoding.

    Jade uses ONLY center-length encoding for better gradient flow.

    Args:
        outputs: Model outputs dict with 'pointers_cl' key [B, 2] = [center, length]
        expansion_start: Target start indices [B] in range [0, 104]
        expansion_end: Target end indices [B] in range [0, 104]

    Returns:
        Huber loss for pointer regression with δ≈0.08
    """
    from moola.data.pointer_transforms import start_end_to_center_length

    # Convert ground truth to center-length encoding
    center_target, length_target = start_end_to_center_length(
        expansion_start.float(), expansion_end.float(), seq_len=104
    )

    # Stack targets [B, 2]
    targets_cl = torch.stack([center_target, length_target], dim=1)

    # Get predictions [B, 2]
    preds_cl = outputs.get("pointers_cl", outputs["pointers"])

    # Huber loss with δ≈0.08 (0.08 * 105 ≈ 8 timesteps transition)
    center_loss = F.huber_loss(preds_cl[:, 0], targets_cl[:, 0], delta=0.08)
    length_loss = F.huber_loss(preds_cl[:, 1], targets_cl[:, 1], delta=0.08)

    # Weighted combination: center (1.0) > length (0.8)
    loss = center_loss + 0.8 * length_loss

    return loss


class JadeModel(BaseModel):
    """Jade Architecture - Production BiLSTM with Multi-task Learning.

    Implements Stones non-negotiables for robust multi-task learning.
    """

    # Model metadata for registry
    MODEL_ID = "moola-lstm-m-v1.0"
    CODENAME = "Jade"

    def __init__(
        self,
        seed: int = 1337,
        hidden_size: int = 128,
        num_layers: int = 2,  # Jade: 2 layers (Stones requirement)
        n_epochs: int = 60,
        batch_size: int = 29,
        learning_rate: float = 3e-4,
        max_grad_norm: float = 2.0,  # Jade: 2.0 (Stones: 1.5-2.0)
        device: str = "cpu",
        use_amp: bool = True,
        num_workers: int = 16,
        early_stopping_patience: int = 20,  # Jade: 20 (Stones requirement)
        val_split: float = 0.15,
        mixup_alpha: float = 0.4,
        cutmix_prob: float = 0.5,
        use_temporal_aug: bool = True,
        jitter_prob: float = 0.8,
        jitter_sigma: float = 0.03,
        magnitude_warp_prob: float = 0.5,
        magnitude_warp_sigma: float = 0.2,
        magnitude_warp_knots: int = 4,
        scaling_prob: float = 0.0,
        time_warp_prob: float = 0.0,
        predict_pointers: bool = False,  # Multi-task: predict expansion center/length
        use_latent_mixup: bool = True,
        latent_mixup_alpha: float = 0.4,
        latent_mixup_prob: float = 0.5,
        # Jade: ReduceLROnPlateau configuration (Stones requirement)
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 10,
        scheduler_threshold: float = 0.001,
        scheduler_cooldown: int = 0,
        scheduler_min_lr: float = 1e-6,
        save_checkpoints: bool = False,
        **kwargs,
    ):
        """Initialize Jade model with Stones non-negotiables.

        Args:
            seed: Random seed for reproducibility
            hidden_size: LSTM hidden dimension (default: 128)
            num_layers: Number of LSTM layers (default: 2, Stones requirement)
            n_epochs: Number of training epochs (default: 60)
            batch_size: Training batch size (default: 29)
            learning_rate: Learning rate for optimizer (default: 3e-4)
            max_grad_norm: Gradient clipping threshold (default: 2.0, Stones: 1.5-2.0)
            device: Device to train on ('cpu' or 'cuda')
            use_amp: Use automatic mixed precision (FP16) when device='cuda'
            num_workers: Number of DataLoader worker processes
            early_stopping_patience: Epochs to wait before stopping (default: 20,
                                    Stones requirement)
            val_split: Validation split ratio (default: 0.15)
            mixup_alpha: Mixup interpolation strength (default: 0.4)
            cutmix_prob: Probability of applying cutmix vs mixup (default: 0.5)
            use_temporal_aug: Enable temporal augmentation
            jitter_prob: Probability of applying jitter (default: 0.8)
            jitter_sigma: Jitter noise std (default: 0.03)
            magnitude_warp_prob: Probability of magnitude warping (default: 0.5)
            magnitude_warp_sigma: Magnitude warp std (default: 0.2)
            magnitude_warp_knots: Number of warp control points (default: 4)
            scaling_prob: [DEPRECATED] Use magnitude_warp_prob instead (default: 0.0)
            time_warp_prob: [DEPRECATED] Use magnitude_warp_prob instead (default: 0.0)
            predict_pointers: Enable multi-task pointer prediction (default: False)
            use_latent_mixup: Apply mixup in latent space after encoder (default: True)
            latent_mixup_alpha: Beta distribution parameter for mixup strength (default: 0.4)
            latent_mixup_prob: Probability of applying latent mixup (default: 0.5)
            scheduler_factor: Factor to reduce LR by when plateauing (default: 0.5)
            scheduler_patience: Epochs to wait before reducing LR (default: 10)
            scheduler_threshold: Minimum change to qualify as improvement (default: 0.001)
            scheduler_cooldown: Epochs to wait before resuming normal operation (default: 0)
            scheduler_min_lr: Minimum learning rate threshold (default: 1e-6)
            save_checkpoints: Save best model checkpoints during training (default: False)
            **kwargs: Additional parameters (ignored for Jade)
        """
        super().__init__(seed=seed)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
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
        self.jitter_sigma = jitter_sigma
        self.magnitude_warp_prob = magnitude_warp_prob
        self.magnitude_warp_sigma = magnitude_warp_sigma
        self.magnitude_warp_knots = magnitude_warp_knots
        self.scaling_prob = scaling_prob
        self.time_warp_prob = time_warp_prob
        self.predict_pointers = predict_pointers
        self.use_latent_mixup = use_latent_mixup
        self.latent_mixup_alpha = latent_mixup_alpha
        self.latent_mixup_prob = latent_mixup_prob
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_threshold = scheduler_threshold
        self.scheduler_cooldown = scheduler_cooldown
        self.scheduler_min_lr = scheduler_min_lr
        self.save_checkpoints = save_checkpoints

        set_seed(seed)

        # Initialize temporal augmentation pipeline
        if self.use_temporal_aug:
            self.temporal_aug = TemporalAugmentation(
                jitter_prob=jitter_prob,
                jitter_sigma=jitter_sigma,
                magnitude_warp_prob=magnitude_warp_prob,
                magnitude_warp_sigma=magnitude_warp_sigma,
                magnitude_warp_knots=magnitude_warp_knots,
                scaling_prob=scaling_prob,
                scaling_sigma=0.1,
                permutation_prob=0.0,
                time_warp_prob=time_warp_prob,
                time_warp_sigma=0.2,
                rotation_prob=0.0,
            )

        # Model will be built after seeing input dimension and num classes
        self.model = None
        self.n_classes = None
        self.input_dim = None

    def _build_model(self, input_dim: int, n_classes: int) -> nn.Module:
        """Build Jade neural network architecture.

        Args:
            input_dim: Input feature dimension (expected: 11 for RelativeTransform)
            n_classes: Number of output classes (expected: 3 for Jade)

        Returns:
            PyTorch model implementing Jade architecture
        """
        if input_dim != 11:
            logger.warning(
                f"Jade expects 11-dim input, got {input_dim}. Architecture may be suboptimal."
            )

        if n_classes != 3:
            logger.warning(
                f"Jade expects 3 classes, got {n_classes}. Architecture may be suboptimal."
            )

        class JadeNet(nn.Module):
            def __init__(
                self,
                input_dim: int,
                hidden_size: int,
                num_layers: int,
                n_classes: int,
                predict_pointers: bool = False,
            ):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.predict_pointers = predict_pointers

                # Jade: Input dropout 0.2-0.3 (Stones requirement)
                self.input_dropout = nn.Dropout(0.25)

                # Jade: BiLSTM(11→128×2, 2 layers) (Stones requirement)
                self.lstm = nn.LSTM(
                    input_dim,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=0.65 if num_layers > 1 else 0,  # Jade: recurrent 0.6-0.7
                    bidirectional=True,
                )

                # Jade: Global average pooling (Stones requirement)
                # Instead of using last timestep, use global average pooling
                self.global_pool = nn.AdaptiveAvgPool1d(1)

                # Jade: Dense dropout 0.4-0.5 (Stones requirement)
                self.dense_dropout = nn.Dropout(0.45)

                # Jade: Type head: 3-way logits (Stones requirement)
                self.type_head = nn.Sequential(
                    nn.Linear(hidden_size * 2, 64),
                    nn.ReLU(),
                    self.dense_dropout,
                    nn.Linear(64, n_classes),
                )

                # Jade: Pointer head: center(sigmoid), length(sigmoid) (Stones requirement)
                if predict_pointers:
                    self.pointer_head = nn.Sequential(
                        nn.Linear(hidden_size * 2, 64),
                        nn.ReLU(),
                        self.dense_dropout,
                        nn.Linear(64, 2),  # center, length
                    )

            def forward(self, x: torch.Tensor) -> torch.Tensor | dict:
                """Forward pass with Jade architecture.

                Args:
                    x: Input tensor [batch, seq_len, input_dim]

                Returns:
                    If predict_pointers=False: logits [batch, n_classes]
                    If predict_pointers=True: dict with type_logits and pointers_cl
                """
                # Jade: Input dropout
                x = self.input_dropout(x)

                # Jade: BiLSTM processing
                lstm_out, _ = self.lstm(x)  # [B, 105, 256]

                # Jade: Global average pooling instead of last timestep
                # Transpose for pooling: [B, 256, 105] -> [B, 256, 1] -> [B, 256]
                lstm_out_transposed = lstm_out.transpose(1, 2)  # [B, 256, 105]
                pooled = self.global_pool(lstm_out_transposed).squeeze(-1)  # [B, 256]

                # Jade: Dense dropout
                pooled = self.dense_dropout(pooled)

                # Jade: Type head
                type_logits = self.type_head(pooled)  # [B, n_classes]

                # Return early if single-task mode
                if not self.predict_pointers:
                    return type_logits

                # Jade: Pointer head with center+length encoding
                pointers_cl = self.pointer_head(pooled)  # [B, 2]
                pointers_cl = torch.sigmoid(pointers_cl)  # Both in [0, 1]

                return {
                    "type_logits": type_logits,
                    "pointers": pointers_cl,  # [center, length] in [0, 1]
                    "pointers_cl": pointers_cl,  # Explicit alias
                }

        model = JadeNet(
            input_dim=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            n_classes=n_classes,
            predict_pointers=self.predict_pointers,
        )

        return model.to(self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        expansion_start: np.ndarray | None = None,
        expansion_end: np.ndarray | None = None,
        unfreeze_encoder_after: int = 0,
        pretrained_encoder_path: Path | None = None,
        freeze_encoder: bool = True,
        monitor_gradients: bool = False,
        gradient_log_freq: int = 10,
    ) -> "JadeModel":
        """Train Jade model with uncertainty-weighted multi-task loss.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, D]
            y: Target labels of shape [N]
            expansion_start: Expansion start indices [N] (required if predict_pointers=True)
            expansion_end: Expansion end indices [N] (required if predict_pointers=True)
            unfreeze_encoder_after: Epoch to unfreeze encoder (0 = never unfreeze)
            pretrained_encoder_path: Optional path to pre-trained encoder checkpoint
            freeze_encoder: If True and pretrained_encoder_path provided, freeze encoder
            monitor_gradients: If True, monitor gradient statistics during training
            gradient_log_freq: Frequency (in epochs) to log gradient statistics

        Returns:
            Self for method chaining
        """
        set_seed(self.seed)

        # Validate multi-task configuration
        has_pointers = (expansion_start is not None) and (expansion_end is not None)

        if self.predict_pointers and not has_pointers:
            raise ValueError(
                "predict_pointers=True but pointer labels not provided. "
                "Please provide both expansion_start and expansion_end arrays."
            )

        if has_pointers and not self.predict_pointers:
            logger.warning(
                "Pointer labels provided but predict_pointers=False. "
                "Ignoring pointer labels. Set predict_pointers=True to enable multi-task learning."
            )
            has_pointers = False

        # Prepare and validate data
        X, y_indices, self.label_to_idx, self.idx_to_label, self.n_classes = (
            DataValidator.prepare_training_data(X, y, expected_features=4)
        )

        N, T, D = X.shape
        self.input_dim = D

        logger.info(
            f"Jade ({self.MODEL_ID} // {self.CODENAME}) training: {X.shape} samples, "
            f"multi_task={has_pointers}"
        )

        # Build model
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
            if has_pointers:
                (
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    ptr_start_train,
                    ptr_start_val,
                    ptr_end_train,
                    ptr_end_val,
                ) = train_test_split(
                    X,
                    y_indices,
                    expansion_start,
                    expansion_end,
                    test_size=self.val_split,
                    random_state=self.seed,
                    stratify=y_indices,
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y_indices,
                    test_size=self.val_split,
                    random_state=self.seed,
                    stratify=y_indices,
                )
                ptr_start_train = ptr_start_val = None
                ptr_end_train = ptr_end_val = None
        else:
            X_train, y_train = X, y_indices
            X_val, y_val = None, None
        if has_pointers:
            ptr_start_train, ptr_end_train = expansion_start, expansion_end
        else:
            ptr_start_train, ptr_end_train = None, None
        ptr_start_val, ptr_end_val = None, None

        # Create training dataloader
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)

        if has_pointers:
            ptr_start_train_tensor = torch.FloatTensor(ptr_start_train)
            ptr_end_train_tensor = torch.FloatTensor(ptr_end_train)
            train_dataset = TensorDataset(
                X_train_tensor, y_train_tensor, ptr_start_train_tensor, ptr_end_train_tensor
            )
        else:
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        num_workers = self.num_workers if self.device.type == "cuda" else 0
        train_dataloader = DataLoader(
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

            if has_pointers:
                ptr_start_val_tensor = torch.FloatTensor(ptr_start_val)
                ptr_end_val_tensor = torch.FloatTensor(ptr_end_val)
                val_dataset = TensorDataset(
                    X_val_tensor, y_val_tensor, ptr_start_val_tensor, ptr_end_val_tensor
                )
            else:
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.device.type == "cuda" else False,
            )

        # Setup optimizer
        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.model.lstm.parameters(),
                    "lr": self.learning_rate,
                    "weight_decay": 1e-3,
                },  # BiLSTM encoder
                {
                    "params": self.model.type_head.parameters(),
                    "lr": self.learning_rate,
                    "weight_decay": 1e-2,
                },  # Type head
            ]
            + (
                [
                    {
                        "params": self.model.pointer_head.parameters(),
                        "lr": self.learning_rate,
                        "weight_decay": 1e-2,
                    }
                ]  # Pointer head
                if self.predict_pointers
                else []
            )
        )

        # Use Focal Loss for class imbalance
        class_weights = torch.tensor([1.0, 1.17], dtype=torch.float32, device=self.device)
        criterion = FocalLoss(gamma=2.0, alpha=class_weights, reduction="mean")

        # Jade: Uncertainty-weighted loss (DEFAULT, no manual λ)
        uncertainty_loss = None
        if self.predict_pointers:
            uncertainty_loss = UncertaintyWeightedLoss().to(self.device)
            # Add uncertainty parameters to optimizer
            optimizer.add_param_group(
                {
                    "params": uncertainty_loss.parameters(),
                    "lr": self.learning_rate,
                    "weight_decay": 0.0,  # No weight decay for uncertainty parameters
                }
            )
            logger.info("Jade: Using uncertainty-weighted multi-task loss (DEFAULT)")

        # Jade: ReduceLROnPlateau scheduler (Stones requirement)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # Monitor validation loss
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            threshold=self.scheduler_threshold,
            threshold_mode="rel",
            cooldown=self.scheduler_cooldown,
            min_lr=self.scheduler_min_lr,
            verbose=True,
        )
        logger.info(
            f"Jade: ReduceLROnPlateau scheduler enabled (factor={self.scheduler_factor}, "
            f"patience={self.scheduler_patience})"
        )

        # Setup mixed precision training
        scaler = GradScaler() if self.use_amp else None

        # Setup early stopping
        early_stopping = None
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        if val_dataloader is not None:
            early_stopping = EarlyStopping(
                patience=self.early_stopping_patience, mode="min", verbose=True
            )

        # Training loop
        for epoch in range(self.n_epochs):
            # Training phase
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_data in train_dataloader:
                # Unpack batch
                if has_pointers:
                    batch_X, batch_y, batch_ptr_start, batch_ptr_end = batch_data
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    batch_ptr_start = batch_ptr_start.to(self.device, non_blocking=True)
                    batch_ptr_end = batch_ptr_end.to(self.device, non_blocking=True)
                else:
                    batch_X, batch_y = batch_data
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
                if self.use_amp and scaler is not None:
                    with autocast():
                        outputs = self.model(batch_X_aug)

                        if has_pointers:
                            # Multi-task loss with uncertainty weighting
                            type_logits = outputs["type_logits"]
                            class_loss = mixup_criterion(criterion, type_logits, y_a, y_b, lam)
                            pointer_loss = compute_pointer_regression_loss(
                                outputs, batch_ptr_start, batch_ptr_end
                            )
                            loss = uncertainty_loss(pointer_loss, class_loss)
                            logits = type_logits
                        else:
                            # Single-task classification loss
                            logits = outputs
                            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

                    scaler.scale(loss).backward()
                    # Jade: Gradient clipping 1.5-2.0 (Stones requirement)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm
                    )
                    if uncertainty_loss is not None:
                        torch.nn.utils.clip_grad_norm_(
                            uncertainty_loss.parameters(), max_norm=self.max_grad_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(batch_X_aug)

                    if has_pointers:
                        # Multi-task loss with uncertainty weighting
                        type_logits = outputs["type_logits"]
                        class_loss = mixup_criterion(criterion, type_logits, y_a, y_b, lam)
                        pointer_loss = compute_pointer_regression_loss(
                            outputs, batch_ptr_start, batch_ptr_end
                        )
                        loss = uncertainty_loss(pointer_loss, class_loss)
                        logits = type_logits
                    else:
                        # Single-task classification loss
                        logits = outputs
                        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

                    loss.backward()
                    # Jade: Gradient clipping 1.5-2.0 (Stones requirement)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm
                    )
                    if uncertainty_loss is not None:
                        torch.nn.utils.clip_grad_norm_(
                            uncertainty_loss.parameters(), max_norm=self.max_grad_norm
                        )
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
                        # Unpack batch
                        if has_pointers:
                            batch_X, batch_y, batch_ptr_start, batch_ptr_end = batch_data
                            batch_X = batch_X.to(self.device, non_blocking=True)
                            batch_y = batch_y.to(self.device, non_blocking=True)
                            batch_ptr_start = batch_ptr_start.to(self.device, non_blocking=True)
                            batch_ptr_end = batch_ptr_end.to(self.device, non_blocking=True)
                        else:
                            batch_X, batch_y = batch_data
                            batch_X = batch_X.to(self.device, non_blocking=True)
                            batch_y = batch_y.to(self.device, non_blocking=True)

                        if self.use_amp and scaler is not None:
                            with autocast():
                                outputs = self.model(batch_X)

                                if has_pointers:
                                    type_logits = outputs["type_logits"]
                                    class_loss = criterion(type_logits, batch_y)
                                    pointer_loss = compute_pointer_regression_loss(
                                        outputs, batch_ptr_start, batch_ptr_end
                                    )
                                    loss = uncertainty_loss(pointer_loss, class_loss)
                                    logits = type_logits
                                else:
                                    logits = outputs
                                    loss = criterion(logits, batch_y)
                        else:
                            outputs = self.model(batch_X)

                            if has_pointers:
                                type_logits = outputs["type_logits"]
                                class_loss = criterion(type_logits, batch_y)
                                pointer_loss = compute_pointer_regression_loss(
                                    outputs, batch_ptr_start, batch_ptr_end
                                )
                                loss = uncertainty_loss(pointer_loss, class_loss)
                                logits = type_logits
                            else:
                                logits = outputs
                                loss = criterion(logits, batch_y)

                        val_loss += loss.item()
                        _, predicted = torch.max(logits, 1)
                        val_correct += (predicted == batch_y).sum().item()
                        val_total += batch_y.size(0)

                avg_val_loss = val_loss / len(val_dataloader)
                val_accuracy = val_correct / val_total

                # Jade: Step ReduceLROnPlateau scheduler (Stones requirement)
                scheduler.step(avg_val_loss)

                # Enhanced early stopping tracking
                if avg_val_loss < best_val_loss:
                    improvement = best_val_loss - avg_val_loss
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    patience_counter = 0

                    logger.info(
                        f"  ✓ New best validation loss: {best_val_loss:.4f} "
                        f"(improved by {improvement:.4f})"
                    )

                    # Save best model checkpoint if enabled
                    if self.save_checkpoints:
                        from pathlib import Path

                        checkpoint_dir = Path("artifacts/models/supervised/checkpoints")
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        checkpoint_path = checkpoint_dir / "jade_best_checkpoint.pt"

                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "best_val_loss": best_val_loss,
                                "train_loss": avg_train_loss,
                                "val_accuracy": val_accuracy,
                                "model_id": self.MODEL_ID,
                                "codename": self.CODENAME,
                            },
                            checkpoint_path,
                        )
                        logger.info(f"  Saved checkpoint to {checkpoint_path}")
                else:
                    patience_counter += 1
                    logger.info(
                        f"  No improvement for {patience_counter}/"
                        f"{self.early_stopping_patience} epochs"
                    )

                # Check early stopping
                if early_stopping(avg_val_loss, self.model):
                    logger.info(
                        f"\nJade: Early stopping triggered at epoch {epoch + 1}\n"
                        f"  Best validation loss: {best_val_loss:.4f} (epoch {best_epoch + 1})\n"
                        f"  Final learning rate: {optimizer.param_groups[0]['lr']:.2e}"
                    )
                    break

                if (epoch + 1) % max(1, self.n_epochs // 10) == 0:
                    gpu_mem = (
                        f" GPU: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB"
                        if self.device.type == "cuda"
                        else ""
                    )
                    logger.info(
                        f"Epoch [{epoch+1}/{self.n_epochs}] [Jade] "
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
                    logger.info(
                        f"Epoch [{epoch+1}/{self.n_epochs}] [Jade] "
                        f"Loss: {avg_train_loss:.4f} Acc: {train_accuracy:.4f}{gpu_mem}"
                    )

        # Restore best model if early stopping was used
        if early_stopping is not None and self.model is not None:
            early_stopping.load_best_model(self.model)

        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        expansion_start: np.ndarray | None = None,
        expansion_end: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Reshape input
        X = DataValidator.reshape_input(X, expected_features=4)

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)

            # Handle both single-task and multi-task outputs
            if isinstance(outputs, dict):
                logits = outputs["type_logits"]
            else:
                logits = outputs

            _, predicted = torch.max(logits, 1)

        # Convert indices back to original labels
        predicted_labels = np.array([self.idx_to_label[idx.item()] for idx in predicted])

        return predicted_labels

    def predict_proba(
        self,
        X: np.ndarray,
        expansion_start: np.ndarray | None = None,
        expansion_end: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Reshape input
        X = DataValidator.reshape_input(X, expected_features=4)

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)

            # Handle both single-task and multi-task outputs
            if isinstance(outputs, dict):
                logits = outputs["type_logits"]
            else:
                logits = outputs

            probs = F.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def predict_with_pointers(self, X: np.ndarray) -> dict:
        """Predict class labels AND pointer start/end.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, D]

        Returns:
            Dictionary containing:
            {
                'labels': [N] predicted class labels (strings),
                'probabilities': [N, n_classes] class probabilities,
                'pointers': [N, 2] expansion [center, length] in range [0, 1]
            }

        Raises:
            ValueError: If model not trained with predict_pointers=True
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if not self.predict_pointers:
            raise ValueError(
                "Model not trained with pointer prediction. "
                "Set predict_pointers=True when initializing the model."
            )

        # Reshape input
        X = DataValidator.reshape_input(X, expected_features=4)

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)

            # Extract outputs
            type_logits = outputs["type_logits"]
            pointers = outputs["pointers"]  # [N, 2]

            # Classification predictions
            class_probs = F.softmax(type_logits, dim=1)
            _, class_preds = torch.max(type_logits, 1)

        # Convert to numpy and original labels
        predicted_labels = np.array([self.idx_to_label[idx.item()] for idx in class_preds])

        return {
            "labels": predicted_labels,
            "probabilities": class_probs.cpu().numpy(),
            "pointers": pointers.cpu().numpy(),
        }

    def save(self, path: Path) -> None:
        """Save Jade model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "model_state_dict": self.model.state_dict() if self.model is not None else None,
            "label_to_idx": self.label_to_idx if hasattr(self, "label_to_idx") else None,
            "idx_to_label": self.idx_to_label if hasattr(self, "idx_to_label") else None,
            "n_classes": self.n_classes,
            "input_dim": self.input_dim,
            "model_id": self.MODEL_ID,
            "codename": self.CODENAME,
            "hyperparams": {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "max_grad_norm": self.max_grad_norm,
                "early_stopping_patience": self.early_stopping_patience,
            },
        }

        torch.save(save_dict, path)

    def load(self, path: Path) -> "JadeModel":
        """Load Jade model from disk."""
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
        self.max_grad_norm = hyperparams["max_grad_norm"]
        self.early_stopping_patience = hyperparams["early_stopping_patience"]

        # Rebuild and load model
        self.model = self._build_model(self.input_dim, self.n_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.is_fitted = True
        return self

    def load_pretrained_encoder(
        self, encoder_path: Path, freeze_encoder: bool = True
    ) -> "JadeModel":
        """Load pre-trained encoder weights with strict validation.

        Args:
            encoder_path: Path to pre-trained encoder checkpoint (.pt file)
            freeze_encoder: If True, freeze LSTM weights during initial training

        Returns:
            Self with pre-trained encoder loaded
        """
        if self.model is None:
            raise ValueError(
                "Model must be built first. Call fit() or _build_model() before loading encoder."
            )

        # TODO: Implement pretrained encoder loading for Jade
        logger.warning(f"Jade: Pretrained encoder loading not yet implemented for {encoder_path}")
        logger.info("Jade: Using randomly initialized weights")

        return self
