"""CNN → Transformer hybrid model for hierarchical feature extraction with multi-task learning.

This model combines local pattern detection via CNNs with global context modeling
via Transformers. Designed for time series classification with optional pointer prediction.

Architecture:
- Multi-scale CNN blocks (Conv1d with kernels {3, 5, 9} - final kernel increased to capture longer trends)
- Causal padding for temporal consistency
- Transformer encoder with relative positional encoding
- 3 layers × 4 heads

Channels: 3× [64, 128, 128] with dropout 0.25 (balanced regularization)

Multi-Task Learning (Phase 3):
- Shared backbone: CNN + Transformer feature extractor
- Task-specific heads:
  1. Classification head: Predict pattern type (currently: consolidation/retracement; dynamically adapts to num_classes)
  2. Pointer start head: Identify expansion start within inner window [30:75]
  3. Pointer end head: Identify expansion end within inner window [30:75]
- Balanced loss weighting: alpha=0.5 (class), beta=0.25 (each pointer)

Training Enhancements (Phase 2 augmentation improvements):
- Mixup + CutMix augmentation (alpha=0.4, increased for better generalization)
- Temporal augmentation: jitter (50%), scaling (30%), time_warp (30%)
- Early stopping with patience=20 (optimized for Phase 2 with augmentation)
- Learning rate: 5e-4 (reverted from 1e-3 - higher LR caused gradient explosion)
- Dropout: 0.25 (reverted from 0.1 - essential regularization for small dataset)
- Max epochs: 60 (increased from 10)
- AdamW optimizer with weight_decay=1e-4
- Validation split: 15% (~20 samples for stable early stopping)
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from ..utils.augmentation import mixup_criterion, mixup_cutmix
from ..utils.early_stopping import EarlyStopping
from ..utils.focal_loss import FocalLoss
from ..utils.losses import compute_multitask_loss
from ..utils.seeds import get_device, set_seed
from ..utils.temporal_augmentation import TemporalAugmentation
from .base import BaseModel


class CausalConv1d(nn.Module):
    """Conv1d with causal padding to preserve temporal order."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = 0.2):
        super().__init__()
        self.padding = (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with causal padding.

        Args:
            x: Input [batch, channels, seq_len]

        Returns:
            Output [batch, out_channels, seq_len]
        """
        # Left-pad to maintain causality
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        return self.dropout(x)


class CNNBlock(nn.Module):
    """Multi-scale CNN block with residual connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: list[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.kernels = kernels or [3, 5, 7]
        self.out_channels = out_channels

        # Distribute output channels evenly across kernels
        channels_per_kernel = out_channels // len(self.kernels)
        remainder = out_channels % len(self.kernels)

        # Multi-scale convolutions with proper channel distribution
        self.convs = nn.ModuleList()
        for i, k in enumerate(self.kernels):
            # Add remainder to first conv to ensure total matches out_channels
            ch = channels_per_kernel + (1 if i < remainder else 0)
            self.convs.append(CausalConv1d(in_channels, ch, k, dropout))

        # Batch normalization and activation
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

        # Residual connection (if dimensions match)
        self.residual = None
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale convolutions.

        Args:
            x: Input [batch, in_channels, seq_len]

        Returns:
            Output [batch, out_channels, seq_len]
        """
        # Multi-scale convolutions
        conv_outs = [conv(x) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1)

        # Batch norm and activation
        out = self.bn(out)
        out = self.activation(out)

        # Residual connection
        if self.residual is not None:
            x = self.residual(x)

        return out + x


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for Transformers.

    Uses learnable relative position embeddings instead of absolute positions.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Learnable relative position embeddings
        self.rel_pos_emb = nn.Parameter(torch.randn(2 * max_len - 1, d_model))

    def forward(self, seq_len: int) -> torch.Tensor:
        """Generate relative positional encodings.

        Args:
            seq_len: Sequence length

        Returns:
            Relative position encodings [seq_len, seq_len, d_model]
        """
        # Simple relative positions: i - j for all pairs (i, j)
        positions = torch.arange(seq_len, device=self.rel_pos_emb.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]
        rel_pos = rel_pos + self.max_len - 1  # Shift to positive indices

        # Clamp to valid range
        rel_pos = torch.clamp(rel_pos, 0, 2 * self.max_len - 2)

        return self.rel_pos_emb[rel_pos]  # [seq_len, seq_len, d_model]


class WindowAwarePositionalEncoding(nn.Module):
    """Boost attention for inner prediction window [30:75]."""

    def __init__(self, window_size: int = 105):
        super().__init__()
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply window-aware positional weighting.

        Args:
            x: Input tensor [B, seq_len, d_model]

        Returns:
            Weighted tensor with inner window [30:75] boosted by 1.5x
        """
        pos = torch.arange(self.window_size, device=x.device)

        # Create attention boost
        weights = torch.ones_like(pos, dtype=torch.float32)
        weights[30:75] = 1.5  # 50% boost for prediction region

        return x * weights[None, :, None]


class CnnTransformerModel(BaseModel):
    """CNN → Transformer hybrid for time series classification.

    Combines local pattern detection (CNN) with global context modeling (Transformer).
    """

    def __init__(
        self,
        seed: int = 1337,
        cnn_channels: list[int] = None,
        cnn_kernels: list[int] = None,
        transformer_layers: int = 3,
        transformer_heads: int = 4,
        dropout: float = 0.25,
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
        predict_pointers: bool = False,
        loss_alpha: float = 0.5,
        loss_beta: float = 0.25,
        use_temporal_aug: bool = True,
        jitter_prob: float = 0.5,
        scaling_prob: float = 0.3,
        time_warp_prob: float = 0.3,
        **kwargs,
    ):
        """Initialize CNN→Transformer model with optional multi-task pointer prediction.

        Args:
            seed: Random seed for reproducibility
            cnn_channels: CNN channel sizes (default: [64, 128, 128])
            cnn_kernels: CNN kernel sizes (default: [3, 5, 9] - final kernel increased)
            transformer_layers: Number of Transformer encoder layers
            transformer_heads: Number of attention heads
            dropout: Dropout rate (0.25 balanced regularization for small dataset)
            n_epochs: Number of training epochs (increased to 60 with early stopping)
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer (5e-4 stable for small dataset)
            device: Device to train on ('cpu' or 'cuda')
            use_amp: Use automatic mixed precision (FP16) when device='cuda'
            num_workers: Number of DataLoader worker processes
            early_stopping_patience: Epochs to wait before stopping (default: 20, Phase 2 optimized)
            val_split: Validation split ratio for early stopping (default: 0.15, ~20 samples)
            mixup_alpha: Mixup interpolation strength (default: 0.4, Phase 2 augmentation)
            cutmix_prob: Probability of applying cutmix vs mixup (default: 0.5)
            predict_pointers: Enable multi-task pointer prediction (default: False)
            loss_alpha: Weight for classification loss in multi-task mode (default: 0.5)
            loss_beta: Weight for each pointer loss in multi-task mode (default: 0.25)
            use_temporal_aug: Enable temporal augmentation (jitter, scaling, time_warp)
            jitter_prob: Probability of applying jitter (default: 0.5)
            scaling_prob: Probability of applying scaling (default: 0.3)
            time_warp_prob: Probability of applying time warping (default: 0.3)
            **kwargs: Additional parameters
        """
        super().__init__(seed=seed)
        self.cnn_channels = cnn_channels or [64, 128, 128]
        self.cnn_kernels = cnn_kernels or [3, 5, 9]
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
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
        self.predict_pointers = predict_pointers
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
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
        """Build the CNN→Transformer neural network architecture.

        Args:
            input_dim: Input feature dimension
            n_classes: Number of output classes

        Returns:
            PyTorch model
        """
        class CnnTransformerNet(nn.Module):
            def __init__(
                self,
                input_dim: int,
                cnn_channels: list[int],
                cnn_kernels: list[int],
                transformer_layers: int,
                transformer_heads: int,
                n_classes: int,
                dropout: float,
                predict_pointers: bool = False,
            ):
                super().__init__()
                self.input_dim = input_dim
                self.cnn_channels = cnn_channels
                self.d_model = cnn_channels[-1]  # Final CNN output becomes Transformer input
                self.predict_pointers = predict_pointers

                # CNN blocks (shared backbone)
                in_channels = [input_dim] + cnn_channels[:-1]
                self.cnn_blocks = nn.ModuleList([
                    CNNBlock(in_ch, out_ch, cnn_kernels, dropout)
                    for in_ch, out_ch in zip(in_channels, cnn_channels)
                ])

                # Transformer encoder (shared backbone)
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

                # Window-aware positional weighting
                self.window_pos_weight = WindowAwarePositionalEncoding(window_size=105)

                # Classification head (always present)
                self.output_norm = nn.LayerNorm(self.d_model)
                self.classifier = nn.Linear(self.d_model, n_classes)
                self.dropout = nn.Dropout(dropout)

                # Pointer prediction heads (optional)
                if predict_pointers:
                    self.pointer_start_head = nn.Linear(self.d_model, 1)
                    self.pointer_end_head = nn.Linear(self.d_model, 1)

            def _create_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
                """Prevent buffers [0:30] and [75:105] from attending to prediction region.

                Creates an additive attention mask for the transformer.
                - Buffers [0:30] and [75:105] can attend to each other but NOT to [30:75]
                - Prediction region [30:75] can attend to itself

                Args:
                    seq_len: Sequence length (should be 105)
                    device: Device to place mask on

                Returns:
                    Additive mask [seq_len, seq_len] where 0.0 = allow, -inf = block
                """
                # Start with all positions allowed (0.0)
                mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)

                # Block left buffer [0:30] from attending to prediction region [30:75]
                mask[0:30, 30:75] = float('-inf')

                # Block right buffer [75:105] from attending to prediction region [30:75]
                mask[75:105, 30:75] = float('-inf')

                return mask

            def forward(self, x: torch.Tensor) -> torch.Tensor | dict:
                """Forward pass with optional multi-task output.

                Args:
                    x: Input tensor [batch, seq_len, input_dim]
                       Expected shape: [B, 105, 4] for OHLC data

                Returns:
                    If predict_pointers=False:
                        Classification logits [batch, n_classes]
                    If predict_pointers=True:
                        Dictionary with keys:
                        - 'classification': [B, 3] class logits
                        - 'start': [B, 45] pointer start logits for inner window
                        - 'end': [B, 45] pointer end logits for inner window
                """
                # Transpose to [batch, input_dim, seq_len] for Conv1d
                x = x.transpose(1, 2)

                # Apply CNN blocks (shared backbone)
                for cnn_block in self.cnn_blocks:
                    x = cnn_block(x)

                # Transpose back to [batch, seq_len, d_model] for Transformer
                x = x.transpose(1, 2)

                # Apply window-aware positional weighting (before Transformer)
                x = self.window_pos_weight(x)

                # Add relative positional encoding
                # Simplified approach: use mean of relative position embeddings as additive bias
                # Full implementation would require modifying attention mechanism
                B, T, C = x.shape
                rel_pos_encoding = self.rel_pos_enc(T)  # [T, T, C]
                # Average over relative positions for each position
                pos_bias = rel_pos_encoding.mean(dim=1)  # [T, C]
                x = x + pos_bias.unsqueeze(0)  # Add positional bias [1, T, C]

                # Create attention mask: block buffers from attending to prediction region
                attention_mask = self._create_attention_mask(T, x.device)  # [105, 105]

                # Apply Transformer encoder with attention masking
                x = self.transformer(x, mask=attention_mask)  # [B, 105, d_model]

                # DEBUG: Verify masking is active (log once per model instance)
                if not hasattr(self, '_mask_verified'):
                    blocked_positions = (attention_mask == float('-inf')).sum().item()
                    print(f"[MASK] Attention mask applied | shape={attention_mask.shape} | blocked={blocked_positions}/11025")
                    self._mask_verified = True

                # TASK 1: Classification head
                # Region-specific pooling (only pool bars [30:75])
                pooled = x[:, 30:75, :].mean(dim=1)  # [B, d_model]
                pooled = self.output_norm(pooled)
                pooled = self.dropout(pooled)
                class_logits = self.classifier(pooled)  # [B, n_classes]

                # Return early if single-task mode
                if not self.predict_pointers:
                    return class_logits

                # TASK 2 & 3: Pointer prediction heads
                # Extract inner window [30:75] for pointer prediction
                inner_features = x[:, 30:75, :]  # [B, 45, d_model]

                # Pointer start logits (per-timestep prediction)
                start_logits = self.pointer_start_head(inner_features).squeeze(-1)  # [B, 45]

                # Pointer end logits (per-timestep prediction)
                end_logits = self.pointer_end_head(inner_features).squeeze(-1)  # [B, 45]

                return {
                    'classification': class_logits,
                    'start': start_logits,
                    'end': end_logits
                }

                # TODO: Add evidential deep learning layer here
                # - Replace final linear layer with evidential output
                # - Predict Dirichlet distribution parameters (alpha)
                # - Add evidential loss function (MSE + KL divergence)

                # TODO: Add conformal prediction layer here
                # - Compute calibrated prediction sets
                # - Track non-conformity scores during training
                # - Provide coverage guarantees for predictions

        model = CnnTransformerNet(
            input_dim=input_dim,
            cnn_channels=self.cnn_channels,
            cnn_kernels=self.cnn_kernels,
            transformer_layers=self.transformer_layers,
            transformer_heads=self.transformer_heads,
            n_classes=n_classes,
            dropout=self.dropout_rate,
            predict_pointers=self.predict_pointers,
        )

        return model.to(self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        expansion_start: np.ndarray | None = None,
        expansion_end: np.ndarray | None = None,
    ) -> "CnnTransformerModel":
        """Train CNN→Transformer model with optional multi-task pointer prediction.

        Args:
            X: Feature matrix of shape [N, D] or [N, T, D]
                For OHLC data: [N, 105, 4] or flattened [N, 420]
            y: Target labels of shape [N]
                Classification labels (e.g., 'consolidation', 'retracement'; model adapts to available classes)
            expansion_start: Optional expansion start indices of shape [N] (unused for single-task, can be used for multi-task pointer prediction)
            expansion_end: Optional expansion end indices of shape [N] (unused for single-task, can be used for multi-task pointer prediction)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If predict_pointers=True but pointer labels not provided
            ValueError: If pointer labels provided but predict_pointers=False (warning only)
        """
        set_seed(self.seed)

        # Map expansion parameters to pointer parameters for internal use
        pointer_starts = expansion_start
        pointer_ends = expansion_end

        # Validate multi-task mode configuration
        has_pointers = (pointer_starts is not None) and (pointer_ends is not None)

        if has_pointers and not self.predict_pointers:
            print("[WARNING] Pointer labels provided but predict_pointers=False. Ignoring pointer labels.")
            print("          Set predict_pointers=True to enable multi-task learning.")
            has_pointers = False

        if not has_pointers and self.predict_pointers:
            raise ValueError(
                "predict_pointers=True but pointer labels not provided. "
                "Please provide both pointer_starts and pointer_ends arrays."
            )

        # Handle input shape
        if X.ndim == 2:
            N, D = X.shape
            # Reshape to [N, T, F] - try to infer T and F
            if D % 4 == 0:
                T = D // 4
                X = X.reshape(N, T, 4)
            else:
                X = X.reshape(N, 1, D)

        N, T, F = X.shape
        self.input_dim = F

        # Validate and convert pointer indices if provided
        if has_pointers:
            assert pointer_starts.shape == (N,), f"pointer_starts shape mismatch: expected ({N},), got {pointer_starts.shape}"
            assert pointer_ends.shape == (N,), f"pointer_ends shape mismatch: expected ({N},), got {pointer_ends.shape}"

            # Convert from absolute window positions [30:75] to relative inner window positions [0:44]
            # The training data provides absolute positions in the 105-bar window
            # We need positions relative to the inner window [30:75]
            pointer_starts = np.clip(pointer_starts - 30, 0, 44)
            pointer_ends = np.clip(pointer_ends - 30, 0, 44)

            assert np.all((pointer_starts >= 0) & (pointer_starts < 45)), f"pointer_starts must be in range [0, 44], got {pointer_starts}"
            assert np.all((pointer_ends >= 0) & (pointer_ends < 45)), f"pointer_ends must be in range [0, 44], got {pointer_ends}"
            print(f"[MULTI-TASK] Training with pointer prediction enabled")
            print(f"[MULTI-TASK] Converted expansion indices from absolute to relative positions")
            print(f"[MULTI-TASK] Loss weights: alpha={self.loss_alpha}, beta={self.loss_beta}")

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

        # Load pre-trained encoder if path was provided
        use_pretrained = False
        if hasattr(self, '_pretrained_encoder_path'):
            print(f"[SSL] Loading pre-trained encoder from {self._pretrained_encoder_path}")
            self.load_pretrained_encoder(self._pretrained_encoder_path)
            use_pretrained = True

            # Freeze encoder initially
            from ..config.training_config import CNNTR_FREEZE_EPOCHS, CNNTR_GRADUAL_UNFREEZE
            if CNNTR_GRADUAL_UNFREEZE:
                self.freeze_encoder()
                print(f"[SSL] Encoder will be frozen for first {CNNTR_FREEZE_EPOCHS} epochs")
                print(f"[SSL] Gradual unfreezing enabled: stages at epochs {CNNTR_FREEZE_EPOCHS}, {CNNTR_FREEZE_EPOCHS + 10}, {CNNTR_FREEZE_EPOCHS + 20}")

        # GPU diagnostic logging
        if self.device.type == "cuda":
            print(f"[GPU] Training on: {torch.cuda.get_device_name(0)}")
            print(f"[GPU] Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"[GPU] Mixed precision (FP16): {self.use_amp}")

        # Convert labels to indices
        y_indices = np.array([self.label_to_idx[label] for label in y])

        # Split into train/val for early stopping if val_split > 0
        if self.val_split > 0:
            if has_pointers:
                X_train, X_val, y_train, y_val, ptr_start_train, ptr_start_val, ptr_end_train, ptr_end_val = train_test_split(
                    X, y_indices, pointer_starts, pointer_ends,
                    test_size=self.val_split, random_state=self.seed, stratify=y_indices
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y_indices, test_size=self.val_split, random_state=self.seed, stratify=y_indices
                )
                ptr_start_train = ptr_start_val = None
                ptr_end_train = ptr_end_val = None
        else:
            X_train, y_train = X, y_indices
            X_val, y_val = None, None
            ptr_start_train, ptr_end_train = pointer_starts, pointer_ends if has_pointers else (None, None)
            ptr_start_val = ptr_end_val = None

        # Convert training data to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)

        # Create training dataset and dataloader
        if has_pointers:
            ptr_start_train_tensor = torch.LongTensor(ptr_start_train)
            ptr_end_train_tensor = torch.LongTensor(ptr_end_train)
            train_dataset = torch.utils.data.TensorDataset(
                X_train_tensor, y_train_tensor,
                ptr_start_train_tensor, ptr_end_train_tensor
            )
        else:
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
            if has_pointers:
                ptr_start_val_tensor = torch.LongTensor(ptr_start_val)
                ptr_end_val_tensor = torch.LongTensor(ptr_end_val)
                val_dataset = torch.utils.data.TensorDataset(
                    X_val_tensor, y_val_tensor,
                    ptr_start_val_tensor, ptr_end_val_tensor
                )
            else:
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
            # Gradual unfreezing for pre-trained encoder
            if use_pretrained:
                from ..config.training_config import CNNTR_FREEZE_EPOCHS, CNNTR_GRADUAL_UNFREEZE, CNNTR_UNFREEZE_SCHEDULE
                if CNNTR_GRADUAL_UNFREEZE:
                    stage1_epoch = CNNTR_UNFREEZE_SCHEDULE['stage1_epoch']
                    stage2_epoch = CNNTR_UNFREEZE_SCHEDULE['stage2_epoch']
                    stage3_epoch = CNNTR_UNFREEZE_SCHEDULE['stage3_epoch']

                    if epoch == stage1_epoch:
                        self.unfreeze_encoder_gradual(stage=1)
                    elif epoch == stage2_epoch:
                        self.unfreeze_encoder_gradual(stage=2)
                    elif epoch == stage3_epoch:
                        self.unfreeze_encoder_gradual(stage=3)

            # Progressive loss weighting: start classification-heavy, gradually add pointer tasks
            # 97 samples too small for strong multi-task - let classification converge first
            if has_pointers:
                epoch_ratio = min(epoch / 50, 1.0)  # 0→1 over 50 epochs
                current_alpha = 1.0  # Classification weight stays constant
                current_beta = 0.1 * epoch_ratio  # Pointer weight: 0.0 → 0.1 over 50 epochs

                if epoch == 0:
                    print(f"[PROGRESSIVE LOSS] Epoch {epoch}: alpha={current_alpha:.2f}, beta={current_beta:.4f} (pointer tasks disabled)")
                elif epoch == 10:
                    print(f"[PROGRESSIVE LOSS] Epoch {epoch}: alpha={current_alpha:.2f}, beta={current_beta:.4f} (pointer tasks at {epoch_ratio*100:.0f}%)")
                elif epoch == 50:
                    print(f"[PROGRESSIVE LOSS] Epoch {epoch}: alpha={current_alpha:.2f}, beta={current_beta:.4f} (pointer tasks at full strength)")
            else:
                current_alpha = self.loss_alpha
                current_beta = self.loss_beta

            # Training phase
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, batch_data in enumerate(train_dataloader):
                # Unpack batch (handles both single-task and multi-task)
                if has_pointers:
                    batch_X, batch_y, batch_start, batch_end = batch_data
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    batch_start = batch_start.to(self.device, non_blocking=True)
                    batch_end = batch_end.to(self.device, non_blocking=True)
                else:
                    batch_X, batch_y = batch_data
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)

                # Apply temporal augmentation first (jitter, scaling, time_warp)
                if self.use_temporal_aug:
                    batch_X = self.temporal_aug.apply_augmentation(batch_X)

                # Then apply mixup/cutmix
                # NOTE: Augmentation only applies to classification labels, not pointers
                batch_X_aug, y_a, y_b, lam = mixup_cutmix(
                    batch_X, batch_y,
                    mixup_alpha=self.mixup_alpha,
                    cutmix_prob=self.cutmix_prob,
                )

                optimizer.zero_grad()

                # Forward pass with mixed precision
                if self.use_amp:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(batch_X_aug)

                        if has_pointers:
                            # Multi-task loss with progressive weighting
                            targets = {
                                'class': batch_y,
                                'start_idx': batch_start,
                                'end_idx': batch_end
                            }
                            loss, loss_dict = compute_multitask_loss(
                                outputs, targets,
                                alpha=current_alpha,
                                beta=current_beta,
                                device=str(self.device)
                            )
                            logits = outputs['classification']
                        else:
                            # Single-task classification loss
                            logits = outputs
                            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard forward/backward pass
                    outputs = self.model(batch_X_aug)

                    if has_pointers:
                        # Multi-task loss with progressive weighting
                        targets = {
                            'class': batch_y,
                            'start_idx': batch_start,
                            'end_idx': batch_end
                        }
                        loss, loss_dict = compute_multitask_loss(
                            outputs, targets,
                            alpha=current_alpha,
                            beta=current_beta,
                            device=str(self.device)
                        )
                        logits = outputs['classification']
                    else:
                        # Single-task classification loss
                        logits = outputs
                        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

                    loss.backward()
                    optimizer.step()

                # Track metrics (using original labels for accuracy)
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

                # Log multi-task losses periodically
                if has_pointers and (batch_idx % 10 == 0) and (epoch % max(1, self.n_epochs // 10) == 0):
                    print(f"  Batch {batch_idx:3d} | Class: {loss_dict['class']:.4f} | "
                          f"Start: {loss_dict['start']:.4f} | End: {loss_dict['end']:.4f}")

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
                        # Unpack batch (handles both single-task and multi-task)
                        if has_pointers:
                            batch_X, batch_y, batch_start, batch_end = batch_data
                            batch_X = batch_X.to(self.device, non_blocking=True)
                            batch_y = batch_y.to(self.device, non_blocking=True)
                            batch_start = batch_start.to(self.device, non_blocking=True)
                            batch_end = batch_end.to(self.device, non_blocking=True)
                        else:
                            batch_X, batch_y = batch_data
                            batch_X = batch_X.to(self.device, non_blocking=True)
                            batch_y = batch_y.to(self.device, non_blocking=True)

                        if self.use_amp:
                            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                                outputs = self.model(batch_X)

                                if has_pointers:
                                    # Multi-task loss with progressive weighting
                                    targets = {
                                        'class': batch_y,
                                        'start_idx': batch_start,
                                        'end_idx': batch_end
                                    }
                                    loss, _ = compute_multitask_loss(
                                        outputs, targets,
                                        alpha=current_alpha,
                                        beta=current_beta,
                                        device=str(self.device)
                                    )
                                    logits = outputs['classification']
                                else:
                                    logits = outputs
                                    loss = criterion(logits, batch_y)
                        else:
                            outputs = self.model(batch_X)

                            if has_pointers:
                                # Multi-task loss with progressive weighting
                                targets = {
                                    'class': batch_y,
                                    'start_idx': batch_start,
                                    'end_idx': batch_end
                                }
                                loss, _ = compute_multitask_loss(
                                    outputs, targets,
                                    alpha=current_alpha,
                                    beta=current_beta,
                                    device=str(self.device)
                                )
                                logits = outputs['classification']
                            else:
                                logits = outputs
                                loss = criterion(logits, batch_y)

                        val_loss += loss.item()
                        _, predicted = torch.max(logits, 1)
                        val_correct += (predicted == batch_y).sum().item()
                        val_total += batch_y.size(0)

                avg_val_loss = val_loss / len(val_dataloader)
                val_accuracy = val_correct / val_total

                # Per-class accuracy tracking (every 10 epochs or on collapse warning)
                if (epoch + 1) % 10 == 0 or epoch < 5:
                    from ..validation.training_validator import detect_class_collapse

                    # Get validation predictions for per-class analysis
                    self.model.eval()
                    with torch.no_grad():
                        val_X_tensor = torch.FloatTensor(X_val).to(self.device)
                        val_outputs = self.model(val_X_tensor)
                        if isinstance(val_outputs, dict):
                            val_logits = val_outputs['classification']
                        else:
                            val_logits = val_outputs
                        _, val_preds = torch.max(val_logits, 1)

                    # Detect class collapse
                    val_preds_np = val_preds.cpu().numpy()
                    class_accs = detect_class_collapse(
                        val_preds_np,
                        y_val,
                        epoch + 1,
                        threshold=0.1,
                        class_names=self.idx_to_label
                    )

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

        Note:
            For pointer predictions, use predict_with_pointers()
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
            outputs = self.model(X_tensor)

            # Handle multi-task output
            if isinstance(outputs, dict):
                logits = outputs['classification']
            else:
                logits = outputs

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

        Note:
            For pointer predictions, use predict_with_pointers()
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
            outputs = self.model(X_tensor)

            # Handle multi-task output
            if isinstance(outputs, dict):
                logits = outputs['classification']
            else:
                logits = outputs

            probs = F.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def predict_with_pointers(self, X: np.ndarray) -> dict:
        """Predict class labels AND pointer start/end (if model trained for pointers).

        This method returns comprehensive predictions including classification and
        pointer localization within the inner window [30:75].

        Args:
            X: Feature matrix of shape [N, D] or [N, T, D]
                For OHLC data: [N, 105, 4] or flattened [N, 420]

        Returns:
            Dictionary containing:
            {
                'labels': [N] predicted class labels (strings),
                'probabilities': [N, 3] class probabilities,
                'start_probabilities': [N, 45] pointer start probabilities (sigmoid),
                'end_probabilities': [N, 45] pointer end probabilities (sigmoid),
                'start_predictions': [N] predicted start indices (argmax) in [0, 44],
                'end_predictions': [N] predicted end indices (argmax) in [0, 44]
            }

        Raises:
            ValueError: If model not trained with predict_pointers=True

        Example:
            >>> model = CnnTransformerModel(predict_pointers=True)
            >>> model.fit(X_train, y_train, pointer_starts=starts, pointer_ends=ends)
            >>> results = model.predict_with_pointers(X_test)
            >>> print(results['labels'])  # ['consolidation', 'retracement', ...]
            >>> print(results['start_predictions'])  # [5, 12, 8, ...]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if not self.predict_pointers:
            raise ValueError(
                "Model not trained with pointer prediction. "
                "Set predict_pointers=True when initializing the model."
            )

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
            outputs = self.model(X_tensor)

            # Extract outputs
            class_logits = outputs['classification']
            start_logits = outputs['start']
            end_logits = outputs['end']

            # Classification predictions
            class_probs = F.softmax(class_logits, dim=1)
            _, class_preds = torch.max(class_logits, 1)

            # Pointer predictions (apply sigmoid to logits)
            start_probs = torch.sigmoid(start_logits)
            end_probs = torch.sigmoid(end_logits)
            start_preds = torch.argmax(start_probs, dim=1)
            end_preds = torch.argmax(end_probs, dim=1)

        # Convert to numpy and original labels
        predicted_labels = np.array([self.idx_to_label[idx.item()] for idx in class_preds])

        return {
            'labels': predicted_labels,
            'probabilities': class_probs.cpu().numpy(),
            'start_probabilities': start_probs.cpu().numpy(),
            'end_probabilities': end_probs.cpu().numpy(),
            'start_predictions': start_preds.cpu().numpy(),
            'end_predictions': end_preds.cpu().numpy()
        }

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
            'predict_pointers': self.predict_pointers,
            'hyperparams': {
                'cnn_channels': self.cnn_channels,
                'cnn_kernels': self.cnn_kernels,
                'transformer_layers': self.transformer_layers,
                'transformer_heads': self.transformer_heads,
                'dropout_rate': self.dropout_rate,
                'loss_alpha': self.loss_alpha,
                'loss_beta': self.loss_beta,
            }
        }

        torch.save(save_dict, path)

    def load(self, path: Path) -> "CnnTransformerModel":
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
        self.predict_pointers = checkpoint.get('predict_pointers', False)  # Default False for backward compatibility

        # Restore hyperparameters
        hyperparams = checkpoint['hyperparams']
        self.cnn_channels = hyperparams['cnn_channels']
        self.cnn_kernels = hyperparams['cnn_kernels']
        self.transformer_layers = hyperparams['transformer_layers']
        self.transformer_heads = hyperparams['transformer_heads']
        self.dropout_rate = hyperparams['dropout_rate']
        self.loss_alpha = hyperparams.get('loss_alpha', 0.5)  # Default for backward compatibility
        self.loss_beta = hyperparams.get('loss_beta', 0.25)   # Default for backward compatibility

        # Rebuild and load model
        self.model = self._build_model(self.input_dim, self.n_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.is_fitted = True
        return self

    def load_pretrained_encoder(self, encoder_path: Path) -> "CnnTransformerModel":
        """Load pre-trained encoder weights from SSL pre-training.

        This method loads weights from a TS-TCC pre-trained encoder and maps them
        to this model's CNN and Transformer layers. The classification head
        remains randomly initialized and will be trained on labeled data.

        Args:
            encoder_path: Path to pre-trained encoder weights (.pt file)
                         Expected format: {'encoder_state_dict': {...}, 'hyperparams': {...}}

        Returns:
            Self with pre-trained encoder weights loaded

        Example:
            >>> model = CnnTransformerModel()
            >>> model.load_pretrained_encoder('data/artifacts/pretrained/encoder_weights.pt')
            >>> model.fit(X_train, y_train)  # Fine-tune on labeled data
        """
        print(f"[SSL] Loading pre-trained encoder from: {encoder_path}")

        # Load checkpoint
        checkpoint = torch.load(encoder_path, map_location=self.device)
        encoder_state_dict = checkpoint['encoder_state_dict']
        hyperparams = checkpoint['hyperparams']

        # Verify architecture compatibility
        if self.cnn_channels != hyperparams['cnn_channels']:
            raise ValueError(
                f"Architecture mismatch: cnn_channels {self.cnn_channels} != "
                f"pre-trained {hyperparams['cnn_channels']}"
            )
        if self.cnn_kernels != hyperparams['cnn_kernels']:
            raise ValueError(
                f"Architecture mismatch: cnn_kernels {self.cnn_kernels} != "
                f"pre-trained {hyperparams['cnn_kernels']}"
            )

        # Map encoder weights to model
        # The TS-TCC encoder has the same structure as CnnTransformerNet:
        # - cnn_blocks (CNN layers)
        # - transformer (Transformer encoder)
        # - rel_pos_enc (Relative positional encoding)
        #
        # We load these weights but keep the classification head randomly initialized

        if self.model is None:
            raise ValueError(
                "Model not built yet. Call fit() first or build model manually "
                "with _build_model()"
            )

        # Get current model state dict
        model_state_dict = self.model.state_dict()

        # Map encoder weights
        pretrained_keys_loaded = 0
        for key, value in encoder_state_dict.items():
            # Encoder keys should match model keys directly
            # (both use the same architecture)
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    model_state_dict[key] = value
                    pretrained_keys_loaded += 1
                else:
                    print(f"[SSL] WARNING: Shape mismatch for {key}: "
                          f"{model_state_dict[key].shape} != {value.shape}")
            else:
                print(f"[SSL] WARNING: Key {key} not found in model")

        # Load mapped weights
        self.model.load_state_dict(model_state_dict)

        print(f"[SSL] Loaded {pretrained_keys_loaded} pre-trained layers")
        print(f"[SSL] Encoder pre-training complete - ready for fine-tuning on labeled data")
        print(f"[SSL] Classification head will be trained from scratch")

        return self

    def freeze_encoder(self) -> None:
        """Freeze encoder weights (CNN blocks + Transformer) to prevent updates during training.

        Use this after loading pre-trained encoder to preserve learned representations
        while training only the classification head.
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call fit() or _build_model() first")

        # Freeze CNN blocks
        for param in self.model.cnn_blocks.parameters():
            param.requires_grad = False

        # Freeze transformer layers
        for param in self.model.transformer.parameters():
            param.requires_grad = False

        # Freeze positional encoding
        for param in self.model.rel_pos_enc.parameters():
            param.requires_grad = False

        frozen_count = sum(1 for p in self.model.parameters() if not p.requires_grad)
        trainable_count = sum(1 for p in self.model.parameters() if p.requires_grad)

        print(f"[FREEZE] Encoder frozen: {frozen_count} frozen params, {trainable_count} trainable params")

    def unfreeze_encoder_gradual(self, stage: int) -> None:
        """Gradually unfreeze encoder layers for fine-tuning.

        Args:
            stage: Unfreezing stage:
                1: Unfreeze last transformer layer only
                2: Unfreeze all transformer layers
                3: Unfreeze everything (CNN blocks + Transformer)
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call fit() or _build_model() first")

        if stage == 1:
            # Unfreeze last transformer layer
            last_layer = self.model.transformer.layers[-1]
            for param in last_layer.parameters():
                param.requires_grad = True
            print(f"[UNFREEZE] Stage 1: Last transformer layer unfrozen")

        elif stage == 2:
            # Unfreeze all transformer layers
            for param in self.model.transformer.parameters():
                param.requires_grad = True
            print(f"[UNFREEZE] Stage 2: All transformer layers unfrozen")

        elif stage == 3:
            # Unfreeze everything
            for param in self.model.cnn_blocks.parameters():
                param.requires_grad = True
            for param in self.model.transformer.parameters():
                param.requires_grad = True
            for param in self.model.rel_pos_enc.parameters():
                param.requires_grad = True
            print(f"[UNFREEZE] Stage 3: Full model unfrozen (fine-tuning all layers)")

        else:
            raise ValueError(f"Invalid stage {stage}. Must be 1, 2, or 3")

        # Log current status
        frozen_count = sum(1 for p in self.model.parameters() if not p.requires_grad)
        trainable_count = sum(1 for p in self.model.parameters() if p.requires_grad)
        print(f"[UNFREEZE] Current status: {frozen_count} frozen params, {trainable_count} trainable params")
