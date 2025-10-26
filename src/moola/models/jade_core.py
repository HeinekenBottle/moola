"""Jade Compact - Minimal Viable BiLSTM Architecture.

Clean nn.Module with no training logic - just architecture.
Implements Stones non-negotiables for robust multi-task learning.

Architecture:
    - BiLSTM(11→96×2, 1 layer) → projection (64) → classification head
    - Dropout: recurrent 0.7, dense 0.6, input 0.3
    - Center+length pointer encoding
    - Learnable uncertainty weighting (Kendall et al., CVPR 2018)

Uncertainty Weighting:
    - Learnable σ_ptr, σ_type parameters for automatic task balancing
    - Loss: L = (1/2σ_ptr²)L_ptr + (1/2σ_type²)L_type + log(σ_ptr × σ_type)
    - Prevents manual λ tuning, adapts during training

Usage:
    >>> from moola.models.jade_core import JadeCompact
    >>> model = JadeCompact(input_size=11, hidden_size=96, num_layers=1, predict_pointers=True)
    >>> x = torch.randn(32, 105, 11)  # (batch, time, features)
    >>> output = model(x)  # {"logits": ..., "pointers": ..., "sigma_ptr": ..., "sigma_type": ...}
"""

from typing import Optional

import torch
import torch.nn as nn


class JadeCompact(nn.Module):
    """Compact variant of Jade for small datasets.
    
    Reduced parameter count for 174-sample regime:
    - 1 layer BiLSTM (vs 2)
    - 96 hidden size (vs 128)
    - Projection head to 64 dimensions
    
    Total params: ~52K (vs ~85K for full Jade)
    """
    
    MODEL_ID = "moola-lstm-s-v1.1"
    CODENAME = "Jade-Compact"
    
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 96,
        num_layers: int = 1,
        dropout: float = 0.7,
        input_dropout: float = 0.3,
        dense_dropout: float = 0.6,
        num_classes: int = 3,
        predict_pointers: bool = False,
        predict_expansion_sequence: bool = False,
        proj_head: bool = True,
        head_width: int = 64,
        seed: Optional[int] = None,
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.seed = seed
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.predict_pointers = predict_pointers
        self.predict_expansion_sequence = predict_expansion_sequence
        
        # Input dropout (stronger for small dataset)
        self.input_dropout = nn.Dropout(input_dropout)
        
        # BiLSTM encoder
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        # PERFORMANCE: Enable gradient checkpointing for memory efficiency
        self.use_checkpointing = False
        
        # Projection head (optional dimensionality reduction)
        lstm_output_size = hidden_size * 2
        if proj_head:
            self.projection = nn.Sequential(
                nn.Linear(lstm_output_size, head_width),
                nn.ReLU(),
            )
            backbone_out = head_width
        else:
            self.projection = nn.Identity()
            backbone_out = lstm_output_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dense_dropout),
            nn.Linear(backbone_out, num_classes),
        )
        
        # Optional pointer prediction head
        if predict_pointers:
            self.pointer_head = nn.Sequential(
                nn.Dropout(dense_dropout),
                nn.Linear(backbone_out, 2),
            )

            # Uncertainty weighting (Kendall et al., CVPR 2018)
            # Learnable log-variance parameters for automatic task balancing
            # L_total = (1/2σ_ptr²)L_ptr + (1/2σ_type²)L_type + log(σ_ptr × σ_type)
            # Initialize: log_sigma_ptr ~ -0.3 (σ ≈ 0.74), log_sigma_type ~ 0.0 (σ ≈ 1.0)
            self.log_sigma_ptr = nn.Parameter(torch.tensor(-0.30, dtype=torch.float32))
            self.log_sigma_type = nn.Parameter(torch.tensor(0.00, dtype=torch.float32))
        else:
            self.pointer_head = None
            self.log_sigma_ptr = None
            self.log_sigma_type = None

        # Optional per-timestep expansion detection heads
        if predict_expansion_sequence:
            # Binary head: Per-timestep classification (is this bar part of expansion?)
            # Input: lstm_out (batch, 105, lstm_output_size)
            # Output: (batch, 105, 1) sigmoid probabilities
            self.expansion_binary_head = nn.Sequential(
                nn.Dropout(dense_dropout),
                nn.Linear(lstm_output_size, 1),
            )

            # Countdown head: Per-timestep regression (bars until expansion starts)
            # Input: lstm_out (batch, 105, lstm_output_size)
            # Output: (batch, 105, 1) continuous countdown values
            self.expansion_countdown_head = nn.Sequential(
                nn.Dropout(dense_dropout),
                nn.Linear(lstm_output_size, 1),
            )

            # Uncertainty parameters for expansion tasks
            self.log_sigma_binary = nn.Parameter(torch.tensor(0.00, dtype=torch.float32))
            self.log_sigma_countdown = nn.Parameter(torch.tensor(0.00, dtype=torch.float32))
        else:
            self.expansion_binary_head = None
            self.expansion_countdown_head = None
            self.log_sigma_binary = None
            self.log_sigma_countdown = None
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        num_classes: int = 3,
        predict_pointers: bool = False,
        freeze_encoder: bool = True,
        **kwargs
    ) -> "JadeCompact":
        """Load JadeCompact with pre-trained encoder weights from JadePretrainer.

        Args:
            pretrained_path: Path to pretrained checkpoint (.pt file)
            num_classes: Number of output classes (default 3)
            predict_pointers: Enable pointer prediction head (default False)
            freeze_encoder: Freeze encoder weights for fine-tuning (default True)
            **kwargs: Additional model initialization arguments

        Returns:
            JadeCompact model with pre-trained encoder

        Example:
            >>> model = JadeCompact.from_pretrained(
            ...     "artifacts/jade_pretrain_20ep/checkpoint_best.pt",
            ...     predict_pointers=True,
            ...     freeze_encoder=True
            ... )
        """
        import torch
        from pathlib import Path

        # Load pretrained checkpoint
        checkpoint_path = Path(pretrained_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location="cpu")

        # Extract encoder config from checkpoint
        # Handle both legacy (model_config) and new (config.model) checkpoint formats
        if "model_config" in checkpoint:
            model_config = checkpoint["model_config"]
        elif "config" in checkpoint and "model" in checkpoint["config"]:
            model_config = checkpoint["config"]["model"]
        else:
            model_config = {}
        input_size = model_config.get("input_size", 11)
        hidden_size = model_config.get("hidden_size", 128)
        num_layers = model_config.get("num_layers", 2)
        dropout = model_config.get("dropout", 0.7)

        # Create JadeCompact model with matching encoder architecture
        # Note: JadePretrainer uses 2 layers, 128 hidden by default
        # JadeCompact defaults to 1 layer, 96 hidden, so we override with checkpoint config
        model = cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
            predict_pointers=predict_pointers,
            **kwargs
        )

        # Load encoder weights (LSTM only)
        # JadePretrainer structure: encoder.weight_ih_l0, encoder.weight_hh_l0, etc.
        # JadeCompact structure: lstm.weight_ih_l0, lstm.weight_hh_l0, etc.

        pretrained_state = checkpoint["model_state_dict"]
        encoder_weights = {}

        for key, value in pretrained_state.items():
            if key.startswith("encoder."):
                # Map encoder.* → lstm.*
                new_key = key.replace("encoder.", "lstm.")
                encoder_weights[new_key] = value

        # Load encoder weights with strict=False (ignore decoder, classification head)
        model.load_state_dict(encoder_weights, strict=False)

        # Freeze encoder if requested (recommended for small datasets)
        if freeze_encoder:
            for param in model.lstm.parameters():
                param.requires_grad = False

        print(f"✓ Loaded pre-trained encoder from {pretrained_path}")
        print(f"  - Encoder: {num_layers} layer BiLSTM, {hidden_size} hidden × 2 directions")
        print(f"  - Frozen: {freeze_encoder}")
        print(f"  - Trainable params: {model.get_num_parameters()['trainable']:,}")

        return model

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass with Jade-Compact architecture."""
        # Input dropout
        x = self.input_dropout(x)

        # BiLSTM encoding with optional gradient checkpointing
        if self.use_checkpointing and self.training:
            # PERFORMANCE: Use gradient checkpointing for memory efficiency
            def lstm_forward(inp):
                return self.lstm(inp)[0]
            lstm_out = torch.utils.checkpoint.checkpoint(lstm_forward, x)
        else:
            lstm_out, _ = self.lstm(x)

        # Global average pooling
        pooled = lstm_out.mean(dim=1)

        # Projection
        features = self.projection(pooled)

        # Classification
        logits = self.classifier(features)

        output = {"logits": logits}

        # Optional pointer prediction
        if self.predict_pointers and self.pointer_head is not None:
            pointers = torch.sigmoid(self.pointer_head(features))
            output["pointers"] = pointers

            # Include uncertainty parameters for loss computation
            # Convert from log-space: σ = exp(log_sigma)
            output["sigma_ptr"] = torch.exp(self.log_sigma_ptr)
            output["sigma_type"] = torch.exp(self.log_sigma_type)
            output["log_sigma_ptr"] = self.log_sigma_ptr
            output["log_sigma_type"] = self.log_sigma_type

        # Optional per-timestep expansion detection
        if self.predict_expansion_sequence:
            # Binary head: (batch, 105, 1) -> (batch, 105)
            expansion_binary_logits = self.expansion_binary_head(lstm_out).squeeze(-1)
            output["expansion_binary_logits"] = expansion_binary_logits
            output["expansion_binary"] = torch.sigmoid(expansion_binary_logits)

            # Countdown head: (batch, 105, 1) -> (batch, 105)
            expansion_countdown = self.expansion_countdown_head(lstm_out).squeeze(-1)
            output["expansion_countdown"] = expansion_countdown

            # Uncertainty parameters for expansion tasks
            output["sigma_binary"] = torch.exp(self.log_sigma_binary)
            output["sigma_countdown"] = torch.exp(self.log_sigma_countdown)

        return output
    
    def get_num_parameters(self) -> dict:
        """Get parameter count statistics."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.use_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.use_checkpointing = False

