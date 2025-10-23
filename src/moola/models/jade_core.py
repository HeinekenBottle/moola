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
        input_size: int = 11,
        hidden_size: int = 96,
        num_layers: int = 1,
        dropout: float = 0.7,
        input_dropout: float = 0.3,
        dense_dropout: float = 0.6,
        num_classes: int = 3,
        predict_pointers: bool = False,
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

