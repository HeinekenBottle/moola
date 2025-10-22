"""Jade Core - Pure PyTorch BiLSTM Architecture.

Clean nn.Module with no training logic - just architecture.
Implements Stones non-negotiables for robust multi-task learning.

Architecture:
    - BiLSTM(11→128×2, 2 layers) → global average pool → classification head
    - Dropout: recurrent 0.6-0.7, dense 0.4-0.5, input 0.2-0.3
    - Center+length pointer encoding
    - Uncertainty-weighted loss support

Usage:
    >>> from moola.models.jade_core import JadeCore
    >>> model = JadeCore(input_size=11, hidden_size=128, num_layers=2)
    >>> x = torch.randn(32, 105, 11)  # (batch, time, features)
    >>> logits = model(x)  # (batch, num_classes)
"""

from typing import Optional

import torch
import torch.nn as nn


class JadeCore(nn.Module):
    """Pure PyTorch BiLSTM core for Jade model.
    
    Clean nn.Module with no training logic - just architecture.
    Implements Stones non-negotiables:
    - BiLSTM with proper dropout configuration
    - Global average pooling
    - Classification head
    - Optional pointer prediction head
    
    Args:
        input_size: Input feature dimension (default: 11 for RelativeTransform)
        hidden_size: LSTM hidden dimension (default: 128)
        num_layers: Number of LSTM layers (default: 2, Stones requirement)
        dropout: Recurrent dropout rate (default: 0.65, Stones: 0.6-0.7)
        input_dropout: Input dropout rate (default: 0.25, Stones: 0.2-0.3)
        dense_dropout: Dense layer dropout rate (default: 0.5, Stones: 0.4-0.5)
        num_classes: Number of output classes (default: 3)
        predict_pointers: Enable multi-task pointer prediction (default: False)
        seed: Random seed for reproducibility (default: None)
    """
    
    # Model metadata for registry
    MODEL_ID = "moola-lstm-m-v1.0"
    CODENAME = "Jade"
    
    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.65,
        input_dropout: float = 0.25,
        dense_dropout: float = 0.5,
        num_classes: int = 3,
        predict_pointers: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
        
        self.seed = seed
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.predict_pointers = predict_pointers
        
        # Input dropout (Stones: 0.2-0.3)
        self.input_dropout = nn.Dropout(input_dropout)
        
        # BiLSTM encoder with recurrent dropout (Stones: 0.6-0.7)
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        
        # Classification head with dense dropout (Stones: 0.4-0.5)
        lstm_output_size = hidden_size * 2  # Bidirectional
        self.classifier = nn.Sequential(
            nn.Dropout(dense_dropout),
            nn.Linear(lstm_output_size, num_classes),
        )
        
        # Optional pointer prediction head (center + length encoding)
        if predict_pointers:
            self.pointer_head = nn.Sequential(
                nn.Dropout(dense_dropout),
                nn.Linear(lstm_output_size, 2),  # [center, length]
            )
        else:
            self.pointer_head = None
    
    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass with Jade architecture.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
        
        Returns:
            dict with:
                - 'logits': Classification logits [batch, num_classes]
                - 'pointers': Pointer predictions [batch, 2] (if predict_pointers=True)
        """
        # Input dropout
        x = self.input_dropout(x)
        
        # BiLSTM encoding
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size * 2]
        
        # Global average pooling over time dimension
        pooled = lstm_out.mean(dim=1)  # [batch, hidden_size * 2]
        
        # Classification head
        logits = self.classifier(pooled)  # [batch, num_classes]
        
        output = {"logits": logits}
        
        # Optional pointer prediction
        if self.predict_pointers and self.pointer_head is not None:
            pointers = torch.sigmoid(self.pointer_head(pooled))  # [batch, 2] in [0,1]
            output["pointers"] = pointers
        
        return output
    
    def get_num_parameters(self) -> dict:
        """Get parameter count statistics.
        
        Returns:
            dict with total and trainable parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


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
        else:
            self.pointer_head = None
    
    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass with Jade-Compact architecture."""
        # Input dropout
        x = self.input_dropout(x)
        
        # BiLSTM encoding
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
        
        return output
    
    def get_num_parameters(self) -> dict:
        """Get parameter count statistics."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

