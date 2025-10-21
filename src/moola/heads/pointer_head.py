"""Pointer head for multi-task learning.

Predicts expansion center and length for temporal localization tasks.
"""

import torch
import torch.nn as nn


class PointerHead(nn.Module):
    """Pointer head for predicting expansion center and length.

    Outputs:
        - center: Predicted center of expansion (normalized 0-1)
        - length: Predicted length of expansion (normalized 0-1)
    """

    def __init__(self, hidden_size: int, dropout: float = 0.5):
        """Initialize pointer head.

        Args:
            hidden_size: Hidden dimension from encoder
            dropout: Dropout rate
        """
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),  # center, length
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Encoder features [batch, hidden_size * 2]

        Returns:
            Pointer predictions [batch, 2] (center, length)
        """
        return self.head(x)


class TypeHead(nn.Module):
    """Classification head for pattern type prediction.

    Predicts binary classification: consolidation vs retracement.
    """

    def __init__(self, hidden_size: int, n_classes: int = 2, dropout: float = 0.5):
        """Initialize type head.

        Args:
            hidden_size: Hidden dimension from encoder
            n_classes: Number of output classes (default: 2)
            dropout: Dropout rate
        """
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Encoder features [batch, hidden_size * 2]

        Returns:
            Class logits [batch, n_classes]
        """
        return self.head(x)
