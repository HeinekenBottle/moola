"""Jade_Compact Architecture - Production BiLSTM with Multi-task Learning.

Jade_Compact: 1-layer BiLSTM with hidden_size=96, bidirectional=True, proj_head=true
Preserves bidirectional context while reducing parameters from 85K to ~52K.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JadeModel(nn.Module):
    """Jade_Compact Architecture - Production BiLSTM with Multi-task Learning.

    Implements Stones non-negotiables for robust multi-task learning.
    """

    # Model metadata for registry
    MODEL_ID = "moola-lstm-m-v1.0"
    CODENAME = "Jade"

    def __init__(self, input_size, hidden_size=96, num_layers=1, bidirectional=True,
                 proj_head=True, head_width=64, pointer_encoding="center_length"):
        super().__init__()
        
        # Core encoder - 1-layer BiLSTM with 96 hidden size
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                               bidirectional=bidirectional, batch_first=True, dropout=0.0)
        enc_out = hidden_size * (2 if bidirectional else 1)
        
        # Projection head - reduces to 64 dimensions
        rep = nn.Linear(enc_out, head_width) if proj_head else nn.Identity()
        self.backbone = nn.Sequential(rep, nn.ReLU())
        
        # Task heads with proper dropout (Stones: 0.4-0.5)
        backbone_out = head_width if proj_head else enc_out
        self.cls_head = nn.Sequential(nn.Dropout(0.5), nn.Linear(backbone_out, 3))
        self.ptr_head = nn.Sequential(nn.Dropout(0.5), nn.Linear(backbone_out, 2))
        
        # Kendall uncertainty parameters with proper initialization
        self.log_sigma_ptr = nn.Parameter(torch.tensor(-0.30))
        self.log_sigma_cls = nn.Parameter(torch.tensor(0.00))
        
        # Assert center_length encoding (PAPER-STRICT)
        assert pointer_encoding == "center_length"
        self.pointer_encoding = pointer_encoding

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass with Jade_Compact architecture.

        Args:
            x: Input tensor [batch, seq_len, input_dim]

        Returns:
            dict with type_logits and pointers_cl (center, length)
        """
        # BiLSTM encoding
        lstm_out, _ = self.encoder(x)  # [B, T, enc_out]
        
        # Global average pooling
        pooled = lstm_out.mean(dim=1)  # [B, enc_out]
        
        # Projection
        features = self.backbone(pooled)  # [B, head_width]
        
        # Task heads
        type_logits = self.cls_head(features)  # [B, 3]
        pointers_cl = torch.sigmoid(self.ptr_head(features))  # [B, 2] in [0,1]
        
        return {
            "type_logits": type_logits,
            "pointers": pointers_cl,
            "pointers_cl": pointers_cl,  # Explicit alias
        }

    def compute_loss(self, outputs: dict, labels: torch.Tensor, 
                    ptr_start: torch.Tensor = None, ptr_end: torch.Tensor = None) -> dict:
        """Compute uncertainty-weighted multi-task loss.

        Args:
            outputs: Model outputs dict
            labels: Classification labels [B]
            ptr_start: Pointer start positions [B] (optional)
            ptr_end: Pointer end positions [B] (optional)

        Returns:
            dict with losses and uncertainties
        """
        # Classification loss
        type_loss = F.cross_entropy(outputs["type_logits"], labels)
        
        total_loss = type_loss
        loss_dict = {"type_loss": type_loss.item(), "total_loss": type_loss.item()}
        
        # Pointer regression loss (if available)
        if ptr_start is not None and ptr_end is not None:
            # Convert to center-length
            center_target = (ptr_start + ptr_end).float() / 208.0  # Normalize to [0,1]
            length_target = (ptr_end - ptr_start).float() / 208.0   # Normalize to [0,1]
            
            targets_cl = torch.stack([center_target, length_target], dim=1)
            preds_cl = outputs["pointers_cl"]
            
            # Huber loss with δ≈0.08
            ptr_loss = F.huber_loss(preds_cl, targets_cl, delta=0.08)
            
            # Kendall uncertainty weighting
            precision_ptr = torch.exp(-self.log_sigma_ptr)
            precision_cls = torch.exp(-self.log_sigma_cls)
            
            weighted_ptr = 0.5 * precision_ptr * ptr_loss + self.log_sigma_ptr
            weighted_cls = precision_cls * type_loss + self.log_sigma_cls
            
            total_loss = weighted_ptr + weighted_cls
            
            loss_dict.update({
                "ptr_loss": ptr_loss.item(),
                "weighted_ptr": weighted_ptr.item(),
                "weighted_cls": weighted_cls.item(),
                "total_loss": total_loss.item(),
                "sigma_ptr": torch.exp(0.5 * self.log_sigma_ptr).item(),
                "sigma_cls": torch.exp(0.5 * self.log_sigma_cls).item(),
            })
        
        return loss_dict

    def get_uncertainties(self) -> dict:
        """Return current σ values for monitoring."""
        return {
            "sigma_ptr": torch.exp(0.5 * self.log_sigma_ptr).item(),
            "sigma_cls": torch.exp(0.5 * self.log_sigma_cls).item(),
        }