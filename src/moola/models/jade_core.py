"""Jade Compact - Minimal Viable BiLSTM Architecture.

Clean nn.Module with no training logic - just architecture.
Implements Stones non-negotiables for robust multi-task learning.

Architecture:
    - BiLSTM(11→96×2, 1 layer) → projection (64) → classification head
    - Dropout: recurrent 0.7, dense 0.6, input 0.3
    - Center+length pointer encoding
    - Learnable uncertainty weighting (Kendall et al., CVPR 2018)
    - Optional CRF for contiguous span detection

Uncertainty Weighting:
    - Learnable σ_ptr, σ_type parameters for automatic task balancing
    - Loss: L = (1/2σ_ptr²)L_ptr + (1/2σ_type²)L_type + log(σ_ptr × σ_type)
    - Prevents manual λ tuning, adapts during training

Soft Span Masks (CRF):
    - Replaces binary 0/1 labels with soft probabilities [0-1]
    - CRF layer ensures contiguous spans (no isolated predictions)
    - Addresses zero bias (99% "no" drowning signal)
    - +10-15% accuracy on small datasets (Zhong et al. 2023)

Usage:
    >>> from moola.models.jade_core import JadeCompact
    >>> model = JadeCompact(input_size=11, hidden_size=96, num_layers=1,
    ...                     predict_pointers=True, predict_expansion_sequence=True, use_crf=True)
    >>> x = torch.randn(32, 105, 11)  # (batch, time, features)
    >>> output = model(x)  # {"logits": ..., "pointers": ..., "expansion_spans": ..., ...}
"""

from typing import Optional

import torch
import torch.nn as nn

try:
    from torchcrf import CRF

    HAS_CRF = True
except ImportError:
    HAS_CRF = False
    CRF = None


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
        use_crf: bool = False,
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
        self.use_crf = use_crf

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
            # Soft span head: Per-timestep classification (soft probabilities, not binary)
            # Input: lstm_out (batch, 105, lstm_output_size)
            # Output: (batch, 105, 2) for CRF (2 states: out-of-span, in-span)
            #      OR (batch, 105, 1) for soft sigmoid (no CRF)
            if use_crf:
                if not HAS_CRF:
                    raise ImportError(
                        "torchcrf is required for CRF integration. "
                        "Install with: pip install pytorch-crf"
                    )
                # CRF expects (batch, seq_len, num_tags)
                self.expansion_span_emission = nn.Sequential(
                    nn.Dropout(dense_dropout),
                    nn.Linear(lstm_output_size, 2),  # 2 states: out/in span
                )
                self.crf = CRF(num_tags=2, batch_first=True)
            else:
                # Soft span head without CRF (just sigmoid probabilities)
                self.expansion_span_emission = nn.Sequential(
                    nn.Dropout(dense_dropout),
                    nn.Linear(lstm_output_size, 1),
                )
                self.crf = None

            # Countdown head: Per-timestep regression (bars until expansion starts)
            # Input: lstm_out (batch, 105, lstm_output_size)
            # Output: (batch, 105, 1) continuous countdown values
            self.expansion_countdown_head = nn.Sequential(
                nn.Dropout(dense_dropout),
                nn.Linear(lstm_output_size, 1),
            )

            # Uncertainty parameters for expansion tasks
            self.log_sigma_span = nn.Parameter(torch.tensor(0.00, dtype=torch.float32))
            self.log_sigma_countdown = nn.Parameter(torch.tensor(0.00, dtype=torch.float32))
        else:
            self.expansion_span_emission = None
            self.expansion_countdown_head = None
            self.crf = None
            self.log_sigma_span = None
            self.log_sigma_countdown = None

        # Legacy attribute names for backward compatibility
        if predict_expansion_sequence and not use_crf:
            self.expansion_binary_head = self.expansion_span_emission
            self.log_sigma_binary = self.log_sigma_span

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        num_classes: int = 3,
        predict_pointers: bool = False,
        freeze_encoder: bool = True,
        **kwargs,
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
        from pathlib import Path

        import torch

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
            **kwargs,
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
            if self.use_crf and self.crf is not None:
                # CRF mode: Emission scores for 2 states (out/in span)
                emissions = self.expansion_span_emission(lstm_out)  # (batch, 105, 2)
                output["span_emissions"] = emissions

                # Viterbi decoding for best path (inference)
                if not self.training:
                    # During inference, decode best path
                    # CRF.decode returns List[List[int]], convert to tensor
                    decoded_spans = self.crf.decode(emissions)  # List[List[int]]
                    decoded_tensor = torch.tensor(decoded_spans, device=emissions.device)
                    output["expansion_spans"] = decoded_tensor  # (batch, 105) with 0/1

                    # Also provide soft probabilities from emissions
                    # P(in-span) = softmax(emissions)[:, :, 1]
                    span_probs = torch.softmax(emissions, dim=-1)[:, :, 1]
                    output["expansion_binary"] = span_probs
                else:
                    # During training, CRF forward returns log-likelihood
                    # This will be computed in the loss function with target tags
                    # Just provide soft probabilities for monitoring
                    span_probs = torch.softmax(emissions, dim=-1)[:, :, 1]
                    output["expansion_binary"] = span_probs

                # Uncertainty parameters
                output["sigma_span"] = torch.exp(self.log_sigma_span)
                output["log_sigma_span"] = self.log_sigma_span

            else:
                # Soft span mode (no CRF): sigmoid probabilities [0-1]
                span_logits = self.expansion_span_emission(lstm_out).squeeze(-1)  # (batch, 105)
                output["expansion_binary_logits"] = span_logits
                output["expansion_binary"] = torch.sigmoid(span_logits)

                # Uncertainty parameters (backward compatibility)
                output["sigma_binary"] = torch.exp(self.log_sigma_span)
                output["sigma_span"] = torch.exp(self.log_sigma_span)

            # Countdown head: (batch, 105, 1) -> (batch, 105)
            expansion_countdown = self.expansion_countdown_head(lstm_out).squeeze(-1)
            output["expansion_countdown"] = expansion_countdown

            # Countdown uncertainty
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


# ============================================================================
# Soft Span Mask Loss and Metrics
# ============================================================================


def soft_span_loss(
    pred_probs: torch.Tensor,
    target_soft: torch.Tensor,
    reduction: str = "mean",
    epsilon: float = 1e-7,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """Compute soft probability loss for span masks with class weighting.

    Replaces binary cross-entropy to handle soft (non-binary) targets.
    Addresses gradient scale bias by normalizing per-element contributions.
    Supports positive class weighting to handle class imbalance.

    Args:
        pred_probs: Predicted probabilities, shape (batch, seq_len) or (batch, seq_len, 1)
        target_soft: Soft target probabilities [0-1], same shape as pred_probs
        reduction: 'mean', 'sum', or 'none'
        epsilon: Small constant for numerical stability
        pos_weight: Weight for positive class (in-span). Use 1/positive_ratio for balance.
                    E.g., if 7.1% in-span, use pos_weight=14.1 (1/0.071)

    Returns:
        Loss value

    Example:
        >>> pred = torch.sigmoid(logits)  # (batch, 105)
        >>> target = torch.tensor([0.0, 0.2, 0.9, 1.0, 0.8, ...])  # Soft labels
        >>> loss = soft_span_loss(pred, target, pos_weight=13.1)  # For 7.1% positive class
    """
    # Ensure same shape
    if pred_probs.dim() == 3 and pred_probs.size(-1) == 1:
        pred_probs = pred_probs.squeeze(-1)
    if target_soft.dim() == 3 and target_soft.size(-1) == 1:
        target_soft = target_soft.squeeze(-1)

    # Clamp probabilities for numerical stability
    pred_probs = torch.clamp(pred_probs, epsilon, 1 - epsilon)

    # Weighted binary cross-entropy with soft targets:
    # BCE_weighted = -[w * t * log(p) + (1-t) * log(1-p)]
    # where t can be non-binary [0-1], w is positive class weight
    bce = -(
        pos_weight * target_soft * torch.log(pred_probs)
        + (1 - target_soft) * torch.log(1 - pred_probs)
    )

    # Reduction
    if reduction == "mean":
        return bce.mean()
    elif reduction == "sum":
        return bce.sum()
    elif reduction == "none":
        return bce
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def crf_span_loss(
    model: nn.Module,
    emissions: torch.Tensor,
    target_tags: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute CRF negative log-likelihood loss.

    Args:
        model: JadeCompact model with CRF layer
        emissions: Emission scores from model, shape (batch, seq_len, num_tags)
        target_tags: Target tag sequence, shape (batch, seq_len) with values in [0, num_tags-1]
        mask: Optional mask for variable-length sequences, shape (batch, seq_len)

    Returns:
        Negative log-likelihood loss (scalar)

    Example:
        >>> emissions = model(x)["span_emissions"]  # (batch, 105, 2)
        >>> target_tags = (target_spans > 0.5).long()  # Convert soft to binary tags
        >>> loss = crf_span_loss(model, emissions, target_tags)
    """
    if not hasattr(model, "crf") or model.crf is None:
        raise ValueError("Model does not have a CRF layer")

    # CRF forward returns log-likelihood
    # Negative log-likelihood is the loss
    if mask is None:
        mask = torch.ones_like(target_tags, dtype=torch.bool)

    log_likelihood = model.crf(emissions, target_tags, mask=mask, reduction="mean")
    return -log_likelihood


def compute_span_f1(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute F1 score for span detection.

    Args:
        pred: Predicted span probabilities, shape (seq_len,) or (batch, seq_len)
        target: Target span labels (0/1 or soft), same shape as pred
        threshold: Threshold for converting soft predictions to binary

    Returns:
        F1 score (scalar)

    Example:
        >>> pred = torch.tensor([0.1, 0.3, 0.8, 0.9, 0.7, 0.2])
        >>> target = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        >>> f1 = compute_span_f1(pred, target)
    """
    # Convert to binary if needed
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    # True positives, false positives, false negatives
    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()

    # Precision, recall, F1
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return f1


def compute_span_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    """Compute precision, recall, and F1 for span detection.

    Args:
        pred: Predicted span probabilities, shape (seq_len,) or (batch, seq_len)
        target: Target span labels (0/1 or soft), same shape as pred
        threshold: Threshold for converting soft predictions to binary

    Returns:
        Dictionary with 'precision', 'recall', 'f1' keys

    Example:
        >>> pred = torch.tensor([0.1, 0.3, 0.8, 0.9, 0.7, 0.2])
        >>> target = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        >>> metrics = compute_span_metrics(pred, target)
        >>> print(f"Precision: {metrics['precision']:.3f}, F1: {metrics['f1']:.3f}")
    """
    # Convert to binary if needed
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    # True positives, false positives, false negatives
    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()

    # Precision, recall, F1
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_hit_at_k(
    pred_start: int, pred_end: int, target_start: int, target_end: int, k: int = 3
) -> float:
    """Compute Hit@±k metric for span boundaries.

    Returns 1.0 if both boundaries are within k bars of target, else 0.0.

    Args:
        pred_start: Predicted expansion start
        pred_end: Predicted expansion end
        target_start: Target expansion start
        target_end: Target expansion end
        k: Tolerance in bars (default: 3)

    Returns:
        1.0 if hit, 0.0 otherwise

    Example:
        >>> hit = compute_hit_at_k(38, 62, 40, 60, k=3)
        >>> print(hit)  # 1.0 (both within ±3 bars)
    """
    start_hit = abs(pred_start - target_start) <= k
    end_hit = abs(pred_end - target_end) <= k

    return 1.0 if (start_hit and end_hit) else 0.0
