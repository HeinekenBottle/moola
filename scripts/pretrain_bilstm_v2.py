#!/usr/bin/env python3
"""
BiLSTM Masked Autoencoder Pre-training v2
==========================================
Creates pretrained encoder that MATCHES the dual-task architecture (type + pointers).

Key difference from v1:
- Builds FULL EnhancedSimpleLSTM with pointer head
- Only trains encoder + decoder (freeze task heads)
- Saves complete model state_dict for perfect architecture match

Usage:
    python3 scripts/pretrain_bilstm_v2.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

# Setup logging
logger.remove()
logger.add(lambda msg: print(msg, end=""), colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")


def mae_pretrain_loss(model, decoder, X, mask_ratio=0.15):
    """
    Masked Autoencoder loss for pre-training.

    Args:
        model: DualTaskBiLSTM (BiLSTM with dual task heads)
        decoder: MLP decoder to reconstruct masked features
        X: Input tensor [B, L, D]
        mask_ratio: Fraction of timesteps to mask

    Returns:
        Reconstruction loss (MSE on masked timesteps only)
    """
    B, L, D = X.shape
    device = X.device

    # Random mask (15% of timesteps)
    mask = torch.rand(B, L, device=device) < mask_ratio
    X_masked = X.clone()
    X_masked[mask] = 0  # Zero out masked timesteps

    # Encode (use only encoder, ignore task heads)
    lstm_out, _ = model.lstm(X_masked)  # [B, L, 256] (128*2 for bidirectional)

    # Decode
    reconstructed = decoder(lstm_out)  # [B, L, D]

    # Loss only on masked timesteps
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    loss = F.mse_loss(reconstructed[mask], X[mask])
    return loss


def main():
    # Configuration
    INPUT_DIM = 11  # RelativeTransform features
    HIDDEN_DIM = 128  # BiLSTM hidden (256 total with bidirectional)
    MASK_RATIO = 0.15
    BATCH_SIZE = 256
    EPOCHS = 75
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    OUTPUT_PATH = Path('artifacts/encoders/pretrained/bilstm_mae_11d_v2_with_pointers.pt')
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("BiLSTM Masked Autoencoder Pre-training v2")
    logger.info("=" * 80)
    logger.info(f"Input dim: {INPUT_DIM} (RelativeTransform)")
    logger.info(f"Hidden dim: {HIDDEN_DIM} (bidirectional → 256 total)")
    logger.info(f"Mask ratio: {MASK_RATIO}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Device: {DEVICE}")
    logger.info("=" * 80)

    # 1. Load unlabeled data
    logger.info("Loading unlabeled data...")
    data_path = Path('data/processed/unlabeled/unlabeled_11d_relative.npy')
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    # Load 11D RelativeTransform features
    X_np = np.load(data_path)  # Shape: [N, 105, 11]
    X = torch.FloatTensor(X_np)

    logger.info(f"Loaded {len(X)} unlabeled samples, shape: {X.shape}")

    # 2. Build FULL architecture (with pointer head)
    logger.info("Building BiLSTM with dual task heads (type + pointers)...")

    class DualTaskBiLSTM(nn.Module):
        """BiLSTM with type classification and pointer regression heads."""
        def __init__(self, input_dim, hidden_dim, num_classes=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=True
            )
            self.dropout = nn.Dropout(dropout)

            # Type classification head
            self.type_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)
            )

            # Pointer regression head (predicts start/end indices)
            self.pointer_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 2)  # start_idx, end_idx
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)  # [B, L, hidden*2]
            pooled = lstm_out.mean(dim=1)  # [B, hidden*2]
            pooled = self.dropout(pooled)

            type_logits = self.type_head(pooled)
            pointers = self.pointer_head(pooled)
            return lstm_out, type_logits, pointers

    model = DualTaskBiLSTM(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=2,
        dropout=0.2
    ).to(DEVICE)

    # 3. Add decoder for reconstruction
    logger.info("Adding MLP decoder...")
    decoder = nn.Sequential(
        nn.Linear(256, 128),  # 256 = HIDDEN_DIM * 2 (bidirectional)
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, INPUT_DIM)
    ).to(DEVICE)

    # 4. Freeze task heads (only train encoder + decoder)
    logger.info("Freezing task heads (type_head, pointer_head)...")
    for param in model.type_head.parameters():
        param.requires_grad = False
    for param in model.pointer_head.parameters():
        param.requires_grad = False

    # Count trainable parameters
    encoder_params = sum(p.numel() for p in model.lstm.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in decoder.parameters())
    frozen_params = sum(p.numel() for p in model.type_head.parameters()) + \
                   sum(p.numel() for p in model.pointer_head.parameters())

    logger.info(f"Encoder params: {encoder_params:,} (trainable)")
    logger.info(f"Decoder params: {decoder_params:,} (trainable)")
    logger.info(f"Task heads params: {frozen_params:,} (frozen)")
    logger.info(f"Total trainable: {encoder_params + decoder_params:,}")

    # 5. Setup optimizer
    optimizer = torch.optim.Adam(
        list(model.lstm.parameters()) + list(decoder.parameters()),
        lr=LR
    )

    # 6. DataLoader
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 7. Training loop
    logger.info("=" * 80)
    logger.info("Starting pre-training...")
    logger.info("=" * 80)

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for (batch_X,) in loader:
            batch_X = batch_X.to(DEVICE)

            # MAE loss
            loss = mae_pretrain_loss(model, decoder, batch_X, mask_ratio=MASK_RATIO)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.lstm.parameters()) + list(decoder.parameters()),
                max_norm=1.0
            )
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Log progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:3d}/{EPOCHS}: Loss = {avg_loss:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save FULL model state_dict (encoder + frozen heads)
            torch.save(
                model.state_dict(),
                OUTPUT_PATH
            )
            if (epoch + 1) % 10 == 0:
                logger.info(f"  → Saved checkpoint (best loss: {best_loss:.6f})")

    # 8. Final save
    logger.info("=" * 80)
    logger.info("Pre-training complete!")
    logger.info(f"Best loss: {best_loss:.6f}")
    logger.info(f"Encoder saved: {OUTPUT_PATH}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Train with frozen encoder:")
    logger.info(f"     python3 -m moola.cli train --model enhanced_simple_lstm \\")
    logger.info(f"       --predict-pointers \\")
    logger.info(f"       --pretrained-encoder {OUTPUT_PATH} \\")
    logger.info(f"       --freeze-encoder --epochs 60")
    logger.info("")
    logger.info("  2. Expected improvement: +5-10% accuracy over baseline")


if __name__ == '__main__':
    main()
