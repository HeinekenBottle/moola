"""Masked LSTM Pre-training Infrastructure.

Main pre-training class for bidirectional masked LSTM autoencoder.
Handles training loop, early stopping, checkpointing, and encoder extraction.

Expected Performance:
    - Pre-training time: ~20 minutes on H100 GPU (11K samples, 50 epochs)
    - Fine-tuning improvement: +8-12% accuracy over baseline
    - Class collapse: Broken (Class 1: 0% → 45-55%)

Usage:
    >>> from moola.pretraining import MaskedLSTMPretrainer
    >>> pretrainer = MaskedLSTMPretrainer(
    ...     hidden_dim=128,
    ...     mask_strategy="patch",
    ...     device="cuda"
    ... )
    >>> history = pretrainer.pretrain(
    ...     X_unlabeled,
    ...     n_epochs=50,
    ...     save_path=Path("artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt")
    ... )
    >>> # Encoder saved and ready for transfer learning
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..models.bilstm_masked_autoencoder import (
    BiLSTMMaskedAutoencoder,
    apply_masking,
)
from ..utils.early_stopping import EarlyStopping
from ..utils.seeds import get_device, set_seed


class MaskedLSTMPretrainer:
    """Pre-trainer for bidirectional masked LSTM autoencoder.

    Implements complete pre-training pipeline:
        1. Data loading and train/val split
        2. Masked autoencoding training loop
        3. Early stopping and checkpointing
        4. Encoder weight extraction for transfer learning

    Args:
        input_dim: Input feature dimension (4 for OHLC)
        hidden_dim: LSTM hidden dimension per direction (total: 2*hidden_dim)
        num_layers: Number of stacked LSTM layers
        mask_ratio: Proportion of timesteps to mask (0.15 = 15%)
        mask_strategy: Masking approach ("random", "block", "patch")
        patch_size: Patch size for patch masking strategy
        learning_rate: Learning rate for AdamW optimizer
        batch_size: Training batch size
        device: Device to train on ("cpu" or "cuda")
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        mask_ratio: float = 0.15,
        mask_strategy: Literal["random", "block", "patch"] = "patch",
        patch_size: int = 7,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        device: str = "cuda",
        seed: int = 1337,
    ):
        set_seed(seed)
        self.device = get_device(device)
        self.seed = seed

        # Hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.patch_size = patch_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Build model
        self.model = BiLSTMMaskedAutoencoder(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout
        ).to(self.device)

        # Optimizer with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )

        # LR scheduler: cosine annealing for smooth learning rate decay
        self.scheduler = None  # Will be initialized in pretrain()

        # Track best model
        self.best_model_state = None

    def pretrain(
        self,
        X_unlabeled: np.ndarray,
        n_epochs: int = 50,
        val_split: float = 0.1,
        patience: int = 10,
        save_path: Path | None = None,
        verbose: bool = True,
    ) -> dict[str, list]:
        """Pre-train on unlabeled data using masked reconstruction.

        Args:
            X_unlabeled: [N, seq_len, features] unlabeled OHLC sequences
            n_epochs: Number of training epochs
            val_split: Validation split ratio (0.1 = 10%)
            patience: Early stopping patience (epochs without improvement)
            save_path: Path to save best encoder (None = don't save)
            verbose: Print training progress

        Returns:
            history: Dictionary with training metrics
                Keys: 'train_loss', 'val_loss', 'train_recon', 'val_recon'
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"MASKED LSTM PRE-TRAINING")
            print(f"{'='*70}")
            print(f"  Dataset size: {len(X_unlabeled)} samples")
            print(f"  Mask strategy: {self.mask_strategy}")
            print(f"  Mask ratio: {self.mask_ratio}")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Epochs: {n_epochs}")
            print(f"  Device: {self.device}")
            print(f"{'='*70}\n")

        # Initialize LR scheduler now that we know n_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=1e-5
        )

        # Split train/val
        N = len(X_unlabeled)
        val_size = int(N * val_split)
        indices = np.random.permutation(N)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        X_train = torch.FloatTensor(X_unlabeled[train_indices])
        X_val = torch.FloatTensor(X_unlabeled[val_indices])

        if verbose:
            print(f"[DATA SPLIT]")
            print(f"  Train: {len(X_train)} samples ({(1-val_split)*100:.0f}%)")
            print(f"  Val: {len(X_val)} samples ({val_split*100:.0f}%)")
            print()

        # Create DataLoader with optimized settings
        train_dataset = torch.utils.data.TensorDataset(X_train)

        # Import optimized DataLoader kwargs
        try:
            from ..config.performance_config import get_optimized_dataloader_kwargs

            dataloader_kwargs = get_optimized_dataloader_kwargs(is_training=True)
        except ImportError:
            # Fallback to manual configuration
            dataloader_kwargs = {
                "num_workers": 8 if self.device.type == "cuda" else 0,
                "pin_memory": True if self.device.type == "cuda" else False,
                "prefetch_factor": 2 if self.device.type == "cuda" else None,
                "persistent_workers": True if self.device.type == "cuda" else False,
            }
            dataloader_kwargs = {k: v for k, v in dataloader_kwargs.items() if v is not None}

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, **dataloader_kwargs
        )

        # Early stopping
        early_stopping = EarlyStopping(patience=patience, mode="min", verbose=verbose)

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_recon": [],
            "val_recon": [],
            "learning_rate": [],
        }

        # Setup AMP scaler for mixed precision training
        use_amp = self.device.type == "cuda" and torch.cuda.is_available()
        scaler = None
        if use_amp:
            try:
                from ..config.performance_config import get_amp_scaler

                scaler = get_amp_scaler()
                if scaler and verbose:
                    print("[PERFORMANCE] Using automatic mixed precision (AMP) for 1.5-2× speedup")
            except ImportError:
                scaler = torch.amp.GradScaler("cuda")
                if verbose:
                    print("[PERFORMANCE] Using AMP with default settings")

        # Training loop
        for epoch in range(n_epochs):
            # ============================================================
            # TRAINING PHASE
            # ============================================================
            self.model.train()
            train_losses = []
            train_recon_losses = []

            # Progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", disable=not verbose)

            for (batch_X,) in pbar:
                batch_X = batch_X.to(self.device, non_blocking=True)

                # Apply masking strategy
                x_masked, mask = apply_masking(
                    batch_X,
                    self.model.mask_token,
                    mask_strategy=self.mask_strategy,
                    mask_ratio=self.mask_ratio,
                    patch_size=self.patch_size,
                )

                self.optimizer.zero_grad()

                # Forward pass with optional AMP
                if scaler:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        reconstruction = self.model(x_masked)
                        loss, loss_dict = self.model.compute_loss(reconstruction, batch_X, mask)

                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    reconstruction = self.model(x_masked)
                    loss, loss_dict = self.model.compute_loss(reconstruction, batch_X, mask)

                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                # Track metrics
                train_losses.append(loss_dict["total"])
                train_recon_losses.append(loss_dict["reconstruction"])

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{loss_dict['total']:.4f}",
                        "recon": f"{loss_dict['reconstruction']:.4f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.6f}",
                    }
                )

            # Update learning rate
            self.scheduler.step()

            # ============================================================
            # VALIDATION PHASE
            # ============================================================
            self.model.eval()
            val_losses = []
            val_recon_losses = []

            with torch.no_grad():
                # Process validation data in batches
                for i in range(0, len(X_val), self.batch_size):
                    batch_X = X_val[i : i + self.batch_size].to(self.device, non_blocking=True)

                    # Apply masking
                    x_masked, mask = apply_masking(
                        batch_X,
                        self.model.mask_token,
                        mask_strategy=self.mask_strategy,
                        mask_ratio=self.mask_ratio,
                        patch_size=self.patch_size,
                    )

                    # Forward pass with optional AMP
                    if scaler:
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            reconstruction = self.model(x_masked)
                            loss, loss_dict = self.model.compute_loss(reconstruction, batch_X, mask)
                    else:
                        reconstruction = self.model(x_masked)
                        loss, loss_dict = self.model.compute_loss(reconstruction, batch_X, mask)

                    val_losses.append(loss_dict["total"])
                    val_recon_losses.append(loss_dict["reconstruction"])

            # ============================================================
            # LOGGING AND CHECKPOINTING
            # ============================================================
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            avg_train_recon = np.mean(train_recon_losses)
            avg_val_recon = np.mean(val_recon_losses)
            current_lr = self.scheduler.get_last_lr()[0]

            # Record history
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["train_recon"].append(avg_train_recon)
            history["val_recon"].append(avg_val_recon)
            history["learning_rate"].append(current_lr)

            # Print epoch summary
            if verbose:
                print(f"\nEpoch [{epoch+1}/{n_epochs}]")
                print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                print(f"  Train Recon: {avg_train_recon:.4f} | Val Recon: {avg_val_recon:.4f}")
                print(f"  LR: {current_lr:.6f}")

            # Early stopping check
            if early_stopping(avg_val_loss, self.model):
                if verbose:
                    print(f"\n[EARLY STOPPING] Triggered at epoch {epoch+1}")
                break

        # ============================================================
        # RESTORE BEST MODEL AND SAVE
        # ============================================================
        # Load best model from early stopping
        early_stopping.load_best_model(self.model)

        if verbose:
            print(f"\n{'='*70}")
            print(f"PRE-TRAINING COMPLETE")
            print(f"{'='*70}")
            print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
            print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
            print(f"  Best val loss: {min(history['val_loss']):.4f}")

        # Save encoder if path provided
        if save_path is not None:
            self.save_encoder(save_path)
            if verbose:
                print(f"  Encoder saved: {save_path}")

        if verbose:
            print(f"{'='*70}\n")

        return history

    def save_encoder(self, path: Path) -> None:
        """Save encoder weights and hyperparameters for transfer learning.

        Saves only the encoder portion of the model (not decoder) along
        with architecture hyperparameters needed to rebuild the encoder.

        Args:
            path: Path to save encoder checkpoint (.pt file)
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "encoder_state_dict": self.model.encoder_lstm.state_dict(),
            "hyperparams": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
            "mask_token": self.model.mask_token.data,
            "training_config": {
                "mask_strategy": self.mask_strategy,
                "mask_ratio": self.mask_ratio,
                "patch_size": self.patch_size,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
            },
        }

        torch.save(checkpoint, path)

    def load_encoder(self, path: Path) -> None:
        """Load pre-trained encoder weights.

        Args:
            path: Path to encoder checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Restore encoder weights
        self.model.encoder_lstm.load_state_dict(checkpoint["encoder_state_dict"])

        # Restore mask token
        self.model.mask_token.data = checkpoint["mask_token"]

        print(f"[PRETRAINING] Loaded encoder from {path}")


def visualize_reconstruction(
    model: BiLSTMMaskedAutoencoder,
    X_sample: torch.Tensor,
    mask_strategy: str = "random",
    mask_ratio: float = 0.15,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Visualize masked reconstruction quality (for debugging/analysis).

    Args:
        model: Trained BiLSTMMaskedAutoencoder
        X_sample: [1, seq_len, features] single sample
        mask_strategy: Masking strategy to use
        mask_ratio: Masking ratio
        device: Device for computation

    Returns:
        Dictionary with original, masked, reconstructed arrays and mask
    """
    model.eval()
    X_sample = X_sample.to(device)

    with torch.no_grad():
        # Apply masking
        x_masked, mask = apply_masking(
            X_sample, model.mask_token, mask_strategy=mask_strategy, mask_ratio=mask_ratio
        )

        # Reconstruct
        reconstruction = model(x_masked)

    return {
        "original": X_sample.cpu().numpy(),
        "masked": x_masked.cpu().numpy(),
        "reconstructed": reconstruction.cpu().numpy(),
        "mask": mask.cpu().numpy(),
    }
