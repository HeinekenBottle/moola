"""Feature-Aware Masked LSTM Pre-training Infrastructure.

Enhanced pre-training pipeline for feature-aware bidirectional masked LSTM autoencoder.
Handles dual-input pre-training with both OHLC data and engineered features.

Expected Performance:
    - Pre-training time: ~25-30 minutes on H100 GPU (11K samples, 50 epochs)
    - Fine-tuning improvement: +10-15% accuracy over OHLC-only pre-training
    - Rich feature representations: Better understanding of market patterns

Usage:
    >>> from moola.pretraining import FeatureAwareMaskedLSTMPretrainer
    >>> from moola.features import AdvancedFeatureEngineer
    >>>
    >>> # Feature engineer the OHLC data first
    >>> engineer = AdvancedFeatureEngineer()
    >>> X_features = engineer.transform(X_ohlc)  # [N, 105, 4] -> [N, 105, ~30]
    >>>
    >>> pretrainer = FeatureAwareMaskedLSTMPretrainer(
    ...     ohlc_dim=4,
    ...     feature_dim=X_features.shape[-1],
    ...     feature_fusion="concat",
    ...     mask_strategy="patch",
    ...     device="cuda"
    ... )
    >>> history = pretrainer.pretrain(
    ...     X_ohlc,
    ...     X_features,
    ...     n_epochs=50,
    ...     save_path=Path("artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt")
    ... )
    >>> # Encoder saved and ready for transfer learning to enhanced SimpleLSTM
"""

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..models.feature_aware_bilstm_masked_autoencoder import (
    FeatureAwareBiLSTMMaskedAutoencoder,
    apply_feature_aware_masking,
)
from ..utils.early_stopping import EarlyStopping
from ..utils.seeds import get_device, set_seed


class FeatureAwareMaskedLSTMPretrainer:
    """Pre-trainer for feature-aware bidirectional masked LSTM autoencoder.

    Implements complete dual-input pre-training pipeline:
        1. Data loading and train/val split for dual inputs
        2. Feature-aware masked autoencoding training loop
        3. Dual reconstruction loss computation
        4. Early stopping and checkpointing
        5. Encoder weight extraction for transfer learning

    Args:
        ohlc_dim: OHLC feature dimension (4 for standard OHLC)
        feature_dim: Engineered feature dimension
        hidden_dim: LSTM hidden dimension per direction (total: 2*hidden_dim)
        num_layers: Number of stacked LSTM layers
        feature_fusion: Fusion strategy for OHLC and features ('concat', 'add', 'gate')
        mask_ratio: Proportion of timesteps to mask (0.15 = 15%)
        mask_strategy: Masking approach ("random", "block", "patch")
        patch_size: Patch size for patch masking strategy
        loss_weights: Weights for different loss components
        learning_rate: Learning rate for AdamW optimizer
        batch_size: Training batch size
        device: Device to train on ("cpu" or "cuda")
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        ohlc_dim: int = 4,
        feature_dim: int = 25,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        feature_fusion: Literal["concat", "add", "gate"] = "concat",
        mask_ratio: float = 0.15,
        mask_strategy: Literal["random", "block", "patch"] = "patch",
        patch_size: int = 7,
        loss_weights: Optional[dict] = None,
        learning_rate: float = 1e-3,
        batch_size: int = 256,  # Reduced due to dual inputs
        device: str = "cuda",
        seed: int = 1337
    ):
        set_seed(seed)
        self.device = get_device(device)
        self.seed = seed

        # Hyperparameters
        self.ohlc_dim = ohlc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.feature_fusion = feature_fusion
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.patch_size = patch_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Default loss weights
        if loss_weights is None:
            loss_weights = {
                'ohlc_weight': 0.4,
                'feature_weight': 0.4,
                'regularization_weight': 0.2
            }
        self.loss_weights = loss_weights

        # Build model
        self.model = FeatureAwareBiLSTMMaskedAutoencoder(
            ohlc_dim=ohlc_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            feature_fusion=feature_fusion
        ).to(self.device)

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )

        # LR scheduler
        self.scheduler = None  # Will be initialized in pretrain()

        # Track best model
        self.best_model_state = None

    def pretrain(
        self,
        X_ohlc: np.ndarray,
        X_features: np.ndarray,
        n_epochs: int = 50,
        val_split: float = 0.1,
        patience: int = 10,
        save_path: Optional[Path] = None,
        verbose: bool = True
    ) -> dict[str, list]:
        """Pre-train on unlabeled dual data using masked reconstruction.

        Args:
            X_ohlc: [N, seq_len, ohlc_dim] unlabeled OHLC sequences
            X_features: [N, seq_len, feature_dim] engineered features
            n_epochs: Number of training epochs
            val_split: Validation split ratio (0.1 = 10%)
            patience: Early stopping patience (epochs without improvement)
            save_path: Path to save best encoder (None = don't save)
            verbose: Print training progress

        Returns:
            history: Dictionary with training metrics
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"FEATURE-AWARE MASKED LSTM PRE-TRAINING")
            print(f"{'='*80}")
            print(f"  Dataset size: {len(X_ohlc)} samples")
            print(f"  OHLC shape: {X_ohlc.shape}")
            print(f"  Features shape: {X_features.shape}")
            print(f"  Feature fusion: {self.feature_fusion}")
            print(f"  Mask strategy: {self.mask_strategy}")
            print(f"  Mask ratio: {self.mask_ratio}")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Epochs: {n_epochs}")
            print(f"  Device: {self.device}")
            print(f"  Loss weights: {self.loss_weights}")
            print(f"{'='*80}\n")

        # Validate input shapes
        assert X_ohlc.shape[0] == X_features.shape[0], "OHLC and features must have same batch size"
        assert X_ohlc.shape[1] == X_features.shape[1], "OHLC and features must have same sequence length"
        assert X_ohlc.shape[2] == self.ohlc_dim, f"OHLC dim mismatch: expected {self.ohlc_dim}, got {X_ohlc.shape[2]}"
        assert X_features.shape[2] == self.feature_dim, f"Feature dim mismatch: expected {self.feature_dim}, got {X_features.shape[2]}"

        # Initialize LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=n_epochs,
            eta_min=1e-5
        )

        # Split train/val
        N = len(X_ohlc)
        val_size = int(N * val_split)
        indices = np.random.permutation(N)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        X_ohlc_train = torch.FloatTensor(X_ohlc[train_indices])
        X_features_train = torch.FloatTensor(X_features[train_indices])
        X_ohlc_val = torch.FloatTensor(X_ohlc[val_indices])
        X_features_val = torch.FloatTensor(X_features[val_indices])

        if verbose:
            print(f"[DATA SPLIT]")
            print(f"  Train: {len(X_ohlc_train)} samples ({(1-val_split)*100:.0f}%)")
            print(f"  Val: {len(X_ohlc_val)} samples ({val_split*100:.0f}%)")
            print()

        # Create DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_ohlc_train, X_features_train)

        # Optimized DataLoader settings
        try:
            from ..config.performance_config import get_optimized_dataloader_kwargs
            dataloader_kwargs = get_optimized_dataloader_kwargs(is_training=True)
        except ImportError:
            dataloader_kwargs = {
                'num_workers': 8 if self.device.type == "cuda" else 0,
                'pin_memory': True if self.device.type == "cuda" else False,
                'prefetch_factor': 2 if self.device.type == "cuda" else None,
                'persistent_workers': True if self.device.type == "cuda" else False,
            }
            dataloader_kwargs = {k: v for k, v in dataloader_kwargs.items() if v is not None}

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            **dataloader_kwargs
        )

        # Early stopping
        early_stopping = EarlyStopping(
            patience=patience,
            mode="min",
            verbose=verbose
        )

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_ohlc_recon': [],
            'val_ohlc_recon': [],
            'train_feature_recon': [],
            'val_feature_recon': [],
            'learning_rate': []
        }

        # Setup AMP scaler
        use_amp = self.device.type == "cuda" and torch.cuda.is_available()
        scaler = None
        if use_amp:
            try:
                from ..config.performance_config import get_amp_scaler
                scaler = get_amp_scaler()
                if scaler and verbose:
                    print("[PERFORMANCE] Using automatic mixed precision (AMP) for 1.5-2× speedup")
            except ImportError:
                scaler = torch.amp.GradScaler('cuda')
                if verbose:
                    print("[PERFORMANCE] Using AMP with default settings")

        # Training loop
        for epoch in range(n_epochs):
            # ============================================================
            # TRAINING PHASE
            # ============================================================
            self.model.train()
            train_losses = []
            train_ohlc_recon_losses = []
            train_feature_recon_losses = []

            # Progress bar
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{n_epochs}",
                disable=not verbose
            )

            for batch_ohlc, batch_features in pbar:
                batch_ohlc = batch_ohlc.to(self.device, non_blocking=True)
                batch_features = batch_features.to(self.device, non_blocking=True)

                # Apply feature-aware masking
                ohlc_masked, features_masked, ohlc_mask, feature_mask = apply_feature_aware_masking(
                    batch_ohlc,
                    batch_features,
                    self.model.ohlc_mask_token,
                    self.model.feature_mask_token,
                    mask_strategy=self.mask_strategy,
                    mask_ratio=self.mask_ratio,
                    patch_size=self.patch_size
                )

                self.optimizer.zero_grad()

                # Forward pass with optional AMP
                if scaler:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        ohlc_recon, feature_recon = self.model(ohlc_masked, features_masked)
                        loss, loss_dict = self.model.compute_loss(
                            ohlc_recon, feature_recon,
                            batch_ohlc, batch_features,
                            ohlc_mask, feature_mask,
                            self.loss_weights
                        )

                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    ohlc_recon, feature_recon = self.model(ohlc_masked, features_masked)
                    loss, loss_dict = self.model.compute_loss(
                        ohlc_recon, feature_recon,
                        batch_ohlc, batch_features,
                        ohlc_mask, feature_mask,
                        self.loss_weights
                    )

                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                # Track metrics
                train_losses.append(loss_dict['total'])
                train_ohlc_recon_losses.append(loss_dict['ohlc_reconstruction'])
                train_feature_recon_losses.append(loss_dict['feature_reconstruction'])

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'ohlc': f"{loss_dict['ohlc_reconstruction']:.4f}",
                    'feat': f"{loss_dict['feature_reconstruction']:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
                })

            # Update learning rate
            self.scheduler.step()

            # ============================================================
            # VALIDATION PHASE
            # ============================================================
            self.model.eval()
            val_losses = []
            val_ohlc_recon_losses = []
            val_feature_recon_losses = []

            with torch.no_grad():
                # Process validation data in batches
                for i in range(0, len(X_ohlc_val), self.batch_size):
                    batch_ohlc = X_ohlc_val[i:i+self.batch_size].to(self.device, non_blocking=True)
                    batch_features = X_features_val[i:i+self.batch_size].to(self.device, non_blocking=True)

                    # Apply masking
                    ohlc_masked, features_masked, ohlc_mask, feature_mask = apply_feature_aware_masking(
                        batch_ohlc,
                        batch_features,
                        self.model.ohlc_mask_token,
                        self.model.feature_mask_token,
                        mask_strategy=self.mask_strategy,
                        mask_ratio=self.mask_ratio,
                        patch_size=self.patch_size
                    )

                    # Forward pass with optional AMP
                    if scaler:
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            ohlc_recon, feature_recon = self.model(ohlc_masked, features_masked)
                            loss, loss_dict = self.model.compute_loss(
                                ohlc_recon, feature_recon,
                                batch_ohlc, batch_features,
                                ohlc_mask, feature_mask,
                                self.loss_weights
                            )
                    else:
                        ohlc_recon, feature_recon = self.model(ohlc_masked, features_masked)
                        loss, loss_dict = self.model.compute_loss(
                            ohlc_recon, feature_recon,
                            batch_ohlc, batch_features,
                            ohlc_mask, feature_mask,
                            self.loss_weights
                        )

                    val_losses.append(loss_dict['total'])
                    val_ohlc_recon_losses.append(loss_dict['ohlc_reconstruction'])
                    val_feature_recon_losses.append(loss_dict['feature_reconstruction'])

            # ============================================================
            # LOGGING AND CHECKPOINTING
            # ============================================================
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            avg_train_ohlc_recon = np.mean(train_ohlc_recon_losses)
            avg_val_ohlc_recon = np.mean(val_ohlc_recon_losses)
            avg_train_feature_recon = np.mean(train_feature_recon_losses)
            avg_val_feature_recon = np.mean(val_feature_recon_losses)
            current_lr = self.scheduler.get_last_lr()[0]

            # Record history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_ohlc_recon'].append(avg_train_ohlc_recon)
            history['val_ohlc_recon'].append(avg_val_ohlc_recon)
            history['train_feature_recon'].append(avg_train_feature_recon)
            history['val_feature_recon'].append(avg_val_feature_recon)
            history['learning_rate'].append(current_lr)

            # Print epoch summary
            if verbose:
                print(f"\nEpoch [{epoch+1}/{n_epochs}]")
                print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                print(f"  Train OHLC: {avg_train_ohlc_recon:.4f} | Val OHLC: {avg_val_ohlc_recon:.4f}")
                print(f"  Train Features: {avg_train_feature_recon:.4f} | Val Features: {avg_val_feature_recon:.4f}")
                print(f"  LR: {current_lr:.6f}")

            # Early stopping check
            if early_stopping(avg_val_loss, self.model):
                if verbose:
                    print(f"\n[EARLY STOPPING] Triggered at epoch {epoch+1}")
                break

        # ============================================================
        # RESTORE BEST MODEL AND SAVE
        # ============================================================
        early_stopping.load_best_model(self.model)

        if verbose:
            print(f"\n{'='*80}")
            print(f"FEATURE-AWARE PRE-TRAINING COMPLETE")
            print(f"{'='*80}")
            print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
            print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
            print(f"  Best val loss: {min(history['val_loss']):.4f}")
            print(f"  Final OHLC recon: {history['val_ohlc_recon'][-1]:.4f}")
            print(f"  Final Feature recon: {history['val_feature_recon'][-1]:.4f}")

        # Save encoder if path provided
        if save_path is not None:
            self.save_encoder(save_path)
            if verbose:
                print(f"  Encoder saved: {save_path}")

        if verbose:
            print(f"{'='*80}\n")

        return history

    def save_encoder(self, path: Path) -> None:
        """Save encoder weights and hyperparameters for transfer learning.

        Args:
            path: Path to save encoder checkpoint (.pt file)
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'encoder_state_dict': self.model.encoder_lstm.state_dict(),
            'hyperparams': {
                'ohlc_dim': self.ohlc_dim,
                'feature_dim': self.feature_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'feature_fusion': self.feature_fusion,
            },
            'ohlc_mask_token': self.model.ohlc_mask_token.data,
            'feature_mask_token': self.model.feature_mask_token.data,
            'training_config': {
                'mask_strategy': self.mask_strategy,
                'mask_ratio': self.mask_ratio,
                'patch_size': self.patch_size,
                'loss_weights': self.loss_weights,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
            }
        }

        torch.save(checkpoint, path)

    def load_encoder(self, path: Path) -> None:
        """Load pre-trained encoder weights.

        Args:
            path: Path to encoder checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Restore encoder weights
        self.model.encoder_lstm.load_state_dict(checkpoint['encoder_state_dict'])

        # Restore mask tokens
        self.model.ohlc_mask_token.data = checkpoint['ohlc_mask_token']
        self.model.feature_mask_token.data = checkpoint['feature_mask_token']

        print(f"[FEATURE-AWARE PRETRAINING] Loaded encoder from {path}")


def visualize_feature_aware_reconstruction(
    model: FeatureAwareBiLSTMMaskedAutoencoder,
    ohlc_sample: torch.Tensor,
    feature_sample: torch.Tensor,
    mask_strategy: str = "random",
    mask_ratio: float = 0.15,
    device: str = "cuda"
) -> dict[str, np.ndarray]:
    """Visualize feature-aware masked reconstruction quality.

    Args:
        model: Trained FeatureAwareBiLSTMMaskedAutoencoder
        ohlc_sample: [1, seq_len, ohlc_dim] single OHLC sample
        feature_sample: [1, seq_len, feature_dim] single feature sample
        mask_strategy: Masking strategy to use
        mask_ratio: Masking ratio
        device: Device for computation

    Returns:
        Dictionary with original, masked, reconstructed arrays and masks
    """
    model.eval()
    ohlc_sample = ohlc_sample.to(device)
    feature_sample = feature_sample.to(device)

    with torch.no_grad():
        # Apply masking
        ohlc_masked, features_masked, ohlc_mask, feature_mask = apply_feature_aware_masking(
            ohlc_sample,
            feature_sample,
            model.ohlc_mask_token,
            model.feature_mask_token,
            mask_strategy=mask_strategy,
            mask_ratio=mask_ratio
        )

        # Reconstruct
        ohlc_recon, feature_recon = model(ohlc_masked, features_masked)

    return {
        'ohlc_original': ohlc_sample.cpu().numpy(),
        'ohlc_masked': ohlc_masked.cpu().numpy(),
        'ohlc_reconstructed': ohlc_recon.cpu().numpy(),
        'features_original': feature_sample.cpu().numpy(),
        'features_masked': features_masked.cpu().numpy(),
        'features_reconstructed': feature_recon.cpu().numpy(),
        'ohlc_mask': ohlc_mask.cpu().numpy(),
        'feature_mask': feature_mask.cpu().numpy()
    }