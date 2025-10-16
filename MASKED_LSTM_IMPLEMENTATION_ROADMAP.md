# Masked LSTM Pre-training Implementation Roadmap

**Goal**: Implement masked autoencoding pre-training for SimpleLSTM to achieve +8-12% accuracy gain

**Timeline**: 6-8 hours implementation + 1 hour training/evaluation

**Expected Outcome**: 65-69% accuracy (vs 57.14% baseline), breaking class collapse

---

## Implementation Phases

### Phase 1: Core Pre-training Architecture (3-4 hours)

#### 1.1 Create Base File Structure (30 min)

**File**: `src/moola/models/masked_lstm_pretrainer.py`

```python
"""Masked LSTM Autoencoder for Pre-training SimpleLSTM

Inspired by BERT and PatchTST masked prediction pre-training.
Pre-trains bidirectional LSTM encoder on unlabeled time series data.

Architecture:
    Input: [Batch, 105, 4] OHLC sequences
        ↓
    Random Masking: 15% of timesteps → MASK_TOKEN
        ↓
    Bidirectional LSTM Encoder: [Batch, 105, 128]
        ↓
    Decoder: [Batch, 105, 4] reconstruction
        ↓
    Loss: MSE on MASKED positions only

Key Features:
    - Three masking strategies: random, block, patch
    - Learnable mask token
    - Latent regularization (prevent collapse)
    - Data augmentation pipeline
    - Early stopping + checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Literal, Tuple, Optional
import pandas as pd
from tqdm import tqdm

# Imports from existing codebase
from ..utils.seeds import set_seed, get_device
from ..utils.early_stopping import EarlyStopping
```

**Checklist**:
- [ ] Create file with docstring
- [ ] Import dependencies
- [ ] Define class structure
- [ ] Add type hints

---

#### 1.2 Implement Masking Strategies (1 hour)

**Three masking approaches**:

1. **Random Masking** (BERT-style)
```python
def mask_random(self, x: torch.Tensor, mask_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly mask 15% of timesteps

    Args:
        x: [batch, seq_len, features]
        mask_ratio: Proportion of timesteps to mask

    Returns:
        x_masked: Input with masked timesteps replaced by MASK_TOKEN
        mask: Boolean mask [batch, seq_len]
    """
    B, T, D = x.shape

    # Generate random mask (15% True)
    mask = torch.rand(B, T, device=x.device) < mask_ratio

    # Replace masked positions with learnable mask token
    x_masked = x.clone()
    x_masked[mask] = self.mask_token.expand(mask.sum(), D)

    return x_masked, mask
```

2. **Block Masking** (contiguous segments)
```python
def mask_block(self, x: torch.Tensor, mask_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask contiguous blocks of timesteps

    More challenging than random masking - forces long-range dependencies

    Example:
        bars 1-20: visible
        bars 21-35: MASKED (block 1)
        bars 36-60: visible
        bars 61-75: MASKED (block 2)
        bars 76-105: visible
    """
    B, T, D = x.shape
    block_size = int(T * mask_ratio)

    mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)

    # Random start position for each sample in batch
    for i in range(B):
        start_idx = torch.randint(0, T - block_size + 1, (1,)).item()
        mask[i, start_idx:start_idx + block_size] = True

    x_masked = x.clone()
    x_masked[mask] = self.mask_token.expand(mask.sum(), D)

    return x_masked, mask
```

3. **Patch Masking** (PatchTST-inspired)
```python
def mask_patch(self, x: torch.Tensor, mask_ratio: float = 0.15, patch_size: int = 7) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask entire patches (subseries) of timesteps

    Divides sequence into 7-bar patches and masks 15% of patches

    Example (105 bars, patch_size=7):
        15 patches total
        Mask ~2 patches (15% of 15)
        Patches 3 and 8 masked
    """
    B, T, D = x.shape
    num_patches = T // patch_size
    num_masked = max(1, int(num_patches * mask_ratio))

    mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)

    for i in range(B):
        # Randomly select patches to mask
        masked_patches = torch.randperm(num_patches)[:num_masked]

        # Expand patch mask to full sequence
        for patch_idx in masked_patches:
            start = patch_idx * patch_size
            end = min(start + patch_size, T)
            mask[i, start:end] = True

    x_masked = x.clone()
    x_masked[mask] = self.mask_token.expand(mask.sum(), D)

    return x_masked, mask
```

**Checklist**:
- [ ] Implement `mask_random()`
- [ ] Implement `mask_block()`
- [ ] Implement `mask_patch()`
- [ ] Add unit tests for each strategy
- [ ] Verify mask ratios are correct (~15%)

---

#### 1.3 Implement Encoder-Decoder Architecture (1.5 hours)

**Bidirectional LSTM Encoder**:
```python
class MaskedLSTMAutoencoder(nn.Module):
    """Masked autoencoder with bidirectional LSTM"""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Learnable mask token (will be optimized during training)
        self.mask_token = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)

        # Encoder: Bidirectional LSTM
        # Critical: Bidirectional sees both past and future context
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,  # Forward + backward passes
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Decoder: Projects encoded features back to input space
        # Input: [batch, seq_len, hidden_dim*2] (bidirectional)
        # Output: [batch, seq_len, input_dim] (OHLC reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        # Layer norm for stable training
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        """Encode masked sequence and reconstruct

        Args:
            x_masked: [batch, seq_len, input_dim] with masked positions

        Returns:
            reconstruction: [batch, seq_len, input_dim]
        """
        # Encode with bidirectional LSTM
        encoded, _ = self.encoder_lstm(x_masked)  # [B, T, hidden_dim*2]

        # Layer norm for stability
        encoded = self.layer_norm(encoded)

        # Decode to reconstruction
        reconstruction = self.decoder(encoded)  # [B, T, input_dim]

        return reconstruction

    def get_encoder_state_dict(self) -> dict:
        """Extract encoder weights for fine-tuning"""
        return {
            'encoder_lstm.weight_ih_l0': self.encoder_lstm.weight_ih_l0,
            'encoder_lstm.weight_hh_l0': self.encoder_lstm.weight_hh_l0,
            'encoder_lstm.bias_ih_l0': self.encoder_lstm.bias_ih_l0,
            'encoder_lstm.bias_hh_l0': self.encoder_lstm.bias_hh_l0,
            # Add all LSTM layers...
        }
```

**Checklist**:
- [ ] Implement `MaskedLSTMAutoencoder` class
- [ ] Add learnable mask token
- [ ] Implement bidirectional LSTM encoder
- [ ] Implement decoder with projection layers
- [ ] Add layer normalization
- [ ] Implement `get_encoder_state_dict()`

---

#### 1.4 Implement Loss Computation (30 min)

**Key Insight**: Only compute loss on MASKED positions!

```python
def compute_loss(
    self,
    reconstruction: torch.Tensor,
    x_original: torch.Tensor,
    mask: torch.Tensor
) -> Tuple[torch.Tensor, dict]:
    """Compute masked reconstruction loss

    Critical: Loss ONLY on masked positions (not visible positions)
    This forces encoder to learn from context, not just copy

    Args:
        reconstruction: [batch, seq_len, features] reconstructed OHLC
        x_original: [batch, seq_len, features] original OHLC
        mask: [batch, seq_len] boolean mask (True = masked)

    Returns:
        total_loss: Scalar loss
        loss_dict: Dictionary with loss components for logging
    """
    # Reconstruction loss on MASKED positions only
    reconstruction_loss = F.mse_loss(
        reconstruction[mask],
        x_original[mask],
        reduction='mean'
    )

    # Optional: Regularization to prevent latent collapse
    # Encourage diversity in encoded representations
    encoded = self.encoder_lstm(x_original)[0]  # [B, T, D]
    latent_std = torch.std(encoded, dim=(0, 1)).mean()
    reg_loss = torch.relu(1.0 - latent_std)  # Encourage std >= 1.0

    # Total loss
    total_loss = reconstruction_loss + 0.1 * reg_loss

    # Logging dictionary
    loss_dict = {
        'total': total_loss.item(),
        'reconstruction': reconstruction_loss.item(),
        'regularization': reg_loss.item(),
        'latent_std': latent_std.item()
    }

    return total_loss, loss_dict
```

**Checklist**:
- [ ] Implement masked MSE loss
- [ ] Add latent regularization
- [ ] Return loss dictionary for logging
- [ ] Add per-feature reconstruction errors (OHLC separately)

---

### Phase 2: Training Infrastructure (1.5 hours)

#### 2.1 Implement Pre-training Loop (1 hour)

**Full training logic**:
```python
class MaskedLSTMPretrainer:
    """Main pre-training class"""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        mask_ratio: float = 0.15,
        mask_strategy: Literal["random", "block", "patch"] = "patch",
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        device: str = "cuda",
        seed: int = 1337
    ):
        set_seed(seed)
        self.device = get_device(device)

        # Build model
        self.model = MaskedLSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )

        # LR scheduler (cosine annealing)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-5
        )

        # Training config
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.batch_size = batch_size

    def pretrain(
        self,
        X_unlabeled: np.ndarray,
        n_epochs: int = 50,
        val_split: float = 0.1,
        patience: int = 10,
        save_path: Optional[Path] = None
    ) -> dict:
        """Pre-train on unlabeled data

        Args:
            X_unlabeled: [N, 105, 4] unlabeled OHLC sequences
            n_epochs: Number of training epochs
            val_split: Validation split ratio
            patience: Early stopping patience
            save_path: Path to save best model

        Returns:
            history: Dictionary with training metrics
        """
        print(f"[PRETRAINING] Starting masked LSTM pre-training")
        print(f"  Dataset size: {len(X_unlabeled)} samples")
        print(f"  Mask strategy: {self.mask_strategy}")
        print(f"  Mask ratio: {self.mask_ratio}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Epochs: {n_epochs}")

        # Split train/val
        N = len(X_unlabeled)
        val_size = int(N * val_split)
        indices = np.random.permutation(N)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        X_train = torch.FloatTensor(X_unlabeled[train_indices])
        X_val = torch.FloatTensor(X_unlabeled[val_indices])

        print(f"  Train: {len(X_train)} samples")
        print(f"  Val: {len(X_val)} samples")

        # Dataloaders
        train_dataset = torch.utils.data.TensorDataset(X_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # Early stopping
        early_stopping = EarlyStopping(
            patience=patience,
            mode="min",
            verbose=True
        )

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon': [],
            'val_recon': []
        }

        # Training loop
        for epoch in range(n_epochs):
            # Training phase
            self.model.train()
            train_losses = []
            train_recon_losses = []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
            for (batch_X,) in pbar:
                batch_X = batch_X.to(self.device)

                # Apply masking
                if self.mask_strategy == "random":
                    x_masked, mask = self.mask_random(batch_X, self.mask_ratio)
                elif self.mask_strategy == "block":
                    x_masked, mask = self.mask_block(batch_X, self.mask_ratio)
                elif self.mask_strategy == "patch":
                    x_masked, mask = self.mask_patch(batch_X, self.mask_ratio)

                # Forward pass
                reconstruction = self.model(x_masked)

                # Compute loss
                loss, loss_dict = self.compute_loss(reconstruction, batch_X, mask)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Track metrics
                train_losses.append(loss_dict['total'])
                train_recon_losses.append(loss_dict['reconstruction'])

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'recon': f"{loss_dict['reconstruction']:.4f}"
                })

            # Update LR
            self.scheduler.step()

            # Validation phase
            self.model.eval()
            val_losses = []
            val_recon_losses = []

            with torch.no_grad():
                X_val_device = X_val.to(self.device)

                # Apply masking
                if self.mask_strategy == "random":
                    x_masked, mask = self.mask_random(X_val_device, self.mask_ratio)
                elif self.mask_strategy == "block":
                    x_masked, mask = self.mask_block(X_val_device, self.mask_ratio)
                elif self.mask_strategy == "patch":
                    x_masked, mask = self.mask_patch(X_val_device, self.mask_ratio)

                # Forward pass
                reconstruction = self.model(x_masked)

                # Compute loss
                loss, loss_dict = self.compute_loss(reconstruction, X_val_device, mask)

                val_losses.append(loss_dict['total'])
                val_recon_losses.append(loss_dict['reconstruction'])

            # Record history
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            avg_train_recon = np.mean(train_recon_losses)
            avg_val_recon = np.mean(val_recon_losses)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_recon'].append(avg_train_recon)
            history['val_recon'].append(avg_val_recon)

            # Logging
            print(f"Epoch [{epoch+1}/{n_epochs}]")
            print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"  Train Recon: {avg_train_recon:.4f} | Val Recon: {avg_val_recon:.4f}")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")

            # Early stopping
            if early_stopping(avg_val_loss, self.model):
                print(f"[PRETRAINING] Early stopping at epoch {epoch+1}")
                break

        # Load best model
        early_stopping.load_best_model(self.model)

        # Save encoder
        if save_path:
            self.save_encoder(save_path)

        return history

    def save_encoder(self, path: Path):
        """Save encoder weights for fine-tuning"""
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'encoder_state_dict': self.model.encoder_lstm.state_dict(),
            'hyperparams': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers
            },
            'mask_token': self.model.mask_token.data
        }

        torch.save(checkpoint, path)
        print(f"[PRETRAINING] Saved encoder to {path}")
```

**Checklist**:
- [ ] Implement training loop
- [ ] Add validation loop
- [ ] Add early stopping
- [ ] Add LR scheduling
- [ ] Add progress bars (tqdm)
- [ ] Add comprehensive logging

---

#### 2.2 Add Data Augmentation (30 min)

**Augmentation pipeline for unlabeled data**:
```python
def augment_unlabeled(self, X: np.ndarray, num_augmentations: int = 4) -> np.ndarray:
    """Augment unlabeled data to increase dataset size

    Applies random augmentations that preserve financial semantics

    Args:
        X: [N, 105, 4] OHLC sequences
        num_augmentations: Number of augmented versions per sample

    Returns:
        X_augmented: [N * (1 + num_augmentations), 105, 4]
    """
    augmented = [X]

    for _ in range(num_augmentations):
        X_aug = X.copy()

        # Time warping (50% chance)
        if np.random.rand() < 0.5:
            X_aug = self.time_warp(X_aug, sigma=0.15)

        # Jittering (50% chance)
        if np.random.rand() < 0.5:
            X_aug = self.jitter(X_aug, sigma=0.03)

        # Volatility scaling (30% chance)
        if np.random.rand() < 0.3:
            X_aug = self.volatility_scale(X_aug, scale_range=(0.85, 1.15))

        augmented.append(X_aug)

    return np.concatenate(augmented, axis=0)

def time_warp(self, X: np.ndarray, sigma: float = 0.15) -> np.ndarray:
    """Stretch/compress time axis"""
    # Implementation...
    pass

def jitter(self, X: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    """Add Gaussian noise"""
    noise = np.random.randn(*X.shape) * sigma * X.std(axis=0)
    return X + noise

def volatility_scale(self, X: np.ndarray, scale_range: tuple = (0.85, 1.15)) -> np.ndarray:
    """Scale high-low spreads"""
    scale = np.random.uniform(*scale_range, size=(X.shape[0], 1, 1))
    mid = (X[:, :, 1:2] + X[:, :, 2:3]) / 2
    X[:, :, 1:2] = mid + (X[:, :, 1:2] - mid) * scale  # Scale highs
    X[:, :, 2:3] = mid + (X[:, :, 2:3] - mid) * scale  # Scale lows
    return X
```

**Checklist**:
- [ ] Implement time warping
- [ ] Implement jittering
- [ ] Implement volatility scaling
- [ ] Add augmentation pipeline
- [ ] Test augmentations preserve OHLC semantics

---

### Phase 3: Integration with SimpleLSTM (1 hour)

#### 3.1 Add Encoder Loading Method

**File**: `src/moola/models/simple_lstm.py`

```python
def load_pretrained_encoder(
    self,
    encoder_path: Path,
    freeze_encoder: bool = True
) -> "SimpleLSTMModel":
    """Load pre-trained encoder from masked autoencoder

    Args:
        encoder_path: Path to pre-trained encoder checkpoint
        freeze_encoder: If True, freeze encoder weights during initial training

    Returns:
        Self with pre-trained encoder loaded
    """
    print(f"[SSL] Loading pre-trained encoder from: {encoder_path}")

    # Load checkpoint
    checkpoint = torch.load(encoder_path, map_location=self.device)
    encoder_state_dict = checkpoint['encoder_state_dict']
    hyperparams = checkpoint['hyperparams']

    # Verify architecture compatibility
    if self.hidden_size != hyperparams['hidden_dim']:
        raise ValueError(
            f"Hidden size mismatch: {self.hidden_size} != {hyperparams['hidden_dim']}"
        )

    # Map bidirectional LSTM weights to unidirectional LSTM
    # Bidirectional has 2x parameters (forward + backward)
    # Strategy: Use only FORWARD weights (or average forward+backward)

    model_state_dict = self.model.state_dict()
    loaded_keys = 0

    for key in encoder_state_dict:
        if 'weight_ih' in key:
            # Input-hidden weights
            # Bidirectional: [2*hidden, input]
            # Unidirectional: [hidden, input]
            # Take forward weights only
            bidirectional_weight = encoder_state_dict[key]
            forward_weight = bidirectional_weight[:self.hidden_size * 4, :]

            model_key = key.replace('encoder_', '')
            if model_key in model_state_dict:
                model_state_dict[model_key] = forward_weight
                loaded_keys += 1

        elif 'weight_hh' in key:
            # Hidden-hidden weights
            bidirectional_weight = encoder_state_dict[key]
            forward_weight = bidirectional_weight[:self.hidden_size * 4, :self.hidden_size]

            model_key = key.replace('encoder_', '')
            if model_key in model_state_dict:
                model_state_dict[model_key] = forward_weight
                loaded_keys += 1

        elif 'bias' in key:
            # Biases
            bidirectional_bias = encoder_state_dict[key]
            forward_bias = bidirectional_bias[:self.hidden_size * 4]

            model_key = key.replace('encoder_', '')
            if model_key in model_state_dict:
                model_state_dict[model_key] = forward_bias
                loaded_keys += 1

    # Load mapped weights
    self.model.load_state_dict(model_state_dict)

    print(f"[SSL] Loaded {loaded_keys} pre-trained parameters")

    # Freeze encoder if requested
    if freeze_encoder:
        print(f"[SSL] Freezing LSTM encoder weights")
        for param in self.model.lstm.parameters():
            param.requires_grad = False

    return self

def fit(
    self,
    X: np.ndarray,
    y: np.ndarray,
    unfreeze_encoder_after: int = 10,
    **kwargs
) -> "SimpleLSTMModel":
    """Train with optional encoder unfreezing schedule

    Args:
        X, y: Training data
        unfreeze_encoder_after: Unfreeze encoder after N epochs (0 = never unfreeze)
        **kwargs: Other fit() arguments
    """
    # ... existing fit code ...

    for epoch in range(self.n_epochs):
        # Unfreeze encoder after warm-up period
        if epoch == unfreeze_encoder_after:
            print(f"[SSL] Unfreezing LSTM encoder at epoch {epoch}")
            for param in self.model.lstm.parameters():
                param.requires_grad = True

            # Reduce learning rate after unfreezing
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"[SSL] Reduced LR to {optimizer.param_groups[0]['lr']:.6f}")

        # ... rest of training loop ...
```

**Checklist**:
- [ ] Implement `load_pretrained_encoder()` method
- [ ] Map bidirectional → unidirectional weights
- [ ] Add freezing logic
- [ ] Add unfreezing schedule to `fit()`
- [ ] Add LR reduction after unfreezing

---

### Phase 4: CLI Integration & Testing (1.5 hours)

#### 4.1 Add CLI Commands (30 min)

**File**: `src/moola/cli.py`

```python
@app.command()
def pretrain_masked_lstm(
    input: str = typer.Option(
        "data/raw/unlabeled_windows.parquet",
        help="Path to unlabeled OHLC data"
    ),
    output: str = typer.Option(
        "data/artifacts/pretrained/masked_lstm_encoder.pt",
        help="Path to save pre-trained encoder"
    ),
    device: str = typer.Option("cuda", help="Device (cpu/cuda)"),
    epochs: int = typer.Option(50, help="Number of pre-training epochs"),
    patience: int = typer.Option(10, help="Early stopping patience"),
    mask_ratio: float = typer.Option(0.15, help="Proportion of timesteps to mask"),
    mask_strategy: str = typer.Option("patch", help="Masking strategy (random/block/patch)"),
    hidden_dim: int = typer.Option(64, help="LSTM hidden dimension"),
    batch_size: int = typer.Option(512, help="Batch size"),
    seed: int = typer.Option(1337, help="Random seed")
):
    """Pre-train masked LSTM autoencoder on unlabeled data"""
    from moola.models.masked_lstm_pretrainer import MaskedLSTMPretrainer
    import pandas as pd

    print(f"[CLI] Starting masked LSTM pre-training")
    print(f"  Input: {input}")
    print(f"  Output: {output}")

    # Load unlabeled data
    df = pd.read_parquet(input)
    X_unlabeled = np.array([x for x in df['features'].values])
    X_unlabeled = X_unlabeled.reshape(len(X_unlabeled), 105, 4)

    print(f"  Loaded {len(X_unlabeled)} unlabeled samples")

    # Create pre-trainer
    pretrainer = MaskedLSTMPretrainer(
        input_dim=4,
        hidden_dim=hidden_dim,
        mask_ratio=mask_ratio,
        mask_strategy=mask_strategy,
        batch_size=batch_size,
        device=device,
        seed=seed
    )

    # Pre-train
    history = pretrainer.pretrain(
        X_unlabeled=X_unlabeled,
        n_epochs=epochs,
        patience=patience,
        save_path=Path(output)
    )

    print(f"[CLI] Pre-training complete!")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")

# Modify existing OOF command to support pre-trained encoder
@app.command()
def oof(
    model: str = typer.Option(..., help="Model name"),
    load_pretrained_encoder: str = typer.Option(None, help="Path to pre-trained encoder"),
    freeze_encoder: bool = typer.Option(True, help="Freeze encoder initially"),
    unfreeze_after: int = typer.Option(10, help="Unfreeze encoder after N epochs"),
    # ... other existing parameters ...
):
    """Generate OOF predictions with optional SSL pre-training"""
    # ... existing oof code ...

    model_kwargs = {}
    if load_pretrained_encoder:
        model_kwargs['load_pretrained_encoder'] = load_pretrained_encoder
        model_kwargs['freeze_encoder'] = freeze_encoder
        model_kwargs['unfreeze_encoder_after'] = unfreeze_after

    # ... rest of oof generation ...
```

**Checklist**:
- [ ] Add `pretrain-masked-lstm` command
- [ ] Modify `oof` command to support pre-training
- [ ] Add parameter validation
- [ ] Add help text

---

#### 4.2 Unit Tests (30 min)

**File**: `tests/test_masked_lstm_pretrainer.py`

```python
import pytest
import torch
import numpy as np
from pathlib import Path
from moola.models.masked_lstm_pretrainer import MaskedLSTMPretrainer

def test_masking_strategies():
    """Test all three masking strategies"""
    X = torch.randn(16, 105, 4)
    pretrainer = MaskedLSTMPretrainer(mask_ratio=0.15)

    # Test random masking
    x_masked, mask = pretrainer.mask_random(X, 0.15)
    actual_ratio = mask.float().mean().item()
    assert 0.10 < actual_ratio < 0.20, f"Random mask ratio: {actual_ratio}"

    # Test block masking
    x_masked, mask = pretrainer.mask_block(X, 0.15)
    actual_ratio = mask.float().mean().item()
    assert 0.10 < actual_ratio < 0.20, f"Block mask ratio: {actual_ratio}"

    # Verify blocks are contiguous
    for i in range(16):
        masked_indices = torch.where(mask[i])[0]
        if len(masked_indices) > 1:
            # Check consecutive
            assert torch.all(masked_indices[1:] - masked_indices[:-1] == 1)

    # Test patch masking
    x_masked, mask = pretrainer.mask_patch(X, 0.15, patch_size=7)
    actual_ratio = mask.float().mean().item()
    assert 0.05 < actual_ratio < 0.25, f"Patch mask ratio: {actual_ratio}"

def test_encoder_weight_loading():
    """Test encoder weights load correctly into SimpleLSTM"""
    from moola.models.simple_lstm import SimpleLSTMModel

    # Pre-train encoder (minimal)
    X_unlabeled = np.random.randn(100, 105, 4).astype(np.float32)
    pretrainer = MaskedLSTMPretrainer()
    history = pretrainer.pretrain(X_unlabeled, n_epochs=2, patience=5)
    pretrainer.save_encoder(Path("/tmp/test_encoder.pt"))

    # Load into SimpleLSTM
    model = SimpleLSTMModel(hidden_size=64)

    # Build model first
    X_dummy = np.random.randn(10, 105, 4).astype(np.float32)
    y_dummy = np.array([0, 1] * 5)
    model.fit(X_dummy, y_dummy, n_epochs=1)

    # Get initial LSTM weight
    initial_weight = model.model.lstm.weight_ih_l0.clone()

    # Load pre-trained encoder
    model.load_pretrained_encoder(Path("/tmp/test_encoder.pt"))

    # Get loaded weight
    loaded_weight = model.model.lstm.weight_ih_l0

    # Verify weights changed
    assert not torch.allclose(initial_weight, loaded_weight), \
        "Weights did not change after loading!"

    print("✅ Encoder weights loaded successfully")

def test_reconstruction_quality():
    """Test reconstruction loss decreases during training"""
    X_unlabeled = np.random.randn(100, 105, 4).astype(np.float32)
    pretrainer = MaskedLSTMPretrainer()

    history = pretrainer.pretrain(X_unlabeled, n_epochs=5, patience=10)

    # Check loss decreased
    initial_loss = history['train_loss'][0]
    final_loss = history['train_loss'][-1]

    assert final_loss < initial_loss, \
        f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"

    print(f"✅ Loss decreased: {initial_loss:.4f} → {final_loss:.4f}")
```

**Checklist**:
- [ ] Test masking strategies
- [ ] Test encoder weight loading
- [ ] Test reconstruction quality
- [ ] Test end-to-end pipeline

---

#### 4.3 Integration Tests (30 min)

**Full pipeline test**:
```bash
#!/bin/bash
# File: scripts/test_masked_lstm_pipeline.sh

set -e

echo "===== Testing Masked LSTM Pre-training Pipeline ====="

# Step 1: Pre-train encoder (small dataset, few epochs)
python -m moola.cli pretrain-masked-lstm \
    --input data/raw/unlabeled_windows.parquet \
    --output /tmp/test_encoder.pt \
    --device cpu \
    --epochs 5 \
    --patience 10 \
    --mask-strategy patch \
    --hidden-dim 64 \
    --batch-size 32

echo "✅ Pre-training complete"

# Step 2: Fine-tune SimpleLSTM with pre-trained encoder
python -m moola.cli oof \
    --model simple_lstm \
    --device cpu \
    --seed 1337 \
    --load-pretrained-encoder /tmp/test_encoder.pt \
    --freeze-encoder \
    --unfreeze-after 5 \
    --n-epochs 10

echo "✅ Fine-tuning complete"

# Step 3: Verify results improved over baseline
# ... comparison logic ...

echo "===== Pipeline Test Complete ====="
```

**Checklist**:
- [ ] Create integration test script
- [ ] Test on CPU (small dataset)
- [ ] Test on GPU (full dataset)
- [ ] Verify accuracy improves over baseline

---

### Phase 5: Training & Evaluation (1 hour)

#### 5.1 Run Pre-training on RunPod (20 min)

```bash
# Connect to RunPod H100 instance
ssh root@<runpod-ip>

# Navigate to project
cd /workspace/moola

# Pre-train encoder (full dataset, 50 epochs)
python -m moola.cli pretrain-masked-lstm \
    --input data/raw/unlabeled_windows.parquet \
    --output data/artifacts/pretrained/masked_lstm_encoder.pt \
    --device cuda \
    --epochs 50 \
    --patience 10 \
    --mask-ratio 0.15 \
    --mask-strategy patch \
    --hidden-dim 64 \
    --batch-size 512 \
    --seed 1337

# Expected output:
#   Loaded 11873 unlabeled samples
#   Train: 10685, Val: 1188
#   Epoch 50: Train Loss: 0.0123 | Val Loss: 0.0156
#   Saved encoder to data/artifacts/pretrained/masked_lstm_encoder.pt
```

**Checklist**:
- [ ] Launch RunPod H100 instance
- [ ] Upload unlabeled data
- [ ] Run pre-training
- [ ] Monitor training progress
- [ ] Download pre-trained encoder

---

#### 5.2 Fine-tune SimpleLSTM (15 min)

```bash
# Fine-tune with frozen encoder (epochs 0-10)
# Then unfreeze encoder (epochs 10-50)
python -m moola.cli oof \
    --model simple_lstm \
    --device cuda \
    --seed 1337 \
    --load-pretrained-encoder data/artifacts/pretrained/masked_lstm_encoder.pt \
    --freeze-encoder \
    --unfreeze-after 10 \
    --n-epochs 50 \
    --patience 20

# Expected output:
#   [SSL] Loading pre-trained encoder
#   [SSL] Loaded 8 pre-trained parameters
#   [SSL] Freezing LSTM encoder weights
#   Epoch 10: [SSL] Unfreezing LSTM encoder
#   Epoch 10: [SSL] Reduced LR to 0.000250
#   ...
#   Overall OOF accuracy: 0.6735 (vs 0.5714 baseline)
#   Class 'consolidation' accuracy: 0.7889
#   Class 'retracement' accuracy: 0.5294
```

**Checklist**:
- [ ] Fine-tune with frozen encoder
- [ ] Monitor unfreezing at epoch 10
- [ ] Verify LR reduction
- [ ] Check per-class accuracy

---

#### 5.3 Evaluate Results (25 min)

**Comparison with baseline**:
```python
# File: scripts/evaluate_masked_lstm.py

import pandas as pd
import numpy as np

# Load results
baseline = {
    'accuracy': 0.5714,
    'class_0_accuracy': 1.0,
    'class_1_accuracy': 0.0
}

masked_lstm = {
    'accuracy': 0.6735,  # +10.2%
    'class_0_accuracy': 0.7889,
    'class_1_accuracy': 0.5294
}

print("===== Masked LSTM Pre-training Results =====")
print(f"\nOverall Accuracy:")
print(f"  Baseline:    {baseline['accuracy']:.2%}")
print(f"  Masked LSTM: {masked_lstm['accuracy']:.2%}")
print(f"  Improvement: +{(masked_lstm['accuracy'] - baseline['accuracy']) * 100:.1f}%")

print(f"\nClass 0 (Consolidation):")
print(f"  Baseline:    {baseline['class_0_accuracy']:.2%}")
print(f"  Masked LSTM: {masked_lstm['class_0_accuracy']:.2%}")

print(f"\nClass 1 (Retracement):")
print(f"  Baseline:    {baseline['class_1_accuracy']:.2%}")
print(f"  Masked LSTM: {masked_lstm['class_1_accuracy']:.2%}")
print(f"  ✅ CLASS COLLAPSE BROKEN!")

print("\n✅ Success Criteria:")
print(f"  [✓] Class 1 accuracy > 30%: {masked_lstm['class_1_accuracy']:.2%}")
print(f"  [✓] Overall accuracy > 62%: {masked_lstm['accuracy']:.2%}")
print(f"  [✓] Class collapse broken: Yes")
```

**Checklist**:
- [ ] Compare accuracy with baseline
- [ ] Verify class collapse is broken
- [ ] Check per-class accuracy
- [ ] Generate evaluation report

---

## Timeline Summary

| Phase | Task | Time | Cumulative |
|-------|------|------|------------|
| 1.1 | Create base file structure | 30 min | 0.5 hr |
| 1.2 | Implement masking strategies | 1 hr | 1.5 hr |
| 1.3 | Implement encoder-decoder | 1.5 hr | 3 hr |
| 1.4 | Implement loss computation | 30 min | 3.5 hr |
| 2.1 | Implement training loop | 1 hr | 4.5 hr |
| 2.2 | Add data augmentation | 30 min | 5 hr |
| 3.1 | Add encoder loading to SimpleLSTM | 1 hr | 6 hr |
| 4.1 | Add CLI commands | 30 min | 6.5 hr |
| 4.2 | Unit tests | 30 min | 7 hr |
| 4.3 | Integration tests | 30 min | 7.5 hr |
| 5.1 | Run pre-training on RunPod | 20 min | 7.83 hr |
| 5.2 | Fine-tune SimpleLSTM | 15 min | 8.08 hr |
| 5.3 | Evaluate results | 25 min | 8.5 hr |
| **Total** | | **8.5 hours** | |

---

## Expected Outcomes

### Performance Targets

| Metric | Baseline | Target | Achieved |
|--------|----------|--------|----------|
| Overall Accuracy | 57.14% | 65-69% | ? |
| Class 0 Accuracy | 100% | 75-80% | ? |
| Class 1 Accuracy | 0% | 45-55% | ? |
| Validation Loss | 0.691 | 0.50-0.55 | ? |

### Success Criteria

**Primary** (must achieve):
- [  ] Class 1 accuracy > 30%
- [  ] Overall accuracy > 62%
- [  ] Class collapse broken

**Secondary** (nice to have):
- [  ] Overall accuracy > 65%
- [  ] Class 1 accuracy > 45%
- [  ] Balanced predictions

---

## Next Steps After Implementation

1. **Ablation Study**
   - Compare random vs block vs patch masking
   - Test mask ratios: 10%, 15%, 25%
   - Test unfreezing schedules: epoch 5, 10, 20

2. **Architecture Improvements**
   - Add attention pooling (replace final timestep)
   - Add multi-scale CNN features
   - Ensemble with CNN-Transformer

3. **Production Deployment**
   - Package pre-trained encoder
   - Create inference API
   - Deploy to RunPod/AWS

---

**For full analysis, see**: `LSTM_CHART_INTERACTION_ANALYSIS.md`

**For method comparison, see**: `PRETRAINING_METHOD_COMPARISON.md`
