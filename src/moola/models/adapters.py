"""Model Adapters - Backward Compatibility Shims.

Thin fit/predict/save wrappers to preserve old BaseModel contract.
Provides fit(), predict(), save(), load() methods that legacy code expects.

Usage:
    >>> from moola.models.jade_core import JadeCompact
    >>> from moola.models.adapters import ModuleAdapter, TrainCfg
    >>>
    >>> core = JadeCompact(input_size=10, hidden_size=96)
    >>> cfg = TrainCfg(epochs=60, lr=3e-4, device="cuda")
    >>> model = ModuleAdapter(core, cfg=cfg)
    >>> 
    >>> # Train
    >>> hist = model.fit(train_dl, val_dl)
    >>> 
    >>> # Predict
    >>> preds, probs = model.predict(test_dl)
    >>> 
    >>> # Save/Load
    >>> model.save("artifacts/models/jade.pt")
    >>> loaded = ModuleAdapter.load("artifacts/models/jade.pt")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


@dataclass
class TrainCfg:
    """Training configuration for adapter.

    Args:
        epochs: Number of training epochs (default: 60)
        lr: Learning rate (default: 3e-4)
        device: Device to train on ('cpu' or 'cuda', default: 'cpu')
        batch_size: Batch size for training (default: 29, Stones requirement)
        max_grad_norm: Gradient clipping threshold (default: 2.0, Stones: 1.5-2.0)
        early_stopping_patience: Epochs to wait before stopping (default: 20)
        val_split: Validation split ratio (default: 0.15)
        use_amp: Use automatic mixed precision (default: False)
    """

    epochs: int = 60
    lr: float = 3e-4
    device: str = "cpu"
    batch_size: int = 29
    max_grad_norm: float = 2.0
    early_stopping_patience: int = 20
    val_split: float = 0.15
    use_amp: bool = False


class ModuleAdapter:
    """Thin fit/predict/save wrapper to preserve old BaseModel contract.

    Provides fit(), predict(), save(), load() methods that legacy code expects.
    Wraps a pure nn.Module (like JadeCompact) with training logic.

    Args:
        module: PyTorch nn.Module to wrap (e.g., JadeCompact)
        cfg: Training configuration (default: TrainCfg())
    """

    def __init__(self, module: nn.Module, *, cfg: Optional[TrainCfg] = None):
        self.module = module
        self.cfg = cfg or TrainCfg()
        self.is_fitted = False
        self.label_to_idx = None
        self.idx_to_label = None

    def fit(self, train_dl, val_dl=None) -> dict[str, Any]:
        """Train the model on provided DataLoaders.

        Args:
            train_dl: Training DataLoader yielding (x, y, ptr) tuples
            val_dl: Optional validation DataLoader

        Returns:
            dict with training history (loss, val_loss, etc.)
        """
        dev = self.cfg.device
        self.module.to(dev)
        self.module.train()

        # Optimizer
        opt = torch.optim.AdamW(self.module.parameters(), lr=self.cfg.lr)

        # Loss function
        loss_fn = nn.CrossEntropyLoss()

        # Training history
        hist = {"train_loss": [], "val_loss": [], "val_acc": []}

        # Early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Training {self.module.__class__.__name__} for {self.cfg.epochs} epochs")
        logger.info(f"Device: {dev} | LR: {self.cfg.lr} | Batch size: {self.cfg.batch_size}")

        for epoch in range(self.cfg.epochs):
            # Training phase
            self.module.train()
            running_loss = 0.0
            n_batches = 0

            for xb, yb, _ptr in train_dl:
                xb, yb = xb.to(dev), yb.to(dev)

                # Forward pass
                opt.zero_grad()
                outputs = self.module(xb)
                logits = outputs["logits"]
                loss = loss_fn(logits, yb)

                # Backward pass
                loss.backward()

                # Gradient clipping (Stones: 1.5-2.0)
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.cfg.max_grad_norm)

                opt.step()

                running_loss += float(loss.detach())
                n_batches += 1

            avg_train_loss = running_loss / max(1, n_batches)
            hist["train_loss"].append(avg_train_loss)

            # Validation phase
            if val_dl is not None:
                val_loss, val_acc = self._validate(val_dl, loss_fn, dev)
                hist["val_loss"].append(val_loss)
                hist["val_acc"].append(val_acc)

                logger.info(
                    f"Epoch {epoch+1}/{self.cfg.epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.cfg.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                logger.info(f"Epoch {epoch+1}/{self.cfg.epochs} | Train Loss: {avg_train_loss:.4f}")

        self.is_fitted = True
        return hist

    def _validate(self, val_dl, loss_fn, dev) -> tuple[float, float]:
        """Run validation and return loss and accuracy."""
        self.module.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.inference_mode():
            for xb, yb, _ptr in val_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                outputs = self.module(xb)
                logits = outputs["logits"]
                loss = loss_fn(logits, yb)
                running_loss += float(loss.detach())

                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        avg_loss = running_loss / max(1, len(val_dl))
        accuracy = correct / max(1, total)
        return avg_loss, accuracy

    def predict(self, dl) -> tuple[np.ndarray, np.ndarray]:
        """Generate predictions and probabilities for provided DataLoader.

        Args:
            dl: DataLoader yielding (x, y, ptr) tuples

        Returns:
            Tuple of (predictions, probabilities)
            - predictions: np.ndarray of shape [N] with class indices
            - probabilities: np.ndarray of shape [N, num_classes]
        """
        self.module.eval()
        preds_list = []
        probs_list = []

        with torch.inference_mode():
            for xb, _yb, _ptr in dl:
                xb = xb.to(self.cfg.device)
                outputs = self.module(xb)
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                probs_list.append(probs.cpu().numpy())
                preds_list.append(preds.cpu().numpy())

        predictions = np.concatenate(preds_list, axis=0)
        probabilities = np.concatenate(probs_list, axis=0)

        return predictions, probabilities

    def predict_proba(self, dl) -> np.ndarray:
        """Generate class probabilities for provided DataLoader.

        Args:
            dl: DataLoader yielding (x, y, ptr) tuples

        Returns:
            np.ndarray of shape [N, num_classes] with class probabilities
        """
        _, probs = self.predict(dl)
        return probs

    def save(self, path: str | Path):
        """Save model state dict to disk.

        Args:
            path: Path to save model checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.module.state_dict(),
            "model_class": self.module.__class__.__name__,
            "model_id": getattr(self.module, "MODEL_ID", "unknown"),
            "codename": getattr(self.module, "CODENAME", "unknown"),
            "config": {
                "input_size": getattr(self.module, "input_size", None),
                "hidden_size": getattr(self.module, "hidden_size", None),
                "num_layers": getattr(self.module, "num_layers", None),
                "num_classes": getattr(self.module, "num_classes", None),
            },
        }

        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(
        cls, path: str | Path, *, core_kwargs: Optional[dict] = None, cfg: Optional[TrainCfg] = None
    ):
        """Load model from saved state dict.

        Args:
            path: Path to saved model checkpoint
            core_kwargs: Optional kwargs to override saved config
            cfg: Optional training config for loaded model

        Returns:
            ModuleAdapter instance with loaded model
        """
        from .jade_core import JadeCompact

        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu")

        # Only JadeCompact is supported
        model_class_name = checkpoint.get("model_class", "JadeCompact")
        if model_class_name != "JadeCompact":
            raise ValueError(
                f"Unsupported model class: {model_class_name}. Only JadeCompact is supported."
            )
        model_cls = JadeCompact

        # Build core with saved or overridden config
        saved_config = checkpoint.get("config", {})
        if core_kwargs is not None:
            saved_config.update(core_kwargs)

        core = model_cls(**{k: v for k, v in saved_config.items() if v is not None})

        # Load state dict
        core.load_state_dict(checkpoint["state_dict"], strict=True)

        logger.info(f"Model loaded from {path}")
        logger.info(
            f"Model: {checkpoint.get('model_id', 'unknown')} // {checkpoint.get('codename', 'unknown')}"
        )

        return cls(core, cfg=cfg)
