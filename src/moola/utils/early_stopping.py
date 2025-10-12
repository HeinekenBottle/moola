"""Early stopping callback for PyTorch model training.

Monitors validation metrics and stops training when performance plateaus,
preventing overfitting on small datasets.
"""

import numpy as np
import torch


class EarlyStopping:
    """Early stopping handler for PyTorch training loops.

    Monitors a validation metric and stops training when it stops improving,
    saving the best model checkpoint.

    Attributes:
        patience: Number of epochs to wait before stopping
        delta: Minimum change to qualify as an improvement
        mode: 'min' for loss, 'max' for accuracy/MCC
        counter: Current patience counter
        best_score: Best monitored score so far
        early_stop: Whether to stop training
        best_model_state: State dict of best model
    """

    def __init__(
        self,
        patience: int = 10,
        delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait after last improvement
            delta: Minimum change to qualify as improvement
            mode: 'min' (for loss) or 'max' (for accuracy/MCC)
            verbose: Whether to print messages
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

        # Set comparison function based on mode
        if mode == "min":
            self.is_better = lambda new, best: new < best - delta
            self.best_score = np.inf
        elif mode == "max":
            self.is_better = lambda new, best: new > best + delta
            self.best_score = -np.inf
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop.

        Args:
            score: Current validation score (loss or metric)
            model: PyTorch model to save if improved

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            # First epoch - initialize
            self.best_score = score
            self.save_checkpoint(score, model)
            return False

        if self.is_better(score, self.best_score):
            # Improvement detected
            if self.verbose:
                direction = "decreased" if self.mode == "min" else "increased"
                print(
                    f"[EarlyStopping] Validation score {direction} "
                    f"({self.best_score:.6f} → {score:.6f}). Saving model..."
                )

            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

        else:
            # No improvement
            self.counter += 1

            if self.verbose:
                print(
                    f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs "
                    f"(best: {self.best_score:.6f}, current: {score:.6f})"
                )

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"[EarlyStopping] Early stopping triggered! Restoring best model.")
                return True

        return False

    def save_checkpoint(self, score: float, model: torch.nn.Module):
        """Save model state dict.

        Args:
            score: Current validation score
            model: Model to save
        """
        self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def load_best_model(self, model: torch.nn.Module):
        """Load the best model state back into the model.

        Args:
            model: Model to restore
        """
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"[EarlyStopping] Restored best model (score: {self.best_score:.6f})")
