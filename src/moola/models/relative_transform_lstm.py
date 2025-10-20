"""RelativeTransform LSTM model for 11-dimensional features.

This model is specifically designed to work with RelativeTransform features
and can load pretrained BiLSTM encoder weights from masked autoencoder training.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from moola.models.base import BaseModel
from moola.models.pretrained_utils import load_pretrained_strict
from moola.utils.seeds import set_seed


class RelativeTransformLSTMNet(nn.Module):
    """Neural network for RelativeTransform features with BiLSTM encoder."""
    
    def __init__(
        self,
        input_dim: int = 11,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_classes = n_classes
        
        # BiLSTM encoder (matches pretrained encoder architecture)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Logits of shape (batch_size, n_classes)
        """
        # BiLSTM encoding
        outputs, (hidden, cell) = self.encoder(x)
        
        # Use final hidden state from both directions
        # hidden shape: (num_layers * 2, batch_size, hidden_size)
        # Take last layer from forward and backward directions
        forward_hidden = hidden[-2]  # Last forward layer
        backward_hidden = hidden[-1]  # Last backward layer
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=-1)
        
        # Classification
        logits = self.classifier(combined_hidden)
        return logits


class RelativeTransformLSTMModel(BaseModel):
    """Model for RelativeTransform features with pretrained encoder support."""
    
    def __init__(
        self,
        seed: int = 1337,
        device: str = "cpu",
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 100,
        early_stopping_patience: int = 20,
        freeze_encoder: bool = False,
    ):
        super().__init__(seed=seed, device=device)
        self.device = torch.device(device)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.freeze_encoder = freeze_encoder
        
        # Will be set after first call to fit()
        self.model = None
        self.input_dim = None
        self.n_classes = None
        
        # For pretrained encoder loading
        self._pretrained_encoder_path = None
        
    def _build_model(self, input_dim: int, n_classes: int) -> RelativeTransformLSTMNet:
        """Build the model architecture.
        
        Args:
            input_dim: Input feature dimension (should be 11 for RelativeTransform)
            n_classes: Number of output classes
            
        Returns:
            Instantiated model
        """
        model = RelativeTransformLSTMNet(
            input_dim=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            n_classes=n_classes,
            dropout=self.dropout,
        )
        
        # Move to device
        model = model.to(self.device)
        
        return model
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "RelativeTransformLSTMModel":
        """Train the model.
        
        Args:
            X: Input features of shape (n_samples, seq_len, input_dim)
            y: Target labels
            **kwargs: Additional arguments including:
                - pretrained_encoder: Path to pretrained encoder weights
                - freeze_encoder: Whether to freeze encoder weights
                
        Returns:
            Self for method chaining
        """
        # Extract pretrained encoder path if provided
        pretrained_encoder = kwargs.get("pretrained_encoder")
        freeze_encoder = kwargs.get("freeze_encoder", self.freeze_encoder)
        
        # Set random seed
        set_seed(self.seed)
        
        # Validate input shape
        if len(X.shape) != 3:
            raise ValueError(f"Expected 3D input (n_samples, seq_len, input_dim), got {X.shape}")
            
        self.input_dim = X.shape[-1]
        self.n_classes = len(np.unique(y))
        
        if self.input_dim != 11:
            raise ValueError(f"RelativeTransformLSTM expects 11 features, got {self.input_dim}")
        
        # Build model
        self.model = self._build_model(self.input_dim, self.n_classes)
        
        # Load pretrained encoder if provided
        if pretrained_encoder:
            self.load_pretrained_encoder(pretrained_encoder, freeze_encoder)
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
                
        # Prepare data - convert string labels to integers if needed
        if y.dtype == 'object':
            # Convert string labels to integers
            unique_labels = np.unique(y)
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            y = np.array([label_map[label] for label in y])
        
        dataset = TensorDataset(
            torch.FloatTensor(X).to(self.device),
            torch.LongTensor(y).to(self.device),
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        
        self.model.train()
        for epoch in range(self.max_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            epoch_loss = total_loss / len(dataloader)
            epoch_acc = correct / total
            
            # Early stopping check (simplified - using training loss)
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                patience_counter = 0
                # Save best model
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
                break
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Acc = {epoch_acc:.4f}")
        
        return self
        
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features of shape (n_samples, seq_len, input_dim)
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            
        return predicted.cpu().numpy()
        
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input features of shape (n_samples, seq_len, input_dim)
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
        return probabilities.cpu().numpy()
        
    def load_pretrained_encoder(self, encoder_path: str, freeze: bool = True) -> None:
        """Load pretrained BiLSTM encoder weights.
        
        Args:
            encoder_path: Path to pretrained encoder checkpoint
            freeze: Whether to freeze encoder weights
        """
        if self.model is None:
            raise ValueError("Model must be built before loading encoder")
            
        # Load the checkpoint
        checkpoint = torch.load(encoder_path, map_location=self.device)
        encoder_state_dict = checkpoint["encoder_state_dict"]
        
        # Map pretrained weights to our model's encoder
        mapped_state_dict = {}
        for key, value in encoder_state_dict.items():
            # Add 'encoder.' prefix to match our model structure
            mapped_key = f"encoder.{key}"
            if mapped_key in self.model.state_dict():
                mapped_state_dict[mapped_key] = value
                
        # Load the mapped weights
        self.model.load_state_dict(mapped_state_dict, strict=False)
        
        print(f"Loaded {len(mapped_state_dict)} encoder weights from {encoder_path}")
        
        if freeze:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            print("Encoder weights frozen")
            
    def save(self, path: Path) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "n_classes": self.n_classes,
            "hyperparams": {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "early_stopping_patience": self.early_stopping_patience,
            },
            "seed": self.seed,
        }
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
        
    def load(self, path: Path) -> "RelativeTransformLSTMModel":
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Self for method chaining
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.input_dim = checkpoint["input_dim"]
        self.n_classes = checkpoint["n_classes"]
        
        # Rebuild model
        self.model = self._build_model(self.input_dim, self.n_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        print(f"Model loaded from {path}")
        return self