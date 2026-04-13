# aegis/models/deep_learning/transformer.py
"""
Transformer Model for Network Traffic Classification

Novelty: Self-attention based model to capture temporal patterns in network traffic.
Can reveal which packet features are most important for detection.

Architecture:
- Input Embedding
- Positional Encoding
- Multi-Head Self-Attention
- Feed-Forward Network
- Classification Head
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NetworkTrafficTransformer(nn.Module):
    """
    Transformer for Network Traffic Classification

    Novelty: Uses self-attention to learn which network flow features
    are most important for detecting attacks.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        seq_length: int = 10
    ):
        super(NetworkTrafficTransformer, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.seq_length = seq_length

        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length + 1, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_length, input_dim) or (batch_size, input_dim)

        Returns:
            Class logits (batch_size, num_classes)
        """
        # Handle different input shapes
        if x.dim() == 2:
            # (batch, features) -> (batch, 1, features)
            x = x.unsqueeze(1)

        batch_size, seq_len, features = x.shape

        # Project to d_model dimensions
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Use the last sequence element for classification
        x = x[:, -1, :]  # (batch, d_model)

        # Classification
        x = self.fc(x)

        return x

    def get_attention_weights(self, x):
        """Get attention weights for interpretability."""
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(1)

            x = self.input_projection(x)
            x = self.pos_encoder(x)

            # Get attention from first layer
            attn_layer = self.transformer_encoder.layers[0]
            attn_output, attn_weights = attn_layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False
            )

        return attn_weights.cpu().numpy()

    def get_embedding(self, x):
        """Get embedding before classification."""
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(1)

            x = self.input_projection(x)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            x = x[:, -1, :]

        return x


class TransformerClassifier:
    """
    Training and inference wrapper for Transformer model.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        seq_length: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 0.0001,
        device: str = "auto"
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = NetworkTrafficTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            seq_length=seq_length
        ).to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        self.is_fitted = False
        self.classes_ = None

    def _prepare_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Convert flat features to sequences.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Sequences (n_samples, seq_length, n_features)
        """
        n_samples = len(X)

        # Pad or truncate to seq_length
        if n_samples < self.seq_length:
            # Pad with zeros
            padding = np.zeros((self.seq_length - n_samples, self.input_dim))
            X = np.vstack([np.tile(X, (self.seq_length // n_samples + 1, 1))[:self.seq_length], padding])
        else:
            # Take last seq_length samples (for proper temporal ordering)
            X = X[-self.seq_length:]

        return X

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 128,
        validation_split: float = 0.2,
        class_weight: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the Transformer model.

        Args:
            X: Training features
            y: Training labels
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Validation fraction
            class_weight: Class weights for imbalanced data
            verbose: Print progress

        Returns:
            Training history
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        # Split data
        n_samples = len(X)
        n_val = int(n_samples * validation_split)

        if n_val > 0:
            indices = np.random.permutation(n_samples)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            X_train = X_tensor[train_indices]
            y_train = y_tensor[train_indices]
            X_val = X_tensor[val_indices]
            y_val = y_tensor[val_indices]
        else:
            X_train = X_tensor
            y_train = y_tensor
            X_val = None
            y_val = None

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Handle class weights
        if class_weight is not None:
            weights = torch.FloatTensor(class_weight).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)

        # Training loop
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            train_loss /= train_total
            train_acc = train_correct / train_total

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = self.criterion(val_outputs, y_val).item()
                    _, val_predicted = torch.max(val_outputs, 1)
                    val_acc = (val_predicted == y_val).sum().item() / len(y_val)

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()

            if verbose and (epoch + 1) % 5 == 0:
                msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                logger.info(msg)

        # Restore best model
        if X_val is not None and hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)

        self.is_fitted = True
        self.classes_ = np.arange(self.num_classes)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probas = torch.softmax(outputs, dim=1)

        return probas.cpu().numpy()

    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """Get prediction confidence."""
        probas = self.predict_proba(X)
        return np.max(probas, axis=1)

    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """Get attention weights for interpretability."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return self.model.get_attention_weights(X)

    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Get embeddings for XAI."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            embeddings = self.model.get_embedding(X_tensor)

        return embeddings.cpu().numpy()

    def save(self, path: str):
        """Save model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'seq_length': self.seq_length,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'is_fitted': self.is_fitted,
            'classes_': self.classes_
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "auto"):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=device if device != "auto" else None)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        instance = cls(
            input_dim=checkpoint['input_dim'],
            num_classes=checkpoint['num_classes'],
            seq_length=checkpoint['seq_length'],
            d_model=checkpoint['d_model'],
            nhead=checkpoint['nhead'],
            num_layers=checkpoint['num_layers'],
            dim_feedforward=checkpoint['dim_feedforward'],
            dropout=checkpoint['dropout'],
            learning_rate=checkpoint['learning_rate'],
            device=device
        )

        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.is_fitted = checkpoint['is_fitted']
        instance.classes_ = checkpoint['classes_']

        logger.info(f"Model loaded from {path}")
        return instance

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": "Transformer",
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "seq_length": self.seq_length,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "device": str(self.device),
            "is_fitted": self.is_fitted,
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


def create_transformer_model(
    input_dim: int,
    num_classes: int = 2,
    seq_length: int = 10,
    architecture: str = "light"
) -> TransformerClassifier:
    """
    Factory function to create Transformer model with predefined architectures.

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        seq_length: Sequence length
        architecture: One of 'light', 'standard', 'deep'

    Returns:
        TransformerClassifier instance
    """
    architectures = {
        "light": {
            "d_model": 32,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.1
        },
        "standard": {
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1
        },
        "deep": {
            "d_model": 128,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 256,
            "dropout": 0.2
        }
    }

    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}")

    config = architectures[architecture]

    return TransformerClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        seq_length=seq_length,
        **config
    )