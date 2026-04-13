# aegis/models/deep_learning/cnn1d.py
"""
1D CNN for Network Traffic Classification

Novelty: 1D Convolutional Neural Network specifically designed for
network flow features. Uses temporal patterns in packet-level data.

Architecture:
- Conv1D → BatchNorm → ReLU → MaxPool
- Conv1D → BatchNorm → ReLU → GlobalAvgPool
- Dense → Dropout → Dense → Softmax
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CNN1D(nn.Module):
    """
    1D CNN for Network Traffic Classification

    Handles network flow data as 1D sequences.
    Input: (batch_size, sequence_length, n_features)
    Output: (batch_size, num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        sequence_length: int = 1,
        filters: list = [64, 128],
        kernel_sizes: list = [3, 3],
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super(CNN1D, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm

        # If sequence_length is 1, treat input as (batch, features)
        # Otherwise treat as (batch, seq_len, features)
        if sequence_length == 1:
            # Reshape to (batch, 1, features) for 1D conv
            self.reshape_input = True
        else:
            self.reshape_input = False

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        in_channels = 1  # Start with single channel
        current_dim = input_dim * sequence_length

        for i, (out_channels, kernel_size) in enumerate(zip(filters, kernel_sizes)):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            self.conv_layers.append(conv)

            if use_batch_norm:
                bn = nn.BatchNorm1d(out_channels)
                self.bn_layers.append(bn)

            # MaxPool after each conv (except last)
            if i < len(filters) - 1:
                self.pool_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                current_dim = current_dim // 2
            else:
                self.pool_layers.append(nn.Identity())

            in_channels = out_channels

        # Calculate feature dimension after convolutions
        self.feature_dim = filters[-1]

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Handle different input shapes
        if x.dim() == 2:
            # (batch, features) -> (batch, 1, features)
            x = x.unsqueeze(1)
        elif x.dim() == 3 and self.reshape_input:
            # (batch, seq, features) -> (batch, seq*features)
            batch_size = x.size(0)
            x = x.reshape(batch_size, -1).unsqueeze(1)

        # Convolutional blocks
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.use_batch_norm and i < len(self.bn_layers):
                x = self.bn_layers[i](x)
            x = torch.relu(x)
            x = self.pool_layers[i](x)

        # Global average pooling for final conv layer
        x = torch.mean(x, dim=2)

        # Fully connected classification
        x = self.fc_layers(x)

        return x

    def get_embedding(self, x):
        """Get the penultimate layer embedding for XAI."""
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(1)
            elif x.dim() == 3 and self.reshape_input:
                batch_size = x.size(0)
                x = x.reshape(batch_size, -1).unsqueeze(1)

            for i, conv in enumerate(self.conv_layers):
                x = conv(x)
                if self.use_batch_norm and i < len(self.bn_layers):
                    x = self.bn_layers[i](x)
                x = torch.relu(x)
                x = self.pool_layers[i](x)

            x = torch.mean(x, dim=2)
            return x


class CNN1DClassifier:
    """
    Training and inference wrapper for CNN1D model.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        sequence_length: int = 1,
        filters: list = [64, 128],
        kernel_sizes: list = [3, 3],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        device: str = "auto"
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = CNN1D(
            input_dim=input_dim,
            num_classes=num_classes,
            sequence_length=sequence_length,
            filters=filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout
        ).to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.is_fitted = False
        self.classes_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 256,
        validation_split: float = 0.2,
        class_weight: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the CNN model.

        Args:
            X: Training features (n_samples, n_features) or (n_samples, seq_len, n_features)
            y: Training labels
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction for validation
            class_weight: Weights for classes (for imbalanced data)
            verbose: Print training progress

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

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                logger.info(msg)

        self.is_fitted = True
        self.classes_ = np.arange(self.num_classes)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Input features

        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probas = torch.softmax(outputs, dim=1)

        return probas.cpu().numpy()

    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction confidence (max probability).

        Args:
            X: Input features

        Returns:
            Confidence scores (n_samples,)
        """
        probas = self.predict_proba(X)
        return np.max(probas, axis=1)

    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Get embeddings for XAI purposes.

        Args:
            X: Input features

        Returns:
            Embeddings (n_samples, embedding_dim)
        """
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
            'sequence_length': self.sequence_length,
            'filters': self.filters,
            'kernel_sizes': self.kernel_sizes,
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
            sequence_length=checkpoint['sequence_length'],
            filters=checkpoint['filters'],
            kernel_sizes=checkpoint['kernel_sizes'],
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
            "model_type": "CNN1D",
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "sequence_length": self.sequence_length,
            "filters": self.filters,
            "kernel_sizes": self.kernel_sizes,
            "dropout": self.dropout,
            "device": str(self.device),
            "is_fitted": self.is_fitted,
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


def create_cnn1d_model(
    input_dim: int,
    num_classes: int = 2,
    sequence_length: int = 1,
    architecture: str = "standard"
) -> CNN1DClassifier:
    """
    Factory function to create CNN1D model with predefined architectures.

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        sequence_length: Sequence length for temporal data
        architecture: One of 'light', 'standard', 'deep'

    Returns:
        CNN1DClassifier instance
    """
    architectures = {
        "light": {
            "filters": [32, 64],
            "kernel_sizes": [3, 3],
            "dropout": 0.3
        },
        "standard": {
            "filters": [64, 128],
            "kernel_sizes": [3, 3],
            "dropout": 0.3
        },
        "deep": {
            "filters": [64, 128, 256],
            "kernel_sizes": [3, 3, 3],
            "dropout": 0.4
        }
    }

    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(architectures.keys())}")

    config = architectures[architecture]

    return CNN1DClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        sequence_length=sequence_length,
        filters=config["filters"],
        kernel_sizes=config["kernel_sizes"],
        dropout=config["dropout"]
    )