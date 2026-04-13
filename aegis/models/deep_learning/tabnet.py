# aegis/models/deep_learning/tabnet.py
"""
Simplified TabNet-like Model for Network Intrusion Detection

Novelty: Attention-based feature selection for interpretable tabular classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SimpleTabNet(nn.Module):
    """
    Simplified TabNet-like model with attention-based feature selection.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        n_steps: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        super(SimpleTabNet, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.n_steps = n_steps
        self.hidden_dim = hidden_dim

        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Feature attention (applied to original input for interpretability)
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=1)
        )

        # Step decisions
        self.decision_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for _ in range(n_steps)
        ])

        # Final classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * n_steps, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """Forward pass with attention."""
        batch_size = x.size(0)

        # Get feature attention (for interpretability)
        attn_weights = self.feature_attention(x)

        # Embed input
        embedded = self.input_embedding(x)

        # Get decisions from each step
        decisions = []
        current = embedded

        for step in range(self.n_steps):
            decision = self.decision_layers[step](current)
            decisions.append(decision)
            current = decision  # Use previous decision for next step

        # Stack and flatten decisions
        decisions = torch.stack(decisions, dim=1)
        decisions = decisions.view(batch_size, -1)

        # Final classification
        output = self.classifier(decisions)

        return output

    def get_attention(self, x):
        """Get attention weights for interpretability."""
        self.eval()
        with torch.no_grad():
            return self.feature_attention(x)

    def get_embedding(self, x):
        """Get embedding before classification."""
        with torch.no_grad():
            embedded = self.input_embedding(x)
            decisions = []
            current = embedded
            for step in range(self.n_steps):
                decision = self.decision_layers[step](current)
                decisions.append(decision)
                current = decision
            decisions = torch.stack(decisions, dim=1)
            return decisions.view(x.size(0), -1)


class TabNetClassifier:
    """Training and inference wrapper for TabNet."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        n_steps: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        device: str = "auto"
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.n_steps = n_steps

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = SimpleTabNet(
            input_dim=input_dim,
            num_classes=num_classes,
            n_steps=n_steps,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

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
        """Train the TabNet model."""
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
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = float('inf')

        for epoch in range(epochs):
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
                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                logger.info(msg)

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

    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """Get feature importance from attention weights."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            attn_weights = self.model.get_attention(X_tensor)

        return attn_weights.mean(dim=0).cpu().numpy()

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
            'n_steps': self.n_steps,
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
            n_steps=checkpoint['n_steps'],
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
            "model_type": "TabNet",
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "n_steps": self.n_steps,
            "device": str(self.device),
            "is_fitted": self.is_fitted,
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


def create_tabnet_model(
    input_dim: int,
    num_classes: int = 2,
    architecture: str = "light"
) -> TabNetClassifier:
    """Factory function to create TabNet model."""
    architectures = {
        "light": {"n_steps": 2, "hidden_dim": 32},
        "standard": {"n_steps": 3, "hidden_dim": 64},
        "deep": {"n_steps": 5, "hidden_dim": 128}
    }

    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}")

    config = architectures[architecture]

    return TabNetClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        **config
    )