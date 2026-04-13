# aegis/models/deep_learning/vae.py
"""
Variational Autoencoder (VAE) for Network Anomaly Detection

Novelty: Unsupervised anomaly detection by learning the "normal" traffic
distribution. Attacks are detected as outliers based on reconstruction error.

Architecture:
- Encoder: x -> mu, log_var
- Reparameterization: z = mu + sigma * epsilon
- Decoder: z -> x_reconstructed
- Loss: Reconstruction + KL Divergence
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class VariationalAutoencoder(nn.Module):
    """
    VAE for Network Anomaly Detection

    Novelty: Learns distribution of normal traffic. High reconstruction
    error indicates anomalies (attacks).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: list = [64, 32]
    ):
        super(VariationalAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """Encode to latent space."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode from latent space."""
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def get_embedding(self, x):
        """Get latent representation."""
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu

    def get_reconstruction_error(self, x):
        """Get reconstruction error for anomaly scoring."""
        with torch.no_grad():
            x_recon, _, _ = self.forward(x)
            # MSE per sample
            error = torch.mean((x - x_recon) ** 2, dim=1)
        return error.cpu().numpy()


class VAEAnomalyDetector:
    """
    Training and inference wrapper for VAE-based anomaly detection.

    Novelty: Train only on normal data, detect anomalies by reconstruction error.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: list = [64, 32],
        learning_rate: float = 0.001,
        beta: float = 1.0,  # KL divergence weight
        device: str = "auto"
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.beta = beta

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = VariationalAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.is_fitted = False
        self.threshold = None
        self.training_recon_errors = None

    def vae_loss(self, x_recon, x, mu, log_var):
        """VAE loss = Reconstruction + Beta * KL Divergence."""
        # Reconstruction loss (MSE)
        recon_loss = torch.mean((x - x_recon) ** 2)

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + log_var - mu ** 2 - torch.exp(log_var))

        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 50,
        batch_size: int = 256,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the VAE.

        Note: Should be trained on NORMAL data only for anomaly detection.
        """
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Split data
        n_samples = len(X)
        n_val = int(n_samples * validation_split)

        if n_val > 0:
            indices = np.random.permutation(n_samples)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            X_train = X_tensor[train_indices]
            X_val = X_tensor[val_indices]
        else:
            X_train = X_tensor
            X_val = None

        # Create data loader
        train_dataset = TensorDataset(X_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        history = {
            "train_loss": [],
            "train_recon": [],
            "train_kl": [],
            "val_loss": [],
            "val_recon": []
        }

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_recon = 0.0
            train_kl = 0.0
            n_batches = 0

            for batch_x, in train_loader:
                self.optimizer.zero_grad()
                x_recon, mu, log_var = self.model(batch_x)
                loss, recon, kl = self.vae_loss(x_recon, batch_x, mu, log_var)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_recon += recon.item()
                train_kl += kl.item()
                n_batches += 1

            train_loss /= n_batches
            train_recon /= n_batches
            train_kl /= n_batches

            history["train_loss"].append(train_loss)
            history["train_recon"].append(train_recon)
            history["train_kl"].append(train_kl)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_recon, _, _ = self.model(X_val)
                    val_loss = torch.mean((X_val - val_recon) ** 2).item()

                history["val_loss"].append(val_loss)
                history["val_recon"].append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                logger.info(msg)

        # Calculate threshold from training data
        self.model.eval()
        with torch.no_grad():
            train_recon_errors = self.model.get_reconstruction_error(X_tensor)

        self.training_recon_errors = train_recon_errors
        # Use 95th percentile as threshold
        self.threshold = np.percentile(train_recon_errors, 95)

        self.is_fitted = True
        logger.info(f"Training complete. Anomaly threshold (95th percentile): {self.threshold:.6f}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (1 = anomaly/attack, 0 = normal).

        Returns binary predictions based on reconstruction error threshold.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        recon_errors = self.get_reconstruction_error(X)
        return (recon_errors > self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly probability based on reconstruction error.

        Returns probability of being an anomaly (higher = more likely attack).
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        recon_errors = self.get_reconstruction_error(X)

        # Convert to probability using error magnitude
        # Higher error = higher anomaly probability
        if self.threshold > 0:
            proba = np.clip(recon_errors / self.threshold, 0, 1)
        else:
            proba = np.zeros_like(recon_errors)

        # Return as [normal_prob, anomaly_prob]
        return np.column_stack([1 - proba, proba])

    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Get reconstruction error for each sample."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        return self.model.get_reconstruction_error(X_tensor)

    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """Get confidence (inverse of anomaly probability)."""
        probas = self.predict_proba(X)
        return 1 - probas[:, 1]  # Normal probability as confidence

    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Get latent embeddings for XAI."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            embeddings = self.model.get_embedding(X_tensor)

        return embeddings.cpu().numpy()

    def set_threshold(self, threshold: float):
        """Manually set anomaly detection threshold."""
        self.threshold = threshold
        logger.info(f"Threshold set to: {threshold:.6f}")

    def fit_threshold(
        self,
        X_normal: np.ndarray,
        X_anomaly: np.ndarray,
        target_fpr: float = 0.05
    ):
        """
        Fit threshold using known normal and anomaly data.

        Args:
            X_normal: Known normal samples
            X_anomaly: Known anomaly samples
            target_fpr: Target false positive rate
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get reconstruction errors
        normal_errors = self.get_reconstruction_error(X_normal)
        anomaly_errors = self.get_reconstruction_error(X_anomaly)

        # Find threshold that achieves target FPR on normal data
        self.threshold = np.percentile(normal_errors, (1 - target_fpr) * 100)

        # Calculate actual metrics
        fpr = np.mean(normal_errors > self.threshold)
        tpr = np.mean(anomaly_errors > self.threshold)

        logger.info(f"Threshold fitted: {self.threshold:.6f}")
        logger.info(f"Actual FPR: {fpr:.4f}, TPR: {tpr:.4f}")

        return {"threshold": self.threshold, "fpr": fpr, "tpr": tpr}

    def save(self, path: str):
        """Save model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'learning_rate': self.learning_rate,
            'beta': self.beta,
            'threshold': self.threshold,
            'is_fitted': self.is_fitted
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
            latent_dim=checkpoint['latent_dim'],
            hidden_dims=checkpoint['hidden_dims'],
            learning_rate=checkpoint['learning_rate'],
            beta=checkpoint['beta'],
            device=device
        )

        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.threshold = checkpoint['threshold']
        instance.is_fitted = checkpoint['is_fitted']

        logger.info(f"Model loaded from {path}")
        return instance

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": "VAE",
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "beta": self.beta,
            "threshold": self.threshold,
            "device": str(self.device),
            "is_fitted": self.is_fitted,
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


def create_vae_model(
    input_dim: int,
    latent_dim: int = 16,
    architecture: str = "light"
) -> VAEAnomalyDetector:
    """
    Factory function to create VAE model.

    Args:
        input_dim: Number of input features
        latent_dim: Latent space dimension
        architecture: One of 'light', 'standard', 'deep'

    Returns:
        VAEAnomalyDetector instance
    """
    architectures = {
        "light": {"hidden_dims": [32]},
        "standard": {"hidden_dims": [64, 32]},
        "deep": {"hidden_dims": [128, 64, 32]}
    }

    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}")

    config = architectures[architecture]

    return VAEAnomalyDetector(
        input_dim=input_dim,
        latent_dim=latent_dim,
        **config
    )