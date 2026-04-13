# aegis/models/deep_learning/__init__.py
"""
Deep Learning Models for Network Intrusion Detection

This module contains novel deep learning architectures specifically designed
for network traffic classification and anomaly detection.

Models:
- CNN1D: 1D Convolutional Neural Network for packet-level features
- Transformer: Self-attention model for temporal patterns
- TabNet: Interpretable decision tree-based model
- VAE: Variational Autoencoder for unsupervised anomaly detection
"""

from aegis.models.deep_learning.cnn1d import CNN1D, CNN1DClassifier, create_cnn1d_model
from aegis.models.deep_learning.transformer import (
    NetworkTrafficTransformer,
    TransformerClassifier,
    create_transformer_model
)
from aegis.models.deep_learning.tabnet import SimpleTabNet, TabNetClassifier, create_tabnet_model
from aegis.models.deep_learning.vae import (
    VariationalAutoencoder,
    VAEAnomalyDetector,
    create_vae_model
)

__all__ = [
    # CNN1D
    "CNN1D",
    "CNN1DClassifier",
    "create_cnn1d_model",
    # Transformer
    "NetworkTrafficTransformer",
    "TransformerClassifier",
    "create_transformer_model",
    # TabNet
    "SimpleTabNet",
    "TabNetClassifier",
    "create_tabnet_model",
    # VAE
    "VariationalAutoencoder",
    "VAEAnomalyDetector",
    "create_vae_model",
]

__version__ = "1.0.0"