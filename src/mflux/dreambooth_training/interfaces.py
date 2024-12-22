"""
Interface definitions for DreamBooth components.

This module defines the abstract base classes and interfaces that must be
implemented by various components of the DreamBooth training pipeline.
"""

from abc import ABC, abstractmethod

import mlx.core as mx


class VAEInterface(ABC):
    """Interface for Variational Autoencoder models."""

    @abstractmethod
    def encode(self, x: mx.array) -> mx.array:
        """Encode input images to latent space.

        Args:
            x: Input images tensor

        Returns:
            Latent space representation
        """
        pass

    @abstractmethod
    def decode(self, z: mx.array) -> mx.array:
        """Decode latent vectors to images.

        Args:
            z: Latent space tensor

        Returns:
            Reconstructed images
        """
        pass
