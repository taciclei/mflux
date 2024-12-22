"""
VAE adapter module for DreamBooth training.

This module provides adapter classes to make different VAE implementations
compatible with the DreamBooth training pipeline.
"""

import mlx.core as mx
import mlx.nn as nn

from .interfaces import VAEInterface


class VAEAdapter(VAEInterface):
    """Adapter for VAE models to make them compatible with DreamBooth training."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_channels: int = 4,
        scaling_factor: float = 0.18215,
    ) -> None:
        """Initialize the VAE adapter.

        Args:
            encoder: Encoder network
            decoder: Decoder network
            latent_channels: Number of channels in latent space
            scaling_factor: Scaling factor for latent vectors
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor

    def encode(self, x: mx.array) -> mx.array:
        """Encode input images to latent space.

        Args:
            x: Input images tensor of shape (B, C, H, W)

        Returns:
            Latent vectors of shape (B, latent_channels, H/8, W/8)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {x.shape}")

        h = self.encoder.encode(x)
        return h

    def decode(self, z: mx.array) -> mx.array:
        """Decode latent vectors to images.

        Args:
            z: Latent vectors of shape (B, latent_channels, H/8, W/8)

        Returns:
            Reconstructed images of shape (B, C, H, W)
        """
        if z.ndim != 4:
            raise ValueError(f"Expected 4D latent tensor, got shape {z.shape}")

        z = z / self.scaling_factor
        return self.decoder(z)
