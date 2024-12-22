"""VAE implementation."""

import mlx.core as mx
import mlx.nn as nn


class VAE(nn.Module):
    """Variational autoencoder module."""

    def __init__(self, in_channels: int = 3, latent_channels: int = 4):
        """Initialize VAE.

        Args:
            in_channels: Number of input channels
            latent_channels: Number of latent channels
        """
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, latent_channels * 2, kernel_size=3, stride=2, padding=1),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, in_channels, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: mx.array) -> mx.array:
        """Encode input to latent space.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Latent tensor (B, C', H', W')
        """
        # Encode
        h = self.encoder(x)  # (B, C'*2, H', W')

        # Split into mean and logvar
        B, _, H, W = h.shape
        h = h.reshape(B, 2, self.latent_channels, H, W)
        mean, logvar = h[:, 0], h[:, 1]

        # Sample
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(mean.shape)
        z = mean + eps * std

        return z

    def decode(self, z: mx.array) -> mx.array:
        """Decode latent tensor to image.

        Args:
            z: Latent tensor (B, C', H', W')

        Returns:
            Decoded tensor (B, C, H, W)
        """
        return self.decoder(z)
