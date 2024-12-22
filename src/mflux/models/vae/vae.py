"""VAE module optimized for MLX."""

import logging
from typing import Dict, Tuple

import mlx.core as mx
import mlx.nn as nn

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Only add handler if not already added to avoid duplicate logs
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)


def log_tensor_info(name: str, tensor: mx.array, level: str = "debug"):
    """Log tensor information at specified level."""
    msg = f"{name} shape: {tensor.shape}"
    if level == "debug":
        logger.debug(msg)
    elif level == "info":
        logger.info(msg)
    elif level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)


class ResBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        identity = x

        x = self.conv1(x)
        x = mx.maximum(x * 0.2, x)  # LeakyReLU
        x = self.conv2(x)
        x = mx.maximum(x * 0.2, x)  # LeakyReLU

        return x + identity


class Encoder(nn.Module):
    """Encoder network."""

    def __init__(self, latent_channels: int = 4):
        super().__init__()

        self.conv_in = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            ResBlock(128),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            ResBlock(256),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            ResBlock(512),
        )

        self.bottleneck = ResBlock(512)
        self.proj_mu = nn.Conv2d(512, latent_channels, kernel_size=1)
        self.proj_logvar = nn.Conv2d(512, latent_channels, kernel_size=1)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        log_tensor_info("Encoder input", x, level="debug")

        x = self.conv_in(x)
        x = mx.maximum(x * 0.2, x)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.bottleneck(x)

        mu = self.proj_mu(x)
        logvar = self.proj_logvar(x)

        log_tensor_info("Encoder mu", mu, level="debug")
        log_tensor_info("Encoder logvar", logvar, level="debug")

        return mu, logvar


class Decoder(nn.Module):
    """Decoder network."""

    def __init__(self, latent_channels: int = 4):
        super().__init__()

        self.conv_in = nn.Conv2d(latent_channels, 512, kernel_size=3, stride=1, padding=1)
        self.bottleneck = ResBlock(512)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            ResBlock(256),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            ResBlock(128),
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            ResBlock(64),
        )

        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        log_tensor_info("Decoder input", x, level="debug")

        x = self.conv_in(x)
        x = mx.maximum(x * 0.2, x)
        x = self.bottleneck(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        x = self.conv_out(x)
        x = mx.tanh(x)

        log_tensor_info("Decoder output", x, level="debug")
        return x


class VAE(nn.Module):
    """Complete VAE architecture."""

    def __init__(self, latent_channels: int = 4):
        super().__init__()
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)
        logger.info(f"Initialized VAE with {latent_channels} latent channels")

    def reparameterize(self, mu: mx.array, logvar: mx.array) -> mx.array:
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(mu.shape)
        return mu + eps * std

    def encode(self, x: mx.array) -> mx.array:
        """Encode input to latent space."""
        log_tensor_info("VAE input", x, level="debug")

        # Normalize and convert format
        x = x / 127.5 - 1.0
        x = mx.transpose(x, [0, 2, 3, 1])

        # Encode
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        # Convert back
        z = mx.transpose(z, [0, 3, 1, 2])

        log_tensor_info("VAE latent", z, level="debug")
        return z

    def decode(self, z: mx.array) -> mx.array:
        """Decode from latent space."""
        z = mx.transpose(z, [0, 2, 3, 1])
        x = self.decoder(z)
        x = mx.transpose(x, [0, 3, 1, 2])
        x = (x + 1.0) * 127.5
        return x

    def forward(self, x: mx.array) -> Dict[str, mx.array]:
        """Forward pass through the VAE."""
        # Process
        x = x / 127.5 - 1.0
        x = mx.transpose(x, [0, 2, 3, 1])

        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)

        # Format output
        recon = mx.transpose(recon, [0, 3, 1, 2])
        recon = (recon + 1.0) * 127.5

        return {"reconstruction": recon, "mu": mu, "logvar": logvar}
