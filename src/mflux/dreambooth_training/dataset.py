"""DreamBooth dataset module."""

import os

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.text_encoder.text_encoder import TextEncoder
from mflux.models.vae.vae import VAE

from .tokenizer import Tokenizer


class DreamBoothDataset:
    """DreamBooth dataset class."""

    def __init__(
        self,
        instance_data_dir: str,
        size: int,
        tokenizer: Tokenizer,
        instance_prompt: str,
        text_encoder: TextEncoder,
        vae: VAE,
    ):
        """Initialize dataset.

        Args:
            instance_data_dir: Path to directory with instance images
            size: Size of images
            tokenizer: Tokenizer
            instance_prompt: Prompt to use for instance images
            text_encoder: Text encoder
            vae: VAE model
        """
        self.instance_data_dir = instance_data_dir
        self.size = size
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.text_encoder = text_encoder
        self.vae = vae

        # Get list of image files
        self.image_paths = [
            os.path.join(instance_data_dir, f)
            for f in os.listdir(instance_data_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]

        # Tokenize prompt
        token_ids = self.tokenizer.encode(instance_prompt)
        token_ids = mx.array(token_ids)

        # Get text embeddings
        self.text_embeds, self.pooled_text_embeds = self.text_encoder(token_ids)

        # Convert to float32
        self.text_embeds = self.text_embeds.astype(mx.float32)
        self.pooled_text_embeds = self.pooled_text_embeds.astype(mx.float32)

    def __iter__(self) -> "DreamBoothDataset":
        """Get iterator.

        Returns:
            Dataset iterator
        """
        return self

    def __next__(self) -> dict:
        """Get next batch.

        Returns:
            Dictionary containing:
                - pixel_values: Image tensor [batch_size, channels, height, width]
                - input_ids: Token IDs [batch_size, sequence_length]
                - text_embeds: Text embeddings [batch_size, sequence_length, hidden_size]
                - pooled_text_embeds: Pooled text embeddings [batch_size, hidden_size]
        """
        # Get random image
        image_path = np.random.choice(self.image_paths)

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.size, self.size), Image.BILINEAR)
        image = np.array(image)

        # Normalize to [-1, 1]
        image = (image / 127.5) - 1.0
        image = image.astype(np.float32)

        # Convert from HWC to CHW
        image = np.transpose(image, (2, 0, 1))

        # Convert to MLX array
        image = mx.array(image)

        # Add batch dimension [C, H, W] -> [B, C, H, W]
        image = mx.expand_dims(image, axis=0)

        # Verify dimensions
        assert image.shape == (1, 3, self.size, self.size), f"Invalid image shape: {image.shape}"

        # Return batch
        return {
            "pixel_values": image,
            "input_ids": mx.expand_dims(self.tokenizer.encode(self.instance_prompt), axis=0),
            "text_embeds": mx.expand_dims(self.text_embeds, axis=0),
            "pooled_text_embeds": mx.expand_dims(self.pooled_text_embeds, axis=0),
        }
