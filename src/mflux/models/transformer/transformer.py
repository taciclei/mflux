"""Transformer module."""

import mlx.core as mx
import mlx.nn as nn

from .transformer_block import TransformerBlock


class Transformer(nn.Module):
    """Transformer class."""

    def __init__(self, config):
        """Initialize transformer.

        Args:
            config: Model configuration
        """
        super().__init__()

        # Create patch embedding
        self.patch_embedding = nn.Linear(
            config.patch_size * config.patch_size * config.in_channels,
            config.hidden_size,
        )

        # Create position embedding
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )

        # Create transformer blocks
        self.transformer_blocks = [
            TransformerBlock(config)
            for _ in range(config.num_hidden_layers)
        ]

        # Create final layer norm
        self.ln = nn.LayerNorm(config.hidden_size)

        # Create output projection
        self.out_proj = nn.Linear(config.hidden_size, config.patch_size * config.patch_size * config.out_channels)

        # Store config
        self.config = config

    def __call__(
        self,
        pixel_values: mx.array,
        input_ids: mx.array,
        text_embeds: mx.array,
        pooled_text_embeds: mx.array,
    ) -> mx.array:
        """Forward pass.

        Args:
            pixel_values: Image tensor [batch_size, channels, height, width]
            input_ids: Token IDs [batch_size, sequence_length]
            text_embeds: Text embeddings [batch_size, sequence_length, hidden_size]
            pooled_text_embeds: Pooled text embeddings [batch_size, hidden_size]

        Returns:
            Output tensor [batch_size, channels, height, width]
        """
        # Get dimensions
        batch_size = pixel_values.shape[0]
        h_patches = self.config.height // self.config.patch_size
        w_patches = self.config.width // self.config.patch_size
        num_patches = h_patches * w_patches

        print(f"Transformer input shape: {pixel_values.shape}")
        print(f"h_patches: {h_patches}, w_patches: {w_patches}, num_patches: {num_patches}")

        # Reshape to patches [B, C, H, W] -> [B, num_patches, patch_size * patch_size * C]
        patches = mx.reshape(
            pixel_values,
            (
                batch_size,
                self.config.in_channels,
                h_patches,
                self.config.patch_size,
                w_patches,
                self.config.patch_size,
            ),
        )
        print(f"After first reshape: {patches.shape}")

        patches = mx.transpose(patches, [0, 2, 4, 1, 3, 5])
        print(f"After transpose: {patches.shape}")

        patches = mx.reshape(patches, (batch_size, num_patches, -1))
        print(f"After final reshape: {patches.shape}")

        # Embed patches
        x = self.patch_embedding(patches)
        print(f"After patch embedding: {x.shape}")

        # Add position embeddings
        positions = mx.arange(x.shape[1])
        x = x + self.position_embedding(positions)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Apply final layer norm
        x = self.ln(x)

        # Project to output patch size
        x = self.out_proj(x)

        # Reshape back to image
        x = mx.reshape(x, (batch_size, h_patches, w_patches, self.config.patch_size, self.config.patch_size, self.config.out_channels))
        x = mx.transpose(x, [0, 5, 1, 3, 2, 4])
        x = mx.reshape(x, (batch_size, self.config.out_channels, self.config.height, self.config.width))
        print(f"Transformer output shape: {x.shape}")

        return x
