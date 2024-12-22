"""Text encoder implementation."""

import mlx.core as mx
import mlx.nn as nn


class TextEncoder(nn.Module):
    """Text encoder module."""

    def __init__(self, hidden_size: int = 768):
        """Initialize text encoder.

        Args:
            hidden_size: Hidden size for embeddings
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(256, hidden_size)  # Simple ASCII embedding
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, token_ids: mx.array) -> mx.array:
        """Forward pass.

        Args:
            token_ids: Input token ids (B, L)

        Returns:
            Text embeddings (B, L, D)
        """
        # Embed tokens
        x = self.embedding(token_ids)  # (B, L, D)

        # Encode
        x = self.encoder(x)  # (B, L, D)

        return x

    def encode_pooled(self, token_ids: mx.array) -> mx.array:
        """Encode and pool text.

        Args:
            token_ids: Input token ids (B, L)

        Returns:
            Pooled text embeddings (B, D)
        """
        # Get embeddings
        x = self.forward(token_ids)  # (B, L, D)

        # Pool
        x = mx.mean(x, axis=1)  # (B, D)

        return x
