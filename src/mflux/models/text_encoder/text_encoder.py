"""Text encoder module."""

import mlx.core as mx
import mlx.nn as nn


class TextEncoder(nn.Module):
    """Text encoder class."""

    def __init__(self):
        """Initialize text encoder."""
        super().__init__()
        self.embedding = nn.Embedding(50257, 768)

    def __call__(self, token_ids: mx.array) -> tuple[mx.array, mx.array]:
        """Encode text.

        Args:
            token_ids: Token IDs [batch_size, sequence_length]

        Returns:
            Tuple of:
                - Text embeddings [batch_size, sequence_length, hidden_size]
                - Pooled text embeddings [batch_size, hidden_size]
        """
        # Get embeddings
        embeddings = self.embedding(token_ids)

        # Pool embeddings
        pooled = mx.mean(embeddings, axis=1)

        return embeddings, pooled
