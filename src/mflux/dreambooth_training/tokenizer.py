"""Tokenizer module."""

import mlx.core as mx


class Tokenizer:
    """Simple tokenizer class."""

    def __init__(self):
        """Initialize tokenizer."""
        pass

    def encode(self, text: str) -> mx.array:
        """Encode text.

        Args:
            text: Text to encode

        Returns:
            Token IDs [batch_size, sequence_length]
        """
        # Convert text to list of bytes
        tokens = [ord(c) for c in text]

        # Add batch dimension
        tokens = mx.array([tokens])

        return tokens
