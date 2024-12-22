"""Transformer block module."""

import mlx.core as mx
import mlx.nn as nn

from .model_config import ModelConfig


class TransformerBlock(nn.Module):
    """Transformer block class."""

    def __init__(self, config: ModelConfig):
        """Initialize transformer block.

        Args:
            config: Model configuration
        """
        super().__init__()

        # Multi-head attention
        self.attention = nn.MultiHeadAttention(
            num_heads=config.num_attention_heads,
            dims=config.hidden_size,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )

        # Layer normalization
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, sequence_length, hidden_size]

        Returns:
            Output tensor [batch_size, sequence_length, hidden_size]
        """
        # Multi-head attention
        attn_out = self.attention(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_out

        # Feed-forward network
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out

        return x
