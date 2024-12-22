"""Layer normalization implementation."""

import mlx.core as mx
import mlx.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization module."""

    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.bias = mx.zeros((hidden_size,))
        self.eps = eps

    def forward(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (B, L, D)

        Returns:
            Normalized tensor (B, L, D)
        """
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return self.weight * (x - mean) / mx.sqrt(var + self.eps) + self.bias
