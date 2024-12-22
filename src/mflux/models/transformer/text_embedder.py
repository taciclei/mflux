import mlx.core as mx
from mlx import nn

from mflux.nn.linear import Linear  # Import du Linear personnalisé


class TextEmbedder(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.linear_1 = Linear(dim, 4 * dim)
        self.linear_2 = Linear(4 * dim, dim)  #  Dimension de sortie = dim

    def forward(self, caption: mx.array) -> mx.array:
        hidden_states = self.linear_1(caption)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
