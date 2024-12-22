import mlx.core as mx
from mlx import nn


class EmbedND(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 128  # Dimension of embeddings (same as head_dimension in JointAttention)
        self.theta = 10000  # Base for frequencies
        self.num_heads = 24  # Number of attention heads
        self.max_position_embeddings = 1024  # Maximum sequence length

    def forward(self, position_ids: mx.array) -> mx.array:
        # position_ids shape: (batch_size, seq_length)
        seq_length = position_ids.shape[1]
        encoder_seq_length = 1  # Length of encoder sequence
        total_seq_length = seq_length + encoder_seq_length  # Total sequence length after concatenation

        # Generate frequencies for each dimension
        freqs = EmbedND.precompute_freqs_cis(
            dim=self.dim // 2,  # Half dimension for complex numbers
            theta=self.theta,
            seq_length=total_seq_length
        )

        # Reshape for attention heads
        # From (total_seq_length, dim//2, 2) to (1, num_heads, total_seq_length, dim//2, 2)
        freqs = mx.expand_dims(freqs, axis=0)  # Add batch dimension
        freqs = mx.expand_dims(freqs, axis=1)  # Add head dimension
        freqs = mx.broadcast_to(freqs, (1, self.num_heads, total_seq_length, self.dim // 2, 2))

        return freqs

    @staticmethod
    def precompute_freqs_cis(dim: int, theta: float, seq_length: int) -> mx.array:
        # Generate frequency bands
        freqs = mx.exp(
            -mx.arange(0, dim, dtype=mx.float32) * (mx.log(theta) / dim)
        )

        # Generate positions
        positions = mx.arange(seq_length, dtype=mx.float32)

        # Compute angles
        angles = mx.outer(positions, freqs)

        # Return complex components: (seq_length, dim, 2)
        return mx.stack(
            [mx.cos(angles), mx.sin(angles)],
            axis=-1
        )
