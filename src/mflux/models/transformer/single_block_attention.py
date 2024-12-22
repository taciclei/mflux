import mlx.core as mx
from mlx import nn


class SingleBlockAttention(nn.Module):
    head_dimension = 128
    batch_size = 1
    num_heads = 24

    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(3072, 3072)
        self.to_k = nn.Linear(3072, 3072)
        self.to_v = nn.Linear(3072, 3072)
        self.norm_q = nn.RMSNorm(128)
        self.norm_k = nn.RMSNorm(128)

    def forward(self, hidden_states: mx.array, image_rotary_emb: mx.array) -> (mx.array, mx.array):
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = mx.transpose(mx.reshape(query, (1, -1, 24, 128)), (0, 2, 1, 3))
        key = mx.transpose(mx.reshape(key, (1, -1, 24, 128)), (0, 2, 1, 3))
        value = mx.transpose(mx.reshape(value, (1, -1, 24, 128)), (0, 2, 1, 3))

        query = self.norm_q(query)
        key = self.norm_k(key)

        query, key = SingleBlockAttention.apply_rope(query, key, image_rotary_emb)

        hidden_states = SingleBlockAttention.attention(query, key, value)
        hidden_states = mx.transpose(hidden_states, (0, 2, 1, 3))
        hidden_states = mx.reshape(
            hidden_states,
            (self.batch_size, -1, self.num_heads * self.head_dimension),
        )

        return hidden_states

    @staticmethod
    def attention(query, key, value):
        scale = 1 / mx.sqrt(query.shape[-1])
        scores = (query * scale) @ key.transpose(0, 1, 3, 2)
        attn = mx.softmax(scores, axis=-1)
        hidden_states = attn @ value
        return hidden_states

    @staticmethod
    def apply_rope(xq: mx.array, xk: mx.array, freqs_cis: mx.array):
        # Split last dimension into real and imaginary parts
        # Original shape: (batch, num_heads, seq_len, head_dim)
        # After reshape: (batch, num_heads, seq_len, head_dim//2, 2)
        xq_ = xq.astype(mx.float32).reshape(*xq.shape[:-1], xq.shape[-1] // 2, 2)
        xk_ = xk.astype(mx.float32).reshape(*xk.shape[:-1], xk.shape[-1] // 2, 2)

        # Apply rotary embeddings
        real_part = freqs_cis[..., 0] * xq_[..., 0] - freqs_cis[..., 1] * xq_[..., 1]
        imag_part = freqs_cis[..., 0] * xq_[..., 1] + freqs_cis[..., 1] * xq_[..., 0]
        xq_out = mx.stack([real_part, imag_part], axis=-1)

        real_part = freqs_cis[..., 0] * xk_[..., 0] - freqs_cis[..., 1] * xk_[..., 1]
        imag_part = freqs_cis[..., 0] * xk_[..., 1] + freqs_cis[..., 1] * xk_[..., 0]
        xk_out = mx.stack([real_part, imag_part], axis=-1)

        # Reshape back to original dimensions
        # From (batch, num_heads, seq_len, head_dim//2, 2) to (batch, num_heads, seq_len, head_dim)
        return (mx.reshape(xq_out, xq.shape).astype(mx.float32), mx.reshape(xk_out, xk.shape).astype(mx.float32))
