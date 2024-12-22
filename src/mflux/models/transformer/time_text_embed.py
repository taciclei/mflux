# src/mflux/models/transformer/time_text_embed.py
"""Time and text embedding implementation."""

import math

import mlx.core as mx
import mlx.nn as nn

from mflux.config.model_config import ModelConfig
from mflux.models.transformer.text_embedder import TextEmbedder


class TimeTextEmbed(nn.Module):
    """Time and text embedding module."""

    def __init__(self, config: ModelConfig, embedding_dim: int = 3072):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # Text projection
        self.text_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.text_embedder = TextEmbedder(embedding_dim)

    def forward(self, time_step, pooled_prompt_embeds, guidance):
        time_embeddings = self.time_embed(time_step)

        pooled_projection = self.text_proj(pooled_prompt_embeds)

        pooled_projections = self.text_embedder.forward(pooled_projection)

        if pooled_projections.shape != time_embeddings.shape:
            pooled_projections = mx.reshape(pooled_projections, time_embeddings.shape)

        return time_embeddings + guidance * pooled_projections

    @staticmethod
    def _time_proj(time_steps: mx.array) -> mx.array:
        max_period = 10000
        half_dim = 128
        exponent = -math.log(max_period) * mx.arange(start=0, stop=half_dim, step=None, dtype=mx.float32)
        exponent = exponent / half_dim
        emb = mx.exp(exponent)
        emb = time_steps[:, None].astype(mx.float32) * emb[None, :]
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
        return emb
