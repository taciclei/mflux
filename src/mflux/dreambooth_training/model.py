"""
DreamBooth model implementation.

This module provides the DreamBooth model classes that wrap the transformer,
text encoder, and VAE components for fine-tuning.
"""

import mlx.core as mx
import mlx.nn

from mflux.config.runtime_config import RuntimeConfig
from mflux.models.transformer.transformer import Transformer

from .interfaces import VAEInterface


class TransformerWrapper(mlx.nn.Module):
    """Wrapper for the transformer model to handle input preprocessing."""

    def __init__(self, transformer: Transformer) -> None:
        """Initialize the transformer wrapper.

        Args:
            transformer: Base transformer model
        """
        super().__init__()
        self.transformer = transformer

    def predict(
        self,
        t: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        hidden_states: mx.array,
        config: RuntimeConfig,
    ) -> mx.array:
        """Make a prediction using the transformer.

        Args:
            t: Timestep
            prompt_embeds: Text prompt embeddings
            pooled_prompt_embeds: Pooled text embeddings
            hidden_states: Input hidden states
            config: Runtime configuration

        Returns:
            Predicted noise
        """
        if hidden_states.ndim == 3:
            hidden_states = mx.expand_dims(hidden_states, axis=0)

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = mx.reshape(hidden_states, (batch_size, channels, -1))
        hidden_states = mx.transpose(hidden_states, (0, 2, 1))  # (B, H*W, C)

        return self.transformer.predict(t, prompt_embeds, pooled_prompt_embeds, hidden_states, config)


class DreamBoothModel(mlx.nn.Module):
    """Main DreamBooth model combining transformer, text encoder, and VAE."""

    def __init__(self, transformer: Transformer, text_encoder: mlx.nn.Module, vae: VAEInterface) -> None:
        """Initialize the DreamBooth model.

        Args:
            transformer: Base transformer model
            text_encoder: Text encoder model
            vae: VAE model interface
        """
        super().__init__()
        self.transformer = TransformerWrapper(transformer)
        self.text_encoder = text_encoder
        self.vae = vae

    def noise_prediction(
        self,
        x: mx.array,
        t: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        config: RuntimeConfig,
    ) -> mx.array:
        """Predict noise for the input tensor.

        Args:
            x: Input tensor (B, C, H, W)
            t: Timestep
            prompt_embeds: Text prompt embeddings
            pooled_prompt_embeds: Pooled text embeddings
            config: Runtime configuration

        Returns:
            Predicted noise
        """
        # Ensure input has the right shape (B, C, H, W)
        if x.ndim == 3:
            x = mx.expand_dims(x, axis=0)

        # First encode the images through VAE to get latents
        latents = self.vae.encode(x)  # (B, C, H, W)

        # Predict noise
        noise_pred = self.transformer.predict(
            t,
            prompt_embeds,
            pooled_prompt_embeds,
            latents,
            config,
        )

        return noise_pred

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        config: RuntimeConfig,
    ) -> mx.array:
        """Forward pass through the model.

        Args:
            x: Input tensor (B, C, H, W)
            t: Timestep
            prompt_embeds: Text prompt embeddings
            pooled_prompt_embeds: Pooled text embeddings
            config: Runtime configuration

        Returns:
            Model prediction
        """
        return self.noise_prediction(x, t, prompt_embeds, pooled_prompt_embeds, config)
