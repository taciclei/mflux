"""Runtime configuration."""

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class RuntimeConfig:
    """Runtime configuration for training."""

    guidance_scale: float
    batch_size: int
    learning_rate: float
    num_warmup_steps: int
    save_steps: int
    eval_steps: int
    max_grad_norm: float
    height: int
    width: int
    in_channels: int
    out_channels: int

    def __post_init__(self):
        """Initialize runtime config."""
        # Convert learning rate to MLX array
        self.learning_rate = mx.array(self.learning_rate, dtype=mx.float32)

        # Create timestep map
        self.timestep_map = mx.array(
            mx.linspace(0.00085**0.5, 0.012**0.5, 1000, dtype=mx.float32) ** 2
        )
