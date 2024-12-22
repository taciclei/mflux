"""DreamBooth trainer module."""

import os

import mlx.core as mx
import mlx.optimizers as optim

from mflux.config.runtime_config import RuntimeConfig
from mflux.models.transformer.transformer import Transformer

from .dataset import DreamBoothDataset


class DreamBoothTrainer:
    """DreamBooth trainer class."""

    def __init__(
        self,
        model: Transformer,
        config: RuntimeConfig,
        dataset: DreamBoothDataset,
        output_dir: str,
    ):
        """Initialize trainer.

        Args:
            model: Transformer model
            config: Runtime configuration
            dataset: DreamBooth dataset
            output_dir: Output directory
        """
        self.model = model
        self.config = config
        self.dataset = dataset
        self.output_dir = output_dir

        # Create optimizer
        self.optimizer = optim.Adam(
            learning_rate=config.learning_rate,
        )

    def _mse_loss(self, output: mx.array, target: mx.array) -> mx.array:
        """Calculate MSE loss.

        Args:
            output: Model output
            target: Target output

        Returns:
            MSE loss
        """
        diff = output - target
        return mx.mean(diff * diff)

    def _loss_fn(self, model_params):
        """Loss function.

        Args:
            model_params: Model parameters

        Returns:
            Loss value
        """
        # Update model parameters
        self.model.update(model_params)

        # Get batch
        batch = next(self.dataset)

        # Encode image with VAE
        latents = self.dataset.vae.encode(batch["pixel_values"])

        # Forward pass through transformer
        output = self.model(
            latents,  # Use VAE latents instead of raw pixels
            batch["input_ids"],
            batch["text_embeds"],
            batch["pooled_text_embeds"],
        )

        # Calculate loss against original latents
        loss = self._mse_loss(output, latents)

        return loss

    def train(self, max_train_steps: int, config: RuntimeConfig) -> None:
        """Train the model.

        Args:
            max_train_steps: Maximum number of training steps
            config: Runtime configuration
        """
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Training loop
        for step in range(max_train_steps):
            # Get gradients
            loss_and_grad_fn = mx.value_and_grad(self._loss_fn)
            loss, grads = loss_and_grad_fn(self.model.parameters())

            # Update parameters
            self.optimizer.update(self.model.parameters(), grads)

            # Save checkpoint
            if step % config.save_steps == 0:
                self._save_checkpoint(step)

            # Print progress
            if step % config.eval_steps == 0:
                print(f"Step {step}: loss = {loss}")

    def _save_checkpoint(self, step: int) -> None:
        """Save checkpoint.

        Args:
            step: Current training step
        """
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Convert parameters to numpy arrays
        params = self.model.parameters()
        param_dict = {}
        for i, (name, param) in enumerate(params.items()):
            try:
                # Get the raw value of the parameter
                if isinstance(param, dict):
                    # For nested parameters (like in Sequential)
                    for subname, subparam in param.items():
                        if hasattr(subparam, "value"):
                            param_dict[f"{name}.{subname}"] = subparam.value
                else:
                    # For direct parameters
                    if hasattr(param, "value"):
                        param_dict[name] = param.value
            except Exception as e:
                print(f"Error converting parameter {name}: {e}")
                continue

        # Save model weights
        try:
            mx.savez(
                os.path.join(checkpoint_dir, "weights.npz"),
                **param_dict,
            )
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
