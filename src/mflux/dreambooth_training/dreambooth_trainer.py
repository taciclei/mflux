"""
DreamBooth Trainer module for fine-tuning diffusion models.

This module provides the DreamBoothTrainer class which handles the training loop,
gradient computation, and model optimization for DreamBooth fine-tuning.
"""

import os
from typing import Iterator, Tuple

import mlx.core as mx
import mlx.nn
import mlx.optimizers
from mlx.nn.losses import mse_loss
from tqdm import tqdm

from mflux.config.runtime_config import RuntimeConfig
from mflux.dreambooth_training.config import DreamBoothTrainingConfig
from mflux.dreambooth_training.model import DreamBoothModel


class DreamBoothTrainer:
    """Trainer class for DreamBooth fine-tuning of diffusion models."""

    def __init__(
        self,
        config: DreamBoothTrainingConfig,
        model: DreamBoothModel,
        dataset: Iterator[Tuple[mx.array, mx.array, mx.array]],
    ) -> None:
        """Initialize the DreamBooth trainer.

        Args:
            config: Training configuration
            model: DreamBooth model to train
            dataset: Training dataset iterator
        """
        self.config = config
        self.model = model
        self.dataset = dataset
        self.optimizer = mlx.optimizers.AdamW(learning_rate=config.learning_rate)
        self.max_grad_norm = 1.0

    def compute_loss(self, params, x, t, prompt_embeds, pooled_prompt_embeds, config):
        """Compute loss for training.

        Args:
            params: Model parameters
            x: Input tensor (B, C, H, W)
            t: Timestep
            prompt_embeds: Text prompt embeddings
            pooled_prompt_embeds: Pooled text embeddings
            config: Runtime configuration

        Returns:
            Loss value
        """
        # Add noise to the input images
        noise = mx.random.normal(x.shape)
        noisy_x = x + t[:, None, None, None] * noise

        # Predict noise using the model with the current parameters
        def forward_fn(model, x, t, prompt_embeds, pooled_prompt_embeds, config):
            noise_pred = model(x, t, prompt_embeds, pooled_prompt_embeds, config)
            return mse_loss(noise_pred, noise)

        loss = forward_fn(self.model, noisy_x, t, prompt_embeds, pooled_prompt_embeds, config)
        return loss

    def train(self, num_steps: int, runtime_config: RuntimeConfig) -> None:
        """Train the model.

        Args:
            num_steps: Number of training steps
            runtime_config: Runtime configuration
        """
        self.model.train()
        optimizer = mlx.optimizers.Adam(learning_rate=self.config.learning_rate)

        # Create grad function
        grad_fn = mx.grad(self.compute_loss)

        losses = []
        progress_bar = tqdm(range(num_steps), desc="Training")

        dataset_iterator = iter(self.dataset)
        for step in progress_bar:
            # Get a batch from the dataset
            try:
                batch = next(dataset_iterator)
            except StopIteration:
                dataset_iterator = iter(self.dataset)
                batch = next(dataset_iterator)

            image, prompt_embeds, pooled_prompt_embeds = batch

            # Ensure correct dimensions
            if prompt_embeds.ndim == 2:
                prompt_embeds = mx.expand_dims(prompt_embeds, axis=0)
            if pooled_prompt_embeds.ndim == 1:
                pooled_prompt_embeds = mx.expand_dims(pooled_prompt_embeds, axis=0)
            if image.ndim == 3:
                image = mx.expand_dims(image, axis=0)

            # Sample timesteps
            timesteps_orig = mx.random.randint(0, self.config.num_train_steps, (1,))
            timesteps = mx.array([runtime_config.timestep_map[int(t.item())] for t in timesteps_orig], dtype=mx.float32)

            # Get model parameters
            params = self.model.parameters()

            # Calculate loss and gradients
            grads = grad_fn(
                params,
                image,
                timesteps,
                prompt_embeds,
                pooled_prompt_embeds,
                runtime_config,
            )
            loss = self.compute_loss(
                params,
                image,
                timesteps,
                prompt_embeds,
                pooled_prompt_embeds,
                runtime_config,
            )

            if mx.isnan(loss):
                raise ValueError("Loss is NaN")

            # Update model parameters
            for k, g in grads.items():
                if g is not None:
                    params[k] = params[k] - self.config.learning_rate * g

            # Update progress bar
            losses.append(loss.item())
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Save checkpoint if needed
            if step > 0 and step % self.config.save_steps == 0:
                self._save_checkpoint(step, avg_loss)

            # Evaluate if needed
            if step > 0 and step % self.config.eval_steps == 0:
                self.evaluate(runtime_config)

        # Final save
        self._save_checkpoint(num_steps, avg_loss)

    def evaluate(self, runtime_config: RuntimeConfig) -> float:
        """Evaluate the model on the validation set.

        Args:
            runtime_config: Runtime configuration

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Run evaluation
        for batch in self.dataset:
            image, prompt_embeds, pooled_prompt_embeds = batch

            # Ensure correct dimensions
            if prompt_embeds.ndim == 2:
                prompt_embeds = mx.expand_dims(prompt_embeds, axis=0)
            if pooled_prompt_embeds.ndim == 1:
                pooled_prompt_embeds = mx.expand_dims(pooled_prompt_embeds, axis=0)
            if image.ndim == 3:
                image = mx.expand_dims(image, axis=0)

            # Sample timesteps
            timesteps = mx.random.randint(0, self.config.num_train_steps, (1,))

            # Add noise to the input images
            noise = mx.random.normal(image.shape)
            noisy_image = image + timesteps[:, None, None, None] * noise

            # Predict noise
            noise_pred = self.model(
                noisy_image,
                timesteps,
                prompt_embeds,
                pooled_prompt_embeds,
                runtime_config,
            )

            # Calculate loss
            loss = mse_loss(noise_pred, noise)
            total_loss += loss.item()
            num_batches += 1

            if num_batches >= 10:  # Limit validation to 10 batches
                break

        self.model.train()
        return total_loss / num_batches if num_batches > 0 else float("inf")

    def _save_checkpoint(self, step: int, loss: float) -> None:
        """Save a model checkpoint.

        Args:
            step: Current training step
            loss: Current loss value
        """
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model state
        model_path = os.path.join(checkpoint_dir, "model.npz")
        mx.savez(model_path, **dict(self.model.transformer.named_parameters()))

        # Save training info
        with open(os.path.join(checkpoint_dir, "training_info.txt"), "w") as f:
            f.write(f"step: {step}\n")
            f.write(f"loss: {loss}\n")

        print(f"Saved checkpoint to {checkpoint_dir}")
