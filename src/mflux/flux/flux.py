import logging
from pathlib import Path

import mlx.core as mx
import safetensors.numpy

from mflux.config.config import Config
from mflux.config.runtime_config import RuntimeConfigValidation
from mflux.flux.transformer import Transformer

log = logging.getLogger(__name__)


class DummyTokenizer:
    def tokenize(self, text: str) -> mx.array:
        # Return a random tensor to simulate tokens
        return mx.random.normal((1, 77, 768))


class DummyEncoder:
    def __call__(self, tokens: mx.array) -> mx.array:
        # Return a random tensor to simulate embeddings
        return tokens


class DummyVAE:
    def encode(self, image: mx.array) -> mx.array:
        # Return a random tensor to simulate latents
        return mx.random.normal((1, 4, 64, 64))


class Flux:
    """Main class for the diffusion model"""

    def __init__(
        self,
        model_path: Path,
        runtime_config: RuntimeConfigValidation,
        config: Config | None = None,
    ):
        self.model_path = model_path
        self.runtime_config = runtime_config
        self.config = config

        # Create the transformer
        self.transformer = Transformer(config=config)

        # Create the tokenizers and encoders
        self.t5_tokenizer = DummyTokenizer()
        self.clip_tokenizer = DummyTokenizer()
        self.t5_text_encoder = DummyEncoder()
        self.clip_text_encoder = DummyEncoder()
        self.vae = DummyVAE()

        # Load the model parameters
        if model_path.exists():
            try:
                params = safetensors.numpy.load_file(str(model_path))
                if params:
                    for name, value in params.items():
                        if name.startswith("transformer."):
                            # Convert the parameters to MLX tensors
                            value = mx.array(value).astype(mx.float32)
                            # Assign the parameter to the transformer
                            self._set_parameter(name[len("transformer.") :], value)
            except Exception as e:
                log.error(f"Error loading model: {str(e)}")

        # Initialize the parameters
        mx.eval(self.transformer.parameters())

    def _set_parameter(self, name: str, value: mx.array):
        """Assign a parameter to the transformer"""
        try:
            # Split the name into components
            components = name.split(".")

            # Find the target module
            target = self.transformer
            for comp in components[:-1]:
                if hasattr(target, comp):
                    target = getattr(target, comp)
                else:
                    print(f"Warning: Module {comp} not found")
                    return

            # Assign the parameter
            if hasattr(target, components[-1]):
                setattr(target, components[-1], value)
            else:
                print(f"Warning: Parameter {components[-1]} not found")

        except Exception as e:
            print(f"Error assigning parameter: {str(e)}")

    def trainable_parameters(self) -> dict:
        """Return the trainable parameters of the model"""
        try:
            # Get the transformer parameters
            params = {}
            transformer_params = self.transformer.trainable_parameters()

            if not transformer_params:
                print("Warning: No trainable parameters found in the transformer")
                return {}

            for name, value in transformer_params.items():
                if isinstance(value, mx.array):
                    params[f"transformer.{name}"] = value.astype(mx.float32)

            if not params:
                print("Warning: No trainable parameters found")
                return {}

            return params

        except Exception as e:
            print(f"Error retrieving parameters: {str(e)}")
            return {}

    def compute_loss(
        self,
        params: dict,
        encoded_images: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        guidance: float,
        config=None,
    ) -> mx.array:
        """Compute the loss for Dreambooth training"""
        try:
            # Check that the tensors are valid
            if not all([encoded_images is not None, prompt_embeds is not None, pooled_prompt_embeds is not None]):
                print("Warning: Missing tensors")
                return mx.array(0.0, dtype=mx.float32)

            # Predict the noise
            noise = self.transformer(
                t=0,  # TODO: Add the timestep
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                hidden_states=encoded_images,
                config=config,
            )

            if noise is None:
                return mx.array(0.0, dtype=mx.float32)

            # Calculate the MSE loss
            target = mx.zeros_like(noise)  # TODO: Add the target noise
            loss = mx.mean((noise - target) ** 2)

            if guidance != 1.0:
                loss = loss * guidance

            return loss

        except Exception as e:
            print(f"Error computing loss: {str(e)}")
            return mx.array(0.0, dtype=mx.float32)

    def train_step(self, batch):
        """Performs an optimized training step for MLX on Mac"""
        try:
            # Get batch data
            latents = batch.latents
            noise = batch.noise
            timesteps = batch.timesteps

            # Vectorized loss function
            def loss_fn(params):
                self.transformer.update(params)

                # Predict noise (automatically uses GPU)
                noise_pred = self.transformer(
                    timesteps,
                    batch.prompt_embeds,
                    batch.pooled_prompt_embeds,
                    latents,
                )

                if noise_pred is None:
                    return mx.array(float("inf"), dtype=mx.float32)

                # Calculate MSE loss in a vectorized way
                loss = mx.mean((noise_pred - noise) ** 2)
                return loss

            # Compile loss function for speed
            loss_fn = mx.compile(loss_fn)

            # Calculate loss and gradients asynchronously
            loss, grads = mx.value_and_grad(loss_fn)(self.transformer.parameters())

            # Update parameters with optimizer
            self.opt.update(self.transformer, grads)

            # Evaluate pending calculations
            mx.eval(loss)
            mx.eval(self.transformer.parameters())

            # Return loss
            return loss

        except Exception as e:
            print(f"Error during training: {str(e)}")
            return mx.array(0.0, dtype=mx.float32)

    def save_checkpoint(self, params: dict, path: Path):
        """Save the model parameters to a safetensor file"""
        try:
            import numpy as np

            # Ensure all calculations are finished
            mx.eval(params)

            # Convert parameters to numpy arrays in an optimized way
            numpy_params = {}
            for name, param in params.items():
                if isinstance(param, mx.array):
                    # Convert to numpy array via tolist
                    param_list = param.astype(mx.float32).tolist()
                    numpy_params[name] = np.array(param_list)

            # Save the parameters
            safetensors.numpy.save_file(numpy_params, str(path))

        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
