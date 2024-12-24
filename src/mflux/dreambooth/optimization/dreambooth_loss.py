import random

import mlx.core as mx

from mflux import Config, Flux1
from mflux.config.runtime_config import RuntimeConfig
from mflux.dreambooth.dataset.batch import Batch
from mflux.dreambooth.dataset.dataset import Example
from mflux.latent_creator.latent_creator import LatentCreator


class DreamBoothLoss:
    @staticmethod
    def compute_loss(flux: Flux1, config: RuntimeConfig, batch: Batch) -> mx.array:
        # Compute losses for all examples in the batch
        losses = [DreamBoothLoss._single_example_loss(flux, config, example, batch.rng) for example in batch.examples]
        # Stack losses into a single tensor and take the mean
        return mx.mean(mx.stack(losses))

    @staticmethod
    def _single_example_loss(flux: Flux1, config: RuntimeConfig, example: Example, rng: random.Random) -> mx.array:
        # Must be a better way to handle the randomness than this, but we already
        # save/restore the random state via the iterator so this is a continent shortcut.
        time_seed = rng.randint(0, 2**32 - 1)
        noise_seed = rng.randint(0, 2**32 - 1)

        # Draw a random timestep t from [0, num_inference_steps]
        t = mx.random.randint(low=0, high=config.num_inference_steps, shape=[], key=mx.random.key(time_seed))

        # Get the clean image latent
        clean_image = example.clean_latents

        # Generate pure noise
        pure_noise = mx.random.normal(shape=clean_image.shape, dtype=Config.precision, key=mx.random.key(noise_seed))

        # By linear interpolation between the clean image and pure noise, construct a latent at time t
        latents_t = LatentCreator.add_noise_by_interpolation(
            clean=clean_image, noise=pure_noise, sigma=config.sigmas[int(t)]
        )

        # Predict the noise from timestep t
        predicted_noise = flux.transformer.predict(
            t=t,
            prompt_embeds=example.prompt_embeds,
            pooled_prompt_embeds=example.pooled_prompt_embeds,
            hidden_states=latents_t,
            config=config,
        )

        # Construct the loss (derivation in src/mflux/dreambooth/optimization/_loss_derivation)
        # Convert to scalar by taking mean over all dimensions
        return mx.mean(mx.square(clean_image + predicted_noise - pure_noise))
