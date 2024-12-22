import unittest

import mlx.core as mx

from mflux.config.model_config import ModelConfig
from mflux.dreambooth_training.dummy_vae import VAE
from mflux.dreambooth_training.main import DreamBoothTrainingConfig, loss_fn  # importer les functions à tester
from mflux.dreambooth_training.model import DreamBoothModel
from mflux.dreambooth_training.vae_adapter import VAEAdapter
from mflux.models.transformer.transformer import Transformer


class TestTrainer(unittest.TestCase):
    def test_loss_fn(self):
        model_config = ModelConfig(name="test", num_train_steps=1000, max_sequence_length=256)
        config = DreamBoothTrainingConfig(model_config)
        model = DreamBoothModel(Transformer(model_config), None, VAEAdapter(VAE()))  # VAE factice

        images = [mx.ones((3, 256, 256)) for _ in range(2)]
        prompt_embeds = [mx.ones((77, 768)) for _ in range(2)]
        pooled_prompt_embeds = [mx.ones(768) for _ in range(2)]

        batch = list(zip(images, prompt_embeds, pooled_prompt_embeds))

        timesteps = mx.array([0, 1])
        noise = mx.random.normal((2, 4, 16 * 32))  # Forme correcte pour le bruit

        loss = loss_fn(model, batch, timesteps, noise, config)  # Appeler la function avec config

        self.assertTrue(isinstance(loss, mx.array))  # Vérifier que la function renvoie bien un tenseur


if __name__ == "__main__":
    unittest.main()
