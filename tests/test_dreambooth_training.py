"""
Tests for DreamBooth training functionality.

This module contains tests for the DreamBooth training pipeline, including
model initialization, training loop, and output validation.
"""

import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.dreambooth_training.config import DreamBoothTrainingConfig
from mflux.dreambooth_training.dreambooth_trainer import DreamBoothTrainer
from mflux.dreambooth_training.model import DreamBoothModel
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.transformer.transformer import Transformer
from mflux.models.vae.vae import VAE


@pytest.fixture
def test_dirs():
    """Create and manage test directories.

    Yields:
        dict: Dictionary containing test directory paths
    """
    instance_dir = Path("./test_instance_data")
    output_dir = Path("./test_output")

    # Create directories
    instance_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Create test image
    image = np.random.normal(0, 1, (512, 512, 3)).astype(np.float32)
    image_path = instance_dir / "test_image.npy"
    np.save(str(image_path), image)

    yield {"instance_dir": instance_dir, "output_dir": output_dir, "image_path": image_path}

    # Cleanup
    if instance_dir.exists():
        shutil.rmtree(instance_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture
def model_fixtures():
    """Create model components for testing.

    Returns:
        tuple: (transformer, text_encoder, vae, model)
    """
    transformer = Transformer(ModelConfig.FLUX1_SCHNELL)
    text_encoder = T5Encoder()
    vae = VAE()
    model = DreamBoothModel(transformer=transformer, text_encoder=text_encoder, vae=vae)
    return transformer, text_encoder, vae, model


def create_test_dataset(batch_size: int = 1) -> list:
    """Create a dummy dataset for testing.

    Args:
        batch_size: Batch size for the dataset

    Returns:
        list: List of (image, prompt_embeds, pooled_prompt_embeds) tuples
    """
    hidden_size = 64
    sequence_length = 77  # Standard length for text tokens
    channels = 4
    height = width = 32  # Reduced size for testing

    image = mx.random.normal((batch_size, channels, height, width))
    prompt_embeds = mx.random.normal((batch_size, sequence_length, hidden_size))
    pooled_prompt_embeds = mx.random.normal((batch_size, hidden_size))
    return [(image, prompt_embeds, pooled_prompt_embeds)]


def test_model_initialization(model_fixtures):
    """Test model component initialization."""
    transformer, text_encoder, vae, model = model_fixtures

    assert isinstance(transformer, Transformer), "Transformer initialization failed"
    assert isinstance(text_encoder, T5Encoder), "Text encoder initialization failed"
    assert isinstance(vae, VAE), "VAE initialization failed"
    assert isinstance(model, DreamBoothModel), "DreamBooth model initialization failed"


def test_training_config(test_dirs):
    """Test training configuration initialization."""
    config = DreamBoothTrainingConfig(
        model_config=ModelConfig.FLUX1_SCHNELL,
        instance_data_dir=str(test_dirs["instance_dir"]),
        output_dir=str(test_dirs["output_dir"]),
        instance_prompt="a photo of sks",
        train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        lr_scheduler="cosine",
        lr_warmup_steps=100,
        max_train_steps=1000,
        guidance=7.5,
    )

    assert config.train_batch_size == 1, "Incorrect batch size"
    assert config.learning_rate == 5e-6, "Incorrect learning rate"
    assert config.guidance == 7.5, "Incorrect guidance value"
    assert isinstance(config.sigmas, mx.array), "Sigmas not properly initialized"


def test_training_loop(test_dirs, model_fixtures):
    """Test the complete training loop."""
    _, _, _, model = model_fixtures

    # Setup configuration
    config = DreamBoothTrainingConfig(
        model_config=ModelConfig.FLUX1_SCHNELL,
        instance_data_dir=str(test_dirs["instance_dir"]),
        output_dir=str(test_dirs["output_dir"]),
        instance_prompt="a photo of sks",
        train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        lr_scheduler="cosine",
        lr_warmup_steps=100,
        max_train_steps=1000,
        guidance=7.5,
    )
    runtime_config = RuntimeConfig(config=config, model_config=ModelConfig.FLUX1_SCHNELL)

    # Create trainer
    dataset = create_test_dataset()
    trainer = DreamBoothTrainer(config=config, model=model, dataset=dataset)

    # Run training
    trainer.train(num_steps=2, runtime_config=runtime_config)

    # Verify outputs
    assert test_dirs["output_dir"].exists(), "Output directory not created"
    assert any(test_dirs["output_dir"].iterdir()), "No output files generated"


def test_model_inference(model_fixtures):
    """Test model inference with dummy inputs."""
    _, _, _, model = model_fixtures

    # Setup inputs
    batch_size = 1
    hidden_size = 64
    sequence_length = 77  # Standard length for text tokens
    channels = 4
    height = width = 32  # Reduced size for testing

    test_input = mx.random.normal((batch_size, 24, height, width))
    timestep = mx.array([0], dtype=mx.float32)
    prompt_embeds = mx.random.normal((batch_size, sequence_length, hidden_size))
    pooled_prompt_embeds = mx.random.normal((batch_size, hidden_size))

    # Run inference
    config = RuntimeConfig(
        config=DreamBoothTrainingConfig(
            model_config=ModelConfig.FLUX1_SCHNELL,
            instance_data_dir="dummy",
            output_dir="dummy",
            instance_prompt="dummy",
            train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=5e-6,
            lr_scheduler="cosine",
            lr_warmup_steps=100,
            max_train_steps=1000,
            guidance=7.5,
        ),
        model_config=ModelConfig.FLUX1_SCHNELL,
    )

    output = model(timestep, prompt_embeds, pooled_prompt_embeds, test_input, config)

    # Verify output shape
    expected_shape = (batch_size, 24, height, width)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
