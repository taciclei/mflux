"""Main module for DreamBooth training."""

import os
import sys
from argparse import ArgumentParser

from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.text_encoder.text_encoder import TextEncoder
from mflux.models.transformer.transformer import Transformer
from mflux.models.vae.vae import VAE

from .dataset import DreamBoothDataset
from .tokenizer import Tokenizer
from .trainer import DreamBoothTrainer


def main():
    """Main function."""
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        required=True,
        help="Path to directory with instance images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to directory to save model",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
        help="Prompt to use for instance images",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=50,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=800,
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create model config
    model_config = ModelConfig.from_alias(args.model)

    # Create runtime config
    runtime_config = RuntimeConfig(
        guidance_scale=args.guidance,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        num_warmup_steps=args.lr_warmup_steps,
        save_steps=100,
        eval_steps=100,
        max_grad_norm=1.0,
        height=512,
        width=512,
        in_channels=4,
        out_channels=4,
    )

    # Create model components
    tokenizer = Tokenizer()
    text_encoder = TextEncoder()
    vae = VAE()

    # Create dataset
    dataset = DreamBoothDataset(
        args.instance_data_dir,
        512,  # image size
        tokenizer,
        args.instance_prompt,
        text_encoder,
        vae,
    )

    # Create model
    transformer = Transformer(model_config)

    # Create trainer
    trainer = DreamBoothTrainer(
        model=transformer,
        config=runtime_config,
        dataset=dataset,
        output_dir=args.output_dir,
    )

    # Train model
    trainer.train(args.max_train_steps, runtime_config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
