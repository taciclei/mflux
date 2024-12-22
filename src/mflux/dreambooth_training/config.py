"""
Configuration classes for DreamBooth training.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Base configuration class."""

    guidance: float
    num_inference_steps: int
    num_train_steps: int


@dataclass
class DreamBoothTrainingConfig:
    """Configuration for DreamBooth training."""

    instance_data_dir: str
    output_dir: str
    instance_prompt: str
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 50
    max_train_steps: int = 800
    guidance: float = 7.5
