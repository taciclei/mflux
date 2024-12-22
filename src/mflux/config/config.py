import logging
from pathlib import Path

import mlx.core as mx

log = logging.getLogger(__name__)


class Config:
    precision: mx.Dtype = mx.bfloat16

    def __init__(
        self,
        num_inference_steps: int = 4,
        width: int = 1024,
        height: int = 1024,
        guidance: float = 4.0,
        init_image_path: Path | None = None,
        init_image_strength: float | None = None,
    ):
        if width % 16 != 0 or height % 16 != 0:
            log.warning("Width and height should be multiples of 16. Rounding down.")
        self.width = 16 * (width // 16)
        self.height = 16 * (height // 16)
        self.num_inference_steps = num_inference_steps
        self.guidance = guidance
        self.init_image_path = init_image_path
        self.init_image_strength = init_image_strength


class ConfigControlnet(Config):
    def __init__(
        self,
        num_inference_steps: int = 4,
        width: int = 1024,
        height: int = 1024,
        guidance: float = 4.0,
        controlnet_strength: float = 1.0,
    ):
        super().__init__(num_inference_steps=num_inference_steps, width=width, height=height, guidance=guidance)
        self.controlnet_strength = controlnet_strength


class DreamBoothConfig(Config):
    def __init__(
        self,
        model_name: str,
        instance_data_dir: str,
        output_dir: str,
        instance_prompt: str,
        train_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 5e-6,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        max_train_steps: int = 2,
        guidance: float = 8.0,
        height: int = 512,
        width: int = 512,
    ):
        super().__init__(guidance=guidance, height=height, width=width)
        self.model_name = model_name
        self.instance_data_dir = instance_data_dir
        self.output_dir = output_dir
        self.instance_prompt = instance_prompt
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.max_train_steps = max_train_steps

    @property
    def num_train_steps(self) -> int:
        return self.max_train_steps
