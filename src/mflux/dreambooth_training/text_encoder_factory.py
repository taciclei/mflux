from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.weights.weight_handler import WeightHandler


class TextEncoderFactory:
    def __init__(self, model_name: str, local_path: str = None, lora_paths: list = None, lora_scales: list = None):
        self.weights = WeightHandler(
            repo_id=model_name,
            local_path=local_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    def create_text_encoder(self, text_encoder_type: str):
        if text_encoder_type == "clip":
            text_encoder = CLIPEncoder()
            text_encoder.update(self.weights.clip_encoder)
            return text_encoder
        elif text_encoder_type == "t5":
            text_encoder = T5Encoder()
            text_encoder.update(self.weights.t5_encoder)
            return text_encoder
        else:
            raise ValueError(f"Unknown text_encoder_type: {text_encoder_type}")
