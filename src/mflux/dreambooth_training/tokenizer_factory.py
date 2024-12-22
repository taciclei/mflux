from mflux.tokenizer.clip_tokenizer import TokenizerCLIP
from mflux.tokenizer.t5_tokenizer import TokenizerT5
from mflux.tokenizer.tokenizer_handler import TokenizerHandler


class TokenizerFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizers = TokenizerHandler(model_name)

    def create_tokenizer(self, text_encoder_type: str, model_config=None):
        if text_encoder_type == "clip":
            return TokenizerCLIP(self.tokenizers.clip)
        elif text_encoder_type == "t5":
            if model_config is None:
                raise ValueError("model_config must be provided for t5 tokenizer")
            return TokenizerT5(self.tokenizers.t5, max_length=model_config.max_sequence_length)
        else:
            raise ValueError(f"Unknown text_encoder_type: {text_encoder_type}")
