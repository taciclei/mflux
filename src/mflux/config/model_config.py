"""Model configuration."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration class."""

    model_name: str
    alias: str
    num_train_steps: int
    max_sequence_length: int
    height: int
    width: int
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    patch_size: int = 8
    in_channels: int = 4
    out_channels: int = 4

    @staticmethod
    def from_alias(alias: str) -> "ModelConfig":
        """Create model config from alias.

        Args:
            alias: Model alias

        Returns:
            Model config
        """
        if alias.lower() == "schnell":
            return ModelConfig(
                model_name="black-forest-labs/FLUX.1-schnell",
                alias="schnell",
                num_train_steps=1000,
                max_sequence_length=256,
                height=64,
                width=64,
            )
        elif alias.lower() == "dev":
            return ModelConfig(
                model_name="black-forest-labs/FLUX.1-dev",
                alias="dev",
                num_train_steps=1000,
                max_sequence_length=512,
                height=64,
                width=64,
            )
        else:
            raise ValueError(f"Unknown model alias: {alias}")
