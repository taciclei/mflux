from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from mflux.dreambooth.lora_layers.linear_lora_layer import LoRALinear
from mflux.dreambooth.state.training_spec import SingleTransformerBlocks, TrainingSpec, TransformerBlocks
from mflux.dreambooth.state.zip_util import ZipUtil
from mflux.models.transformer.joint_transformer_block import JointTransformerBlock
from mflux.models.transformer.single_transformer_block import SingleTransformerBlock
from mflux.weights.weight_handler import MetaData, WeightHandler

if TYPE_CHECKING:
    from mflux import Flux1


class LoRALayers:
    def __init__(self, weights: "WeightHandler"):
        self.layers = weights

    @staticmethod
    def from_spec(flux: "Flux1", training_spec: TrainingSpec) -> "LoRALayers":
        if training_spec.lora_layers.state_path is not None:
            # Load from state if present in the spec
            from mflux.weights.weight_handler_lora import WeightHandlerLoRA

            weights = ZipUtil.unzip(
                zip_path=training_spec.checkpoint_path,
                filename=training_spec.lora_layers.state_path,
                loader=lambda x: WeightHandlerLoRA.load_lora_weights(
                    transformer=flux.transformer, lora_files=[x], lora_scales=[1.0]
                ),
            )
            return LoRALayers(weights=weights[0])
        else:
            # Construct the LoRA weights from the spec
            transformer_lora_layers = {}
            single_transformer_lora_layers = {}

            if training_spec.lora_layers.transformer_blocks:
                transformer_lora_layers = LoRALayers._construct_layers(
                    blocks=flux.transformer.transformer_blocks,
                    block_spec=training_spec.lora_layers.transformer_blocks,
                    block_prefix="transformer_blocks",
                )

            if training_spec.lora_layers.single_transformer_blocks:
                single_transformer_lora_layers = LoRALayers._construct_layers(
                    blocks=flux.transformer.single_transformer_blocks,
                    block_spec=training_spec.lora_layers.single_transformer_blocks,
                    block_prefix="single_transformer_blocks",
                )

            lora_layers = {"transformer": {**transformer_lora_layers, **single_transformer_lora_layers}}

            weights = WeightHandler(
                meta_data=MetaData(is_mflux=True),
                transformer=lora_layers["transformer"],
            )  # fmt:off

            return LoRALayers(weights=weights)

    @staticmethod
    def _construct_layers(
        block_spec: TransformerBlocks | SingleTransformerBlocks,
        blocks: list[JointTransformerBlock] | list[SingleTransformerBlock],
        block_prefix: str,
    ) -> dict:
        start = block_spec.block_range.start
        end = block_spec.block_range.end
        lora_layers = {}
        for i in range(start, end):
            block = blocks[i]

            for layer_type in block_spec.layer_types:
                if layer_type == "attn":
                    # Handle attention layers
                    attn = block.attn
                    for attr in [
                        "to_q",
                        "to_k",
                        "to_v",
                        "to_out",
                        "add_q_proj",
                        "add_k_proj",
                        "add_v_proj",
                        "to_add_out",
                    ]:
                        if hasattr(attn, attr):
                            layer = getattr(attn, attr)
                            if isinstance(layer, list):
                                layer = layer[0]
                            if isinstance(layer, nn.Linear):
                                lora_layer = LoRALinear.from_linear(
                                    linear=layer,
                                    r=block_spec.lora_rank,
                                )
                                layer_path = f"{block_prefix}.{i}.{layer_type}.{attr}"
                                lora_layers[layer_path] = lora_layer
                elif layer_type == "mlp":
                    # Handle MLP layers
                    for ff_attr in ["ff", "ff_context"]:
                        if hasattr(block, ff_attr):
                            ff = getattr(block, ff_attr)
                            for mlp_attr in ["linear1", "linear2"]:
                                if hasattr(ff, mlp_attr):
                                    layer = getattr(ff, mlp_attr)
                                    if isinstance(layer, nn.Linear):
                                        lora_layer = LoRALinear.from_linear(
                                            linear=layer,
                                            r=block_spec.lora_rank,
                                        )
                                        layer_path = f"{block_prefix}.{i}.{ff_attr}.{mlp_attr}"
                                        lora_layers[layer_path] = lora_layer

        return lora_layers

    @staticmethod
    def _get_nested_attr(obj, attr):
        """Get a nested attribute from an object using dot notation."""
        if "." in attr:
            parts = attr.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            return getattr(obj, parts[-1])
        return getattr(obj, attr)

    @staticmethod
    def transformer_dict_from_template(weights: dict, transformer: nn.Module, scale: float) -> dict:
        lora_layers = {}
        for key in weights.keys():
            if key.endswith(".lora_A"):
                base_path = key[: -len(".lora_A")]
                parts = base_path.split(".")

                if parts[1] == "transformer_blocks":
                    block = transformer.transformer_blocks[int(parts[2])]
                elif parts[1] == "single_transformer_blocks":
                    block = transformer.single_transformer_blocks[int(parts[2])]
                else:
                    continue

                layer = LoRALayers._get_nested_attr(block, ".".join(parts[3:]))
                if isinstance(layer, list):
                    layer = layer[0]

                lora_layer = LoRALinear.from_linear(
                    linear=layer,
                    r=weights[key].shape[1],
                )
                lora_layer.lora_A = weights[key]
                lora_layer.lora_B = weights[base_path + ".lora_B"]
                lora_layer.scale = scale

                layer_path = ".".join(parts)
                lora_layers[layer_path] = [lora_layer] if isinstance(layer, list) else lora_layer

        return lora_layers

    def save(self, path: Path) -> None:
        state = tree_flatten(self.layers.transformer)
        mx.save_safetensors(str(path), dict(state))
