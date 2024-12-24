import gc
from typing import Any, Dict, Union

import mlx.core as mx

from mflux.dreambooth.dataset.batch import Batch, Example


def optimize_memory_usage(config):
    """Optimise l'utilisation de la mémoire pour les Mac M1/M2"""
    # Forcer le garbage collection
    gc.collect()


def convert_to_mlx(data: Any) -> Any:
    """Convertit les données en format MLX avec optimisation pour M1/M2"""
    if data is None:
        return None

    if isinstance(data, (list, tuple)):
        return [convert_to_mlx(x) for x in data]
    elif isinstance(data, dict):
        return {k: convert_to_mlx(v) for k, v in data.items()}
    elif hasattr(data, "numpy"):  # Tensor PyTorch
        arr = data.numpy()
        if arr is None:
            return None
        return mx.array(arr, dtype=mx.float32)
    elif hasattr(data, "__array__"):  # Array NumPy
        if data is None:
            return None
        return mx.array(data, dtype=mx.float32)
    elif isinstance(data, mx.array):
        return data
    return data


def example_to_dict(example: Example) -> Dict[str, Any]:
    """Convertit un Example en dictionnaire avec les tenseurs MLX"""
    try:
        example_dict = {
            "example_id": example.example_id,
            "prompt": example.prompt,
            "image_name": example.image_name,
        }

        # Convertir les tenseurs en MLX arrays
        if hasattr(example, "clean_latents") and example.clean_latents is not None:
            example_dict["clean_latents"] = convert_to_mlx(example.clean_latents)
        if hasattr(example, "prompt_embeds") and example.prompt_embeds is not None:
            example_dict["prompt_embeds"] = convert_to_mlx(example.prompt_embeds)
        if hasattr(example, "pooled_prompt_embeds") and example.pooled_prompt_embeds is not None:
            example_dict["pooled_prompt_embeds"] = convert_to_mlx(example.pooled_prompt_embeds)

        return example_dict
    except Exception as e:
        print(f"Erreur lors de la conversion de l'example: {str(e)}")
        return None


def batch_to_dict(batch: Batch) -> Dict[str, Any]:
    """Convertit un batch en dictionnaire optimisé"""
    try:
        examples_data = []
        for example in batch.examples:
            if example is None:
                continue

            example_dict = example_to_dict(example)
            if example_dict is not None:
                examples_data.append(example_dict)

        if not examples_data:
            print("Attention: Aucun example valid dans le batch")
            return {"examples": []}

        return {
            "examples": examples_data,
        }
    except Exception as e:
        print(f"Erreur lors de la conversion du batch: {str(e)}")
        return {"examples": []}


def prepare_batch_for_mlx(batch: Union[Batch, Dict[str, Any]]) -> Dict[str, Any]:
    """Prépare un batch pour MLX avec optimisations M1/M2"""
    try:
        # Si le batch est déjà un dictionnaire, le convertir directement
        if isinstance(batch, dict):
            return {k: convert_to_mlx(v) for k, v in batch.items()}

        # Sinon, convertir le batch en dictionnaire
        batch_dict = batch_to_dict(batch)
        return batch_dict
    except Exception as e:
        print(f"Erreur lors de la préparation du batch: {str(e)}")
        return {"examples": []}
