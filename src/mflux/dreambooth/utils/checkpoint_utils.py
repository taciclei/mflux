from typing import Any, Dict, Union

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file


def _mlx_to_numpy(arr: mx.array) -> np.ndarray:
    """Convertit un tableau MLX en numpy array"""
    if arr is None:
        return np.array(0.0, dtype=np.float32)
    return np.array(arr.tolist(), dtype=np.float32)


def _flatten_dict(d: Dict[str, Any], parent_key: str = "") -> Dict[str, np.ndarray]:
    """Aplatit les dictionnaires imbriqués et convertit les tableaux MLX en numpy"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key).items())
        elif isinstance(v, mx.array):
            # Convertir MLX array en numpy avec le bon type
            numpy_arr = _mlx_to_numpy(v)
            items.append((new_key, numpy_arr))
        elif isinstance(v, np.ndarray):
            items.append((new_key, v.astype(np.float32)))
        else:
            continue  # Ignorer les valeurs non-tensorielles

    return dict(items)


def _scale_nested_dict(d: Dict[str, Any], scale: float) -> Dict[str, Any]:
    """Scale values in a nested dictionary"""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _scale_nested_dict(v, scale)
        elif isinstance(v, (mx.array, np.ndarray)):
            result[k] = v / scale
        else:
            result[k] = v
    return result


def _update_nested_dict(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """Update values in nested dictionaries"""
    result = {}
    for k in d1.keys():
        if k in d2:
            if isinstance(d1[k], dict) and isinstance(d2[k], dict):
                result[k] = _update_nested_dict(d1[k], d2[k])
            else:
                result[k] = d1[k] + d2[k]
        else:
            result[k] = d1[k]
    return result


def save_checkpoint(params: Dict[str, Union[Dict, mx.array]], path: str) -> None:
    """Sauvegarde les paramètres du modèle dans un fichier checkpoint"""
    try:
        # Aplatir les dictionnaires imbriqués et convertir en numpy
        numpy_params = _flatten_dict(params)

        # Sauvegarder les paramètres aplatis
        save_file(numpy_params, path)
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du checkpoint: {str(e)}")
        # Continuer sans sauvegarder en cas d'erreur


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Charge les paramètres du modèle depuis un fichier checkpoint"""
    # TODO: Implémenter le chargement des checkpoints
    raise NotImplementedError("Le chargement des checkpoints n'est pas encore implémenté")
