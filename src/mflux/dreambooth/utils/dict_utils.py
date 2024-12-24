from typing import Any, Dict

import mlx.core as mx
import numpy as np


def update_nested_dict(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """Met à jour les valeurs dans des dictionnaires imbriqués"""
    result = {}
    for k in d1.keys():
        if k in d2:
            if isinstance(d1[k], dict) and isinstance(d2[k], dict):
                result[k] = update_nested_dict(d1[k], d2[k])
            else:
                result[k] = d1[k] + d2[k]
        else:
            result[k] = d1[k]
    return result


def scale_nested_dict(d: Dict[str, Any], scale: float) -> Dict[str, Any]:
    """Applique une mise à l'échelle aux valeurs dans un dictionnaire imbriqué"""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = scale_nested_dict(v, scale)
        elif isinstance(v, (mx.array, np.ndarray)):
            result[k] = v / scale
        else:
            result[k] = v
    return result
