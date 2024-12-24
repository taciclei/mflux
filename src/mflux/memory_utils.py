"""Utilitaires pour la gestion de la mémoire"""

import gc

import mlx.core as mx


def optimize_memory_usage():
    """Optimise l'utilisation de la mémoire"""
    # Forcer la collection des objects inutilisés
    gc.collect()

    # Vider le cache MLX
    mx.clear_kernel_cache()
    mx.clear_call_cache()

    # Vider le cache du compilateur
    mx.clear_compiler_cache()
