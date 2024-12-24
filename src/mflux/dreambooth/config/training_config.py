from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    """Configuration pour l'optimiseur"""

    learning_rate: float = 1e-6
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8


@dataclass
class MemoryConfig:
    """Configuration pour l'optimisation de la mémoire"""

    clear_cache_interval: int = 10  # Intervalle pour nettoyer le cache MLX
    garbage_collection_interval: int = 50  # Intervalle pour le garbage collection
    memory_efficient_attention: bool = True  # Utiliser l'attention optimisée en mémoire
    use_fp16: bool = True  # Utiliser la précision mixte pour les Mac M1/M2


@dataclass
class DreamBoothConfig:
    """Configuration pour l'entraînement DreamBooth"""

    learning_rate: float = 5e-6
    lr_scheduler: str = "constant"  # "constant", "cosine", "linear"
    lr_warmup_steps: int = 0
    max_train_steps: int = 1000
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    save_steps: int = 100

    # Paramètres de mémoire
    clear_cache_interval: int = 10  # Nettoyer le cache tous les N pas

    def __post_init__(self):
        """Validation des paramètres"""
        if self.learning_rate <= 0:
            raise ValueError("Le taux d'apprentissage doit être positif")

        if self.lr_scheduler not in ["constant", "cosine", "linear"]:
            raise ValueError("Scheduler non supporté")

        if self.lr_warmup_steps < 0:
            raise ValueError("Le nombre de pas de warmup doit être positif ou nul")

        if self.max_train_steps <= 0:
            raise ValueError("Le nombre de pas d'entraînement doit être positif")

        if self.train_batch_size <= 0:
            raise ValueError("La taille du lot doit être positive")

        if self.gradient_accumulation_steps <= 0:
            raise ValueError("Le nombre de pas d'accumulation doit être positif")

        if self.save_steps <= 0:
            raise ValueError("L'intervalle de sauvegarde doit être positif")
