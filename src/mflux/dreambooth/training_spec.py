from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingSpec:
    """Spécification pour l'entraînement DreamBooth"""

    instance_data_dir: Path
    output_dir: Path
    instance_prompt: str
