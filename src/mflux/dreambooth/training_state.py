from typing import Callable, Optional

from mflux.dreambooth.dataset.dataset import DreamBoothDataset


class TrainingState:
    def __init__(self, dataset: DreamBoothDataset):
        self.dataset = dataset
        self.iterator = iter(dataset)
        self.progress_callback: Optional[Callable[[int, float], None]] = None
        self.current_step = 0

    def update_progress(self, step: int, loss: float):
        """Met à jour la progression de l'entraînement"""
        self.current_step = step
        if self.progress_callback:
            self.progress_callback(step, loss)
