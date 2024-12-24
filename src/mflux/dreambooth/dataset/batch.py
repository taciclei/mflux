from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx


@dataclass
class Example:
    """Un example d'entraînement pour DreamBooth"""

    example_id: int
    prompt: str
    image_path: Path
    encoded_image: mx.array
    prompt_embeds: mx.array
    pooled_prompt_embeds: mx.array

    def to_dict(self) -> dict:
        """Convertit l'example en dictionnaire pour l'entraînement"""
        return {
            "example_id": self.example_id,
            "prompt": self.prompt,
            "image_path": str(self.image_path),
            "encoded_image": self.encoded_image,
            "prompt_embeds": self.prompt_embeds,
            "pooled_prompt_embeds": self.pooled_prompt_embeds,
        }


@dataclass
class Batch:
    """Un lot d'examples pour l'entraînement"""

    examples: list[Example]

    def to_dict(self) -> dict:
        """Convertit le lot en dictionnaire pour l'entraînement"""
        try:
            if not self.examples:
                raise ValueError("Le lot est vide")

            # Vérifier que tous les tenseurs sont valides
            for example in self.examples:
                if not all(
                    [
                        example.encoded_image is not None,
                        example.prompt_embeds is not None,
                        example.pooled_prompt_embeds is not None,
                    ]
                ):
                    raise ValueError("Tenseurs manquants dans l'example")

            # Concaténer tous les tenseurs du lot
            encoded_images = mx.stack([ex.encoded_image for ex in self.examples])
            prompt_embeds = mx.stack([ex.prompt_embeds for ex in self.examples])
            pooled_prompt_embeds = mx.stack([ex.pooled_prompt_embeds for ex in self.examples])

            return {
                "encoded_images": encoded_images,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
            }

        except Exception as e:
            print(f"Erreur lors de la conversion du lot: {str(e)}")
            return {}
