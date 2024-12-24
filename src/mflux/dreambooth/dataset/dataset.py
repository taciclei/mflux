import random
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx


class DreamBoothDataset:
    def __init__(
        self,
        instance_data_dir: Path,
        instance_prompt: str,
        tokenizer_t5,
        tokenizer_clip,
        t5_text_encoder,
        clip_text_encoder,
        vae,
        size: int = 512,
    ):
        self.instance_data_dir = instance_data_dir
        self.instance_prompt = instance_prompt
        self.tokenizer_t5 = tokenizer_t5
        self.tokenizer_clip = tokenizer_clip
        self.t5_text_encoder = t5_text_encoder
        self.clip_text_encoder = clip_text_encoder
        self.vae = vae
        self.size = size

        # Charger les images
        self.image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            self.image_paths.extend(instance_data_dir.glob(ext))

        if not self.image_paths:
            raise ValueError(f"Aucune image trouvée dans {instance_data_dir}")

        self.encoded_examples = []

    def __len__(self) -> int:
        return len(self.image_paths)

    def __iter__(self):
        return iter(self.encoded_examples)

    def encode(self, progress_callback: Optional[Callable[[], None]] = None):
        """Encode toutes les images et les prompts"""
        import numpy as np
        from PIL import Image

        for idx, image_path in enumerate(self.image_paths):
            try:
                # Charger et prétraiter l'image
                image = Image.open(image_path).convert("RGB")
                image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
                image_array = np.array(image, dtype=np.float32) / 255.0  # [H, W, C]
                image_tensor = mx.array(image_array)  # [H, W, C]
                image_tensor = image_tensor.transpose(2, 0, 1)  # [C, H, W]
                image_tensor = mx.expand_dims(image_tensor, axis=0)  # [1, C, H, W]

                # Encoder l'image avec le VAE
                latents = self.vae.encode(image_tensor)
                latents = mx.stop_gradient(latents)

                # Tokeniser et encoder le prompt
                t5_tokens = self.tokenizer_t5.tokenize(self.instance_prompt)
                clip_tokens = self.tokenizer_clip.tokenize(self.instance_prompt)

                prompt_embeds = self.t5_text_encoder(t5_tokens)
                prompt_embeds = mx.stop_gradient(prompt_embeds)

                pooled_prompt_embeds = self.clip_text_encoder(clip_tokens)
                pooled_prompt_embeds = mx.stop_gradient(pooled_prompt_embeds)

                # Créer l'example
                from mflux.dreambooth.dataset.batch import Example

                example = Example(
                    example_id=idx,
                    prompt=self.instance_prompt,
                    image_path=image_path,
                    encoded_image=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                )

                self.encoded_examples.append(example)

                if progress_callback:
                    progress_callback()

            except Exception as e:
                print(f"Erreur lors de l'encodage de {image_path}: {str(e)}")
                continue

    def get_batch(self):
        """Retourne un batch aléatoire d'examples"""
        if not self.encoded_examples:
            raise ValueError("Le dataset n'a pas été encodé")

        # Choisir un example aléatoire
        example = random.choice(self.encoded_examples)

        # Créer le batch
        from mflux.dreambooth.dataset.batch import Batch

        batch = Batch(examples=[example])

        return batch
