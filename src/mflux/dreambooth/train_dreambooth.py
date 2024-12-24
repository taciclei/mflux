"""Script d'entraînement DreamBooth"""

import argparse
from pathlib import Path

import mlx.core as mx
from rich.console import Console
from rich.panel import Panel

from mflux.config.config import Config
from mflux.config.runtime_config import RuntimeConfigValidation
from mflux.dreambooth.dataset.batch import Batch
from mflux.dreambooth.dataset.dataset import DreamBoothDataset
from mflux.dreambooth.utils.memory_utils import optimize_memory_usage
from mflux.flux.flux import Flux1

console = Console()


def parse_args():
    """Parse les arguments de la ligne de commande"""
    parser = argparse.ArgumentParser(description="Train a model with DreamBooth")

    # Arguments obligatoires
    parser.add_argument("--model", type=str, required=True, help="Chemin vers le modèle de base")
    parser.add_argument("--instance_data_dir", type=str, required=True, help="Dossier contenant les images d'instance")
    parser.add_argument("--output_dir", type=str, required=True, help="Dossier de sortie pour les checkpoints")
    parser.add_argument("--instance_prompt", type=str, required=True, help="Prompt pour les images d'instance")

    # Arguments optionnels
    parser.add_argument("--train_batch_size", type=int, default=1, help="Taille du batch")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Nombre d'étapes d'accumulation")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Taux d'apprentissage")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="Type de scheduler")
    parser.add_argument("--lr_warmup_steps", type=int, default=30, help="Nombre d'étapes de warmup")
    parser.add_argument("--max_train_steps", type=int, default=200, help="Nombre maximum d'étapes")
    parser.add_argument("--guidance", type=float, default=7.5, help="Facteur de guidance")

    return parser.parse_args()


def main():
    """Function principale"""
    args = parse_args()

    # Afficher la configuration
    console.print(
        Panel.fit(
            "[bold]Configuration de l'Entraînement[/bold]",
            border_style="blue",
        )
    )

    try:
        # Créer les dossiers
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Charger le modèle
        model_path = Path(args.model)
        config = Config()
        runtime_config = RuntimeConfigValidation(
            guidance_scale=args.guidance,
            num_inference_steps=args.max_train_steps,
            use_mixed_precision=True,
            memory_efficient=True,
            use_metal=True,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_weight_decay=1e-2,
            adam_epsilon=1e-8,
        )

        flux = Flux1(
            model_path=model_path,
            runtime_config=runtime_config,
            config=config,
        )

        # Créer le dataset
        dataset = DreamBoothDataset(
            instance_data_dir=Path(args.instance_data_dir),
            instance_prompt=args.instance_prompt,
            tokenizer_t5=flux.t5_tokenizer,
            tokenizer_clip=flux.clip_tokenizer,
            t5_text_encoder=flux.t5_text_encoder,
            clip_text_encoder=flux.clip_text_encoder,
            vae=flux.vae,
        )

        # Encoder le dataset
        dataset.encode()

        # Boucle d'entraînement
        for step in range(args.max_train_steps):
            # Optimiser la mémoire
            optimize_memory_usage(config)

            # Récupérer un batch
            batch: Batch = dataset.get_batch()
            batch_dict = batch.to_dict()

            # Calculer la perte
            loss = flux.compute_loss(
                params=flux.trainable_parameters(),
                encoded_images=batch_dict["encoded_images"],
                prompt_embeds=batch_dict["prompt_embeds"],
                pooled_prompt_embeds=batch_dict["pooled_prompt_embeds"],
                guidance=args.guidance,
                config=config,
            )

            # Mettre à jour les paramètres
            if (step + 1) % args.gradient_accumulation_steps == 0:
                mx.eval(loss)

            # Sauvegarder le checkpoint
            if (step + 1) % 100 == 0:
                checkpoint_path = output_dir / f"checkpoint_{step + 1}.safetensors"
                flux.save_checkpoint(flux.trainable_parameters(), checkpoint_path)

            # Afficher la progression
            if (step + 1) % 10 == 0:
                console.print(f"Étape {step + 1}/{args.max_train_steps}, Perte: {loss.item():.4f}")

    except Exception as e:
        console.print(f"\nErreur lors de l'entraînement: {str(e)}")
        raise


if __name__ == "__main__":
    main()
