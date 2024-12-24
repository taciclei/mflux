import mlx.core as mx
import mlx.optimizers as optim

from mflux.config.config import Config
from mflux.config.runtime_config import RuntimeConfigValidation
from mflux.dreambooth.config.training_config import DreamBoothConfig
from mflux.dreambooth.dataset.batch import Batch
from mflux.dreambooth.training_spec import TrainingSpec
from mflux.dreambooth.training_state import TrainingState
from mflux.flux.flux import Flux1
from mflux.memory_utils import optimize_memory_usage


class DreamBooth:
    def __init__(
        self,
        flux: Flux1,
        config: DreamBoothConfig,
        runtime_config: RuntimeConfigValidation,
    ):
        self.flux = flux
        self.config = config
        self.runtime_config = runtime_config

        # Créer la configuration pour le modèle
        self.model_config = Config(
            num_inference_steps=50,
            guidance=runtime_config.guidance_scale,
            height=512,
            width=512,
        )

        # Créer l'optimiseur
        self.optimizer = optim.Adam(
            learning_rate=config.learning_rate,
            betas=(runtime_config.adam_beta1, runtime_config.adam_beta2),
            eps=runtime_config.adam_epsilon,
        )

    def train_step(self, params: dict, batch: Batch) -> tuple[float, dict]:
        """Effectue une étape d'entraînement"""
        try:
            # Convertir le lot en dictionnaire
            batch_dict = batch.to_dict()

            # Calculer la perte et les gradients
            loss, grads = self.compute_loss_and_gradients(params, batch_dict)

            # Appliquer le weight decay
            if self.runtime_config.adam_weight_decay > 0:
                for k, v in grads.items():
                    if isinstance(v, mx.array) and isinstance(params[k], mx.array):
                        grads[k] = v + self.runtime_config.adam_weight_decay * params[k]

            # Mettre à jour les paramètres
            params = self.optimizer.update(params, grads)

            return float(loss), params

        except Exception as e:
            print(f"Erreur lors de l'étape d'entraînement: {str(e)}")
            raise

    def compute_loss_and_gradients(self, params: dict, batch: dict) -> tuple[float, dict]:
        """Calcule la perte et les gradients pour un lot"""

        def loss_fn(params: dict) -> float:
            try:
                # Extraire les tenseurs du lot
                encoded_images = batch.get("encoded_images")
                prompt_embeds = batch.get("prompt_embeds")
                pooled_prompt_embeds = batch.get("pooled_prompt_embeds")

                # Vérifier que tous les tenseurs sont présents
                if not all([encoded_images is not None, prompt_embeds is not None, pooled_prompt_embeds is not None]):
                    print("Attention: Tenseurs manquants dans le lot")
                    return mx.array(0.0, dtype=mx.float32)

                # Calculer la perte
                loss = self.flux.compute_loss(
                    params=params,
                    encoded_images=encoded_images,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    guidance=self.runtime_config.guidance_scale,
                    config=self.model_config,
                )

                return loss

            except Exception as e:
                print(f"Erreur lors du calcul de la perte: {str(e)}")
                return mx.array(0.0, dtype=mx.float32)

        # Calculer la perte et les gradients
        loss, grads = mx.value_and_grad(loss_fn)(params)

        return loss, grads

    def train(self, training_spec: TrainingSpec, training_state: TrainingState):
        """Exécute la boucle d'entraînement complète"""
        # Créer les répertoires de sortie
        output_dir = training_spec.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        # Initialiser les paramètres
        params = self.flux.trainable_parameters()

        try:
            # Boucle d'entraînement
            step = 0
            accumulated_loss = 0.0

            while step < self.config.max_train_steps:
                # Créer un lot
                examples = []
                for _ in range(self.config.train_batch_size):
                    try:
                        example = next(training_state.iterator)
                        examples.append(example)
                    except StopIteration:
                        break

                if not examples:
                    break

                batch = Batch(examples=examples)

                # Étape d'entraînement
                loss, params = self.train_step(params, batch)
                accumulated_loss += loss

                # Mise à jour de la progression
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
                    training_state.update_progress(step, float(avg_loss))
                    accumulated_loss = 0.0

                    # Sauvegarde du checkpoint
                    if (step + 1) % self.config.save_steps == 0:
                        try:
                            checkpoint_path = checkpoint_dir / f"checkpoint-{step+1}.safetensors"
                            self.flux.save_checkpoint(params, checkpoint_path)
                        except Exception as e:
                            print(f"Erreur lors de la sauvegarde du checkpoint: {str(e)}")

                    # Nettoyage du cache
                    if (step + 1) % self.config.clear_cache_interval == 0:
                        optimize_memory_usage()

                step += 1

            # Sauvegarde finale
            try:
                final_path = output_dir / "final.safetensors"
                self.flux.save_checkpoint(params, final_path)
            except Exception as e:
                print(f"Erreur lors de la sauvegarde finale: {str(e)}")

        except KeyboardInterrupt:
            print("\nEntraînement interrompu par l'utilisateur")
            # Sauvegarde d'urgence
            try:
                emergency_path = output_dir / "emergency-checkpoint.safetensors"
                self.flux.save_checkpoint(params, emergency_path)
            except Exception as e:
                print(f"Erreur lors de la sauvegarde d'urgence: {str(e)}")
            raise

        except Exception as e:
            print(f"Erreur lors de l'entraînement: {str(e)}")
            raise

        return params
