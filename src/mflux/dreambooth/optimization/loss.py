from typing import Any, Dict

import mlx.core as mx


class DreamBoothLoss:
    @staticmethod
    def compute_loss(flux, batch: Dict[str, Any], guidance_scale: float = 7.5) -> mx.array:
        """
        Compute the DreamBooth loss with proper type handling and memory optimization

        Args:
            flux: The flux model instance
            batch: The batch data dictionary
            guidance_scale: The guidance scale for the loss computation

        Returns:
            mx.array: The computed loss value
        """
        try:
            # Forward pass with gradient computation
            with mx.stop_gradient():
                # Get model outputs
                outputs = flux.forward(batch)

                # Compute the main loss components
                reconstruction_loss = outputs.get("reconstruction_loss", 0.0)
                prior_loss = outputs.get("prior_loss", 0.0)

                # Apply guidance scale
                total_loss = reconstruction_loss + guidance_scale * prior_loss

                # Add any regularization terms if present
                if "reg_loss" in outputs:
                    total_loss = total_loss + outputs["reg_loss"]

            return total_loss

        except Exception as e:
            print(f"Error in loss computation: {str(e)}")
            raise
