from mflux.models.vae.vae import VAE


class VAEFactory:
    @staticmethod
    def create_vae():
        return VAE()
