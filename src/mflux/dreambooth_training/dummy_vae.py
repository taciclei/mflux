# mflux/dreambooth_training/dummy_vae.py
import mlx.nn as nn


class VAE(nn.Module):
    def __init__(self):
        print("[dreambooth_training] Dummy VAE initialized")
        super().__init__()

    def encode(self, x):
        pass

    def decode(self, z):
        pass
