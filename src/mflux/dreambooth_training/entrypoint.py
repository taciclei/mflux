# mflux/dreambooth_training/entrypoint.py
import sys

# Monkey patching: Remplacer l'ancien module VAE par un module factice

sys.modules["mflux.models.vae.vae"] = sys.modules["mflux.dreambooth_training.dummy_vae"]

# Importer la function main après le monkey patching
from mflux.dreambooth_training.main import main

if __name__ == "__main__":
    main()
