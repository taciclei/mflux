from abc import ABC, abstractmethod

import mlx.core as mx
from mlx import nn

from .decoder.decoder import Decoder
from .encoder.encoder import Encoder


class VAEInterface(ABC):
    @abstractmethod
    def encode(self, x: mx.array) -> mx.array:
        pass

    @abstractmethod
    def decode(self, z: mx.array) -> mx.array:
        pass


class VAE(nn.Module, VAEInterface):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        moments = self.encoder(x)
        return moments[:, :4, :, :]

    def decode(self, z):
        return self.decoder(z)

    def update(self, parameters):
        self.load_state_dict(parameters, strict=False)
