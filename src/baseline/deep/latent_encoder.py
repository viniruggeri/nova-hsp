import torch
import torch.nn as nn


class LatentEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x: torch.Tensor):
        raise NotImplementedError

    def decode(self, z: torch.Tensor):
        raise NotImplementedError
