import torch
import torch.nn as nn


class DeepHazardModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

    def hazard(self, x: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError