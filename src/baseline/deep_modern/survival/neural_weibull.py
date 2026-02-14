import torch
import torch.nn as nn


class NeuralWeibull(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.k_head = nn.Linear(hidden_dim, 1)
        self.lambda_head = nn.Linear(hidden_dim, 1)

        self.softplus = nn.Softplus()

    def forward(self, x, lengths=None):
        # usa último timestep
        x_last = x[:, -1, :]

        features = self.feature_extractor(x_last)

        k = self.softplus(self.k_head(features)) + 1e-6
        lam = self.softplus(self.lambda_head(features)) + 1e-6

        return k, lam
