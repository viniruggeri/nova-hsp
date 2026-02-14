import torch
import torch.nn as nn


class NeuralStateSpace(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.A = nn.Linear(hidden_dim, hidden_dim)
        self.B = nn.Linear(input_dim, hidden_dim)

        self.activation = nn.Tanh()

        self.norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths=None):
        batch_size, seq_len, _ = x.size()

        h = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=x.device
        )

        for t in range(seq_len):
            h = self.activation(
                self.A(h) + self.B(x[:, t, :])
            )

        h = self.norm(h)
        return self.head(h)
