import torch
import torch.nn as nn


class LSTMRUL(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        direction_factor = 2 if bidirectional else 1

        self.norm = nn.LayerNorm(hidden_dim * direction_factor)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * direction_factor, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x, lengths=None):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        last_hidden = self.norm(last_hidden)
        return self.head(last_hidden)
    
    
#teste amem jesus agr da certo 5.0
