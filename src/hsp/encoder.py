"""
Temporal Encoder — maps raw observations to latent state z_t.

Architecture v3 pipeline:
    raw time series → [Encoder] → z_t ∈ R^d

The encoder is the ONLY learned component needed for analytical systems.
For real-world data, it's paired with a dynamics model (see dynamics.py).

Supported architectures:
    - TransformerEncoder: Multi-head self-attention over input window
    - LSTMEncoder: Recurrent encoding with last hidden state
    - IdentityEncoder: Pass-through for systems where x_t IS the state (synthetic)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IdentityEncoder(nn.Module):
    """
    Pass-through encoder for synthetic systems.

    When we have direct access to the state x_t (e.g., saddle-node, double-well),
    no encoding is needed. This preserves the pipeline interface.
    """

    def __init__(self, state_dim: int = 1):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = state_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window, d) or (batch, d) -> z: (batch, d)"""
        if x.dim() == 3:
            return x[:, -1, :]  # last timestep
        return x


class LSTMEncoder(nn.Module):
    """
    LSTM-based temporal encoder.

    Maps a window of observations to a latent vector using the
    last hidden state of a multi-layer LSTM.

    Args:
        input_dim: Observation dimensionality.
        latent_dim: Latent space dimensionality.
        hidden_dim: LSTM hidden size.
        num_layers: Number of LSTM layers.
        dropout: Dropout between LSTM layers.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window, input_dim) -> z: (batch, latent_dim)"""
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, batch, hidden)
        h_last = h_n[-1]  # (batch, hidden)
        return self.projection(h_last)  # (batch, latent_dim)


class TransformerEncoder(nn.Module):
    """
    Transformer-based temporal encoder.

    Uses multi-head self-attention over the input window, then
    projects the CLS token (or mean-pooled) representation to latent space.

    Args:
        input_dim: Observation dimensionality.
        latent_dim: Latent space dimensionality.
        d_model: Transformer model dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dropout: Dropout rate.
        window_size: Maximum input window length (for positional encoding).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        window_size: int = 50,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(d_model, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window, input_dim) -> z: (batch, latent_dim)"""
        B, W, _ = x.shape
        h = self.input_proj(x) + self.pos_encoding[:, :W, :]
        h = self.transformer(h)  # (B, W, d_model)
        h = h.mean(dim=1)  # mean pool over window
        return self.projection(h)  # (B, latent_dim)
