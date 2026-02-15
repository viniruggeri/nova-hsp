"""
Learned Dynamics Model — g_θ(z_t) → z_{t+1}

For real-world data where the true dynamics f(x, p) are unknown,
we learn a surrogate dynamics model in latent space.

Architecture v3 pipeline:
    z_t → [Dynamics] → z_{t+1} = g_θ(z_t)
    (iterated H times for rollouts)

Supported models:
    - LatentDynamicsMLP: Simple z_{t+1} = MLP(z_t)
    - ResidualDynamics: z_{t+1} = z_t + MLP(z_t) (residual connection)
    - NeuralODE: dz/dt = f_θ(z) integrated by ODE solver (requires torchdiffeq)

For synthetic systems, use IdentityDynamics (rollout is done analytically).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IdentityDynamics(nn.Module):
    """
    Placeholder for synthetic systems where rollouts use the true dynamics.

    Not actually used in forward pass — exists to satisfy the pipeline interface
    when compute_basin_access (analytical) is used instead of compute_basin_access_learned.
    """

    def __init__(self, latent_dim: int = 1):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z


class LatentDynamicsMLP(nn.Module):
    """
    Simple MLP dynamics: z_{t+1} = MLP(z_t).

    Args:
        latent_dim: Dimensionality of latent space.
        hidden_dim: Hidden layer size.
        num_layers: Number of hidden layers.
        activation: Activation function.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        self.latent_dim = latent_dim

        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]

        layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), act_fn()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn()])
        layers.append(nn.Linear(hidden_dim, latent_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, latent_dim) -> z_next: (batch, latent_dim)"""
        return self.net(z)


class ResidualDynamics(nn.Module):
    """
    Residual dynamics: z_{t+1} = z_t + MLP(z_t).

    Learns the increment Δz rather than the full next state.
    Better for systems with slow dynamics (small changes per step).

    Args:
        latent_dim: Dimensionality of latent space.
        hidden_dim: Hidden layer size.
        num_layers: Number of hidden layers.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, latent_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, latent_dim) -> z_next: (batch, latent_dim)"""
        return z + self.net(z)


class NeuralODE(nn.Module):
    """
    Neural ODE dynamics: dz/dt = f_θ(z), integrated by adaptive solver.

    Requires torchdiffeq. Provides continuous-time dynamics, which is
    more natural for physical systems and allows variable integration horizons.

    Args:
        latent_dim: Dimensionality of latent space.
        hidden_dim: Hidden layer size.
        solver: ODE solver method ('dopri5', 'euler', 'rk4').
        rtol: Relative tolerance for adaptive solver.
        atol: Absolute tolerance for adaptive solver.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

        self.func = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def _ode_func(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """ODE right-hand side: dz/dt = f_θ(z)."""
        return self.func(z)

    def forward(self, z: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Integrate one step: z(t) -> z(t + dt).

        Args:
            z: Current state, shape (batch, latent_dim).
            dt: Integration time step.

        Returns:
            Next state z(t+dt), shape (batch, latent_dim).
        """
        try:
            from torchdiffeq import odeint
        except ImportError:
            raise ImportError(
                "NeuralODE requires torchdiffeq. "
                "Install with: pip install torchdiffeq"
            )

        t_span = torch.tensor([0.0, dt], device=z.device)
        # odeint returns (2, batch, d) — take last time point
        z_out = odeint(
            self._ode_func,
            z,
            t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )
        return z_out[-1]
